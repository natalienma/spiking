import time

import numpy as np
import snntorch as snn
import torch
from codecarbon import EmissionsTracker

time_total = 3.0
t_step = 0.1
n_steps = int(time_total / t_step)

n_in, n_hidden1, n_hidden2, n_out = 784, 128, 64, 10

beta_trace = 0.9
A_plus = 0.05  # max weight nudge size
A_minus = 0.05
W_MIN, W_MAX = -1.0, 1.0  # weight clipping bounds
MEM_MIN, MEM_MAX = -1.0, 2.0  # membrane potential clamp bounds

N_REPEATS = 300  # repeat the sim to get a runtime long enough for codecarbon to measure


def dense_forward(layer, x):
    return layer(x)


def sparse_forward(layer, x):
    # spikes are mostly zero (~10% active) -- represent them as a sparse row vector
    # so the matmul only touches the nonzero entries instead of the full dense input
    x_sparse = x.unsqueeze(0).to_sparse()
    return (torch.sparse.mm(x_sparse, layer.weight.t()) + layer.bias).squeeze(0)


def stdp_update(layer, pre_spike, post_spike, pre_trace, post_trace):
    # delta_t > 0 (pre fired recently, post fires now): potentiate, weighted by pre_trace
    # delta_t < 0 (post fired recently, pre fires now): depress, weighted by post_trace
    delta_w = A_plus * torch.outer(post_spike, pre_trace) - A_minus * torch.outer(post_trace, pre_spike)
    layer.weight.data = torch.clamp(layer.weight.data + delta_w, W_MIN, W_MAX)


def run_simulation(forward_fn, seed=0, record_history=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    layer_1 = torch.nn.Linear(n_in, n_hidden1)
    layer_2 = torch.nn.Linear(n_hidden1, n_hidden2)
    layer_3 = torch.nn.Linear(n_hidden2, n_out)

    lif_1 = snn.Leaky(beta=0.9, reset_mechanism="zero")
    lif_2 = snn.Leaky(beta=0.9, reset_mechanism="zero")
    lif_3 = snn.Leaky(beta=0.9, reset_mechanism="zero")

    mem_1, mem_2, mem_3 = torch.zeros(n_hidden1), torch.zeros(n_hidden2), torch.zeros(n_out)
    trace_1 = torch.zeros(n_in)  # tracks recent activity of layer_1's inputs
    trace_2 = torch.zeros(n_hidden1)  # tracks recent spikes of layer_1's output
    trace_3 = torch.zeros(n_hidden2)  # tracks recent spikes of layer_2's output
    trace_4 = torch.zeros(n_out)  # tracks recent spikes of layer_3's output (post-trace for layer_3)

    spk_out, mem_out = [[] for _ in range(n_out)], [[] for _ in range(n_out)]
    input_history = []

    for t in range(n_steps):
        input = torch.tensor(np.random.rand(n_in) < 0.1, dtype=torch.float32)  # fires 10% of the time

        out_1 = forward_fn(layer_1, input)
        spk_1, mem_1 = lif_1(out_1, mem_1)
        mem_1 = torch.clamp(mem_1, min=MEM_MIN, max=MEM_MAX)

        out_2 = forward_fn(layer_2, spk_1)
        spk_2, mem_2 = lif_2(out_2, mem_2)
        mem_2 = torch.clamp(mem_2, min=MEM_MIN, max=MEM_MAX)

        out_3 = forward_fn(layer_3, spk_2)
        spk_3, mem_3 = lif_3(out_3, mem_3)
        mem_3 = torch.clamp(mem_3, min=MEM_MIN, max=MEM_MAX)

        trace_1 = beta_trace * trace_1 + input
        trace_2 = beta_trace * trace_2 + spk_1
        trace_3 = beta_trace * trace_3 + spk_2
        trace_4 = beta_trace * trace_4 + spk_3

        stdp_update(layer_1, input, spk_1, trace_1, trace_2)
        stdp_update(layer_2, spk_1, spk_2, trace_2, trace_3)
        stdp_update(layer_3, spk_2, spk_3, trace_3, trace_4)

        if record_history:
            input_history.append(input)
            for i in range(n_out):
                spk_out[i].append(spk_3[i].item())
                mem_out[i].append(mem_3[i].item())

    return {
        "layer_3_weights": layer_3.weight.data,
        "trace_3": trace_3,
        "spk_out": spk_out,
        "mem_out": mem_out,
        "input_history": input_history,
    }


def measure_energy(forward_fn, label):
    tracker = EmissionsTracker(
        project_name=f"snn_{label}",
        save_to_file=False,
        log_level="error",
        measure_power_secs=0.5,
        allow_multiple_runs=True,
    )
    tracker.start()
    start = time.perf_counter()
    for _ in range(N_REPEATS):
        run_simulation(forward_fn, seed=0, record_history=False)
    elapsed = time.perf_counter() - start
    tracker.stop()
    data = tracker.final_emissions_data
    return {
        "label": label,
        "duration_s": elapsed,
        "energy_kwh": data.energy_consumed,
        "emissions_kg": data.emissions,
        "cpu_model": data.cpu_model,
        "os": data.os,
    }


print(f"measuring energy: dense tensors ({N_REPEATS}x runs)...")
dense_stats = measure_energy(dense_forward, "dense")

print(f"measuring energy: sparse tensors ({N_REPEATS}x runs)...")
sparse_stats = measure_energy(sparse_forward, "sparse")

print("\n=== energy comparison (dense vs. sparse tensor format) ===")
print(f"{'variant':>8} | {'duration (s)':>13} | {'energy (kWh)':>14} | {'CO2eq (kg)':>12}")
print("-" * 56)
for stats in (dense_stats, sparse_stats):
    print(f"{stats['label']:>8} | {stats['duration_s']:>13.4f} | {stats['energy_kwh']:>14.3e} | {stats['emissions_kg']:>12.3e}")

energy_delta_pct = (sparse_stats["energy_kwh"] - dense_stats["energy_kwh"]) / dense_stats["energy_kwh"] * 100
time_delta_pct = (sparse_stats["duration_s"] - dense_stats["duration_s"]) / dense_stats["duration_s"] * 100
print(f"\nsparse vs. dense: {energy_delta_pct:+.1f}% energy, {time_delta_pct:+.1f}% wall time")

# ---- write a markdown report ----
report_path = "energy_report.md"
with open(report_path, "w") as f:
    f.write("# Energy Comparison: Dense vs. Sparse Tensor Format\n\n")
    f.write(f"Measured with [codecarbon](https://github.com/mlco2/codecarbon) on `{dense_stats['cpu_model']}` ({dense_stats['os']}).\n\n")
    f.write(f"Network: {n_in} -> {n_hidden1} -> {n_hidden2} -> {n_out} LIF layers, "
            f"{n_steps} timesteps per run, {N_REPEATS} runs per variant.\n\n")
    f.write("| variant | duration (s) | energy (kWh) | CO2eq (kg) |\n")
    f.write("|---|---|---|---|\n")
    for stats in (dense_stats, sparse_stats):
        f.write(f"| {stats['label']} | {stats['duration_s']:.4f} | {stats['energy_kwh']:.3e} | {stats['emissions_kg']:.3e} |\n")
    f.write(f"\n**sparse vs. dense: {energy_delta_pct:+.1f}% energy, {time_delta_pct:+.1f}% wall time**\n")
print(f"saved energy report to {report_path}")

# ---- single recorded run for the table/plot (dense; not part of the energy measurement) ----
plot_result = run_simulation(dense_forward, seed=0, record_history=True)
spk_out, mem_out, input_history = plot_result["spk_out"], plot_result["mem_out"], plot_result["input_history"]

print("\nfinal output-layer trace:", plot_result["trace_3"])
print("final output-layer weights:\n", plot_result["layer_3_weights"])

# ---- printed table (output layer, 10 neurons) ----
header = " | ".join(f"V{i:<2}" for i in range(n_out)) + " | " + " ".join(f"s{i}" for i in range(n_out))
print(f"\n{'t':>4} | {'in (sum)':>9} | {header}")
print("-" * (20 + 8 * n_out))
for t in range(n_steps):
    in_sum = input_history[t].sum().item()
    voltages = " ".join(f"{mem_out[i][t]:>6.2f}" for i in range(n_out))
    spikes = " ".join(f"{spk_out[i][t]:>2.0f}" for i in range(n_out))
    print(f"{t:>4} | {in_sum:>9.2f} | {voltages} | {spikes}")

# ---- plot (output layer, 10 neurons) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
for i in range(n_out):
    ax.plot(range(n_steps), mem_out[i], label=f"neuron {i} voltage")
    for t in range(n_steps):
        if spk_out[i][t] == 1:
            ax.axvline(t, color=f"C{i}", alpha=0.2)
ax.axhline(1.0, color="black", linestyle="--", label="threshold")
ax.set_xlabel("timestep")
ax.set_ylabel("membrane voltage")
ax.legend(fontsize=6, ncol=2)
ax.set_title(f"3-layer LIF network ({n_in}->{n_hidden1}->{n_hidden2}->{n_out})")
plt.tight_layout()
plt.savefig("step2_plot.png", dpi=120)
print("\nsaved plot to step2_plot.png")
