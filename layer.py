import snntorch as snn
import torch
import numpy as np

time_total = 3.0
t_step = 0.1
n_steps = int(time_total/t_step)

n_in, n_hidden1, n_hidden2, n_out = 784, 128, 64, 10

# Layer 1: input -> hidden1
layer_1 = torch.nn.Linear(n_in, n_hidden1)  # default init: Kaiming-uniform, scaled by 1/sqrt(fan_in)
lif_1 = snn.Leaky(beta=0.9, reset_mechanism="zero")
mem_1 = torch.zeros(n_hidden1)
trace_1 = torch.zeros(n_in)  # tracks recent activity of layer_1's inputs

# Layer 2: hidden1 -> hidden2
layer_2 = torch.nn.Linear(n_hidden1, n_hidden2)
lif_2 = snn.Leaky(beta=0.9, reset_mechanism="zero")
mem_2 = torch.zeros(n_hidden2)
trace_2 = torch.zeros(n_hidden1)  # tracks recent spikes of layer_1's output

# Layer 3: hidden2 -> output (must be 10 neurons)
layer_3 = torch.nn.Linear(n_hidden2, n_out)
lif_3 = snn.Leaky(beta=0.9, reset_mechanism="zero")
mem_3 = torch.zeros(n_out)
trace_3 = torch.zeros(n_hidden2)  # tracks recent spikes of layer_2's output
trace_4 = torch.zeros(n_out)  # tracks recent spikes of layer_3's output (post-trace for layer_3)

beta_trace = 0.9
A_plus = 0.05  # max weight nudge size
A_minus = 0.05
W_MIN, W_MAX = -1.0, 1.0  # weight clipping bounds

spk_out, mem_out = [[] for _ in range(n_out)], [[] for _ in range(n_out)]
input_history = []


def stdp_update(layer, pre_spike, post_spike, pre_trace, post_trace):
    # delta_t > 0 (pre fired recently, post fires now): potentiate, weighted by pre_trace
    # delta_t < 0 (post fired recently, pre fires now): depress, weighted by post_trace
    delta_w = A_plus * torch.outer(post_spike, pre_trace) - A_minus * torch.outer(post_trace, pre_spike)
    layer.weight.data = torch.clamp(layer.weight.data + delta_w, W_MIN, W_MAX)


# Run:
for t in range(n_steps):
    input = torch.tensor(np.random.rand(n_in) < 0.1, dtype=torch.float32)  # fires 10% of the time
    input_history.append(input)

    out_1 = layer_1(input)
    spk_1, mem_1 = lif_1(out_1, mem_1)

    out_2 = layer_2(spk_1)
    spk_2, mem_2 = lif_2(out_2, mem_2)

    out_3 = layer_3(spk_2)
    spk_3, mem_3 = lif_3(out_3, mem_3)

    trace_1 = beta_trace * trace_1 + input
    trace_2 = beta_trace * trace_2 + spk_1
    trace_3 = beta_trace * trace_3 + spk_2
    trace_4 = beta_trace * trace_4 + spk_3

    stdp_update(layer_1, input, spk_1, trace_1, trace_2)
    stdp_update(layer_2, spk_1, spk_2, trace_2, trace_3)
    stdp_update(layer_3, spk_2, spk_3, trace_3, trace_4)

    for i in range(n_out):
        spk_out[i].append(spk_3[i].item())
        mem_out[i].append(mem_3[i].item())

print("final output-layer trace:", trace_3)
print("final output-layer weights:\n", layer_3.weight.data)

# ---- printed table (output layer, 10 neurons) ----
header = " | ".join(f"V{i:<2}" for i in range(n_out)) + " | " + " ".join(f"s{i}" for i in range(n_out))
print(f"{'t':>4} | {'in (sum)':>9} | {header}")
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
