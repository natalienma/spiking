import snntorch as snn
import torch
import numpy as np

time_total = 1.0
t_step = 0.1
n_steps = int(time_total/t_step)

# Layer 1 - Inputs - 5 Neurons
# Linear is not a neuron. it doesn't leak or fire. it's just a one-shot weighted sum
layer_1= torch.nn.Linear(5, 3) 
layer_1.weight.data = torch.rand(3, 5)

# Layer 2 - 3 Neurons
lif = snn.Leaky(beta = 0.9) # one global beta for now
mem = torch.zeros(3)
spk_out, mem_out = [[],[],[]], [[],[],[]]
input_history = []

for t in range(n_steps):
    input = torch.tensor(np.random.rand(5) < 0.1, dtype=torch.float32) # 5 inputs (0 or 1) per timestep -- will fire 10% of the time
    input_history.append(input)
    layer_1_out = layer_1(input) # computes 3 weighted sums -> 3 outputs
    spk, mem = lif(layer_1_out, mem)
    for i in range(3):
        spk_out[i].append(spk[i].item())
        mem_out[i].append(mem[i].item())


# ---- printed table ----
print(f"{'t':>4} | {'in (sum)':>9} | {'V0':>7} {'V1':>7} {'V2':>7} | {'spk0':>5} {'spk1':>5} {'spk2':>5}")
print("-" * 60)
for t in range(n_steps):
    in_sum = input_history[t].sum().item()
    v0, v1, v2 = mem_out[0][t], mem_out[1][t], mem_out[2][t]
    s0, s1, s2 = spk_out[0][t], spk_out[1][t], spk_out[2][t]
    print(f"{t:>4} | {in_sum:>9.2f} | {v0:>7.2f} {v1:>7.2f} {v2:>7.2f} | {s0:>5.0f} {s1:>5.0f} {s2:>5.0f}")

# ---- plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
for i in range(3):
    ax.plot(range(n_steps), mem_out[i], label=f"neuron {i} voltage")
    for t in range(n_steps):
        if spk_out[i][t] == 1:
            ax.axvline(t, color=f"C{i}", alpha=0.2)
ax.axhline(1.0, color="black", linestyle="--", label="threshold")
ax.set_xlabel("timestep")
ax.set_ylabel("membrane voltage")
ax.legend()
ax.set_title("3-neuron LIF layer")
plt.tight_layout()
plt.savefig("step2_plot.png", dpi=120)
print("\nsaved plot to step2_plot.png")