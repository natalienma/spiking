import snntorch as snn
import torch
import numpy as np

time_total = 1.0
t_step = 0.1

# Layer 1 - Inputs - 5 Neurons
# Linear is not a neuron. it doesn't leak or fire. it's just a one-shot weighted sum
layer_1= torch.nn.Linear(5, 3) 
layer_1.weight.data = torch.full((3, 5), 0.5)

# Layer 2 - 3 Neurons
lif = snn.Leaky(beta = 0.9) # one global beta for now
mem = torch.zeros(3)
spk_out, mem_out = [[],[],[]], [[],[],[]]
input_history = []

for t in range(int(time_total/t_step)):
    input = torch.rand(5) # 5 random inputs per timestep
    input_history.append(input)
    layer_1_out = layer_1(input) # computes 3 weighted sums -> 3 outputs
    spk, mem = lif(layer_1_out, mem)
    for i in range(3):
        spk_out[i].append(spk[i].item())
        mem_out[i].append(mem[i].item())


print("input spikes: ", input_history)
print("membrane volt:", mem_out)
print("spike or no spike: ", spk_out)