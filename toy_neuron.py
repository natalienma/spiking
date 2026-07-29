import snntorch as snn
import torch

# steps:
# build the neuron with its 2 properties: threshold and degradation
# initiate the neuron value
# create 2 storage arrays for spikes out and memory out
# give it an input

lif = snn.Leaky(beta = 0.9) # default threshold = 1.0

mem = torch.zeros(1)
spk_out, mem_out = [], []

spikes_in = torch.tensor([0., 0., 1., 0., 1., 1., 0., 0., 0., 0.])

for t in spikes_in:
    spk, mem = lif(t, mem) 
    spk_out.append(spk.item())
    mem_out.append(mem.item())

print("input spikes: ", spikes_in.tolist())
print("membrane volt:", [round(m, 2) for m in mem_out])
print("spike or no spike: ", spk_out)