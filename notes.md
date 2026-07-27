## Syntax vs. our own names — here's the split:

### Ours (we chose these names): 
lif, mem, spikes_in, spk_out, mem_out, t — could be renamed to anything, Python wouldn't care.

### snnTorch's (fixed, defined by the library): 
snn.Leaky(...), the argument beta=, and the behavior of calling lif(x, mem) — that calling convention (takes current input + previous membrane voltage, returns new spike + new voltage) is baked into the snn.Leaky class. We didn't invent that part.

## Next steps
1. variable beta for different neurons (heterogeneous time constants)
