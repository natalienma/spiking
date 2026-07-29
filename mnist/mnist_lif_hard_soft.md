# LIF Reset Mechanism: Soft vs. Hard Reset on MNIST

Same 2-layer `snn.Leaky` network (784->1000->10, beta=0.95, 25 timesteps), same seeded weight init, same training procedure -- only `reset_mechanism` differs: `"subtract"` (soft, `V -= Vth` on spike) vs `"zero"` (hard, `V := 0` on spike). Trained 1 epoch on MNIST, evaluated on the full 10,000-image test set.

## Results

| reset mechanism | test accuracy | test error | test loss | final train loss | train time (s) |
|---|---|---|---|---|---|
| soft (subtract) | 90.96% | 9.04% | 6.46 | 6.73 | 18.6 |
| hard (zero) | 33.71% | 66.29% | 7.99 | 8.33 | 18.3 |

**Error delta (soft - hard): -57.24 percentage points**

## Interpretation

- Both networks start from identical weights (same seed) and see identical batches, so the only source of divergence is what happens to membrane voltage the instant a neuron crosses threshold.
- Soft reset carries the overshoot `V - Vth` into the next timestep; hard reset discards it and restarts from 0. Over many timesteps and two stacked LIF layers, this changes exactly *which* timesteps spike, not just the membrane trace shape -- so it propagates into different spike counts reaching the output layer, and ultimately a different accuracy/error on held-out digits.
- Here **soft reset** wins by a wide margin (57.2 points of test error). Hard reset throws away the overshoot every time a neuron fires, which after 25 timesteps through two stacked layers compounds into weaker/noisier spike-count signal reaching the output layer -- consistent with it converging much more slowly than soft reset within a single epoch.
