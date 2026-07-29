# LIF Reset Mechanism: Soft vs. Hard Reset on MNIST

Same 2-layer `snn.Leaky` network (784->1000->10, beta=0.95, 25 timesteps), same seeded weight init, same training procedure -- only `reset_mechanism` differs: `"subtract"` (soft, `V -= Vth` on spike) vs `"zero"` (hard, `V := 0` on spike). Trained 1 epoch on MNIST, evaluated on the full 10,000-image test set.

## Results

| reset mechanism | test accuracy | test error | test loss | final train loss | train time (s) |
|---|---|---|---|---|---|
| soft (subtract) | 90.96% | 9.04% | 6.46 | 6.73 | 18.6 |
| hard (zero) | 33.71% | 66.29% | 7.99 | 8.33 | 18.3 |

**Error delta (soft - hard): -57.24 percentage points**

## Specs:
- Both of the LIF networks initiate with the same random weights (same seed) and same batches. Everything is the same except the reset behavior.

## Conclusion:
- Soft reset wins by 57.2 points 
- Soft reset spikes more and carries some voltage. Intuitively, I think it performed better because it is more data-rich, while hard reset is noiser. Hard reset discards all granularity after resetting to 0, but not all spikes are created equal. 

## Remaining Questions:
- Maybe over many epochs, hard reset will be able to converge?
