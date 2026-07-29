# LIF Reset Mechanism: Soft vs. Hard Reset

Comparing `snn.Leaky` with `reset_mechanism="subtract"` (soft: `V -= Vth` on spike, keeps the overshoot) against `reset_mechanism="zero"` (hard: `V := 0` on spike, discards the overshoot). Both neurons see the identical input current; beta=0.9, threshold=1.0, 30 timesteps.

Input current: `[1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 0.0, 0.0, 0.0, 0.0, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]`

## Trace

| t | I[t] | V_soft | spike_soft | V_hard | spike_hard | \|error\| |
|---|---|---|---|---|---|---|
| 0 | 1.40 | 1.400 | 1 | 1.400 | 1 | 0.000 |
| 1 | 1.40 | 1.660 | 1 | 1.400 | 1 | 0.260 |
| 2 | 1.40 | 1.894 | 1 | 1.400 | 1 | 0.494 |
| 3 | 1.40 | 2.105 | 1 | 1.400 | 1 | 0.705 |
| 4 | 1.40 | 2.294 | 1 | 1.400 | 1 | 0.894 |
| 5 | 1.40 | 2.465 | 1 | 1.400 | 1 | 1.065 |
| 6 | 0.00 | 1.218 | 1 | 0.000 | 0 | 1.218 |
| 7 | 0.00 | 0.096 | 0 | 0.000 | 0 | 0.096 |
| 8 | 0.00 | 0.087 | 0 | 0.000 | 0 | 0.087 |
| 9 | 0.00 | 0.078 | 0 | 0.000 | 0 | 0.078 |
| 10 | 1.20 | 1.270 | 1 | 1.200 | 1 | 0.070 |
| 11 | 1.20 | 1.343 | 1 | 1.200 | 1 | 0.143 |
| 12 | 1.20 | 1.409 | 1 | 1.200 | 1 | 0.209 |
| 13 | 1.20 | 1.468 | 1 | 1.200 | 1 | 0.268 |
| 14 | 1.20 | 1.521 | 1 | 1.200 | 1 | 0.321 |
| 15 | 1.20 | 1.569 | 1 | 1.200 | 1 | 0.369 |
| 16 | 0.00 | 0.412 | 0 | 0.000 | 0 | 0.412 |
| 17 | 0.00 | 0.371 | 0 | 0.000 | 0 | 0.371 |
| 18 | 0.00 | 0.334 | 0 | 0.000 | 0 | 0.334 |
| 19 | 0.00 | 0.300 | 0 | 0.000 | 0 | 0.300 |
| 20 | 0.30 | 0.570 | 0 | 0.300 | 0 | 0.270 |
| 21 | 0.30 | 0.813 | 0 | 0.570 | 0 | 0.243 |
| 22 | 0.30 | 1.032 | 1 | 0.813 | 0 | 0.219 |
| 23 | 0.30 | 0.229 | 0 | 1.032 | 1 | 0.803 |
| 24 | 0.30 | 0.506 | 0 | 0.300 | 0 | 0.206 |
| 25 | 0.30 | 0.755 | 0 | 0.570 | 0 | 0.185 |
| 26 | 0.30 | 0.980 | 0 | 0.813 | 0 | 0.167 |
| 27 | 0.30 | 1.182 | 1 | 1.032 | 1 | 0.150 |
| 28 | 0.30 | 0.364 | 0 | 0.300 | 0 | 0.064 |
| 29 | 0.30 | 0.627 | 0 | 0.570 | 0 | 0.057 |

## Error summary

- Spike count -- soft: 15.0, hard: 14.0
- Mean absolute error (membrane voltage): 0.3354
- RMSE: 0.4508
- Max error: 1.2183, at t=6

## Interpretation

- Error is zero while the neuron stays sub-threshold (both reset mechanisms are no-ops there).
- Error appears/grows right after a spike during a *sustained* input burst: soft reset carries the overshoot `V - Vth` forward into the next timestep, so it reaches threshold again sooner than the hard-reset neuron, which restarts from exactly 0.
- Once the input drops back to a low/sub-threshold level, both neurons just decay by `beta` from wherever they were left, so the gap between them shrinks but doesn't reset to zero immediately -- it decays at the same rate `beta` as everything else.
- Soft reset is the more common default (matches the discrete LIF update `V[t] = beta*V[t-1] + I[t] - S[t-1]*Vth` used in [notes.md](notes.md)) because it conserves charge instead of discarding it, which matters for high firing rates / large timesteps where hard reset would systematically throw away energy and undercount how excited the neuron really is.
- The accumulated drift can flip a spike's timing: at t=22-23 soft reset spikes one step earlier than hard reset (it crossed threshold sooner because of the carried-over charge), which is exactly the failure mode that matters for training -- reset mechanism choice can change *which* timestep a neuron fires on, not just its membrane trace.
