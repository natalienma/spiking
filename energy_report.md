# Energy Comparison: Dense vs. Sparse Tensor Format

Measured with [codecarbon](https://github.com/mlco2/codecarbon) on `Apple M5` (macOS-26.5.2-arm64-arm-64bit-Mach-O).

Network: 784 -> 128 -> 64 -> 10 LIF layers, 30 timesteps per run, 300 runs per variant.

| variant | duration (s) | energy (kWh) | CO2eq (kg) |
|---|---|---|---|
| dense | 2.6762 | 8.893e-06 | 2.250e-06 |
| sparse | 2.7799 | 9.049e-06 | 2.289e-06 |

**sparse vs. dense: +1.7% energy, +3.9% wall time**
