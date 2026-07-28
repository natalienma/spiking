# Energy Comparison: Dense vs. Sparse Tensor Format

Measured with [codecarbon](https://github.com/mlco2/codecarbon) on `Apple M5` (macOS-26.5.2-arm64-arm-64bit-Mach-O).

Network: 700 -> 128 -> 64 -> 10 LIF layers, input: SHD sample (label=11), 140 timesteps of 5ms bins, 60 runs per variant.

| variant | duration (s) | energy (kWh) | CO2eq (kg) |
|---|---|---|---|
| dense | 4.2868 | 1.403e-05 | 3.549e-06 |
| sparse | 4.6594 | 1.507e-05 | 3.813e-06 |

**sparse vs. dense: +7.4% energy, +8.7% wall time**
