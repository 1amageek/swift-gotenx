# QLKNN Transport Model Configuration

## Overview

QLKNN (QuaLiKiz Neural Network) is a fast surrogate model for the QuaLiKiz turbulent transport code. It predicts heat and particle fluxes from local plasma parameters using neural networks trained on 300M QuaLiKiz simulations.

**Performance**: 4-6 orders of magnitude faster than QuaLiKiz
- QuaLiKiz: ~1 second per radial point
- QLKNN: ~1 millisecond per radial point (GPU-accelerated with MLX)

**Accuracy**: R² > 0.96 for all transport channels

## Platform Requirements

swift-Gotenx targets macOS 26.4 and Metal 4 capable Apple Silicon. QLKNN is included by default.

- ✅ macOS 26.4+ on Metal 4 capable Apple Silicon
- ❌ iOS/visionOS (not targeted)
- ❌ Linux (not supported)

## Configuration Example

See `iter_like_qlknn.json` for a complete ITER-like simulation using QLKNN transport:

```json
{
  "runtime": {
    "dynamic": {
      "transport": {
        "modelType": "qlknn",
        "parameters": {
          "Zeff": 1.5,
          "min_chi": 0.01
        }
      }
    }
  }
}
```

### Transport Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Zeff` | Float | 1.0 | Effective charge for collisionality calculation |
| `min_chi` | Float | 0.01 | Minimum transport coefficient floor [m²/s] |

**Zeff values**:
- 1.0: Pure deuterium plasma
- 1.5: Typical D-T mixture with impurities
- 2.0-3.0: Higher impurity content

**min_chi**: Prevents numerical issues when QLKNN predicts very low transport (e.g., in ITB regions).

## Running a Simulation

```bash
# Build in release mode for optimal performance
swift build -c release

# Run QLKNN simulation
.build/release/GotenxCLI run \
  --config Examples/Configurations/iter_like_qlknn.json \
  --output-dir results/qlknn_test \
  --output-format netcdf \
  --log-progress
```

## Expected Output

QLKNN predicts turbulent transport from:
- **Normalized gradients**: R/L_Ti, R/L_Te, R/L_ne
- **Magnetic geometry**: q, s (shear), x (inverse aspect ratio)
- **Collisionality**: log₁₀(ν*)
- **Temperature ratio**: Ti/Te

Output transport coefficients:
- **χ_i**: Ion thermal diffusivity [m²/s]
- **χ_e**: Electron thermal diffusivity [m²/s]
- **D**: Particle diffusivity [m²/s]

## Fallback Behavior

If QLKNN prediction fails (e.g., input outside training range), the model automatically falls back to **Bohm-GyroBohm** transport:

```
[QLKNNTransportModel] QLKNN prediction failed: <error>
[QLKNNTransportModel] Falling back to Bohm-GyroBohm transport
```

This ensures robustness during edge-case scenarios.

## Comparison with Bohm-GyroBohm

To compare QLKNN with empirical transport:

```bash
# Run with QLKNN
.build/release/GotenxCLI run --config iter_like_qlknn.json --output-dir results/qlknn

# Run with Bohm-GyroBohm
.build/release/GotenxCLI run --config iter_like.json --output-dir results/bohm
```

Expected differences:
- **QLKNN**: More physics-based, captures ITG/TEM/ETG turbulence modes
- **Bohm-GyroBohm**: Empirical scaling, simpler but less accurate
- **Confinement**: QLKNN typically predicts lower transport (better confinement) in H-mode

## References

- **QLKNN Paper**: van de Plassche et al., "Fast modeling of turbulent transport in fusion plasmas using neural networks", Physics of Plasmas 27, 022310 (2020)
- **QuaLiKiz**: Bourdelle et al., "A new gyrokinetic quasilinear transport model applied to particle transport in tokamak plasmas", Physics of Plasmas 14, 112501 (2007)
- **FusionSurrogates Package**: https://github.com/1amageek/swift-fusion-surrogates
- **TORAX (Original Python)**: https://github.com/google-deepmind/torax

## Troubleshooting

### "QLKNN model failed to load"

**Cause**: FusionSurrogates package not properly installed or SafeTensors weights missing.

**Solution**:
```bash
# Rebuild dependencies
swift package clean
swift package resolve
swift build
```

### "swift-gotenx requires a Metal 4 capable device"

**Cause**: Running below macOS 26.4 or on hardware without Metal 4 support.

**Solution**: Run on macOS 26.4+ with a Metal 4 capable Apple Silicon GPU.

### Very low/zero transport predicted

**Cause**: QLKNN may predict very low transport in stable regions (e.g., ITB).

**Solution**: Adjust `min_chi` parameter:
```json
"parameters": {
  "min_chi": 0.1  // Increase floor to 0.1 m²/s
}
```

## Validation

QLKNN has been validated against:
- 300M QuaLiKiz simulations (training dataset)
- ITER baseline scenarios
- JET experimental data
- Multi-code benchmarks (JETTO, CRONOS, TRANSP)

For swift-Gotenx specific validation, see:
- `Tests/GotenxTests/Transport/QLKNNTransportModelTests.swift`
- Benchmark results (coming soon)
