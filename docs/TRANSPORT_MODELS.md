# Transport Models

swift-Gotenx provides three transport models with increasing physics fidelity.

## 1. Constant Transport Model

**Use case**: Testing, debugging, benchmarking

### Configuration

```json
{
  "transport": {
    "modelType": "constant",
    "parameters": {
      "chi_ion": 1.0,
      "chi_electron": 1.0,
      "particle_diffusivity": 0.5
    }
  }
}
```

### Transport Coefficients [m²/s]

- χ_i = 1.0 (uniform)
- χ_e = 1.0 (uniform)
- D = 0.5 (uniform)

### Characteristics

- Spatially uniform
- Time-independent
- No physics dependencies
- Fastest execution (~1μs per timestep)

---

## 2. Bohm-GyroBohm Transport Model

**Use case**: Fast empirical transport, baseline comparisons

### Configuration

```json
{
  "transport": {
    "modelType": "bohmGyrobohm"
  }
}
```

### Physics

Empirical scaling combining:
- **Bohm diffusion**: χ_Bohm = T_e / (16 e B)
- **GyroBohm diffusion**: χ_GB = ρ*² c_s / a

Where:
- T_e: electron temperature
- B: magnetic field
- ρ*: normalized Larmor radius
- c_s: sound speed
- a: minor radius

### Performance

- ~10μs per timestep
- 100× faster than QLKNN
- CPU-only, no GPU acceleration needed

### Accuracy

- Empirical fit to experimental data
- No first-principles turbulence physics
- Reasonable for H-mode baseline scenarios
- Less accurate for ITB (Internal Transport Barrier) plasmas

---

## 3. QLKNN Transport Model (macOS only)

**Use case**: High-fidelity turbulent transport simulation

### Configuration

```json
{
  "transport": {
    "modelType": "qlknn",
    "parameters": {
      "Zeff": 1.5,
      "min_chi": 0.01
    }
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Zeff` | Float | 1.0 | Effective charge for collisionality calculation |
| `min_chi` | Float | 0.01 | Minimum transport coefficient floor [m²/s] |

**Zeff values**:
- 1.0: Pure deuterium plasma
- 1.5: Typical D-T mixture with impurities
- 2.0-3.0: Higher impurity content

**min_chi**: Prevents numerical issues when QLKNN predicts very low transport (e.g., in ITB regions).

### Physics

Neural network surrogate for QuaLiKiz gyrokinetic code:
- Trained on 300M QuaLiKiz simulations
- Captures ITG (Ion Temperature Gradient) turbulence
- Captures TEM (Trapped Electron Mode) turbulence
- Captures ETG (Electron Temperature Gradient) turbulence
- R² > 0.96 accuracy on all transport channels

### Input Features

- **Normalized gradients**: R/L_Ti, R/L_Te, R/L_ne
- **Magnetic geometry**: q (safety factor), s (shear), x = r/R (inverse aspect ratio)
- **Collisionality**: log₁₀(ν*)
- **Temperature ratio**: Ti/Te

### Output Transport Coefficients

- χ_i: Ion thermal diffusivity [m²/s]
- χ_e: Electron thermal diffusivity [m²/s]
- D: Particle diffusivity [m²/s]

### Performance

- ~1ms per timestep (GPU-accelerated with MLX)
- 4-6 orders of magnitude faster than QuaLiKiz
- 100× slower than Bohm-GyroBohm (but physics-based)

### Platform Requirements

swift-Gotenx targets macOS 26.4 and Metal 4 capable Apple Silicon. QLKNN is included by default.

- ✅ macOS 26.4+ on Metal 4 capable Apple Silicon
- ❌ iOS/visionOS (not targeted)
- ❌ Linux (not supported)

### Fallback Behavior

If QLKNN prediction fails (e.g., input outside training range), the model automatically falls back to **Bohm-GyroBohm** transport:

```
[QLKNNTransportModel] QLKNN prediction failed: <error>
[QLKNNTransportModel] Falling back to Bohm-GyroBohm transport
```

This ensures robustness during edge-case scenarios.

### Example Configuration

See `Examples/Configurations/iter_like_qlknn.json` for a complete ITER-like simulation using QLKNN transport.

### Detailed Documentation

For comprehensive QLKNN usage guide, troubleshooting, and validation information, see:
- `Examples/Configurations/README_QLKNN.md`

### References

- **QLKNN Paper**: van de Plassche et al., "Fast modeling of turbulent transport in fusion plasmas using neural networks", Physics of Plasmas 27, 022310 (2020)
- **QuaLiKiz**: Bourdelle et al., "A new gyrokinetic quasilinear transport model applied to particle transport in tokamak plasmas", Physics of Plasmas 14, 112501 (2007)
- **FusionSurrogates Package**: https://github.com/1amageek/swift-fusion-surrogates
- **TORAX (Original Python)**: https://github.com/google-deepmind/torax

---

## Model Comparison

| Aspect | Constant | Bohm-GyroBohm | QLKNN |
|--------|----------|---------------|-------|
| **Physics Fidelity** | None | Empirical | First-principles surrogate |
| **Speed** | ~1μs | ~10μs | ~1ms |
| **Platform** | All | All | macOS only |
| **Use Case** | Testing | Fast baseline | Research-grade |
| **Turbulence** | No | Scaling only | ITG/TEM/ETG |
| **ITB Capability** | No | Poor | Good |

---

*See also: [CLAUDE.md](../CLAUDE.md) for development guidelines*
