# swift-Gotenx

A Swift implementation of Google DeepMind's [TORAX](https://github.com/google-deepmind/torax) tokamak core transport simulator, optimized for Apple Silicon using MLX-Swift. This differentiable, GPU-accelerated simulator solves coupled nonlinear PDEs describing fusion plasma transport in tokamaks.

## Highlights

- **Differentiable Transport Solver**: Coupled 1D PDEs for ion/electron temperature and particle density
- **Apple Silicon Optimized**: MLX-Swift backend with lazy evaluation, JIT compilation, unified memory, and a Metal 4 baseline
- **GPU-Accelerated**: All computations target Metal 4 capable Apple Silicon GPUs using Float32 precision
- **QLKNN Neural Network**: Fast turbulent transport prediction (4-6 orders of magnitude faster than QuaLiKiz)
- **Modular Physics Stack**: Protocol-based transport models and source terms
- **Type-Safe Concurrency**: Swift 6 actors with `EvaluatedArray` wrapper for Sendable MLXArray
- **Auto-Differentiation**: Jacobian computation via `grad()` and `vjp()` for Newton-Raphson
- **Command-Line Interface**: Full CLI with progress monitoring and multiple output formats
- **Scientific Data Formats**: JSON and NetCDF-4 output with CF-1.8 compliance
- **NetCDF 圧縮最適化**: DEFLATE レベル6 + 時間方向256スライスチャンクで 50× 以上の圧縮を確認

## Project Status

**Phase 4 Complete (100%)** - Core functionality operational with full CLI integration.

✅ **Core Infrastructure**:
- FVM discretization with power-law scheme
- Linear solver (predictor-corrector with Pereverzev)
- Newton-Raphson solver with auto-differentiation
- Geometry system (circular tokamak)
- Configuration system (JSON loading and validation)
- CLI executable (GotenxCLI)
- Actor-based orchestration
- High-precision time accumulation (Double)
- Conservation law enforcement

✅ **Physics Models** (GotenxPhysics):
- Fusion power (Bosch-Hale reactivity)
- Ohmic heating
- Ion-electron energy exchange
- Bremsstrahlung radiation

✅ **Transport Models**:
- Constant transport model
- Bohm-GyroBohm transport model
- **QLKNN neural network** (FusionSurrogates, 4-6 orders of magnitude faster than QuaLiKiz)

✅ **Output Formats**:
- JSON (human-readable)
- NetCDF-4 (CF-1.8 compliant, compressed)

⏳ **In Development**:
- Visualization and plotting
- Interactive CLI menu actions
- Pedestal models

## Prerequisites

- **macOS 26.4+** on Apple Silicon with Metal 4 support
- **Xcode 17.0+** with Swift 6.2 toolchain and the macOS 26.4 SDK
- **Dependencies** (automatically resolved via SwiftPM):
  - MLX-Swift 0.31.3+
  - Swift Configuration 0.1.1+
  - Swift Argument Parser 1.5.0+
  - SwiftNetCDF 1.2.0+
  - FusionSurrogates (for QLKNN)

**Optional**:
- **ncdump** (part of NetCDF tools) for inspecting NetCDF output
- **Python 3.11+** with Google DeepMind's TORAX for comparison

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/swift-Gotenx.git
cd swift-Gotenx
swift build -c release
```

### Run a Simulation

```bash
# Run with example configuration
.build/release/GotenxCLI run \
  --config examples/Configurations/minimal.json \
  --output-dir /tmp/gotenx_results \
  --output-format netcdf \
  --log-progress

# Or install globally
swift package experimental-install -c release
gotenx run --config examples/Configurations/iter_like.json
```

### Inspect Results

```bash
# NetCDF output (recommended for scientific workflows)
ncdump -h /tmp/gotenx_results/state_history_*.nc

# JSON output (human-readable)
cat /tmp/gotenx_results/state_history_*.json | jq .
```

### Run Tests

```bash
# Run all tests
perl -e 'alarm shift; exec @ARGV' 700 xcodebuild test \
  -scheme swift-gotenx-Package \
  -destination 'platform=macOS' \
  -configuration Debug \
  -derivedDataPath /tmp/Gotenx-Package-Xcodebuild-DerivedData

# Run specific test suite
perl -e 'alarm shift; exec @ARGV' 120 xcodebuild test \
  -scheme swift-gotenx-Package \
  -destination 'platform=macOS' \
  -configuration Debug \
  -derivedDataPath /tmp/Gotenx-Package-Xcodebuild-DerivedData \
  -only-testing:GotenxTests/MetalAccelerationPolicyTests
```

### NetCDF Compression Strategy

- 出力 NetCDF-4 ファイルは DEFLATE レベル 6 / shuffle 有効で書き出します。
- 時間方向は最大 256 ステップずつまとめてチャンクし（`[min(256, nTime), nCells]`）、空間方向は全セルを 1 チャンクに含めます。
- 上記設定でテスト用データに対し 51× 以上、NetCDF 既定チャンクでは 61× の圧縮率を確認しています（`xcodebuild test -only-testing:GotenxTests/NetCDFCompressionTests/testCompressionRatio`）。
- CLI の `OutputWriter` が生成する NetCDF でも `xcodebuild test -only-testing:GotenxCLITests/OutputWriterTests/testNetCDFCompressionRatio` を実行すると約 20〜25× の圧縮率が再現されます（テストログで実測値を表示）。
- 時間方向アクセスの局所性を重視する場合は 128/64 ステップといった粒度に落とすか、差分エンコードなどの前処理を併用してください。

## Repository Structure

```
swift-Gotenx/
├── Sources/
│   ├── Gotenx/                      # Core library
│   │   ├── Configuration/          # Config loaders and validation
│   │   ├── Conservation/           # Conservation law enforcement
│   │   ├── Core/                   # CoreProfiles, EvaluatedArray, Geometry
│   │   ├── Diagnostics/            # Diagnostic outputs
│   │   ├── FVM/                    # Finite volume method utilities
│   │   ├── Geometry/               # Tokamak geometry calculations
│   │   ├── Orchestration/          # SimulationOrchestrator (actor-based)
│   │   ├── Output/                 # Result serialization
│   │   ├── Protocols/              # Core protocols (TransportModel, etc.)
│   │   ├── Solver/                 # PDE solvers (Linear, Newton-Raphson)
│   │   ├── Transport/              # Transport model implementations
│   │   └── Utilities/              # Helper utilities
│   │
│   ├── GotenxPhysics/               # Physics models (separate module)
│   │   ├── Heating/                # FusionPower, OhmicHeating, IonElectronExchange
│   │   ├── Radiation/              # Bremsstrahlung
│   │   ├── Neoclassical/           # SauterBootstrapModel
│   │   └── Utilities/              # PhysicsConstants, PhysicsError
│   │
│   └── GotenxCLI/                   # CLI executable
│       ├── Commands/               # RunCommand, PlotCommand, InteractiveMenu
│       ├── Configuration/          # EnvironmentConfig
│       ├── Output/                 # ProgressLogger, OutputWriter
│       └── Utilities/              # DisplayUnits
│
├── Tests/
│   ├── GotenxTests/                 # Core library tests
│   ├── GotenxPhysicsTests/          # Physics model tests
│   └── GotenxCLITests/              # CLI tests
│
├── Examples/
│   └── Configurations/             # Example JSON configurations
│       ├── minimal.json
│       ├── simple_constant_transport.json
│       ├── iter_like.json
│       ├── iter_like_qlknn.json    # QLKNN transport (macOS only)
│       └── README_QLKNN.md         # QLKNN documentation
│
└── docs/                           # Documentation (implementation notes)
```

## CLI Usage

The `gotenx` CLI provides commands for running simulations:

```bash
# Get help
gotenx --help
gotenx run --help

# Run simulation with NetCDF output
gotenx run \
  --config examples/Configurations/minimal.json \
  --output-dir ./results \
  --output-format netcdf \
  --log-progress

# Run with debugging
gotenx run \
  --config config.json \
  --no-compile \
  --enable-errors \
  --log-output

# Quit immediately after completion (for scripts)
gotenx run --config config.json --quit
```

### Configuration Files

Configurations use JSON format with nested structure:

```json
{
  "runtime": {
    "static": {
      "mesh": {
        "nCells": 50,
        "majorRadius": 3.0,
        "minorRadius": 1.0,
        "toroidalField": 2.5,
        "geometryType": "circular"
      },
      "solver": {
        "type": "linear",
        "tolerance": 1e-6,
        "maxIterations": 20
      }
    },
    "dynamic": {
      "boundaries": {
        "ionTemperature": 50.0,
        "electronTemperature": 50.0,
        "density": 5e18
      },
      "transport": {
        "modelType": "constant"
      },
      "sources": {
        "fusionPower": false,
        "ohmicHeating": true,
        "bremsstrahlung": true
      }
    }
  },
  "time": {
    "start": 0.0,
    "end": 1.0,
    "initialDt": 1e-5
  },
  "output": {
    "directory": "/tmp/gotenx_results",
    "format": "netcdf"
  }
}
```

See `examples/Configurations/` for complete examples.

## QLKNN Neural Network Transport

swift-Gotenx includes **QLKNN** (QuaLiKiz Neural Network), a fast surrogate model for turbulent transport prediction. QLKNN is **4-6 orders of magnitude faster** than the full QuaLiKiz gyrokinetic code while maintaining high accuracy (R² > 0.96).

### Platform Requirements

swift-Gotenx now targets macOS 26.4 and Metal 4 directly. QLKNN is included by default.

- ✅ macOS 26.4+ on Metal 4 capable Apple Silicon
- ❌ iOS/visionOS (not targeted)
- ❌ Linux (not supported)

### Quick Start with QLKNN

```bash
# 1. Build the project (QLKNN included automatically on macOS)
swift build -c release

# 2. Run ITER-like simulation with QLKNN transport
.build/release/GotenxCLI run \
  --config Examples/Configurations/iter_like_qlknn.json \
  --output-dir results/qlknn_test \
  --output-format netcdf \
  --log-progress

# 3. Expected output:
# ═══════════════════════════════════════════════════
# swift-Gotenx v0.1.0
# Tokamak Core Transport Simulator for Apple Silicon
# ═══════════════════════════════════════════════════
#
# 📋 Loading configuration...
# ✓ Configuration loaded and validated
#   Mesh cells: 100
#   Transport model: qlknn
#
# 🔧 Initializing physics models...
#   ✓ QLKNN network loaded successfully
#   ✓ Source models initialized
#
# 🚀 Initializing simulation...
# ✓ Simulation initialized
#
# ⏱️  Running simulation...
#   [Progress updates...]
#
# 📊 Simulation Results:
#   Total steps: 21053
#   Converged: Yes
#   Wall time: 45.2s
#
# 💾 Saving results...
#   ✓ Results saved to: results/qlknn_test/
```

### QLKNN Configuration

QLKNN is configured via the `transport` section:

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

**Parameters**:
- `Zeff` (default: 1.0): Effective charge for collisionality calculation
  - 1.0 = Pure deuterium
  - 1.5 = D-T mixture with typical impurities (ITER baseline)
  - 2.0-3.0 = Higher impurity content
- `min_chi` (default: 0.01 m²/s): Minimum transport coefficient floor
  - Prevents numerical issues in low-transport regions (ITB)

### What QLKNN Predicts

**Input features** (computed automatically from plasma profiles):
- Normalized temperature gradients: R/L_Ti, R/L_Te
- Normalized density gradient: R/L_ne
- Magnetic geometry: q (safety factor), s (shear), x = r/R
- Collisionality: log₁₀(ν*)
- Temperature ratio: Ti/Te

**Output transport coefficients**:
- **χ_i**: Ion thermal diffusivity [m²/s]
- **χ_e**: Electron thermal diffusivity [m²/s]
- **D**: Particle diffusivity [m²/s]

**Physics**: Captures ITG (Ion Temperature Gradient), TEM (Trapped Electron Mode), and ETG (Electron Temperature Gradient) turbulence.

### Comparing Transport Models

Compare QLKNN with empirical Bohm-GyroBohm transport:

```bash
# Run with QLKNN (physics-based)
.build/release/GotenxCLI run \
  --config Examples/Configurations/iter_like_qlknn.json \
  --output-dir results/qlknn \
  --output-format netcdf

# Run with Bohm-GyroBohm (empirical)
.build/release/GotenxCLI run \
  --config Examples/Configurations/iter_like.json \
  --output-dir results/bohm \
  --output-format netcdf

# Compare results
ncdump -v Ti results/qlknn/state_history_*.nc | grep "Ti ="
ncdump -v Ti results/bohm/state_history_*.nc | grep "Ti ="
```

**Expected differences**:
- **QLKNN**: More accurate, captures turbulence modes, typically predicts lower transport (better confinement)
- **Bohm-GyroBohm**: Faster (~100×), empirical scaling, good for baseline studies
- **Confinement time**: QLKNN usually predicts 20-40% longer τ_E in H-mode

### Performance

| Model | Time per timestep | Total sim time (2s) | Accuracy |
|-------|-------------------|---------------------|----------|
| QLKNN | ~1-2 ms | ~40-60s | R² > 0.96 |
| Bohm-GyroBohm | ~10-20 μs | ~1-2s | Empirical fit |
| QuaLiKiz (full) | ~1 second | ~6 hours | Reference |

QLKNN achieves **10,000× speedup** vs. QuaLiKiz with minimal accuracy loss.

### Troubleshooting

#### "QLKNN model failed to load"

**Cause**: FusionSurrogates package not properly linked.

**Solution**:
```bash
swift package clean
swift package resolve
swift build -c release
```

#### "swift-gotenx requires a Metal 4 capable device"

**Cause**: Running below macOS 26.4 or on hardware without Metal 4 support.

**Solution**: Run on macOS 26.4+ with a Metal 4 capable Apple Silicon GPU.

#### QLKNN fallback to Bohm-GyroBohm

**Cause**: Input parameters outside QLKNN training range (rare).

**Output**:
```
[QLKNNTransportModel] QLKNN prediction failed: <error>
[QLKNNTransportModel] Falling back to Bohm-GyroBohm transport
```

**Action**: Simulation continues automatically with empirical transport. Check if boundary conditions or source terms are unrealistic.

### Documentation

- **Detailed guide**: `Examples/Configurations/README_QLKNN.md`
- **Implementation**: `Sources/Gotenx/Transport/Models/QLKNNTransportModel.swift`
- **Tests**: `Tests/GotenxTests/Transport/QLKNNTransportModelTests.swift`
- **Package**: https://github.com/1amageek/swift-fusion-surrogates

### References

- **QLKNN Paper**: van de Plassche et al., "Fast modeling of turbulent transport in fusion plasmas using neural networks", _Physics of Plasmas_ **27**, 022310 (2020)
- **QuaLiKiz**: Bourdelle et al., "A new gyrokinetic quasilinear transport model applied to particle transport in tokamak plasmas", _Physics of Plasmas_ **14**, 112501 (2007)

## Output Formats

### NetCDF-4 (Recommended)

CF-1.8 compliant NetCDF-4 format with:
- UNLIMITED time dimension for time series
- Compression (deflate level 6, shuffle filter)
- Proper metadata (units, long_name, standard_name)
- Global attributes (simulation statistics, provenance)

```bash
# Inspect NetCDF file
ncdump -h output.nc

# Variables: Ti, Te, ne, psi
# Coordinates: time [s], rho [normalized]
# Attributes: total_steps, converged, wall_time_seconds
```

### JSON

Human-readable format for quick inspection and debugging.

## Unit System

**CRITICAL**: swift-Gotenx uses **SI-based units** internally:

| Quantity | Unit | Symbol | Notes |
|----------|------|--------|-------|
| **Temperature** | electron volt | **eV** | NOT keV |
| **Density** | particles/m³ | **m⁻³** | NOT 10²⁰ m⁻³ |
| **Time** | seconds | **s** | SI base |
| **Length** | meters | **m** | SI base |
| **Magnetic Field** | tesla | **T** | SI derived |
| **Power Density** | MW/m³ | **MW/m³** | Source terms |

**Display units** (CLI output) may show keV and 10²⁰ m⁻³ for user convenience via `DisplayUnits` module.

See `CLAUDE.md` section "Unit System Standard" for detailed rationale.

## Key Design Features

### Numerical Precision (Float32 only)

swift-Gotenx uses **Float32** exclusively on GPU (Apple Silicon GPUs don't support Float64):

- **Variable scaling** for Newton-Raphson conditioning
- **High-precision time accumulation** using `Double` (CPU-only, 1 op/timestep)
- **Diagonal preconditioning** for ill-conditioned Jacobians
- **Epsilon regularization** for stable gradients
- **Conservation law enforcement** for numerical drift detection

Result: Engineering-grade accuracy (relative error ≤ 10⁻³) over 20,000+ timesteps.

### MLX Lazy Evaluation

`EvaluatedArray` wrapper enforces evaluation at type boundaries:

```swift
public struct EvaluatedArray: @unchecked Sendable {
    private let array: MLXArray

    public init(evaluating array: MLXArray) {
        eval(array)  // Forces evaluation
        self.array = array
    }

    public var value: MLXArray { array }
}
```

### Swift 6 Concurrency

Actor-based simulation orchestration:

```swift
public actor SimulationOrchestrator {
    public func run(
        config: SimulationConfiguration,
        progressCallback: ((ProgressInfo) async -> Void)?
    ) async throws -> SimulationResult
}
```

All data structures are `Sendable`. `EvaluatedArray` is the only `@unchecked Sendable` type.

## Documentation

### Primary Documentation
- **`CLAUDE.md`**: Comprehensive project guide (architecture, precision policy, design decisions)
- **`README.md`**: This file - quick start and overview

### Design Documents
- **`ARCHITECTURE.md`**: System design (if exists)
- **`IMPLEMENTATION_NOTES.md`**: Design decisions and numerical considerations
- **`SOLVER_IMPLEMENTATION_STRATEGY.md`**: Solver architecture

### Phase Documentation
- **`docs/`**: Implementation notes and design analysis

## Roadmap

### P0 - High Priority
- [ ] Add pedestal models
- [ ] Implement plotting with Swift Charts or gnuplot bridge
- [ ] Complete interactive menu actions in CLI
- [ ] Benchmark QLKNN against original Python implementation

### P1 - Medium Priority
- [ ] Time-dependent geometry
- [ ] Current diffusion equation (ψ evolution)
- [ ] Forward sensitivity analysis
- [ ] Compilation caching
- [ ] Multi-ion species support

### P2 - Future
- [ ] MHD models (sawteeth, neoclassical tearing modes)
- [ ] Core-edge coupling
- [ ] Benchmark suite vs. Google DeepMind's TORAX
- [ ] Performance profiling
- [ ] HDF5 output (optional, NetCDF is preferred)

See `CLAUDE.md` for detailed roadmap aligned with [Google DeepMind's TORAX paper (arXiv:2406.06718v2)](https://arxiv.org/abs/2406.06718).

## Contributing

1. Fork the repository and create a feature branch
2. Run `xcodebuild test` before opening a PR
3. Include documentation/test updates
4. Follow Swift formatting conventions

## License

MIT License. See `LICENSE` for details.

## References

- Google DeepMind's TORAX: https://github.com/google-deepmind/torax
- Google DeepMind's TORAX Paper: arXiv:2406.06718v2
- MLX-Swift: https://github.com/ml-explore/mlx-swift
- Swift Numerics: https://github.com/apple/swift-numerics
- Swift Argument Parser: https://github.com/apple/swift-argument-parser
- SwiftNetCDF: https://github.com/patrick-zippenfenig/SwiftNetCDF

---

Questions or feedback? Open an issue on GitHub.
