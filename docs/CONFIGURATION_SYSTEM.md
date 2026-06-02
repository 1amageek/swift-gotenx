# Configuration System Architecture

## Overview

swift-Gotenx uses **swift-configuration** for hierarchical, type-safe configuration management. The system provides a clear separation between different configuration sources with well-defined override priority.

## Hierarchical Configuration Priority

Configuration values are resolved in the following priority order (highest to lowest):

1. **CLI Arguments** (highest priority)
   - Passed via command-line flags like `--mesh-cell-count 200`
   - Mapped to hierarchical keys: `runtime.static.mesh.cellCount`

2. **Environment Variables**
   - Prefixed with `GOTENX_` and the full configuration path (e.g., `GOTENX_RUNTIME_STATIC_MESH_CELL_COUNT=150`)
   - Converted by `EnvironmentVariablesProvider().prefixKeys(with: "gotenx")`

3. **JSON Configuration File**
   - Loaded via `JSONProvider` with `FilePath` type
   - Standard simulation configuration format

4. **Default Values** (lowest priority)
   - Hardcoded defaults in configuration structs

## GotenxConfigReader Implementation

```swift
import Configuration
import Foundation
import SystemPackage

/// Gotenx-specific ConfigReader wrapper
public actor GotenxConfigReader {
    private let configReader: ConfigReader

    /// Create with hierarchical providers
    public static func create(
        jsonPath: String,
        cliOverrides: [String: String] = [:]
    ) async throws -> GotenxConfigReader {
        var providers: [any ConfigProvider] = []

        // IMPORTANT: ConfigReader uses FIRST-MATCH priority order
        // First provider in array has HIGHEST priority

        // Priority 1 (highest): CLI arguments
        if !cliOverrides.isEmpty {
            let configValues = cliOverrides.mapValues {
                ConfigValue(.string($0), isSecret: false)
            }
            providers.append(
                InMemoryProvider(values: configValues)
            )
        }

        // Priority 2: Environment variables
        providers.append(
            EnvironmentVariablesProvider()
                .prefixKeys(with: "gotenx")
        )

        // Priority 3 (lowest): JSON file
        let jsonProvider = try await JSONProvider(filePath: FilePath(jsonPath))
        providers.append(jsonProvider)

        let reader = ConfigReader(providers: providers)
        return GotenxConfigReader(configReader: reader)
    }

    /// Fetch complete SimulationConfiguration
    public func fetchConfiguration() async throws -> SimulationConfiguration {
        let runtime = try await fetchRuntimeConfig()
        let time = try await fetchTimeConfig()
        let output = try await fetchOutputConfig()

        return SimulationConfiguration(
            runtime: runtime,
            time: time,
            output: output
        )
    }
}
```

## Configuration Structure

```swift
/// Root simulation configuration
public struct SimulationConfiguration: Codable, Sendable {
    public let runtime: RuntimeConfiguration
    public let time: TimeConfiguration
    public let output: OutputConfiguration
}

/// Runtime configuration (static + dynamic)
public struct RuntimeConfiguration: Codable, Sendable {
    public let static: StaticConfig      // Triggers recompilation
    public let dynamic: DynamicConfig    // Hot-reloadable
}

/// Static parameters (trigger MLX recompilation)
public struct StaticConfig: Codable, Sendable {
    public let mesh: MeshConfig
    public let evolution: EvolutionConfig  // Which PDEs to solve
    public let solver: SolverConfig
    public let scheme: SchemeConfig
}

/// Dynamic parameters (no recompilation)
public struct DynamicConfig: Codable, Sendable {
    public let boundaries: BoundaryConfig
    public let transport: TransportConfig
    public let sources: SourcesConfig
    public let pedestal: PedestalConfig?
    public let mhd: MHDConfig
    public let restart: RestartConfig
}
```

## CLI Integration

**RunCommand** uses `GotenxConfigReader` to apply CLI overrides:

```swift
private func loadConfiguration(from path: String) async throws -> SimulationConfiguration {
    // Build CLI overrides map
    var cliOverrides: [String: String] = [:]

    if let value = meshCellCount {
        cliOverrides["runtime.static.mesh.cellCount"] = String(value)
    }
    if let value = timeEnd {
        cliOverrides["time.end"] = String(value)
    }
    // ... more overrides

    // Create hierarchical config reader
    let configReader = try await GotenxConfigReader.create(
        jsonPath: path,
        cliOverrides: cliOverrides
    )

    return try await configReader.fetchConfiguration()
}
```

## Configuration Reloading

**IMPORTANT**: `ConfigReader` does not have a `reload()` method. To reload configuration:

1. **For static changes**: Create a new `GotenxConfigReader` instance
2. **For dynamic hot-reload**: Use `ReloadingJSONProvider` instead of `JSONProvider`

```swift
// Option 1: Manual reload (recreate reader)
let newReader = try await GotenxConfigReader.create(
    jsonPath: configPath,
    cliOverrides: cliOverrides
)
let newConfig = try await newReader.fetchConfiguration()

// Option 2: Automatic reload (future enhancement)
let reloadingProvider = try await ReloadingJSONProvider(
    filePath: FilePath(jsonPath),
    pollInterval: .seconds(5)
)
// Add to ServiceGroup for background monitoring
```

## Configuration Key Mapping

| CLI Flag | Hierarchical Key | Type |
|----------|------------------|------|
| `--mesh-cell-count` | `runtime.static.mesh.cellCount` | Int |
| `--mesh-major-radius` | `runtime.static.mesh.majorRadius` | Double |
| `--mesh-minor-radius` | `runtime.static.mesh.minorRadius` | Double |
| `--time-end` | `time.end` | Double |
| `--initial-time-step` | `time.initialTimeStep` | Double |
| `--output-dir` | `output.directory` | String |
| `--output-format` | `output.format` | Enum |

## Type Conversion

swift-configuration handles type conversion automatically:

```swift
// String → Int
let cellCount = try await configReader.fetchInt(
    forKey: "runtime.static.mesh.cellCount",
    default: 100
)

// String → Double
let majorRadius = try await configReader.fetchDouble(
    forKey: "runtime.static.mesh.majorRadius",
    default: 3.0
)

// String → Bool
let evolveIonHeat = try await configReader.fetchBool(
    forKey: "runtime.static.evolution.ionTemperature",
    default: true
)

// String → Enum (via RawRepresentable)
let geometryType = try await configReader.fetchString(
    forKey: "runtime.static.mesh.geometryType",
    default: "circular"
)
```

## JSON Configuration Example

```json
{
  "runtime": {
    "static": {
      "mesh": {
        "cellCount": 100,
        "majorRadius": 6.2,
        "minorRadius": 2.0,
        "toroidalField": 5.3,
        "geometryType": "circular"
      },
      "evolution": {
        "ionTemperature": true,
        "electronTemperature": true,
        "electronDensity": true,
        "poloidalFlux": false
      }
    },
    "dynamic": {
      "boundaries": {
        "ionTemperature": 100.0,
        "electronTemperature": 100.0,
        "electronDensity": 1e19
      },
      "transport": {
        "modelType": "bohmGyrobohm"
      },
      "sources": {
        "ohmicHeating": true,
        "fusionPower": true
      }
    }
  },
  "time": {
    "start": 0.0,
    "end": 2.0,
    "initialTimeStep": 0.001,
    "adaptive": {
      "enabled": true,
      "safetyFactor": 0.9,
      "minimumTimeStep": 1e-6,
      "maximumTimeStep": 0.1
    }
  },
  "output": {
    "directory": "/tmp/gotenx_results",
    "format": "netcdf",
    "saveInterval": 0.01
  }
}
```

## Environment Variable Format

```bash
# Mesh configuration
export GOTENX_RUNTIME_STATIC_MESH_CELL_COUNT=150
export GOTENX_RUNTIME_STATIC_MESH_MAJOR_RADIUS=6.5
export GOTENX_RUNTIME_STATIC_MESH_MINOR_RADIUS=2.1

# Time configuration
export GOTENX_TIME_END=3.0
export GOTENX_TIME_INITIAL_TIME_STEP=0.0005

# Output configuration
export GOTENX_OUTPUT_DIRECTORY=/scratch/gotenx_output
export GOTENX_OUTPUT_FORMAT=netcdf

# Run with environment overrides
swift run GotenxCLI run --config examples/Configurations/minimal.json
```

## Validation Strategy

Configuration validation happens at **fetch time**, not file load time:

```swift
public func fetchConfiguration() async throws -> SimulationConfiguration {
    let runtime = try await fetchRuntimeConfig()
    let time = try await fetchTimeConfig()
    let output = try await fetchOutputConfig()

    let config = SimulationConfiguration(
        runtime: runtime,
        time: time,
        output: output
    )

    // Validate complete configuration
    try ConfigurationValidator.validate(config)

    return config
}
```

**Validation Rules**:
- `cellCount > 0`: Mesh must have positive cells
- `majorRadius > minorRadius`: Tokamak geometry constraint
- `time.end > time.start`: Valid time range
- `initialTimeStep > 0`: Positive timestep
- Aspect ratio: `1.0 < majorRadius/minorRadius < 10.0`

## Best Practices

1. **Use Hierarchical Keys**: Always use dotted notation for nested config
   ```swift
   "runtime.static.mesh.cellCount"
   "mesh_cell_count"
   ```

2. **Type Safety**: Let swift-configuration handle type conversion
   ```swift
   let value = try await configReader.fetchInt(forKey: "...")  // ✅
   let value = Int(try await configReader.fetchString(...))    // ❌
   ```

3. **Defaults**: Always provide sensible defaults
   ```swift
   try await configReader.fetchInt(forKey: "cellCount", default: 100)
   try await configReader.fetchInt(forKey: "cellCount")
   ```

4. **Validation**: Validate after fetching, not during
   ```swift
   let config = try await configReader.fetchConfiguration()
   try ConfigurationValidator.validate(config)  // ✅
   ```

5. **Immutability**: Configuration structs are immutable - use Builder for modifications
   ```swift
   var builder = SimulationConfiguration.Builder()
   builder.time.end = 5.0
   let newConfig = builder.build()  // ✅
   ```

## Common Pitfalls

❌ **DON'T**: Add providers in reverse priority order (lowest first)
```swift
providers.append(JSONProvider(...))          // JSON (lowest)
providers.append(EnvironmentVariablesProvider().prefixKeys(with: "gotenx"))  // Env
providers.append(InMemoryProvider(...))      // CLI (highest)
// ❌ WRONG - ConfigReader uses FIRST-MATCH priority!
```

✅ **DO**: Add providers in priority order (highest first)
```swift
providers.append(InMemoryProvider(...))      // CLI (highest)
providers.append(EnvironmentVariablesProvider().prefixKeys(with: "gotenx"))  // Env
providers.append(JSONProvider(...))          // JSON (lowest)
// ✅ CORRECT - First provider has highest priority
```

❌ **DON'T**: Mix ConfigValue constructors
```swift
ConfigValue("100", isSecret: false)  // ❌ Wrong - expects ConfigContent
```

✅ **DO**: Use ConfigContent enum
```swift
ConfigValue(.string("100"), isSecret: false)  // ✅ Correct
```

❌ **DON'T**: Use String for FilePath
```swift
JSONProvider(filePath: "/path/to/config.json")  // ❌ Type error
```

✅ **DO**: Use SystemPackage.FilePath
```swift
import SystemPackage
JSONProvider(filePath: FilePath("/path/to/config.json"))  // ✅
```

❌ **DON'T**: Expect reload() on ConfigReader
```swift
configReader.reload()  // ❌ Method doesn't exist
```

✅ **DO**: Create new reader or use ReloadingJSONProvider
```swift
let newReader = try await GotenxConfigReader.create(...)  // ✅
```

## References

- **swift-configuration**: https://github.com/apple/swift-configuration
- **DeepWiki Docs**: https://deepwiki.com/apple/swift-configuration
- **Gotenx Config Examples**: `Examples/Configurations/`

---

*See also: [CLAUDE.md](../CLAUDE.md) for development guidelines*
