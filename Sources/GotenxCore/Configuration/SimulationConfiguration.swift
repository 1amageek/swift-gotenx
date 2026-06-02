// SimulationConfiguration.swift
// Root simulation configuration

import Foundation

/// Root simulation configuration
///
/// Designed for:
/// - Codable (JSON serialization)
/// - Sendable (Swift 6 concurrency)
/// - Validatable (physics constraints)
/// - Composable (builder pattern)
public struct SimulationConfiguration: Codable, Sendable, Equatable {
    /// Runtime configuration
    public let runtime: RuntimeConfiguration

    /// Simulation time range
    public let time: TimeConfiguration

    /// Output configuration
    public let output: OutputConfiguration

    public init(
        runtime: RuntimeConfiguration,
        time: TimeConfiguration,
        output: OutputConfiguration = .default
    ) {
        self.runtime = runtime
        self.time = time
        self.output = output
    }
}

// MARK: - Builder Pattern for Ergonomics

extension SimulationConfiguration {
    /// Create configuration with builder pattern
    public static func build(
        _ configure: (inout Builder) throws -> Void
    ) rethrows -> SimulationConfiguration {
        var builder = Builder()
        try configure(&builder)
        return builder.build()
    }

    public struct Builder {
        public var runtime: RuntimeBuilder = .init()
        public var time: TimeBuilder = .init()
        public var output: OutputBuilder = .init()

        public init() {}

        public func build() -> SimulationConfiguration {
            SimulationConfiguration(
                runtime: runtime.build(),
                time: time.build(),
                output: output.build()
            )
        }
    }

    public struct RuntimeBuilder {
        public var `static`: StaticBuilder = .init()
        public var dynamic: DynamicBuilder = .init()

        public func build() -> RuntimeConfiguration {
            RuntimeConfiguration(
                static: `static`.build(),
                dynamic: dynamic.build()
            )
        }
    }

    public struct StaticBuilder {
        public var mesh: MeshBuilder = .init()
        public var evolution: EvolutionConfig = .default
        public var solver: SolverConfig = .default
        public var scheme: SchemeConfig = .default

        public func build() -> StaticConfig {
            StaticConfig(
                mesh: mesh.build(),
                evolution: evolution,
                solver: solver,
                scheme: scheme
            )
        }
    }

    public struct MeshBuilder {
        public var cellCount: Int = 100
        public var majorRadius: Float = 3.0
        public var minorRadius: Float = 1.0
        public var toroidalField: Float = 2.5
        public var geometryType: GeometryType = .circular

        public func build() -> MeshConfig {
            MeshConfig(
                cellCount: cellCount,
                majorRadius: majorRadius,
                minorRadius: minorRadius,
                toroidalField: toroidalField,
                geometryType: geometryType
            )
        }
    }

    public struct DynamicBuilder {
        public var boundaries: BoundaryConfig = BoundaryConfig(
            ionTemperature: 100.0,
            electronTemperature: 100.0,
            electronDensity: 1e19
        )
        public var transport: TransportConfig = .defaultConstant
        public var sources: SourcesConfig = .default
        public var pedestal: PedestalConfig? = nil
        public var mhd: MHDConfig = .default
        public var restart: RestartConfig = .default
        public var initialProfile: InitialProfileConfig = .default

        public func build() -> DynamicConfig {
            DynamicConfig(
                boundaries: boundaries,
                transport: transport,
                sources: sources,
                pedestal: pedestal,
                mhd: mhd,
                restart: restart,
                initialProfile: initialProfile
            )
        }
    }

    public struct TimeBuilder {
        public var start: Float = 0.0
        public var end: Float = 1.0
        public var initialTimeStep: Float = 1e-3
        public var adaptive: AdaptiveTimestepConfig? = .default

        public func build() -> TimeConfiguration {
            TimeConfiguration(
                start: start,
                end: end,
                initialTimeStep: initialTimeStep,
                adaptive: adaptive
            )
        }
    }

    public struct OutputBuilder {
        public var saveInterval: Float? = nil
        public var directory: String = "/tmp/gotenx_results"
        public var format: OutputFormat = .json

        public func build() -> OutputConfiguration {
            OutputConfiguration(
                saveInterval: saveInterval,
                directory: directory,
                format: format
            )
        }
    }
}
