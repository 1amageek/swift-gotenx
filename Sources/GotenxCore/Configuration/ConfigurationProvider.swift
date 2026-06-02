// ConfigurationProvider.swift
// Protocol for configuration sources

import Foundation

/// Protocol for configuration sources
///
/// Providers can read configuration from different sources:
/// - JSON files
/// - Environment variables
/// - Command-line arguments
/// - In-memory defaults
public protocol ConfigurationProvider: Sendable {
    /// Load configuration from this provider
    /// Returns nil if configuration is not available from this source
    func load() async throws -> SimulationConfiguration?

    /// Priority of this provider (higher = takes precedence)
    var priority: Int { get }
}

/// Provider priority levels
public enum ProviderPriority {
    /// Defaults have lowest priority
    public static let defaults = 0

    /// JSON files have medium priority
    public static let json = 100

    /// Environment variables have high priority
    public static let environment = 200

    /// CLI arguments have highest priority
    public static let cli = 300
}

/// Default configuration provider
public struct DefaultConfigurationProvider: ConfigurationProvider {
    public let priority: Int = ProviderPriority.defaults

    public init() {}

    public func load() async throws -> SimulationConfiguration? {
        // Return sensible defaults for ITER-like tokamak
        return SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: MeshConfig(
                        cellCount: 100,
                        majorRadius: 6.2,
                        minorRadius: 2.0,
                        toroidalField: 5.3,
                        geometryType: .circular
                    ),
                    evolution: .default,
                    solver: .default,
                    scheme: .default
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: 100.0,
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: try TransportConfig(
                        modelType: .constant,
                        parameters: [
                            "ionHeatDiffusivity": 0.01,
                            "electronHeatDiffusivity": 0.01,
                            "particleDiffusivity": 0.005
                        ]
                    ),
                    sources: .default,
                    pedestal: nil
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 2.0,
                initialTimeStep: 1e-5,
                adaptive: .default
            ),
            output: .default
        )
    }
}

/// JSON file configuration provider
public struct JSONConfigurationProvider: ConfigurationProvider {
    public let priority: Int = ProviderPriority.json
    private let filePath: String

    public init(filePath: String) {
        self.filePath = filePath
    }

    public func load() async throws -> SimulationConfiguration? {
        let fileURL = URL(fileURLWithPath: filePath)

        // Check if file exists
        guard FileManager.default.fileExists(atPath: filePath) else {
            return nil
        }

        // Read and decode JSON
        let data = try Data(contentsOf: fileURL)
        let decoder = JSONDecoder()
        return try decoder.decode(SimulationConfiguration.self, from: data)
    }
}

/// Environment variable configuration provider
///
/// Reads configuration overrides from environment variables with prefix "GOTENX_"
/// Example: GOTENX_RUNTIME_STATIC_MESH_CELL_COUNT=200, GOTENX_TIME_END=5.0
public struct EnvironmentConfigurationProvider: ConfigurationProvider {
    public let priority: Int = ProviderPriority.environment
    private let prefix: String

    public init(prefix: String = "GOTENX_") {
        self.prefix = prefix
    }

    public func load() async throws -> SimulationConfiguration? {
        // For now, environment variables are applied as overrides
        // in ConfigurationLoader. Return nil to indicate no full config.
        // Future: Could build partial config from env vars
        return nil
    }

    /// Get environment variable value
    public func value(for key: String) -> String? {
        let envKey = prefix + key.uppercased()
        return ProcessInfo.processInfo.environment[envKey]
    }
}

/// CLI argument configuration provider
///
/// Parses command-line arguments to override configuration
public struct CLIConfigurationProvider: ConfigurationProvider {
    public let priority: Int = ProviderPriority.cli
    private let arguments: [String: String]

    public init(arguments: [String: String]) {
        self.arguments = arguments
    }

    public func load() async throws -> SimulationConfiguration? {
        // CLI arguments are applied as overrides in ConfigurationLoader
        // Return nil to indicate no full config
        return nil
    }

    /// Get CLI argument value
    public func value(for key: String) -> String? {
        return arguments[key]
    }
}
