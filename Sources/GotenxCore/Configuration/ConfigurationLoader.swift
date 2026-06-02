// ConfigurationLoader.swift
// Orchestrates configuration loading with hierarchical override

import Foundation

/// Configuration loader with hierarchical override
///
/// Override priority (highest to lowest):
/// 1. CLI arguments (--mesh-cell-count=200)
/// 2. Environment variables (GOTENX_RUNTIME_STATIC_MESH_CELL_COUNT=200)
/// 3. JSON configuration file
/// 4. Default values
public struct ConfigurationLoader: Sendable {
    private let providers: [any ConfigurationProvider]
    private let overrides: [ConfigurationOverrides]

    /// Initialize with custom providers
    public init(
        providers: [any ConfigurationProvider],
        overrides: [ConfigurationOverrides] = []
    ) {
        // Sort by priority (highest first)
        self.providers = providers.sorted { $0.priority > $1.priority }
        self.overrides = overrides
    }

    /// Initialize with standard providers
    public static func standard(
        configFile: String? = nil,
        cliArguments: [String: String] = [:]
    ) -> ConfigurationLoader {
        var providers: [any ConfigurationProvider] = []

        // JSON provider
        if let configFile = configFile {
            providers.append(JSONConfigurationProvider(filePath: configFile))
        }

        // Default provider (lowest priority)
        providers.append(DefaultConfigurationProvider())

        return ConfigurationLoader(
            providers: providers,
            overrides: [
                ConfigurationOverrides.fromEnvironment(),
                ConfigurationOverrides.fromCLI(cliArguments)
            ]
        )
    }

    /// Load configuration with hierarchical override
    public func load() async throws -> SimulationConfiguration {
        // Start with base configuration from lowest priority provider
        var config: SimulationConfiguration?

        // Load from providers in reverse order (lowest to highest priority)
        for provider in providers.reversed() {
            if let loaded = try await provider.load() {
                if config == nil {
                    config = loaded
                } else {
                    // Merge/override would go here
                    // For now, higher priority replaces lower priority
                    config = loaded
                }
            }
        }

        guard var finalConfig = config else {
            throw ConfigurationError.missingRequired(key: "configuration")
        }

        for override in overrides where !override.isEmpty {
            finalConfig = try Self.applyingOverrides(
                to: finalConfig,
                overrides: override
            )
        }

        // Validate final configuration
        try ConfigurationValidator.validate(finalConfig)

        return finalConfig
    }

    /// Load configuration from a JSON file
    public static func loadFromJSON(_ filePath: String) async throws -> SimulationConfiguration {
        let provider = JSONConfigurationProvider(filePath: filePath)
        guard let config = try await provider.load() else {
            throw ConfigurationError.missingRequired(key: "configuration file: \(filePath)")
        }

        try ConfigurationValidator.validate(config)
        return config
    }

    /// Load configuration with builder and overrides
    public static func loadWithOverrides(
        baseConfig: SimulationConfiguration,
        overrides: ConfigurationOverrides
    ) throws -> SimulationConfiguration {
        let config = try applyingOverrides(to: baseConfig, overrides: overrides)
        try ConfigurationValidator.validate(config)
        return config
    }

    private static func applyingOverrides(
        to baseConfig: SimulationConfiguration,
        overrides: ConfigurationOverrides
    ) throws -> SimulationConfiguration {
        var builder = SimulationConfiguration.Builder()

        // Apply base config to builder
        builder.runtime.static.mesh = baseConfig.runtime.static.mesh.toBuilder()
        builder.runtime.static.evolution = baseConfig.runtime.static.evolution
        builder.runtime.static.solver = baseConfig.runtime.static.solver
        builder.runtime.static.scheme = baseConfig.runtime.static.scheme

        builder.runtime.dynamic.boundaries = baseConfig.runtime.dynamic.boundaries
        builder.runtime.dynamic.transport = baseConfig.runtime.dynamic.transport
        builder.runtime.dynamic.sources = baseConfig.runtime.dynamic.sources
        builder.runtime.dynamic.pedestal = baseConfig.runtime.dynamic.pedestal
        builder.runtime.dynamic.mhd = baseConfig.runtime.dynamic.mhd
        builder.runtime.dynamic.restart = baseConfig.runtime.dynamic.restart

        builder.time.start = baseConfig.time.start
        builder.time.end = baseConfig.time.end
        builder.time.initialTimeStep = baseConfig.time.initialTimeStep
        builder.time.adaptive = baseConfig.time.adaptive

        builder.output.saveInterval = baseConfig.output.saveInterval
        builder.output.directory = baseConfig.output.directory
        builder.output.format = baseConfig.output.format

        // Apply overrides
        if let cellCount = overrides.meshCellCount {
            builder.runtime.static.mesh.cellCount = cellCount
        }
        if let majorRadius = overrides.meshMajorRadius {
            builder.runtime.static.mesh.majorRadius = majorRadius
        }
        if let minorRadius = overrides.meshMinorRadius {
            builder.runtime.static.mesh.minorRadius = minorRadius
        }
        if let timeEnd = overrides.timeEnd {
            builder.time.end = timeEnd
        }
        if let initialTimeStep = overrides.initialTimeStep {
            builder.time.initialTimeStep = initialTimeStep
        }
        if let outputDir = overrides.outputDirectory {
            builder.output.directory = outputDir
        }

        return builder.build()
    }
}

/// Configuration overrides from CLI/environment
public struct ConfigurationOverrides: Sendable {
    public var meshCellCount: Int?
    public var meshMajorRadius: Float?
    public var meshMinorRadius: Float?
    public var timeEnd: Float?
    public var initialTimeStep: Float?
    public var outputDirectory: String?

    public init(
        meshCellCount: Int? = nil,
        meshMajorRadius: Float? = nil,
        meshMinorRadius: Float? = nil,
        timeEnd: Float? = nil,
        initialTimeStep: Float? = nil,
        outputDirectory: String? = nil
    ) {
        self.meshCellCount = meshCellCount
        self.meshMajorRadius = meshMajorRadius
        self.meshMinorRadius = meshMinorRadius
        self.timeEnd = timeEnd
        self.initialTimeStep = initialTimeStep
        self.outputDirectory = outputDirectory
    }

    public var isEmpty: Bool {
        meshCellCount == nil &&
        meshMajorRadius == nil &&
        meshMinorRadius == nil &&
        timeEnd == nil &&
        initialTimeStep == nil &&
        outputDirectory == nil
    }

    /// Parse from CLI arguments
    public static func fromCLI(_ arguments: [String: String]) -> ConfigurationOverrides {
        var overrides = ConfigurationOverrides()

        if let value = arguments["mesh-cell-count"], let intValue = Int(value) {
            overrides.meshCellCount = intValue
        }
        if let value = arguments["mesh-major-radius"], let floatValue = Float(value) {
            overrides.meshMajorRadius = floatValue
        }
        if let value = arguments["mesh-minor-radius"], let floatValue = Float(value) {
            overrides.meshMinorRadius = floatValue
        }
        if let value = arguments["time-end"], let floatValue = Float(value) {
            overrides.timeEnd = floatValue
        }
        if let value = arguments["initial-time-step"], let floatValue = Float(value) {
            overrides.initialTimeStep = floatValue
        }
        if let value = arguments["output-dir"] {
            overrides.outputDirectory = value
        }

        return overrides
    }

    /// Parse from environment variables
    public static func fromEnvironment(prefix: String = "GOTENX_") -> ConfigurationOverrides {
        var overrides = ConfigurationOverrides()
        let env = ProcessInfo.processInfo.environment

        if let value = env[prefix + "RUNTIME_STATIC_MESH_CELL_COUNT"], let intValue = Int(value) {
            overrides.meshCellCount = intValue
        }
        if let value = env[prefix + "RUNTIME_STATIC_MESH_MAJOR_RADIUS"], let floatValue = Float(value) {
            overrides.meshMajorRadius = floatValue
        }
        if let value = env[prefix + "RUNTIME_STATIC_MESH_MINOR_RADIUS"], let floatValue = Float(value) {
            overrides.meshMinorRadius = floatValue
        }
        if let value = env[prefix + "TIME_END"], let floatValue = Float(value) {
            overrides.timeEnd = floatValue
        }
        if let value = env[prefix + "TIME_INITIAL_TIME_STEP"], let floatValue = Float(value) {
            overrides.initialTimeStep = floatValue
        }
        if let value = env[prefix + "OUTPUT_DIRECTORY"] {
            overrides.outputDirectory = value
        }

        return overrides
    }
}

/// Extension to support builder from existing config
extension MeshConfig {
    public func toBuilder() -> SimulationConfiguration.MeshBuilder {
        var builder = SimulationConfiguration.MeshBuilder()
        builder.cellCount = self.cellCount
        builder.majorRadius = self.majorRadius
        builder.minorRadius = self.minorRadius
        builder.toroidalField = self.toroidalField
        builder.geometryType = self.geometryType
        return builder
    }
}
