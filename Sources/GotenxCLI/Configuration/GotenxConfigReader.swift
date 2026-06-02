// GotenxConfigReader.swift
// ConfigReader-based configuration loading for Gotenx

import Configuration
import Foundation
import SystemPackage
import GotenxCore

/// TORAX-specific ConfigReader wrapper
///
/// Provides hierarchical configuration loading with the following priority:
/// 1. CLI arguments (highest)
/// 2. Environment variables
/// 3. JSON file (reloadable)
/// 4. Default values (lowest)
public actor GotenxConfigReader {
    private let configReader: ConfigReader
    private let jsonPath: String
    private let cliOverrides: [String: String]
    private let environment: [String: String]

    private init(
        configReader: ConfigReader,
        jsonPath: String,
        cliOverrides: [String: String],
        environment: [String: String]
    ) {
        self.configReader = configReader
        self.jsonPath = jsonPath
        self.cliOverrides = cliOverrides
        self.environment = environment
    }

    /// Create GotenxConfigReader with hierarchical providers
    ///
    /// - Parameters:
    ///   - jsonPath: Path to JSON configuration file
    ///   - cliOverrides: CLI argument overrides as key-value pairs
    /// - Returns: Configured GotenxConfigReader
    public static func create(
        jsonPath: String,
        cliOverrides: [String: String] = [:]
    ) async throws -> GotenxConfigReader {
        try await create(
            jsonPath: jsonPath,
            cliOverrides: cliOverrides,
            environment: ProcessInfo.processInfo.environment
        )
    }

    /// Create GotenxConfigReader with an explicit environment snapshot.
    ///
    /// Use this overload in tests to avoid mutating process-wide environment variables.
    public static func create(
        jsonPath: String,
        cliOverrides: [String: String] = [:],
        environment: [String: String]
    ) async throws -> GotenxConfigReader {
        var providers: [any ConfigProvider] = []

        // IMPORTANT: ConfigReader uses FIRST-MATCH priority order
        // First provider in array has HIGHEST priority
        // (This is the OPPOSITE of what the initial assumption was)

        // Priority 1 (highest): CLI arguments
        if !cliOverrides.isEmpty {
            // Convert [String: String] to [String: ConfigValue]
            // Note: Values are provided as strings, ConfigReader will handle type conversion
            let configValues = cliOverrides.mapValues { value in
                // Attempt to parse as different types
                if let intValue = Int(value) {
                    return ConfigValue(.int(intValue), isSecret: false)
                } else if let doubleValue = Double(value) {
                    return ConfigValue(.double(doubleValue), isSecret: false)
                } else if let boolValue = Bool(value) {
                    return ConfigValue(.bool(boolValue), isSecret: false)
                } else {
                    return ConfigValue(.string(value), isSecret: false)
                }
            }
            providers.append(
                InMemoryProvider(values: configValues)
            )
        }

        // Priority 2: Environment variables
        providers.append(
            EnvironmentVariablesProvider(environmentVariables: environment)
                .prefixKeys(with: "gotenx")
        )

        // Priority 3 (lowest): JSON file
        let jsonProvider = try await JSONProvider(filePath: FilePath(jsonPath))
        providers.append(jsonProvider)

        let reader = ConfigReader(providers: providers)
        return GotenxConfigReader(
            configReader: reader,
            jsonPath: jsonPath,
            cliOverrides: cliOverrides,
            environment: environment
        )
    }

    // MARK: - Configuration Fetching

    /// Fetch complete SimulationConfiguration
    public func fetchConfiguration() async throws -> SimulationConfiguration {
        // Fetch basic configuration first
        let time = try await fetchTimeConfig()
        let runtime = try await fetchRuntimeConfig(time: time)
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

    /// Compute CFL-safe transport parameter defaults
    ///
    /// Design constraint: CFL = χ * Δt / Δx² < cflLimit
    ///   => χ_max = cflLimit * Δx² / Δt
    ///
    /// - Parameters:
    ///   - modelType: Transport model type
    ///   - mesh: Mesh configuration (for Δx)
    ///   - time: Time configuration (for Δt)
    ///   - cflLimit: CFL stability limit (default: 0.5)
    /// - Returns: CFL-safe default parameters
    private func computeCFLSafeDefaults(
        modelType: TransportModelType,
        mesh: MeshConfig,
        time: TimeConfiguration,
        cflLimit: Float = 0.5
    ) throws -> [String: Float] {
        guard mesh.cellCount > 0 else {
            throw ConfigurationError.invalidValue(
                key: "runtime.static.mesh.cellCount",
                value: "\(mesh.cellCount)",
                reason: "Cell count must be positive before computing CFL-safe transport defaults"
            )
        }

        guard mesh.minorRadius.isFinite, mesh.minorRadius > 0 else {
            throw ConfigurationError.invalidValue(
                key: "runtime.static.mesh.minorRadius",
                value: "\(mesh.minorRadius)",
                reason: "Minor radius must be finite and positive before computing CFL-safe transport defaults"
            )
        }

        guard time.initialTimeStep.isFinite, time.initialTimeStep > 0 else {
            throw ConfigurationError.invalidValue(
                key: "time.initialTimeStep",
                value: "\(time.initialTimeStep)",
                reason: "Initial timestep must be finite and positive before computing CFL-safe transport defaults"
            )
        }

        guard cflLimit.isFinite, cflLimit > 0 else {
            throw ConfigurationError.invalidValue(
                key: "transport.cflLimit",
                value: "\(cflLimit)",
                reason: "CFL limit must be finite and positive"
            )
        }

        // Calculate cell spacing
        let dx = mesh.minorRadius / Float(mesh.cellCount)
        let timeStep = time.initialTimeStep

        // CFL-safe maximum diffusivity
        let chiMax = cflLimit * dx * dx / timeStep
        guard chiMax.isFinite, chiMax > 0 else {
            throw ConfigurationError.invalidValue(
                key: "runtime.dynamic.transport",
                value: "\(chiMax)",
                reason: "CFL-safe diffusivity default must be finite and positive"
            )
        }

        switch modelType {
        case .constant:
            // Conservative defaults: Use 90% of CFL limit for safety margin
            let safetyFactor: Float = 0.9
            return [
                "ionHeatDiffusivity": chiMax * safetyFactor,
                "electronHeatDiffusivity": chiMax * safetyFactor,
                "particleDiffusivity": chiMax * safetyFactor * 0.2  // Typically lower
            ]

        case .bohmGyrobohm, .qlknn:
            // Computed by model, no explicit parameters needed
            return [:]

        case .densityTransition:
            // Model-specific parameters (not CFL-limited)
            return [
                "riCoefficient": 0.5,
                "transitionDensity": 2.5e19,
                "transitionWidth": 0.5e19,
                "ionMassNumber": 2.0
            ]
        }
    }

    // MARK: - Runtime Configuration

    private func fetchRuntimeConfig(time: TimeConfiguration) async throws -> RuntimeConfiguration {
        let staticConfig = try await fetchStaticConfig()
        let dynamicConfig = try await fetchDynamicConfig(
            mesh: staticConfig.mesh,
            time: time
        )

        return RuntimeConfiguration(
            static: staticConfig,
            dynamic: dynamicConfig
        )
    }

    private func fetchStaticConfig() async throws -> StaticConfig {
        // Mesh configuration
        let meshCellCount = try await fetchInt(
            forKeys: ["runtime.static.mesh.cellCount"],
            default: 100
        )
        let majorRadius = try await configReader.fetchDouble(
            forKey: "runtime.static.mesh.majorRadius",
            default: 3.0
        )
        let minorRadius = try await configReader.fetchDouble(
            forKey: "runtime.static.mesh.minorRadius",
            default: 1.0
        )
        let toroidalField = try await configReader.fetchDouble(
            forKey: "runtime.static.mesh.toroidalField",
            default: 2.5
        )
        let geometryType = try await fetchEnum(
            forKey: "runtime.static.mesh.geometryType",
            default: GeometryType.circular
        )

        let mesh = MeshConfig(
            cellCount: meshCellCount,
            majorRadius: Float(majorRadius),
            minorRadius: Float(minorRadius),
            toroidalField: Float(toroidalField),
            geometryType: geometryType
        )

        // Evolution configuration
        let evolveIonHeat = try await configReader.fetchBool(
            forKey: "runtime.static.evolution.ionTemperature",
            default: true
        )
        let evolveElectronHeat = try await configReader.fetchBool(
            forKey: "runtime.static.evolution.electronTemperature",
            default: true
        )
        let evolveElectronDensity = try await fetchBool(
            forKeys: ["runtime.static.evolution.electronDensity"],
            default: true
        )
        let evolvePoloidalFlux = try await fetchBool(
            forKeys: ["runtime.static.evolution.poloidalFlux"],
            default: false
        )

        let evolution = EvolutionConfig(
            ionHeat: evolveIonHeat,
            electronHeat: evolveElectronHeat,
            electronDensity: evolveElectronDensity,
            poloidalFlux: evolvePoloidalFlux
        )

        // Solver configuration
        let solverType = try await configReader.fetchString(
            forKey: "runtime.static.solver.type",
            default: "linear"
        )
        let solverMaxIter = try await fetchInt(
            forKeys: ["runtime.static.solver.maximumIterations"],
            default: 30
        )
        let solverTolerance = try await configReader.fetchDouble(
            forKey: "runtime.static.solver.tolerance",
            default: 1e-6
        )

        let solver = SolverConfig(
            type: solverType,
            tolerance: Float(solverTolerance),
            maximumIterations: solverMaxIter
        )

        // Scheme configuration
        let theta = try await configReader.fetchDouble(
            forKey: "runtime.static.scheme.theta",
            default: 1.0
        )

        let scheme = SchemeConfig(theta: Float(theta))

        return StaticConfig(
            mesh: mesh,
            evolution: evolution,
            solver: solver,
            scheme: scheme
        )
    }

    private func fetchDynamicConfig(mesh: MeshConfig, time: TimeConfiguration) async throws -> DynamicConfig {
        // Boundary conditions
        let ionTemp = try await configReader.fetchDouble(
            forKey: "runtime.dynamic.boundaries.ionTemperature",
            default: 100.0
        )
        let electronTemp = try await configReader.fetchDouble(
            forKey: "runtime.dynamic.boundaries.electronTemperature",
            default: 100.0
        )
        let electronDensity = try await fetchDouble(
            forKeys: ["runtime.dynamic.boundaries.electronDensity"],
            default: 1e19
        )

        let boundaries = BoundaryConfig(
            ionTemperature: Float(ionTemp),
            electronTemperature: Float(electronTemp),
            electronDensity: Float(electronDensity)
        )

        // Transport configuration
        let transport = try await fetchTransportConfig(mesh: mesh, time: time)

        // Sources configuration
        let sources = try await fetchSourcesConfig()

        // Pedestal configuration (optional)
        let pedestalModel = try await configReader.fetchString(
            forKey: "runtime.dynamic.pedestal.model",
            default: "none"
        )
        let pedestal = pedestalModel != "none" ? PedestalConfig(model: pedestalModel) : nil

        // MHD configuration
        let mhd = try await fetchMHDConfig()

        // Restart configuration
        let restart = try await fetchRestartConfig()

        return DynamicConfig(
            boundaries: boundaries,
            transport: transport,
            sources: sources,
            pedestal: pedestal,
            mhd: mhd,
            restart: restart
        )
    }

    private func fetchTransportConfig(mesh: MeshConfig, time: TimeConfiguration) async throws -> TransportConfig {
        let modelType = try await fetchEnum(
            forKey: "runtime.dynamic.transport.modelType",
            default: TransportModelType.constant
        )

        try validateConfiguredTransportParameterKeys(for: modelType)

        var parameters = try await fetchTransportParameters(modelType: modelType)
        if parameters.isEmpty {
            parameters = try computeCFLSafeDefaults(
                modelType: modelType,
                mesh: mesh,
                time: time
            )
        }

        return try TransportConfig(
            modelType: modelType,
            parameters: parameters
        )
    }

    private func fetchTransportParameters(modelType: TransportModelType) async throws -> [String: Float] {
        var parameters: [String: Float] = [:]
        for key in modelType.allowedParameterKeys.sorted() {
            if let value = try await configReader.fetchDouble(forKey: "runtime.dynamic.transport.parameters.\(key)") {
                parameters[key] = Float(value)
            }
        }

        return parameters
    }

    private func validateConfiguredTransportParameterKeys(for modelType: TransportModelType) throws {
        let configuredKeys = try configuredTransportParameterKeys()
        let allowedKeys = modelType.allowedParameterKeys

        for key in configuredKeys.sorted() where !allowedKeys.contains(key) {
            throw ConfigurationValidationError.unknownTransportParameter(
                parameter: key,
                modelType: modelType,
                allowed: allowedKeys.sorted()
            )
        }
    }

    private func configuredTransportParameterKeys() throws -> Set<String> {
        var keys = try jsonTransportParameterKeys()
        keys.formUnion(Self.transportParameterKeys(inOverrides: cliOverrides))
        keys.formUnion(Self.transportParameterKeys(inEnvironment: environment))
        return keys
    }

    private func jsonTransportParameterKeys() throws -> Set<String> {
        let data = try Data(contentsOf: URL(fileURLWithPath: jsonPath))
        let root = try JSONSerialization.jsonObject(with: data)

        guard let rootDictionary = root as? [String: Any] else {
            throw ConfigurationError.invalidValue(
                key: "configuration",
                value: "\(Swift.type(of: root))",
                reason: "Configuration root must be a JSON object"
            )
        }

        guard let transport = Self.dictionaryValue(
            in: rootDictionary,
            path: ["runtime", "dynamic", "transport"]
        ) else {
            return []
        }

        guard let parametersValue = transport["parameters"] else {
            return []
        }

        guard let parameters = parametersValue as? [String: Any] else {
            throw ConfigurationError.invalidValue(
                key: "runtime.dynamic.transport.parameters",
                value: "\(Swift.type(of: parametersValue))",
                reason: "Transport parameters must be a JSON object"
            )
        }

        return Set(parameters.keys)
    }

    private static func dictionaryValue(
        in root: [String: Any],
        path: [String]
    ) -> [String: Any]? {
        var current: Any = root

        for component in path {
            guard let dictionary = current as? [String: Any],
                  let next = dictionary[component] else {
                return nil
            }
            current = next
        }

        return current as? [String: Any]
    }

    private static func transportParameterKeys(inOverrides overrides: [String: String]) -> Set<String> {
        let prefix = "runtime.dynamic.transport.parameters."
        return Set(
            overrides.keys.compactMap { key in
                guard key.hasPrefix(prefix) else {
                    return nil
                }
                return String(key.dropFirst(prefix.count))
            }
        )
    }

    private static func transportParameterKeys(inEnvironment environment: [String: String]) -> Set<String> {
        let prefix = environmentName(
            forComponents: ["gotenx", "runtime", "dynamic", "transport", "parameters"]
        ) + "_"
        let knownParameterNames = Dictionary(
            uniqueKeysWithValues: allTransportParameterKeys.map { key in
                (environmentName(forComponents: [key]), key)
            }
        )

        return Set(
            environment.keys.compactMap { key in
                guard key.hasPrefix(prefix) else {
                    return nil
                }

                let encodedParameterName = String(key.dropFirst(prefix.count))
                return knownParameterNames[encodedParameterName] ?? encodedParameterName
            }
        )
    }

    private static var allTransportParameterKeys: Set<String> {
        Set(TransportModelType.allCases.flatMap { $0.allowedParameterKeys })
    }

    private static func environmentName(forComponents components: [String]) -> String {
        components
            .map(environmentComponentName)
            .joined(separator: "_")
    }

    private static func environmentComponentName(_ component: String) -> String {
        var normalized = ""
        var previousWasLowercase = false

        for character in component {
            if previousWasLowercase, character.isUppercase {
                normalized.append("_")
            }

            normalized.append(character)
            previousWasLowercase = character.isLowercase
        }

        return normalized
            .uppercased()
            .map { character in
                character.isLetter || character.isNumber ? String(character) : "_"
            }
            .joined()
    }

    private func fetchSourcesConfig() async throws -> SourcesConfig {
        let ohmicEnabled = try await configReader.fetchBool(
            forKey: "runtime.dynamic.sources.ohmicHeating",
            default: true
        )
        let fusionEnabled = try await configReader.fetchBool(
            forKey: "runtime.dynamic.sources.fusionPower",
            default: true
        )
        let ionElectronEnabled = try await configReader.fetchBool(
            forKey: "runtime.dynamic.sources.ionElectronExchange",
            default: true
        )
        let bremsstrahlungEnabled = try await configReader.fetchBool(
            forKey: "runtime.dynamic.sources.bremsstrahlung",
            default: true
        )

        return SourcesConfig(
            ohmicHeating: ohmicEnabled,
            fusionPower: fusionEnabled,
            ionElectronExchange: ionElectronEnabled,
            bremsstrahlung: bremsstrahlungEnabled
        )
    }

    private func fetchMHDConfig() async throws -> MHDConfig {
        let sawtoothEnabled = try await configReader.fetchBool(
            forKey: "runtime.dynamic.mhd.sawtoothEnabled",
            default: false
        )

        // Sawtooth parameters
        let minimumRadius = try await configReader.fetchDouble(
            forKey: "runtime.dynamic.mhd.sawtooth.minimumRadius",
            default: 0.2
        )
        let sCritical = try await configReader.fetchDouble(
            forKey: "runtime.dynamic.mhd.sawtooth.sCritical",
            default: 0.2
        )
        let minimumCrashInterval = try await configReader.fetchDouble(
            forKey: "runtime.dynamic.mhd.sawtooth.minimumCrashInterval",
            default: 0.01
        )
        let flatteningFactor = try await configReader.fetchDouble(
            forKey: "runtime.dynamic.mhd.sawtooth.flatteningFactor",
            default: 1.01
        )
        let mixingRadiusMultiplier = try await configReader.fetchDouble(
            forKey: "runtime.dynamic.mhd.sawtooth.mixingRadiusMultiplier",
            default: 1.5
        )
        let crashStepDuration = try await configReader.fetchDouble(
            forKey: "runtime.dynamic.mhd.sawtooth.crashStepDuration",
            default: 1e-3
        )

        let sawtoothParameters = SawtoothParameters(
            minimumRadius: Float(minimumRadius),
            sCritical: Float(sCritical),
            minimumCrashInterval: Float(minimumCrashInterval),
            flatteningFactor: Float(flatteningFactor),
            mixingRadiusMultiplier: Float(mixingRadiusMultiplier),
            crashStepDuration: Float(crashStepDuration)
        )

        let ntmEnabled = try await configReader.fetchBool(
            forKey: "runtime.dynamic.mhd.ntmEnabled",
            default: false
        )

        return MHDConfig(
            sawtoothEnabled: sawtoothEnabled,
            sawtoothParameters: sawtoothParameters,
            ntmEnabled: ntmEnabled
        )
    }

    private func fetchRestartConfig() async throws -> RestartConfig {
        let doRestart = try await configReader.fetchBool(
            forKey: "runtime.dynamic.restart.doRestart",
            default: false
        )

        let filename = try await configReader.fetchString(
            forKey: "runtime.dynamic.restart.filename"
        )

        let time = try await configReader.fetchDouble(
            forKey: "runtime.dynamic.restart.time"
        )

        let stitch = try await configReader.fetchBool(
            forKey: "runtime.dynamic.restart.stitch",
            default: true
        )

        return RestartConfig(
            filename: filename,
            time: time.map { Float($0) },
            doRestart: doRestart,
            stitch: stitch
        )
    }

    // MARK: - Time Configuration

    private func fetchTimeConfig() async throws -> TimeConfiguration {
        let start = try await configReader.fetchDouble(
            forKey: "time.start",
            default: 0.0
        )
        let end = try await configReader.fetchDouble(
            forKey: "time.end",
            default: 1.0
        )
        let initialTimeStep = try await fetchDouble(
            forKeys: ["time.initialTimeStep"],
            default: 1e-3
        )

        // Adaptive timestep configuration (optional)
        let adaptiveEnabled = try await configReader.fetchBool(
            forKey: "time.adaptive.enabled",
            default: true
        )

        let adaptive: AdaptiveTimestepConfig?
        if adaptiveEnabled {
            let safetyFactor = try await configReader.fetchDouble(
                forKey: "time.adaptive.safetyFactor",
                default: 0.9
            )
            let minimumTimeStep = try await fetchDouble(
                forKeys: ["time.adaptive.minimumTimeStep"],
                default: 1e-6
            )
            let maximumTimeStep = try await fetchDouble(
                forKeys: ["time.adaptive.maximumTimeStep"],
                default: 1e-1
            )

            adaptive = AdaptiveTimestepConfig(
                minimumTimeStep: Float(minimumTimeStep),
                maximumTimeStep: Float(maximumTimeStep),
                safetyFactor: Float(safetyFactor)
            )
        } else {
            adaptive = nil
        }

        return TimeConfiguration(
            start: Float(start),
            end: Float(end),
            initialTimeStep: Float(initialTimeStep),
            adaptive: adaptive
        )
    }

    // MARK: - Output Configuration

    private func fetchOutputConfig() async throws -> OutputConfiguration {
        let saveInterval = try await configReader.fetchDouble(
            forKey: "output.saveInterval"
        )

        let directory = try await configReader.fetchString(
            forKey: "output.directory",
            default: "/tmp/gotenx_results"
        )

        let format = try await fetchEnum(
            forKey: "output.format",
            default: GotenxCore.OutputFormat.json
        )

        return OutputConfiguration(
            saveInterval: saveInterval.map { Float($0) },
            directory: directory,
            format: format
        )
    }

    // MARK: - Generic Helpers

    private func fetchInt(forKeys keys: [String], default defaultValue: Int) async throws -> Int {
        for key in keys {
            if let value = try await configReader.fetchInt(forKey: key) {
                return value
            }
        }
        return defaultValue
    }

    private func fetchDouble(forKeys keys: [String], default defaultValue: Double) async throws -> Double {
        try await fetchDouble(forKeys: keys) ?? defaultValue
    }

    private func fetchDouble(forKeys keys: [String]) async throws -> Double? {
        for key in keys {
            if let value = try await configReader.fetchDouble(forKey: key) {
                return value
            }
        }
        return nil
    }

    private func fetchBool(forKeys keys: [String], default defaultValue: Bool) async throws -> Bool {
        for key in keys {
            if let value = try await configReader.fetchBool(forKey: key) {
                return value
            }
        }
        return defaultValue
    }

    /// Fetch string-based enum with validation
    ///
    /// Provides type-safe enum conversion with clear error messages.
    ///
    /// - Parameters:
    ///   - key: Configuration key (hierarchical dot notation)
    ///   - defaultValue: Default enum value if key not found
    /// - Returns: Parsed enum value
    /// - Throws: ConfigurationError.invalidValue if string doesn't match any enum case
    ///
    /// Example:
    /// ```swift
    /// let modelType = try await fetchEnum(
    ///     forKey: "runtime.dynamic.transport.modelType",
    ///     default: TransportModelType.constant
    /// )
    /// ```
    private func fetchEnum<T>(
        forKey key: String,
        default defaultValue: T
    ) async throws -> T where T: RawRepresentable, T.RawValue == String, T: CaseIterable {
        let rawValue = try await configReader.fetchString(
            forKey: key,
            default: defaultValue.rawValue
        )

        guard let value = T(rawValue: rawValue) else {
            // Generate helpful error message with all valid cases
            let validCases = T.allCases
                .map { "\($0)" }
                .joined(separator: ", ")

            throw ConfigurationError.invalidValue(
                key: key,
                value: rawValue,
                reason: "Expected one of: \(validCases)"
            )
        }

        return value
    }
}
