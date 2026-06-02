// RunCommand.swift
// Command for running Gotenx simulations with full Torax compatibility

import ArgumentParser
import Foundation
import GotenxCore
import GotenxPhysics

struct RunCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Run a Gotenx simulation",
        discussion: """
            Execute a Gotenx tokamak core transport simulation with the specified configuration.

            The configuration file must be in JSON format and contain all required simulation parameters.
            Relative paths are resolved first against the working directory, then the Gotenx base directory.

            Example:
              torax run --config examples/basic_config.json --log-progress
            """
    )

    // MARK: - Required Arguments

    @Option(
        name: .long,
        help: "Path to configuration file (JSON). Relative paths resolved against cwd, then Gotenx base."
    )
    var config: String

    // MARK: - Output Options

    @Option(
        name: .long,
        help: "Output directory for results (overrides config file if specified)"
    )
    var outputDir: String?

    @Option(
        name: .long,
        help: "Output format: json, hdf5, netcdf (default: json)"
    )
    var outputFormat: OutputFormat = .json

    // MARK: - Logging Options

    @Flag(
        name: .long,
        help: "Log simulation progress (time, timeStep, iterations) to stdout"
    )
    var logProgress: Bool = false

    @Flag(
        name: .long,
        help: "Log detailed output for debugging to stdout/stderr"
    )
    var logOutput: Bool = false

    // MARK: - Plotting Options

    @Flag(
        name: .long,
        help: "Plot simulation progress in real-time (experimental)"
    )
    var plotProgress: Bool = false

    @Option(
        name: .long,
        help: "Path to reference run output for comparison plotting"
    )
    var referenceRun: String?

    @Option(
        name: .long,
        help: "Path to plot configuration file (JSON)"
    )
    var plotConfig: String?

    // MARK: - Interactive Mode

    @Flag(
        name: .long,
        help: "Quit immediately after simulation completes (disable interactive menu)"
    )
    var quit: Bool = false

    // MARK: - Performance Options

    @Flag(
        name: .long,
        help: "Disable MLX JIT compilation (for debugging)"
    )
    var noCompile: Bool = false

    @Flag(
        name: .long,
        help: "Enable additional error checking (performance impact)"
    )
    var enableErrors: Bool = false

    @Option(
        name: .long,
        help: "MLX GPU cache limit in MB"
    )
    var cacheLimit: Int?

    @Flag(
        name: .long,
        help: "Enable performance profiling"
    )
    var profile: Bool = false

    @Option(
        name: .long,
        help: "Profile output file path"
    )
    var profileOutput: String?

    // MARK: - Configuration Overrides

    @Option(
        name: .long,
        help: "Override mesh cell count"
    )
    var meshCellCount: Int?

    @Option(
        name: .long,
        help: "Override mesh major radius (m)"
    )
    var meshMajorRadius: Double?

    @Option(
        name: .long,
        help: "Override mesh minor radius (m)"
    )
    var meshMinorRadius: Double?

    @Option(
        name: .long,
        help: "Override simulation end time (s)"
    )
    var timeEnd: Double?

    @Option(
        name: .long,
        help: "Override initial timestep (s)"
    )
    var initialTimeStep: Double?

    // MARK: - Execution

    mutating func run() async throws {
        GotenxLogging.bootstrap()
        printBanner()

        // Resolve configuration path (Torax-compatible)
        let resolvedConfigPath = try resolveConfigPath(config)
        print("Configuration: \(resolvedConfigPath)")

        // Setup environment
        let envConfig = EnvironmentConfig(
            compilationEnabled: !noCompile,
            errorsEnabled: enableErrors,
            cacheLimitMB: cacheLimit
        )
        try envConfig.apply()

        // Setup logger
        let logger = ProgressLogger(
            logProgress: logProgress,
            logOutput: logOutput
        )

        // Start profiling if enabled
        let profilingContext = profile ? ProfilingContext(outputPath: profileOutput) : nil
        profilingContext?.start()

        // Load configuration with overrides
        print("\n📋 Loading configuration...")
        let simulationConfig = try await loadConfiguration(from: resolvedConfigPath)
        print("✓ Configuration loaded and validated")
        print("  Mesh cells: \(simulationConfig.runtime.static.mesh.cellCount)")
        print("  Major radius: \(simulationConfig.runtime.static.mesh.majorRadius) m")
        print("  Time range: [\(simulationConfig.time.start), \(simulationConfig.time.end)] s")
        print("  Initial timeStep: \(simulationConfig.time.initialTimeStep) s")

        // Create output directory (after config loaded)
        try createOutputDirectory(config: simulationConfig)

        // Initialize physics models
        print("\n🔧 Initializing physics models...")
        let transportModel = try createTransportModel(config: simulationConfig)
        print("  ✓ Transport model: \(simulationConfig.runtime.dynamic.transport.modelType)")

        let sourceModel = try createSourceModel(config: simulationConfig)
        print("  ✓ Source models initialized")

        // Initialize simulation runner
        print("\n🚀 Initializing simulation...")
        let runner = SimulationRunner(config: simulationConfig)

        try await runner.initialize(
            transportModel: transportModel,
            sourceModels: [sourceModel]
        )

        // Run simulation
        print("\n⏱️  Running simulation...")
        let result = try await runner.run(
            progressCallback: logProgress ? makeProgressCallback() : nil
        )

        // Display results summary
        print("\n📊 Simulation Results:")
        print("  Total steps: \(result.statistics.totalSteps)")
        print("  Total iterations: \(result.statistics.totalIterations)")
        print("  Wall time: \(String(format: "%.2f", result.statistics.wallTime))s")
        print("  Converged: \(result.statistics.converged ? "Yes" : "No")")

        // Log final state with display units
        logger.logFinalState(createSummary(from: result))

        // Save results
        print("\n💾 Saving results...")
        try await saveResults(result, config: simulationConfig)
        print("  ✓ Results saved to: \(simulationConfig.output.directory)")

        // Stop profiling
        if let context = profilingContext {
            let stats = context.stop()
            logger.logProfilingStats(stats)
        }

        // Interactive menu (unless --quit specified)
        if !quit {
            try await interactiveMenu(
                logger: logger,
                config: simulationConfig,
                configPath: resolvedConfigPath,
                result: result
            )
        }
    }

    // MARK: - Configuration Loading

    /// Load configuration using GotenxConfigReader with hierarchical overrides
    ///
    /// Override priority (highest to lowest):
    /// 1. CLI arguments
    /// 2. Environment variables (GOTENX_*)
    /// 3. JSON configuration file
    /// 4. Default values
    private func loadConfiguration(from path: String) async throws -> SimulationConfiguration {
        // Build CLI overrides map (only include explicitly specified values)
        var cliOverrides: [String: String] = [:]

        if let value = meshCellCount {
            cliOverrides["runtime.static.mesh.cellCount"] = String(value)
        }
        if let value = meshMajorRadius {
            cliOverrides["runtime.static.mesh.majorRadius"] = String(value)
        }
        if let value = meshMinorRadius {
            cliOverrides["runtime.static.mesh.minorRadius"] = String(value)
        }
        if let value = timeEnd {
            cliOverrides["time.end"] = String(value)
        }
        if let value = initialTimeStep {
            cliOverrides["time.initialTimeStep"] = String(value)
        }
        if let value = outputDir {
            cliOverrides["output.directory"] = value
        }

        // Create GotenxConfigReader with hierarchical configuration
        let configReader = try await GotenxConfigReader.create(
            jsonPath: path,
            cliOverrides: cliOverrides
        )

        // Log override sources if any CLI args were specified
        if !cliOverrides.isEmpty {
            print("  Applying hierarchical configuration:")
            print("    1. CLI arguments (\(cliOverrides.count) override\(cliOverrides.count == 1 ? "" : "s"))")
            print("    2. Environment variables (GOTENX_*)")
            print("    3. JSON file: \(path)")
        }

        // Fetch complete configuration
        return try await configReader.fetchConfiguration()
    }

    // MARK: - Helper Methods

    /// Resolve configuration path (Torax-compatible behavior)
    private func resolveConfigPath(_ path: String) throws -> String {
        // 1. Absolute path
        if path.hasPrefix("/") {
            guard FileManager.default.fileExists(atPath: path) else {
                throw CLIError.configNotFound(path)
            }
            return path
        }

        // 2. Relative to current working directory
        let cwdPath = FileManager.default.currentDirectoryPath + "/" + path
        if FileManager.default.fileExists(atPath: cwdPath) {
            return cwdPath
        }

        // 3. Relative to Gotenx base directory
        if let basePath = findGotenxBaseDirectory() {
            let gotenxPath = basePath + "/" + path
            if FileManager.default.fileExists(atPath: gotenxPath) {
                return gotenxPath
            }
        }

        throw CLIError.configNotFound(path)
    }

    /// Find Gotenx base directory by looking for characteristic directories
    private func findGotenxBaseDirectory() -> String? {
        var currentPath = FileManager.default.currentDirectoryPath

        // Look up directory tree for Gotenx markers
        for _ in 0..<5 {
            let examplesPath = currentPath + "/examples"
            let sourcesPath = currentPath + "/Sources/Gotenx"
            let packageSwift = currentPath + "/Package.swift"

            if FileManager.default.fileExists(atPath: examplesPath) ||
               FileManager.default.fileExists(atPath: sourcesPath) ||
               FileManager.default.fileExists(atPath: packageSwift) {
                return currentPath
            }

            // Go up one level
            currentPath = (currentPath as NSString).deletingLastPathComponent

            // Stop at root
            if currentPath == "/" {
                break
            }
        }

        return nil
    }

    private func createOutputDirectory(config: SimulationConfiguration) throws {
        // Determine output directory: CLI override > config file
        let directory: String
        let source: String

        if let cliOverride = outputDir {
            // CLI flag takes precedence
            directory = cliOverride
            source = "CLI override"
        } else {
            // Use directory from config (has default value)
            directory = config.output.directory
            source = "from config"
        }

        let url = URL(fileURLWithPath: directory)
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: true
        )

        print("Output directory: \(directory) (\(source))")
    }

    private func interactiveMenu(
        logger: ProgressLogger,
        config: SimulationConfiguration,
        configPath: String,
        result: SimulationResult
    ) async throws {
        var menu = InteractiveMenu(
            logger: logger,
            plotConfig: plotConfig,
            referenceRun: referenceRun,
            config: config,
            configPath: configPath,
            lastResult: result
        )

        try await menu.run()
    }

    private func printBanner() {
        print("""
        ═══════════════════════════════════════════════════
        swift-Gotenx v0.1.0
        Tokamak Core Transport Simulator for Apple Silicon
        ═══════════════════════════════════════════════════
        """)
    }

    // MARK: - Model Factories

    /// Create transport model from configuration
    private func createTransportModel(config: SimulationConfiguration) throws -> any TransportModel {
        return try TransportModelFactory.create(config: config.runtime.dynamic.transport)
    }

    /// Create source model from configuration
    private func createSourceModel(config: SimulationConfiguration) throws -> any SourceModel {
        return try SourceModelFactory.create(config: config.runtime.dynamic.sources)
    }

    // MARK: - Progress Callback

    /// Create progress callback for logging
    private func makeProgressCallback() -> @Sendable (Float, ProgressInfo) -> Void {
        return { fraction, progress in
            // Simple progress logging - could be enhanced with ETA, etc.
            let percentage = Int(fraction * 100)
            print("  Progress: \(percentage)% | Time: \(String(format: "%.6f", progress.currentTime))s | timeStep: \(String(format: "%.8f", progress.lastTimeStep))s")
        }
    }

    // MARK: - Results Handling

    /// Save simulation results to file
    private func saveResults(
        _ result: SimulationResult,
        config: SimulationConfiguration
    ) async throws {
        let directory = config.output.directory

        // Convert Gotenx.OutputFormat to gotenx_cli.OutputFormat
        let cliFormat: OutputFormat
        switch config.output.format {
        case .json:
            cliFormat = .json
        case .hdf5:
            cliFormat = .hdf5
        case .netcdf:
            cliFormat = .netcdf
        }

        // Generate filename with timestamp
        let filename = OutputFileNaming.generateFilename(
            prefix: "state_history",
            format: cliFormat
        )
        let filepath = (directory as NSString).appendingPathComponent(filename)

        // Write results
        let writer = OutputWriter(format: cliFormat)
        try writer.write(result, to: URL(fileURLWithPath: filepath))
    }

    /// Create summary for display from simulation result
    private func createSummary(from result: SimulationResult) -> SimulationStateSummary {
        let profiles = result.finalProfiles

        return SimulationStateSummary(
            ionTemperature: ProfileStats(
                min: Double(profiles.ionTemperatureArray.min() ?? 0),
                max: Double(profiles.ionTemperatureArray.max() ?? 0),
                core: Double(profiles.ionTemperatureArray.first ?? 0),
                edge: Double(profiles.ionTemperatureArray.last ?? 0)
            ),
            electronTemperature: ProfileStats(
                min: Double(profiles.electronTemperatureArray.min() ?? 0),
                max: Double(profiles.electronTemperatureArray.max() ?? 0),
                core: Double(profiles.electronTemperatureArray.first ?? 0),
                edge: Double(profiles.electronTemperatureArray.last ?? 0)
            ),
            electronDensity: ProfileStats(
                min: Double(profiles.electronDensityArray.min() ?? 0),
                max: Double(profiles.electronDensityArray.max() ?? 0),
                core: Double(profiles.electronDensityArray.first ?? 0),
                edge: Double(profiles.electronDensityArray.last ?? 0)
            ),
            // Safety factor and magnetic shear - not yet computed
            safetyFactor: ProfileStats(min: 0, max: 0, core: 0, edge: 0),
            magneticShear: ProfileStats(min: 0, max: 0, core: 0, edge: 0)
        )
    }
}

// MARK: - Placeholder Types

struct PlaceholderResults {
    // Placeholder until SimulationResults is implemented
}
