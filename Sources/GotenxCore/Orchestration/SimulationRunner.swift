// SimulationRunner.swift
// High-level simulation runner that integrates configuration with execution

import Foundation
import MLX

/// High-level simulation runner
///
/// Integrates the configuration system with simulation execution.
/// Responsible for:
/// - Running the simulation loop with provided models
/// - Managing timesteps and progress
/// - Coordinating with SimulationOrchestrator
///
/// **Protocol Conformance**: Conforms to `SimulationRunnable` for dependency injection.
/// This enables testing with `MockSimulationRunner` and future implementations
/// like `RemoteSimulationRunner`.
///
/// ## Example
///
/// ```swift
/// // Direct usage (production)
/// let runner = SimulationRunner(config: config)
/// try await runner.initialize(...)
/// let result = try await runner.run()
///
/// // Protocol usage (testing)
/// let runner: any SimulationRunnable = SimulationRunner(config: config)
/// try await runner.initialize(...)
/// let result = try await runner.run()
/// ```
public actor SimulationRunner: SimulationRunnable {
    private let config: SimulationConfiguration
    private var orchestrator: SimulationOrchestrator?

    public init(config: SimulationConfiguration) {
        self.config = config
    }

    /// Initialize the simulation with pre-created models
    ///
    /// - Parameters:
    ///   - transportModel: Pre-created transport model
    ///   - sourceModels: Pre-created source models
    ///   - mhdModels: MHD models (optional, default: created from config)
    /// - Throws: ConfigurationError if initialization fails
    public func initialize(
        transportModel: any TransportModel,
        sourceModels: [any SourceModel],
        mhdModels: [any MHDModel]? = nil
    ) async throws {
        let metalDevice = try MetalAccelerationPolicy.requireMetal4Device()

        // Create static runtime parameters
        let staticParameters = try config.runtime.static.runtimeParameters()

        // Get ProfileConditions from DynamicConfig
        let profileConditions = config.runtime.dynamic.toProfileConditions()

        // Generate initial profiles using ProfileConditions
        let initialProfiles = try generateInitialProfiles(
            mesh: config.runtime.static.mesh,
            profileConditions: profileConditions
        )

        // Create MHD models from config if not provided
        let mhdModelsToUse: [any MHDModel]
        if let provided = mhdModels {
            mhdModelsToUse = provided
        } else {
            mhdModelsToUse = MHDModelFactory.createAllModels(config: config.runtime.dynamic.mhd)
        }

        // Get adaptive config from time configuration (or use default if not specified)
        let adaptiveConfig = config.time.adaptive ?? .default

        // Initialize orchestrator with provided models
        self.orchestrator = await SimulationOrchestrator(
            staticParameters: staticParameters,
            initialProfiles: initialProfiles,
            transport: transportModel,
            sources: sourceModels,
            mhdModels: mhdModelsToUse,
            samplingConfig: .realTimePlotting,
            adaptiveConfig: adaptiveConfig
        )

        print("✓ Simulation initialized")
        print("  Mesh: \(staticParameters.mesh.cellCount) cells")
        print("  Solver: \(staticParameters.solverType)")
        print("  Transport: \(config.runtime.dynamic.transport.modelType)")
        print("  Metal: \(metalDevice.name) (Metal 4)")
        if !mhdModelsToUse.isEmpty {
            print("  MHD models: \(mhdModelsToUse.count) enabled")
        }
    }

    /// Run the simulation
    ///
    /// Executes the simulation loop for the configured time range.
    /// - Parameter progressCallback: Optional callback for progress updates
    /// - Returns: Final simulation result
    /// - Throws: Error if simulation fails
    public func run(
        progressCallback: (@Sendable (Float, ProgressInfo) -> Void)? = nil
    ) async throws -> SimulationResult {
        guard let orchestrator = orchestrator else {
            throw SimulationError.notInitialized
        }

        let timeConfig = config.time
        let dynamicConfig = config.runtime.dynamic

        // Create dynamic parameters
        let dynamicParameters = dynamicConfig.dynamicRuntimeParameters(timeStep: timeConfig.initialTimeStep)

        let endTime = timeConfig.end

        print("\n🚀 Starting simulation")
        print("  Time range: [\(timeConfig.start), \(timeConfig.end)] s")
        print("  Initial timeStep: \(timeConfig.initialTimeStep) s")

        // Start background task for progress monitoring
        let progressTask: Task<Void, Never>? = if let callback = progressCallback {
            Task {
                var iterationCount = 0
                while !Task.isCancelled {
                    iterationCount += 1

                    let progress = await orchestrator.progress()

                    let fraction = progress.currentTime / endTime
                    callback(fraction, progress)

                    // Update every 0.1 seconds
                    do {
                        try await Task.sleep(for: .milliseconds(100))
                    } catch {
                        break
                    }

                    // Stop when simulation completes or if no progress is being made.
                    if progress.currentTime >= endTime {
                        break
                    }

                    if iterationCount > 50 && progress.totalSteps == 0 {
                        break
                    }
                }
            }
        } else {
            nil
        }

        // Always stop progress monitoring, including when the run throws — otherwise the
        // background task keeps polling progress() and invoking the UI/log callback
        // after the simulation has already failed.
        defer { progressTask?.cancel() }

        // Run simulation via orchestrator
        let result = try await orchestrator.run(
            until: endTime,
            dynamicParameters: dynamicParameters,
            saveInterval: nil  // Could be made configurable
        )

        print("✓ Simulation complete")
        print("  Steps: \(result.statistics.totalSteps)")
        print("  Final time: \(endTime) s")
        print("  Wall time: \(result.statistics.wallTime) s")

        return result
    }

    /// Pause the simulation
    ///
    /// **App Integration**: Call from UI to pause long-running simulation.
    ///
    /// ## Example
    ///
    /// ```swift
    /// // From SwiftUI button
    /// Button("Pause") {
    ///     Task {
    ///         await runner.pause()
    ///     }
    /// }
    /// ```
    public func pause() async {
        await orchestrator?.pause()
    }

    /// Resume the simulation
    ///
    /// **App Integration**: Call from UI to resume paused simulation.
    ///
    /// ## Example
    ///
    /// ```swift
    /// // From SwiftUI button
    /// Button("Resume") {
    ///     Task {
    ///         await runner.resume()
    ///     }
    /// }
    /// ```
    public func resume() async {
        await orchestrator?.resume()
    }

    /// Check if simulation is paused
    ///
    /// - Returns: true if simulation is currently paused
    public func isPaused() async -> Bool {
        await orchestrator?.isPaused() ?? false
    }

    /// Generate initial profiles from ProfileConditions
    ///
    /// Uses ProfileConditions.evaluate(at:) to generate initial plasma profiles.
    /// This replaces the previous hardcoded profile generation with configuration-driven approach.
    ///
    /// - Parameters:
    ///   - mesh: Mesh configuration
    ///   - profileConditions: Profile conditions from DynamicConfig
    /// - Returns: Initial CoreProfiles
    private func generateInitialProfiles(
        mesh: MeshConfig,
        profileConditions: ProfileConditions
    ) throws -> CoreProfiles {
        let cellCount = mesh.cellCount

        // Generate normalized radial coordinate [0, 1]
        var rNorm = [Float](repeating: 0.0, count: cellCount)
        for i in 0..<cellCount {
            rNorm[i] = Float(i) / Float(cellCount - 1)
        }

        // Evaluate profiles using ProfileConditions
        var ti = [Float](repeating: 0.0, count: cellCount)
        var te = [Float](repeating: 0.0, count: cellCount)
        var ne = [Float](repeating: 0.0, count: cellCount)

        for i in 0..<cellCount {
            ti[i] = profileConditions.ionTemperature.evaluate(at: rNorm[i])
            te[i] = profileConditions.electronTemperature.evaluate(at: rNorm[i])
            ne[i] = profileConditions.electronDensity.evaluate(at: rNorm[i])
        }

        // Check for missing electron temperature.
        // If Te is zero/missing, use physically sound fallback Te = Ti
        let te_max = te.max() ?? 0.0
        if te_max <= 0.0 {
            print("[INIT-WARNING] Electron temperature missing or zero, using Te = Ti fallback")
            te = ti
        }

        // Apply the same density floor used by the coefficient builder.
        let ne_floor: Float = 1e18
        for i in 0..<cellCount {
            ne[i] = max(ne[i], ne_floor)
        }

        // Poloidal flux: initially zero (physical initial condition)
        // Note: In Phase 2, this could be made configurable via ProfileConditions
        let psi = [Float](repeating: 0.0, count: cellCount)

        // Create evaluated arrays
        let evaluated = EvaluatedArray.evaluatingBatch([
            MLXArray(ti),
            MLXArray(te),
            MLXArray(ne),
            MLXArray(psi)
        ])

        let profiles = CoreProfiles(
            ionTemperature: evaluated[0],
            electronTemperature: evaluated[1],
            electronDensity: evaluated[2],
            poloidalFlux: evaluated[3]
        )

        let validatedProfiles: ValidatedProfiles
        do {
            validatedProfiles = try ValidatedProfiles.validate(profiles)
        } catch {
            // User-configuration-derived failure: surface it as a typed error so the
            // app/CLI can report it, instead of crashing the process with fatalError.
            throw SimulationError.invalidConfiguration("""
                Initial profile validation failed (check ProfileConditions in the configuration).
                - Ti range: [\(ti.min() ?? Float.nan), \(ti.max() ?? Float.nan)] eV
                - Te range: [\(te.min() ?? Float.nan), \(te.max() ?? Float.nan)] eV
                - ne range: [\(ne.min() ?? Float.nan), \(ne.max() ?? Float.nan)] m⁻³
                - Error: \(error)
                """)
        }

        return validatedProfiles.toCoreProfiles()
    }

    /// Adapt timestep based on stability criteria
    private func adaptTimestep(
        currentDt: Float,
        minimumTimeStep: Float,
        maximumTimeStep: Float,
        safetyFactor: Float
    ) -> Float {
        // Simple adaptive scheme: could be enhanced with error estimation
        var newDt = currentDt * 1.1  // Increase by 10%

        // Apply safety factor
        newDt *= safetyFactor

        // Clamp to limits
        newDt = max(minimumTimeStep, min(maximumTimeStep, newDt))

        return newDt
    }
}

/// Simulation errors with user-friendly descriptions
///
/// **App Integration**: Conforms to `LocalizedError` for user-friendly error messages
/// and recovery suggestions in the UI.
///
/// **Design Note**: This is a high-level error type for app integration.
/// Internal errors (like `SolverError`) are thrown directly by lower-level components.
/// Apps should catch both `SimulationError` and `SolverError`.
///
/// ## Example
///
/// ```swift
/// do {
///     try await runner.run()
/// } catch let error as SimulationError {
///     // High-level errors (initialization, configuration, validation)
///     print(error.localizedDescription)
///     if let suggestion = error.recoverySuggestion {
///         print("Suggestion: \(suggestion)")
///     }
/// } catch let error as SolverError {
///     // Low-level solver errors (convergence, numerical instability)
///     print("Solver error: \(error)")
/// }
/// ```
public enum SimulationError: Error, LocalizedError {
    case notInitialized
    case invalidConfiguration(String)
    case executionFailed(String)

    // MARK: - Specific Errors

    /// Model initialization failed
    case modelInitializationFailed(modelName: String, reason: String)

    /// Numeric instability detected during simulation
    case numericInstability(time: Float, variable: String, value: Float)

    /// Solver failed to converge
    case convergenceFailure(iterations: Int, residual: Float)

    /// Invalid boundary conditions
    case invalidBoundaryConditions(String)

    /// Mesh resolution too coarse
    case meshTooCoarse(cellCount: Int, minimum: Int)

    /// Time step too small
    case timeStepTooSmall(timeStep: Float, minimum: Float)

    // MARK: - LocalizedError

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Simulation not initialized. Call initialize() first."

        case .invalidConfiguration(let msg):
            return "Invalid configuration: \(msg)"

        case .executionFailed(let msg):
            return "Simulation execution failed: \(msg)"

        case .modelInitializationFailed(let model, let reason):
            return "Failed to initialize \(model): \(reason)"

        case .numericInstability(let time, let variable, let value):
            return "Numeric instability detected at t=\(time)s: \(variable) = \(value)"

        case .convergenceFailure(let iters, let residual):
            return "Solver failed to converge after \(iters) iterations (residual: \(residual))"

        case .invalidBoundaryConditions(let msg):
            return "Invalid boundary conditions: \(msg)"

        case .meshTooCoarse(let cellCount, let minimum):
            return "Mesh too coarse: \(cellCount) cells (minimum: \(minimum))"

        case .timeStepTooSmall(let timeStep, let minimum):
            return "Time step too small: \(timeStep)s (minimum: \(minimum))"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .meshTooCoarse(_, let minimum):
            return "Increase mesh resolution to at least \(minimum) cells"

        case .timeStepTooSmall:
            return "Increase initial time step or check for numerical instabilities"

        case .convergenceFailure:
            return "Try reducing time step or using a more robust solver"

        case .invalidBoundaryConditions:
            return "Check that temperature and density values are positive and realistic"

        case .modelInitializationFailed:
            return "Check model configuration and ensure all required parameters are provided"

        default:
            return nil
        }
    }
}
