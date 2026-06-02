// SimulationRunnable.swift
// Protocol for simulation execution

import Foundation

/// Protocol for simulation execution
///
/// Defines the contract for running tokamak transport simulations.
/// Implementations must be actors for thread-safe execution.
///
/// **App Integration**: This protocol enables dependency injection for testing.
/// Use `SimulationRunner` for production and `MockSimulationRunner` for tests.
///
/// ## Implementations
///
/// - `SimulationRunner`: Standard local execution with full physics models
/// - `MockSimulationRunner`: Test double for unit tests (fast, controllable)
/// - `RemoteSimulationRunner`: Future implementation for remote/cloud execution
/// - `CachedSimulationRunner`: Future implementation with result caching
///
/// ## Example
///
/// ```swift
/// // Production code
/// let runner: any SimulationRunnable = SimulationRunner(config: config)
/// try await runner.initialize(
///     transportModel: transport,
///     sourceModels: [sources],
///     mhdModels: mhdModels
/// )
/// let result = try await runner.run()
///
/// // Test code
/// let mockRunner: any SimulationRunnable = MockSimulationRunner()
/// mockRunner.simulationDuration = 0.1  // Fast for tests
/// try await mockRunner.initialize(...)
/// let result = try await mockRunner.run()
/// ```
///
/// ## Thread Safety
///
/// All implementations must be actors to ensure thread-safe access.
/// Progress callbacks are `@Sendable` to cross actor boundaries safely.
public protocol SimulationRunnable: Actor {
    /// Initialize simulation with physics models
    ///
    /// Must be called before `run()`. Prepares the simulation with transport,
    /// source, and MHD models.
    ///
    /// - Parameters:
    ///   - transportModel: Transport model (QLKNN, Bohm-GyroBohm, Constant, etc.)
    ///   - sourceModels: Heating and current drive sources (typically array with single CompositeSourceModel)
    ///   - mhdModels: MHD models (sawteeth, NTMs, etc.), optional
    /// - Throws: SimulationError if initialization fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let transport = try TransportModelFactory.create(config: transportConfig)
    /// let sources = try SourceModelFactory.create(config: sourcesConfig)
    /// let mhdModels = MHDModelFactory.createAllModels(config: mhdConfig)
    ///
    /// try await runner.initialize(
    ///     transportModel: transport,
    ///     sourceModels: [sources],
    ///     mhdModels: mhdModels.isEmpty ? nil : mhdModels
    /// )
    /// ```
    func initialize(
        transportModel: any TransportModel,
        sourceModels: [any SourceModel],
        mhdModels: [any MHDModel]?
    ) async throws

    /// Run simulation with progress callback
    ///
    /// Executes the simulation loop until the configured end time.
    /// Progress callback is called periodically (throttled to ~100ms).
    ///
    /// - Parameter progressCallback: Optional callback for progress updates
    ///   - Parameter fraction: Progress fraction [0.0, 1.0]
    ///   - Parameter info: Detailed progress information (time, steps, profiles, etc.)
    /// - Returns: Final simulation result with profiles and statistics
    /// - Throws:
    ///   - SimulationError: High-level errors (not initialized, configuration, etc.)
    ///   - SolverError: Solver-specific errors (convergence failure, etc.)
    ///   - CancellationError: Task was cancelled
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try await runner.run { fraction, progressInfo in
    ///     print("Progress: \(fraction * 100)%")
    ///     print("  Time: \(progressInfo.currentTime)s")
    ///     print("  Steps: \(progressInfo.totalSteps)")
    ///
    ///     if let profiles = progressInfo.profiles {
    ///         updatePlot(profiles)  // Live plotting
    ///     }
    /// }
    ///
    /// print("Final ionTemperature: \(result.finalProfiles.ionTemperature)")
    /// print("Wall time: \(result.statistics.wallTime)s")
    /// ```
    func run(
        progressCallback: (@Sendable (Float, ProgressInfo) -> Void)?
    ) async throws -> SimulationResult

    /// Pause the simulation
    ///
    /// **App Integration**: Call from UI to pause long-running simulation.
    /// Simulation will pause at the beginning of the next timestep.
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
    func pause() async

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
    func resume() async

    /// Check if simulation is paused
    ///
    /// - Returns: true if simulation is currently paused
    ///
    /// ## Example
    ///
    /// ```swift
    /// let paused = await runner.isPaused()
    /// if paused {
    ///     print("Simulation is paused")
    /// }
    /// ```
    func isPaused() async -> Bool
}

// MARK: - Default Implementations

extension SimulationRunnable {
    /// Initialize simulation without MHD models
    ///
    /// Convenience method when MHD models are not needed.
    /// Equivalent to calling `initialize(transportModel:sourceModels:mhdModels:)` with `mhdModels: nil`.
    ///
    /// - Parameters:
    ///   - transportModel: Transport model (QLKNN, Bohm-GyroBohm, Constant, etc.)
    ///   - sourceModels: Heating and current drive sources
    /// - Throws: SimulationError if initialization fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Without MHD models
    /// try await runner.initialize(
    ///     transportModel: transport,
    ///     sourceModels: sources
    /// )
    ///
    /// // With MHD models
    /// try await runner.initialize(
    ///     transportModel: transport,
    ///     sourceModels: sources,
    ///     mhdModels: mhdModels
    /// )
    /// ```
    public func initialize(
        transportModel: any TransportModel,
        sourceModels: [any SourceModel]
    ) async throws {
        try await initialize(
            transportModel: transportModel,
            sourceModels: sourceModels,
            mhdModels: nil
        )
    }

    /// Run simulation without progress callback
    ///
    /// Convenience method for batch processing without UI updates.
    ///
    /// - Returns: Final simulation result
    /// - Throws: SimulationError, SolverError, or CancellationError
    public func run() async throws -> SimulationResult {
        try await run(progressCallback: nil)
    }
}
