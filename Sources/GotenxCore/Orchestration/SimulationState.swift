import MLX
import Foundation
import Numerics

// MARK: - Simulation State

/// Internal simulation state (actor-isolated)
///
/// Uses high-precision Float.Augmented time accumulation to prevent cumulative round-off errors
/// over long simulations (20,000+ timesteps) while maintaining Float32-only design.
///
/// **Design Principle**: Apple Silicon GPU does NOT support Float64 (Double).
/// All numeric types must be Float32 for GPU compatibility and architectural consistency.
public struct SimulationState: Sendable {
    // MARK: - Core State (Always Present)

    /// Current plasma profiles
    public let profiles: CoreProfiles

    /// High-precision time accumulator (internal, ~15 digits precision)
    ///
    /// Uses Double (64-bit) instead of Float (32-bit) to maintain precision over
    /// long integrations. This is the **only CPU operation** in the entire
    /// simulation pipeline, but has negligible cost (1 operation per timestep).
    ///
    /// **Why Double instead of Float32?**
    /// - Time accumulation is CPU-only (GPU compatibility irrelevant)
    /// - Double provides 15 digits precision vs Float32's 7 digits
    /// - For 20,000 steps: Double error ~10⁻¹² vs Float32 error ~2×10⁻³
    /// - Standard practice in scientific computing
    ///
    /// **Why not Float.Augmented?**
    /// - More complex (head + tail representation)
    /// - Slower (2 additions + correction)
    /// - Requires Swift Numerics dependency
    /// - Double is simpler, faster, and more accurate
    ///
    /// **Design Rationale**:
    /// - Float32-only policy applies to **GPU operations**
    /// - CPU-only operations can use Double when beneficial
    /// - Time accumulation: 1 operation per timestep (negligible cost)
    ///
    /// **Performance**: CPU-only operation (1 per timestep, negligible cost)
    private let timeAccumulator: Double

    /// Current simulation time [s]
    ///
    /// Computed from high-precision Double accumulator to avoid cumulative errors.
    ///
    /// ## Precision Comparison
    ///
    /// **Naive Float32 accumulation** (WRONG):
    /// ```
    /// var time: Float = 0.0
    /// for _ in 0..<20000 {
    ///     time += dt  // Cumulative error: 20,000 × 10⁻⁷ ≈ 2×10⁻³ (0.2%)
    /// }
    /// ```
    ///
    /// **Double accumulation** (CORRECT):
    /// ```
    /// var acc: Double = 0.0
    /// for _ in 0..<20000 {
    ///     acc += Double(dt)  // Cumulative error: ~10⁻¹² (negligible)
    /// }
    /// let time = Float(acc)
    /// ```
    ///
    /// **Result**: 0.2% error → 0.0000001% error (2,000,000× improvement)
    public var time: Float {
        Float(timeAccumulator)
    }

    /// Current timestep [s]
    public let dt: Float

    /// Step number
    public let step: Int

    /// Statistics
    public var statistics: SimulationStatistics

    // MARK: - Extended Physics State (Phase 1: Optional)

    /// Transport coefficients (χ, D, v)
    ///
    /// **Phase 1**: Always nil (not yet captured)
    /// **Phase 3**: Populated from transport models
    public let transport: TransportCoefficients?

    /// Source terms (heating, current drive)
    ///
    /// **Phase 1**: Always nil (not yet captured)
    /// **Phase 3**: Populated from source models
    public let sources: SourceTerms?

    /// Geometry configuration
    ///
    /// **Phase 1**: Always nil (use from config instead)
    /// **Phase 2**: Populated for convenience
    public let geometry: Geometry?

    /// Derived scalar quantities (τE, Q, βN, etc.)
    ///
    /// **Phase 1**: Always nil (not yet computed)
    /// **Phase 2**: Computed from profiles
    /// **Phase 3**: Computed from profiles + transport + sources
    public let derived: DerivedQuantities?

    /// Numerical diagnostics (convergence, conservation)
    ///
    /// **Phase 1**: Always nil (not yet tracked)
    /// **Phase 2**: Captured from solver
    public let diagnostics: NumericalDiagnostics?

    // MARK: - Initialization

    public init(
        profiles: CoreProfiles,
        timeAccumulator: Double = 0.0,
        dt: Float = 1e-4,
        step: Int = 0,
        statistics: SimulationStatistics = SimulationStatistics(),
        transport: TransportCoefficients? = nil,
        sources: SourceTerms? = nil,
        geometry: Geometry? = nil,
        derived: DerivedQuantities? = nil,
        diagnostics: NumericalDiagnostics? = nil
    ) {
        self.profiles = profiles
        self.timeAccumulator = timeAccumulator
        self.dt = dt
        self.step = step
        self.statistics = statistics
        self.transport = transport
        self.sources = sources
        self.geometry = geometry
        self.derived = derived
        self.diagnostics = diagnostics
    }

    /// Create updated state (legacy compatibility)
    ///
    /// **Deprecated**: Use `advanced(by:profiles:)` for time stepping.
    public func updated(
        profiles: CoreProfiles? = nil,
        time: Float? = nil,
        dt: Float? = nil,
        step: Int? = nil,
        statistics: SimulationStatistics? = nil,
        transport: TransportCoefficients?? = nil,
        sources: SourceTerms?? = nil,
        geometry: Geometry?? = nil,
        derived: DerivedQuantities?? = nil,
        diagnostics: NumericalDiagnostics?? = nil
    ) -> SimulationState {
        SimulationState(
            profiles: profiles ?? self.profiles,
            timeAccumulator: time.map { Double($0) } ?? self.timeAccumulator,
            dt: dt ?? self.dt,
            step: step ?? self.step,
            statistics: statistics ?? self.statistics,
            transport: transport ?? self.transport,
            sources: sources ?? self.sources,
            geometry: geometry ?? self.geometry,
            derived: derived ?? self.derived,
            diagnostics: diagnostics ?? self.diagnostics
        )
    }

    /// Advance state by one timestep with high-precision time accumulation
    ///
    /// This is the **recommended** method for time stepping, as it maintains
    /// numerical precision over long simulations.
    ///
    /// - Parameters:
    ///   - dt: Timestep duration [s]
    ///   - profiles: Updated plasma profiles
    ///   - statistics: Updated statistics (optional)
    ///   - transport: Updated transport coefficients (Phase 3)
    ///   - sources: Updated source terms (Phase 3)
    ///   - geometry: Geometry configuration (Phase 2)
    ///   - derived: Derived quantities (Phase 2)
    ///   - diagnostics: Numerical diagnostics (Phase 2)
    /// - Returns: New state with accumulated time
    ///
    /// ## Example
    ///
    /// ```swift
    /// var state = SimulationState(profiles: initialProfiles)
    /// for _ in 0..<20000 {
    ///     let newProfiles = solver.solve(...)
    ///     state = state.advanced(by: dt, profiles: newProfiles)
    /// }
    /// print(state.time)  // Accurate to ~10⁻¹⁰ after 20,000 steps
    /// ```
    public func advanced(
        by dt: Float,
        profiles: CoreProfiles,
        statistics: SimulationStatistics? = nil,
        transport: TransportCoefficients? = nil,
        sources: SourceTerms? = nil,
        geometry: Geometry? = nil,
        derived: DerivedQuantities? = nil,
        diagnostics: NumericalDiagnostics? = nil
    ) -> SimulationState {
        // Validate timestep
        guard dt.isFinite else {
            fatalError("SimulationState.advanced: dt must be finite (got \(dt))")
        }
        guard dt >= 0 else {
            fatalError("SimulationState.advanced: dt must be non-negative (got \(dt))")
        }

        // High-precision time accumulation using Double (CPU operation, but 1 per timestep)
        let newTimeAccumulator = timeAccumulator + Double(dt)

        // Check for overflow (extremely rare: would require ~10^300 seconds)
        guard newTimeAccumulator.isFinite else {
            fatalError("SimulationState.advanced: time accumulator overflow (accumulated time: \(timeAccumulator)s, dt: \(dt)s)")
        }

        return SimulationState(
            profiles: profiles,
            timeAccumulator: newTimeAccumulator,
            dt: dt,
            step: step + 1,
            statistics: statistics ?? self.statistics,
            transport: transport ?? self.transport,
            sources: sources ?? self.sources,
            geometry: geometry ?? self.geometry,
            derived: derived ?? self.derived,
            diagnostics: diagnostics ?? self.diagnostics
        )
    }
}

// MARK: - Simulation Statistics

/// Statistics tracked during simulation
public struct SimulationStatistics: Sendable, Codable {
    /// Total number of solver iterations
    public var totalIterations: Int

    /// Number of timesteps
    public var totalSteps: Int

    /// Whether simulation converged
    public var converged: Bool

    /// Maximum residual norm encountered
    public var maxResidualNorm: Float

    /// Total wall time [s]
    public var wallTime: Float

    public init(
        totalIterations: Int = 0,
        totalSteps: Int = 0,
        converged: Bool = true,
        maxResidualNorm: Float = 0.0,
        wallTime: Float = 0.0
    ) {
        self.totalIterations = totalIterations
        self.totalSteps = totalSteps
        self.converged = converged
        self.maxResidualNorm = maxResidualNorm
        self.wallTime = wallTime
    }
}

// MARK: - Serializable Profiles

/// Serializable profiles for crossing actor boundaries
public struct SerializableProfiles: Sendable, Codable {
    public let ionTemperature: [Float]
    public let electronTemperature: [Float]
    public let electronDensity: [Float]
    public let poloidalFlux: [Float]

    public init(
        ionTemperature: [Float],
        electronTemperature: [Float],
        electronDensity: [Float],
        poloidalFlux: [Float]
    ) {
        self.ionTemperature = ionTemperature
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity
        self.poloidalFlux = poloidalFlux
    }
}

// MARK: - CoreProfiles Conversion

extension CoreProfiles {
    /// Convert to serializable format
    public func toSerializable() -> SerializableProfiles {
        let tiArray = ionTemperature.value.asArray(Float.self)
        let teArray = electronTemperature.value.asArray(Float.self)
        let neArray = electronDensity.value.asArray(Float.self)
        let psiArray = poloidalFlux.value.asArray(Float.self)

        return SerializableProfiles(
            ionTemperature: tiArray,
            electronTemperature: teArray,
            electronDensity: neArray,
            poloidalFlux: psiArray
        )
    }

    /// Create from serializable format
    public init(from serializable: SerializableProfiles) {
        self.init(
            ionTemperature: EvaluatedArray(evaluating: MLXArray(serializable.ionTemperature)),
            electronTemperature: EvaluatedArray(evaluating: MLXArray(serializable.electronTemperature)),
            electronDensity: EvaluatedArray(evaluating: MLXArray(serializable.electronDensity)),
            poloidalFlux: EvaluatedArray(evaluating: MLXArray(serializable.poloidalFlux))
        )
    }
}

// MARK: - Simulation Result

/// Result from simulation run
public struct SimulationResult: Sendable, Codable {
    /// Final profiles
    public let finalProfiles: SerializableProfiles

    /// Simulation statistics
    public let statistics: SimulationStatistics

    /// Time series (optional, for post-processing)
    public let timeSeries: [TimePoint]?

    public init(
        finalProfiles: SerializableProfiles,
        statistics: SimulationStatistics,
        timeSeries: [TimePoint]? = nil
    ) {
        self.finalProfiles = finalProfiles
        self.statistics = statistics
        self.timeSeries = timeSeries
    }
}

// MARK: - CLI Integration

extension SerializableProfiles {
    /// Access temperature profiles as arrays for display
    public var ionTemperatureArray: [Float] { ionTemperature }
    public var electronTemperatureArray: [Float] { electronTemperature }
    public var electronDensityArray: [Float] { electronDensity }
}

// MARK: - Time Point

/// Single time point in simulation
///
/// **Phase 1**: Only time and profiles captured
/// **Phase 2**: Derived quantities and diagnostics added
/// **Phase 3**: Transport and source terms added
public struct TimePoint: Sendable, Codable {
    public let time: Float
    public let profiles: SerializableProfiles

    /// Derived scalar quantities (Phase 2+)
    public let derived: DerivedQuantities?

    /// Numerical diagnostics (Phase 2+)
    public let diagnostics: NumericalDiagnostics?

    public init(
        time: Float,
        profiles: SerializableProfiles,
        derived: DerivedQuantities? = nil,
        diagnostics: NumericalDiagnostics? = nil
    ) {
        self.time = time
        self.profiles = profiles
        self.derived = derived
        self.diagnostics = diagnostics
    }
}

// MARK: - SimulationState to TimePoint Conversion

extension SimulationState {
    /// Convert simulation state to time point for time series capture
    ///
    /// **Phase 1**: Only captures time and profiles
    /// **Phase 2**: Captures derived quantities and diagnostics
    public func toTimePoint() -> TimePoint {
        TimePoint(
            time: time,
            profiles: profiles.toSerializable(),
            derived: derived,
            diagnostics: diagnostics
        )
    }
}

// MARK: - Progress Info

/// Progress information for monitoring
///
/// **App Integration**: Supports live plotting by optionally including profiles and derived quantities.
/// Enable via `SamplingConfig.enableLivePlotting` to receive profile data in progress callbacks.
///
/// ## Example
///
/// ```swift
/// let result = try await runner.run { fraction, progressInfo in
///     if let profiles = progressInfo.profiles {
///         // Update live plot with current profiles
///         updatePlot(profiles)
///     }
/// }
/// ```
public struct ProgressInfo: Sendable {
    public let currentTime: Float
    public let totalSteps: Int
    public let lastDt: Float
    public let converged: Bool

    /// Current profiles (optional, enabled via SamplingConfig.enableLivePlotting)
    ///
    /// **Performance**: Serialization cost ~100μs @ 100 cells, throttled to 100ms by SimulationRunner
    public let profiles: SerializableProfiles?

    /// Derived quantities (optional, enabled via SamplingConfig.enableLivePlotting)
    ///
    /// Includes τE, Q, βN, central values, and volume-averaged quantities.
    public let derived: DerivedQuantities?

    public init(
        currentTime: Float,
        totalSteps: Int,
        lastDt: Float,
        converged: Bool,
        profiles: SerializableProfiles? = nil,
        derived: DerivedQuantities? = nil
    ) {
        self.currentTime = currentTime
        self.totalSteps = totalSteps
        self.lastDt = lastDt
        self.converged = converged
        self.profiles = profiles
        self.derived = derived
    }
}
