// ScenarioOptimizer.swift
// High-level optimization scenarios for tokamak operation
//
// Use cases:
// 1. Maximize fusionGain (fusion gain)
// 2. Match experimental target profiles
// 3. Optimize ramp-up/ramp-down trajectories

import Foundation
import MLX

/// Scenario optimizer for tokamak operation optimization
///
/// **Purpose**: High-level interface for common optimization scenarios
///
/// **Example 1: Maximize fusionGain**
/// ```swift
/// let result = try await ScenarioOptimizer.maximizeQFusion(
///     initialProfiles: profiles,
///     geometry: geometry,
///     staticParameters: staticParameters,
///     dynamicParameters: dynamicParameters,
///     timeHorizon: 2.0,
///     timeStep: 0.01,
///     constraints: .iter
/// )
///
/// print("Optimized fusionGain: \(result.fusionGain)")
/// ```
///
/// **Example 2: Match target profiles**
/// ```swift
/// let result = try ScenarioOptimizer.matchTargetProfiles(
///     initialProfiles: profiles,
///     targetProfiles: experimentalData,
///     ...
/// )
/// ```
public struct ScenarioOptimizer {

    // MARK: - fusionGain Maximization

    /// Optimize actuator trajectory to maximize fusion gain (fusionGain)
    ///
    /// **Objective**: Maximize Q = fusionPower / (auxiliaryPower + ohmicPower)
    ///
    /// **Method**: Adam optimizer with gradient-based search
    ///
    /// - Parameters:
    ///   - initialProfiles: Initial plasma profiles
    ///   - geometry: Tokamak geometry
    ///   - staticParameters: Static runtime parameters
    ///   - dynamicParameters: Dynamic runtime parameters (initial guess)
    ///   - timeHorizon: Simulation time [s]
    ///   - timeStep: Fixed timestep [s]
    ///   - constraints: Actuator constraints (power limits, etc.)
    ///   - optimizerConfig: Adam optimizer configuration
    ///
    /// - Returns: Optimization result with optimal actuators and achieved fusionGain
    public static func maximizeQFusion(
        initialProfiles: CoreProfiles,
        geometry: Geometry,
        staticParameters: StaticRuntimeParameters,
        dynamicParameters: DynamicRuntimeParameters,
        timeHorizon: Float,
        timeStep: Float,
        constraints: ActuatorConstraints = .iter,
        optimizerConfig: AdamConfig = .default
    ) throws -> ScenarioOptimizationResult {
        let stepCount = Int(timeHorizon / timeStep)

        // Initial guess: constant baseline actuators
        let initialActuators = ActuatorTimeSeries.constant(
            ecrhPower: 15.0,   // 15 MW ECRH
            icrhPower: 7.5,    // 7.5 MW ICRH
            gasPuffRate: 5e20, // 5×10²⁰ particles/s
            plasmaCurrent: 15.0, // 15 MA
            stepCount: stepCount
        )

        // Create differentiable simulation
        let simulation = DifferentiableSimulation(
            staticParameters: staticParameters,
            transport: createTransportModel(from: dynamicParameters),
            sources: createSourceModels(from: dynamicParameters),
            geometry: geometry
        )

        // Define optimization problem (maximize fusionGain)
        let problem = QFusionMaximization(
            simulation: simulation,
            initialProfiles: initialProfiles,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )

        // Create optimizer
        let optimizer = Adam(
            learningRate: optimizerConfig.learningRate,
            maximumIterations: optimizerConfig.maximumIterations,
            tolerance: optimizerConfig.tolerance,
            logInterval: optimizerConfig.logInterval
        )

        // Run optimization
        print("🎯 Optimizing for maximum fusionGain...")
        let result = optimizer.optimize(
            problem: problem,
            initialParams: initialActuators,
            constraints: constraints
        )

        // Compute final fusionGain
        let (finalProfiles, _) = simulation.forward(
            initialProfiles: initialProfiles,
            actuators: result.actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )

        let derived = DerivedQuantitiesComputer.compute(
            profiles: finalProfiles,
            geometry: geometry
        )

        return ScenarioOptimizationResult(
            actuators: result.actuators,
            finalProfiles: finalProfiles,
            fusionGain: derived.fusionGain,
            energyConfinementTime: derived.energyConfinementTime,
            normalizedBeta: derived.normalizedBeta,
            iterations: result.iterations,
            converged: result.converged,
            lossHistory: result.lossHistory
        )
    }

    // MARK: - Profile Matching

    /// Optimize actuators to match experimental target profiles
    ///
    /// **Objective**: Minimize L2 error between simulated and target profiles
    ///
    /// **Use case**: Reproduce experimental scenarios
    ///
    /// - Parameters:
    ///   - initialProfiles: Initial plasma profiles
    ///   - targetProfiles: Experimental target profiles to match
    ///   - geometry: Tokamak geometry
    ///   - staticParameters: Static runtime parameters
    ///   - dynamicParameters: Dynamic runtime parameters
    ///   - timeHorizon: Simulation time [s]
    ///   - timeStep: Fixed timestep [s]
    ///   - constraints: Actuator constraints
    ///   - optimizerConfig: Adam optimizer configuration
    ///
    /// - Returns: Optimization result with matched profiles
    public static func matchTargetProfiles(
        initialProfiles: CoreProfiles,
        targetProfiles: TargetProfiles,
        geometry: Geometry,
        staticParameters: StaticRuntimeParameters,
        dynamicParameters: DynamicRuntimeParameters,
        timeHorizon: Float,
        timeStep: Float,
        constraints: ActuatorConstraints = .iter,
        optimizerConfig: AdamConfig = .default
    ) throws -> ScenarioOptimizationResult {
        let stepCount = Int(timeHorizon / timeStep)

        // Initial guess
        let initialActuators = ActuatorTimeSeries.constant(
            ecrhPower: 10.0,
            icrhPower: 5.0,
            gasPuffRate: 1e20,
            plasmaCurrent: 15.0,
            stepCount: stepCount
        )

        // Create simulation
        let simulation = DifferentiableSimulation(
            staticParameters: staticParameters,
            transport: createTransportModel(from: dynamicParameters),
            sources: createSourceModels(from: dynamicParameters),
            geometry: geometry
        )

        // Define problem (minimize profile mismatch)
        let problem = ProfileMatching(
            simulation: simulation,
            initialProfiles: initialProfiles,
            targetProfiles: targetProfiles,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )

        // Optimize
        let optimizer = Adam(
            learningRate: optimizerConfig.learningRate,
            maximumIterations: optimizerConfig.maximumIterations,
            tolerance: optimizerConfig.tolerance
        )

        print("🎯 Optimizing to match target profiles...")
        let result = optimizer.optimize(
            problem: problem,
            initialParams: initialActuators,
            constraints: constraints
        )

        // Get final profiles
        let (finalProfiles, _) = simulation.forward(
            initialProfiles: initialProfiles,
            actuators: result.actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )

        let derived = DerivedQuantitiesComputer.compute(
            profiles: finalProfiles,
            geometry: geometry
        )

        return ScenarioOptimizationResult(
            actuators: result.actuators,
            finalProfiles: finalProfiles,
            fusionGain: derived.fusionGain,
            energyConfinementTime: derived.energyConfinementTime,
            normalizedBeta: derived.normalizedBeta,
            iterations: result.iterations,
            converged: result.converged,
            lossHistory: result.lossHistory
        )
    }

    // MARK: - Helper Functions

    /// Create transport model from dynamic parameters
    private static func createTransportModel(from parameters: DynamicRuntimeParameters) -> any TransportModel {
        // Use transport model from parameters
        // For now, return Bohm-GyroBohm as default
        return BohmGyroBohmTransportModel()
    }

    /// Create source models from dynamic parameters
    private static func createSourceModels(from parameters: DynamicRuntimeParameters) -> [any SourceModel] {
        // Import required: GotenxPhysics module for source adapters
        // For now, return empty array (TODO: wire up source models)
        // This requires importing GotenxPhysics which provides:
        // - FusionPowerSource
        // - OhmicHeatingSource
        // - BremsstrahlungSource
        // - IonElectronExchangeSource

        // Return empty for now to avoid circular dependency
        return []
    }
}

// MARK: - Optimization Problems

/// fusionGain maximization problem
struct QFusionMaximization: OptimizationProblem {
    let simulation: DifferentiableSimulation
    let initialProfiles: CoreProfiles
    let dynamicParameters: DynamicRuntimeParameters
    let timeHorizon: Float
    let timeStep: Float

    func objective(_ actuators: ActuatorTimeSeries) -> Float {
        let (_, loss) = simulation.forward(
            initialProfiles: initialProfiles,
            actuators: actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )
        return loss.item(Float.self)
    }

    func gradient(_ actuators: ActuatorTimeSeries) -> ActuatorTimeSeries {
        let sensitivity = ForwardSensitivity(simulation: simulation)
        return sensitivity.computeGradient(
            initialProfiles: initialProfiles,
            actuators: actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )
    }
}

/// Profile matching problem
struct ProfileMatching: OptimizationProblem {
    let simulation: DifferentiableSimulation
    let initialProfiles: CoreProfiles
    let targetProfiles: TargetProfiles
    let dynamicParameters: DynamicRuntimeParameters
    let timeHorizon: Float
    let timeStep: Float

    func objective(_ actuators: ActuatorTimeSeries) -> Float {
        let (finalProfiles, _) = simulation.forward(
            initialProfiles: initialProfiles,
            actuators: actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )

        // Compute L2 error
        let loss = simulation.computeProfileMatchingLoss(
            profiles: finalProfiles,
            target: targetProfiles
        )

        return loss.item(Float.self)
    }

    func gradient(_ actuators: ActuatorTimeSeries) -> ActuatorTimeSeries {
        let sensitivity = ForwardSensitivity(simulation: simulation)

        // Custom objective for profile matching
        let objectiveFn: (CoreProfiles) -> MLXArray = { profiles in
            return self.simulation.computeProfileMatchingLoss(
                profiles: profiles,
                target: self.targetProfiles
            )
        }

        return sensitivity.computeGradientWithCustomObjective(
            initialProfiles: initialProfiles,
            actuators: actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep,
            objectiveFn: objectiveFn
        )
    }
}

// MARK: - Configuration

/// Adam optimizer configuration
public struct AdamConfig: Sendable {
    public let learningRate: Float
    public let maximumIterations: Int
    public let tolerance: Float
    public let logInterval: Int

    public init(
        learningRate: Float,
        maximumIterations: Int,
        tolerance: Float,
        logInterval: Int = 10
    ) {
        self.learningRate = learningRate
        self.maximumIterations = maximumIterations
        self.tolerance = tolerance
        self.logInterval = logInterval
    }

    /// Default configuration for fusionGain optimization
    public static let `default` = AdamConfig(
        learningRate: 0.001,
        maximumIterations: 100,
        tolerance: 1e-4
    )

    /// Fast configuration (fewer iterations)
    public static let fast = AdamConfig(
        learningRate: 0.01,
        maximumIterations: 50,
        tolerance: 1e-3
    )

    /// Precise configuration (more iterations, tighter tolerance)
    public static let precise = AdamConfig(
        learningRate: 0.0005,
        maximumIterations: 200,
        tolerance: 1e-5
    )
}

// MARK: - Result

/// Scenario optimization result
public struct ScenarioOptimizationResult {
    /// Optimized actuator trajectory
    public let actuators: ActuatorTimeSeries

    /// Final plasma profiles
    public let finalProfiles: CoreProfiles

    /// Achieved fusion gain
    public let fusionGain: Float

    /// Energy confinement time [s]
    public let energyConfinementTime: Float

    /// Normalized beta
    public let normalizedBeta: Float

    /// Number of optimization iterations
    public let iterations: Int

    /// Whether optimization converged
    public let converged: Bool

    /// Loss history
    public let lossHistory: [Float]

    public init(
        actuators: ActuatorTimeSeries,
        finalProfiles: CoreProfiles,
        fusionGain: Float,
        energyConfinementTime: Float,
        normalizedBeta: Float,
        iterations: Int,
        converged: Bool,
        lossHistory: [Float]
    ) {
        self.actuators = actuators
        self.finalProfiles = finalProfiles
        self.fusionGain = fusionGain
        self.energyConfinementTime = energyConfinementTime
        self.normalizedBeta = normalizedBeta
        self.iterations = iterations
        self.converged = converged
        self.lossHistory = lossHistory
    }

    /// Summary description
    public func summary() -> String {
        let status = converged ? "✅ Converged" : "⚠️ Max iterations"
        return """
        Scenario Optimization Result: \(status)
          fusionGain: \(fusionGain)
          τE: \(energyConfinementTime) s
          βN: \(normalizedBeta)
          Iterations: \(iterations)
        """
    }
}
