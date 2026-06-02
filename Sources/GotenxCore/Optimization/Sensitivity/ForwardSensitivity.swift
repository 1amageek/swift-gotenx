// ForwardSensitivity.swift
// Forward sensitivity analysis using MLX automatic differentiation
//
// Computes: ∂outputs / ∂parameters
//
// Use cases:
// 1. Parameter sensitivity analysis (which parameters affect fusionGain most?)
// 2. Actuator ranking (prioritize control levers)
// 3. Uncertainty quantification

import Foundation
import MLX

/// Forward sensitivity analysis using automatic differentiation
///
/// **Purpose**: Compute gradients of simulation outputs w.r.t. control parameters
///
/// **Example**:
/// ```swift
/// let sensitivity = ForwardSensitivity(simulation: simulation)
///
/// let gradient = sensitivity.computeGradient(
///     initialProfiles: profiles,
///     actuators: actuators,
///     dynamicParameters: parameters,
///     timeHorizon: 2.0,
///     timeStep: 0.01
/// )
///
/// // gradient shows ∂fusionGain / ∂[ecrhPower, icrhPower, gasPuffRate, plasmaCurrent]
/// ```
public struct ForwardSensitivity {
    /// Differentiable simulation
    private let simulation: DifferentiableSimulation

    public init(simulation: DifferentiableSimulation) {
        self.simulation = simulation
    }

    // MARK: - Gradient Computation

    /// Compute gradient of loss w.r.t. actuators
    ///
    /// **Returns**: ∂loss / ∂actuators
    ///
    /// Uses MLX automatic differentiation to compute exact gradients.
    ///
    /// - Parameters:
    ///   - initialProfiles: Initial plasma profiles
    ///   - actuators: Control parameter time series
    ///   - dynamicParameters: Dynamic runtime parameters
    ///   - timeHorizon: Simulation time [s]
    ///   - timeStep: Fixed timestep [s]
    ///
    /// - Returns: Gradient actuators (same structure as input)
    ///
    /// **Performance**: ~10× slower than forward pass (backpropagation overhead)
    public func computeGradient(
        initialProfiles: CoreProfiles,
        actuators: ActuatorTimeSeries,
        dynamicParameters: DynamicRuntimeParameters,
        timeHorizon: Float,
        timeStep: Float
    ) -> ActuatorTimeSeries {
        let stepCount = actuators.stepCount
        let actuatorsArray = actuators.asMLXArray()

        // Define loss function (closure captures context)
        func lossFn(_ parameters: MLXArray) -> MLXArray {
            let acts = ActuatorTimeSeries(mlxArray: parameters, stepCount: stepCount)

            let (_, loss) = simulation.forward(
                initialProfiles: initialProfiles,
                actuators: acts,
                dynamicParameters: dynamicParameters,
                timeHorizon: timeHorizon,
                timeStep: timeStep
            )

            return loss
        }

        // Compute gradient via MLX automatic differentiation
        let gradFn = grad(lossFn)
        let gradient = gradFn(actuatorsArray)

        // Force evaluation (gradient is lazy)
        eval(gradient)

        // Convert back to ActuatorTimeSeries
        return ActuatorTimeSeries(mlxArray: gradient, stepCount: stepCount)
    }

    /// Compute gradient for custom objective function
    ///
    /// **Use case**: Optimize for objectives other than default (e.g., minimize power consumption)
    ///
    /// - Parameters:
    ///   - initialProfiles: Initial profiles
    ///   - actuators: Control parameters
    ///   - dynamicParameters: Dynamic parameters
    ///   - timeHorizon: Simulation time [s]
    ///   - timeStep: Timestep [s]
    ///   - objectiveFn: Custom objective (profiles → scalar loss)
    ///
    /// - Returns: Gradient w.r.t. actuators
    public func computeGradientWithCustomObjective(
        initialProfiles: CoreProfiles,
        actuators: ActuatorTimeSeries,
        dynamicParameters: DynamicRuntimeParameters,
        timeHorizon: Float,
        timeStep: Float,
        objectiveFn: @escaping (CoreProfiles) -> MLXArray
    ) -> ActuatorTimeSeries {
        let stepCount = actuators.stepCount
        let actuatorsArray = actuators.asMLXArray()

        func lossFn(_ parameters: MLXArray) -> MLXArray {
            let acts = ActuatorTimeSeries(mlxArray: parameters, stepCount: stepCount)

            let (finalProfiles, _) = simulation.forward(
                initialProfiles: initialProfiles,
                actuators: acts,
                dynamicParameters: dynamicParameters,
                timeHorizon: timeHorizon,
                timeStep: timeStep
            )

            // Apply custom objective
            return objectiveFn(finalProfiles)
        }

        let gradFn = grad(lossFn)
        let gradient = gradFn(actuatorsArray)
        eval(gradient)

        return ActuatorTimeSeries(mlxArray: gradient, stepCount: stepCount)
    }

    // MARK: - Sensitivity Matrix

    /// Compute sensitivity matrix: ∂outputs / ∂inputs
    ///
    /// **Purpose**: Analyze which actuators affect which outputs most
    ///
    /// **Returns**: Matrix[nOutputs, nActuators × stepCount]
    ///
    /// Each row shows sensitivity of one output to all actuator parameters.
    ///
    /// - Parameters:
    ///   - initialProfiles: Initial profiles
    ///   - actuators: Control parameters
    ///   - dynamicParameters: Dynamic parameters
    ///   - timeHorizon: Simulation time [s]
    ///   - timeStep: Timestep [s]
    ///   - outputs: Output quantities to analyze
    ///
    /// - Returns: Sensitivity matrix (each row = gradient for one output)
    ///
    /// **Example outputs**: ["Q_fusion", "tau_E", "beta_N", "H_factor"]
    public func computeSensitivityMatrix(
        initialProfiles: CoreProfiles,
        actuators: ActuatorTimeSeries,
        dynamicParameters: DynamicRuntimeParameters,
        timeHorizon: Float,
        timeStep: Float,
        outputs: [SensitivityOutput]
    ) -> SensitivityMatrix {
        var gradients: [[Float]] = []
        var outputNames: [String] = []

        let geometry = simulation.geometry

        for output in outputs {
            // Define objective for this output
            let objectiveFn: (CoreProfiles) -> MLXArray = { profiles in
                return output.evaluate(profiles: profiles, geometry: geometry)
            }

            // Compute gradient
            let gradient = computeGradientWithCustomObjective(
                initialProfiles: initialProfiles,
                actuators: actuators,
                dynamicParameters: dynamicParameters,
                timeHorizon: timeHorizon,
                timeStep: timeStep,
                objectiveFn: objectiveFn
            )

            // Convert to flat array
            let gradientArray = gradient.asMLXArray().asArray(Float.self)
            gradients.append(gradientArray)
            outputNames.append(output.name)
        }

        return SensitivityMatrix(
            outputs: outputNames,
            gradients: gradients,
            stepCount: actuators.stepCount
        )
    }

    // MARK: - Parameter Importance Analysis

    /// Analyze parameter importance (which actuators have most impact)
    ///
    /// **Returns**: Ranking of actuators by L2 norm of gradient
    ///
    /// High gradient magnitude → high sensitivity → important parameter
    ///
    /// - Parameters:
    ///   - initialProfiles: Initial profiles
    ///   - actuators: Control parameters
    ///   - dynamicParameters: Dynamic parameters
    ///   - timeHorizon: Simulation time [s]
    ///   - timeStep: Timestep [s]
    ///
    /// - Returns: Parameter importance ranking
    public func analyzeParameterImportance(
        initialProfiles: CoreProfiles,
        actuators: ActuatorTimeSeries,
        dynamicParameters: DynamicRuntimeParameters,
        timeHorizon: Float,
        timeStep: Float
    ) -> ParameterImportance {
        let gradient = computeGradient(
            initialProfiles: initialProfiles,
            actuators: actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )

        // Compute L2 norm for each actuator type
        let P_ECRH_importance = l2Norm(gradient.ecrhPower)
        let P_ICRH_importance = l2Norm(gradient.icrhPower)
        let gas_puff_importance = l2Norm(gradient.gasPuffRate)
        let I_plasma_importance = l2Norm(gradient.plasmaCurrent)

        return ParameterImportance(
            ecrhPower: P_ECRH_importance,
            icrhPower: P_ICRH_importance,
            gasPuffRate: gas_puff_importance,
            plasmaCurrent: I_plasma_importance
        )
    }

    // MARK: - Validation

    /// Validate gradient correctness using finite differences
    ///
    /// **Purpose**: Verify that MLX AD gradients are correct
    ///
    /// **Method**: Compare analytical gradient (AD) vs numerical gradient (finite diff)
    ///
    /// - Parameters:
    ///   - initialProfiles: Initial profiles
    ///   - actuators: Control parameters
    ///   - dynamicParameters: Dynamic parameters
    ///   - timeHorizon: Simulation time [s]
    ///   - timeStep: Timestep [s]
    ///   - epsilon: Finite difference step size
    ///   - sampleSize: Number of parameters to check (random sample)
    ///
    /// - Returns: Validation result with relative error
    public func validateGradient(
        initialProfiles: CoreProfiles,
        actuators: ActuatorTimeSeries,
        dynamicParameters: DynamicRuntimeParameters,
        timeHorizon: Float,
        timeStep: Float,
        epsilon: Float = 1e-4,
        sampleSize: Int = 10
    ) -> GradientValidationResult {
        // Analytical gradient (via AD)
        let analyticalGrad = computeGradient(
            initialProfiles: initialProfiles,
            actuators: actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )

        // Numerical gradient (via finite differences)
        let numericalGrad = computeNumericalGradient(
            initialProfiles: initialProfiles,
            actuators: actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep,
            epsilon: epsilon,
            sampleSize: sampleSize
        )

        // Compute relative error
        let relativeError = computeRelativeError(
            analytical: analyticalGrad,
            numerical: numericalGrad
        )

        return GradientValidationResult(
            relativeError: relativeError,
            passed: relativeError < 0.01,  // 1% threshold
            sampleSize: sampleSize
        )
    }

    /// Compute numerical gradient using finite differences (for validation)
    private func computeNumericalGradient(
        initialProfiles: CoreProfiles,
        actuators: ActuatorTimeSeries,
        dynamicParameters: DynamicRuntimeParameters,
        timeHorizon: Float,
        timeStep: Float,
        epsilon: Float,
        sampleSize: Int
    ) -> ActuatorTimeSeries {
        // Baseline loss
        let (_, baselineLoss) = simulation.forward(
            initialProfiles: initialProfiles,
            actuators: actuators,
            dynamicParameters: dynamicParameters,
            timeHorizon: timeHorizon,
            timeStep: timeStep
        )
        let baselineLossValue = baselineLoss.item(Float.self)

        // Sample random indices to perturb
        let totalParams = actuators.stepCount * 4
        let indices = (0..<totalParams).shuffled().prefix(sampleSize)

        // Compute numerical gradients
        var numericalGradients = [Float](repeating: 0.0, count: totalParams)

        for idx in indices {
            // Perturb parameter
            let perturbed = perturbActuator(actuators, at: idx, by: epsilon)

            // Compute perturbed loss
            let (_, perturbedLoss) = simulation.forward(
                initialProfiles: initialProfiles,
                actuators: perturbed,
                dynamicParameters: dynamicParameters,
                timeHorizon: timeHorizon,
                timeStep: timeStep
            )
            let perturbedLossValue = perturbedLoss.item(Float.self)

            // Finite difference gradient
            let gradient = (perturbedLossValue - baselineLossValue) / epsilon
            numericalGradients[idx] = gradient
        }

        return ActuatorTimeSeries(mlxArray:
            MLXArray(numericalGradients),
            stepCount: actuators.stepCount
        )
    }

    /// Perturb single actuator parameter
    private func perturbActuator(
        _ actuators: ActuatorTimeSeries,
        at index: Int,
        by epsilon: Float
    ) -> ActuatorTimeSeries {
        var array = actuators.asMLXArray().asArray(Float.self)
        array[index] += epsilon

        return ActuatorTimeSeries(mlxArray:
            MLXArray(array),
            stepCount: actuators.stepCount
        )
    }

    /// Compute relative error between analytical and numerical gradients
    private func computeRelativeError(
        analytical: ActuatorTimeSeries,
        numerical: ActuatorTimeSeries
    ) -> Float {
        let analyticalArray = analytical.asMLXArray().asArray(Float.self)
        let numericalArray = numerical.asMLXArray().asArray(Float.self)

        // L2 relative error
        var sumSquaredDiff: Float = 0.0
        var sumSquaredNumerical: Float = 0.0

        for i in 0..<analyticalArray.count {
            let diff = analyticalArray[i] - numericalArray[i]
            sumSquaredDiff += diff * diff
            sumSquaredNumerical += numericalArray[i] * numericalArray[i]
        }

        let l2Diff = sqrt(sumSquaredDiff)
        let l2Numerical = sqrt(sumSquaredNumerical)

        return l2Numerical > 1e-10 ? l2Diff / l2Numerical : l2Diff
    }

    // MARK: - Helper Functions

    /// Compute L2 norm of array
    private func l2Norm(_ array: [Float]) -> Float {
        let sumSquares = array.reduce(0.0) { $0 + $1 * $1 }
        return sqrt(sumSquares)
    }
}

// MARK: - Supporting Types

/// Sensitivity output quantity
public enum SensitivityOutput {
    case fusionGain
    case energyConfinementTime
    case normalizedBeta
    case confinementHFactor
    case custom(name: String, evaluator: (CoreProfiles, Geometry) -> MLXArray)

    public var name: String {
        switch self {
        case .fusionGain: return "Q_fusion"
        case .energyConfinementTime: return "tau_E"
        case .normalizedBeta: return "beta_N"
        case .confinementHFactor: return "H_factor"
        case .custom(let name, _): return name
        }
    }

    public func evaluate(profiles: CoreProfiles, geometry: Geometry) -> MLXArray {
        let derived = DerivedQuantitiesComputer.compute(
            profiles: profiles,
            geometry: geometry
        )

        switch self {
        case .fusionGain:
            return MLXArray(-derived.fusionGain)  // Negative for maximization
        case .energyConfinementTime:
            return MLXArray(-derived.energyConfinementTime)
        case .normalizedBeta:
            // Constraint: normalizedBeta < 3.5
            return smoothReLU(derived.normalizedBeta - 3.5)
        case .confinementHFactor:
            return MLXArray(-derived.confinementHFactor)
        case .custom(_, let evaluator):
            return evaluator(profiles, geometry)
        }
    }
}

/// Smooth ReLU for differentiable constraints
private func smoothReLU(_ x: Float) -> MLXArray {
    // ReLU(x) ≈ log(1 + exp(kx)) / k
    // Smooth approximation with k=10
    let k: Float = 10.0
    let xArray = MLXArray(x)
    return log(1.0 + exp(k * xArray)) / k
}

/// Sensitivity matrix result
public struct SensitivityMatrix {
    /// Output names
    public let outputs: [String]

    /// Gradients (one row per output)
    public let gradients: [[Float]]

    /// Number of timesteps
    public let stepCount: Int

    /// Get gradient for specific output
    public func gradient(for output: String) -> [Float]? {
        guard let index = outputs.firstIndex(of: output) else {
            return nil
        }
        return gradients[index]
    }

    /// Summary description
    public func summary() -> String {
        var lines: [String] = []
        lines.append("Sensitivity Matrix:")
        lines.append("  Outputs: \(outputs.count)")
        lines.append("  Parameters: \(gradients.first?.count ?? 0)")

        for (i, output) in outputs.enumerated() {
            let grad = gradients[i]
            let maxGrad = grad.max() ?? 0.0
            let l2Norm = sqrt(grad.reduce(0.0) { $0 + $1 * $1 })
            lines.append("  \(output): max=\(maxGrad), L2=\(l2Norm)")
        }

        return lines.joined(separator: "\n")
    }
}

/// Parameter importance ranking
public struct ParameterImportance {
    public let ecrhPower: Float
    public let icrhPower: Float
    public let gasPuffRate: Float
    public let plasmaCurrent: Float

    /// Sorted ranking (most important first)
    public var ranking: [(name: String, importance: Float)] {
        let parameters: [(name: String, importance: Float)] = [
            ("P_ECRH", ecrhPower),
            ("P_ICRH", icrhPower),
            ("gas_puff", gasPuffRate),
            ("I_plasma", plasmaCurrent)
        ]
        return parameters.sorted { $0.1 > $1.1 }
    }

    /// Summary description
    public func summary() -> String {
        var lines: [String] = []
        lines.append("Parameter Importance (L2 norm of gradient):")

        for (name, importance) in ranking {
            lines.append("  \(name): \(importance)")
        }

        return lines.joined(separator: "\n")
    }
}

/// Gradient validation result
public struct GradientValidationResult {
    /// Relative error (L2 norm)
    public let relativeError: Float

    /// Whether gradient passed validation (< 1% error)
    public let passed: Bool

    /// Number of parameters sampled
    public let sampleSize: Int

    /// Summary description
    public func summary() -> String {
        let status = passed ? "✅ PASSED" : "❌ FAILED"
        return """
        Gradient Validation: \(status)
          Relative Error: \(relativeError) (threshold: 0.01)
          Sample Size: \(sampleSize) parameters
        """
    }
}
