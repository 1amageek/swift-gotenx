// Adam.swift
// Adaptive Moment Estimation optimizer for gradient-based optimization
//
// Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015)
//
// Key features:
// 1. Adaptive learning rates for each parameter
// 2. Momentum (first moment) for stable convergence
// 3. RMSprop (second moment) for adaptive scaling
// 4. Bias correction for early iterations

import Foundation
import MLX

/// Adam optimizer (Adaptive Moment Estimation)
///
/// **Algorithm**:
/// ```
/// m_t = β₁ m_{t-1} + (1-β₁) g_t           // First moment (momentum)
/// v_t = β₂ v_{t-1} + (1-β₂) g_t²          // Second moment (RMSprop)
/// m̂_t = m_t / (1 - β₁^t)                  // Bias correction
/// v̂_t = v_t / (1 - β₂^t)                  // Bias correction
/// θ_t = θ_{t-1} - α m̂_t / (√v̂_t + ε)    // Parameter update
/// ```
///
/// **Advantages over gradient descent**:
/// - Adaptive learning rate per parameter
/// - More stable convergence
/// - Better for noisy gradients
/// - Widely used in deep learning
///
/// **Example**:
/// ```swift
/// let optimizer = Adam(
///     learningRate: 0.001,
///     maximumIterations: 100
/// )
///
/// let result = optimizer.optimize(
///     problem: qFusionProblem,
///     initialParams: baselineActuators,
///     constraints: .iter
/// )
/// ```
public struct Adam {
    /// Learning rate (α)
    public let learningRate: Float

    /// First moment decay rate (β₁)
    ///
    /// **Default**: 0.9 (recommended by original paper)
    /// Higher values → more momentum
    public let beta1: Float

    /// Second moment decay rate (β₂)
    ///
    /// **Default**: 0.999 (recommended by original paper)
    /// Higher values → smoother adaptive scaling
    public let beta2: Float

    /// Numerical stability constant (ε)
    ///
    /// **Default**: 1e-8
    /// Prevents division by zero
    public let epsilon: Float

    /// Maximum number of iterations
    public let maximumIterations: Int

    /// Convergence tolerance
    ///
    /// Optimization stops when |Δloss| < tolerance
    public let tolerance: Float

    /// Logging interval (print progress every N iterations)
    public let logInterval: Int

    // MARK: - Initialization

    public init(
        learningRate: Float = 0.001,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        maximumIterations: Int = 100,
        tolerance: Float = 1e-4,
        logInterval: Int = 10
    ) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.maximumIterations = maximumIterations
        self.tolerance = tolerance
        self.logInterval = logInterval
    }

    // MARK: - Optimization

    /// Optimize using Adam algorithm
    ///
    /// - Parameters:
    ///   - problem: Optimization problem (provides objective + gradient)
    ///   - initialParams: Initial parameter guess
    ///   - constraints: Parameter constraints (hard limits)
    ///
    /// - Returns: Optimization result with final parameters and loss
    public func optimize(
        problem: OptimizationProblem,
        initialParams: ActuatorTimeSeries,
        constraints: ActuatorConstraints
    ) -> OptimizationResult {
        var parameters = initialParams
        var paramsArray = parameters.asMLXArray()

        // Initialize moments
        var m = MLXArray.zeros(like: paramsArray)  // First moment
        var v = MLXArray.zeros(like: paramsArray)  // Second moment

        var bestLoss = Float.infinity
        var bestParams = parameters
        var lossHistory: [Float] = []

        print("Adam Optimizer starting...")
        print("  Learning rate: \(learningRate)")
        print("  Max iterations: \(maximumIterations)")
        print("  Tolerance: \(tolerance)")

        for t in 1...maximumIterations {
            // Compute gradient
            let gradient = problem.gradient(parameters)
            let gradArray = gradient.asMLXArray()

            // Update biased first moment estimate
            // m_t = β₁ m_{t-1} + (1-β₁) g_t
            m = beta1 * m + (1 - beta1) * gradArray

            // Update biased second moment estimate
            // v_t = β₂ v_{t-1} + (1-β₂) g_t²
            v = beta2 * v + (1 - beta2) * (gradArray * gradArray)

            // Bias correction
            // m̂_t = m_t / (1 - β₁^t)
            let beta1_t = pow(beta1, Float(t))
            let beta2_t = pow(beta2, Float(t))
            let mHat = m / (1 - beta1_t)
            let vHat = v / (1 - beta2_t)

            // Parameter update
            // θ_t = θ_{t-1} - α m̂_t / (√v̂_t + ε)
            paramsArray = paramsArray - learningRate * mHat / (sqrt(vHat) + epsilon)

            // Apply constraints directly on MLXArray (gradient-preserving)
            paramsArray = applyConstraintsMLX(
                paramsArray,
                constraints: constraints,
                stepCount: parameters.stepCount
            )

            // Evaluate parameters array
            eval(paramsArray)

            // Convert back to ActuatorTimeSeries (gradient-preserving)
            parameters = ActuatorTimeSeries(mlxArray: paramsArray, stepCount: parameters.stepCount)

            // Evaluate loss
            let loss = problem.objective(parameters)
            lossHistory.append(loss)

            // Update best
            if loss < bestLoss {
                bestLoss = loss
                bestParams = parameters
            }

            // Log progress
            if t % logInterval == 0 || t == 1 {
                let gradSquared = gradArray * gradArray
                let gradNorm = sqrt(MLX.sum(gradSquared).item(Float.self))
                print("  Iteration \(t): loss = \(loss), grad_norm = \(gradNorm)")
            }

            // Check convergence
            if t > 1 {
                let deltaLoss = abs(lossHistory[t-1] - lossHistory[t-2])
                if deltaLoss < tolerance {
                    print("✅ Converged at iteration \(t) (Δloss = \(deltaLoss))")
                    return OptimizationResult(
                        actuators: bestParams,
                        finalLoss: bestLoss,
                        iterations: t,
                        converged: true,
                        lossHistory: lossHistory
                    )
                }
            }
        }

        print("⚠️ Max iterations reached without convergence")
        return OptimizationResult(
            actuators: bestParams,
            finalLoss: bestLoss,
            iterations: maximumIterations,
            converged: false,
            lossHistory: lossHistory
        )
    }

    // MARK: - Helper Functions

    /// Apply constraints on MLXArray (gradient-preserving)
    ///
    /// Uses MLX clamp operation to maintain differentiability
    private func applyConstraintsMLX(
        _ array: MLXArray,
        constraints: ActuatorConstraints,
        stepCount: Int
    ) -> MLXArray {
        // Create constraint bounds as MLXArrays
        let nActuators = 4
        var minBounds = [Float](repeating: 0, count: stepCount * nActuators)
        var maxBounds = [Float](repeating: 0, count: stepCount * nActuators)

        // ecrhPower bounds
        for i in 0..<stepCount {
            minBounds[i] = constraints.minimumECRHPower
            maxBounds[i] = constraints.maximumECRHPower
        }

        // icrhPower bounds
        for i in stepCount..<(2*stepCount) {
            minBounds[i] = constraints.minimumICRHPower
            maxBounds[i] = constraints.maximumICRHPower
        }

        // gasPuffRate bounds
        for i in (2*stepCount)..<(3*stepCount) {
            minBounds[i] = constraints.minimumGasPuffRate
            maxBounds[i] = constraints.maximumGasPuffRate
        }

        // plasmaCurrent bounds
        for i in (3*stepCount)..<(4*stepCount) {
            minBounds[i] = constraints.minimumCurrent
            maxBounds[i] = constraints.maximumCurrent
        }

        let minArray = MLXArray(minBounds)
        let maxArray = MLXArray(maxBounds)

        // Apply clamp (differentiable!)
        return clip(array, min: minArray, max: maxArray)
    }
}

// MARK: - Optimization Problem Protocol

/// Optimization problem interface
///
/// Implement this protocol to define custom optimization objectives
public protocol OptimizationProblem {
    /// Objective function (to minimize)
    func objective(_ parameters: ActuatorTimeSeries) -> Float

    /// Gradient of objective w.r.t. parameters
    func gradient(_ parameters: ActuatorTimeSeries) -> ActuatorTimeSeries
}

// MARK: - Optimization Result

/// Optimization result
public struct OptimizationResult {
    /// Optimized actuator trajectory
    public let actuators: ActuatorTimeSeries

    /// Final loss value
    public let finalLoss: Float

    /// Number of iterations
    public let iterations: Int

    /// Whether optimization converged
    public let converged: Bool

    /// Loss history (one per iteration)
    public let lossHistory: [Float]

    public init(
        actuators: ActuatorTimeSeries,
        finalLoss: Float,
        iterations: Int,
        converged: Bool,
        lossHistory: [Float] = []
    ) {
        self.actuators = actuators
        self.finalLoss = finalLoss
        self.iterations = iterations
        self.converged = converged
        self.lossHistory = lossHistory
    }

    /// Summary description
    public func summary() -> String {
        let status = converged ? "✅ Converged" : "⚠️ Max iterations"
        return """
        Optimization Result: \(status)
          Final loss: \(finalLoss)
          Iterations: \(iterations)
          Improvement: \(improvementPercentage())%
        """
    }

    /// Improvement percentage (baseline = first loss)
    private func improvementPercentage() -> Float {
        guard let firstLoss = lossHistory.first, firstLoss > 0 else {
            return 0.0
        }
        return ((firstLoss - finalLoss) / firstLoss) * 100.0
    }
}
