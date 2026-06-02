// ActuatorTimeSeries.swift
// Differentiable control parameters for optimization

import Foundation
import MLX

/// Actuator time series for optimization
///
/// Represents control parameters (ECRH power, ICRH power, gas puff, plasma current)
/// that can be optimized using gradient-based methods.
///
/// **Design (Gradient-preserving)**:
/// - Internal representation: MLXArray (preserves gradient tape)
/// - External interface: [Float] accessors (for convenience)
/// - Shape: [stepCount, 4] where 4 = [ecrhPower, icrhPower, gasPuffRate, plasmaCurrent]
/// - All operations maintain differentiability
public struct ActuatorTimeSeries {
    /// Internal MLXArray representation (gradient-preserving)
    /// Shape: [stepCount × 4] (flattened for optimization)
    private let data: MLXArray

    /// Number of timesteps
    public let stepCount: Int

    // MARK: - Read-only accessors (for display/logging)

    /// ECRH power at each timestep [MW]
    public var ecrhPower: [Float] {
        let start = 0
        let end = stepCount
        return Array(data.asArray(Float.self)[start..<end])
    }

    /// ICRH power at each timestep [MW]
    public var icrhPower: [Float] {
        let start = stepCount
        let end = 2 * stepCount
        return Array(data.asArray(Float.self)[start..<end])
    }

    /// Gas puff rate at each timestep [particles/s]
    public var gasPuffRate: [Float] {
        let start = 2 * stepCount
        let end = 3 * stepCount
        return Array(data.asArray(Float.self)[start..<end])
    }

    /// Plasma current at each timestep [MA]
    public var plasmaCurrent: [Float] {
        let start = 3 * stepCount
        let end = 4 * stepCount
        return Array(data.asArray(Float.self)[start..<end])
    }

    // MARK: - Initialization

    /// Create actuator time series from Float arrays
    public init(
        ecrhPower: [Float],
        icrhPower: [Float],
        gasPuffRate: [Float],
        plasmaCurrent: [Float]
    ) {
        precondition(ecrhPower.count == icrhPower.count, "All actuators must have same length")
        precondition(ecrhPower.count == gasPuffRate.count, "All actuators must have same length")
        precondition(ecrhPower.count == plasmaCurrent.count, "All actuators must have same length")
        precondition(ecrhPower.count > 0, "Must have at least one timestep")

        self.stepCount = ecrhPower.count

        // Create flat MLXArray (gradient-preserving)
        let flat = ecrhPower + icrhPower + gasPuffRate + plasmaCurrent
        self.data = MLXArray(flat)
    }

    /// Create from MLXArray (preserves gradient tape)
    public init(mlxArray: MLXArray, stepCount: Int) {
        precondition(mlxArray.shape[0] == stepCount * 4,
                    "Array shape \(mlxArray.shape[0]) != stepCount (\(stepCount)) × 4")

        self.data = mlxArray
        self.stepCount = stepCount
    }

    /// Create constant actuators (same value at all timesteps)
    public static func constant(
        ecrhPower: Float,
        icrhPower: Float,
        gasPuffRate: Float,
        plasmaCurrent: Float,
        stepCount: Int
    ) -> ActuatorTimeSeries {
        precondition(stepCount > 0, "Must have at least one timestep")

        return ActuatorTimeSeries(
            ecrhPower: [Float](repeating: ecrhPower, count: stepCount),
            icrhPower: [Float](repeating: icrhPower, count: stepCount),
            gasPuffRate: [Float](repeating: gasPuffRate, count: stepCount),
            plasmaCurrent: [Float](repeating: plasmaCurrent, count: stepCount)
        )
    }

    // MARK: - MLXArray Conversion (Gradient-preserving)

    /// Convert to MLXArray for differentiation
    ///
    /// **Critical**: Returns internal MLXArray directly (no copy)
    /// This preserves the gradient tape for automatic differentiation
    ///
    /// Layout: [P_ECRH_0, ..., P_ECRH_N, P_ICRH_0, ..., P_ICRH_N, ...]
    /// Total length: stepCount × 4
    public func asMLXArray() -> MLXArray {
        return data  // Return internal representation (gradient-preserving!)
    }

    /// Get actuator values at specific timestep index
    public func values(atStep step: Int) -> ActuatorValues {
        precondition(step >= 0 && step < stepCount, "Step \(step) out of range [0, \(stepCount))")

        return ActuatorValues(
            ecrhPower: ecrhPower[step],
            icrhPower: icrhPower[step],
            gasPuffRate: gasPuffRate[step],
            plasmaCurrent: plasmaCurrent[step]
        )
    }

    /// Get actuator values at specific time (interpolated)
    public func values(atTime time: Float, timeStep: Float) -> ActuatorValues {
        let step = Int(time / timeStep)

        // Clamp to valid range
        let clampedStep = max(0, min(step, stepCount - 1))

        return values(atStep: clampedStep)
    }
}

/// Actuator values at a single timestep
public struct ActuatorValues {
    /// ECRH power [MW]
    public let ecrhPower: Float

    /// ICRH power [MW]
    public let icrhPower: Float

    /// Gas puff rate [particles/s]
    public let gasPuffRate: Float

    /// Plasma current [MA]
    public let plasmaCurrent: Float

    public init(
        ecrhPower: Float,
        icrhPower: Float,
        gasPuffRate: Float,
        plasmaCurrent: Float
    ) {
        self.ecrhPower = ecrhPower
        self.icrhPower = icrhPower
        self.gasPuffRate = gasPuffRate
        self.plasmaCurrent = plasmaCurrent
    }
}

/// Actuator constraints (physical limits)
public struct ActuatorConstraints: Sendable {
    public let minimumECRHPower: Float
    public let maximumECRHPower: Float
    public let minimumICRHPower: Float
    public let maximumICRHPower: Float
    public let minimumCurrent: Float
    public let maximumCurrent: Float
    public let minimumGasPuffRate: Float
    public let maximumGasPuffRate: Float

    public init(
        minimumECRHPower: Float,
        maximumECRHPower: Float,
        minimumICRHPower: Float,
        maximumICRHPower: Float,
        minimumCurrent: Float,
        maximumCurrent: Float,
        minimumGasPuffRate: Float,
        maximumGasPuffRate: Float
    ) {
        self.minimumECRHPower = minimumECRHPower
        self.maximumECRHPower = maximumECRHPower
        self.minimumICRHPower = minimumICRHPower
        self.maximumICRHPower = maximumICRHPower
        self.minimumCurrent = minimumCurrent
        self.maximumCurrent = maximumCurrent
        self.minimumGasPuffRate = minimumGasPuffRate
        self.maximumGasPuffRate = maximumGasPuffRate
    }

    /// ITER Baseline constraints
    public static let iter = ActuatorConstraints(
        minimumECRHPower: 0.0,
        maximumECRHPower: 30.0,        // 30 MW maximum
        minimumICRHPower: 0.0,
        maximumICRHPower: 20.0,        // 20 MW maximum
        minimumCurrent: 5.0,      // 5 MA minimum
        maximumCurrent: 20.0,     // 20 MA maximum (ITER: 15 MA baseline)
        minimumGasPuffRate: 0.0,
        maximumGasPuffRate: 1e21      // 10²¹ particles/s maximum
    )

    /// Apply constraints (clamp to limits)
    public func apply(to actuators: ActuatorTimeSeries) -> ActuatorTimeSeries {
        return ActuatorTimeSeries(
            ecrhPower: actuators.ecrhPower.map { clamp($0, min: minimumECRHPower, max: maximumECRHPower) },
            icrhPower: actuators.icrhPower.map { clamp($0, min: minimumICRHPower, max: maximumICRHPower) },
            gasPuffRate: actuators.gasPuffRate.map { clamp($0, min: minimumGasPuffRate, max: maximumGasPuffRate) },
            plasmaCurrent: actuators.plasmaCurrent.map { clamp($0, min: minimumCurrent, max: maximumCurrent) }
        )
    }

    private func clamp(_ value: Float, min: Float, max: Float) -> Float {
        return Swift.max(min, Swift.min(max, value))
    }
}
