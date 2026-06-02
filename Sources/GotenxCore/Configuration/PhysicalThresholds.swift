// PhysicalThresholds.swift
// Physical quantity thresholds for diagnostics and validation

import Foundation

/// Physical quantity thresholds (scaled to problem)
///
/// These thresholds replace hardcoded magic numbers (especially 1e-6)
/// with physically meaningful values that scale with problem size.
public struct PhysicalThresholds: Codable, Sendable, Equatable, Hashable {
    /// Fusion fuel fraction sum tolerance (default: 1e-4)
    /// Physical tolerance: 0.01% is reasonable for fraction sums
    /// (was 1e-6, which is too strict for float arithmetic)
    public let fuelFractionTolerance: Float

    /// Minimum fusion power for Q calculation [MW] (default: 1e-3)
    /// Below 1 kW, fusion gain Q is meaningless
    public let minimumFusionPowerForGain: Float

    /// Minimum heating power for τE calculation [MW] (default: 1e-2)
    /// Below 10 kW, energy confinement time is unreliable
    public let minimumHeatingPowerForEnergyConfinementTime: Float

    /// Poloidal flux relative variation threshold (default: 1e-5)
    /// Skip Ohmic heating calculation if dψ/ψ < threshold
    public let fluxVariationThreshold: Float

    /// Minimum stored energy for diagnostics [MJ] (default: 1e-3)
    /// Below 1 kJ, plasma is negligible
    public let minimumStoredEnergy: Float

    public init(
        fuelFractionTolerance: Float,
        minimumFusionPowerForGain: Float,
        minimumHeatingPowerForEnergyConfinementTime: Float,
        fluxVariationThreshold: Float,
        minimumStoredEnergy: Float
    ) {
        self.fuelFractionTolerance = fuelFractionTolerance
        self.minimumFusionPowerForGain = minimumFusionPowerForGain
        self.minimumHeatingPowerForEnergyConfinementTime = minimumHeatingPowerForEnergyConfinementTime
        self.fluxVariationThreshold = fluxVariationThreshold
        self.minimumStoredEnergy = minimumStoredEnergy
    }

    /// Default thresholds for ITER-scale tokamaks
    public static let `default` = PhysicalThresholds(
        fuelFractionTolerance: 1e-4,      // 0.01% (was 1e-6, too strict)
        minimumFusionPowerForGain: 1e-3,         // 1 kW (was 1e-6 MW, unrealistic)
        minimumHeatingPowerForEnergyConfinementTime: 1e-2,     // 10 kW (was implicit 1e-6)
        fluxVariationThreshold: 1e-5,     // 0.001% flux change (was 1e-6)
        minimumStoredEnergy: 1e-3             // 1 kJ (was 1e-6 MJ, too small)
    )
}
