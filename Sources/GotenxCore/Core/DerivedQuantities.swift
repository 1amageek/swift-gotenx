// DerivedQuantities.swift
// Derived scalar quantities for performance monitoring and visualization
//
// Phase 1 Implementation: Minimal structure with default values
// Phase 2 Implementation: Actual computation from CoreProfiles
// Phase 3 Implementation: Advanced metrics using transport/sources

import Foundation
import MLX

/// Derived scalar quantities computed from simulation state
///
/// **Design Philosophy**:
/// - All quantities are scalars (0D) - cheap to compute and store
/// - Computed at every timestep for real-time monitoring
/// - Independent of transport models (Phase 1-2) or dependent (Phase 3)
///
/// **Implementation Phases**:
/// - Phase 1 (Current): Returns default values (zeros)
/// - Phase 2: Computes central values, volume averages, total energies
/// - Phase 3: Computes confinement metrics (τE, Q, βN) using transport/sources
public struct DerivedQuantities: Sendable, Codable, Equatable {
    // MARK: - Central Values

    /// Ion temperature at magnetic axis (ρ=0) [eV]
    ///
    /// **Unit**: eV (TORAX internal standard, NOT keV)
    public let coreIonTemperature: Float

    /// Electron temperature at magnetic axis (ρ=0) [eV]
    ///
    /// **Unit**: eV (TORAX internal standard, NOT keV)
    public let coreElectronTemperature: Float

    /// Electron density at magnetic axis (ρ=0) [m^-3]
    ///
    /// **Unit**: m^-3 (TORAX internal standard, NOT 10^20 m^-3)
    public let coreElectronDensity: Float

    // MARK: - Volume Averages

    /// Volume-averaged electron density [m^-3]
    ///
    /// **Unit**: m^-3 (TORAX internal standard, NOT 10^20 m^-3)
    public let averageElectronDensity: Float

    /// Volume-averaged ion temperature [eV]
    ///
    /// **Unit**: eV (TORAX internal standard, NOT keV)
    public let averageIonTemperature: Float

    /// Volume-averaged electron temperature [eV]
    ///
    /// **Unit**: eV (TORAX internal standard, NOT keV)
    public let averageElectronTemperature: Float

    // MARK: - Total Energies

    /// Total thermal energy [MJ]
    public let thermalEnergy: Float

    /// Ion thermal energy [MJ]
    public let ionThermalEnergy: Float

    /// Electron thermal energy [MJ]
    public let electronThermalEnergy: Float

    // MARK: - Fusion Performance

    /// Fusion power [MW]
    public let fusionPower: Float

    /// Alpha particle heating power [MW]
    public let alphaPower: Float

    /// Auxiliary heating power [MW]
    public let auxiliaryPower: Float

    /// Ohmic heating power [MW]
    ///
    /// **Added**: For complete Q = fusionPower / (auxiliaryPower + ohmicPower) calculation
    public let ohmicPower: Float

    /// Fusion gain Q = fusionPower / P_input
    ///
    /// **Definition**: Q = fusionPower / (auxiliaryPower + ohmicPower)
    ///
    /// **Key values**:
    /// - Q = 1: Breakeven (fusion power equals input power)
    /// - Q = 10: ITER design target
    /// - Q = ∞: Ignition (self-sustaining fusion)
    ///
    /// **Note**: Alpha power (alphaPower) is NOT included in P_input since it's
    /// internally generated. Only external heating (auxiliary + ohmic) counts.
    public let fusionGain: Float

    // MARK: - Confinement Metrics

    /// Energy confinement time [s]
    public let energyConfinementTime: Float

    /// Energy confinement time from scaling law [s]
    public let scalingEnergyConfinementTime: Float

    /// H-factor (τE / τE_scaling)
    public let confinementHFactor: Float

    // MARK: - Beta Limits

    /// Toroidal beta [%]
    public let toroidalBeta: Float

    /// Poloidal beta
    public let poloidalBeta: Float

    /// Normalized beta βN = β(%) × a(m) × B(T) / Ip(MA)
    public let normalizedBeta: Float

    /// Troyon beta limit
    public let normalizedBetaLimit: Float

    // MARK: - Current Drive

    /// Total plasma current [MA]
    public let plasmaCurrent: Float

    /// Bootstrap current [MA]
    public let bootstrapCurrent: Float

    /// Bootstrap fraction f_bs = bootstrapCurrent / plasmaCurrent
    public let bootstrapFraction: Float

    // MARK: - Triple Product

    /// Lawson triple product n⟨T⟩τE [eV s m^-3]
    ///
    /// **Unit**: eV s m^-3 (TORAX internal standard)
    ///
    /// **Note**: Commonly displayed as 10^21 keV s m^-3 in literature.
    /// For conversion: value_displayed = tripleProduct / 1e24
    public let tripleProduct: Float

    // MARK: - Initialization

    public init(
        coreIonTemperature: Float,
        coreElectronTemperature: Float,
        coreElectronDensity: Float,
        averageElectronDensity: Float,
        averageIonTemperature: Float,
        averageElectronTemperature: Float,
        thermalEnergy: Float,
        ionThermalEnergy: Float,
        electronThermalEnergy: Float,
        fusionPower: Float,
        alphaPower: Float,
        auxiliaryPower: Float,
        ohmicPower: Float,
        fusionGain: Float,
        energyConfinementTime: Float,
        scalingEnergyConfinementTime: Float,
        confinementHFactor: Float,
        toroidalBeta: Float,
        poloidalBeta: Float,
        normalizedBeta: Float,
        normalizedBetaLimit: Float,
        plasmaCurrent: Float,
        bootstrapCurrent: Float,
        bootstrapFraction: Float,
        tripleProduct: Float
    ) {
        self.coreIonTemperature = coreIonTemperature
        self.coreElectronTemperature = coreElectronTemperature
        self.coreElectronDensity = coreElectronDensity
        self.averageElectronDensity = averageElectronDensity
        self.averageIonTemperature = averageIonTemperature
        self.averageElectronTemperature = averageElectronTemperature
        self.thermalEnergy = thermalEnergy
        self.ionThermalEnergy = ionThermalEnergy
        self.electronThermalEnergy = electronThermalEnergy
        self.fusionPower = fusionPower
        self.alphaPower = alphaPower
        self.auxiliaryPower = auxiliaryPower
        self.ohmicPower = ohmicPower
        self.fusionGain = fusionGain
        self.energyConfinementTime = energyConfinementTime
        self.scalingEnergyConfinementTime = scalingEnergyConfinementTime
        self.confinementHFactor = confinementHFactor
        self.toroidalBeta = toroidalBeta
        self.poloidalBeta = poloidalBeta
        self.normalizedBeta = normalizedBeta
        self.normalizedBetaLimit = normalizedBetaLimit
        self.plasmaCurrent = plasmaCurrent
        self.bootstrapCurrent = bootstrapCurrent
        self.bootstrapFraction = bootstrapFraction
        self.tripleProduct = tripleProduct
    }
}

// MARK: - Phase 1: Default Values

extension DerivedQuantities {
    /// Phase 1 implementation: Return default zeros
    ///
    /// **Rationale**: Allows compilation and testing without breaking existing code.
    /// Actual computation will be implemented in Phase 2.
    public static let zero = DerivedQuantities(
        coreIonTemperature: 0,
        coreElectronTemperature: 0,
        coreElectronDensity: 0,
        averageElectronDensity: 0,
        averageIonTemperature: 0,
        averageElectronTemperature: 0,
        thermalEnergy: 0,
        ionThermalEnergy: 0,
        electronThermalEnergy: 0,
        fusionPower: 0,
        alphaPower: 0,
        auxiliaryPower: 0,
        ohmicPower: 0,
        fusionGain: 0,
        energyConfinementTime: 0,
        scalingEnergyConfinementTime: 0,
        confinementHFactor: 0,
        toroidalBeta: 0,
        poloidalBeta: 0,
        normalizedBeta: 0,
        normalizedBetaLimit: 0,
        plasmaCurrent: 0,
        bootstrapCurrent: 0,
        bootstrapFraction: 0,
        tripleProduct: 0
    )
}
