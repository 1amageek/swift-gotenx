// PlotData.swift
// 2D simulation data container for plotting
//
// Converts Gotenx simulation data to display units:
// - Temperature: eV → keV
// - Density: m^-3 → 10^20 m^-3

import Foundation
import GotenxCore

/// Complete simulation output data for 2D plotting
public struct PlotData: Sendable {
    // MARK: - Coordinates

    /// Normalized radius ρ ∈ [0, 1] [cellCount]
    public let normalizedRadius: [Float]

    /// Time [s] [timeCount]
    public let time: [Float]

    // MARK: - Temperature & Density Profiles [timeCount, cellCount]

    /// Ion temperature [keV]
    public let ionTemperature: [[Float]]

    /// Electron temperature [keV]
    public let electronTemperature: [[Float]]

    /// Electron density [10^20 m^-3]
    public let electronDensity: [[Float]]

    // MARK: - Magnetic Field Profiles [timeCount, cellCount]

    /// Safety factor (dimensionless)
    public let safetyFactor: [[Float]]

    /// Magnetic shear (dimensionless)
    public let magneticShear: [[Float]]

    /// Poloidal flux [Wb]
    public let poloidalFlux: [[Float]]

    // MARK: - Transport Coefficients [timeCount, cellCount] [m^2/s]

    /// Total ion heat conductivity
    public let totalIonHeatConductivity: [[Float]]

    /// Total electron heat conductivity
    public let totalElectronHeatConductivity: [[Float]]

    /// Turbulent ion heat conductivity
    public let turbulentIonHeatConductivity: [[Float]]

    /// Turbulent electron heat conductivity
    public let turbulentElectronHeatConductivity: [[Float]]

    /// Particle diffusivity
    public let particleDiffusivity: [[Float]]

    // MARK: - Current Density Profiles [timeCount, cellCount] [MA/m^2]

    /// Total toroidal current density
    public let totalCurrentDensity: [[Float]]

    /// Ohmic current density
    public let ohmicCurrentDensity: [[Float]]

    /// Bootstrap current density
    public let bootstrapCurrentDensity: [[Float]]

    /// ECRH-driven current density
    public let ecrhCurrentDensity: [[Float]]

    // MARK: - Source Terms [timeCount, cellCount] [MW/m^3]

    /// Ohmic heating source
    public let ohmicHeatSource: [[Float]]

    /// Fusion heating source
    public let fusionHeatSource: [[Float]]

    /// ICRH ion heating density
    public let icrhIonHeatingPowerDensity: [[Float]]

    /// ICRH electron heating density
    public let icrhElectronHeatingPowerDensity: [[Float]]

    /// ECRH electron heating density
    public let ecrhElectronHeatingPowerDensity: [[Float]]

    // MARK: - Time Series Scalars [timeCount]

    /// Plasma current [MA]
    public let plasmaCurrent: [Float]

    /// Bootstrap current [MA]
    public let bootstrapCurrent: [Float]

    /// ECRH-driven current [MA]
    public let ecrhCurrent: [Float]

    /// Fusion gain (dimensionless)
    public let fusionGain: [Float]

    /// Auxiliary heating power [MW]
    public let auxiliaryHeatingPower: [Float]

    /// Ohmic heating power (electron) [MW]
    public let ohmicElectronHeatingPower: [Float]

    /// Alpha particle heating power [MW]
    public let totalAlphaPower: [Float]

    /// Bremsstrahlung radiation loss [MW]
    public let bremsstrahlungPower: [Float]

    /// Total radiation loss [MW]
    public let radiationPower: [Float]

    // MARK: - Utilities

    /// Number of time points
    public var timeCount: Int { time.count }

    /// Number of radial cells
    public var cellCount: Int { normalizedRadius.count }

    /// Time range
    public var timeRange: ClosedRange<Float> { time.first!...time.last! }

    /// Normalized radius range
    public var normalizedRadiusRange: ClosedRange<Float> { 0.0...1.0 }

    // MARK: - Initialization

    public init(
        normalizedRadius: [Float],
        time: [Float],
        ionTemperature: [[Float]],
        electronTemperature: [[Float]],
        electronDensity: [[Float]],
        safetyFactor: [[Float]],
        magneticShear: [[Float]],
        poloidalFlux: [[Float]],
        totalIonHeatConductivity: [[Float]],
        totalElectronHeatConductivity: [[Float]],
        turbulentIonHeatConductivity: [[Float]],
        turbulentElectronHeatConductivity: [[Float]],
        particleDiffusivity: [[Float]],
        totalCurrentDensity: [[Float]],
        ohmicCurrentDensity: [[Float]],
        bootstrapCurrentDensity: [[Float]],
        ecrhCurrentDensity: [[Float]],
        ohmicHeatSource: [[Float]],
        fusionHeatSource: [[Float]],
        icrhIonHeatingPowerDensity: [[Float]],
        icrhElectronHeatingPowerDensity: [[Float]],
        ecrhElectronHeatingPowerDensity: [[Float]],
        plasmaCurrent: [Float],
        bootstrapCurrent: [Float],
        ecrhCurrent: [Float],
        fusionGain: [Float],
        auxiliaryHeatingPower: [Float],
        ohmicElectronHeatingPower: [Float],
        totalAlphaPower: [Float],
        bremsstrahlungPower: [Float],
        radiationPower: [Float]
    ) {
        self.normalizedRadius = normalizedRadius
        self.time = time
        self.ionTemperature = ionTemperature
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity
        self.safetyFactor = safetyFactor
        self.magneticShear = magneticShear
        self.poloidalFlux = poloidalFlux
        self.totalIonHeatConductivity = totalIonHeatConductivity
        self.totalElectronHeatConductivity = totalElectronHeatConductivity
        self.turbulentIonHeatConductivity = turbulentIonHeatConductivity
        self.turbulentElectronHeatConductivity = turbulentElectronHeatConductivity
        self.particleDiffusivity = particleDiffusivity
        self.totalCurrentDensity = totalCurrentDensity
        self.ohmicCurrentDensity = ohmicCurrentDensity
        self.bootstrapCurrentDensity = bootstrapCurrentDensity
        self.ecrhCurrentDensity = ecrhCurrentDensity
        self.ohmicHeatSource = ohmicHeatSource
        self.fusionHeatSource = fusionHeatSource
        self.icrhIonHeatingPowerDensity = icrhIonHeatingPowerDensity
        self.icrhElectronHeatingPowerDensity = icrhElectronHeatingPowerDensity
        self.ecrhElectronHeatingPowerDensity = ecrhElectronHeatingPowerDensity
        self.plasmaCurrent = plasmaCurrent
        self.bootstrapCurrent = bootstrapCurrent
        self.ecrhCurrent = ecrhCurrent
        self.fusionGain = fusionGain
        self.auxiliaryHeatingPower = auxiliaryHeatingPower
        self.ohmicElectronHeatingPower = ohmicElectronHeatingPower
        self.totalAlphaPower = totalAlphaPower
        self.bremsstrahlungPower = bremsstrahlungPower
        self.radiationPower = radiationPower
    }
}

// MARK: - Conversion from SimulationResult

extension PlotData {
    /// Create PlotData from SimulationResult with unit conversion
    ///
    /// **Unit Conversions**:
    /// - Temperature: eV → keV (÷ 1000)
    /// - Density: m^-3 → 10^20 m^-3 (÷ 1e20)
    /// - Other quantities: No conversion
    ///
    /// - Parameter result: Simulation result with time series
    /// - Throws: If time series is missing
    public init(from result: SimulationResult) throws {
        guard let timeSeries = result.timeSeries, !timeSeries.isEmpty else {
            throw PlotDataError.missingTimeSeries
        }

        let timeCount = timeSeries.count
        let cellCount = timeSeries[0].profiles.ionTemperature.count

        // Generate normalizedRadius coordinate
        self.normalizedRadius = (0..<cellCount).map { Float($0) / Float(max(cellCount - 1, 1)) }

        // Extract time
        self.time = timeSeries.map { $0.time }

        // Convert temperature profiles: eV → keV
        self.ionTemperature = timeSeries.map { timePoint in
            timePoint.profiles.ionTemperature.map { $0 / 1000.0 }
        }
        self.electronTemperature = timeSeries.map { timePoint in
            timePoint.profiles.electronTemperature.map { $0 / 1000.0 }
        }

        // Convert density profiles: m^-3 → 10^20 m^-3
        self.electronDensity = timeSeries.map { timePoint in
            timePoint.profiles.electronDensity.map { $0 / 1e20 }
        }

        // Poloidal flux (no conversion)
        self.poloidalFlux = timeSeries.map { timePoint in
            timePoint.profiles.poloidalFlux
        }

        // Placeholder for unimplemented fields (filled with zeros)
        let zeroProfile = Array(repeating: Float(0.0), count: cellCount)
        let zeroProfiles = Array(repeating: zeroProfile, count: timeCount)

        self.safetyFactor = zeroProfiles
        self.magneticShear = zeroProfiles
        self.totalIonHeatConductivity = zeroProfiles
        self.totalElectronHeatConductivity = zeroProfiles
        self.turbulentIonHeatConductivity = zeroProfiles
        self.turbulentElectronHeatConductivity = zeroProfiles
        self.particleDiffusivity = zeroProfiles
        self.totalCurrentDensity = zeroProfiles
        self.ohmicCurrentDensity = zeroProfiles
        self.bootstrapCurrentDensity = zeroProfiles
        self.ecrhCurrentDensity = zeroProfiles
        self.ohmicHeatSource = zeroProfiles
        self.fusionHeatSource = zeroProfiles
        self.icrhIonHeatingPowerDensity = zeroProfiles
        self.icrhElectronHeatingPowerDensity = zeroProfiles
        self.ecrhElectronHeatingPowerDensity = zeroProfiles

        // Time series scalars
        // Phase 1: Attempt to extract from derived quantities if available
        // Phase 2+: Derived quantities should always be populated

        // Check if any time point has derived quantities
        let hasDerived = timeSeries.contains { $0.derived != nil }

        if hasDerived {
            // Extract from derived quantities with fallback to zero
            self.plasmaCurrent = timeSeries.map { $0.derived?.plasmaCurrent ?? 0.0 }
            self.bootstrapCurrent = timeSeries.map { $0.derived?.bootstrapCurrent ?? 0.0 }
            self.ecrhCurrent = Array(repeating: Float(0.0), count: timeCount)  // Not in DerivedQuantities yet

            // Fusion performance metrics
            self.fusionGain = timeSeries.map { timePoint in
                // Q = fusionPower / (auxiliaryPower + ohmicPower)
                guard let derived = timePoint.derived else { return 0.0 }
                let inputPower = derived.auxiliaryPower + derived.ohmicPower + 1e-10
                return derived.fusionPower / inputPower
            }

            self.auxiliaryHeatingPower = timeSeries.map { $0.derived?.auxiliaryPower ?? 0.0 }
            self.ohmicElectronHeatingPower = timeSeries.map { $0.derived?.ohmicPower ?? 0.0 }
            self.totalAlphaPower = timeSeries.map { $0.derived?.alphaPower ?? 0.0 }
            self.bremsstrahlungPower = Array(repeating: Float(0.0), count: timeCount)  // Not in DerivedQuantities
            self.radiationPower = Array(repeating: Float(0.0), count: timeCount)  // Not in DerivedQuantities
        } else {
            // Phase 1: No derived quantities available, use zeros
            let zeroScalar = Array(repeating: Float(0.0), count: timeCount)
            self.plasmaCurrent = zeroScalar
            self.bootstrapCurrent = zeroScalar
            self.ecrhCurrent = zeroScalar
            self.fusionGain = zeroScalar
            self.auxiliaryHeatingPower = zeroScalar
            self.ohmicElectronHeatingPower = zeroScalar
            self.totalAlphaPower = zeroScalar
            self.bremsstrahlungPower = zeroScalar
            self.radiationPower = zeroScalar
        }
    }
}

// MARK: - Errors

public enum PlotDataError: LocalizedError {
    case missingTimeSeries
    case inconsistentDataShape

    public var errorDescription: String? {
        switch self {
        case .missingTimeSeries:
            return "SimulationResult must contain time series data for plotting"
        case .inconsistentDataShape:
            return "Data arrays have inconsistent shapes"
        }
    }
}
