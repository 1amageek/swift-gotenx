// DerivedQuantitiesComputer.swift
// Computes derived scalar quantities from simulation state
//
// Phase 2 Implementation: Basic metrics (central values, averages, energies)
// Phase 3 Implementation: Advanced metrics (τE, Q, βN) requiring transport/sources

import Foundation
import MLX

/// Computes derived scalar quantities from simulation state
///
/// **Design Philosophy**:
/// - Pure functions: No side effects, deterministic outputs
/// - GPU-optimized: All MLXArray operations stay on GPU
/// - Unit-aware: Explicit unit conversions documented
///
/// **Implementation Phases**:
/// - Phase 2 (Current): Central values, volume averages, total energies
/// - Phase 3: Confinement metrics (τE, H-factor), beta limits, fusion performance
public enum DerivedQuantitiesComputer {

    // MARK: - Phase 2: Basic Metrics

    /// Compute derived quantities from simulation state
    ///
    /// **Phase 2**: Computes only basic metrics (central values, averages, energies)
    /// **Phase 3**: Will add advanced metrics (τE, Q, βN)
    ///
    /// - Parameters:
    ///   - profiles: Current plasma profiles
    ///   - geometry: Tokamak geometry
    ///   - transport: Transport coefficients (Phase 3)
    ///   - sources: Source terms (Phase 3)
    /// - Returns: Computed derived quantities
    public static func compute(
        profiles: CoreProfiles,
        geometry: Geometry,
        transport: TransportCoefficients? = nil,
        sources: SourceTerms? = nil
    ) -> DerivedQuantities {

        // Phase 2: Compute basic metrics
        let centralValues = computeCentralValues(profiles: profiles)
        let volumeAverages = computeVolumeAverages(profiles: profiles, geometry: geometry)
        let totalEnergies = computeTotalEnergies(profiles: profiles, geometry: geometry)

        // Phase 3: Placeholder for advanced metrics (will be implemented later)
        let advancedMetrics = computeAdvancedMetrics(
            profiles: profiles,
            geometry: geometry,
            transport: transport,
            sources: sources,
            totalEnergies: totalEnergies
        )

        return DerivedQuantities(
            coreIonTemperature: centralValues.ionTemperature,
            coreElectronTemperature: centralValues.electronTemperature,
            coreElectronDensity: centralValues.electronDensity,
            averageElectronDensity: volumeAverages.electronDensity,
            averageIonTemperature: volumeAverages.ionTemperature,
            averageElectronTemperature: volumeAverages.electronTemperature,
            thermalEnergy: totalEnergies.thermal,
            ionThermalEnergy: totalEnergies.ion,
            electronThermalEnergy: totalEnergies.electron,
            fusionPower: advancedMetrics.fusionPower,
            alphaPower: advancedMetrics.alphaPower,
            auxiliaryPower: advancedMetrics.auxiliaryPower,
            ohmicPower: advancedMetrics.ohmicPower,
            fusionGain: advancedMetrics.fusionGain,
            energyConfinementTime: advancedMetrics.energyConfinementTime,
            scalingEnergyConfinementTime: advancedMetrics.scalingEnergyConfinementTime,
            confinementHFactor: advancedMetrics.confinementHFactor,
            toroidalBeta: advancedMetrics.toroidalBeta,
            poloidalBeta: advancedMetrics.poloidalBeta,
            normalizedBeta: advancedMetrics.normalizedBeta,
            normalizedBetaLimit: advancedMetrics.normalizedBetaLimit,
            plasmaCurrent: advancedMetrics.plasmaCurrent,
            bootstrapCurrent: advancedMetrics.bootstrapCurrent,
            bootstrapFraction: advancedMetrics.bootstrapFraction,
            tripleProduct: advancedMetrics.tripleProduct
        )
    }

    // MARK: - Central Values (ρ=0)

    private static func computeCentralValues(profiles: CoreProfiles) -> (ionTemperature: Float, electronTemperature: Float, electronDensity: Float) {
        // Extract central values (first cell, index 0)
        let Ti_array = profiles.ionTemperature.value
        let Te_array = profiles.electronTemperature.value
        let ne_array = profiles.electronDensity.value

        // GPU → CPU transfer (scalar only, cheap)
        let coreIonTemperature = Ti_array[0].item(Float.self)
        let coreElectronTemperature = Te_array[0].item(Float.self)
        let coreElectronDensity = ne_array[0].item(Float.self)

        return (coreIonTemperature, coreElectronTemperature, coreElectronDensity)
    }

    // MARK: - Volume Averages

    private static func computeVolumeAverages(
        profiles: CoreProfiles,
        geometry: Geometry
    ) -> (ionTemperature: Float, electronTemperature: Float, electronDensity: Float) {

        // Get cell volumes from geometry
        let geometricFactors = GeometricFactors.from(geometry: geometry)
        let volumes = geometricFactors.cellVolumes.value
        eval(volumes)

        let Ti = profiles.ionTemperature.value
        let Te = profiles.electronTemperature.value
        let ne = profiles.electronDensity.value

        // Volume-weighted averages: ⟨Q⟩ = ∫ Q dV / ∫ dV
        let totalVolume = volumes.sum()

        let Ti_weighted = (Ti * volumes).sum()
        let Te_weighted = (Te * volumes).sum()
        let ne_weighted = (ne * volumes).sum()

        // Batch evaluation for efficiency
        eval(totalVolume, Ti_weighted, Te_weighted, ne_weighted)

        let averageIonTemperature = (Ti_weighted / totalVolume).item(Float.self)
        let averageElectronTemperature = (Te_weighted / totalVolume).item(Float.self)
        let averageElectronDensity = (ne_weighted / totalVolume).item(Float.self)

        return (averageIonTemperature, averageElectronTemperature, averageElectronDensity)
    }

    // MARK: - Total Energies

    private static func computeTotalEnergies(
        profiles: CoreProfiles,
        geometry: Geometry
    ) -> (thermal: Float, ion: Float, electron: Float) {

        // Physical constants
        let eV_to_J: Float = 1.602176634e-19  // eV to Joule conversion

        // Get cell volumes
        let geometricFactors = GeometricFactors.from(geometry: geometry)
        let volumes = geometricFactors.cellVolumes.value
        eval(volumes)

        let Ti = profiles.ionTemperature.value  // [eV]
        let Te = profiles.electronTemperature.value  // [eV]
        let ne = profiles.electronDensity.value  // [m^-3]

        // Assume quasineutrality: n_i ≈ n_e
        // Thermal energy density: 3/2 * n * T [eV/m^3]
        let w_ion_density = 1.5 * ne * Ti  // [eV/m^3]
        let w_electron_density = 1.5 * ne * Te  // [eV/m^3]

        // Total energy: ∫ w dV → [eV/m^3] × [m^3] = [eV]
        let W_ion_eV = (w_ion_density * volumes).sum()
        let W_electron_eV = (w_electron_density * volumes).sum()
        let W_thermal_eV = W_ion_eV + W_electron_eV

        // Batch evaluation
        eval(W_ion_eV, W_electron_eV, W_thermal_eV)

        // Convert eV → J → MJ
        let J_to_MJ: Float = 1e-6

        let W_ion_MJ = W_ion_eV.item(Float.self) * eV_to_J * J_to_MJ
        let W_electron_MJ = W_electron_eV.item(Float.self) * eV_to_J * J_to_MJ
        let W_thermal_MJ = W_thermal_eV.item(Float.self) * eV_to_J * J_to_MJ

        return (W_thermal_MJ, W_ion_MJ, W_electron_MJ)
    }

    // MARK: - Advanced Metrics (Phase 3)

    /// Compute advanced metrics (τE, Q, βN, etc.)
    ///
    /// **Phase 3**: Full implementation using transport/sources
    ///
    /// - Parameters:
    ///   - profiles: Current plasma profiles
    ///   - geometry: Tokamak geometry
    ///   - transport: Transport coefficients (optional)
    ///   - sources: Source terms (optional)
    ///   - totalEnergies: Pre-computed total energies
    /// - Returns: Advanced metrics
    private static func computeAdvancedMetrics(
        profiles: CoreProfiles,
        geometry: Geometry,
        transport: TransportCoefficients?,
        sources: SourceTerms?,
        totalEnergies: (thermal: Float, ion: Float, electron: Float)
    ) -> AdvancedMetrics {

        let geometricFactors = GeometricFactors.from(geometry: geometry)
        let volumes = geometricFactors.cellVolumes.value

        // 1. Power balance
        let powers = computePowerBalance(
            sources: sources,
            profiles: profiles,
            geometry: geometry,
            volumes: volumes
        )

        // 2. Confinement time
        let confinement = computeConfinementMetrics(
            thermalEnergy: totalEnergies.thermal,
            powers: powers,
            profiles: profiles,
            geometry: geometry
        )

        // 3. Current metrics (MUST compute before beta to get plasmaCurrent)
        let current = computeCurrentMetrics(
            profiles: profiles,
            geometry: geometry,
            transport: transport
        )

        // 4. Beta limits (uses plasmaCurrent from current metrics)
        let beta = computeBetaMetrics(
            profiles: profiles,
            geometry: geometry,
            volumes: volumes,
            plasmaCurrent: current.plasmaCurrent
        )

        // 5. Triple product
        let tripleProduct = computeTripleProduct(
            profiles: profiles,
            geometry: geometry,
            volumes: volumes,
            energyConfinementTime: confinement.energyConfinementTime
        )

        // 6. Fusion gain Q = fusionPower / P_input
        let fusionGain = computeFusionGain(
            fusionPower: powers.fusionPower,
            auxiliaryPower: powers.auxiliaryPower,
            ohmicPower: powers.ohmicPower
        )

        return AdvancedMetrics(
            fusionPower: powers.fusionPower,
            alphaPower: powers.alphaPower,
            auxiliaryPower: powers.auxiliaryPower,
            ohmicPower: powers.ohmicPower,
            fusionGain: fusionGain,
            energyConfinementTime: confinement.energyConfinementTime,
            scalingEnergyConfinementTime: confinement.scalingEnergyConfinementTime,
            confinementHFactor: confinement.confinementHFactor,
            toroidalBeta: beta.toroidal,
            poloidalBeta: beta.poloidal,
            normalizedBeta: beta.normalized,
            normalizedBetaLimit: beta.troyon_limit,
            plasmaCurrent: current.plasmaCurrent,
            bootstrapCurrent: current.bootstrapCurrent,
            bootstrapFraction: current.bootstrapFraction,
            tripleProduct: tripleProduct
        )
    }

    // MARK: - Power Balance

    private static func computePowerBalance(
        sources: SourceTerms?,
        profiles: CoreProfiles,
        geometry: Geometry,
        volumes: MLXArray
    ) -> (fusionPower: Float, alphaPower: Float, auxiliaryPower: Float, ohmicPower: Float) {

        guard let sources = sources else {
            return (0, 0, 0, 0)
        }

        guard let metadata = sources.metadata else {
            preconditionFailure(
                """
                SourceTerms.metadata is required for accurate power balance computation.

                All SourceModel implementations must provide SourceMetadata.

                Fix: Update SourceModel to return SourceTerms with metadata:
                    let metadata = SourceMetadata(
                        modelName: "your_model",
                        category: .fusion/.auxiliary/.ohmic,
                        ionPower: computed_ion_power,
                        electronPower: computed_electron_power
                    )
                    return SourceTerms(..., metadata: SourceMetadataCollection(entries: [metadata]))
                """
            )
        }

        do {
            try metadata.validatePowerAccounting()
        } catch {
            preconditionFailure("Invalid SourceTerms.metadata for power balance computation: \(error)")
        }

        // Convert from W to MW
        let fusionPower = metadata.fusionPower / 1e6       // [W] → [MW]
        let alphaPower = metadata.alphaPower / 1e6         // [W] → [MW]
        let auxiliaryPower = metadata.auxiliaryPower / 1e6 // [W] → [MW]
        let ohmicPower = metadata.ohmicPower / 1e6         // [W] → [MW]

        return (fusionPower, alphaPower, auxiliaryPower, ohmicPower)
    }

    // MARK: - Confinement Metrics

    private static func computeConfinementMetrics(
        thermalEnergy: Float,
        powers: (fusionPower: Float, alphaPower: Float, auxiliaryPower: Float, ohmicPower: Float),
        profiles: CoreProfiles,
        geometry: Geometry
    ) -> (energyConfinementTime: Float, scalingEnergyConfinementTime: Float, confinementHFactor: Float) {

        // Energy confinement time: τE = W / P_loss
        // P_loss = P_input + alphaPower (external heating + alpha particle heating)
        // Note: fusionPower is NOT included in P_loss (it's already counted via alphaPower)
        let P_input = powers.auxiliaryPower + powers.ohmicPower
        let P_loss = P_input + powers.alphaPower  // Total heating power

        let energyConfinementTime: Float
        if P_loss > PhysicalThresholds.default.minimumHeatingPowerForEnergyConfinementTime {
            energyConfinementTime = thermalEnergy / P_loss  // [MJ / MW = s]
        } else {
            energyConfinementTime = 0
        }

        // ITER98y2 scaling law for H-mode
        // τE = 0.0562 * Ip^0.93 * Bt^0.15 * P^(-0.69) * n^0.41 * M^0.19 * R^1.97 * ε^0.58 * κ^0.78
        // Simplified version for circular geometry
        let scalingEnergyConfinementTime = computeITER98Scaling(
            profiles: profiles,
            geometry: geometry,
            P_loss: P_loss
        )

        let confinementHFactor: Float
        if scalingEnergyConfinementTime > 0 {
            confinementHFactor = energyConfinementTime / scalingEnergyConfinementTime
        } else {
            confinementHFactor = 0
        }

        return (energyConfinementTime, scalingEnergyConfinementTime, confinementHFactor)
    }

    private static func computeITER98Scaling(
        profiles: CoreProfiles,
        geometry: Geometry,
        P_loss: Float
    ) -> Float {

        // Extract parameters
        let R0 = geometry.majorRadius  // [m]
        let a = geometry.minorRadius   // [m]
        let Bt = geometry.toroidalField  // [T]

        // Estimate plasma current from geometry (very rough)
        let epsilon = a / R0
        let Ip_est: Float = 15.0  // [MA] - typical ITER-scale value

        // Volume-averaged density
        let geometricFactors = GeometricFactors.from(geometry: geometry)
        let volumes = geometricFactors.cellVolumes.value
        let ne = profiles.electronDensity.value
        let ne_weighted = (ne * volumes).sum()
        let total_volume = volumes.sum()
        eval(ne_weighted, total_volume)

        let averageElectronDensity = (ne_weighted / total_volume).item(Float.self)  // [m^-3]
        let ne_19 = averageElectronDensity / 1e19  // [10^19 m^-3]

        // Mass number (assume deuterium-tritium)
        let M: Float = 2.5

        // Elongation (assume circular for now)
        let kappa: Float = 1.0

        // ITER98y2 formula
        let tau_scaling = 0.0562 *
            pow(Ip_est, 0.93) *
            pow(Bt, 0.15) *
            pow(max(P_loss, 0.1), -0.69) *
            pow(ne_19, 0.41) *
            pow(M, 0.19) *
            pow(R0, 1.97) *
            pow(epsilon, 0.58) *
            pow(kappa, 0.78)

        return tau_scaling  // [s]
    }

    // MARK: - Beta Metrics

    private static func computeBetaMetrics(
        profiles: CoreProfiles,
        geometry: Geometry,
        volumes: MLXArray,
        plasmaCurrent: Float
    ) -> (toroidal: Float, poloidal: Float, normalized: Float, troyon_limit: Float) {

        let mu0: Float = 4.0 * .pi * 1e-7  // Permeability [H/m]
        let eV_to_J: Float = 1.602176634e-19

        // Volume-averaged pressure
        let Ti = profiles.ionTemperature.value  // [eV]
        let Te = profiles.electronTemperature.value  // [eV]
        let ne = profiles.electronDensity.value  // [m^-3]

        // Pressure: p = n_e * (T_i + T_e) [eV/m^3]
        let pressure_eV = ne * (Ti + Te)
        let pressure_weighted = (pressure_eV * volumes).sum()
        let total_volume = volumes.sum()
        eval(pressure_weighted, total_volume)

        let p_avg_eV = (pressure_weighted / total_volume).item(Float.self)
        let p_avg = p_avg_eV * eV_to_J  // [Pa]

        // Toroidal beta: βt = 2μ0⟨p⟩ / Bt^2
        let Bt = geometry.toroidalField
        let toroidalBeta = (2.0 * mu0 * p_avg) / (Bt * Bt) * 100.0  // [%]

        // Poloidal beta: rough estimate as βp ≈ 2 * βt for typical tokamaks
        let poloidalBeta = 2.0 * toroidalBeta

        // Normalized beta: βN = β(%) * a(m) * Bt(T) / Ip(MA)
        // Use minimum 0.1 MA for small tokamaks (avoids unrealistic βN for low-current plasmas)
        let Ip_MA = max(plasmaCurrent, 0.1)  // Avoid division by zero
        let normalizedBeta = toroidalBeta * geometry.minorRadius * Bt / Ip_MA

        // Troyon limit: βN_limit ≈ 2.8 (empirical)
        let normalizedBetaLimit: Float = 2.8

        return (toroidalBeta, poloidalBeta, normalizedBeta, normalizedBetaLimit)
    }

    // MARK: - Current Metrics

    private static func computeCurrentMetrics(
        profiles: CoreProfiles,
        geometry: Geometry,
        transport: TransportCoefficients?
    ) -> (plasmaCurrent: Float, bootstrapCurrent: Float, bootstrapFraction: Float) {

        // Compute plasma current from poloidal flux gradient
        let psi = profiles.poloidalFlux.value
        let geometricFactors = GeometricFactors.from(geometry: geometry)

        // Check if we have meaningful flux data
        let psiRange = MLX.max(psi).item(Float.self) - MLX.min(psi).item(Float.self)

        if psiRange > 0.01 {
            // Compute current density: j_∥ ≈ (1/μ₀R) * ∂ψ/∂r
            let cellCount = psi.shape[0]
            let df = psi[1...] - psi[..<(cellCount - 1)]
            let radialSpacing = geometricFactors.cellDistances.value + 1e-10
            let grad_psi_faces = df / radialSpacing

            // Interpolate gradient to cell centers
            let grad0 = grad_psi_faces[0..<1]
            let left = grad_psi_faces[0..<(cellCount - 2)]
            let right = grad_psi_faces[1..<(cellCount - 1)]
            let gradInterior = (left + right) / 2.0
            let gradN = grad_psi_faces[(cellCount - 2)..<(cellCount - 1)]
            let grad_psi = concatenated([grad0, gradInterior, gradN], axis: 0)

            let mu0: Float = 4.0 * .pi * 1e-7
            let R0 = geometry.majorRadius
            let j_parallel = grad_psi / (mu0 * R0)  // [A/m²]

            // Integrate over cross-section: I = ∫ j dA
            // For circular geometry, use cell volumes divided by 2πR
            // Volume = 2π²Rr²Δr → dA ≈ Volume / (2πR)
            let volumes = geometricFactors.cellVolumes.value  // [cellCount]

            // Current density × area element
            let I_elements = abs(j_parallel) * volumes / (2.0 * Float.pi * R0)  // [A·m]

            // Total current
            let I_total = I_elements.sum().item(Float.self)  // [A]
            let plasmaCurrent = I_total * 1e-6  // [MA]

            // Bootstrap fraction: typical range 0.2-0.5 for ITER-like plasmas
            let bootstrapFraction: Float = 0.3  // Rough estimate
            let bootstrapCurrent = plasmaCurrent * bootstrapFraction  // [MA]

            return (plasmaCurrent, bootstrapCurrent, bootstrapFraction)
        } else {
            // Fallback: Estimate from geometry when flux is not available
            let a = geometry.minorRadius
            let Bt = geometry.toroidalField
            let q_edge: Float = 3.0  // Typical edge safety factor
            let mu0: Float = 4.0 * .pi * 1e-7
            let R0 = geometry.majorRadius

            let plasmaCurrent = (a * Bt) / (q_edge * mu0 * R0) * 1e-6  // [MA]
            let bootstrapFraction: Float = 0.3
            let bootstrapCurrent = plasmaCurrent * bootstrapFraction

            return (plasmaCurrent, bootstrapCurrent, bootstrapFraction)
        }
    }

    // MARK: - Triple Product

    private static func computeTripleProduct(
        profiles: CoreProfiles,
        geometry: Geometry,
        volumes: MLXArray,
        energyConfinementTime: Float
    ) -> Float {

        // Lawson triple product: n⟨T⟩τE
        let Ti = profiles.ionTemperature.value  // [eV]
        let Te = profiles.electronTemperature.value  // [eV]
        let ne = profiles.electronDensity.value  // [m^-3]

        // Volume-averaged temperature
        let T_avg_eV = (((Ti + Te) * 0.5 * ne * volumes).sum() / ((ne * volumes).sum() + 1e-10))
        eval(T_avg_eV)
        let T_avg = T_avg_eV.item(Float.self)  // [eV]

        // Volume-averaged density
        let ne_weighted = (ne * volumes).sum()
        let total_volume = volumes.sum()
        eval(ne_weighted, total_volume)
        let averageElectronDensity = (ne_weighted / total_volume).item(Float.self)  // [m^-3]

        // Triple product: n⟨T⟩τE [eV s m^-3]
        let tripleProduct = averageElectronDensity * T_avg * energyConfinementTime

        return tripleProduct
    }

    // MARK: - Fusion Gain

    /// Compute fusion gain Q = fusionPower / P_input
    ///
    /// **Definition**: Q = fusionPower / (auxiliaryPower + ohmicPower)
    ///
    /// **Physics**:
    /// - Q < 1: More input power than fusion power (typical for small devices)
    /// - Q = 1: Breakeven (fusion power equals input power)
    /// - Q = 5-10: High-performance operation (ITER target: Q = 10)
    /// - Q → ∞: Ignition (self-sustaining fusion, no external heating needed)
    ///
    /// **Note**: Alpha power (alphaPower) is NOT counted as input since it's internally
    /// generated. Only external heating sources count toward P_input.
    ///
    /// - Parameters:
    ///   - fusionPower: Total fusion power [MW]
    ///   - auxiliaryPower: Auxiliary heating power [MW]
    ///   - ohmicPower: Ohmic heating power [MW]
    /// - Returns: Fusion gain Q (dimensionless)
    private static func computeFusionGain(
        fusionPower: Float,
        auxiliaryPower: Float,
        ohmicPower: Float
    ) -> Float {
        // Input power = external heating only (exclude alpha power)
        let P_input = auxiliaryPower + ohmicPower

        // Handle edge cases
        guard P_input > PhysicalThresholds.default.minimumFusionPowerForGain else {
            // No input power: return 0 (avoid division by zero)
            return 0
        }

        // Fusion gain
        let Q = fusionPower / P_input

        // Clamp to reasonable range [0, 100]
        // Q > 100 is unrealistic and likely indicates numerical issues
        return max(0, min(Q, 100))
    }
}

// MARK: - Internal Data Structures

private struct AdvancedMetrics {
    let fusionPower: Float
    let alphaPower: Float
    let auxiliaryPower: Float
    let ohmicPower: Float
    let fusionGain: Float
    let energyConfinementTime: Float
    let scalingEnergyConfinementTime: Float
    let confinementHFactor: Float
    let toroidalBeta: Float
    let poloidalBeta: Float
    let normalizedBeta: Float
    let normalizedBetaLimit: Float
    let plasmaCurrent: Float
    let bootstrapCurrent: Float
    let bootstrapFraction: Float
    let tripleProduct: Float
}
