import Foundation
import MLX
import GotenxCore

/// Ohmic heating model
///
/// Computes resistive heating power from plasma current:
/// Q_ohm = η_∥ * j_∥²
///
/// Uses Spitzer resistivity with optional neoclassical correction
/// for trapped particles.
///
/// Spitzer resistivity:
/// η_Spitzer = 5.2 × 10⁻⁵ * effectiveCharge * ln(Λ) / T_e^(3/2)  [Ω·m]
///
/// Neoclassical correction:
/// η_neo = η_Spitzer * (1 + ε^(3/2))
/// where ε = r/R₀ (inverse aspect ratio)
public struct OhmicHeating: Sendable {

    /// Effective charge
    public let effectiveCharge: Float

    /// Coulomb logarithm
    public let coulombLogarithm: Float

    /// Apply neoclassical correction for trapped particles
    public let useNeoclassical: Bool

    /// Physical thresholds for validation
    public let thresholds: PhysicalThresholds

    /// Create Ohmic heating model
    ///
    /// - Parameters:
    ///   - effectiveCharge: Effective charge (default: 1.5)
    ///   - coulombLogarithm: Coulomb logarithm (default: 17.0)
    ///   - useNeoclassical: Apply neoclassical correction (default: true)
    ///   - thresholds: Physical thresholds (default: .default)
    public init(
        effectiveCharge: Float = 1.5,
        coulombLogarithm: Float = 17.0,
        useNeoclassical: Bool = true,
        thresholds: PhysicalThresholds = .default
    ) {
        self.effectiveCharge = effectiveCharge
        self.coulombLogarithm = coulombLogarithm
        self.useNeoclassical = useNeoclassical
        self.thresholds = thresholds
    }

    /// Compute Ohmic heating power density
    ///
    /// - Parameters:
    ///   - electronTemperature: Electron temperature [eV], shape [cellCount]
    ///   - jParallel: Parallel current density [A/m²], shape [cellCount]
    ///   - geometry: Tokamak geometry
    /// - Returns: Heating power [W/m³], shape [cellCount]
    /// - Throws: PhysicsError if inputs are invalid
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func compute(
        electronTemperature: MLXArray,
        jParallel: MLXArray,
        geometry: Geometry
    ) throws -> MLXArray {

        // Validate inputs (CRITICAL FIX #3)
        try PhysicsValidation.validateTemperature(electronTemperature, name: "electronTemperature")
        try PhysicsValidation.validateFinite(jParallel, name: "jParallel")
        try PhysicsValidation.validateShapes([electronTemperature, jParallel], names: ["electronTemperature", "jParallel"])

        // Spitzer resistivity [Ω·m]
        // η_Spitzer = 5.2 × 10⁻⁵ * effectiveCharge * ln(Λ) / T_e^(3/2)
        let eta_Spitzer = PhysicsConstants.spitzerPrefactor * effectiveCharge * coulombLogarithm / pow(electronTemperature, 1.5)

        var eta = eta_Spitzer

        if useNeoclassical {
            // Neoclassical correction for trapped particles
            // Inverse aspect ratio: ε = r/R₀
            let geomFactors = GeometricFactors.from(geometry: geometry)
            let epsilon = geomFactors.cellRadii.value / geometry.majorRadius

            // Trapped particle correction factor: f_trap ≈ 1 + ε^(3/2)
            let f_trap = 1.0 + pow(epsilon, 1.5)
            eta = eta * f_trap
        }

        // Ohmic power [W/m³]
        // Q_ohm = η * j_∥²
        let Q_ohm_watts = eta * jParallel * jParallel

        // Return lazy MLXArray - caller will eval() when needed
        return Q_ohm_watts
    }

    /// Compute Spitzer resistivity (without neoclassical correction)
    ///
    /// - Parameters:
    ///   - electronTemperature: Electron temperature [eV]
    ///   - effectiveCharge: Effective charge (optional override)
    ///   - coulombLogarithm: Coulomb logarithm (optional override)
    /// - Returns: Resistivity [Ω·m]
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func computeSpitzerResistivity(
        electronTemperature: MLXArray,
        effectiveCharge: Float? = nil,
        coulombLogarithm: Float? = nil
    ) -> MLXArray {
        let Z = effectiveCharge ?? self.effectiveCharge
        let ln = coulombLogarithm ?? self.coulombLogarithm

        return PhysicsConstants.spitzerPrefactor * Z * ln / pow(electronTemperature, 1.5)
    }

    /// Compute neoclassical resistivity
    ///
    /// - Parameters:
    ///   - electronTemperature: Electron temperature [eV]
    ///   - geometry: Tokamak geometry
    /// - Returns: Resistivity [Ω·m]
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func computeNeoclassicalResistivity(
        electronTemperature: MLXArray,
        geometry: Geometry
    ) -> MLXArray {
        let eta_Spitzer = computeSpitzerResistivity(electronTemperature: electronTemperature)

        // Trapped particle correction
        let geomFactors = GeometricFactors.from(geometry: geometry)
        let epsilon = geomFactors.cellRadii.value / geometry.majorRadius
        let f_trap = 1.0 + pow(epsilon, 1.5)

        return eta_Spitzer * f_trap
    }

    /// Compute source metadata for power balance tracking
    ///
    /// - Parameters:
    ///   - profiles: Current plasma profiles
    ///   - geometry: Geometry for volume integration
    ///   - plasmaCurrentDensity: Optional externally provided current density [A/m²]
    /// - Returns: Source metadata with ohmic power
    /// - Throws: PhysicsError if computation fails
    public func computeMetadata(
        profiles: CoreProfiles,
        geometry: Geometry,
        plasmaCurrentDensity: MLXArray? = nil
    ) throws -> SourceMetadata {

        // Compute parallel current density
        let jParallel: MLXArray
        if let providedCurrent = plasmaCurrentDensity {
            jParallel = providedCurrent
        } else {
            jParallel = try computeParallelCurrentFromProfiles(
                profiles: profiles,
                geometry: geometry
            )
        }

        let Q_ohm_watts = try compute(
            electronTemperature: profiles.electronTemperature.value,
            jParallel: jParallel,
            geometry: geometry
        )

        // Volume integration: ∫ Q dV → [W/m³] × [m³] = [W]
        let cellVolumes = GeometricFactors.from(geometry: geometry).cellVolumes.value
        let P_ohmic_total = (Q_ohm_watts * cellVolumes).sum()
        eval(P_ohmic_total)

        let ohmicPower = P_ohmic_total.item(Float.self)

        return SourceMetadata(
            modelName: "ohmic_heating",
            category: .ohmic,
            ionPower: 0,  // All Ohmic power goes to electrons
            electronPower: ohmicPower
        )
    }
}

// MARK: - Source Model Protocol Conformance

extension OhmicHeating {

    /// Apply Ohmic heating to source terms
    ///
    /// All Ohmic power goes to electron heating (electrons carry the current).
    ///
    /// - Parameters:
    ///   - sources: Source terms to modify
    ///   - profiles: Current plasma profiles
    ///   - geometry: Tokamak geometry
    /// - Returns: Modified source terms with Ohmic heating
    /// Apply Ohmic heating to source terms
    ///
    /// CRITICAL FIX #1: Improved implementation with current density computation
    ///
    /// Computes parallel current from:
    /// 1. Bootstrap current (from profiles)
    /// 2. Ohmic current (from resistive diffusion)
    /// 3. External current drive
    ///
    /// - Parameters:
    ///   - sources: Source terms to modify
    ///   - profiles: Current plasma profiles
    ///   - geometry: Tokamak geometry
    ///   - plasmaCurrentDensity: Optional externally provided current density [A/m²]
    ///                           If nil, estimates from profiles
    /// - Returns: Modified source terms with Ohmic heating
    /// - Throws: PhysicsError if computation fails
    public func applyToSources(
        _ sources: SourceTerms,
        profiles: CoreProfiles,
        geometry: Geometry,
        plasmaCurrentDensity: MLXArray? = nil
    ) throws -> SourceTerms {

        // Compute parallel current density
        let jParallel: MLXArray
        if let providedCurrent = plasmaCurrentDensity {
            jParallel = providedCurrent
        } else {
            // Estimate from poloidal flux if available
            jParallel = try computeParallelCurrentFromProfiles(
                profiles: profiles,
                geometry: geometry
            )
        }

        let Q_ohm_watts = try compute(
            electronTemperature: profiles.electronTemperature.value,
            jParallel: jParallel,
            geometry: geometry
        )

        // Convert to MW/m³ for SourceTerms
        let Q_ohm = PhysicsConstants.wattsToMegawatts(Q_ohm_watts)

        // Compute metadata for power balance tracking
        // Reuse Q_ohm_watts to avoid duplicate computation
        let cellVolumes = GeometricFactors.from(geometry: geometry).cellVolumes.value
        let P_ohmic_total = (Q_ohm_watts * cellVolumes).sum()
        eval(P_ohmic_total)
        let ohmicPower = P_ohmic_total.item(Float.self)

        let ohmicMetadata = SourceMetadata(
            modelName: "ohmic_heating",
            category: .ohmic,
            ionPower: 0,
            electronPower: ohmicPower
        )

        // Merge with existing metadata
        let mergedMetadata: SourceMetadataCollection
        if let existingMetadata = sources.metadata {
            mergedMetadata = SourceMetadataCollection(
                entries: existingMetadata.entries + [ohmicMetadata]
            )
        } else {
            mergedMetadata = SourceMetadataCollection(entries: [ohmicMetadata])
        }

        // Create new SourceTerms with updated electron heating and metadata
        return SourceTerms(
            ionHeating: sources.ionHeating,
            electronHeating: EvaluatedArray(
                evaluating: sources.electronHeating.value + Q_ohm
            ),
            particleSource: sources.particleSource,
            currentSource: sources.currentSource,
            metadata: mergedMetadata
        )
    }

    public func applyToSourcesForSolver(
        _ sources: SourceTerms,
        profiles: CoreProfiles,
        geometry: Geometry,
        plasmaCurrentDensity: MLXArray? = nil
    ) throws -> SourceTerms {
        let jParallel: MLXArray
        if let providedCurrent = plasmaCurrentDensity {
            jParallel = providedCurrent
        } else {
            jParallel = computeParallelCurrentFromProfilesForSolver(
                profiles: profiles,
                geometry: geometry
            )
        }

        let qOhmWatts = computeForSolver(
            electronTemperature: profiles.electronTemperature.value,
            jParallel: jParallel,
            geometry: geometry
        )
        let qOhm = PhysicsConstants.wattsToMegawatts(qOhmWatts)

        return SourceTerms(
            ionHeating: sources.ionHeating,
            electronHeating: EvaluatedArray(
                evaluating: sources.electronHeating.value + qOhm
            ),
            particleSource: sources.particleSource,
            currentSource: sources.currentSource,
            metadata: sources.metadata,
            validateDebugUnits: false
        )
    }

    /// Compute parallel current density from plasma profiles (CRITICAL FIX #1)
    ///
    /// Implements simplified current density model:
    /// j_∥ ≈ (1/μ₀R) * ∂ψ/∂r
    ///
    /// where ψ is poloidal flux, R is major radius
    ///
    /// **Note**: This is a simplified implementation suitable for:
    /// - Circular cross-section tokamaks
    /// - Moderate aspect ratio (R/a > 2)
    ///
    /// For shaped plasmas, need full MHD equilibrium solver.
    ///
    /// - Parameters:
    ///   - profiles: Current plasma profiles
    ///   - geometry: Tokamak geometry
    /// - Returns: Parallel current density [A/m²]
    /// - Throws: PhysicsError if computation fails
    private func computeParallelCurrentFromProfiles(
        profiles: CoreProfiles,
        geometry: Geometry
    ) throws -> MLXArray {

        let psi = profiles.poloidalFlux.value
        let cellCount = psi.shape[0]

        // Check if we have meaningful flux data
        let psiRange = MLX.max(psi).item(Float.self) - MLX.min(psi).item(Float.self)
        let psiMax = MLX.max(abs(psi)).item(Float.self)

        // Use relative threshold: dψ/ψ_max < threshold
        let relativeVariation = psiRange / max(psiMax, 1e-10)

        guard relativeVariation > thresholds.fluxVariationThreshold else {
            // Poloidal flux variation is negligible → no meaningful current
            // This happens in startup or when psi solver hasn't run yet
            return MLXArray.zeros([cellCount])
        }

        let geomFactors = GeometricFactors.from(geometry: geometry)
        let cellRadii = geomFactors.cellRadii.value

        // Compute radial derivative of psi using central differences
        // ∂ψ/∂r ≈ (ψ[i+1] - ψ[i-1]) / (r[i+1] - r[i-1])

        guard cellCount >= 3 else {
            // Not enough points for gradient
            return MLXArray.zeros([cellCount])
        }

        // Interior points: central difference
        let dr_interior = cellRadii[2..<cellCount] - cellRadii[0..<(cellCount-2)]
        let dpsi_interior = psi[2..<cellCount] - psi[0..<(cellCount-2)]
        let grad_psi_interior = dpsi_interior / (dr_interior + 1e-10)

        // Boundaries: forward/backward difference
        let dr_left = cellRadii[1] - cellRadii[0]
        let dpsi_left = psi[1] - psi[0]
        let grad_psi_left = dpsi_left / (dr_left + 1e-10)

        let dr_right = cellRadii[cellCount-1] - cellRadii[cellCount-2]
        let dpsi_right = psi[cellCount-1] - psi[cellCount-2]
        let grad_psi_right = dpsi_right / (dr_right + 1e-10)

        // Concatenate
        let grad_psi = concatenated([
            grad_psi_left.reshaped([1]),
            grad_psi_interior,
            grad_psi_right.reshaped([1])
        ], axis: 0)

        // Parallel current density: j_∥ ≈ (1/μ₀R) * ∂ψ/∂r
        let mu0 = PhysicsConstants.mu0
        let R0 = geometry.majorRadius
        let j_parallel = grad_psi / (mu0 * R0)

        return j_parallel
    }

    private func computeForSolver(
        electronTemperature: MLXArray,
        jParallel: MLXArray,
        geometry: Geometry
    ) -> MLXArray {
        let etaSpitzer = PhysicsConstants.spitzerPrefactor * effectiveCharge * coulombLogarithm / pow(electronTemperature, 1.5)

        let eta: MLXArray
        if useNeoclassical {
            let geomFactors = GeometricFactors.from(geometry: geometry)
            let epsilon = geomFactors.cellRadii.value / geometry.majorRadius
            eta = etaSpitzer * (1.0 + pow(epsilon, 1.5))
        } else {
            eta = etaSpitzer
        }

        return eta * jParallel * jParallel
    }

    private func computeParallelCurrentFromProfilesForSolver(
        profiles: CoreProfiles,
        geometry: Geometry
    ) -> MLXArray {
        let psi = profiles.poloidalFlux.value
        let cellCount = psi.shape[0]

        guard cellCount >= 3 else {
            return MLXArray.zeros([cellCount])
        }

        let geomFactors = GeometricFactors.from(geometry: geometry)
        let cellRadii = geomFactors.cellRadii.value

        let drInterior = cellRadii[2..<cellCount] - cellRadii[0..<(cellCount - 2)]
        let dpsiInterior = psi[2..<cellCount] - psi[0..<(cellCount - 2)]
        let gradPsiInterior = dpsiInterior / (drInterior + 1e-10)

        let drLeft = cellRadii[1] - cellRadii[0]
        let dpsiLeft = psi[1] - psi[0]
        let gradPsiLeft = dpsiLeft / (drLeft + 1e-10)

        let drRight = cellRadii[cellCount - 1] - cellRadii[cellCount - 2]
        let dpsiRight = psi[cellCount - 1] - psi[cellCount - 2]
        let gradPsiRight = dpsiRight / (drRight + 1e-10)

        let gradPsi = concatenated([
            gradPsiLeft.reshaped([1]),
            gradPsiInterior,
            gradPsiRight.reshaped([1])
        ], axis: 0)

        return gradPsi / (PhysicsConstants.mu0 * geometry.majorRadius)
    }
}
