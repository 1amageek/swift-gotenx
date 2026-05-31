import Foundation
import MLX
import GotenxCore

/// Ion-electron collisional heat exchange model
///
/// Computes power density transferred from electrons to ions (or vice versa)
/// through Coulomb collisions.
///
/// Physical equation:
/// Q_ie = (3/2) * (m_e/m_i) * n_e * ν_ei * (T_e - T_i)
///
/// Where collision frequency:
/// ν_ei = 2.91 × 10⁻⁶ * n_e * Z_eff * ln(Λ) / T_e^(3/2)
///
/// Coulomb logarithm:
/// ln(Λ) = 24 - ln(√(n_e[m⁻³]/10⁶) / T_e[eV])
///
/// Units:
/// - Input: n_e [m⁻³], T_e [eV], T_i [eV]
/// - Output: Q_ie [W/m³] (positive = heating ions)
public struct IonElectronExchange: Sendable {

    /// Effective charge number
    public let Zeff: Float

    /// Ion mass in atomic mass units
    public let ionMass: Float

    /// Physical constants
    private let kB: Float = PhysicsConstants.eV           // eV to Joules
    private let me: Float = PhysicsConstants.electronMass  // electron mass [kg]
    private let mp: Float = PhysicsConstants.protonMass   // proton mass [kg]

    /// Create ion-electron exchange model
    ///
    /// - Parameters:
    ///   - Zeff: Effective charge number (default: 1.5)
    ///   - ionMass: Ion mass in amu (default: 2.014 for deuterium)
    public init(Zeff: Float = 1.5, ionMass: Float = 2.014) {
        self.Zeff = Zeff
        self.ionMass = ionMass
    }

    /// Compute ion-electron heat exchange power density
    ///
    /// - Parameters:
    ///   - ne: Electron density [m⁻³], shape [nCells]
    ///   - Te: Electron temperature [eV], shape [nCells]
    ///   - Ti: Ion temperature [eV], shape [nCells]
    /// - Returns: Heat exchange power [W/m³], shape [nCells]
    ///            Positive = heating ions, Negative = heating electrons
    /// - Throws: PhysicsError if inputs are invalid
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func compute(
        ne: MLXArray,
        Te: MLXArray,
        Ti: MLXArray
    ) throws -> MLXArray {

        // Validate inputs (CRITICAL FIX #3)
        try PhysicsValidation.validateDensity(ne, name: "ne")
        try PhysicsValidation.validateTemperature(Te, name: "Te")
        try PhysicsValidation.validateTemperature(Ti, name: "Ti")
        try PhysicsValidation.validateShapes([ne, Te, Ti], names: ["ne", "Te", "Ti"])

        // Coulomb logarithm with bounds (MEDIUM FIX #1)
        // ln(Λ) = 24 - ln(√(n_e/10⁶) / T_e)
        let lnLambda_raw = Float(24.0) - log(sqrt(ne / Float(1e6)) / Te)
        let lnLambda = PhysicsValidation.clampCoulombLog(lnLambda_raw)

        // Electron-ion collision frequency [Hz]
        // ν_ei = 2.91 × 10⁻⁶ * n_e * Z_eff * ln(Λ) / T_e^(3/2)
        let nu_ei = PhysicsConstants.collisionFrequencyPrefactor * ne * Zeff * lnLambda / pow(Te, Float(1.5))

        // Ion mass [kg]
        let mi = PhysicsConstants.amuToKg(ionMass)

        // Exchange power density [W/m³]
        // Q_ie = (3/2) * (m_e/m_i) * n_e * ν_ei * k_B * (T_e - T_i)
        let Q_ie_watts = (Float(3.0)/Float(2.0)) * (me/mi) * ne * nu_ei * kB * (Te - Ti)

        // Return lazy MLXArray - caller will eval() when needed
        return Q_ie_watts
    }

    /// Compute Coulomb logarithm
    ///
    /// - Parameters:
    ///   - ne: Electron density [m⁻³]
    ///   - Te: Electron temperature [eV]
    /// - Returns: Coulomb logarithm (dimensionless)
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func computeCoulombLogarithm(ne: MLXArray, Te: MLXArray) -> MLXArray {
        return Float(24.0) - log(sqrt(ne / Float(1e6)) / Te)
    }

    /// Compute electron-ion collision frequency
    ///
    /// - Parameters:
    ///   - ne: Electron density [m⁻³]
    ///   - Te: Electron temperature [eV]
    /// - Returns: Collision frequency [Hz]
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func computeCollisionFrequency(ne: MLXArray, Te: MLXArray) -> MLXArray {
        let lnLambda = computeCoulombLogarithm(ne: ne, Te: Te)
        return PhysicsConstants.collisionFrequencyPrefactor * ne * Zeff * lnLambda / pow(Te, Float(1.5))
    }

    /// Compute source metadata for power balance tracking
    ///
    /// Ion-electron exchange is energy conservative: power transferred to ions
    /// equals power removed from electrons (and vice versa).
    ///
    /// - Parameters:
    ///   - profiles: Current plasma profiles
    ///   - geometry: Geometry for volume integration
    /// - Returns: Source metadata with ion-electron exchange power
    /// - Throws: PhysicsError if computation fails
    public func computeMetadata(
        profiles: CoreProfiles,
        geometry: Geometry
    ) throws -> SourceMetadata {

        let Q_ie_watts = try compute(
            ne: profiles.electronDensity.value,
            Te: profiles.electronTemperature.value,
            Ti: profiles.ionTemperature.value
        )

        // Volume integration: ∫ Q dV → [W/m³] × [m³] = [W]
        let cellVolumes = GeometricFactors.from(geometry: geometry).cellVolumes.value
        let P_ie_total = (Q_ie_watts * cellVolumes).sum()
        eval(P_ie_total)

        let exchangePower = P_ie_total.item(Float.self)

        // Positive = heating ions, negative = heating electrons
        return SourceMetadata(
            modelName: "ion_electron_exchange",
            category: .other,  // Energy transfer, not a source
            ionPower: exchangePower,
            electronPower: -exchangePower  // Energy conserved
        )
    }
}

// MARK: - Source Model Protocol Conformance

extension IonElectronExchange {

    /// Apply heat exchange to source terms
    ///
    /// Updates ion and electron heating sources with collisional energy exchange.
    /// Energy is conserved: Q_ion = -Q_electron
    ///
    /// - Parameters:
    ///   - sources: Source terms to modify
    ///   - profiles: Current plasma profiles
    ///   - geometry: Tokamak geometry for metadata computation
    /// - Returns: Modified source terms with heat exchange
    public func applyToSources(
        _ sources: SourceTerms,
        profiles: CoreProfiles,
        geometry: Geometry
    ) throws -> SourceTerms {

        // Phase 1b: Input validation (Sprint 1 robustness)
        // CRITICAL: Prevents NaN propagation that caused production crash
        guard let validatedProfiles = ValidatedProfiles.validateMinimal(profiles) else {
            print("[ION-ELECTRON-EXCHANGE] Invalid input profiles, skipping heat exchange")
            // Fail-safe: Return current sources unchanged (preserve simulation)
            return sources
        }

        let Q_ie_watts = try compute(
            ne: validatedProfiles.electronDensity.value,
            Te: validatedProfiles.electronTemperature.value,
            Ti: validatedProfiles.ionTemperature.value
        )

        // Convert to MW/m³ for SourceTerms
        let Q_ie = PhysicsConstants.wattsToMegawatts(Q_ie_watts)

        // Phase 1b: Output validation (Sprint 1 robustness)
        // Detect NaN/Inf in computed exchange power before adding to sources
        let Q_ie_min = Q_ie.min().item(Float.self)
        let Q_ie_max = Q_ie.max().item(Float.self)
        guard !Q_ie_min.isNaN && !Q_ie_min.isInfinite &&
              !Q_ie_max.isNaN && !Q_ie_max.isInfinite else {
            print("[ION-ELECTRON-EXCHANGE] Output contains NaN/Inf, skipping heat exchange")
            print("  Q_ie range: [\(Q_ie_min), \(Q_ie_max)] MW/m³")
            // Fail-safe: Return current sources unchanged
            return sources
        }

        // Compute metadata for power balance tracking
        // Reuse Q_ie_watts to avoid duplicate computation
        let cellVolumes = GeometricFactors.from(geometry: geometry).cellVolumes.value
        let P_ie_total = (Q_ie_watts * cellVolumes).sum()
        eval(P_ie_total)
        let exchangePower = P_ie_total.item(Float.self)

        let exchangeMetadata = SourceMetadata(
            modelName: "ion_electron_exchange",
            category: .other,
            ionPower: exchangePower,
            electronPower: -exchangePower
        )

        // Merge with existing metadata
        let mergedMetadata: SourceMetadataCollection
        if let existingMetadata = sources.metadata {
            mergedMetadata = SourceMetadataCollection(
                entries: existingMetadata.entries + [exchangeMetadata]
            )
        } else {
            mergedMetadata = SourceMetadataCollection(entries: [exchangeMetadata])
        }

        // Create new SourceTerms with updated heating and metadata
        return SourceTerms(
            ionHeating: EvaluatedArray(
                evaluating: sources.ionHeating.value + Q_ie
            ),
            electronHeating: EvaluatedArray(
                evaluating: sources.electronHeating.value - Q_ie
            ),
            particleSource: sources.particleSource,
            currentSource: sources.currentSource,
            metadata: mergedMetadata
        )
    }

    public func applyToSourcesForSolver(
        _ sources: SourceTerms,
        profiles: CoreProfiles
    ) throws -> SourceTerms {
        let qIEWatts = computeForSolver(
            ne: profiles.electronDensity.value,
            Te: profiles.electronTemperature.value,
            Ti: profiles.ionTemperature.value
        )
        let qIE = PhysicsConstants.wattsToMegawatts(qIEWatts)

        return SourceTerms(
            ionHeating: EvaluatedArray(
                evaluating: sources.ionHeating.value + qIE
            ),
            electronHeating: EvaluatedArray(
                evaluating: sources.electronHeating.value - qIE
            ),
            particleSource: sources.particleSource,
            currentSource: sources.currentSource,
            metadata: sources.metadata,
            validateDebugUnits: false
        )
    }

    private func computeForSolver(
        ne: MLXArray,
        Te: MLXArray,
        Ti: MLXArray
    ) -> MLXArray {
        let lnLambdaRaw = Float(24.0) - log(sqrt(ne / Float(1e6)) / Te)
        let lnLambda = PhysicsValidation.clampCoulombLog(lnLambdaRaw)
        let nuEI = PhysicsConstants.collisionFrequencyPrefactor * ne * Zeff * lnLambda / pow(Te, Float(1.5))
        let mi = PhysicsConstants.amuToKg(ionMass)

        return (Float(3.0) / Float(2.0)) * (me / mi) * ne * nuEI * kB * (Te - Ti)
    }
}
