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
/// ν_ei = 2.91 × 10⁻⁶ * n_e * effectiveCharge * ln(Λ) / T_e^(3/2)
///
/// Coulomb logarithm:
/// ln(Λ) = 24 - ln(√(n_e[m⁻³]/10⁶) / T_e[eV])
///
/// Units:
/// - Input: n_e [m⁻³], T_e [eV], T_i [eV]
/// - Output: Q_ie [W/m³] (positive = heating ions)
public struct IonElectronExchange: Sendable {

    /// Effective charge number
    public let effectiveCharge: Float

    /// Ion mass in atomic mass units
    public let ionMass: Float

    /// Physical constants
    private let kB: Float = PhysicsConstants.electronVolt           // eV to Joules
    private let me: Float = PhysicsConstants.electronMass  // electron mass [kg]
    private let mp: Float = PhysicsConstants.protonMass   // proton mass [kg]

    /// Create ion-electron exchange model
    ///
    /// - Parameters:
    ///   - effectiveCharge: Effective charge number (default: 1.5)
    ///   - ionMass: Ion mass in atomicMassUnits (default: 2.014 for deuterium)
    public init(effectiveCharge: Float = 1.5, ionMass: Float = 2.014) {
        self.effectiveCharge = effectiveCharge
        self.ionMass = ionMass
    }

    /// Compute ion-electron heat exchange power density
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m⁻³], shape [cellCount]
    ///   - electronTemperature: Electron temperature [eV], shape [cellCount]
    ///   - ionTemperature: Ion temperature [eV], shape [cellCount]
    /// - Returns: Heat exchange power [W/m³], shape [cellCount]
    ///            Positive = heating ions, Negative = heating electrons
    /// - Throws: PhysicsError if inputs are invalid
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func compute(
        electronDensity: MLXArray,
        electronTemperature: MLXArray,
        ionTemperature: MLXArray
    ) throws -> MLXArray {
        try compute(
            electronDensity: electronDensity,
            electronTemperature: electronTemperature,
            ionTemperature: ionTemperature,
            validatesInputs: true
        )
    }

    package func compute(
        electronDensity: MLXArray,
        electronTemperature: MLXArray,
        ionTemperature: MLXArray,
        validatesInputs: Bool
    ) throws -> MLXArray {
        if validatesInputs {
            try PhysicsValidation.validateDensity(electronDensity, name: "electronDensity")
            try PhysicsValidation.validateTemperature(electronTemperature, name: "electronTemperature")
            try PhysicsValidation.validateTemperature(ionTemperature, name: "ionTemperature")
            try PhysicsValidation.validateShapes([electronDensity, electronTemperature, ionTemperature], names: ["electronDensity", "electronTemperature", "ionTemperature"])
        }

        // Coulomb logarithm with bounds (MEDIUM FIX #1)
        // ln(Λ) = 24 - ln(√(n_e/10⁶) / T_e)
        let lnLambda_raw = Float(24.0) - log(sqrt(electronDensity / Float(1e6)) / electronTemperature)
        let coulombLogarithm = PhysicsValidation.clampCoulombLog(lnLambda_raw)

        // Electron-ion collision frequency [Hz]
        // ν_ei = 2.91 × 10⁻⁶ * n_e * effectiveCharge * ln(Λ) / T_e^(3/2)
        let nu_ei = PhysicsConstants.collisionFrequencyPrefactor * electronDensity * effectiveCharge * coulombLogarithm / pow(electronTemperature, Float(1.5))

        // Ion mass [kg]
        let mi = PhysicsConstants.atomicMassUnitsToKilograms(ionMass)

        // Exchange power density [W/m³]
        // Q_ie = (3/2) * (m_e/m_i) * n_e * ν_ei * k_B * (T_e - T_i)
        let Q_ie_watts = (Float(3.0)/Float(2.0)) * (me/mi) * electronDensity * nu_ei * kB * (electronTemperature - ionTemperature)

        // Return lazy MLXArray - caller will eval() when needed
        return Q_ie_watts
    }

    /// Compute Coulomb logarithm
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m⁻³]
    ///   - electronTemperature: Electron temperature [eV]
    /// - Returns: Coulomb logarithm (dimensionless)
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func computeCoulombLogarithm(electronDensity: MLXArray, electronTemperature: MLXArray) -> MLXArray {
        return Float(24.0) - log(sqrt(electronDensity / Float(1e6)) / electronTemperature)
    }

    /// Compute electron-ion collision frequency
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m⁻³]
    ///   - electronTemperature: Electron temperature [eV]
    /// - Returns: Collision frequency [Hz]
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func computeCollisionFrequency(electronDensity: MLXArray, electronTemperature: MLXArray) -> MLXArray {
        let coulombLogarithm = computeCoulombLogarithm(electronDensity: electronDensity, electronTemperature: electronTemperature)
        return PhysicsConstants.collisionFrequencyPrefactor * electronDensity * effectiveCharge * coulombLogarithm / pow(electronTemperature, Float(1.5))
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
            electronDensity: profiles.electronDensity.value,
            electronTemperature: profiles.electronTemperature.value,
            ionTemperature: profiles.ionTemperature.value
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
    private static func metadataCollection(
        existing: SourceMetadataCollection?,
        appending metadata: SourceMetadata
    ) -> SourceMetadataCollection {
        if let existing {
            SourceMetadataCollection(entries: existing.entries + [metadata])
        } else {
            SourceMetadataCollection(entries: [metadata])
        }
    }

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
        try applyToSources(
            sources,
            profiles: profiles,
            geometricFactors: GeometricFactors.from(geometry: geometry),
            evaluationMode: .eager,
            includesMetadata: true,
            validateDebugUnits: true,
            validatesInputs: true
        )
    }

    package func applyToSources(
        _ sources: SourceTerms,
        profiles: CoreProfiles,
        context: SourceEvaluationContext
    ) throws -> SourceTerms {
        try applyToSources(
            sources,
            profiles: profiles,
            geometricFactors: context.geometricFactors,
            evaluationMode: context.evaluationMode,
            includesMetadata: context.includesMetadata,
            validateDebugUnits: context.validatesDebugUnits,
            validatesInputs: context.includesMetadata
        )
    }

    package func applyToSources(
        _ sources: SourceTerms,
        profiles: CoreProfiles,
        geometricFactors: GeometricFactors,
        evaluationMode: MLXEvaluationMode,
        includesMetadata: Bool,
        validateDebugUnits: Bool,
        validatesInputs: Bool
    ) throws -> SourceTerms {
        let activeProfiles = if validatesInputs {
            try ValidatedProfiles.validate(profiles).toCoreProfiles()
        } else {
            profiles
        }

        let Q_ie_watts = try compute(
            electronDensity: activeProfiles.electronDensity.value,
            electronTemperature: activeProfiles.electronTemperature.value,
            ionTemperature: activeProfiles.ionTemperature.value,
            validatesInputs: validatesInputs
        )

        // Convert to MW/m³ for SourceTerms
        let Q_ie = PhysicsConstants.wattsToMegawatts(Q_ie_watts)

        let metadata: SourceMetadataCollection?
        if includesMetadata {
            let Q_ie_min = Q_ie.min().item(Float.self)
            let Q_ie_max = Q_ie.max().item(Float.self)
            guard !Q_ie_min.isNaN && !Q_ie_min.isInfinite &&
                  !Q_ie_max.isNaN && !Q_ie_max.isInfinite else {
                throw NumericalValidationError.nonFinite(
                    field: "ionElectronExchange",
                    minimum: Q_ie_min,
                    maximum: Q_ie_max
                )
            }

            let cellVolumes = geometricFactors.cellVolumes.value
            let P_ie_total = (Q_ie_watts * cellVolumes).sum()
            eval(P_ie_total)
            let exchangePower = P_ie_total.item(Float.self)

            let exchangeMetadata = SourceMetadata(
                modelName: "ion_electron_exchange",
                category: .other,
                ionPower: exchangePower,
                electronPower: -exchangePower
            )
            metadata = Self.metadataCollection(
                existing: sources.metadata,
                appending: exchangeMetadata
            )
        } else {
            metadata = sources.metadata
        }

        // Create new SourceTerms with updated heating and metadata
        return SourceTerms(
            ionHeating: evaluationMode.wrap(sources.ionHeating.value + Q_ie),
            electronHeating: evaluationMode.wrap(sources.electronHeating.value - Q_ie),
            particleSource: sources.particleSource,
            currentSource: sources.currentSource,
            metadata: metadata,
            validateDebugUnits: validateDebugUnits
        )
    }
}
