import Foundation
import MLX
import GotenxCore

/// Bremsstrahlung radiation model
///
/// Free electrons radiating when deflected by ions.
/// Always a loss term (negative power).
///
/// Physical equation:
/// P_brems = -C_brems * n_e² * effectiveCharge * √T_e * (1 + f_rel)
///
/// Where:
/// - C_brems = 5.35 × 10⁻³⁷ [W·m³·eV^(-1/2)]
/// - f_rel = (T_e/511000) * (4√2 - 1) / π (relativistic correction)
///
/// Units:
/// - Input: n_e [m⁻³], T_e [eV]
/// - Output: P_brems [W/m³] (negative = loss)
public struct Bremsstrahlung: Sendable {

    /// Effective charge number
    public let effectiveCharge: Float

    /// Include relativistic correction for high temperatures
    public let includeRelativistic: Bool

    /// Bremsstrahlung coefficient [W·m³·eV^(-1/2)]
    private let C_brems: Float = PhysicsConstants.bremsCoefficient

    /// Electron rest mass energy [eV]
    private let m_e_c2: Float = PhysicsConstants.electronRestMass

    /// Create Bremsstrahlung radiation model
    ///
    /// - Parameters:
    ///   - effectiveCharge: Effective charge (default: 1.5)
    ///   - includeRelativistic: Apply relativistic correction (default: true)
    public init(effectiveCharge: Float = 1.5, includeRelativistic: Bool = true) {
        self.effectiveCharge = effectiveCharge
        self.includeRelativistic = includeRelativistic
    }

    /// Compute Bremsstrahlung radiation power
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m⁻³], shape [cellCount]
    ///   - electronTemperature: Electron temperature [eV], shape [cellCount]
    /// - Returns: Radiation power [W/m³] (negative = loss), shape [cellCount]
    /// - Throws: PhysicsError if inputs are invalid
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func compute(electronDensity: MLXArray, electronTemperature: MLXArray) throws -> MLXArray {
        try compute(
            electronDensity: electronDensity,
            electronTemperature: electronTemperature,
            validatesInputs: true
        )
    }

    package func compute(
        electronDensity: MLXArray,
        electronTemperature: MLXArray,
        validatesInputs: Bool
    ) throws -> MLXArray {
        if validatesInputs {
            try PhysicsValidation.validateDensity(electronDensity, name: "electronDensity")
            try PhysicsValidation.validateTemperature(electronTemperature, name: "electronTemperature")
            try PhysicsValidation.validateShapes([electronDensity, electronTemperature], names: ["electronDensity", "electronTemperature"])
        }

        var f_rel = MLXArray.zeros(like: electronTemperature)

        if includeRelativistic {
            // Relativistic correction: only significant for electronTemperature > 1 keV
            // f_rel = (T_e / m_e c²) * (4√2 - 1) / π
            let mask = MLX.greater(electronTemperature, Float(1000.0))  // Only apply for electronTemperature > 1 keV
            let mask_float = mask.asType(.float32)  // Convert Bool to 0/1

            let relativistic_factor = (electronTemperature / m_e_c2) * (Float(4.0) * sqrt(Float(2.0)) - Float(1.0)) / Float.pi
            f_rel = mask_float * relativistic_factor
        }

        // Bremsstrahlung power (negative = energy loss) [W/m³]
        // P_brems = -C * n_e² * effectiveCharge * √T_e * (1 + f_rel)
        // CRITICAL: Multiply small values first to prevent Float32 overflow
        // electronDensity ≈ 10^20, C_brems ≈ 5.35e-37, sqrt(electronTemperature) ≈ 100
        // Order: (-C_brems * electronDensity) * sqrt(electronTemperature) * electronDensity * effectiveCharge avoids 10^40 overflow
        let P_brems_watts = -C_brems * electronDensity * sqrt(electronTemperature) * electronDensity * effectiveCharge * (Float(1.0) + f_rel)

        // Return lazy MLXArray - caller will eval() when needed
        return P_brems_watts
    }

    /// Compute classical Bremsstrahlung (no relativistic correction)
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m⁻³]
    ///   - electronTemperature: Electron temperature [eV]
    /// - Returns: Classical Bremsstrahlung power [W/m³]
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func computeClassical(electronDensity: MLXArray, electronTemperature: MLXArray) -> MLXArray {
        // CRITICAL: Multiply small values first to prevent Float32 overflow
        // Order: (-C_brems * electronDensity) * sqrt(electronTemperature) * electronDensity * effectiveCharge avoids 10^40 overflow
        return -C_brems * electronDensity * sqrt(electronTemperature) * electronDensity * effectiveCharge
    }

    /// Compute relativistic correction factor
    ///
    /// - Parameter electronTemperature: Electron temperature [eV]
    /// - Returns: Relativistic correction factor f_rel (dimensionless)
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func computeRelativisticCorrection(electronTemperature: MLXArray) -> MLXArray {
        let mask = MLX.greater(electronTemperature, Float(1000.0))
        let mask_float = mask.asType(.float32)  // Convert Bool to 0/1
        let factor = (electronTemperature / m_e_c2) * (Float(4.0) * sqrt(Float(2.0)) - Float(1.0)) / Float.pi
        let result = mask_float * factor
        // Return lazy MLXArray - caller will eval() when needed
        return result
    }

    /// Check if relativistic effects are significant
    ///
    /// - Parameter electronTemperature: Electron temperature [eV]
    /// - Returns: True if relativistic correction > 1%
    ///
    /// - Note: Returns a lazy MLXArray. Call `eval()` before using `.item()` to extract values.
    ///   When used with `EvaluatedArray(evaluating:)`, evaluation is automatic.
    public func isRelativisticSignificant(electronTemperature: MLXArray) -> MLXArray {
        let f_rel = computeRelativisticCorrection(electronTemperature: electronTemperature)
        let result = MLX.greater(f_rel, Float(0.01))
        // Return lazy MLXArray - caller will eval() when needed
        return result
    }

    /// Compute source metadata for power balance tracking
    ///
    /// - Parameters:
    ///   - profiles: Current plasma profiles
    ///   - geometry: Geometry for volume integration
    /// - Returns: Source metadata with Bremsstrahlung radiation power (negative)
    /// - Throws: PhysicsError if computation fails
    public func computeMetadata(
        profiles: CoreProfiles,
        geometry: Geometry
    ) throws -> SourceMetadata {

        let P_brems_watts = try compute(
            electronDensity: profiles.electronDensity.value,
            electronTemperature: profiles.electronTemperature.value
        )

        // Volume integration: ∫ P dV → [W/m³] × [m³] = [W]
        let cellVolumes = GeometricFactors.from(geometry: geometry).cellVolumes.value
        let P_brems_total = (P_brems_watts * cellVolumes).sum()
        eval(P_brems_total)

        let bremsPower = P_brems_total.item(Float.self)

        // Bremsstrahlung is a power loss (negative value)
        return SourceMetadata(
            modelName: "bremsstrahlung",
            category: .radiation,
            ionPower: 0,  // Only affects electrons
            electronPower: bremsPower  // Already negative from compute()
        )
    }
}

// MARK: - Source Model Protocol Conformance

extension Bremsstrahlung {
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

    /// Apply Bremsstrahlung radiation to source terms
    ///
    /// Subtracts radiation losses from electron heating.
    ///
    /// - Parameters:
    ///   - sources: Source terms to modify
    ///   - profiles: Current plasma profiles
    ///   - geometry: Tokamak geometry for metadata computation
    /// - Returns: Modified source terms with radiation losses
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
        let P_brems_watts = try compute(
            electronDensity: profiles.electronDensity.value,
            electronTemperature: profiles.electronTemperature.value,
            validatesInputs: validatesInputs
        )

        // Convert to MW/m³ for SourceTerms
        let P_brems = PhysicsConstants.wattsToMegawatts(P_brems_watts)

        let metadata: SourceMetadataCollection?
        if includesMetadata {
            let cellVolumes = geometricFactors.cellVolumes.value
            let P_brems_total = (P_brems_watts * cellVolumes).sum()
            eval(P_brems_total)
            let bremsPower = P_brems_total.item(Float.self)

            let bremsMetadata = SourceMetadata(
                modelName: "bremsstrahlung",
                category: .radiation,
                ionPower: 0,
                electronPower: bremsPower
            )
            metadata = Self.metadataCollection(
                existing: sources.metadata,
                appending: bremsMetadata
            )
        } else {
            metadata = sources.metadata
        }

        // Create new SourceTerms with updated electron heating and metadata
        return SourceTerms(
            ionHeating: sources.ionHeating,
            electronHeating: evaluationMode.wrap(sources.electronHeating.value + P_brems),
            particleSource: sources.particleSource,
            currentSource: sources.currentSource,
            metadata: metadata,
            validateDebugUnits: validateDebugUnits
        )
    }
}

// MARK: - Diagnostic Output

extension Bremsstrahlung {

    /// Compute total radiated power
    ///
    /// Integrates Bremsstrahlung power over plasma volume.
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m⁻³]
    ///   - electronTemperature: Electron temperature [eV]
    ///   - geometry: Tokamak geometry
    /// - Returns: Total radiated power [W]
    public func computeTotalPower(
        electronDensity: MLXArray,
        electronTemperature: MLXArray,
        geometry: Geometry
    ) throws -> Float {

        let P_brems_density = try compute(electronDensity: electronDensity, electronTemperature: electronTemperature)

        // Integrate over volume: P_total = Σ P_brems * V_cell
        let cellVolumes = GeometricFactors.from(geometry: geometry).cellVolumes.value
        let P_total = (P_brems_density * cellVolumes).sum()

        return P_total.item(Float.self)
    }

    /// Compute radiation fraction (P_rad / P_input)
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m⁻³]
    ///   - electronTemperature: Electron temperature [eV]
    ///   - geometry: Tokamak geometry
    ///   - inputPower: Total input power [W]
    /// - Returns: Radiation fraction (dimensionless)
    public func computeRadiationFraction(
        electronDensity: MLXArray,
        electronTemperature: MLXArray,
        geometry: Geometry,
        inputPower: Float
    ) throws -> Float {

        let P_rad = abs(try computeTotalPower(electronDensity: electronDensity, electronTemperature: electronTemperature, geometry: geometry))
        return P_rad / (inputPower + 1e-10)
    }
}
