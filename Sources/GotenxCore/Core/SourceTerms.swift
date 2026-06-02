import Foundation
import MLX

// MARK: - Source Terms

/// Source and sink terms for plasma equations
///
/// Metadata enables accurate separation of fusion, auxiliary, ohmic, and
/// radiation contributions without fixed-ratio estimation.
public struct SourceTerms: Sendable, Equatable {
    /// Ion heating [MW/m^3]
    public let ionHeating: EvaluatedArray

    /// Electron heating [MW/m^3]
    public let electronHeating: EvaluatedArray

    /// Particle source [m^-3/s]
    public let particleSource: EvaluatedArray

    /// Current source [MA/m^2]
    public let currentSource: EvaluatedArray

    /// Source metadata for power balance
    ///
    /// Diagnostic paths must provide metadata. Solver-only paths may omit it because
    /// metadata integration performs host-side scalar reads and is not differentiable.
    public let metadata: SourceMetadataCollection?

    public init(
        ionHeating: EvaluatedArray,
        electronHeating: EvaluatedArray,
        particleSource: EvaluatedArray,
        currentSource: EvaluatedArray,
        metadata: SourceMetadataCollection? = nil,
        validateDebugUnits: Bool = true
    ) {
        #if DEBUG
        if validateDebugUnits {
        // ═══════════════════════════════════════════════════════════════
        // DEFENSE LAYER: Detect unit errors early (Debug builds only)
        // ═══════════════════════════════════════════════════════════════

        // Validate array shapes
        let cellCount = ionHeating.shape[0]
        precondition(electronHeating.shape[0] == cellCount,
                    "SourceTerms: electron heating shape mismatch (expected \(cellCount), got \(electronHeating.shape[0]))")
        precondition(particleSource.shape[0] == cellCount,
                    "SourceTerms: particle source shape mismatch (expected \(cellCount), got \(particleSource.shape[0]))")
        precondition(currentSource.shape[0] == cellCount,
                    "SourceTerms: current source shape mismatch (expected \(cellCount), got \(currentSource.shape[0]))")

        // Validate heating units (should be MW/m³, NOT eV/(m³·s)).
        // This guard is a unit-conversion sentinel, not a physics limiter:
        // localized exchange terms can exceed ordinary external-heating densities,
        // while an already-converted eV/(m³·s) value is typically O(1e24).
        let heatingUnitSentinel: Float = 1e12
        let maxIonHeatingMagnitude = abs(ionHeating.value).max().item(Float.self)
        let maxElectronHeatingMagnitude = abs(electronHeating.value).max().item(Float.self)

        precondition(maxIonHeatingMagnitude < heatingUnitSentinel,
            """
            SourceTerms: Suspicious ion heating magnitude: \(maxIonHeatingMagnitude) MW/m³

            If this value is ~1e24, you likely returned eV/(m³·s) instead of MW/m³!

            EXPECTED: Physics models return MW/m³
            ACTUAL: You may have converted to eV/(m³·s)

            FIX: Return MW/m³ from your SourceModel.computeTerms()
            Conversion to eV/(m³·s) happens in Block1DCoeffsBuilder, not in physics models.
            """)

        precondition(maxElectronHeatingMagnitude < heatingUnitSentinel,
            """
            SourceTerms: Suspicious electron heating magnitude: \(maxElectronHeatingMagnitude) MW/m³

            If this value is ~1e24, you likely returned eV/(m³·s) instead of MW/m³!

            EXPECTED: Physics models return MW/m³
            ACTUAL: You may have converted to eV/(m³·s)

            FIX: Return MW/m³ from your SourceModel.computeTerms()
            Conversion to eV/(m³·s) happens in Block1DCoeffsBuilder, not in physics models.
            """)

        // Validate particle source units (should be m^-3/s)
        // ITER gas puff: ~1e21 particles/s over ~1000 m³ → ~1e18 m^-3/s average
        // Allow up to 1e20 m^-3/s for localized injection
        let maxParticleSource = abs(particleSource.value).max().item(Float.self)

        precondition(maxParticleSource < 1e20,
            """
            SourceTerms: Suspicious particle source value: \(maxParticleSource) m^-3/s

            Typical range: 1e16 - 1e19 m^-3/s
            If value is much larger, check your calculation.
            """)

        // Validate current density (should be MA/m²)
        // ITER: ~15 MA total current, ~30 m² cross-section → ~0.5 MA/m² average
        // Allow up to 100 MA/m² for localized current drive
        let maxCurrentSource = abs(currentSource.value).max().item(Float.self)

        precondition(maxCurrentSource < 100.0,
            """
            SourceTerms: Suspicious current source value: \(maxCurrentSource) MA/m²

            Typical range: 0.01 - 10 MA/m²
            If value is much larger, check your calculation.
            """)
        }
        #endif

        self.ionHeating = ionHeating
        self.electronHeating = electronHeating
        self.particleSource = particleSource
        self.currentSource = currentSource
        self.metadata = metadata
    }

    /// Equatable conformance (metadata ignored for array comparison)
    public static func == (lhs: SourceTerms, rhs: SourceTerms) -> Bool {
        lhs.ionHeating == rhs.ionHeating &&
        lhs.electronHeating == rhs.electronHeating &&
        lhs.particleSource == rhs.particleSource &&
        lhs.currentSource == rhs.currentSource
        // Note: metadata intentionally excluded from equality check
    }

    /// Zero source terms
    public static func zero(
        cellCount: Int,
        metadata: SourceMetadataCollection? = SourceMetadataCollection.empty,
        validateDebugUnits: Bool = true
    ) -> SourceTerms {
        SourceTerms(
            ionHeating: .zeros([cellCount]),
            electronHeating: .zeros([cellCount]),
            particleSource: .zeros([cellCount]),
            currentSource: .zeros([cellCount]),
            metadata: metadata,
            validateDebugUnits: validateDebugUnits
        )
    }

    public static func invalidNumerics(cellCount: Int) -> SourceTerms {
        let invalid = EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float.nan)))
        return SourceTerms(
            ionHeating: invalid,
            electronHeating: invalid,
            particleSource: invalid,
            currentSource: invalid,
            metadata: nil,
            validateDebugUnits: false
        )
    }

    /// Add two source terms
    ///
    /// Phase 4a: Merges metadata collections when both are present
    public static func + (lhs: SourceTerms, rhs: SourceTerms) -> SourceTerms {
        lhs.adding(rhs)
    }

    public func adding(_ other: SourceTerms, validateDebugUnits: Bool = true) -> SourceTerms {
        // Merge metadata collections
        let mergedMetadata: SourceMetadataCollection?
        switch (metadata, other.metadata) {
        case (let lm?, let rm?):
            mergedMetadata = SourceMetadataCollection(entries: lm.entries + rm.entries)
        case (let lm?, nil):
            mergedMetadata = lm
        case (nil, let rm?):
            mergedMetadata = rm
        case (nil, nil):
            mergedMetadata = nil
        }

        return SourceTerms(
            ionHeating: EvaluatedArray(evaluating: ionHeating.value + other.ionHeating.value),
            electronHeating: EvaluatedArray(evaluating: electronHeating.value + other.electronHeating.value),
            particleSource: EvaluatedArray(evaluating: particleSource.value + other.particleSource.value),
            currentSource: EvaluatedArray(evaluating: currentSource.value + other.currentSource.value),
            metadata: mergedMetadata,
            validateDebugUnits: validateDebugUnits
        )
    }
}

extension SourceTerms {
    public func validateNumerics(
        expectedCellCount: Int,
        requiresMetadata: Bool = false
    ) throws {
        try NumericalValidation.validateShape(ionHeating.value, field: "ionHeating", expected: [expectedCellCount])
        try NumericalValidation.validateShape(electronHeating.value, field: "electronHeating", expected: [expectedCellCount])
        try NumericalValidation.validateShape(particleSource.value, field: "particleSource", expected: [expectedCellCount])
        try NumericalValidation.validateShape(currentSource.value, field: "currentSource", expected: [expectedCellCount])

        try NumericalValidation.validateFinite(ionHeating.value, field: "ionHeating")
        try NumericalValidation.validateFinite(electronHeating.value, field: "electronHeating")
        try NumericalValidation.validateFinite(particleSource.value, field: "particleSource")
        try NumericalValidation.validateFinite(currentSource.value, field: "currentSource")

        guard let metadata else {
            if requiresMetadata {
                throw NumericalValidationError.missingMetadata(field: "SourceTerms.metadata")
            }
            return
        }

        try metadata.validatePowerAccounting()
    }
}
