import Foundation
import MLX

// MARK: - Source Terms

/// Source and sink terms for plasma equations
///
/// Phase 4a: Added optional metadata field for power balance tracking.
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

    /// Phase 4a: Source metadata for power balance (optional)
    ///
    /// When present, enables accurate power categorization.
    /// When nil, falls back to Phase 3 fixed-ratio estimation.
    public let metadata: SourceMetadataCollection?

    public init(
        ionHeating: EvaluatedArray,
        electronHeating: EvaluatedArray,
        particleSource: EvaluatedArray,
        currentSource: EvaluatedArray,
        metadata: SourceMetadataCollection? = nil
    ) {
        #if DEBUG
        // ═══════════════════════════════════════════════════════════════
        // DEFENSE LAYER: Detect unit errors early (Debug builds only)
        // ═══════════════════════════════════════════════════════════════

        // Validate array shapes
        let nCells = ionHeating.shape[0]
        precondition(electronHeating.shape[0] == nCells,
                    "SourceTerms: electron heating shape mismatch (expected \(nCells), got \(electronHeating.shape[0]))")
        precondition(particleSource.shape[0] == nCells,
                    "SourceTerms: particle source shape mismatch (expected \(nCells), got \(particleSource.shape[0]))")
        precondition(currentSource.shape[0] == nCells,
                    "SourceTerms: current source shape mismatch (expected \(nCells), got \(currentSource.shape[0]))")

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
    public static func zero(nCells: Int) -> SourceTerms {
        SourceTerms(
            ionHeating: .zeros([nCells]),
            electronHeating: .zeros([nCells]),
            particleSource: .zeros([nCells]),
            currentSource: .zeros([nCells]),
            metadata: SourceMetadataCollection.empty  // Always provide empty metadata
        )
    }

    /// Add two source terms
    ///
    /// Phase 4a: Merges metadata collections when both are present
    public static func + (lhs: SourceTerms, rhs: SourceTerms) -> SourceTerms {
        // Merge metadata collections
        let mergedMetadata: SourceMetadataCollection?
        switch (lhs.metadata, rhs.metadata) {
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
            ionHeating: EvaluatedArray(evaluating: lhs.ionHeating.value + rhs.ionHeating.value),
            electronHeating: EvaluatedArray(evaluating: lhs.electronHeating.value + rhs.electronHeating.value),
            particleSource: EvaluatedArray(evaluating: lhs.particleSource.value + rhs.particleSource.value),
            currentSource: EvaluatedArray(evaluating: lhs.currentSource.value + rhs.currentSource.value),
            metadata: mergedMetadata
        )
    }
}
