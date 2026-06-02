import MLX
import Foundation

// MARK: - Particle Conservation

/// Enforces particle conservation: ∫ nₑ dV = const
///
/// ## Physics Background
///
/// In a closed tokamak (no particle sources at boundaries), the total particle number
/// must remain constant over time. This is a fundamental conservation law from the
/// continuity equation:
///
/// ```
/// ∂nₑ/∂t + ∇·Γ = Sₙ
/// ```
///
/// For closed system (Sₙ = 0, Γ·n̂ = 0 at boundary):
///
/// ```
/// dN/timeStep = 0  →  N = ∫ nₑ dV = const
/// ```
///
/// ## Numerical Drift
///
/// Over long simulations (20,000+ timesteps), floating-point round-off errors cause
/// numerical drift:
///
/// ```
/// Initial:  N₀ = 1.0 × 10²¹ particles
/// After 20k steps: N = 0.99 × 10²¹  (1% drift)
/// ```
///
/// This violates physics and can lead to unphysical plasma states.
///
/// ## Correction Method
///
/// We apply **uniform density scaling** to restore particle conservation:
///
/// ```swift
/// correctionFactor = N₀ / N
/// nₑ_corrected = nₑ × correctionFactor
/// ```
///
/// **Why uniform scaling works**:
/// - Preserves density profile shape (gradients maintained)
/// - Small corrections (< 1%) don't affect physics
/// - GPU-efficient (element-wise multiplication)
///
/// ## Safety Features
///
/// - **Zero division protection**: Returns 1.0 if N ≤ 0
/// - **Non-finite detection**: Catches NaN/Inf before correction
/// - **Clamping**: Limits correction to ±20% to prevent instability
///
/// ## Example Usage
///
/// ```swift
/// let conservation = ParticleConservation(driftTolerance: 0.005)  // 0.5%
///
/// // Initial reference
/// let N0 = conservation.computeConservedQuantity(
///     profiles: initialProfiles,
///     geometry: geometry
/// )
///
/// // After many timesteps
/// let N = conservation.computeConservedQuantity(
///     profiles: currentProfiles,
///     geometry: geometry
/// )
///
/// let drift = abs(N - N0) / N0
/// if drift > 0.005 {
///     let factor = conservation.computeCorrectionFactor(current: N, reference: N0)
///     let corrected = conservation.applyCorrection(
///         profiles: currentProfiles,
///         correctionFactor: factor
///     )
/// }
/// ```
public struct ParticleConservation: ConservationLaw {
    public let name = "ParticleConservation"
    public let driftTolerance: Float

    /// Initialize particle conservation law
    ///
    /// - Parameter driftTolerance: Relative drift threshold for correction (default: 0.005 = 0.5%)
    public init(driftTolerance: Float = 0.005) {
        self.driftTolerance = driftTolerance
    }

    // MARK: - ConservationLaw Protocol

    public func computeConservedQuantity(
        profiles: CoreProfiles,
        geometry: Geometry
    ) -> Float {
        // Extract electron density and cell volumes
        let ne = profiles.electronDensity.value                                    // [cellCount], m^-3
        let volumes = GeometricFactors.from(geometry: geometry).cellVolumes.value  // [cellCount], m^3

        // Total particle number: N = ∫ nₑ dV ≈ Σ nₑ,i × Vᵢ
        let totalParticles = (ne * volumes).sum()    // GPU sum reduction
        eval(totalParticles)

        // Extract scalar value
        return totalParticles.item(Float.self)       // Total particles (dimensionless)
    }

    public func computeCorrectionFactor(
        current: Float,
        reference: Float
    ) -> Float {
        // Guard against invalid values
        guard current > 0, current.isFinite else {
            print("[ParticleConservation] Invalid current value: \(current), no correction applied")
            return 1.0
        }
        guard reference > 0, reference.isFinite else {
            print("[ParticleConservation] Invalid reference value: \(reference), no correction applied")
            return 1.0
        }

        // Compute correction factor: factor = N₀ / N
        let factor = reference / current

        // Clamp to ±20% to prevent large corrections (which could cause instability)
        if abs(factor - 1.0) > 0.2 {
            let clampedFactor: Float = factor > 1.0 ? 1.2 : 0.8
            print("""
                [ParticleConservation] Large correction (\(String(format: "%.3f", factor))×) clamped to \
                \(String(format: "%.3f", clampedFactor))×
                """)
            return clampedFactor
        }

        return factor
    }

    public func applyCorrection(
        profiles: CoreProfiles,
        correctionFactor: Float
    ) -> CoreProfiles {
        // Apply uniform density scaling: nₑ_new = nₑ × factor
        let ne_corrected = profiles.electronDensity.value * correctionFactor
        eval(ne_corrected)

        // Return corrected profiles (only density changed)
        return CoreProfiles(
            ionTemperature: profiles.ionTemperature,
            electronTemperature: profiles.electronTemperature,
            electronDensity: EvaluatedArray(evaluating: ne_corrected),
            poloidalFlux: profiles.poloidalFlux
        )
    }
}

// MARK: - Diagnostics

extension ParticleConservation {
    /// Compute relative drift for diagnostics
    ///
    /// - Parameters:
    ///   - current: Current total particle number
    ///   - reference: Reference (initial) particle number
    /// - Returns: Relative drift: |N - N₀| / N₀
    public func computeRelativeDrift(current: Float, reference: Float) -> Float {
        guard reference > 0 else { return 0.0 }
        return abs(current - reference) / reference
    }

    /// Check if correction is needed
    ///
    /// - Parameters:
    ///   - current: Current total particle number
    ///   - reference: Reference (initial) particle number
    /// - Returns: True if drift exceeds tolerance
    public func needsCorrection(current: Float, reference: Float) -> Bool {
        let drift = computeRelativeDrift(current: current, reference: reference)
        return drift > driftTolerance
    }
}
