// CollisionalityHelpers.swift
// Collisionality and neoclassical transport helpers

import MLX

/// Collisionality and neoclassical transport helpers
///
/// **References**:
/// - Sauter et al., "Neoclassical conductivity and bootstrap current", PoP 6, 2834 (1999)
/// - Wesson, "Tokamak Physics" (2nd ed.), Chapter 7
public struct CollisionalityHelpers {

    /// Compute electron-ion collision time τₑ [s]
    ///
    /// **Formula**: τₑ = C * Tₑ^(3/2) / (nₑ * ln(Λ))
    ///
    /// **Coefficient derivation**:
    /// NRL Plasma Formulary: τₑ [s] = 3.44 × 10^5 * Tₑ[eV]^(3/2) / (nₑ[cm⁻³] * ln Λ)
    ///
    /// Converting to m⁻³ (Gotenx uses nₑ in m⁻³, not cm⁻³):
    /// - nₑ[cm⁻³] = nₑ[m⁻³] / 10⁶
    ///
    /// Substituting:
    /// τₑ = 3.44e5 * Tₑ[eV]^1.5 / ((nₑ[m⁻³] / 1e6) * ln Λ)
    ///    = 3.44e5 * 1e6 * Tₑ[eV]^1.5 / (nₑ[m⁻³] * ln Λ)
    ///    = 3.44e11 * Tₑ[eV]^1.5 / (nₑ[m⁻³] * ln Λ)
    ///
    /// **Verification** (ITER core: Tₑ = 10 keV = 10000 eV, nₑ = 1e20 m⁻³):
    /// τₑ = 3.44e11 * (10000)^1.5 / (1e20 * 17)
    ///    = 3.44e11 * 1e6 / 1.7e21
    ///    = 3.44e17 / 1.7e21
    ///    ≈ 2.0e-4 s = 0.2 ms ✓ (correct order of magnitude)
    ///
    /// - Parameters:
    ///   - electronTemperature: Electron temperature [eV], shape [cellCount]
    ///   - electronDensity: Electron density [m⁻³], shape [cellCount]
    ///   - coulombLog: Coulomb logarithm (default: 17.0)
    /// - Returns: Collision time [s], shape [cellCount]
    public static func computeCollisionTime(
        electronTemperature: MLXArray,
        electronDensity: MLXArray,
        coulombLog: Float = 17.0
    ) -> MLXArray {
        // Correct coefficient for T_e[eV], n_e[m⁻³] → τₑ[s]
        return 3.44e11 * pow(electronTemperature, 1.5) / (electronDensity * coulombLog)
    }

    /// Compute normalized collisionality ν*
    ///
    /// Formula: ν* = (R₀ q) / (ε^(3/2) vₜₕ τₑ)
    ///
    /// Where:
    /// - ε = r/R₀ (inverse aspect ratio)
    /// - q: safety factor
    /// - vₜₕ = √(2Tₑ/mₑ): thermal velocity
    ///
    /// - Parameters:
    ///   - electronTemperature: Electron temperature [eV], shape [cellCount]
    ///   - electronDensity: Electron density [m⁻³], shape [cellCount]
    ///   - geometry: Tokamak geometry
    /// - Returns: Normalized collisionality ν* [dimensionless], shape [cellCount]
    public static func computeNormalizedCollisionality(
        electronTemperature: MLXArray,
        electronDensity: MLXArray,
        geometry: Geometry
    ) -> MLXArray {
        let collisionTime = computeCollisionTime(
            electronTemperature: electronTemperature,
            electronDensity: electronDensity
        )

        let epsilon = geometry.radii.value / geometry.majorRadius
        let q = approximateSafetyFactor(geometry: geometry)

        // Thermal velocity: vₜₕ = √(2Tₑ/mₑ)
        // With Tₑ in eV: vₜₕ = √(3.514e11 * Tₑ)  [m/s]
        let thermalVelocity = sqrt(3.514e11 * electronTemperature)

        let normalizedCollisionality = (geometry.majorRadius * q) / (
            pow(epsilon, 1.5) * thermalVelocity * collisionTime
        )

        return normalizedCollisionality
    }

    /// Approximate safety factor q from geometry
    ///
    /// Parabolic approximation: q ≈ 1 + (r/a)²
    ///
    /// - Parameter geometry: Tokamak geometry
    /// - Returns: Safety factor [dimensionless], shape [cellCount]
    private static func approximateSafetyFactor(geometry: Geometry) -> MLXArray {
        let r_norm = geometry.radii.value / geometry.minorRadius
        return 1.0 + r_norm * r_norm
    }
}
