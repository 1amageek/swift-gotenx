import Foundation
import MLX
import GotenxCore

/// Bootstrap current using Sauter model
///
/// Reference: Sauter et al., Physics of Plasmas 6(7), 2834-2839 (1999)
///
/// Computes self-generated toroidal current from pressure gradients
/// and trapped particles in tokamak geometry.
///
/// Bootstrap current formula:
/// j_bs = σ_bs * (L31 * ∇p_e/p_e + L32 * ∇n_e/n_e + L34 * ∇T_e/T_e)
///
/// where L31, L32, L34 are Sauter coefficients depending on:
/// - Trapped fraction: f_trap
/// - Collisionality: ν*
/// - Inverse aspect ratio: ε = r/R₀
///
/// Units:
/// - Input: plasma profiles (n_e [m⁻³], T_e [eV], T_i [eV]), geometry
/// - Output: j_bs [A/m²]
public struct SauterBootstrapModel: Sendable {

    /// Effective charge number
    public let effectiveCharge: Float

    /// Coulomb logarithm
    public let coulombLogarithm: Float

    /// Create Sauter bootstrap current model
    ///
    /// - Parameters:
    ///   - effectiveCharge: Effective charge (default: 1.5)
    ///   - coulombLogarithm: Coulomb logarithm (default: 17.0)
    public init(effectiveCharge: Float = 1.5, coulombLogarithm: Float = 17.0) {
        self.effectiveCharge = effectiveCharge
        self.coulombLogarithm = coulombLogarithm
    }

    /// Compute bootstrap current density
    ///
    /// - Parameters:
    ///   - profiles: Core plasma profiles
    ///   - geometry: Tokamak geometry
    ///   - q: Safety factor [cellCount]
    /// - Returns: Bootstrap current density [A/m²], shape [cellCount]
    public func compute(
        profiles: CoreProfiles,
        geometry: Geometry,
        safetyFactor: MLXArray
    ) -> MLXArray {

        let electronDensity = profiles.electronDensity.value
        let electronTemperature = profiles.electronTemperature.value

        let R0 = geometry.majorRadius
        let geomFactors = GeometricFactors.from(geometry: geometry)
        let r = geomFactors.cellRadii.value
        let epsilon = r / R0
        let sqrt_eps = sqrt(epsilon)

        // Trapped particle fraction (Sauter formula)
        // f_trap = 1.46 * √ε / (1 + 0.46 * √ε)
        let f_trap = 1.46 * sqrt_eps / (1.0 + 0.46 * sqrt_eps)

        // Collisionality (normalized)
        // ν* = 6.921×10⁻¹⁸ * q * R₀ * n_e * effectiveCharge * ln(Λ) / (T_e² * ε^(3/2))
        let normalizedCollisionality = 6.921e-18 * safetyFactor * R0 * electronDensity * effectiveCharge * coulombLogarithm
                      / (electronTemperature * electronTemperature * pow(epsilon, 1.5))

        // Sauter F-functions
        let F31 = computeF31(normalizedCollisionality: normalizedCollisionality, epsilon: epsilon)
        let F32_eff = computeF32_eff(normalizedCollisionality: normalizedCollisionality, epsilon: epsilon)
        let F32_ee = computeF32_ee(normalizedCollisionality: normalizedCollisionality, epsilon: epsilon)

        // Sauter L-coefficients (broken into parts for compiler)
        let term1 = (1.0 + 0.15 / (f_trap * f_trap)) * F31
        let term2 = 0.4 / (1.0 + 0.5 * effectiveCharge) * sqrt_eps * F32_eff / (f_trap * f_trap)
        let denominator = 1.0 + 0.7 * sqrt(effectiveCharge - 1.0)
        let L31 = (term1 + term2) / denominator

        let L32 = (1.0 + 0.15 / (f_trap * f_trap)) * F32_ee / f_trap
        let L34 = -F32_ee / f_trap

        // Compute gradients using central differences
        let electronPressureGradient = computeGradient(electronDensity * electronTemperature, geometry: geometry)
        let electronDensityGradient = computeGradient(electronDensity, geometry: geometry)
        let electronTemperatureGradient = computeGradient(electronTemperature, geometry: geometry)

        // Electron pressure [Pa]
        let electronPressure = electronDensity * electronTemperature * PhysicsConstants.electronVolt

        // Bootstrap current formula
        // j_bs = L31 * ∇p_e/p_e + L32 * ∇n_e/n_e + L34 * ∇T_e/T_e
        let normalizedBootstrapCurrent = L31 * electronPressureGradient / (electronPressure + 1e-10)
                            + L32 * electronDensityGradient / (electronDensity + 1e-10)
                            + L34 * electronTemperatureGradient / (electronTemperature + 1e-10)

        // Multiply by conductivity factor to get actual current
        let sigma_factor = computeConductivityFactor(
            electronTemperature: electronTemperature,
            electronDensity: electronDensity,
            B: geometry.toroidalField
        )

        let j_bs = sigma_factor * normalizedBootstrapCurrent

        return j_bs
    }

    // MARK: - Sauter F-Functions

    /// Compute F31 function (pressure gradient coefficient)
    ///
    /// - Parameters:
    ///   - normalizedCollisionality: Normalized collisionality
    ///   - epsilon: Inverse aspect ratio
    /// - Returns: F31 coefficient
    private func computeF31(normalizedCollisionality: MLXArray, epsilon: MLXArray) -> MLXArray {
        let sqrt_eps = sqrt(epsilon)

        // Banana regime (low collisionality)
        let F31_banana = sqrt_eps * (0.75 + 0.25 * normalizedCollisionality)

        // Plateau regime (intermediate collisionality)
        let F31_plateau = epsilon / (1.0 + 0.5 * normalizedCollisionality)

        // Interpolate between regimes
        let F31 = F31_banana * exp(-normalizedCollisionality) + F31_plateau * (1.0 - exp(-normalizedCollisionality))

        return F31
    }

    /// Compute F32_eff function (density gradient coefficient - effective)
    ///
    /// - Parameters:
    ///   - normalizedCollisionality: Normalized collisionality
    ///   - epsilon: Inverse aspect ratio
    /// - Returns: F32_eff coefficient
    private func computeF32_eff(normalizedCollisionality: MLXArray, epsilon: MLXArray) -> MLXArray {
        let sqrt_eps = sqrt(epsilon)
        return sqrt_eps * (1.0 + normalizedCollisionality) / pow(1.0 + 0.15 * normalizedCollisionality, 2)
    }

    /// Compute F32_ee function (density gradient coefficient - electron-electron)
    ///
    /// - Parameters:
    ///   - normalizedCollisionality: Normalized collisionality
    ///   - epsilon: Inverse aspect ratio
    /// - Returns: F32_ee coefficient
    private func computeF32_ee(normalizedCollisionality: MLXArray, epsilon: MLXArray) -> MLXArray {
        let sqrt_eps = sqrt(epsilon)
        let Z = effectiveCharge

        // Split scalar prefactor from MLXArray term to keep the type-checker fast
        let prefactor: Float = (0.05 + 0.62 * Z) / (Z * Z)
        let collisionalityTerm = sqrt_eps / (1.0 + 0.44 * normalizedCollisionality)
        return prefactor * collisionalityTerm
    }

    // MARK: - Helper Functions

    /// Compute gradient using central differences
    ///
    /// - Parameters:
    ///   - field: Field to differentiate [cellCount]
    ///   - geometry: Tokamak geometry
    /// - Returns: Gradient [cellCount]
    private func computeGradient(_ field: MLXArray, geometry: Geometry) -> MLXArray {
        let cellCount = field.shape[0]

        guard cellCount > 2 else {
            // Not enough points for gradient
            return MLXArray.zeros([cellCount])
        }

        let geomFactors = GeometricFactors.from(geometry: geometry)
        let cellRadii = geomFactors.cellRadii.value

        // Interior points: central difference
        // grad[i] = (field[i+1] - field[i-1]) / (r[i+1] - r[i-1])
        let dr_interior = cellRadii[2..<cellCount] - cellRadii[0..<(cellCount-2)]
        let df_interior = field[2..<cellCount] - field[0..<(cellCount-2)]
        let grad_interior = df_interior / (dr_interior + 1e-10)

        // Left boundary: forward difference
        // grad[0] = (field[1] - field[0]) / (r[1] - r[0])
        let dr_left = cellRadii[1] - cellRadii[0]
        let df_left = field[1] - field[0]
        let grad_left = df_left / (dr_left + 1e-10)

        // Right boundary: backward difference
        // grad[n-1] = (field[n-1] - field[n-2]) / (r[n-1] - r[n-2])
        let dr_right = cellRadii[cellCount-1] - cellRadii[cellCount-2]
        let df_right = field[cellCount-1] - field[cellCount-2]
        let grad_right = df_right / (dr_right + 1e-10)

        // Concatenate
        let grad = concatenated([
            grad_left.reshaped([1]),
            grad_interior,
            grad_right.reshaped([1])
        ], axis: 0)

        return grad
    }

    /// Compute conductivity factor for bootstrap current
    ///
    /// Simplified model: σ ∝ n_e * T_e^(3/2) / B²
    ///
    /// - Parameters:
    ///   - electronTemperature: Electron temperature [eV]
    ///   - electronDensity: Electron density [m⁻³]
    ///   - B: Magnetic field [T]
    /// - Returns: Conductivity factor
    private func computeConductivityFactor(
        electronTemperature: MLXArray,
        electronDensity: MLXArray,
        B: Float
    ) -> MLXArray {

        // Simplified conductivity
        // σ ∝ n_e * T_e^(3/2) / B²
        let sigma = electronDensity * pow(electronTemperature, 1.5) / (B * B + 1e-10)

        // Normalize to get reasonable current densities
        let normalization: Float = 1e-3

        return sigma * normalization
    }

    /// Compute trapped particle fraction
    ///
    /// - Parameter epsilon: Inverse aspect ratio ε = r/R₀
    /// - Returns: Trapped fraction f_trap
    public func computeTrappedFraction(epsilon: MLXArray) -> MLXArray {
        let sqrt_eps = sqrt(epsilon)
        return 1.46 * sqrt_eps / (1.0 + 0.46 * sqrt_eps)
    }

    /// Compute collisionality parameter
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m⁻³]
    ///   - electronTemperature: Electron temperature [eV]
    ///   - q: Safety factor
    ///   - epsilon: Inverse aspect ratio
    ///   - R0: Major radius [m]
    /// - Returns: Normalized collisionality ν*
    public func computeCollisionality(
        electronDensity: MLXArray,
        electronTemperature: MLXArray,
        safetyFactor: MLXArray,
        epsilon: MLXArray,
        R0: Float
    ) -> MLXArray {

        let normalizedCollisionality = 6.921e-18 * safetyFactor * R0 * electronDensity * effectiveCharge * coulombLogarithm
                      / (electronTemperature * electronTemperature * pow(epsilon, 1.5))

        return normalizedCollisionality
    }
}

// MARK: - Diagnostic Output

extension SauterBootstrapModel {

    /// Compute total bootstrap current
    ///
    /// - Parameters:
    ///   - profiles: Core plasma profiles
    ///   - geometry: Tokamak geometry
    ///   - q: Safety factor
    /// - Returns: Total bootstrap current [A]
    public func computeTotalCurrent(
        profiles: CoreProfiles,
        geometry: Geometry,
        safetyFactor: MLXArray
    ) -> Float {

        let j_bs = compute(profiles: profiles, geometry: geometry, safetyFactor: safetyFactor)

        // Integrate over cross-section: I_bs = Σ j_bs * A_cell
        // For toroidal geometry: A_cell = 2π r * Δr
        let geomFactors = GeometricFactors.from(geometry: geometry)
        let cellRadii = geomFactors.cellRadii.value
        let radialSpacing = geometry.radialSpacing

        // Cell area (approximate)
        let A_cell = 2.0 * Float.pi * cellRadii * radialSpacing

        let I_bs = (j_bs * A_cell).sum()

        return I_bs.item(Float.self)
    }

    /// Compute bootstrap current fraction
    ///
    /// - Parameters:
    ///   - profiles: Core plasma profiles
    ///   - geometry: Tokamak geometry
    ///   - q: Safety factor
    ///   - totalCurrent: Total plasma current [A]
    /// - Returns: Bootstrap fraction f_bs = I_bs / I_total
    public func computeBootstrapFraction(
        profiles: CoreProfiles,
        geometry: Geometry,
        safetyFactor: MLXArray,
        totalCurrent: Float
    ) -> Float {

        let I_bs = computeTotalCurrent(profiles: profiles, geometry: geometry, safetyFactor: safetyFactor)
        return I_bs / (totalCurrent + 1e-10)
    }

    /// Check collisionality regime
    ///
    /// - Parameter normalizedCollisionality: Normalized collisionality
    /// - Returns: Regime classification
    public func classifyCollisionalityRegime(normalizedCollisionality: Float) -> String {
        if normalizedCollisionality < 0.01 {
            return "Banana regime (low collisionality)"
        } else if normalizedCollisionality < 1.0 {
            return "Plateau regime (intermediate)"
        } else {
            return "Collisional regime (high)"
        }
    }
}
