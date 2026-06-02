import MLX
import Foundation

// MARK: - MLX Gradient Computation

/// Vectorized gradient computation for MLX arrays
///
/// **Design Principle**: No for-loops, pure tensor operations
///
/// MLX is designed for vectorized tensor operations. Element-wise loops
/// are not compilable and defeat GPU optimization. All operations here
/// use array slicing and element-wise arithmetic.
public struct MLXGradient {
    // MARK: - Radial Gradient

    /// Compute radial gradient using 2nd-order finite differences
    ///
    /// **Scheme**:
    /// - Interior points (i = 1 to n-2): Central difference
    ///   ```
    ///   ∇f[i] = (f[i+1] - f[i-1]) / (r[i+1] - r[i-1])
    ///   ```
    /// - Left boundary (i = 0): Forward difference
    ///   ```
    ///   ∇f[0] = (f[1] - f[0]) / (r[1] - r[0])
    ///   ```
    /// - Right boundary (i = n-1): Backward difference
    ///   ```
    ///   ∇f[n-1] = (f[n-1] - f[n-2]) / (r[n-1] - r[n-2])
    ///   ```
    ///
    /// **Accuracy**: 2nd-order in interior, 1st-order at boundaries
    ///
    /// - Parameters:
    ///   - field: Field to differentiate [n]
    ///   - radii: Radial coordinates [n]
    /// - Returns: Gradient [n]
    public static func radialGradient(
        field: MLXArray,
        radii: MLXArray
    ) -> MLXArray {
        let n = field.shape[0]
        guard n >= 2 else {
            // Degenerate case: return zeros
            return MLXArray.zeros([n])
        }

        if n == 2 {
            // Only two points: use simple difference
            let df = field[1] - field[0]
            let radialSpacing = radii[1] - radii[0]
            let grad = df / radialSpacing
            return stacked([grad, grad], axis: 0)
        }

        // Interior points: central difference (i = 1 to n-2)
        // Create shifted arrays for vectorized computation
        let f_forward = field[2...]           // f[i+1]: [f[2], f[3], ..., f[n-1]]
        let f_backward = field[..<(n-2)]      // f[i-1]: [f[0], f[1], ..., f[n-3]]
        let r_forward = radii[2...]           // r[i+1]
        let r_backward = radii[..<(n-2)]      // r[i-1]

        // Central difference: (f[i+1] - f[i-1]) / (r[i+1] - r[i-1])
        let df_interior = f_forward - f_backward
        let dr_interior = r_forward - r_backward
        let grad_interior = df_interior / dr_interior

        // Left boundary: forward difference
        let df_left = field[1] - field[0]
        let dr_left = radii[1] - radii[0]
        let grad_left = df_left / dr_left

        // Right boundary: backward difference
        let df_right = field[n-1] - field[n-2]
        let dr_right = radii[n-1] - radii[n-2]
        let grad_right = df_right / dr_right

        // Concatenate: [left, interior, right]
        return concatenated([
            grad_left.reshaped([1]),
            grad_interior,
            grad_right.reshaped([1])
        ], axis: 0)
    }

    // MARK: - Normalized Gradient

    /// Compute normalized gradient: L/L_T = -L(∇T/T)
    ///
    /// **General Formula**: `L/L_T = -L * (dT/radialSpacing) / T`
    ///
    /// where L is an arbitrary normalization length scale:
    /// - L = a (minor radius): gives a/L_T (TORAX convention)
    /// - L = R (major radius): gives R/L_T (QLKNN convention)
    ///
    /// **Physical Interpretation**:
    /// - L/L_T > 0: Temperature decreases outward (normal)
    /// - L/L_T = 2: Gaussian profile (T ∝ exp(-r²/2σ²) with σ = L/2)
    /// - L/L_T > 10: Very steep gradient (ITG unstable)
    ///
    /// **Usage Examples**:
    /// ```swift
    /// // Gotenx-style: a/L_T
    /// let aLT = normalizedGradient(profile: T, radii: r, normalizationLength: minorRadius)
    ///
    /// // QLKNN-style: R/L_T
    /// let RLT = normalizedGradient(profile: T, radii: r, normalizationLength: majorRadius)
    /// ```
    ///
    /// - Parameters:
    ///   - profile: Temperature or density profile [n]
    ///   - radii: Radial coordinates [n]
    ///   - normalizationLength: Length scale for normalization (e.g., a or R) [m]
    /// - Returns: Normalized gradient L/L_T [n]
    public static func normalizedGradient(
        profile: MLXArray,
        radii: MLXArray,
        normalizationLength: Float
    ) -> MLXArray {
        let gradient = radialGradient(field: profile, radii: radii)
        return -(normalizationLength * gradient) / profile
    }

    // MARK: - Magnetic Shear

    /// Compute magnetic shear: s = (r/q)(dq/radialSpacing)
    ///
    /// **Physical Interpretation**:
    /// - s > 0: Positive shear (q increases with r, stabilizing)
    /// - s < 0: Negative shear (q decreases with r, destabilizing)
    /// - s = 0: Shearless (special configuration)
    ///
    /// Typical tokamak: s ≈ 0.5 - 3.0
    ///
    /// - Parameters:
    ///   - q: Safety factor profile [n]
    ///   - radii: Radial coordinates [n]
    /// - Returns: Magnetic shear s [n]
    public static func magneticShear(
        q: MLXArray,
        radii: MLXArray
    ) -> MLXArray {
        let dqdr = radialGradient(field: q, radii: radii)
        return (radii / q) * dqdr
    }

    // MARK: - Inverse Aspect Ratio

    /// Compute inverse aspect ratio: x = r/R
    ///
    /// **Physical Interpretation**:
    /// - x = 0: On magnetic axis
    /// - x < 0.3: Core region
    /// - x = 0.5: Mid-radius (typical for tokamaks)
    /// - x > 0.7: Edge region
    ///
    /// - Parameters:
    ///   - radii: Minor radius coordinates [n]
    ///   - majorRadius: Major radius R [m]
    /// - Returns: Inverse aspect ratio x [n]
    public static func inverseAspectRatio(
        radii: MLXArray,
        majorRadius: Float
    ) -> MLXArray {
        return radii / majorRadius
    }

    // MARK: - Collisionality

    /// Compute electron-ion collisionality: ν*
    ///
    /// **Formula**:
    /// ```
    /// ν* = (q R / ε^1.5) * C * effectiveCharge * ne * log(Λ) / Te^2
    /// ```
    ///
    /// where:
    /// - C = 6.92×10^-15 [m³ eV² / s] (collision frequency constant)
    /// - log(Λ) ≈ 15.2 - 0.5*log(ne/10²⁰) + log(Te/1000) (Coulomb logarithm)
    /// - ε = r/R (inverse aspect ratio)
    ///
    /// **Physical Interpretation**:
    /// - ν* < 0.01: Collisionless regime (banana orbits dominate)
    /// - ν* ≈ 1: Plateau regime (transitional)
    /// - ν* > 10: Collisional regime (Pfirsch-Schlüter transport)
    ///
    /// **Units**:
    /// - electronDensity: [m^-3]
    /// - electronTemperature: [eV]
    /// - ionTemperature: [eV] (not used in this simplified formula)
    /// - Output: log10(ν*) (dimensionless)
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m^-3]
    ///   - electronTemperature: Electron temperature [eV]
    ///   - q: Safety factor profile
    ///   - radii: Minor radius coordinates [m]
    ///   - majorRadius: Major radius R [m]
    ///   - effectiveCharge: Effective charge (default: 1.0 for pure deuterium)
    /// - Returns: log10(ν*) [dimensionless]
    public static func collisionality(
        electronDensity: MLXArray,
        electronTemperature: MLXArray,
        q: MLXArray,
        radii: MLXArray,
        majorRadius: Float,
        effectiveCharge: Float = 1.0
    ) -> MLXArray {
        // Inverse aspect ratio
        let epsilon = radii / majorRadius

        // Coulomb logarithm: log(Λ) ≈ 15.2 - 0.5*log(ne/1e20) + log(Te/1000)
        let logLambda = Float(15.2) - Float(0.5) * log(electronDensity / Float(1e20)) + log(electronTemperature / Float(1000.0))

        // Collision frequency constant [m³ eV² / s]
        let C: Float = 6.92e-15

        // ν* = (q R / ε^1.5) * C * effectiveCharge * ne * log(Λ) / Te²
        let nuStar = (q * majorRadius / pow(epsilon, Float(1.5)))
                   * C * effectiveCharge * electronDensity * logLambda / pow(electronTemperature, Float(2))

        // Return log10(ν*) for QLKNN input
        return log10(nuStar)
    }

    // MARK: - Temperature Ratio

    /// Compute temperature ratio: Ti/Te
    ///
    /// **Typical Values**:
    /// - Ti/Te ≈ 1.0: Equal ion and electron temperatures (typical)
    /// - Ti/Te > 1.5: Ion-dominated heating (NBI, ICRH)
    /// - Ti/Te < 0.8: Electron-dominated heating (ECRH, Ohmic)
    ///
    /// - Parameters:
    ///   - ionTemperature: Ion temperature [eV]
    ///   - electronTemperature: Electron temperature [eV]
    /// - Returns: Ti/Te ratio [dimensionless]
    public static func temperatureRatio(
        ionTemperature: MLXArray,
        electronTemperature: MLXArray
    ) -> MLXArray {
        return ionTemperature / electronTemperature
    }

    // MARK: - Density Ratio

    /// Compute normalized ion density: ni/ne
    ///
    /// For quasi-neutrality with single ion species: ni/ne ≈ 1.0
    ///
    /// For impurities: ni/ne < 1.0 (dilution effect)
    ///
    /// - Parameters:
    ///   - ionDensity: Ion density [m^-3]
    ///   - electronDensity: Electron density [m^-3]
    /// - Returns: ni/ne ratio [dimensionless]
    public static func densityRatio(
        ionDensity: MLXArray,
        electronDensity: MLXArray
    ) -> MLXArray {
        return ionDensity / electronDensity
    }
}
