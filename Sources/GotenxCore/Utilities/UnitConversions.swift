import Foundation
import MLX

/// Unit conversion utilities for Gotenx
///
/// This module provides essential unit conversions for the eV/m⁻³ unit system
/// used throughout TORAX. See UNIT_SYSTEM_UNIFIED.md for unit standardization details.
public enum UnitConversions {

    // MARK: - Fundamental Constants

    /// Elementary charge / electron volt conversion [J/eV]
    /// Used for converting between Joules and electron volts
    public static let electronVolt: Float = 1.602176634e-19

    // MARK: - Power Density Unit Conversions for Temperature Equations

    /// Convert power density from MW/m³ to eV/(m³·s)
    ///
    /// This conversion is **essential** for temperature equations in non-conservation form:
    /// ```
    /// n_e ∂T_i/∂t = ∇·(n_e χ_i ∇T_i) + Q_i
    /// ```
    ///
    /// **Dimensional Analysis**:
    /// - Left side: [m⁻³] × [eV/s] = [eV/(m³·s)]
    /// - Diffusion term: ∇·([m⁻³] × [m²/s] × [eV/m]) = [eV/(m³·s)]
    /// - Source term Q_i must also be [eV/(m³·s)]
    ///
    /// **Conversion derivation**:
    /// ```
    /// 1 MW/m³ = 10⁶ W/m³
    ///         = 10⁶ J/(m³·s)
    ///         = 10⁶ J/(m³·s) × (1 eV / 1.602176634×10⁻¹⁹ J)
    ///         = 6.2415090744×10²⁴ eV/(m³·s)
    /// ```
    ///
    /// **Physical Interpretation**:
    /// - SourceTerms.swift provides heating in [MW/m³] (standard for plasma physics)
    /// - Temperature equations require [eV/(m³·s)] to match time derivative
    /// - Without this conversion, heating power would be off by a factor of 6.24×10²⁴
    ///
    /// **References**:
    /// - PHYSICS_VALIDATION_ISSUES.md Issue 1 (ソース項の単位変換)
    /// - UNIT_SYSTEM_UNIFIED.md (eV/m⁻³ unit standardization)
    public static let megawattsPerCubicMeterToElectronVoltsPerCubicMeterPerSecond: Float = 6.2415090744e24

    /// Convert power density from MW/m³ to eV/(m³·s) (scalar version)
    ///
    /// Use this when converting heating source terms for temperature equations.
    ///
    /// **Example**:
    /// ```swift
    /// let Q_MW: Float = 1.0  // [MW/m³] - heating power density
    /// let Q_eV = UnitConversions.megawattsToElectronVoltDensity(Q_MW)  // [eV/(m³·s)]
    /// ```
    ///
    /// **Unit validation**:
    /// ```swift
    /// // For Q_i [MW/m³] in temperature equation:
    /// //   n_e ∂T_i/∂t = ∇·(n_e χ_i ∇T_i) + Q_i
    /// //
    /// // All terms must have dimension [eV/(m³·s)]:
    /// //   Left side:  [m⁻³] × [eV/s] = [eV/(m³·s)] ✓
    /// //   Diffusion:  ∇·([m⁻³ × m²/s × eV/m]) = [eV/(m³·s)] ✓
    /// //   Source:     [MW/m³] → [eV/(m³·s)] via this function ✓
    /// ```
    ///
    /// - Parameter megawatts: Power density in [MW/m³]
    /// - Returns: Power density in [eV/(m³·s)]
    public static func megawattsToElectronVoltDensity(_ megawatts: Float) -> Float {
        return megawatts * megawattsPerCubicMeterToElectronVoltsPerCubicMeterPerSecond
    }

    /// Convert power density from MW/m³ to eV/(m³·s) (array version)
    ///
    /// Use this when converting heating source terms for temperature equations.
    ///
    /// **Example**:
    /// ```swift
    /// // In Block1DCoeffsBuilder.swift:
    /// let Q_MW = sources.ionHeating.value  // [MW/m³]
    /// let Q_eV = UnitConversions.megawattsToElectronVoltDensity(Q_MW)  // [eV/(m³·s)]
    /// // Result is wrapped in EvaluatedArray, which calls eval() automatically
    /// ```
    ///
    /// **Why this conversion is critical**:
    /// Without this conversion, the temperature equation would have inconsistent dimensions:
    /// - Left side would expect [eV/(m³·s)]
    /// - Source term would provide [MW/m³] (incompatible!)
    /// - Simulation would produce physically meaningless results
    ///
    /// **Implementation note**:
    /// This conversion is applied in:
    /// - `buildIonEquationCoeffs()` for ion heating sources
    /// - `buildElectronEquationCoeffs()` for electron heating sources
    ///
    /// **Implementation strategy**:
    /// - Uses MLX Float32 operations (GPU-compatible)
    /// - Apple Silicon GPU does NOT support Float64
    /// - **Returns Float32 MLXArray** on same device as input
    ///
    /// **Rationale for Float32**:
    /// - Input: Q_MW ~ 0.1-10 MW/m³
    /// - Output: Q_eV ~ 10²³-10²⁵ eV/(m³·s)
    /// - Float32 max: ~3.4×10³⁸ (sufficient for 10²⁵)
    /// - Float32 precision: 7 significant digits
    /// - For value ~6×10²⁴, relative precision: ~10⁻⁷ (0.00001%)
    /// - This is **acceptable** for physics simulation (solver tolerance ~10⁻⁵)
    ///
    /// **Performance**:
    /// - All operations stay on GPU (no CPU transfer)
    /// - Typical array size: 25-100 elements (radial cells)
    /// - Overhead: < 0.1ms, negligible compared to solver (10-100ms per timestep)
    ///
    /// - Parameter megawatts: Power density array in [MW/m³]
    /// - Returns: Power density array in [eV/(m³·s)] as **Float32 MLXArray**
    public static func megawattsToElectronVoltDensity(_ megawatts: MLXArray) -> MLXArray {
        // **CRITICAL**: Apple Silicon GPU does NOT support Float64
        //
        // Problem: Converting MW/m³ to eV/(m³·s) involves large coefficient (~6.24×10²⁴)
        // - Input: 0.1-10 MW/m³
        // - Output: 6×10²³ - 6×10²⁵ eV/(m³·s)
        //
        // Float32 precision analysis:
        // - Float32 max: ~3.4×10³⁸ (sufficient for 10²⁵)
        // - Float32 precision: 7 significant digits
        // - For value 6×10²⁴: precision is ~6×10¹⁷ (absolute error)
        // - Relative error: ~10⁻⁷ (0.00001%) - **acceptable for physics simulation**
        //
        // Solution: Use Float32 arithmetic throughout (GPU-compatible)
        // - All MLX operations stay on GPU
        // - No Float64 → no CPU fallback
        // - Physics accuracy maintained (relative error < 10⁻⁶ is negligible)
        //
        // Alternative (if higher precision needed):
        // - Move to CPU, use Float64, move back to GPU (expensive)
        // - Not justified for this use case (solver tolerance is ~10⁻⁵)

        // Conversion coefficient as Float32
        let coefficient = Float(megawattsPerCubicMeterToElectronVoltsPerCubicMeterPerSecond)

        // Multiply using MLX operations (stays on same device as input)
        let coeffArray = MLXArray(coefficient)
        return megawatts * coeffArray
    }
}
