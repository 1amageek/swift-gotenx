// PlasmaPhysics.swift
// Fundamental plasma physics formulas and utilities

import Foundation
import MLX

/// Collection of fundamental plasma physics formulas
///
/// **Unit System** (from docs/UNIT_SYSTEM.md):
/// - Temperature: eV (NOT keV)
/// - Density: m⁻³ (NOT 10²⁰ m⁻³)
/// - Magnetic field: Tesla
/// - Resistivity: Ω·m
public struct PlasmaPhysics {

    // MARK: - Physical Constants

    /// Elementary charge [C]
    public static let elementaryCharge: Float = 1.602e-19

    /// Electron mass [kg]
    public static let electronMass: Float = 9.109e-31

    /// Proton mass [kg]
    public static let protonMass: Float = 1.673e-27

    /// Vacuum permeability μ₀ [H/m]
    public static let vacuumPermeability: Float = 4e-7 * Float.pi

    /// Boltzmann constant [J/K]
    public static let boltzmannConstant: Float = 1.381e-23

    // MARK: - Spitzer Resistivity

    /// Compute Spitzer resistivity η = η₀ effectiveCharge ln(Λ) / T_e^(3/2)
    ///
    /// **Formula** (SI units):
    /// ```
    /// η = 5.2×10⁻⁵ × effectiveCharge × ln(Λ) / T_e^(3/2)  [Ω·m]
    /// ```
    ///
    /// **Parameters**:
    /// - Te_eV: Electron temperature [cellCount] in eV
    /// - ne_m3: Electron density [cellCount] in m⁻³
    /// - effectiveCharge: Effective charge (default: 1.0 for pure deuterium)
    ///
    /// **Returns**: Resistivity η [cellCount] in Ω·m
    ///
    /// **Units Verification**:
    /// - η₀ = 5.2×10⁻⁵ [Ω·m·eV^(3/2)]
    /// - T_e in [eV]
    /// - Result: [Ω·m·eV^(3/2)] / [eV^(3/2)] = [Ω·m] ✓
    ///
    /// **Reference**: NRL Plasma Formulary (2019), p.34
    public static func spitzerResistivity(
        Te_eV: MLXArray,
        ne_m3: MLXArray,
        effectiveCharge: Float = 1.0
    ) -> MLXArray {
        // Spitzer coefficient in SI units
        let eta0: Float = 5.2e-5  // Ω·m·eV^(3/2)

        // Coulomb logarithm
        let coulombLogarithm = coulombLogarithm(Te_eV: Te_eV, ne_m3: ne_m3)

        // Spitzer formula: η = η₀ effectiveCharge ln(Λ) / T_e^(3/2)
        let eta = eta0 * effectiveCharge * coulombLogarithm / pow(Te_eV, 1.5)
        eval(eta)
        return eta
    }

    /// Compute Coulomb logarithm ln(Λ)
    ///
    /// **Formula** (for T_e > 10 eV):
    /// ```
    /// ln(Λ) ≈ 15.2 - 0.5·ln(n_e/10²⁰) + ln(T_e[keV])
    /// ```
    ///
    /// **Parameters**:
    /// - Te_eV: Electron temperature [cellCount] in eV
    /// - ne_m3: Electron density [cellCount] in m⁻³
    ///
    /// **Returns**: Coulomb logarithm ln(Λ) [cellCount] (dimensionless)
    ///
    /// **Typical values**: 15-20 for fusion plasmas
    ///
    /// **Reference**: NRL Plasma Formulary (2019), p.34
    public static func coulombLogarithm(
        Te_eV: MLXArray,
        ne_m3: MLXArray
    ) -> MLXArray {
        // Convert to conventional units for formula
        let ne_1e20 = ne_m3 / 1e20  // Density in 10²⁰ m⁻³
        let Te_keV = Te_eV / 1000.0  // Temperature in keV

        // Coulomb logarithm formula
        let coulombLogarithm = 15.2 - 0.5 * log(ne_1e20) + log(Te_keV)

        // Clamp to reasonable range [10, 25]
        let lnLambda_clamped = clip(coulombLogarithm, min: MLXArray(10.0), max: MLXArray(25.0))
        eval(lnLambda_clamped)
        return lnLambda_clamped
    }

    /// Compute resistive diffusion time τ_R = μ₀ a² / η
    ///
    /// Characteristic time for magnetic field diffusion.
    ///
    /// **Parameters**:
    /// - eta: Resistivity [cellCount] in Ω·m
    /// - minorRadius: Plasma minor radius in meters
    ///
    /// **Returns**: Resistive time τ_R [cellCount] in seconds
    ///
    /// **Units Verification**:
    /// - μ₀: [H/m] = [Ω·s/m]
    /// - a²: [m²]
    /// - η: [Ω·m]
    /// - τ_R = [Ω·s/m]·[m²] / [Ω·m] = [s] ✓
    ///
    /// **Typical values**: 0.1 - 10 seconds for tokamaks
    public static func resistiveDiffusionTime(
        eta: MLXArray,
        minorRadius: Float
    ) -> MLXArray {
        let tau_R = vacuumPermeability * minorRadius * minorRadius / eta

        // Clamp to reasonable range [1e-6, 1e6] seconds
        let tau_R_clamped = clip(tau_R, min: MLXArray(1e-6), max: MLXArray(1e6))
        eval(tau_R_clamped)
        return tau_R_clamped
    }

    // MARK: - Plasma Beta

    /// Compute plasma beta β = 2μ₀ p / B²
    ///
    /// Ratio of plasma pressure to magnetic pressure.
    ///
    /// **Parameters**:
    /// - profiles: Core plasma profiles
    /// - magneticField: Total magnetic field B [cellCount] in Tesla
    ///
    /// **Returns**: Plasma beta β [cellCount] (dimensionless)
    ///
    /// **Units Verification**:
    /// - p = n_e × (T_e + T_i) × e [Pa]
    /// - B: [T]
    /// - β = [H/m]·[Pa] / [T²] = dimensionless ✓
    ///
    /// **Typical values**: 0.01 - 0.05 for tokamaks
    public static func plasmaBeta(
        profiles: CoreProfiles,
        magneticField: MLXArray
    ) -> MLXArray {
        // Pressure in Pascals: p = n_e × (T_e + T_i) × e
        let n_e = profiles.electronDensity.value
        let T_e = profiles.electronTemperature.value
        let T_i = profiles.ionTemperature.value

        let pressure = n_e * (T_e + T_i) * elementaryCharge  // Pa
        eval(pressure)

        // Magnetic pressure: B² / (2μ₀)
        let B_squared = magneticField * magneticField
        eval(B_squared)

        // Plasma beta: β = 2μ₀ p / B²
        let beta = (2.0 * vacuumPermeability * pressure) / B_squared

        // Clamp to prevent unphysical values
        let beta_clamped = clip(beta, min: MLXArray(1e-6), max: MLXArray(0.2))
        eval(beta_clamped)
        return beta_clamped
    }

    /// Compute total magnetic field B_total = √(B_tor² + B_pol²)
    ///
    /// **Parameters**:
    /// - toroidalField: Toroidal field B_tor (constant) in Tesla
    /// - poloidalField: Poloidal field B_pol [cellCount] in Tesla (optional)
    /// - cellCount: Number of radial cells for array shape consistency
    ///
    /// **Returns**: Total magnetic field B_total [cellCount] in Tesla
    ///
    /// **Fallback**: If B_pol not provided, returns constant B_tor array [cellCount]
    public static func totalMagneticField(
        toroidalField: Float,
        poloidalField: MLXArray?,
        cellCount: Int
    ) -> MLXArray {
        guard let B_pol = poloidalField else {
            // No poloidal field: use constant toroidal field array
            let B_total = MLXArray.full([cellCount], values: MLXArray(toroidalField))
            eval(B_total)
            return B_total
        }

        // Total field: B_total = √(B_tor² + B_pol²)
        let B_tor_squared = toroidalField * toroidalField
        let B_pol_squared = B_pol * B_pol
        let B_total = sqrt(B_tor_squared + B_pol_squared)

        eval(B_total)
        return B_total
    }

    // MARK: - Characteristic Scales

    /// Compute ion sound Larmor radius ρ_s = c_s / ω_ci
    ///
    /// Characteristic spatial scale for drift-wave turbulence.
    ///
    /// **Formula**:
    /// ```
    /// ρ_s = √(T_e / m_i) / (eB / m_i) = √(T_e × m_i) / (eB)
    /// ```
    ///
    /// **Parameters**:
    /// - Te_eV: Electron temperature [cellCount] in eV
    /// - magneticField: Magnetic field B [cellCount] in Tesla
    /// - ionMass: Ion mass in kg (default: deuterium = 2 × m_p)
    ///
    /// **Returns**: Ion sound Larmor radius ρ_s [cellCount] in meters
    ///
    /// **Typical values**: 1-5 mm for tokamaks
    public static func ionSoundLarmorRadius(
        Te_eV: MLXArray,
        magneticField: MLXArray,
        ionMass: Float = 2.0 * protonMass  // Deuterium
    ) -> MLXArray {
        // CRITICAL: Reformulate to avoid Float32 underflow
        // Standard: ρ_s = c_s / ω_ci = √(T_e / m_i) / (eB / m_i)
        // Simplify: ρ_s = √(m_i × T_e) / (e × B)
        //                = √(m_i × T_e[eV] × e) / (e × B)    [T = T_e[eV] × e]
        //                = √(m_i × T_e[eV] / e) / B          [cancel e]
        //
        // This avoids underflow: m_i × T_e[eV] / e ≈ 3e-27 × 2500 / 1.6e-19 ≈ 5e-5 ✓
        let ionMass_array = MLXArray(Float(ionMass))
        let rho_s = sqrt(ionMass_array * Te_eV / elementaryCharge) / magneticField
        eval(rho_s)
        return rho_s
    }

    /// Compute ion mass from isotope mass number
    ///
    /// **Parameters**:
    /// - massNumber: Isotope mass number (1=H, 2=D, 3=T)
    ///
    /// **Returns**: Ion mass in kg
    public static func ionMass(massNumber: Float) -> Float {
        return massNumber * protonMass
    }
}
