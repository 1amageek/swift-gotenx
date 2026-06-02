import Foundation
import MLX

/// Physical constants for plasma physics calculations
///
/// All constants use SI units unless otherwise specified.
public enum PhysicsConstants {

    // MARK: - Fundamental Constants

    /// Elementary charge [C]
    public static let elementaryCharge: Float = 1.602176634e-19

    /// Electron volt to Joules conversion [J/eV]
    public static let electronVolt: Float = 1.602176634e-19

    /// Electron mass [kg]
    public static let electronMass: Float = 9.1093837015e-31

    /// Proton mass [kg]
    public static let protonMass: Float = 1.67262192369e-27

    /// Atomic mass unit [kg]
    public static let atomicMassUnit: Float = 1.66053906660e-27

    /// Boltzmann constant [J/K]
    public static let boltzmann: Float = 1.380649e-23

    /// Speed of light [m/s]
    public static let speedOfLight: Float = 2.99792458e8

    /// Vacuum permittivity [F/m]
    public static let epsilon0: Float = 8.8541878128e-12

    /// Vacuum permeability [H/m]
    public static let mu0: Float = 1.25663706212e-6

    // MARK: - Plasma-Specific Constants

    /// Bremsstrahlung radiation coefficient [W·m³·eV^(-1/2)]
    public static let bremsCoefficient: Float = 5.35e-37

    /// Rest mass energy of electron [eV]
    public static let electronRestMass: Float = 510998.95  // m_e c² in eV

    /// Spitzer resistivity prefactor [Ω·m·eV^(3/2)]
    public static let spitzerPrefactor: Float = 5.2e-5

    /// Collisional frequency prefactor [Hz·m³·eV^(-3/2)]
    public static let collisionFrequencyPrefactor: Float = 2.91e-6

    // MARK: - Common Ion Masses

    /// Deuterium mass [atomicMassUnits]
    public static let deuteriumMass: Float = 2.014

    /// Tritium mass [atomicMassUnits]
    public static let tritiumMass: Float = 3.016

    /// Helium-4 mass [atomicMassUnits]
    public static let helium4Mass: Float = 4.003

    // MARK: - Fusion-Specific Constants

    /// D-T fusion alpha particle energy [MeV]
    public static let deuteriumTritiumAlphaEnergy: Float = 3.5

    /// D-T fusion neutron energy [MeV]
    public static let deuteriumTritiumNeutronEnergy: Float = 14.1

    /// D-T fusion Q-value (total energy release) [MeV]
    public static let deuteriumTritiumQValue: Float = 17.6

    // MARK: - Unit Conversions

    /// Convert eV to Joules
    public static func electronVoltsToJoules(_ electronVolts: Float) -> Float {
        return electronVolts * PhysicsConstants.electronVolt
    }

    /// Convert Joules to eV
    public static func joulesToElectronVolts(_ joules: Float) -> Float {
        return joules / PhysicsConstants.electronVolt
    }

    /// Convert keV to eV
    public static func kiloelectronVoltsToElectronVolts(_ kiloelectronVolts: Float) -> Float {
        return kiloelectronVolts * 1e3
    }

    /// Convert eV to keV
    public static func electronVoltsToKiloelectronVolts(_ electronVolts: Float) -> Float {
        return electronVolts / 1e3
    }

    /// Convert MeV to Joules
    public static func megaelectronVoltsToJoules(_ megaelectronVolts: Float) -> Float {
        return megaelectronVolts * 1e6 * PhysicsConstants.electronVolt
    }

    /// Convert atomic mass units to kg
    public static func atomicMassUnitsToKilograms(_ atomicMassUnits: Float) -> Float {
        return atomicMassUnits * atomicMassUnit
    }

    // MARK: - Power Unit Conversions

    /// Convert W/m³ to MW/m³
    public static func wattsToMegawatts(_ watts: Float) -> Float {
        return watts / 1e6
    }

    /// Convert MW/m³ to W/m³
    public static func megawattsToWatts(_ megawatts: Float) -> Float {
        return megawatts * 1e6
    }

    /// Convert W/m³ array to MW/m³ array
    public static func wattsToMegawatts(_ watts: MLXArray) -> MLXArray {
        return watts / 1e6
    }

    /// Convert MW/m³ array to W/m³ array
    public static func megawattsToWatts(_ megawatts: MLXArray) -> MLXArray {
        return megawatts * 1e6
    }

    // MARK: - Power Density Unit Conversions for Temperature Equations

    /// Convert power density from MW/m³ to eV/(m³·s)
    ///
    /// This conversion is essential for temperature equations in non-conservation form:
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
    /// - Parameter megawatts: Power density in MW/m³
    /// - Returns: Power density in eV/(m³·s)
    public static let megawattsPerCubicMeterToElectronVoltsPerCubicMeterPerSecond: Float = 6.2415090744e24

    /// Convert power density from MW/m³ to eV/(m³·s) (scalar version)
    ///
    /// Use this when converting heating source terms for temperature equations.
    ///
    /// **Example**:
    /// ```swift
    /// let Q_MW: Float = 1.0  // [MW/m³] - heating power density
    /// let Q_eV = PhysicsConstants.megawattsToElectronVoltDensity(Q_MW)  // [eV/(m³·s)]
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
    /// let Q_eV = PhysicsConstants.megawattsToElectronVoltDensity(Q_MW)  // [eV/(m³·s)]
    /// ```
    ///
    /// - Parameter megawatts: Power density array in [MW/m³]
    /// - Returns: Power density array in [eV/(m³·s)]
    public static func megawattsToElectronVoltDensity(_ megawatts: MLXArray) -> MLXArray {
        return megawatts * megawattsPerCubicMeterToElectronVoltsPerCubicMeterPerSecond
    }
}
