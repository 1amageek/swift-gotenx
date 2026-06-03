import MLX
import Foundation
import GotenxCore

// MARK: - Impurity Radiation Model

/// Impurity radiation model using ADAS polynomial approximation
///
/// **Physics**: Line radiation from impurity ions
/// - Process: Electron-ion collisional excitation + radiative de-excitation
/// - Effect: Power loss (cooling) from plasma
/// - Species: C, Ne, Ar, W (various charge states)
///
/// **Key Features**:
/// - Temperature-dependent radiation function L_z(T_e)
/// - ADAS database polynomial fits
/// - Coronal equilibrium assumption
///
/// **Model**: ADAS polynomial approximation
/// - Real codes use full ADAS database
/// - This model uses polynomial fits from Mavrin (2018):
///   - log₁₀(L_z) = Σ c_i × [log₁₀(T_e)]^i
///   - Valid range: **0.1 keV < T_e < 100 keV** (100 eV ~ 100,000 eV)
///
/// **Reference**:
/// - Mavrin et al. (2018) - Polynomial fits to ADAS data
/// - Summers, H.P., et al., Plasma Phys. Control. Fusion 48, 263 (2006)
/// - ADAS database: https://open.adas.ac.uk
/// - TORAX implementation: google-deepmind/torax
///
/// **Units**:
/// - Input: T_e [eV], n_e [m⁻³], n_imp [m⁻³]
/// - L_z: Radiation coefficient [W⋅m³]
/// - Output: P_rad [W/m³]
public struct ImpurityRadiationModel: Sendable {
    // MARK: - Properties

    /// Impurity fraction (n_imp / n_e)
    /// Typical range: 1e-4 - 1e-2
    public let impurityFraction: Float

    /// Impurity species
    public let species: ImpuritySpecies

    // MARK: - Impurity Species

    /// Supported impurity species with ADAS polynomial coefficients
    public enum ImpuritySpecies: String, Sendable, Codable {
        case carbon   // C (Z=6)
        case neon     // Ne (Z=10)
        case argon    // Ar (Z=18)
        case tungsten // W (Z=74)

        /// ADAS Mavrin (2018) polynomial coefficients for log₁₀(L_z) vs log₁₀(T_e)
        ///
        /// **Source**: TORAX implementation (google-deepmind/torax)
        /// **Format**: Array of coefficient sets, each set is [c₄, c₃, c₂, c₁, c₀]
        /// **Polynomial**: log₁₀(L_z) = c₄⋅X⁴ + c₃⋅X³ + c₂⋅X² + c₁⋅X + c₀
        ///   where X = log₁₀(T_e[eV])
        ///
        /// **Temperature Intervals**: Different coefficients used for different T_e ranges
        /// **Valid Range**: 0.1 - 100 keV (100 - 100,000 eV)
        /// **Units**: L_z [W⋅m³]
        var mavrinCoefficients: [[Float]] {
            switch self {
            case .carbon:
                // Carbon (Z=6) - 2 temperature intervals
                return [
                    [-7.2904e00, -1.6637e01, -1.2788e01, -5.0085e00, -3.4738e01],
                    [4.4470e-02, -2.9191e-01, 6.8856e-01, -3.6687e-01, -3.4174e01]
                ]

            case .neon:
                // Neon (Z=10) - 3 temperature intervals
                return [
                    [1.5648e01, 2.8939e01, 1.5230e01, 1.7309e00, -3.3132e01],
                    [1.7244e-01, -3.9544e-01, 8.6842e-01, -8.7750e-01, -3.3290e01],
                    [-2.6930e-02, 4.3960e-02, 2.9731e-01, -4.5345e-01, -3.3410e01]
                ]

            case .argon:
                // Argon (Z=18) - 3 temperature intervals
                return [
                    [1.5353e01, 3.9161e01, 3.0769e01, 6.5221e00, -3.2155e01],
                    [4.9806e00, -7.6887e00, 1.5389e00, 5.4490e-01, -3.2530e01],
                    [-8.2260e-02, 1.7480e-01, 6.1339e-01, -1.6674e00, -3.1853e01]
                ]

            case .tungsten:
                // Tungsten (Z=74) - 3 temperature intervals
                return [
                    [-1.0103e-01, -1.0311e00, -9.5126e-01, 3.8304e-01, -3.0374e01],
                    [5.1849e01, -6.3303e01, 2.2824e01, -2.9208e00, -3.0238e01],
                    [-3.6759e-01, 2.6627e00, -6.2740e00, 5.2499e00, -3.2153e01]
                ]
            }
        }

        /// Temperature interval boundaries [keV]
        ///
        /// These define which coefficient set to use:
        /// - T_e < intervals[0]: Use coefficients[0]
        /// - intervals[0] <= T_e < intervals[1]: Use coefficients[1]
        /// - etc.
        var temperatureIntervals: [Float] {
            switch self {
            case .carbon:
                return [0.5]  // keV
            case .neon:
                return [0.7, 5.0]  // keV
            case .argon:
                return [0.6, 3.0]  // keV
            case .tungsten:
                return [1.5, 4.0]  // keV
            }
        }

        /// Atomic number
        var atomicNumber: Int {
            switch self {
            case .carbon: return 6
            case .neon: return 10
            case .argon: return 18
            case .tungsten: return 74
            }
        }
    }

    // MARK: - Initialization

    /// Initialize impurity radiation model
    ///
    /// - Parameters:
    ///   - impurityFraction: Impurity fraction (n_imp / n_e)
    ///   - species: Impurity species
    public init(
        impurityFraction: Float,
        species: ImpuritySpecies = .argon
    ) {
        self.impurityFraction = impurityFraction
        self.species = species
    }

    // MARK: - Radiation Calculation

    /// Compute radiation coefficient L_z(T_e) using ADAS polynomial
    ///
    /// **Model**: Polynomial fit to ADAS database (Mavrin 2018)
    /// ```
    /// log₁₀(L_z) = Σ c_i × [log₁₀(T_e)]^i
    /// L_z = 10^(polynomial)
    /// ```
    ///
    /// **Temperature Range**: 0.1 - 100 keV (100 - 100,000 eV)
    /// - Values outside this range are clamped
    /// - Extrapolation beyond validity range may be unreliable
    ///
    /// - Parameter electronTemperature: Electron temperature [eV]
    /// - Returns: Radiation coefficient L_z [W⋅m³]
    private func computeRadiationCoefficient(
        electronTemperature: MLXArray,
        emitDebugWarning: Bool = true
    ) -> MLXArray {
        // Clamp temperature to valid range [0.1 keV, 100 keV] = [100 eV, 100,000 eV]
        // This matches TORAX implementation (Mavrin 2018 polynomial validity range)
        let Te_clamped = MLX.clip(electronTemperature, min: 100.0, max: 100000.0)

        // Diagnostic warning for values outside validity range
        #if DEBUG
        if emitDebugWarning {
            let teRange = MLX.stacked([
                electronTemperature.min(),
                electronTemperature.max()
            ], axis: 0).asArray(Float.self)
            if teRange[0] < 100.0 || teRange[1] > 100000.0 {
                print("Warning: T_e outside ADAS validity range [\(teRange[0]), \(teRange[1])] eV")
                print("   Valid range: [100, 100000] eV (0.1 - 100 keV)")
                print("   Polynomial extrapolation may be unreliable")
            }
        }
        #endif

        // Convert temperature to keV for interval selection
        let Te_keV = Te_clamped / 1000.0  // eV → keV

        // Select appropriate coefficient set based on temperature intervals
        let coefficientSets = species.mavrinCoefficients
        let intervals = species.temperatureIntervals

        // Find which interval each temperature falls into
        // This implements searchsorted from TORAX
        var log10_Lz = MLXArray.zeros(electronTemperature.shape)

        for (idx, coeffs) in coefficientSets.enumerated() {
            // Create mask for this temperature interval
            let mask: MLXArray
            if idx == 0 {
                // First interval: T_e < intervals[0]
                if intervals.count > 0 {
                    mask = Te_keV .< intervals[0]
                } else {
                    // Only one coefficient set, use for all temperatures
                    mask = MLXArray.ones(electronTemperature.shape, dtype: .bool)
                }
            } else if idx < intervals.count {
                // Middle intervals: intervals[idx-1] <= T_e < intervals[idx]
                mask = (Te_keV .>= intervals[idx-1]) .&& (Te_keV .< intervals[idx])
            } else {
                // Last interval: T_e >= intervals[last]
                mask = Te_keV .>= intervals[idx-1]
            }

            // Compute log₁₀(L_z) for this interval using 4th-order polynomial
            // Polynomial: log₁₀(L_z) = c₄⋅X⁴ + c₃⋅X³ + c₂⋅X² + c₁⋅X + c₀
            // Coefficients are ordered: [c₄, c₃, c₂, c₁, c₀]
            let X = log10(Te_clamped)  // X = log₁₀(T_e[eV])

            let c4 = coeffs[0]
            let c3 = coeffs[1]
            let c2 = coeffs[2]
            let c1 = coeffs[3]
            let c0 = coeffs[4]

            // Horner's method for numerical stability:
            // ((((c₄⋅X + c₃)⋅X + c₂)⋅X + c₁)⋅X + c₀)
            var poly = MLXArray(c4)
            poly = poly * X + c3
            poly = poly * X + c2
            poly = poly * X + c1
            poly = poly * X + c0

            // Apply this polynomial result where mask is true
            log10_Lz = MLX.where(mask, poly, log10_Lz)
        }

        // L_z = 10^(log₁₀(L_z))
        let Lz = pow(10.0, log10_Lz)

        // Clamp to prevent numerical overflow
        // Typical range: 10^-32 to 10^-30 W·m³ for impurity radiation
        // Allow up to 10^-25 for safety margin
        let Lz_clamped = MLX.clip(Lz, min: 1e-35, max: 1e-25)

        return Lz_clamped
    }

    /// Compute impurity radiation power density
    ///
    /// **Model**: P_rad = -n_e × n_imp × L_z(T_e)
    /// ```
    /// n_imp = f_imp × n_e
    /// P_rad = -n_e² × f_imp × L_z(T_e) [W/m³]
    /// ```
    ///
    /// **Note**: Returns NEGATIVE value (power loss convention)
    ///
    /// - Parameters:
    ///   - electronDensity: Electron density [m⁻³]
    ///   - electronTemperature: Electron temperature [eV]
    /// - Returns: Radiation power loss [W/m³] (NEGATIVE value)
    public func compute(electronDensity: MLXArray, electronTemperature: MLXArray) -> MLXArray {
        compute(
            electronDensity: electronDensity,
            electronTemperature: electronTemperature,
            emitsDebugWarning: true
        )
    }

    package func compute(
        electronDensity: MLXArray,
        electronTemperature: MLXArray,
        emitsDebugWarning: Bool
    ) -> MLXArray {
        let radiationCoefficient = computeRadiationCoefficient(
            electronTemperature: electronTemperature,
            emitDebugWarning: emitsDebugWarning
        )
        let impurityDensity = impurityFraction * electronDensity

        return -(electronDensity * impurityDensity * radiationCoefficient)
    }

    // MARK: - Apply to Source Terms

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

    /// Apply impurity radiation to source terms
    ///
    /// **Important**: Radiation is a LOSS (negative heating)
    /// - compute() returns NEGATIVE values (power loss convention)
    /// - We ADD this negative value to electron heating (matches Bremsstrahlung pattern)
    /// - Does NOT affect ion heating (ions cooled via e-i exchange)
    ///
    /// **Units**:
    /// - Input: Computes W/m³
    /// - Output: Returns MW/m³ (SourceTerms convention)
    ///
    /// - Parameters:
    ///   - sources: Existing source terms
    ///   - profiles: Current core profiles
    /// - Returns: Updated source terms with radiation loss
    public func applyToSources(
        _ sources: SourceTerms,
        profiles: CoreProfiles
    ) throws -> SourceTerms {
        applyToSources(
            sources,
            profiles: profiles,
            geometricFactors: nil,
            evaluationMode: .eager,
            includesMetadata: false,
            validateDebugUnits: true,
            emitsDebugWarning: true
        )
    }

    package func applyToSources(
        _ sources: SourceTerms,
        profiles: CoreProfiles,
        context: SourceEvaluationContext
    ) -> SourceTerms {
        applyToSources(
            sources,
            profiles: profiles,
            geometricFactors: context.geometricFactors,
            evaluationMode: context.evaluationMode,
            includesMetadata: context.includesMetadata,
            validateDebugUnits: context.validatesDebugUnits,
            emitsDebugWarning: context.includesMetadata
        )
    }

    package func applyToSources(
        _ sources: SourceTerms,
        profiles: CoreProfiles,
        geometricFactors: GeometricFactors?,
        evaluationMode: MLXEvaluationMode,
        includesMetadata: Bool,
        validateDebugUnits: Bool,
        emitsDebugWarning: Bool
    ) -> SourceTerms {
        let electronDensity = profiles.electronDensity.value
        let electronTemperature = profiles.electronTemperature.value

        // Compute radiation power loss [W/m³] (returns NEGATIVE value)
        let P_rad_watts = compute(
            electronDensity: electronDensity,
            electronTemperature: electronTemperature,
            emitsDebugWarning: emitsDebugWarning
        )

        // Convert to MW/m³ for SourceTerms
        let P_rad_MW = PhysicsConstants.wattsToMegawatts(P_rad_watts)

        // Add radiation loss (negative value) to electron heating
        let updated_electron = sources.electronHeating.value + P_rad_MW

        let metadata: SourceMetadataCollection?
        if includesMetadata, let geometricFactors {
            let cellVolumes = geometricFactors.cellVolumes.value
            let P_rad_total = (P_rad_watts * cellVolumes).sum()
            eval(P_rad_total)

            let radiationMetadata = SourceMetadata(
                modelName: "impurity_radiation",
                category: .radiation,
                ionPower: 0,
                electronPower: P_rad_total.item(Float.self)
            )
            metadata = Self.metadataCollection(
                existing: sources.metadata,
                appending: radiationMetadata
            )
        } else {
            metadata = sources.metadata
        }

        return SourceTerms(
            ionHeating: sources.ionHeating,
            electronHeating: evaluationMode.wrap(updated_electron),
            particleSource: sources.particleSource,
            currentSource: sources.currentSource,
            metadata: metadata,
            validateDebugUnits: validateDebugUnits
        )
    }

    /// Compute source metadata for power balance tracking
    ///
    /// - Parameters:
    ///   - profiles: Current plasma profiles
    ///   - geometry: Geometry for volume integration
    /// - Returns: Source metadata with impurity radiation power (negative)
    public func computeMetadata(
        profiles: CoreProfiles,
        geometry: Geometry
    ) -> SourceMetadata {

        let electronDensity = profiles.electronDensity.value
        let electronTemperature = profiles.electronTemperature.value

        // Compute radiation power loss [W/m³] (returns NEGATIVE value)
        let P_rad_watts = compute(electronDensity: electronDensity, electronTemperature: electronTemperature)

        // Volume integration: ∫ P dV → [W/m³] × [m³] = [W]
        let cellVolumes = GeometricFactors.from(geometry: geometry).cellVolumes.value
        let P_rad_total = (P_rad_watts * cellVolumes).sum()
        eval(P_rad_total)

        let radPower = P_rad_total.item(Float.self)

        // Impurity radiation is a power loss (negative value)
        return SourceMetadata(
            modelName: "impurity_radiation",
            category: .radiation,
            ionPower: 0,  // Only affects electrons
            electronPower: radPower  // Already negative from compute()
        )
    }
}

// MARK: - Impurity Radiation Error

public enum ImpurityRadiationError: Error, CustomStringConvertible {
    case negativeImpurityFraction(Float)
    case excessiveImpurityFraction(Float)
    case unknownSpecies(String)

    public var description: String {
        switch self {
        case .negativeImpurityFraction(let fraction):
            return "Impurity Radiation: Negative impurity fraction \(fraction). Must be non-negative."
        case .excessiveImpurityFraction(let fraction):
            return "Impurity Radiation: Excessive impurity fraction \(fraction). Must be < 0.1 (10%)."
        case .unknownSpecies(let name):
            return "Impurity Radiation: Unknown species '\(name)'. Valid: carbon, neon, argon, tungsten."
        }
    }
}
