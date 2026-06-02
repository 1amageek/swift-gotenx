import MLX
import Foundation
import GotenxCore

// MARK: - ECRH Model

/// Electron Cyclotron Resonance Heating (ECRH) model
///
/// **Physics**: Microwave heating at electron cyclotron frequency
/// - Frequency: ~140-170 GHz for ITER (B_T ~ 5.3 T)
/// - Power deposition: Highly localized (Gaussian profile)
/// - Current drive: Optional ECCD (Electron Cyclotron Current Drive)
///
/// **Key Features**:
/// - Localized power deposition at specified radius
/// - No fuel dilution (unlike NBI)
/// - Can target specific flux surfaces (ray tracing simplification)
///
/// **Model**: Simplified Gaussian deposition (Lin-Liu approximation)
/// - Real codes (TORBEAM, TRAVIS) use ray tracing
/// - This model assumes:
///   - Power deposits at ρ_dep with width Δρ_dep
///   - Gaussian profile: exp(-(ρ - ρ_dep)² / (2σ²))
///   - Toroidal mode focusing (implicit in width)
///
/// **Reference**:
/// - Lin-Liu, Y.R., et al., Phys. Plasmas 10, 4064 (2003)
/// - ITER Physics Basis, Nucl. Fusion 39, 2495 (1999)
///
/// **Units**:
/// - Input: totalPower [W]
/// - Output: Power density [W/m³] → converted to [MW/m³] in applyToSources()
public struct ECRHModel: Sendable {
    // MARK: - Properties

    /// Total injected power [W]
    public let totalPower: Float

    /// Deposition location (normalized radius ρ)
    /// Typical range: 0.0 (core) - 0.9 (near edge)
    public let normalizedDepositionRadius: Float

    /// Deposition width (3σ width of Gaussian profile)
    /// Typical range: 0.05 - 0.15
    ///
    /// **Definition**: Full width containing 99.7% of power (3-sigma convention)
    /// - The actual Gaussian σ is computed as: σ = depositionWidth / 3
    /// - Profile: exp(-(ρ - ρ_dep)² / (2σ²))
    public let depositionWidth: Float

    /// Launch angle [degrees] (for future ray tracing)
    /// Currently unused in simplified model
    public let launchAngle: Float?

    /// Microwave frequency [Hz]
    /// ITER: 170 GHz
    /// Currently unused in simplified model
    public let frequency: Float?

    /// Enable current drive calculation
    public let enableCurrentDrive: Bool

    // MARK: - Initialization

    /// Initialize ECRH model
    ///
    /// - Parameters:
    ///   - totalPower: Total injected power [W] (e.g., 20e6 for 20 MW)
    ///   - normalizedDepositionRadius: Deposition location [dimensionless, 0-1]
    ///   - depositionWidth: Deposition width [dimensionless]
    ///   - launchAngle: Launch angle [degrees] (optional, for future use)
    ///   - frequency: Microwave frequency [Hz] (optional, for future use)
    ///   - enableCurrentDrive: Whether to calculate ECCD (default: false)
    public init(
        totalPower: Float,
        normalizedDepositionRadius: Float = 0.5,
        depositionWidth: Float = 0.1,
        launchAngle: Float? = nil,
        frequency: Float? = nil,
        enableCurrentDrive: Bool = false
    ) {
        self.totalPower = totalPower
        self.normalizedDepositionRadius = normalizedDepositionRadius
        self.depositionWidth = depositionWidth
        self.launchAngle = launchAngle
        self.frequency = frequency
        self.enableCurrentDrive = enableCurrentDrive
    }

    // MARK: - Power Deposition Calculation

    /// Compute ECRH power deposition profile
    ///
    /// **Model**: Gaussian deposition normalized to total power
    /// ```
    /// profile(ρ) = exp(-(ρ - ρ_dep)² / (2σ²))
    /// P(ρ) = P_total × profile(ρ) / ∫ profile(ρ) dV
    /// ```
    ///
    /// - Parameters:
    ///   - geometry: Tokamak geometry
    /// - Returns: Power density [W/m³]
    public func computePowerDensity(geometry: Geometry) -> MLXArray {
        let geometricFactors = GeometricFactors.from(geometry: geometry)
        let r = geometricFactors.cellRadii.value  // Physical radius r [m]
        let rho = r / geometry.minorRadius    // Normalized radius ρ = r/a
        let volumes = geometricFactors.cellVolumes.value

        // Gaussian profile centered at normalizedDepositionRadius
        let sigma = depositionWidth / 3.0  // 3-sigma width convention
        let delta = rho - normalizedDepositionRadius
        let profile = exp(-0.5 * pow(delta / sigma, 2))

        // Normalize to total power
        // ∫ P(ρ) dV = P_total
        let integral = (profile * volumes).sum()
        let P_density = totalPower * profile / (integral + 1e-10)  // [W/m³]

        return P_density
    }

    /// Compute ECCD current density (future implementation)
    ///
    /// **Model**: Lin-Liu current drive efficiency
    /// ```
    /// η_CD = C × (T_e / m_e c²) / (1 + ξ × effectiveCharge)
    /// j_ECCD = η_CD × P / (n_e × T_e)
    /// ```
    ///
    /// Currently returns zeros (placeholder for future implementation)
    ///
    /// - Parameters:
    ///   - P_density: Power density [W/m³]
    ///   - profiles: Core plasma profiles
    /// - Returns: Current density [MA/m²]
    private func computeCurrentDrive(
        P_density: MLXArray,
        profiles: CoreProfiles
    ) -> MLXArray {
        guard enableCurrentDrive else {
            return MLXArray.zeros(P_density.shape)
        }

        // TODO: Implement Lin-Liu current drive formula
        // For now, return zeros
        return MLXArray.zeros(P_density.shape)
    }

    // MARK: - Apply to Source Terms

    /// Apply ECRH heating to source terms
    ///
    /// **Units**:
    /// - Input: Computes W/m³
    /// - Output: Returns MW/m³ (SourceTerms convention)
    ///
    /// **Heating Partition**:
    /// - 100% to electrons (ECRH is electron-only heating)
    /// - 0% to ions (ions are heated indirectly via collisions)
    ///
    /// - Parameters:
    ///   - sources: Existing source terms
    ///   - profiles: Current core profiles
    ///   - geometry: Tokamak geometry
    /// - Returns: Updated source terms with ECRH contribution
    public func applyToSources(
        _ sources: SourceTerms,
        profiles: CoreProfiles,
        geometry: Geometry
    ) throws -> SourceTerms {
        // Compute power deposition
        let P_watts = computePowerDensity(geometry: geometry)

        // Convert to MW/m³ for SourceTerms
        let P_MW = PhysicsConstants.wattsToMegawatts(P_watts)

        // ECRH heats electrons only
        let updated_electron = sources.electronHeating.value + P_MW

        // Compute current drive (if enabled)
        let j_ECCD = computeCurrentDrive(P_density: P_watts, profiles: profiles)
        let updated_current = sources.currentSource.value + j_ECCD

        return SourceTerms(
            ionHeating: sources.ionHeating,
            electronHeating: EvaluatedArray(evaluating: updated_electron),
            particleSource: sources.particleSource,
            currentSource: EvaluatedArray(evaluating: updated_current)
        )
    }

    /// Compute source metadata for power balance tracking
    ///
    /// - Parameters:
    ///   - geometry: Geometry for volume integration
    /// - Returns: Source metadata with ECRH power
    public func computeMetadata(geometry: Geometry) -> SourceMetadata {

        let P_watts = computePowerDensity(geometry: geometry)

        // Volume integration: ∫ P dV → [W/m³] × [m³] = [W]
        let cellVolumes = GeometricFactors.from(geometry: geometry).cellVolumes.value
        let P_total = (P_watts * cellVolumes).sum()
        eval(P_total)

        let ecrhPower = P_total.item(Float.self)

        // ECRH heats electrons only (100% to electrons)
        return SourceMetadata(
            modelName: "ecrh",
            category: .auxiliary,
            ionPower: 0,
            electronPower: ecrhPower
        )
    }
}

// MARK: - ECRH Configuration Error

public enum ECRHError: Error, CustomStringConvertible {
    case invalidDepositionLocation(Float)
    case invalidDepositionWidth(Float)
    case negativePower(Float)

    public var description: String {
        switch self {
        case .invalidDepositionLocation(let rho):
            return "ECRH: Invalid deposition location ρ=\(rho). Must be in range [0, 1]."
        case .invalidDepositionWidth(let width):
            return "ECRH: Invalid deposition width \(width). Must be positive and < 0.5."
        case .negativePower(let power):
            return "ECRH: Negative power \(power). Must be non-negative."
        }
    }
}
