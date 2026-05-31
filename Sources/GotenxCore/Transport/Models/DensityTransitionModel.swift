// DensityTransitionModel.swift
// Density-dependent turbulence transition from ITG to RI regimes

import MLX
import Foundation

// MARK: - Density Transition Transport Model

/// Density-dependent turbulence transition transport model
///
/// Implements smooth transition from ITG (Ion-Temperature Gradient) turbulence
/// at low density to RI (Resistive-Interchange) turbulence at high density.
///
/// **Physical Basis**: Discovery from Kinoshita et al., PRL 132, 235101 (2024):
/// - **Below n_trans**: ITG turbulence dominates (ion temperature gradient driven)
/// - **At n_trans**: Turbulence minimized (optimal confinement)
/// - **Above n_trans**: RI turbulence dominates (pressure gradient + resistivity driven)
///
/// **Transition Function**:
/// ```
/// α(n_e) = 1 / (1 + exp(-(n_e - n_trans) / Δn))
/// χ_eff = (1 - α) × χ_ITG + α × χ_RI
/// ```
///
/// **Isotope Effects**:
/// - ITG regime: Similar for H/D (χ_H ≈ χ_D)
/// - RI regime: Suppressed for D (χ_D < χ_H)
/// - Scaling: χ ∝ 1 / A_i^exponent
///
/// **Reference**: Kinoshita et al., Phys. Rev. Lett. 132, 235101 (2024)
public struct DensityTransitionModel: TransportModel {
    // MARK: - Properties

    public let name = "density-transition"

    /// Low-density ITG model
    private let itgModel: any TransportModel

    /// High-density RI model
    private let riModel: any TransportModel

    /// Transition density n_trans [m⁻³]
    ///
    /// Density at which turbulence is minimized.
    ///
    /// **Typical value**: 2.5×10¹⁹ m⁻³ for tokamaks
    public let transitionDensity: Float

    /// Transition width Δn [m⁻³]
    ///
    /// Width of sigmoid transition region.
    ///
    /// **Typical value**: 0.5×10¹⁹ m⁻³ (smooth over ~20% range)
    public let transitionWidth: Float

    /// Ion mass number (1=H, 2=D, 3=T)
    ///
    /// **Default**: 2.0 (deuterium)
    /// **Note**: Isotope effects are handled internally by ResistiveInterchangeModel via ρ_s
    public let ionMassNumber: Float

    // MARK: - Initialization

    /// Initialize density transition model
    ///
    /// - Parameters:
    ///   - itgModel: Transport model for low-density ITG regime
    ///   - riModel: Transport model for high-density RI regime
    ///   - transitionDensity: Transition density n_trans [m⁻³]
    ///   - transitionWidth: Transition width Δn [m⁻³]
    ///   - ionMassNumber: Ion mass number (default: 2.0 for D)
    public init(
        itgModel: any TransportModel,
        riModel: any TransportModel,
        transitionDensity: Float,
        transitionWidth: Float,
        ionMassNumber: Float = 2.0
    ) {
        self.itgModel = itgModel
        self.riModel = riModel
        self.transitionDensity = transitionDensity
        self.transitionWidth = transitionWidth
        self.ionMassNumber = ionMassNumber
    }

    // MARK: - TransportModel Protocol

    public func computeCoefficients(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: TransportParameters
    ) -> TransportCoefficients {
        let n_e = profiles.electronDensity.value

        // Compute transition weight α(n_e) ∈ [0, 1]
        // α = 0: Pure ITG (low density)
        // α = 1: Pure RI (high density)
        let alpha = transitionWeight(density: n_e)

        // Compute ITG regime coefficients (low density)
        let chi_itg = itgModel.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            params: params
        )

        // Compute RI regime coefficients (high density)
        // Note: Isotope effects are already included via ρ_s in ResistiveInterchangeModel
        let chi_ri = riModel.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            params: params
        )

        // Smooth blend: χ_eff = (1 - α) × χ_ITG + α × χ_RI
        let chi_blend = blendCoefficients(
            lowDensity: chi_itg,
            highDensity: chi_ri,
            alpha: alpha
        )

        return chi_blend
    }

    // MARK: - Private Methods

    /// Compute transition weight using sigmoid function
    ///
    /// **Formula**:
    /// ```
    /// α(n_e) = 1 / (1 + exp(-(n_e - n_trans) / Δn))
    /// ```
    ///
    /// **Properties**:
    /// - α → 0 as n_e → 0 (pure ITG)
    /// - α = 0.5 at n_e = n_trans (balanced transition)
    /// - α → 1 as n_e → ∞ (pure RI)
    ///
    /// - Parameter density: Electron density [nCells] in m⁻³
    /// - Returns: Transition weight α [nCells] ∈ [0, 1]
    private func transitionWeight(density: MLXArray) -> MLXArray {
        // Sigmoid transition centered at n_trans with width Δn
        let delta_n = (density - transitionDensity) / transitionWidth
        return 1.0 / (1.0 + exp(-delta_n))
    }

    /// Blend ITG and RI coefficients using transition weight
    ///
    /// **Formula**:
    /// ```
    /// χ_eff = (1 - α) × χ_ITG + α × χ_RI
    /// ```
    ///
    /// - Parameters:
    ///   - lowDensity: ITG coefficients (α = 0)
    ///   - highDensity: RI coefficients (α = 1)
    ///   - alpha: Transition weight [nCells] ∈ [0, 1]
    /// - Returns: Blended coefficients
    private func blendCoefficients(
        lowDensity: TransportCoefficients,
        highDensity: TransportCoefficients,
        alpha: MLXArray
    ) -> TransportCoefficients {
        // Blend ion heat diffusivity
        let chiIon_blend = (1.0 - alpha) * lowDensity.chiIon.value
                         + alpha * highDensity.chiIon.value

        // Blend electron heat diffusivity
        let chiElectron_blend = (1.0 - alpha) * lowDensity.chiElectron.value
                              + alpha * highDensity.chiElectron.value

        // Blend particle diffusivity
        let diffusivity_blend = (1.0 - alpha) * lowDensity.particleDiffusivity.value
                              + alpha * highDensity.particleDiffusivity.value

        // Blend convection velocity
        let convection_blend = (1.0 - alpha) * lowDensity.convectionVelocity.value
                             + alpha * highDensity.convectionVelocity.value

        return TransportCoefficients(
            evaluatingChiIon: chiIon_blend,
            chiElectron: chiElectron_blend,
            particleDiffusivity: diffusivity_blend,
            convectionVelocity: convection_blend
        )
    }
}

// MARK: - Factory Method

extension DensityTransitionModel {
    /// Create density transition model with default ITG model
    ///
    /// Uses `BohmGyroBohmTransportModel` as default ITG model.
    ///
    /// - Parameters:
    ///   - riCoefficient: RI coefficient (default: 0.5)
    ///   - transitionDensity: Transition density [m⁻³] (default: 2.5e19)
    ///   - transitionWidth: Transition width [m⁻³] (default: 0.5e19)
    ///   - ionMassNumber: Ion mass number (default: 2.0 for D)
    /// - Returns: Configured density transition model
    public static func createDefault(
        riCoefficient: Float = 0.5,
        transitionDensity: Float = 2.5e19,
        transitionWidth: Float = 0.5e19,
        ionMassNumber: Float = 2.0
    ) -> DensityTransitionModel {
        // Default ITG model: Bohm-GyroBohm
        // CRITICAL: Pass ionMassNumber for isotope scaling via ρ_s
        let itgModel = BohmGyroBohmTransportModel(ionMassNumber: ionMassNumber)

        // RI model
        let riModel = ResistiveInterchangeModel(
            coefficientRI: riCoefficient,
            ionMassNumber: ionMassNumber
        )

        return DensityTransitionModel(
            itgModel: itgModel,
            riModel: riModel,
            transitionDensity: transitionDensity,
            transitionWidth: transitionWidth,
            ionMassNumber: ionMassNumber
        )
    }
}
