import MLX
import Foundation
import FusionSurrogates

// MARK: - QLKNN Transport Model

/// QLKNN neural network transport model
///
/// **QuaLiKiz Neural Network (QLKNN)**: Fast surrogate for QuaLiKiz turbulent transport code.
///
/// **Physics**: Predicts turbulent heat and particle fluxes from local plasma parameters:
/// - Normalized gradients (R/L_Ti, R/L_Te, R/L_ne)
/// - Magnetic geometry (q, s, x = r/R)
/// - Collisionality (log10(ν*))
/// - Temperature ratio (Ti/Te)
///
/// **Performance**: 4-6 orders of magnitude faster than QuaLiKiz:
/// - QuaLiKiz: ~1 second per radial point
/// - QLKNN: ~1 millisecond per radial point (GPU-accelerated)
///
/// **Accuracy**: Trained on 300M QuaLiKiz simulations
/// - R² > 0.96 for all transport channels
/// - Valid for tokamak core plasmas (ITG, TEM, ETG turbulence)
///
/// **Implementation**: Pure MLX-Swift (no Python dependencies!)
/// - Uses bundled SafeTensors weights (289 KB)
/// - Metal GPU acceleration
/// - Float32 precision
///
/// **Reference**: https://github.com/1amageek/swift-fusion-surrogates
public struct QLKNNTransportModel: TransportModel {
    // MARK: - Properties

    public let name = "qlknn"

    /// MLX neural network for QLKNN prediction
    ///
    /// Note: QLKNNNetwork is not Sendable by design (class with mutable state).
    /// We wrap it in SendableQLKNNNetwork because:
    /// 1. TransportModel methods are called synchronously from simulation orchestrator
    /// 2. No concurrent access occurs (actor-isolated usage)
    /// 3. Network is immutable after loading (read-only operations)
    private let network: SendableQLKNNNetwork

    /// Sendable wrapper for QLKNNNetwork
    private struct SendableQLKNNNetwork: @unchecked Sendable {
        let network: QLKNNNetwork

        init(_ network: QLKNNNetwork) {
            self.network = network
        }
    }

    /// Effective charge Z_eff (for collisionality calculation)
    public let Zeff: Float

    /// Minimum transport coefficient floor [m²/s]
    ///
    /// Prevents numerical issues when QLKNN predicts very low transport
    public let minChi: Float

    /// Fallback transport model when QLKNN fails
    private let fallback: BohmGyroBohmTransportModel

    // MARK: - Initialization

    /// Initialize QLKNN transport model
    ///
    /// - Parameters:
    ///   - Zeff: Effective charge (default: 1.0 for pure deuterium)
    ///   - minChi: Minimum transport coefficient floor (default: 0.01 m²/s)
    /// - Throws: If QLKNN model fails to load
    public init(Zeff: Float = 1.0, minChi: Float = 0.01) throws {
        self.network = SendableQLKNNNetwork(try QLKNNNetwork.loadDefault())
        self.Zeff = Zeff
        self.minChi = minChi
        self.fallback = BohmGyroBohmTransportModel()
    }

    public init(params: TransportParameters) throws {
        self.network = SendableQLKNNNetwork(try QLKNNNetwork.loadDefault())
        self.Zeff = params.params["Zeff"] ?? 1.0
        self.minChi = params.params["min_chi"] ?? 0.01
        self.fallback = BohmGyroBohmTransportModel()
    }

    // MARK: - TransportModel Protocol

    public func computeCoefficients(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: TransportParameters
    ) -> TransportCoefficients {
        // Extract profiles as MLXArrays
        let Ti = profiles.ionTemperature.value
        let Te = profiles.electronTemperature.value
        let ne = profiles.electronDensity.value
        let radii = geometry.radii.value
        let q = geometry.safetyFactor.value

        // Compute QLKNN input features
        let inputs = computeQLKNNInputs(
            Ti: Ti,
            Te: Te,
            ne: ne,
            radii: radii,
            q: q,
            geometry: geometry
        )

        // Predict with QLKNN network
        let outputs: [String: MLXArray]
        do {
            try QLKNN.validateInputs(inputs)
            try QLKNN.validateShapes(inputs)
            outputs = try network.network.predict(inputs)
        } catch {
            print("[QLKNNTransportModel] QLKNN prediction failed: \(error)")
            print("[QLKNNTransportModel] Falling back to Bohm-GyroBohm transport")
            return fallback.computeCoefficients(
                profiles: profiles,
                geometry: geometry,
                params: params
            )
        }

        // Convert QLKNN outputs to physical transport coefficients
        let chiIon = computeIonDiffusivity(outputs: outputs, Te: Te, geometry: geometry)
        let chiElectron = computeElectronDiffusivity(outputs: outputs, Te: Te, geometry: geometry)
        let particleDiffusivity = computeParticleDiffusivity(outputs: outputs, Te: Te, geometry: geometry)

        // Apply minimum floor
        let chiIonClamped = maximum(chiIon, MLXArray(minChi))
        let chiElectronClamped = maximum(chiElectron, MLXArray(minChi))
        let particleDiffusivityClamped = maximum(particleDiffusivity, MLXArray(minChi))

        // No convection velocity from QLKNN (set to zero)
        let nCells = radii.shape[0]
        let convectionVelocity = MLXArray.zeros([nCells])

        return TransportCoefficients(
            evaluatingChiIon: chiIonClamped,
            chiElectron: chiElectronClamped,
            particleDiffusivity: particleDiffusivityClamped,
            convectionVelocity: convectionVelocity
        )
    }

    // MARK: - QLKNN Input Computation

    /// Compute QLKNN input parameters from profiles and geometry
    private func computeQLKNNInputs(
        Ti: MLXArray,
        Te: MLXArray,
        ne: MLXArray,
        radii: MLXArray,
        q: MLXArray,
        geometry: Geometry
    ) -> [String: MLXArray] {
        let majorRadius = geometry.majorRadius

        // Normalized gradients using MLXGradient
        let rLnTi = MLXGradient.normalizedGradient(
            profile: Ti,
            radii: radii,
            normalizationLength: majorRadius
        )

        let rLnTe = MLXGradient.normalizedGradient(
            profile: Te,
            radii: radii,
            normalizationLength: majorRadius
        )

        let rLnNe = MLXGradient.normalizedGradient(
            profile: ne,
            radii: radii,
            normalizationLength: majorRadius
        )

        // For single-species (deuterium), ni ≈ ne
        let rLnNi = rLnNe

        // Magnetic shear
        let sHat = MLXGradient.magneticShear(q: q, radii: radii)

        // Inverse aspect ratio
        let x = MLXGradient.inverseAspectRatio(radii: radii, majorRadius: majorRadius)

        // Temperature ratio
        let tiTe = MLXGradient.temperatureRatio(Ti: Ti, Te: Te)

        // Collisionality
        let logNuStar = MLXGradient.collisionality(
            ne: ne,
            Te: Te,
            q: q,
            radii: radii,
            majorRadius: majorRadius,
            Zeff: Zeff
        )

        // Normalized density (ni/ne ≈ 1 for single-species)
        let normni = broadcast(MLXArray(Float(1.0)), to: radii.shape)

        return [
            "Ati": rLnTi,
            "Ate": rLnTe,
            "Ane": rLnNe,
            "Ani": rLnNi,
            "q": q,
            "smag": sHat,
            "x": x,
            "Ti_Te": tiTe,
            "LogNuStar": logNuStar,
            "normni": normni
        ]
    }

    // MARK: - Transport Coefficient Conversion

    /// Compute GyroBohm diffusivity normalization
    ///
    /// **Formula**: χ_GB = T_e^(3/2) * sqrt(m_i) / [(e B)² * a]
    ///
    /// **Units**:
    /// - Te: [eV] (converted to Joules internally)
    /// - Result: [m²/s]
    ///
    /// **Reference**: Kadomtsev (1975), Plasma Physics and Controlled Nuclear Fusion Research
    private func computeGyrBohmDiffusivity(
        Te: MLXArray,
        geometry: Geometry
    ) -> MLXArray {
        // Physical constants
        let electronCharge: Float = 1.602e-19  // [C]
        let protonMass: Float = 1.673e-27      // [kg] (use deuterium ≈ 2 * proton)
        let ionMass = Float(2.0) * protonMass  // Deuterium
        let eV_to_J: Float = 1.602e-19         // [J/eV]

        let B = geometry.toroidalField          // [T]
        let a = geometry.minorRadius            // [m]

        // Convert Te from eV to Joules for SI calculation
        let Te_J = Te * eV_to_J

        // χ_GB = T_e^(3/2) * sqrt(m_i) / [(e B)² * a]
        // Note: Te is in Joules here
        let numerator = pow(Te_J, Float(1.5)) * sqrt(ionMass)
        let denominator = (electronCharge * B) * (electronCharge * B) * a

        return numerator / denominator
    }

    /// Compute ion thermal diffusivity from QLKNN outputs
    ///
    /// **QLKNN Output**: Normalized flux in GyroBohm units
    /// **Conversion**: χ_i = (efiITG + efiTEM) * χ_GB
    ///
    /// **Modes**:
    /// - efiITG: Ion thermal flux from Ion Temperature Gradient mode
    /// - efiTEM: Ion thermal flux from Trapped Electron Mode
    private func computeIonDiffusivity(
        outputs: [String: MLXArray],
        Te: MLXArray,
        geometry: Geometry
    ) -> MLXArray {
        let chiGB = computeGyrBohmDiffusivity(Te: Te, geometry: geometry)

        let efiITG = outputs["efiITG"]!
        let efiTEM = outputs["efiTEM"]!

        // Total ion heat flux (ITG + TEM modes)
        let efi_total = efiITG + efiTEM

        // Convert to physical diffusivity
        return efi_total * chiGB
    }

    /// Compute electron thermal diffusivity from QLKNN outputs
    ///
    /// **QLKNN Output**: Normalized flux in GyroBohm units
    /// **Conversion**: χ_e = (efeITG + efeTEM + efeETG) * χ_GB
    ///
    /// **Modes**:
    /// - efeITG: Electron thermal flux from ITG mode
    /// - efeTEM: Electron thermal flux from TEM mode
    /// - efeETG: Electron thermal flux from Electron Temperature Gradient mode
    private func computeElectronDiffusivity(
        outputs: [String: MLXArray],
        Te: MLXArray,
        geometry: Geometry
    ) -> MLXArray {
        let chiGB = computeGyrBohmDiffusivity(Te: Te, geometry: geometry)

        let efeITG = outputs["efeITG"]!
        let efeTEM = outputs["efeTEM"]!
        let efeETG = outputs["efeETG"]!

        // Total electron heat flux (ITG + TEM + ETG modes)
        let efe_total = efeITG + efeTEM + efeETG

        // Convert to physical diffusivity
        return efe_total * chiGB
    }

    /// Compute particle diffusivity from QLKNN outputs
    ///
    /// **QLKNN Output**: Normalized flux in GyroBohm units
    /// **Conversion**: D = (pfeITG + pfeTEM) * χ_GB
    ///
    /// **Modes**:
    /// - pfeITG: Particle flux from ITG mode
    /// - pfeTEM: Particle flux from TEM mode
    private func computeParticleDiffusivity(
        outputs: [String: MLXArray],
        Te: MLXArray,
        geometry: Geometry
    ) -> MLXArray {
        let chiGB = computeGyrBohmDiffusivity(Te: Te, geometry: geometry)

        let pfeITG = outputs["pfeITG"]!
        let pfeTEM = outputs["pfeTEM"]!

        // Total particle flux (ITG + TEM modes)
        let pfe_total = pfeITG + pfeTEM

        // Convert to physical diffusivity
        return pfe_total * chiGB
    }
}
