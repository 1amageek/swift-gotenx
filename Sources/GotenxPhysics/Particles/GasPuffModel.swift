import MLX
import Foundation
import GotenxCore

// MARK: - Gas Puff Model

/// Gas puff particle fueling model
///
/// **Physics**: Edge particle injection via gas puffing
/// - Method: Gas injection from plasma edge
/// - Profile: Exponential penetration from edge (ρ = 1)
/// - Application: Density control, fueling
///
/// **Key Features**:
/// - Edge-localized particle source
/// - Exponential penetration profile
/// - Total particle conservation
///
/// **Model**: Simplified edge fueling (no neutral transport)
/// - Real codes (FRANTIC, AURORA) use neutral transport
/// - This model assumes:
///   - Instantaneous ionization
///   - Exponential penetration: exp(-(1-ρ)/λ)
///   - Normalized to total puff rate
///
/// **Reference**:
/// - Pégourié, B., et al., Plasma Phys. Control. Fusion 49, 467 (2007)
/// - ITER Physics Basis, Nucl. Fusion 39, 2137 (1999)
///
/// **Units**:
/// - Input: puffRate [particles/s]
/// - Output: Particle source density [m⁻³/s]
public struct GasPuffModel: Sendable {
    // MARK: - Properties

    /// Total particle puff rate [particles/s]
    /// Typical range for ITER: 1e21 - 1e22 particles/s
    public let puffRate: Float

    /// Penetration depth (λ_n in normalized coordinates)
    /// Typical range: 0.05 - 0.2
    /// - Small λ (~0.05): Shallow penetration (edge-only)
    /// - Large λ (~0.2): Deeper penetration
    public let penetrationDepth: Float

    // MARK: - Initialization

    /// Initialize gas puff model
    ///
    /// - Parameters:
    ///   - puffRate: Total particle puff rate [particles/s]
    ///   - penetrationDepth: Exponential decay length [dimensionless, 0-1]
    public init(
        puffRate: Float,
        penetrationDepth: Float = 0.1
    ) {
        self.puffRate = puffRate
        self.penetrationDepth = penetrationDepth
    }

    // MARK: - Particle Source Calculation

    /// Compute gas puff particle source profile
    ///
    /// **Model**: Exponential penetration from edge
    /// ```
    /// profile(ρ) = exp(-(1 - ρ) / λ_n)
    /// S(ρ) = S_total × profile(ρ) / ∫ profile(ρ) dV
    /// ```
    ///
    /// **Physical Interpretation**:
    /// - ρ = 1 (edge): Maximum source
    /// - ρ → 0 (core): Exponentially decaying
    /// - λ_n controls penetration depth
    ///
    /// **Conservation**: Ensures ∫ S(ρ) dV = puffRate (particle conservation)
    ///
    /// - Parameters:
    ///   - geometry: Tokamak geometry
    /// - Returns: Particle source density [m⁻³/s]
    public func computeParticleSource(geometry: Geometry) -> MLXArray {
        computeParticleSource(
            geometry: geometry,
            geometricFactors: GeometricFactors.from(geometry: geometry),
            validatesConservation: true
        )
    }

    package func computeParticleSource(
        geometry: Geometry,
        geometricFactors: GeometricFactors,
        validatesConservation: Bool
    ) -> MLXArray {
        let r = geometricFactors.cellRadii.value  // Physical radius r [m]
        let rho = r / geometry.minorRadius    // Normalized radius ρ = r/a
        let volumes = geometricFactors.cellVolumes.value

        // Exponential penetration from edge
        // (1 - ρ): Distance from edge (0 at edge, 1 at core)
        let distanceFromEdge = 1.0 - rho
        let profile = exp(-distanceFromEdge / penetrationDepth)

        // Normalize to total puff rate
        // ∫ S(ρ) dV = puffRate [particles/s]
        let integralArray = (profile * volumes).sum()
        let integral = validatesConservation ? integralArray.item(Float.self) : 1.0

        // Validate integral is non-zero to prevent division by zero
        if validatesConservation && integral <= 1e-20 {
            #if DEBUG
            print("⚠️  Warning: Gas puff profile integral too small: \(integral)")
            print("   penetrationDepth = \(penetrationDepth) may be invalid")
            print("   Returning zero particle source")
            #endif
            return MLXArray.zeros(profile.shape)
        }

        let S_particles = puffRate * profile / (integralArray + 1e-20)  // [m⁻³/s]

        // Validate particle conservation (DEBUG builds only)
        #if DEBUG
        if validatesConservation {
            let totalParticles = (S_particles * volumes).sum().item(Float.self)
            let conservationError = abs(totalParticles - puffRate) / puffRate

            if conservationError > 0.01 {  // 1% tolerance
                print("⚠️  Warning: Gas puff particle conservation error: \(conservationError * 100)%")
                print("   Expected: \(puffRate) particles/s")
                print("   Computed: \(totalParticles) particles/s")
                print("   Check: ∫ S(ρ) dV = puffRate")
            }
        }
        #endif

        return S_particles
    }

    // MARK: - Apply to Source Terms

    /// Apply gas puff particle source to source terms
    ///
    /// **Units**:
    /// - Input: Computes m⁻³/s
    /// - Output: Returns m⁻³/s (SourceTerms convention for particles)
    ///
    /// **Important**: Unlike heating sources, particle source does NOT need
    /// MW/m³ conversion. SourceTerms.particleSource uses [m⁻³/s] directly.
    ///
    /// - Parameters:
    ///   - sources: Existing source terms
    ///   - geometry: Tokamak geometry
    /// - Returns: Updated source terms with gas puff contribution
    public func applyToSources(
        _ sources: SourceTerms,
        geometry: Geometry
    ) -> SourceTerms {
        applyToSources(
            sources,
            geometry: geometry,
            geometricFactors: GeometricFactors.from(geometry: geometry),
            evaluationMode: .eager,
            validatesConservation: true,
            validateDebugUnits: true
        )
    }

    package func applyToSources(
        _ sources: SourceTerms,
        context: SourceEvaluationContext
    ) -> SourceTerms {
        applyToSources(
            sources,
            geometry: context.geometry,
            geometricFactors: context.geometricFactors,
            evaluationMode: context.evaluationMode,
            validatesConservation: context.includesMetadata,
            validateDebugUnits: context.validatesDebugUnits
        )
    }

    package func applyToSources(
        _ sources: SourceTerms,
        geometry: Geometry,
        geometricFactors: GeometricFactors,
        evaluationMode: MLXEvaluationMode,
        validatesConservation: Bool,
        validateDebugUnits: Bool
    ) -> SourceTerms {
        let S_particles = computeParticleSource(
            geometry: geometry,
            geometricFactors: geometricFactors,
            validatesConservation: validatesConservation
        )

        // Add to existing particle source
        let updated_particles = sources.particleSource.value + S_particles

        return SourceTerms(
            ionHeating: sources.ionHeating,
            electronHeating: sources.electronHeating,
            particleSource: evaluationMode.wrap(updated_particles),
            currentSource: sources.currentSource,
            metadata: sources.metadata,
            validateDebugUnits: validateDebugUnits
        )
    }

    /// Compute source metadata for power balance tracking
    ///
    /// Gas puff is a particle source only, does not contribute to power balance.
    ///
    /// - Parameters:
    ///   - geometry: Geometry (unused, for API consistency)
    /// - Returns: Source metadata with zero power
    public func computeMetadata(geometry: Geometry) -> SourceMetadata {
        return SourceMetadata(
            modelName: "gas_puff",
            category: .other,
            ionPower: 0,  // Particle source only
            electronPower: 0
        )
    }
}

// MARK: - Gas Puff Configuration Error

public enum GasPuffError: Error, CustomStringConvertible {
    case negativePuffRate(Float)
    case invalidPenetrationDepth(Float)

    public var description: String {
        switch self {
        case .negativePuffRate(let rate):
            return "Gas Puff: Negative puff rate \(rate). Must be non-negative."
        case .invalidPenetrationDepth(let depth):
            return "Gas Puff: Invalid penetration depth \(depth). Must be in range (0, 1]."
        }
    }
}
