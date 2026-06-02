import MLX
import Foundation

// MARK: - Conservation Law Protocol

/// Protocol for physics conservation laws (particle, energy, momentum, etc.)
///
/// Conservation laws enforce fundamental physical constraints during long-time simulations.
/// Over 20,000+ timesteps, numerical round-off can cause drift in conserved quantities:
///
/// ```
/// Initial:  N₀ = 1.0 × 10²¹ particles
/// After 20k steps: N = 0.99 × 10²¹  (1% drift - UNPHYSICAL!)
/// ```
///
/// ## Design Pattern
///
/// Each conservation law implements:
/// 1. **computeConservedQuantity**: Calculate total conserved quantity (GPU)
/// 2. **computeCorrectionFactor**: Determine how much correction is needed
/// 3. **applyCorrection**: Modify profiles to restore conservation (GPU)
///
/// ## Example Usage
///
/// ```swift
/// let law = ParticleConservation(driftTolerance: 0.005)  // 0.5% tolerance
/// let N0 = law.computeConservedQuantity(profiles: initial, geometry: geometry)
/// let N = law.computeConservedQuantity(profiles: current, geometry: geometry)
///
/// let drift = abs(N - N0) / N0
/// if drift > law.driftTolerance {
///     let factor = law.computeCorrectionFactor(current: N, reference: N0)
///     let corrected = law.applyCorrection(profiles: current, correctionFactor: factor)
/// }
/// ```
///
/// ## GPU-First Design
///
/// All computations (quantity calculation, correction) execute on GPU for performance:
/// - `computeConservedQuantity`: GPU array operations → single scalar
/// - `applyCorrection`: GPU element-wise operations → corrected profiles
///
/// ## Safety
///
/// Implementations MUST guard against:
/// - Division by zero (zero/negative quantities)
/// - Non-finite values (NaN, Inf)
/// - Large corrections (> 20% should be clamped)
public protocol ConservationLaw: Sendable {
    /// Human-readable name (e.g., "ParticleConservation")
    var name: String { get }

    /// Drift tolerance before correction is applied
    ///
    /// Typical values:
    /// - Particle conservation: 0.005 (0.5%)
    /// - Energy conservation: 0.01 (1%)
    var driftTolerance: Float { get }

    /// Compute total conserved quantity
    ///
    /// Executes on GPU, returns single scalar value.
    ///
    /// - Parameters:
    ///   - profiles: Current plasma profiles
    ///   - geometry: Tokamak geometry (for volume integration)
    /// - Returns: Total conserved quantity (e.g., particle number, total energy)
    ///
    /// ## Example: Particle Conservation
    ///
    /// ```swift
    /// let ne = profiles.electronDensity.value      // [cellCount]
    /// let volumes = geometry.cellVolumes.value     // [cellCount]
    /// let totalParticles = (ne * volumes).sum()    // GPU sum reduction
    /// return totalParticles.item(Float.self)       // Extract scalar
    /// ```
    func computeConservedQuantity(
        profiles: CoreProfiles,
        geometry: Geometry
    ) -> Float

    /// Compute correction factor to restore conservation
    ///
    /// Given current and reference quantities, compute multiplicative correction factor.
    ///
    /// - Parameters:
    ///   - current: Current total quantity
    ///   - reference: Reference (initial) quantity
    /// - Returns: Correction factor (typically `reference / current`)
    ///
    /// ## Safety Requirements
    ///
    /// Implementations MUST:
    /// 1. Check for zero/negative values
    /// 2. Check for non-finite values (NaN, Inf)
    /// 3. Clamp large corrections (> 20%)
    ///
    /// ## Example Implementation
    ///
    /// ```swift
    /// guard current > 0, current.isFinite else { return 1.0 }
    /// guard reference > 0, reference.isFinite else { return 1.0 }
    ///
    /// let factor = reference / current
    ///
    /// // Clamp to ±20%
    /// if abs(factor - 1.0) > 0.2 {
    ///     return factor > 1.0 ? 1.2 : 0.8
    /// }
    ///
    /// return factor
    /// ```
    func computeCorrectionFactor(
        current: Float,
        reference: Float
    ) -> Float

    /// Apply correction to profiles
    ///
    /// Modify profiles to enforce conservation. All operations execute on GPU.
    ///
    /// - Parameters:
    ///   - profiles: Current profiles to correct
    ///   - correctionFactor: Multiplicative correction factor
    /// - Returns: Corrected profiles
    ///
    /// ## Example: Uniform Density Scaling
    ///
    /// ```swift
    /// let ne_corrected = profiles.electronDensity.value * correctionFactor
    /// eval(ne_corrected)
    /// return CoreProfiles(
    ///     ionTemperature: profiles.ionTemperature,
    ///     electronTemperature: profiles.electronTemperature,
    ///     electronDensity: EvaluatedArray(evaluating: ne_corrected),
    ///     poloidalFlux: profiles.poloidalFlux
    /// )
    /// ```
    func applyCorrection(
        profiles: CoreProfiles,
        correctionFactor: Float
    ) -> CoreProfiles
}

// MARK: - Conservation Result

/// Result from conservation enforcement
public struct ConservationResult: Sendable, Codable {
    /// Law name (e.g., "ParticleConservation")
    public let lawName: String

    /// Reference (initial) quantity
    public let referenceQuantity: Float

    /// Current quantity before correction
    public let currentQuantity: Float

    /// Relative drift: |Q - Q₀| / Q₀
    public let relativeDrift: Float

    /// Correction factor applied
    public let correctionFactor: Float

    /// Whether correction was applied
    public let corrected: Bool

    /// Simulation time [s]
    public let time: Float

    /// Timestep number
    public let step: Int

    public init(
        lawName: String,
        referenceQuantity: Float,
        currentQuantity: Float,
        relativeDrift: Float,
        correctionFactor: Float,
        corrected: Bool,
        time: Float,
        step: Int
    ) {
        self.lawName = lawName
        self.referenceQuantity = referenceQuantity
        self.currentQuantity = currentQuantity
        self.relativeDrift = relativeDrift
        self.correctionFactor = correctionFactor
        self.corrected = corrected
        self.time = time
        self.step = step
    }

    /// Human-readable summary
    public func summary() -> String {
        let driftPercent = relativeDrift * 100
        let status = corrected ? "✓ Corrected" : "ℹ️ Monitored"
        return """
        [\(lawName)] \(status)
          Reference: \(referenceQuantity)
          Current:   \(currentQuantity)
          Drift:     \(String(format: "%.3f", driftPercent))%
          Factor:    \(String(format: "%.6f", correctionFactor))×
          Step:      \(step) (t=\(String(format: "%.4f", time))s)
        """
    }
}
