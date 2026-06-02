import MLX
import Foundation

// MARK: - Conservation Enforcer

/// Orchestrates multiple conservation laws during simulation
///
/// ## Purpose
///
/// Over long simulations (20,000+ timesteps), numerical round-off errors accumulate,
/// causing drift in conserved quantities. ConservationEnforcer:
///
/// 1. Computes reference quantities at t=0
/// 2. Periodically checks current quantities
/// 3. Applies corrections if drift exceeds tolerance
/// 4. Logs enforcement results
///
/// ## Design
///
/// - **Multiple laws**: Can enforce particle, energy, momentum conservation simultaneously
/// - **Sequential application**: Laws applied in order (particle first, then energy)
/// - **Periodic checking**: Runs every N steps (default: 1000) to minimize overhead
/// - **Optional enforcement**: Can monitor drift without enforcing (for diagnostics)
///
/// ## Example Usage
///
/// ```swift
/// let enforcer = ConservationEnforcer(
///     laws: [
///         ParticleConservation(driftTolerance: 0.005),  // 0.5%
///         EnergyConservation(driftTolerance: 0.01)      // 1%
///     ],
///     initialProfiles: initialProfiles,
///     geometry: geometry,
///     enforcementInterval: 1000,  // Check every 1000 steps
///     verbose: true
/// )
///
/// // In simulation loop:
/// if step % 1000 == 0 {
///     let (corrected, results) = enforcer.enforce(
///         profiles: currentProfiles,
///         geometry: geometry,
///         step: step,
///         time: time
///     )
///     currentProfiles = corrected
/// }
/// ```
///
/// ## Performance
///
/// - Particle conservation: O(cellCount) → ~0.01% overhead
/// - Energy conservation: O(cellCount) → ~0.01% overhead
/// - Runs every 1000 steps → total overhead < 0.1%
public struct ConservationEnforcer: Sendable {
    /// Conservation laws to enforce
    private let laws: [any ConservationLaw]

    /// Reference (initial) quantities for each law
    private let referenceQuantities: [Float]

    /// Enforcement interval (steps)
    public let enforcementInterval: Int

    /// Verbose logging
    public let verbose: Bool

    /// Initialize enforcer with conservation laws
    ///
    /// - Parameters:
    ///   - laws: Array of conservation laws to enforce
    ///   - initialProfiles: Initial plasma profiles (t=0)
    ///   - geometry: Tokamak geometry
    ///   - enforcementInterval: Check every N steps (default: 1000)
    ///   - verbose: Enable detailed logging (default: true)
    ///
    /// ## Example
    ///
    /// ```swift
    /// let enforcer = ConservationEnforcer(
    ///     laws: [
    ///         ParticleConservation(),
    ///         EnergyConservation()
    ///     ],
    ///     initialProfiles: profiles,
    ///     geometry: geometry
    /// )
    /// ```
    public init(
        laws: [any ConservationLaw],
        initialProfiles: CoreProfiles,
        geometry: Geometry,
        enforcementInterval: Int = 1000,
        verbose: Bool = true
    ) {
        self.laws = laws
        self.enforcementInterval = enforcementInterval
        self.verbose = verbose

        // Compute reference quantities at t=0
        self.referenceQuantities = laws.map { law in
            law.computeConservedQuantity(profiles: initialProfiles, geometry: geometry)
        }

        if verbose {
            print("\n[ConservationEnforcer] Initialized with \(laws.count) law(s):")
            for (law, refQty) in zip(laws, referenceQuantities) {
                print("  • \(law.name): reference = \(refQty)")
            }
        }
    }

    // MARK: - Enforcement

    /// Enforce conservation laws
    ///
    /// For each law:
    /// 1. Compute current quantity
    /// 2. Check drift vs. reference
    /// 3. Apply correction if drift > tolerance
    /// 4. Return corrected profiles and results
    ///
    /// Laws are applied sequentially (particle first, then energy).
    ///
    /// - Parameters:
    ///   - profiles: Current plasma profiles
    ///   - geometry: Tokamak geometry
    ///   - step: Current timestep number
    ///   - time: Current simulation time [s]
    /// - Returns: Tuple of (corrected profiles, enforcement results)
    ///
    /// ## Example
    ///
    /// ```swift
    /// let (corrected, results) = enforcer.enforce(
    ///     profiles: currentProfiles,
    ///     geometry: geometry,
    ///     step: 10000,
    ///     time: 1.0
    /// )
    ///
    /// // Check if any corrections were applied
    /// let anyCorrected = results.contains { $0.corrected }
    /// if anyCorrected {
    ///     print("Conservation enforced at step \(step)")
    /// }
    /// ```
    public func enforce(
        profiles: CoreProfiles,
        geometry: Geometry,
        step: Int,
        time: Float
    ) -> (profiles: CoreProfiles, results: [ConservationResult]) {
        var currentProfiles = profiles
        var results: [ConservationResult] = []

        // Apply each law sequentially
        for (lawIndex, law) in laws.enumerated() {
            let referenceQty = referenceQuantities[lawIndex]

            // Compute current quantity
            let currentQty = law.computeConservedQuantity(
                profiles: currentProfiles,
                geometry: geometry
            )

            // Guard against zero/invalid reference OR current quantity
            // Per design doc 3.1: "ゼロ割・非有限値を防ぐ"
            guard referenceQty > 0, referenceQty.isFinite,
                  currentQty.isFinite else {
                // Invalid quantities mean no meaningful conservation to enforce
                // Record for monitoring but skip correction
                if verbose {
                    print("[ConservationEnforcer] Skipping \(law.name): invalid quantities (ref=\(referenceQty), current=\(currentQty))")
                }
                let result = ConservationResult(
                    lawName: law.name,
                    referenceQuantity: referenceQty,
                    currentQuantity: currentQty,
                    relativeDrift: Float.nan,  // Cannot compute meaningful drift
                    correctionFactor: 1.0,
                    corrected: false,
                    time: time,
                    step: step
                )
                results.append(result)
                continue  // Skip to next law
            }

            // Compute drift (now safe from division by zero and NaN)
            // Note: referenceQty > 0 is guaranteed by guard, so abs() is redundant but kept for clarity
            let relativeDrift = abs(currentQty - referenceQty) / referenceQty

            // Check if correction is needed
            let needsCorrection = relativeDrift > law.driftTolerance

            if needsCorrection {
                // Compute correction factor
                let correctionFactor = law.computeCorrectionFactor(
                    current: currentQty,
                    reference: referenceQty
                )

                // Apply correction
                currentProfiles = law.applyCorrection(
                    profiles: currentProfiles,
                    correctionFactor: correctionFactor
                )

                // Create result
                let result = ConservationResult(
                    lawName: law.name,
                    referenceQuantity: referenceQty,
                    currentQuantity: currentQty,
                    relativeDrift: relativeDrift,
                    correctionFactor: correctionFactor,
                    corrected: true,
                    time: time,
                    step: step
                )
                results.append(result)

                if verbose {
                    print("""
                        [ConservationEnforcer] ✓ Corrected \(law.name) at step \(step):
                          Drift: \(String(format: "%.3f", relativeDrift * 100))%
                          Factor: \(String(format: "%.6f", correctionFactor))×
                        """)
                }
            } else {
                // No correction needed, but record result for diagnostics
                let result = ConservationResult(
                    lawName: law.name,
                    referenceQuantity: referenceQty,
                    currentQuantity: currentQty,
                    relativeDrift: relativeDrift,
                    correctionFactor: 1.0,
                    corrected: false,
                    time: time,
                    step: step
                )
                results.append(result)

                if verbose && relativeDrift > 0.001 {  // Log if drift > 0.1%
                    print("""
                        [ConservationEnforcer] ℹ️ Monitored \(law.name) at step \(step):
                          Drift: \(String(format: "%.3f", relativeDrift * 100))%
                        """)
                }
            }
        }

        return (currentProfiles, results)
    }

    /// Check if enforcement should run at this step
    ///
    /// - Parameter step: Current timestep number
    /// - Returns: True if enforcement should run
    public func shouldEnforce(step: Int) -> Bool {
        return step % enforcementInterval == 0 && step > 0
    }

    // MARK: - Diagnostics

    /// Get summary of all laws
    public func lawsSummary() -> String {
        var summary = "Conservation Laws:\n"
        for (law, refQty) in zip(laws, referenceQuantities) {
            summary += "  • \(law.name):\n"
            summary += "    - Tolerance: \(String(format: "%.2f", law.driftTolerance * 100))%\n"
            summary += "    - Reference: \(refQty)\n"
        }
        return summary
    }

    /// Compute current drift for all laws (diagnostic only, no enforcement)
    ///
    /// - Parameters:
    ///   - profiles: Current profiles
    ///   - geometry: Tokamak geometry
    /// - Returns: Array of (law name, drift)
    public func computeCurrentDrift(
        profiles: CoreProfiles,
        geometry: Geometry
    ) -> [(lawName: String, drift: Float)] {
        var drifts: [(String, Float)] = []

        for (lawIndex, law) in laws.enumerated() {
            let referenceQty = referenceQuantities[lawIndex]
            let currentQty = law.computeConservedQuantity(
                profiles: profiles,
                geometry: geometry
            )
            let drift = abs(currentQty - referenceQty) / abs(referenceQty)
            drifts.append((law.name, drift))
        }

        return drifts
    }
}
