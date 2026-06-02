import MLX
import Foundation

// MARK: - Energy Conservation

/// Enforces energy conservation: ∫ (3/2 nₑ Tₑ + 3/2 nᵢ Tᵢ) dV = const
///
/// ## Physics Background
///
/// Total thermal energy in a tokamak plasma is the sum of electron and ion thermal energy:
///
/// ```
/// E = ∫ (3/2 nₑ Tₑ + 3/2 nᵢ Tᵢ) dV
/// ```
///
/// For an isolated system (no heating, no losses), energy is conserved:
///
/// ```
/// dE/timeStep = 0  →  E = const
/// ```
///
/// With sources/sinks, energy balance becomes:
///
/// ```
/// dE/timeStep = P_heating - P_losses
/// ```
///
/// ## Use Cases
///
/// 1. **Pure conservation test**: No sources/sinks → E should be constant
/// 2. **Energy balance validation**: With sources, track dE/timeStep = P_in - P_out
///
/// This implementation handles **Case 1** (pure conservation). For Case 2, use
/// diagnostics to monitor energy balance without enforcement.
///
/// ## Numerical Drift
///
/// Over 20,000+ timesteps, floating-point round-off causes energy drift:
///
/// ```
/// Initial:  E₀ = 5.0 × 10⁶ J
/// After 20k steps: E = 4.95 × 10⁶ J  (1% drift)
/// ```
///
/// ## Correction Method
///
/// We apply **uniform temperature scaling** to restore energy conservation:
///
/// ```swift
/// correctionFactor = E₀ / E
/// Tₑ_corrected = Tₑ × correctionFactor
/// Tᵢ_corrected = Tᵢ × correctionFactor
/// ```
///
/// **Physics**: Since E ∝ T (for constant density), to restore E → E₀, we need T → T × (E₀/E).
///
/// **Proof**:
/// - E = 3/2 nₑ (Tₑ + Tᵢ) × V
/// - E_new = 3/2 nₑ (Tₑ×factor + Tᵢ×factor) × V = factor × E
/// - To get E_new = E₀, we need factor = E₀ / E
///
/// **Assumption**: We scale both Tₑ and Tᵢ equally. For more sophisticated correction,
/// could weight by relative energy content.
///
/// ## Safety Features
///
/// - **Zero division protection**: Returns 1.0 if E ≤ 0
/// - **Non-finite detection**: Catches NaN/Inf before correction
/// - **Clamping**: Limits correction to ±20% (temperature change ±10%)
///
/// ## Example Usage
///
/// ```swift
/// let conservation = EnergyConservation(driftTolerance: 0.01)  // 1%
///
/// // Initial reference (no sources)
/// let E0 = conservation.computeConservedQuantity(
///     profiles: initialProfiles,
///     geometry: geometry
/// )
///
/// // After many timesteps
/// let E = conservation.computeConservedQuantity(
///     profiles: currentProfiles,
///     geometry: geometry
/// )
///
/// let drift = abs(E - E0) / E0
/// if drift > 0.01 {
///     let factor = conservation.computeCorrectionFactor(current: E, reference: E0)
///     let corrected = conservation.applyCorrection(
///         profiles: currentProfiles,
///         correctionFactor: factor
///     )
/// }
/// ```
public struct EnergyConservation: ConservationLaw {
    public let name = "EnergyConservation"
    public let driftTolerance: Float

    /// Initialize energy conservation law
    ///
    /// - Parameter driftTolerance: Relative drift threshold for correction (default: 0.01 = 1%)
    public init(driftTolerance: Float = 0.01) {
        self.driftTolerance = driftTolerance
    }

    // MARK: - ConservationLaw Protocol

    public func computeConservedQuantity(
        profiles: CoreProfiles,
        geometry: Geometry
    ) -> Float {
        // Extract profiles and geometry
        let ne = profiles.electronDensity.value                                    // [cellCount], m^-3
        let Te = profiles.electronTemperature.value                                // [cellCount], eV
        let Ti = profiles.ionTemperature.value                                     // [cellCount], eV
        let volumes = GeometricFactors.from(geometry: geometry).cellVolumes.value  // [cellCount], m^3

        // Constants (SI units)
        let eV_to_J: Float = 1.602176634e-19        // 1 eV in joules

        // Thermal energy density:
        // E_e = 3/2 nₑ Tₑ  [eV/m³]
        // E_i = 3/2 nᵢ Tᵢ  [eV/m³]
        // Assume quasi-neutrality: nᵢ ≈ nₑ
        let electronEnergyDensity = 1.5 * ne * Te   // [eV/m³]
        let ionEnergyDensity = 1.5 * ne * Ti        // [eV/m³]

        // Total thermal energy: E = ∫ (E_e + E_i) dV
        let totalEnergyDensity = electronEnergyDensity + ionEnergyDensity  // [eV/m³]
        let totalEnergy_eV = (totalEnergyDensity * volumes).sum()          // [eV]
        eval(totalEnergy_eV)

        // Convert to joules
        let totalEnergy_J = totalEnergy_eV.item(Float.self) * eV_to_J     // [J]

        return totalEnergy_J
    }

    public func computeCorrectionFactor(
        current: Float,
        reference: Float
    ) -> Float {
        // Guard against invalid values
        guard current > 0, current.isFinite else {
            print("[EnergyConservation] Invalid current value: \(current), no correction applied")
            return 1.0
        }
        guard reference > 0, reference.isFinite else {
            print("[EnergyConservation] Invalid reference value: \(reference), no correction applied")
            return 1.0
        }

        // Compute correction factor: factor = E₀ / E
        // Since E ∝ T (for constant density), to restore E → E₀, we need T → T × (E₀/E)
        let factor = reference / current

        // Clamp to ±20% to prevent large temperature changes
        if abs(factor - 1.0) > 0.2 {
            let clampedFactor: Float = factor > 1.0 ? 1.2 : 0.8
            print("""
                [EnergyConservation] Large correction (T × \(String(format: "%.3f", factor))) clamped to \
                T × \(String(format: "%.3f", clampedFactor))
                """)
            return clampedFactor
        }

        return factor
    }

    public func applyCorrection(
        profiles: CoreProfiles,
        correctionFactor: Float
    ) -> CoreProfiles {
        // Apply uniform temperature scaling: T_new = T × factor
        let Te_corrected = profiles.electronTemperature.value * correctionFactor
        let Ti_corrected = profiles.ionTemperature.value * correctionFactor

        // Evaluate corrected temperatures
        eval(Te_corrected, Ti_corrected)

        // Return corrected profiles (only temperatures changed)
        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti_corrected),
            electronTemperature: EvaluatedArray(evaluating: Te_corrected),
            electronDensity: profiles.electronDensity,
            poloidalFlux: profiles.poloidalFlux
        )
    }
}

// MARK: - Diagnostics

extension EnergyConservation {
    /// Compute relative drift for diagnostics
    ///
    /// - Parameters:
    ///   - current: Current total energy
    ///   - reference: Reference (initial) energy
    /// - Returns: Relative drift: |E - E₀| / E₀
    public func computeRelativeDrift(current: Float, reference: Float) -> Float {
        guard reference > 0 else { return 0.0 }
        return abs(current - reference) / reference
    }

    /// Check if correction is needed
    ///
    /// - Parameters:
    ///   - current: Current total energy
    ///   - reference: Reference (initial) energy
    /// - Returns: True if drift exceeds tolerance
    public func needsCorrection(current: Float, reference: Float) -> Bool {
        let drift = computeRelativeDrift(current: current, reference: reference)
        return drift > driftTolerance
    }

    /// Compute energy balance (for diagnostics with sources)
    ///
    /// For simulations with heating/losses, track energy rate of change:
    ///
    /// ```
    /// dE/timeStep ≈ (E - E_prev) / timeStep
    /// ```
    ///
    /// Compare with expected: P_heating - P_losses
    ///
    /// - Parameters:
    ///   - current: Current total energy
    ///   - previous: Previous total energy
    ///   - timeStep: Timestep
    /// - Returns: Energy rate of change [W]
    public func computeEnergyRate(
        current: Float,
        previous: Float,
        timeStep: Float
    ) -> Float {
        guard timeStep > 0 else { return 0.0 }
        return (current - previous) / timeStep  // [J/s] = [W]
    }
}
