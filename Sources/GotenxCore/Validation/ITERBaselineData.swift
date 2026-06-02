import Foundation

// MARK: - ITER Baseline Data

/// ITER Baseline Scenario reference data
///
/// Provides reference parameters from ITER Physics Basis (Nuclear Fusion 39(12), 1999)
/// for validation of global quantities and profile shapes.
///
/// ## Purpose
///
/// - Sanity checks for global quantities (Q, βN, τE)
/// - Qualitative profile shape validation
/// - Order-of-magnitude verification
///
/// ## Usage
///
/// ```swift
/// let baseline = ITERBaselineData.load()
///
/// // Check global quantities
/// print("Expected Q: \(baseline.globalQuantities.fusionGain)")
/// print("Expected βN: \(baseline.globalQuantities.normalizedBeta)")
///
/// // Compare profile shapes
/// let refTi = baseline.profiles.ionTemperature
/// let gotenxTi = // ... from simulation
///
/// let result = ProfileComparator.compare(
///     quantity: "ion_temperature",
///     predicted: gotenxTi,
///     reference: refTi,
///     time: 2.0,
///     thresholds: .experimental  // Relaxed for design data
/// )
/// ```
///
/// ## Note
///
/// ITER Baseline data is **design values**, not simulation outputs.
/// For detailed validation, use TORAXReferenceData instead.
public struct ITERBaselineData: Sendable {
    /// Tokamak geometry parameters
    public let geometry: GeometryParameters

    /// Reference profiles at steady state (t = 2s)
    public let profiles: ReferenceProfiles

    /// Global performance quantities
    public let globalQuantities: GlobalQuantities

    public init(
        geometry: GeometryParameters,
        profiles: ReferenceProfiles,
        globalQuantities: GlobalQuantities
    ) {
        self.geometry = geometry
        self.profiles = profiles
        self.globalQuantities = globalQuantities
    }

    /// Load ITER Baseline Scenario data
    ///
    /// Data source: ITER Physics Basis, Nuclear Fusion 39(12), 1999, Table II
    ///
    /// ## Plasma Parameters
    ///
    /// - Major radius: R₀ = 6.2 m
    /// - Minor radius: a = 2.0 m
    /// - Plasma current: Ip = 15 MA
    /// - Toroidal field: B₀ = 5.3 T
    /// - Fusion gain: Q = 10
    ///
    /// ## Profile Assumptions
    ///
    /// - Ti, electronTemperature: Parabolic profiles peaked at core
    ///   - coreIonTemperature = coreElectronTemperature = 20 keV
    ///   - Ti_edge = Te_edge = 100 eV
    ///   - Shape: T(r) = T_edge + (T_core - T_edge) × (1 - (r/a)²)²
    ///
    /// - electronDensity: Linear profile
    ///   - coreElectronDensity = 1.0 × 10²⁰ m⁻³
    ///   - ne_edge = 0.2 × 10²⁰ m⁻³
    ///   - Shape: ne(r) = ne_edge + (coreElectronDensity - ne_edge) × (1 - r/a)
    ///
    /// - Returns: ITER Baseline data structure
    public static func load() -> ITERBaselineData {
        // ITER geometry from Physics Basis Table II
        let geometry = GeometryParameters(
            majorRadius: 6.2,      // [m]
            minorRadius: 2.0,      // [m]
            elongation: 1.7,       // Plasma elongation
            triangularity: 0.33,   // Plasma triangularity
            plasmaCurrent: 15.0,   // [MA]
            toroidalField: 5.3     // [T]
        )

        // Generate parabolic profiles on uniform grid
        let nPoints = 50
        let rho = stride(from: 0.0, through: 1.0, by: 1.0/Float(nPoints-1)).map { Float($0) }

        // Ion temperature: Parabolic profile
        // Ti(r) = Ti_edge + (coreIonTemperature - Ti_edge) × (1 - r²)²
        let coreIonTemperature: Float = 20000.0  // 20 keV = 20,000 eV
        let Ti_edge: Float = 100.0    // 100 eV
        let Ti = rho.map { r in
            Ti_edge + (coreIonTemperature - Ti_edge) * pow(1.0 - r*r, 2.0)
        }

        // Electron temperature: Same as ion temperature
        let Te = Ti

        // Electron density: Linear profile
        // ne(r) = ne_edge + (coreElectronDensity - ne_edge) × (1 - r)
        let coreElectronDensity: Float = 1.0e20   // 1.0 × 10²⁰ m⁻³
        let ne_edge: Float = 0.2e20   // 0.2 × 10²⁰ m⁻³
        let ne = rho.map { r in
            ne_edge + (coreElectronDensity - ne_edge) * (1.0 - r)
        }

        // Global quantities from ITER Physics Basis
        let global = GlobalQuantities(
            fusionPower: 400.0,     // [MW] - ITER Q=10 design (50 MW → 500 MW)
            alphaPower: 80.0,       // [MW] - 20% of fusion power (400 × 0.2)
            energyConfinementTime: 3.7,          // [s] - H98(y,2) = 1.0 scaling
            normalizedBeta: 1.8,         // Normalized beta (typical ITER value)
            fusionGain: 10.0       // Fusion gain (design goal)
        )

        // Steady state time point
        let steadyStateTime: Float = 2.0  // [s]

        return ITERBaselineData(
            geometry: geometry,
            profiles: ReferenceProfiles(
                normalizedRadius: rho,
                ionTemperature: Ti,
                electronTemperature: Te,
                electronDensity: ne,
                time: steadyStateTime
            ),
            globalQuantities: global
        )
    }

    // MARK: - Validation Helpers

    /// Check if global quantities are physically reasonable
    ///
    /// Verifies:
    /// - Q > 5 (fusion relevant)
    /// - 1.0 < βN < 3.5 (MHD stable)
    /// - τE > 1.0 s (good confinement)
    ///
    /// - Parameter actual: Actual global quantities from simulation
    /// - Returns: True if all quantities are reasonable
    public static func validateGlobalQuantities(_ actual: GlobalQuantities) -> Bool {
        // Q should be > 5 for fusion-relevant regime
        guard actual.fusionGain > 5.0 else {
            print("⚠️ Q = \(actual.fusionGain) < 5 (not fusion-relevant)")
            return false
        }

        // βN should be in MHD-stable range
        guard actual.normalizedBeta > 1.0 && actual.normalizedBeta < 3.5 else {
            print("⚠️ βN = \(actual.normalizedBeta) outside [1.0, 3.5] (MHD limits)")
            return false
        }

        // τE should be > 1s for good confinement
        guard actual.energyConfinementTime > 1.0 else {
            print("⚠️ τE = \(actual.energyConfinementTime) < 1s (poor confinement)")
            return false
        }

        return true
    }

    /// Print comparison summary
    ///
    /// - Parameters:
    ///   - predicted: Predicted global quantities
    ///   - reference: Reference (baseline) global quantities
    public static func printComparison(
        predicted: GlobalQuantities,
        reference: GlobalQuantities
    ) {
        print("\n[ITER Baseline Comparison]")
        print("                  Predicted    Reference    Ratio")
        print("  fusionGain:       \(String(format: "%8.2f", predicted.fusionGain))     \(String(format: "%8.2f", reference.fusionGain))     \(String(format: "%6.2f", predicted.fusionGain / reference.fusionGain))×")
        print("  β_N:            \(String(format: "%8.2f", predicted.normalizedBeta))     \(String(format: "%8.2f", reference.normalizedBeta))     \(String(format: "%6.2f", predicted.normalizedBeta / reference.normalizedBeta))×")
        print("  τ_E [s]:        \(String(format: "%8.2f", predicted.energyConfinementTime))     \(String(format: "%8.2f", reference.energyConfinementTime))     \(String(format: "%6.2f", predicted.energyConfinementTime / reference.energyConfinementTime))×")
        print("  fusionPower [MW]:  \(String(format: "%8.1f", predicted.fusionPower))     \(String(format: "%8.1f", reference.fusionPower))     \(String(format: "%6.2f", predicted.fusionPower / reference.fusionPower))×")
        print("")
    }
}
