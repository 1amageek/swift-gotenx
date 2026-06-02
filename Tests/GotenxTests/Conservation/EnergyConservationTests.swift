import Testing
import MLX
import Foundation
@testable import GotenxCore

/// Tests for energy conservation enforcement
@Suite("Energy Conservation Tests")
struct EnergyConservationTests {

    // MARK: - Test Helpers

    /// Create test profiles with uniform temperature and density
    private func createUniformProfiles(cellCount: Int, electronTemperature: Float, ionTemperature: Float, electronDensity: Float) throws -> CoreProfiles {
        let electronTemperatureArray = MLXArray(Array(repeating: electronTemperature, count: cellCount))
        let ionTemperatureArray = MLXArray(Array(repeating: ionTemperature, count: cellCount))
        let electronDensityArray = MLXArray(Array(repeating: electronDensity, count: cellCount))
        let poloidalFlux = MLXArray(Array(repeating: Float(0.0), count: cellCount))

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: ionTemperatureArray),
            electronTemperature: EvaluatedArray(evaluating: electronTemperatureArray),
            electronDensity: EvaluatedArray(evaluating: electronDensityArray),
            poloidalFlux: EvaluatedArray(evaluating: poloidalFlux)
        )
    }

    /// Create simple geometry
    private func createGeometry(cellCount: Int) -> Geometry {
        let config = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )
        return Geometry(config: config)
    }

    // MARK: - Basic Functionality Tests

    @Test("Compute conserved quantity")
    func testComputeConservedQuantity() throws {
        let conservation = EnergyConservation()
        let cellCount = 25

        // Create profiles: Te = Ti = 10 keV = 10000 eV, ne = 1e20 m^-3
        let profiles = try createUniformProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        // Compute total energy
        let totalEnergy = conservation.computeConservedQuantity(
            profiles: profiles,
            geometry: geometry
        )

        // Expected energy calculation:
        // E = ∫ (3/2 nₑ Tₑ + 3/2 nₑ Tᵢ) dV
        //   = 3/2 nₑ (Tₑ + Tᵢ) × V_total
        // With nₑ = 1e20 m^-3, Tₑ = Tᵢ = 10000 eV
        // E_eV = 3/2 × 1e20 × 20000 × V_total
        // Convert to Joules: E_J = E_eV × 1.602e-19

        // Just verify it's positive and finite
        #expect(totalEnergy > 0, "Total energy should be positive")
        #expect(totalEnergy.isFinite, "Total energy should be finite")
    }

    @Test("Correction factor calculation")
    func testCorrectionFactor() {
        let conservation = EnergyConservation()

        // Test normal correction (1% drift)
        let current: Float = 0.99e6  // 0.99 MJ
        let reference: Float = 1.0e6  // 1.0 MJ
        let factor = conservation.computeCorrectionFactor(current: current, reference: reference)

        let expected = reference / current  // Should be E₀/E (not sqrt!)
        #expect(abs(factor - expected) < 1e-6, "Correction factor incorrect: \(factor) vs \(expected)")
    }

    @Test("Correction factor without sqrt")
    func testCorrectionFactorNoSqrt() {
        let conservation = EnergyConservation()

        // Verify correction factor is E₀/E, NOT sqrt(E₀/E)
        let current: Float = 0.9e6
        let reference: Float = 1.0e6
        let factor = conservation.computeCorrectionFactor(current: current, reference: reference)

        // Expected: E₀/E = 1.0/0.9 = 1.111...
        let expectedLinear = reference / current
        #expect(abs(factor - expectedLinear) < 1e-6, "Factor should be linear (E₀/E), not sqrt")

        // Should NOT be sqrt(E₀/E)
        let wrongSqrt = sqrt(reference / current)
        #expect(abs(factor - wrongSqrt) > 0.01, "Factor should not be sqrt(E₀/E)")
    }

    @Test("Correction factor with zero current")
    func testCorrectionFactorZero() {
        let conservation = EnergyConservation()

        // Should return 1.0 (no correction) for zero current
        let factor = conservation.computeCorrectionFactor(current: 0.0, reference: 1.0e6)
        #expect(factor == 1.0, "Should return 1.0 for zero current")
    }

    @Test("Correction factor clamping")
    func testCorrectionFactorClamping() {
        let conservation = EnergyConservation()

        // Large correction (30% drift) should be clamped to 20%
        let current: Float = 0.7e6
        let reference: Float = 1.0e6
        let factor = conservation.computeCorrectionFactor(current: current, reference: reference)

        // Unclamped would be 1.0/0.7 ≈ 1.43
        // Clamped should be 1.2
        #expect(abs(factor - 1.2) < 1e-6, "Large correction should be clamped to 1.2")
    }

    @Test("Apply correction")
    func testApplyCorrection() throws {
        let conservation = EnergyConservation()
        let cellCount = 25

        // Create profiles with Te = Ti = 10000 eV
        let profiles = try createUniformProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)

        // Apply 5% correction (factor = 1.05)
        let corrected = conservation.applyCorrection(
            profiles: profiles,
            correctionFactor: 1.05
        )

        // Check corrected temperatures
        eval(corrected.electronTemperature.value, corrected.ionTemperature.value)
        let Te_corrected = corrected.electronTemperature.value.asArray(Float.self)
        let Ti_corrected = corrected.ionTemperature.value.asArray(Float.self)

        for (te, ti) in zip(Te_corrected, Ti_corrected) {
            let expectedTe: Float = 10000.0 * 1.05
            let expectedTi: Float = 10000.0 * 1.05
            #expect(abs(te - expectedTe) < 1e-3, "Corrected Te incorrect")
            #expect(abs(ti - expectedTi) < 1e-3, "Corrected Ti incorrect")
        }

        // Check density unchanged
        eval(profiles.electronDensity.value, corrected.electronDensity.value)
        let ne_original = profiles.electronDensity.value.asArray(Float.self)
        let ne_corrected = corrected.electronDensity.value.asArray(Float.self)

        for (orig, corr) in zip(ne_original, ne_corrected) {
            #expect(abs(orig - corr) < 1e-6, "Density should be unchanged")
        }
    }

    // MARK: - Round-Trip Tests

    @Test("Round-trip: correction restores conservation")
    func testRoundTrip() throws {
        let conservation = EnergyConservation()
        let cellCount = 25

        // Create initial profiles
        let initialProfiles = try createUniformProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        // Compute reference
        let E0 = conservation.computeConservedQuantity(
            profiles: initialProfiles,
            geometry: geometry
        )

        // Simulate drift: artificially scale temperature by 0.99
        let drifted = conservation.applyCorrection(
            profiles: initialProfiles,
            correctionFactor: 0.99
        )

        // Compute drifted quantity
        let E_drifted = conservation.computeConservedQuantity(
            profiles: drifted,
            geometry: geometry
        )

        // Apply correction
        let factor = conservation.computeCorrectionFactor(current: E_drifted, reference: E0)
        let corrected = conservation.applyCorrection(
            profiles: drifted,
            correctionFactor: factor
        )

        // Verify restoration
        let E_corrected = conservation.computeConservedQuantity(
            profiles: corrected,
            geometry: geometry
        )

        let relativeError = abs(E_corrected - E0) / E0
        #expect(relativeError < 1e-5, "Round-trip should restore conservation: error = \(relativeError)")
    }

    @Test("Energy scales linearly with temperature")
    func testEnergyScalesLinearly() throws {
        let conservation = EnergyConservation()
        let cellCount = 25

        // Create profiles with Te = Ti = 10000 eV
        let profiles1 = try createUniformProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let E1 = conservation.computeConservedQuantity(profiles: profiles1, geometry: geometry)

        // Scale temperature by 2×
        let profiles2 = conservation.applyCorrection(profiles: profiles1, correctionFactor: 2.0)
        let E2 = conservation.computeConservedQuantity(profiles: profiles2, geometry: geometry)

        // Energy should also scale by 2×
        let ratio = E2 / E1
        #expect(abs(ratio - 2.0) < 1e-5, "Energy should scale linearly with temperature: ratio = \(ratio)")
    }

    // MARK: - Drift Diagnostics Tests

    @Test("Compute relative drift")
    func testRelativeDrift() {
        let conservation = EnergyConservation()

        let reference: Float = 1.0e6
        let current: Float = 0.995e6  // 0.5% drift

        let drift = conservation.computeRelativeDrift(current: current, reference: reference)
        let expected: Float = 0.005

        #expect(abs(drift - expected) < 1e-6, "Relative drift incorrect")
    }

    @Test("Needs correction check")
    func testNeedsCorrection() {
        let conservation = EnergyConservation(driftTolerance: 0.01)  // 1%

        // Below tolerance
        let needs1 = conservation.needsCorrection(current: 0.992e6, reference: 1.0e6)
        #expect(!needs1, "Should not need correction for 0.8% drift")

        // Above tolerance
        let needs2 = conservation.needsCorrection(current: 0.98e6, reference: 1.0e6)
        #expect(needs2, "Should need correction for 2% drift")
    }

    @Test("Energy rate computation")
    func testEnergyRate() {
        let conservation = EnergyConservation()

        let E_prev: Float = 1.0e6  // 1 MJ
        let E_curr: Float = 1.1e6  // 1.1 MJ
        let timeStep: Float = 0.1  // 0.1 s

        let rate = conservation.computeEnergyRate(current: E_curr, previous: E_prev, timeStep: timeStep)

        // Expected: dE/timeStep = (1.1 - 1.0) / 0.1 = 1.0 MW
        let expected: Float = 1.0e6  // W
        #expect(abs(rate - expected) < 1e-3, "Energy rate incorrect")
    }

    // MARK: - Edge Cases

    @Test("Conservation with gradient profiles")
    func testGradientProfiles() throws {
        let conservation = EnergyConservation()
        let cellCount = 25

        // Create profiles with linear gradient: T = 15000 * (1 - 0.5*r)
        var Te_values: [Float] = []
        var Ti_values: [Float] = []
        for i in 0..<cellCount {
            let r = Float(i) / Float(cellCount - 1)
            Te_values.append(15000.0 * (1.0 - 0.5 * r))
            Ti_values.append(15000.0 * (1.0 - 0.5 * r))
        }

        let Te = MLXArray(Te_values, [cellCount])
        let Ti = MLXArray(Ti_values, [cellCount])
        let ne = MLXArray(Array(repeating: Float(1e20), count: cellCount))
        let psi = MLXArray(Array(repeating: Float(0.0), count: cellCount))

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let geometry = createGeometry(cellCount: cellCount)

        // Compute initial energy
        let E0 = conservation.computeConservedQuantity(profiles: profiles, geometry: geometry)

        // Apply correction
        let corrected = conservation.applyCorrection(profiles: profiles, correctionFactor: 1.05)

        // Compute corrected energy
        let E_corrected = conservation.computeConservedQuantity(
            profiles: corrected,
            geometry: geometry
        )

        // Should be 5% higher
        let ratio = E_corrected / E0
        #expect(abs(ratio - 1.05) < 1e-5, "Correction should scale total energy by factor")
    }

    @Test("Conservation with zero temperature cells")
    func testZeroTemperatureCells() throws {
        let conservation = EnergyConservation()
        let cellCount = 25

        // Create profiles with some zero temperature cells
        var Te_values: [Float] = []
        var Ti_values: [Float] = []
        for i in 0..<cellCount {
            Te_values.append(i < 20 ? 10000.0 : 0.0)  // Last 5 cells are zero
            Ti_values.append(i < 20 ? 10000.0 : 0.0)
        }

        let Te = MLXArray(Te_values, [cellCount])
        let Ti = MLXArray(Ti_values, [cellCount])
        let ne = MLXArray(Array(repeating: Float(1e20), count: cellCount))
        let psi = MLXArray(Array(repeating: Float(0.0), count: cellCount))

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let geometry = createGeometry(cellCount: cellCount)

        // Should handle zero temperature cells gracefully
        let E = conservation.computeConservedQuantity(profiles: profiles, geometry: geometry)
        #expect(E.isFinite, "Should handle zero temperature cells")

        // Apply correction
        let corrected = conservation.applyCorrection(profiles: profiles, correctionFactor: 1.1)
        eval(corrected.electronTemperature.value)

        let Te_corrected = corrected.electronTemperature.value.asArray(Float.self)
        for value in Te_corrected {
            #expect(value.isFinite, "Corrected temperature should be finite")
        }
    }
}
