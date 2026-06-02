import Testing
import MLX
import Foundation
@testable import GotenxCore

/// Tests for particle conservation enforcement
@Suite("Particle Conservation Tests")
struct ParticleConservationTests {

    // MARK: - Test Helpers

    /// Create test profiles with uniform density
    private func createUniformProfiles(cellCount: Int, electronDensity: Float) throws -> CoreProfiles {
        let ionTemperature = MLXArray(Array(repeating: Float(10000.0), count: cellCount))
        let electronTemperature = MLXArray(Array(repeating: Float(10000.0), count: cellCount))
        let electronDensityArray = MLXArray(Array(repeating: electronDensity, count: cellCount))
        let poloidalFlux = MLXArray(Array(repeating: Float(0.0), count: cellCount))

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: ionTemperature),
            electronTemperature: EvaluatedArray(evaluating: electronTemperature),
            electronDensity: EvaluatedArray(evaluating: electronDensityArray),
            poloidalFlux: EvaluatedArray(evaluating: poloidalFlux)
        )
    }

    /// Create simple geometry with uniform volumes
    private func createUniformGeometry(cellCount: Int, cellVolume: Float) -> Geometry {
        let config = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,    // m
            minorRadius: 2.0,    // m
            toroidalField: 5.3   // T
        )
        return Geometry(config: config)
    }

    // MARK: - Basic Functionality Tests

    @Test("Compute conserved quantity")
    func testComputeConservedQuantity() throws {
        let conservation = ParticleConservation()
        let cellCount = 25

        // Create profiles: ne = 1e20 m^-3
        let profiles = try createUniformProfiles(cellCount: cellCount, electronDensity: 1e20)
        let geometry = createUniformGeometry(cellCount: cellCount, cellVolume: 1.0)

        // Compute total particles
        let totalParticles = conservation.computeConservedQuantity(
            profiles: profiles,
            geometry: geometry
        )

        // Expected: N = ne × V_total
        // V_total = 2π R₀ radialSpacing × cellCount
        // where radialSpacing = a / cellCount = 2.0 / 25 = 0.08 m
        // V_cell = 2π × 6.2 × 0.08 ≈ 3.115 m³
        // V_total = 3.115 × 25 ≈ 77.88 m³
        let R0: Float = 6.2
        let a: Float = 2.0
        let radialSpacing = a / Float(cellCount)
        let cellVolume = 2.0 * Float.pi * R0 * radialSpacing
        let expected: Float = 1e20 * cellVolume * Float(cellCount)
        let relativeError = abs(totalParticles - expected) / expected

        #expect(relativeError < 1e-5, "Total particles incorrect: \(totalParticles) vs \(expected)")
    }

    @Test("Correction factor calculation")
    func testCorrectionFactor() {
        let conservation = ParticleConservation()

        // Test normal correction (1% drift)
        let current: Float = 0.99e21
        let reference: Float = 1.0e21
        let factor = conservation.computeCorrectionFactor(current: current, reference: reference)

        let expected = reference / current
        #expect(abs(factor - expected) < 1e-6, "Correction factor incorrect")
    }

    @Test("Correction factor with zero current")
    func testCorrectionFactorZero() {
        let conservation = ParticleConservation()

        // Should return 1.0 (no correction) for zero current
        let factor = conservation.computeCorrectionFactor(current: 0.0, reference: 1.0e21)
        #expect(factor == 1.0, "Should return 1.0 for zero current")
    }

    @Test("Correction factor clamping")
    func testCorrectionFactorClamping() {
        let conservation = ParticleConservation()

        // Large correction (30% drift) should be clamped to 20%
        let current: Float = 0.7e21
        let reference: Float = 1.0e21
        let factor = conservation.computeCorrectionFactor(current: current, reference: reference)

        // Unclamped would be 1.0/0.7 ≈ 1.43
        // Clamped should be 1.2
        #expect(abs(factor - 1.2) < 1e-6, "Large correction should be clamped to 1.2")
    }

    @Test("Apply correction")
    func testApplyCorrection() throws {
        let conservation = ParticleConservation()
        let cellCount = 25

        // Create profiles with ne = 1e20
        let profiles = try createUniformProfiles(cellCount: cellCount, electronDensity: 1e20)

        // Apply 5% correction (factor = 1.05)
        let corrected = conservation.applyCorrection(
            profiles: profiles,
            correctionFactor: 1.05
        )

        // Check corrected density
        eval(corrected.electronDensity.value)
        let ne_corrected = corrected.electronDensity.value.asArray(Float.self)

        for value in ne_corrected {
            let expected: Float = 1e20 * 1.05
            let relativeError = abs(value - expected) / expected
            #expect(relativeError < 1e-6, "Corrected density incorrect")
        }

        // Check other profiles unchanged
        eval(profiles.ionTemperature.value, corrected.ionTemperature.value)
        let Ti_original = profiles.ionTemperature.value.asArray(Float.self)
        let Ti_corrected = corrected.ionTemperature.value.asArray(Float.self)

        for (orig, corr) in zip(Ti_original, Ti_corrected) {
            #expect(abs(orig - corr) < 1e-6, "Temperature should be unchanged")
        }
    }

    // MARK: - Round-Trip Tests

    @Test("Round-trip: correction restores conservation")
    func testRoundTrip() throws {
        let conservation = ParticleConservation()
        let cellCount = 25

        // Create initial profiles
        let initialProfiles = try createUniformProfiles(cellCount: cellCount, electronDensity: 1e20)
        let geometry = createUniformGeometry(cellCount: cellCount, cellVolume: 1.0)

        // Compute reference
        let N0 = conservation.computeConservedQuantity(
            profiles: initialProfiles,
            geometry: geometry
        )

        // Simulate drift: artificially scale density by 0.99
        let drifted = conservation.applyCorrection(
            profiles: initialProfiles,
            correctionFactor: 0.99
        )

        // Compute drifted quantity
        let N_drifted = conservation.computeConservedQuantity(
            profiles: drifted,
            geometry: geometry
        )

        // Apply correction
        let factor = conservation.computeCorrectionFactor(current: N_drifted, reference: N0)
        let corrected = conservation.applyCorrection(
            profiles: drifted,
            correctionFactor: factor
        )

        // Verify restoration
        let N_corrected = conservation.computeConservedQuantity(
            profiles: corrected,
            geometry: geometry
        )

        let relativeError = abs(N_corrected - N0) / N0
        #expect(relativeError < 1e-6, "Round-trip should restore conservation")
    }

    // MARK: - Drift Diagnostics Tests

    @Test("Compute relative drift")
    func testRelativeDrift() {
        let conservation = ParticleConservation()

        let reference: Float = 1.0e21
        let current: Float = 0.995e21  // 0.5% drift

        let drift = conservation.computeRelativeDrift(current: current, reference: reference)
        let expected: Float = 0.005

        #expect(abs(drift - expected) < 1e-6, "Relative drift incorrect")
    }

    @Test("Needs correction check")
    func testNeedsCorrection() {
        let conservation = ParticleConservation(driftTolerance: 0.005)  // 0.5%

        // Below tolerance
        let needs1 = conservation.needsCorrection(current: 0.997e21, reference: 1.0e21)
        #expect(!needs1, "Should not need correction for 0.3% drift")

        // Above tolerance
        let needs2 = conservation.needsCorrection(current: 0.99e21, reference: 1.0e21)
        #expect(needs2, "Should need correction for 1% drift")
    }

    // MARK: - Edge Cases

    @Test("Conservation with gradient profiles")
    func testGradientProfiles() throws {
        let conservation = ParticleConservation()
        let cellCount = 25

        // Create profiles with linear gradient: ne = 1e20 * (1 - 0.5*r)
        var ne_values: [Float] = []
        for i in 0..<cellCount {
            let r = Float(i) / Float(cellCount - 1)
            ne_values.append(1e20 * (1.0 - 0.5 * r))
        }

        let Ti = MLXArray(Array(repeating: Float(10000.0), count: cellCount))
        let Te = MLXArray(Array(repeating: Float(10000.0), count: cellCount))
        let ne = MLXArray(ne_values, [cellCount])
        let psi = MLXArray(Array(repeating: Float(0.0), count: cellCount))

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let geometry = createUniformGeometry(cellCount: cellCount, cellVolume: 1.0)

        // Compute initial total
        let N0 = conservation.computeConservedQuantity(profiles: profiles, geometry: geometry)

        // Apply correction
        let corrected = conservation.applyCorrection(profiles: profiles, correctionFactor: 1.05)

        // Compute corrected total
        let N_corrected = conservation.computeConservedQuantity(
            profiles: corrected,
            geometry: geometry
        )

        // Should be 5% higher
        let ratio = N_corrected / N0
        #expect(abs(ratio - 1.05) < 1e-5, "Correction should scale total by factor")
    }

    @Test("Conservation with zero density cells")
    func testZeroDensityCells() throws {
        let conservation = ParticleConservation()
        let cellCount = 25

        // Create profiles with some zero density cells
        var ne_values: [Float] = []
        for i in 0..<cellCount {
            ne_values.append(i < 20 ? 1e20 : 0.0)  // Last 5 cells are zero
        }

        let Ti = MLXArray(Array(repeating: Float(10000.0), count: cellCount))
        let Te = MLXArray(Array(repeating: Float(10000.0), count: cellCount))
        let ne = MLXArray(ne_values, [cellCount])
        let psi = MLXArray(Array(repeating: Float(0.0), count: cellCount))

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let geometry = createUniformGeometry(cellCount: cellCount, cellVolume: 1.0)

        // Should handle zero cells gracefully
        let N = conservation.computeConservedQuantity(profiles: profiles, geometry: geometry)
        #expect(N.isFinite, "Should handle zero density cells")

        // Apply correction
        let corrected = conservation.applyCorrection(profiles: profiles, correctionFactor: 1.1)
        eval(corrected.electronDensity.value)

        let ne_corrected = corrected.electronDensity.value.asArray(Float.self)
        for value in ne_corrected {
            #expect(value.isFinite, "Corrected density should be finite")
        }
    }
}
