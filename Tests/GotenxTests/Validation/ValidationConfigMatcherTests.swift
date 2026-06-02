// ValidationConfigMatcherTests.swift
// Tests for validation configuration matching

import Testing
import Foundation
@testable import GotenxCore

@Suite("Validation Config Matcher Tests")
struct ValidationConfigMatcherTests {

    @Test("Match configuration to ITER Baseline")
    func matchToITERBaseline() throws {
        let config = try ValidationConfigMatcher.matchToITERBaseline()

        // Verify mesh configuration
        #expect(config.runtime.static.mesh.cellCount == 50, "Should use 50 cells")
        #expect(config.runtime.static.mesh.majorRadius == 6.2, "Major radius should be 6.2 m")
        #expect(config.runtime.static.mesh.minorRadius == 2.0, "Minor radius should be 2.0 m")
        #expect(config.runtime.static.mesh.toroidalField == 5.3, "Toroidal field should be 5.3 T")

        // Verify evolution configuration
        #expect(config.runtime.static.evolution.ionHeat, "Ion heat should evolve")
        #expect(config.runtime.static.evolution.electronHeat, "Electron heat should evolve")
        #expect(config.runtime.static.evolution.electronDensity, "Density should evolve")
        #expect(!config.runtime.static.evolution.poloidalFlux, "Current should not evolve")

        // Verify solver configuration
        #expect(config.runtime.static.solver.type == "newton_raphson", "Should use Newton-Raphson solver")
        #expect(config.runtime.static.solver.tolerance == 1e-6, "Tolerance should be 1e-6")

        // Verify time configuration
        #expect(config.time.start == 0.0, "Start time should be 0")
        #expect(config.time.end == 2.0, "End time should be 2s")

        // Verify transport model
        #expect(config.runtime.dynamic.transport.modelType == .bohmGyrobohm, "Should use Bohm-GyroBohm")

        // Verify sources
        #expect(config.runtime.dynamic.sources.ohmicHeating, "Ohmic heating should be enabled")
        #expect(config.runtime.dynamic.sources.fusionPower, "Fusion power should be enabled")
        #expect(config.runtime.dynamic.sources.ionElectronExchange, "Ion-electron exchange should be enabled")
        #expect(config.runtime.dynamic.sources.bremsstrahlung, "Bremsstrahlung should be enabled")
    }

    @Test("Match configuration to TORAX reference data")
    func matchToTorax() throws {
        // Create mock TORAX data
        let time: [Float] = Array(stride(from: 0.0, through: 2.0, by: 0.02))  // 101 points
        let normalizedRadius: [Float] = Array(stride(from: 0.0, through: 1.0, by: 0.01))   // 101 points

        let timeCount = time.count
        let nRho = normalizedRadius.count

        // Create mock profiles
        let profiles = (0..<timeCount).map { _ in
            Array(repeating: Float(1000), count: nRho)
        }

        let toraxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: profiles,
            electronTemperature: profiles,
            electronDensity: profiles
        )

        let config = try ValidationConfigMatcher.matchToTorax(toraxData)

        // Verify mesh matches TORAX
        #expect(config.runtime.static.mesh.cellCount == nRho, "Mesh size should match TORAX rho grid")

        // Verify time range matches TORAX
        #expect(config.time.start == 0.0, "Start time should match TORAX")
        #expect(config.time.end == 2.0, "End time should match TORAX")

        // Verify save interval matches TORAX sampling
        let expectedInterval = 2.0 / Float(timeCount - 1)
        #expect(abs(config.output.saveInterval! - expectedInterval) < 1e-6, "Save interval should match TORAX")
    }

    @Test("ValidationConfigError for invalid mesh size - too small")
    func invalidMeshSize() throws {
        // Create TORAX data with too few cells
        let time: [Float] = [0.0, 1.0, 2.0]
        let normalizedRadius: [Float] = [0.0, 0.5, 1.0]  // Only 3 cells (< 10 minimum)

        let profiles = (0..<3).map { _ in [Float(1000), 500, 100] }

        let toraxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: profiles,
            electronTemperature: profiles,
            electronDensity: profiles
        )

        #expect(throws: ValidationConfigError.self) {
            try ValidationConfigMatcher.matchToTorax(toraxData)
        }
    }

    @Test("ValidationConfigError for invalid mesh size - too large")
    func invalidMeshSizeTooLarge() throws {
        // Create TORAX data with too many cells (> 200 maximum)
        let time: [Float] = [0.0, 1.0, 2.0]
        let normalizedRadius: [Float] = (0...250).map { Float($0) / 250.0 }  // 251 cells (> 200 maximum)

        let profiles = (0..<3).map { _ in Array(repeating: Float(1000), count: 251) }

        let toraxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: profiles,
            electronTemperature: profiles,
            electronDensity: profiles
        )

        #expect(throws: ValidationConfigError.self) {
            try ValidationConfigMatcher.matchToTorax(toraxData)
        }
    }

    @Test("Edge detection with dynamic rho finding")
    func edgeDetectionDynamic() throws {
        // Create TORAX data with proper ascending rho (minimum 10 cells required)
        let time: [Float] = [0.0, 1.0, 2.0]
        let normalizedRadius: [Float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.95, 0.98, 1.0]  // 10 cells with edge at index 9

        // Create profiles with edge values clearly identifiable
        let ionTemperature: [[Float]] = [
            [15000, 13000, 11000, 9000, 7000, 4000, 2000, 500, 200, 100],  // Edge = 100 eV at index 9
            [15100, 13100, 11100, 9100, 7100, 4100, 2100, 510, 210, 110],
            [15200, 13200, 11200, 9200, 7200, 4200, 2200, 520, 220, 120]
        ]

        let electronTemperature = ionTemperature
        let electronDensity: [[Float]] = [
            [1e20, 9e19, 8e19, 7e19, 5e19, 3e19, 1e19, 5e18, 3e18, 2e18],  // Edge = 2e18 m⁻³
            [1.1e20, 9.9e19, 8.8e19, 7.7e19, 5.5e19, 3.3e19, 1.1e19, 5.5e18, 3.3e18, 2.2e18],
            [1.2e20, 10.8e19, 9.6e19, 8.4e19, 6e19, 3.6e19, 1.2e19, 6e18, 3.6e18, 2.4e18]
        ]

        let toraxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: ionTemperature,
            electronTemperature: electronTemperature,
            electronDensity: electronDensity
        )

        let config = try ValidationConfigMatcher.matchToTorax(toraxData)

        // Verify that edge boundary conditions are taken from rho ≈ 1.0 (index 9)
        #expect(config.runtime.dynamic.boundaries.ionTemperature == 100.0,
                "Ti edge should be from rho=1.0 (index 9)")
        #expect(config.runtime.dynamic.boundaries.electronTemperature == 100.0,
                "Te edge should be from rho=1.0 (index 9)")
        #expect(config.runtime.dynamic.boundaries.electronDensity == 2e18,
                "ne edge should be from rho=1.0 (index 9)")

        print("✅ Edge detection correctly found rho=1.0 at index 9")
        print("   Ti edge: \(config.runtime.dynamic.boundaries.ionTemperature) eV")
        print("   Te edge: \(config.runtime.dynamic.boundaries.electronTemperature) eV")
        print("   ne edge: \(config.runtime.dynamic.boundaries.electronDensity) m⁻³")
    }

    @Test("Edge rho validation rejects non-normalized grids")
    func edgeRhoValidation() throws {
        // Create TORAX data where maximum rho is not close to 1.0
        let time: [Float] = [0.0, 1.0]
        let normalizedRadius: [Float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  // Max = 0.9, not 1.0!

        let profiles = (0..<2).map { _ in Array(repeating: Float(5000), count: 10) }

        let toraxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: profiles,
            electronTemperature: profiles,
            electronDensity: profiles
        )

        // Should throw because max rho (0.9) is not close to 1.0
        #expect(throws: ValidationConfigError.self) {
            try ValidationConfigMatcher.matchToTorax(toraxData)
        }
    }

    @Test("ValidationConfigError for empty time array")
    func emptyTimeArray() throws {
        // Create TORAX data with empty time array
        let time: [Float] = []
        let normalizedRadius: [Float] = [0.0, 1.0]

        let toraxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: [],
            electronTemperature: [],
            electronDensity: []
        )

        #expect(throws: ValidationConfigError.self) {
            try ValidationConfigMatcher.matchToTorax(toraxData)
        }
    }

    @Test("Compare with TORAX reference data")
    func compareWithTorax() throws {
        // Create mock reference data with sufficient points (10) for correlation
        let time: [Float] = [0.0, 1.0, 2.0]
        let normalizedRadius: [Float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  // 11 points

        // Temperature profiles (parabolic)
        let Ti_ref: [[Float]] = [
            [10000, 9500, 8500, 7000, 5500, 4000, 2800, 1800, 1000, 500, 100],
            [12000, 11400, 10200, 8400, 6600, 4800, 3360, 2160, 1200, 600, 120],
            [15000, 14250, 12750, 10500, 8250, 6000, 4200, 2700, 1500, 750, 150]
        ]

        let Te_ref = Ti_ref

        // Density profiles (linear)
        let ne_ref: [[Float]] = [
            [1.0e20, 0.9e20, 0.8e20, 0.7e20, 0.6e20, 0.5e20, 0.4e20, 0.3e20, 0.2e20, 0.15e20, 0.1e20],
            [1.1e20, 0.99e20, 0.88e20, 0.77e20, 0.66e20, 0.55e20, 0.44e20, 0.33e20, 0.22e20, 0.165e20, 0.11e20],
            [1.2e20, 1.08e20, 0.96e20, 0.84e20, 0.72e20, 0.6e20, 0.48e20, 0.36e20, 0.24e20, 0.18e20, 0.12e20]
        ]

        let toraxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: Ti_ref,
            electronTemperature: Te_ref,
            electronDensity: ne_ref
        )

        // Create mock Gotenx output (0.5% difference)
        let Ti_gotenx: [[Float]] = Ti_ref.map { profile in
            profile.map { $0 * 1.005 }  // 0.5% higher
        }

        let Te_gotenx = Ti_gotenx

        let ne_gotenx: [[Float]] = ne_ref.map { profile in
            profile.map { $0 * 1.01 }  // 1% higher
        }

        let gotenxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: Ti_gotenx,
            electronTemperature: Te_gotenx,
            electronDensity: ne_gotenx
        )

        // Compare
        let results = ValidationConfigMatcher.compareWithTorax(
            gotenx: gotenxData,
            torax: toraxData,
            thresholds: .torax
        )

        // Should have 9 results (3 quantities × 3 time points)
        #expect(results.count == 9, "Should have 9 comparison results")

        // All should pass (small differences)
        let allPassed = results.allSatisfy { $0.passed }
        #expect(allPassed, "All comparisons should pass with small differences")
    }

    @Test("Compare with large differences fails validation")
    func compareWithLargeDifferences() throws {
        // Create mock reference data
        let time: [Float] = [0.0, 1.0]
        let normalizedRadius: [Float] = [0.0, 1.0]

        let Ti_ref: [[Float]] = [
            [10000, 100],
            [12000, 120]
        ]

        let Te_ref = Ti_ref
        let ne_ref: [[Float]] = [
            [1e20, 1e19],
            [1.1e20, 1.1e19]
        ]

        let toraxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: Ti_ref,
            electronTemperature: Te_ref,
            electronDensity: ne_ref
        )

        // Create Gotenx output with large differences (50% higher)
        let Ti_gotenx: [[Float]] = [
            [15000, 150],
            [18000, 180]
        ]

        let Te_gotenx = Ti_gotenx
        let ne_gotenx: [[Float]] = [
            [1.5e20, 1.5e19],
            [1.65e20, 1.65e19]
        ]

        let gotenxData = TORAXReferenceData(
            time: time,
            normalizedRadius: normalizedRadius,
            ionTemperature: Ti_gotenx,
            electronTemperature: Te_gotenx,
            electronDensity: ne_gotenx
        )

        // Compare
        let results = ValidationConfigMatcher.compareWithTorax(
            gotenx: gotenxData,
            torax: toraxData,
            thresholds: .torax
        )

        // Should have 6 results (3 quantities × 2 time points)
        #expect(results.count == 6, "Should have 6 comparison results")

        // All should fail (large differences)
        let anyPassed = results.contains { $0.passed }
        #expect(!anyPassed, "No comparisons should pass with 50% differences")
    }

    @Test("ValidationConfigError descriptions")
    func validationConfigErrorDescriptions() throws {
        let error1 = ValidationConfigError.invalidMeshSize(5)
        #expect(error1.description.contains("5"))
        #expect(error1.description.contains("10-200"))

        let error2 = ValidationConfigError.emptyTimeArray
        #expect(error2.description.contains("empty"))

        let error3 = ValidationConfigError.incompatibleTimeRanges("test message")
        #expect(error3.description.contains("test message"))
    }
}
