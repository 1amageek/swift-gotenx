// ValidationIntegrationTest.swift
// Integration test for Phase 6 validation tools

import Testing
import Foundation
import MLX
import SwiftNetCDF
@testable import GotenxCore
@testable import GotenxPhysics
@testable import GotenxCLI

/// Validation Integration Test Suite
///
/// Tests Phase 6 validation tools with simulated data:
/// 1. Create mock NetCDF files mimicking Gotenx output format
/// 2. Load using ToraxReferenceDataLoader
/// 3. Validate physical quantities
/// 4. Test ValidationConfigMatcher
@Suite("Validation Integration Tests")
struct ValidationIntegrationTest {

    // MARK: - Mock Data Generation

    /// Create mock Gotenx-format NetCDF file
    func createMockGotenxOutput(path: String, nTime: Int = 6, nRho: Int = 50) throws -> (Int, Int) {
        // Create NetCDF file
        let file = try NetCDF.create(path: path, overwriteExisting: true, useNetCDF4: true)

        // Define dimensions
        let timeDim = try file.createDimension(name: "time", length: nTime)
        let rhoDim = try file.createDimension(name: "rho_tor_norm", length: nRho)

        // Create coordinate variables
        var timeVar = try file.createVariable(name: "time", type: Float.self, dimensions: [timeDim])
        try timeVar.setAttribute("long_name", "simulation time")
        try timeVar.setAttribute("units", "s")

        var rhoVar = try file.createVariable(name: "rho_tor_norm", type: Float.self, dimensions: [rhoDim])
        try rhoVar.setAttribute("long_name", "normalized toroidal flux coordinate")
        try rhoVar.setAttribute("units", "1")

        // Write coordinate data
        let timeData: [Float] = (0..<nTime).map { Float($0) * 0.5 / Float(nTime - 1) }  // 0 to 0.5 seconds
        let rhoData: [Float] = (0..<nRho).map { Float($0) / Float(nRho - 1) }  // 0 to 1

        try timeVar.write(timeData)
        try rhoVar.write(rhoData)

        // Create profile variables
        let variables: [(name: String, longName: String, units: String)] = [
            ("ion_temperature", "ion temperature", "eV"),
            ("electron_temperature", "electron temperature", "eV"),
            ("electron_density", "electron density", "m-3")
        ]

        for varSpec in variables {
            var variable = try file.createVariable(
                name: varSpec.name,
                type: Float.self,
                dimensions: [timeDim, rhoDim]
            )

            try variable.setAttribute("long_name", varSpec.longName)
            try variable.setAttribute("units", varSpec.units)

            // Generate ITER-like profiles
            let data: [Float] = (0..<(nTime * nRho)).map { i in
                let rhoIdx = i % nRho
                let rho = Float(rhoIdx) / Float(nRho - 1)

                switch varSpec.name {
                case "ion_temperature", "electron_temperature":
                    // Parabolic: T = T₀(1 - ρ²)²
                    let T0: Float = 20000.0  // 20 keV peak
                    let T_edge: Float = 100.0
                    return T_edge + (T0 - T_edge) * pow(1.0 - rho * rho, 2.0)

                case "electron_density":
                    // Linear: n = n₀(1 - ρ)
                    let n0: Float = 1.0e20
                    let n_edge: Float = 1.0e19
                    return n_edge + (n0 - n_edge) * (1.0 - rho)

                default:
                    return 0.0
                }
            }

            try variable.write(data, offset: [0, 0], count: [nTime, nRho])
        }

        file.sync()
        return (nTime, nRho)
    }

    // MARK: - Tests

    @Test("Load mock Gotenx output with ToraxReferenceDataLoader")
    func testLoadMockGotenxOutput() throws {
        // Setup temp file
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("mock_gotenx_output.nc").path

        // Clean up any existing file
        removeTestItemIfExists(atPath: filePath)

        // Create mock output
        let (nTime, nRho) = try createMockGotenxOutput(path: filePath)

        // Load using ToraxReferenceDataLoader
        let data = try ToraxReferenceData.loadFromNetCDF(path: filePath)

        // Verify dimensions
        #expect(data.time.count == nTime, "Should have \(nTime) time points")
        #expect(data.rho.count == nRho, "Should have \(nRho) rho points")

        // Verify profiles
        #expect(data.Ti.count == nTime, "Ti should have \(nTime) time points")
        #expect(data.Ti[0].count == nRho, "Ti[0] should have \(nRho) rho points")

        // Clean up
        removeTestItemIfExists(atPath: filePath)

        print("✅ Successfully loaded mock Gotenx output")
        print("   Time points: \(data.time.count)")
        print("   Grid size: \(data.rho.count)")
    }

    @Test("Validate physical quantities in mock output")
    func testPhysicalQuantityValidation() throws {
        // Setup temp file
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("mock_gotenx_physics.nc").path

        // Clean up any existing file
        removeTestItemIfExists(atPath: filePath)

        // Create mock output
        try createMockGotenxOutput(path: filePath)

        // Load data
        let data = try ToraxReferenceData.loadFromNetCDF(path: filePath)

        // Get final state
        let finalIdx = data.time.count - 1
        let Ti_final = data.Ti[finalIdx]
        let Te_final = data.Te[finalIdx]
        let ne_final = data.ne[finalIdx]

        // Core values (rho ≈ 0)
        let Ti_core = Ti_final[0]
        let Te_core = Te_final[0]
        let ne_core = ne_final[0]

        // Edge values (rho ≈ 1.0)
        let edgeIdx = data.rho.count - 1
        let Ti_edge = Ti_final[edgeIdx]
        let Te_edge = Te_final[edgeIdx]
        let ne_edge = ne_final[edgeIdx]

        print("   Core values:")
        print("     Ti: \(Ti_core / 1000) keV")
        print("     Te: \(Te_core / 1000) keV")
        print("     ne: \(ne_core / 1e20) × 10²⁰ m⁻³")
        print("   Edge values:")
        print("     Ti: \(Ti_edge) eV")
        print("     Te: \(Te_edge) eV")
        print("     ne: \(ne_edge / 1e19) × 10¹⁹ m⁻³")

        // Validate temperature ranges
        #expect(Ti_core > 10000, "Core Ti should be > 10 keV")
        #expect(Ti_core < 50000, "Core Ti should be < 50 keV")
        #expect(Te_core > 10000, "Core Te should be > 10 keV")

        // Validate density ranges
        #expect(ne_core > 5e19, "Core density should be > 5×10¹⁹ m⁻³")
        #expect(ne_core < 2e20, "Core density should be < 2×10²⁰ m⁻³")

        // Validate edge < core
        #expect(Ti_edge < Ti_core, "Edge Ti should be less than core Ti")
        #expect(Te_edge < Te_core, "Edge Te should be less than core Te")
        #expect(ne_edge < ne_core, "Edge density should be less than core density")

        // Clean up
        removeTestItemIfExists(atPath: filePath)

        print("✅ All physical quantity validations passed")
    }

    @Test("ValidationConfigMatcher with mock output")
    func testValidationConfigMatcherWithMockOutput() throws {
        // Setup temp file
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("mock_gotenx_matcher.nc").path

        // Clean up any existing file
        removeTestItemIfExists(atPath: filePath)

        // Create mock output
        try createMockGotenxOutput(path: filePath)

        // Load data
        let data = try ToraxReferenceData.loadFromNetCDF(path: filePath)

        // Generate config matching the data
        let config = try ValidationConfigMatcher.matchToTorax(data)

        // Verify mesh size matches
        #expect(config.runtime.static.mesh.nCells == data.rho.count,
                "Matched config mesh should equal data grid size")

        // Verify time range matches
        #expect(config.time.start == data.time.first!,
                "Matched config start time should equal data start time")
        #expect(config.time.end == data.time.last!,
                "Matched config end time should equal data end time")

        // Verify geometry matches ITER parameters
        #expect(config.runtime.static.mesh.majorRadius == 6.2,
                "Matched config should use ITER major radius")

        // Verify boundary conditions extracted from edge
        let edgeIdx = data.rho.count - 1
        let expectedTi = data.Ti[0][edgeIdx]
        let expectedTe = data.Te[0][edgeIdx]
        let expectedNe = data.ne[0][edgeIdx]

        #expect(config.runtime.dynamic.boundaries.ionTemperature == expectedTi,
                "Boundary Ti should match data edge value")
        #expect(config.runtime.dynamic.boundaries.electronTemperature == expectedTe,
                "Boundary Te should match data edge value")
        #expect(config.runtime.dynamic.boundaries.density == expectedNe,
                "Boundary ne should match data edge value")

        // Clean up
        removeTestItemIfExists(atPath: filePath)

        print("✅ ValidationConfigMatcher test passed")
        print("   Matched config parameters:")
        print("     Mesh size: \(config.runtime.static.mesh.nCells)")
        print("     Time range: \(config.time.start) - \(config.time.end) s")
    }

    @Test("Profile comparison with self-consistency")
    func testProfileComparison() throws {
        // Setup temp file
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("mock_gotenx_compare.nc").path

        // Clean up any existing file
        removeTestItemIfExists(atPath: filePath)

        // Create mock output
        try createMockGotenxOutput(path: filePath)

        // Load data
        let data = try ToraxReferenceData.loadFromNetCDF(path: filePath)

        // Compare data with itself (should have perfect correlation)
        let results = ValidationConfigMatcher.compareWithTorax(
            gotenx: data,
            torax: data,
            thresholds: .torax
        )

        print("   Comparison results: \(results.count) comparisons")

        // All comparisons should pass with perfect agreement
        let allPassed = results.allSatisfy { $0.passed }
        #expect(allPassed, "Self-comparison should pass perfectly")

        // Check L2 error values (should be ~0 for self-comparison)
        for result in results {
            #expect(result.l2Error < 1e-6,
                    "Self-comparison L2 error should be ~0, got \(result.l2Error)")

            // Correlation can be NaN for constant profiles (zero variance)
            // This is expected for mock data without time evolution
            if !result.correlation.isNaN {
                #expect(abs(result.correlation - 1.0) < 1e-6,
                        "Self-comparison correlation should be 1.0, got \(result.correlation)")
            }
        }

        // Clean up
        removeTestItemIfExists(atPath: filePath)

        print("✅ Self-consistency test passed")
        print("   All \(results.count) comparisons had perfect agreement")
    }
}
