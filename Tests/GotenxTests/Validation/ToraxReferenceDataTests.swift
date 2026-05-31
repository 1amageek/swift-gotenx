// ToraxReferenceDataTests.swift
// Tests for TORAX NetCDF data loading

import Testing
import Foundation
import SwiftNetCDF
@testable import GotenxCore

@Suite("TORAX Reference Data Loading")
struct ToraxReferenceDataTests {

    @Test("Load mock TORAX NetCDF file")
    func testLoadMockToraxData() throws {
        // Create mock TORAX NetCDF file
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("mock_torax_test.nc").path

        // Clean up any existing file
        removeTestItemIfExists(atPath: filePath)

        // Create mock TORAX file
        let (nTime, nRho) = try createMockToraxFile(path: filePath)

        // Load data using the extension method
        let data = try ToraxReferenceData.loadFromNetCDF(path: filePath)

        // Verify dimensions
        #expect(data.time.count == nTime, "Should have \(nTime) time points")
        #expect(data.rho.count == nRho, "Should have \(nRho) rho points")

        // Verify profiles
        #expect(data.Ti.count == nTime, "Ti should have \(nTime) time points")
        #expect(data.Ti[0].count == nRho, "Ti[0] should have \(nRho) rho points")
        #expect(data.Te.count == nTime, "Te should have \(nTime) time points")
        #expect(data.Te[0].count == nRho, "Te[0] should have \(nRho) rho points")
        #expect(data.ne.count == nTime, "ne should have \(nTime) time points")
        #expect(data.ne[0].count == nRho, "ne[0] should have \(nRho) rho points")

        // Verify data ranges (realistic plasma values)
        #expect(data.Ti[0][0] > 0, "Ti should be positive")
        #expect(data.Ti[0][0] < 100000, "Ti should be reasonable (< 100 keV)")
        #expect(data.Te[0][0] > 0, "Te should be positive")
        #expect(data.ne[0][0] > 0, "ne should be positive")
        #expect(data.ne[0][0] < 1e21, "ne should be reasonable (< 10²¹ m⁻³)")

        // Verify psi is optional
        #expect(data.psi != nil, "psi should be present in mock file")
        if let psi = data.psi {
            #expect(psi.count == nTime, "psi should have \(nTime) time points")
            #expect(psi[0].count == nRho, "psi[0] should have \(nRho) rho points")
        }

        // Clean up
        removeTestItemIfExists(atPath: filePath)

        print("✅ Successfully loaded mock TORAX data:")
        print("   Time points: \(data.time.count)")
        print("   Grid size: \(data.rho.count)")
        print("   Ti range: \(data.Ti.flatMap { $0 }.min()!) - \(data.Ti.flatMap { $0 }.max()!) eV")
        print("   Te range: \(data.Te.flatMap { $0 }.min()!) - \(data.Te.flatMap { $0 }.max()!) eV")
        print("   ne range: \(data.ne.flatMap { $0 }.min()!) - \(data.ne.flatMap { $0 }.max()!) m⁻³")
    }

    @Test("Load TORAX file without poloidal flux")
    func testLoadWithoutPsi() throws {
        // Create mock file without psi
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("mock_torax_no_psi.nc").path

        // Clean up any existing file
        removeTestItemIfExists(atPath: filePath)

        try createMockToraxFile(path: filePath, includePsi: false)

        // Load data
        let data = try ToraxReferenceData.loadFromNetCDF(path: filePath)

        // Verify psi is nil
        #expect(data.psi == nil, "psi should be nil when not present in file")

        // Clean up
        removeTestItemIfExists(atPath: filePath)

        print("✅ Successfully loaded TORAX data without psi")
    }

    @Test("Error: File not found")
    func testFileNotFound() throws {
        #expect(throws: ToraxDataError.self) {
            try ToraxReferenceData.loadFromNetCDF(path: "/nonexistent/path.nc")
        }
    }

    @Test("Error: Invalid dimensions (too few cells)")
    func testInvalidDimensions() throws {
        // Create file with only 5 rho points (< 10 minimum)
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("mock_torax_invalid_dims.nc").path

        // Clean up any existing file
        removeTestItemIfExists(atPath: filePath)

        try createMockToraxFile(path: filePath, nRho: 5)

        #expect(throws: ToraxDataError.self) {
            try ToraxReferenceData.loadFromNetCDF(path: filePath)
        }

        // Clean up
        removeTestItemIfExists(atPath: filePath)
    }

    @Test("Time utilities: findTimeIndex")
    func testFindTimeIndex() throws {
        // Create simple mock data
        let toraxData = ToraxReferenceData(
            time: [0.0, 0.5, 1.0, 1.5, 2.0],
            rho: [0.0, 0.5, 1.0],
            Ti: Array(repeating: Array(repeating: Float(1000.0), count: 3), count: 5),
            Te: Array(repeating: Array(repeating: Float(1000.0), count: 3), count: 5),
            ne: Array(repeating: Array(repeating: Float(1e20), count: 3), count: 5)
        )

        // Test exact match
        #expect(toraxData.findTimeIndex(closestTo: 1.0) == 2)

        // Test closest match
        #expect(toraxData.findTimeIndex(closestTo: 0.7) == 1)  // Closer to 0.5 (diff=0.2) than 1.0 (diff=0.3)
        #expect(toraxData.findTimeIndex(closestTo: 0.3) == 1)  // Closer to 0.5 (diff=0.2) than 0.0 (diff=0.3)

        // Test edge cases
        #expect(toraxData.findTimeIndex(closestTo: -1.0) == 0)  // Before start
        #expect(toraxData.findTimeIndex(closestTo: 10.0) == 4)  // After end
    }

    @Test("Time utilities: getProfiles")
    func testGetProfiles() throws {
        // Create mock data with varying profiles
        let time: [Float] = [0.0, 1.0, 2.0]
        let rho: [Float] = [0.0, 0.5, 1.0]

        let Ti: [[Float]] = [
            [15000.0, 10000.0, 100.0],  // t=0
            [16000.0, 11000.0, 110.0],  // t=1
            [17000.0, 12000.0, 120.0]   // t=2
        ]

        let toraxData = ToraxReferenceData(
            time: time,
            rho: rho,
            Ti: Ti,
            Te: Ti,  // Same as Ti for simplicity
            ne: Array(repeating: Array(repeating: Float(1e20), count: 3), count: 3)
        )

        // Test getProfiles(at:)
        let profiles_t1 = toraxData.getProfiles(at: 1)
        #expect(profiles_t1.time == 1.0)
        #expect(profiles_t1.Ti[0] == 16000.0)
        #expect(profiles_t1.Ti[1] == 11000.0)

        // Test getProfiles(closestTo:)
        let profiles_near_1_5 = toraxData.getProfiles(closestTo: 1.5)
        // 1.5 is equidistant from 1.0 and 2.0, but findTimeIndex returns first match (index 1)
        #expect(profiles_near_1_5.time == 1.0)  // Returns first equidistant point
        #expect(profiles_near_1_5.Ti[0] == 16000.0)
    }

    // MARK: - Helper: Create Mock TORAX File

    /// Create a mock TORAX NetCDF file for testing
    ///
    /// - Parameters:
    ///   - path: File path
    ///   - nTime: Number of time points (default: 50)
    ///   - nRho: Number of radial points (default: 50)
    ///   - includePsi: Include poloidal flux variable (default: true)
    /// - Returns: Tuple of (nTime, nRho)
    @discardableResult
    private func createMockToraxFile(
        path: String,
        nTime: Int = 50,
        nRho: Int = 50,
        includePsi: Bool = true
    ) throws -> (Int, Int) {
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
        let timeData: [Float] = (0..<nTime).map { Float($0) * 2.0 / Float(nTime - 1) }  // 0 to 2 seconds
        let rhoData: [Float] = (0..<nRho).map { Float($0) / Float(nRho - 1) }  // 0 to 1

        try timeVar.write(timeData)
        try rhoVar.write(rhoData)

        // Create profile variables with realistic ITER-like data
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

            // Generate realistic profile data
            let data: [Float] = (0..<(nTime * nRho)).map { i in
                let timeIdx = i / nRho
                let rhoIdx = i % nRho
                let rho = Float(rhoIdx) / Float(nRho - 1)
                let time = Float(timeIdx) / Float(nTime - 1)

                switch varSpec.name {
                case "ion_temperature", "electron_temperature":
                    // Parabolic profile: T = T₀(1 - ρ²)^2 with time evolution
                    let T0 = 15000.0 * (1.0 + 0.1 * sin(time * 2.0 * Float.pi))  // 13.5-16.5 keV
                    return T0 * pow(1.0 - rho * rho, 2.0)

                case "electron_density":
                    // Parabolic density: n = n₀(1 - ρ²)^1.5
                    let n0: Float = 1.0e20 * (1.0 + 0.05 * sin(time * 2.0 * Float.pi))
                    let alpha: Float = 1.5
                    return n0 * pow(1.0 - rho * rho, alpha)

                default:
                    return 0.0
                }
            }

            try variable.write(data, offset: [0, 0], count: [nTime, nRho])
        }

        // Optionally create poloidal flux variable
        if includePsi {
            var psiVar = try file.createVariable(
                name: "poloidal_flux",
                type: Float.self,
                dimensions: [timeDim, rhoDim]
            )

            try psiVar.setAttribute("long_name", "poloidal flux")
            try psiVar.setAttribute("units", "Wb")

            let psiData: [Float] = (0..<(nTime * nRho)).map { i in
                let rhoIdx = i % nRho
                let rho = Float(rhoIdx) / Float(nRho - 1)
                let psi0: Float = 10.0  // Wb
                return psi0 * rho * rho
            }

            try psiVar.write(psiData, offset: [0, 0], count: [nTime, nRho])
        }

        file.sync()

        return (nTime, nRho)
    }
}
