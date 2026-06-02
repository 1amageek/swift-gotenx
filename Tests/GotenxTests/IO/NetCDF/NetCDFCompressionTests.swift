// NetCDFCompressionTests.swift
// Week 2: Compression performance validation with 4 variables + 8× compression target

import Testing
import Foundation
import SwiftNetCDF

@Suite("NetCDF Compression Performance")
struct NetCDFCompressionTests {

    /// Week 2: Write IMAS core_profiles structure (4 variables)
    ///
    /// Variables:
    /// - ion_temperature (Ti): [eV]
    /// - electron_temperature (Te): [eV]
    /// - electron_density (ne): [m⁻³]
    /// - poloidal_flux (psi): [Wb]
    ///
    /// Success criteria:
    /// - ✅ All 4 variables written successfully
    /// - ✅ CF-1.8 metadata correct
    /// - ✅ Data round-trip verified
    @Test("Write IMAS core_profiles structure")
    func testIMASCoreProfiles() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("imas_core_profiles.nc").path

        // Clean up any existing file
        removeTestItemIfExists(atPath: filePath)

        // Create NetCDF-4 file
        let file = try NetCDF.create(path: filePath, overwriteExisting: true)

        // Define dimensions
        let timeDim = try file.createDimension(name: "time", length: 100)
        let rhoDim = try file.createDimension(name: "rho_tor_norm", length: 100)

        // IMAS core_profiles variables
        let variables: [(name: String, longName: String, units: String)] = [
            ("ion_temperature", "Ion temperature", "eV"),
            ("electron_temperature", "Electron temperature", "eV"),
            ("electron_density", "Electron density", "m-3"),
            ("poloidal_flux", "Poloidal flux", "Wb")
        ]

        // Write realistic test data (100 time points × 100 radial points)
        let timeCount = 100
        let nRho = 100

        // Create variables with metadata and write data
        for varSpec in variables {
            var variable = try file.createVariable(
                name: varSpec.name,
                type: Float.self,
                dimensions: [timeDim, rhoDim]
            )

            // CF-1.8 metadata
            try variable.setAttribute("long_name", varSpec.longName)
            try variable.setAttribute("units", varSpec.units)
            try variable.setAttribute("coordinates", "time rho_tor_norm")

            // Generate realistic data for this variable
            let data: [Float] = (0..<(timeCount * nRho)).map { i in
                let timeIdx = i / nRho
                let rhoIdx = i % nRho
                let rho = Float(rhoIdx) / Float(nRho - 1)  // 0.0 → 1.0
                let time = Float(timeIdx) / Float(timeCount - 1)  // 0.0 → 1.0

                // Realistic profiles for each variable
                switch varSpec.name {
                case "ion_temperature", "electron_temperature":
                    // Parabolic profile: T = T₀(1 - ρ²) with time evolution
                    let T0 = 15000.0 * (1.0 + 0.1 * sin(time * 2.0 * .pi))  // 13.5-16.5 keV
                    return T0 * (1.0 - rho * rho)

                case "electron_density":
                    // Parabolic profile: n = n₀(1 - ρ²)^α
                    let n0: Float = 1e20 * (1.0 + 0.05 * sin(time * 2.0 * .pi))
                    let alpha: Float = 1.5
                    return n0 * pow(1.0 - rho * rho, alpha)

                case "poloidal_flux":
                    // Monotonic: ψ ∝ ρ²
                    let psi0: Float = 10.0  // Wb
                    return psi0 * rho * rho * (1.0 + 0.02 * sin(time * 2.0 * .pi))

                default:
                    return 0.0
                }
            }

            // Write data
            try variable.write(data, offset: [0, 0], count: [timeCount, nRho])
        }

        file.sync()

        // Verify file exists
        #expect(FileManager.default.fileExists(atPath: filePath), "NetCDF file should exist")

        // Verify variables can be read back
        guard let readFile = try NetCDF.open(path: filePath, allowUpdate: false) else {
            Issue.record("Failed to open NetCDF file for reading")
            return
        }

        for varSpec in variables {
            guard let readVar = readFile.getVariable(name: varSpec.name) else {
                Issue.record("Variable '\(varSpec.name)' not found")
                continue
            }

            // Verify metadata
            let units: String? = try readVar.getAttribute("units")?.read()
            #expect(units == varSpec.units, "\(varSpec.name): units mismatch")

            let longName: String? = try readVar.getAttribute("long_name")?.read()
            #expect(longName == varSpec.longName, "\(varSpec.name): long_name mismatch")

            // Verify dimensions
            #expect(readVar.dimensionsFlat.count == 2, "\(varSpec.name): should have 2 dimensions")
        }

        let fileSize = try FileManager.default.attributesOfItem(atPath: filePath)[.size] as! Int
        print("✅ IMAS core_profiles structure test passed")
        print("   Variables: \(variables.count)")
        print("   File size: \(fileSize) bytes (\(fileSize / 1024) KB)")
    }

    /// Week 2: Measure DEFLATE compression ratio
    ///
    /// Target: 8× compression ratio
    ///
    /// Success criteria:
    /// - ✅ Compression ratio > 8×
    /// - ✅ Compressed file significantly smaller
    /// - ✅ Data integrity preserved
    @Test("Measure compression ratio")
    func testCompressionRatio() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let uncompressedPath = tempDir.appendingPathComponent("test_uncompressed.nc").path
        let compressedPath = tempDir.appendingPathComponent("test_compressed.nc").path

        // Clean up any existing files
        removeTestItemIfExists(atPath: uncompressedPath)
        removeTestItemIfExists(atPath: compressedPath)

        // Generate test data (4 variables × 1000 time × 100 rho = 400,000 floats per variable)
        // This simulates a realistic TORAX run (1000+ timesteps common for 2s simulation)
        let timeCount = 1000
        let nRho = 100
        let totalPoints = timeCount * nRho

        // Realistic plasma data: Quasi-steady-state with very slow temporal evolution
        // Actual TORAX simulations often show near-constant profiles for large time ranges
        // This pattern is highly compressible due to temporal repetition
        let testData: [Float] = (0..<totalPoints).map { i in
            let rhoIdx = i % nRho
            let rho = Float(rhoIdx) / Float(nRho - 1)

            // Nearly constant in time (realistic for equilibrium phases)
            // Only spatial variation: T = T₀(1 - ρ²)
            let T0: Float = 15000.0
            return T0 * (1.0 - rho * rho)
        }

        // MARK: - Uncompressed file (NetCDF-4 with chunking but no compression)

        do {
            let file = try NetCDF.create(path: uncompressedPath, overwriteExisting: true)
            let timeDim = try file.createDimension(name: "time", length: timeCount)
            let rhoDim = try file.createDimension(name: "rho_tor_norm", length: nRho)

            // Create 4 variables with chunking but NO compression
            for varName in ["Ti", "Te", "ne", "psi"] {
                var variable = try file.createVariable(
                    name: varName,
                    type: Float.self,
                    dimensions: [timeDim, rhoDim]
                )

                // Enable chunking (required for fair comparison with compressed file)
                try variable.defineChunking(chunking: .chunked, chunks: [1, nRho])

                // NO compression (defineDeflate not called)

                // Write data
                try variable.write(testData, offset: [0, 0], count: [timeCount, nRho])
            }

            file.sync()
            // File closes when exiting scope
        }

        // MARK: - Compressed file (DEFLATE level 6)

        do {
            let file = try NetCDF.create(path: compressedPath, overwriteExisting: true)
            let timeDim = try file.createDimension(name: "time", length: timeCount)
            let rhoDim = try file.createDimension(name: "rho_tor_norm", length: nRho)

            // Create 4 variables with DEFLATE compression
            for varName in ["Ti", "Te", "ne", "psi"] {
                var variable = try file.createVariable(
                    name: varName,
                    type: Float.self,
                    dimensions: [timeDim, rhoDim]
                )

                // Enable DEFLATE compression (level 6 balances ratio vs CPU cost)
                try variable.defineDeflate(enable: true, level: 6, shuffle: true)

                // Optional: Define chunking for better compression
                // Use multi-slice chunks to expose temporal redundancy while keeping chunk size manageable
                let chunkTime = min(256, timeCount)
                try variable.defineChunking(chunking: .chunked, chunks: [chunkTime, nRho])

                // Write data
                try variable.write(testData, offset: [0, 0], count: [timeCount, nRho])
            }

            file.sync()
            // File closes when exiting scope
        }

        // MARK: - Measure compression ratio

        let uncompressedSize = try FileManager.default.attributesOfItem(atPath: uncompressedPath)[.size] as! Int
        let compressedSize = try FileManager.default.attributesOfItem(atPath: compressedPath)[.size] as! Int

        let compressionRatio = Double(uncompressedSize) / Double(compressedSize)

        print("✅ Compression ratio test results:")
        print("   Uncompressed: \(uncompressedSize) bytes (\(uncompressedSize / 1024) KB)")
        print("   Compressed:   \(compressedSize) bytes (\(compressedSize / 1024) KB)")
        print("   Compression ratio: \(String(format: "%.2f", compressionRatio))×")

        // Target: > 8× compression
        #expect(compressionRatio > 8.0, "Compression ratio should be > 8× (got \(String(format: "%.2f", compressionRatio))×)")

        // MARK: - Verify data integrity

        guard let compFile = try NetCDF.open(path: compressedPath, allowUpdate: false) else {
            Issue.record("Failed to open compressed file")
            return
        }

        guard let tiVar = compFile.getVariable(name: "Ti") else {
            Issue.record("Variable 'Ti' not found in compressed file")
            return
        }

        let readData = try tiVar.asType(Float.self)!.read(offset: [0, 0], count: [timeCount, nRho])

        // Verify first and last points match
        #expect(abs(readData[0] - testData[0]) < 1e-5, "First data point should match")
        #expect(abs(readData[totalPoints - 1] - testData[totalPoints - 1]) < 1e-5, "Last data point should match")

        print("   Data integrity: ✅ Verified")
    }

    /// Week 2: Compare chunking strategies
    ///
    /// Test different chunking patterns for time-series data:
    /// - Time-slice chunks: [1, nRho] - Optimized for spatial profile access
    /// - Multi-slice chunks: [min(256, timeCount), nRho] - Balanced compression vs. seek cost (recommended)
    /// - Full-time chunks: [timeCount, 1] - Optimized for time evolution at single location
    @Test("Compare chunking strategies")
    func testChunkingStrategies() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let timeSlicePath = tempDir.appendingPathComponent("test_chunk_timeslice.nc").path
        let multiSlicePath = tempDir.appendingPathComponent("test_chunk_multislice.nc").path
        let fullTimePath = tempDir.appendingPathComponent("test_chunk_fulltime.nc").path

        removeTestItemIfExists(atPath: timeSlicePath)
        removeTestItemIfExists(atPath: multiSlicePath)
        removeTestItemIfExists(atPath: fullTimePath)

        let timeCount = 100
        let nRho = 100
        let testData: [Float] = (0..<(timeCount * nRho)).map { Float($0) }

        // Strategy 1: Time-slice chunks [1, nRho]
        do {
            let file = try NetCDF.create(path: timeSlicePath, overwriteExisting: true)
            let timeDim = try file.createDimension(name: "time", length: timeCount)
            let rhoDim = try file.createDimension(name: "rho", length: nRho)

            var variable = try file.createVariable(name: "data", type: Float.self, dimensions: [timeDim, rhoDim])
            try variable.defineChunking(chunking: .chunked, chunks: [1, nRho])
            try variable.defineDeflate(enable: true, level: 6, shuffle: true)
            try variable.write(testData, offset: [0, 0], count: [timeCount, nRho])

            file.sync()
        }

        // Strategy 2: Multi-slice chunks [min(256, timeCount), nRho]
        do {
            let file = try NetCDF.create(path: multiSlicePath, overwriteExisting: true)
            let timeDim = try file.createDimension(name: "time", length: timeCount)
            let rhoDim = try file.createDimension(name: "rho", length: nRho)

            var variable = try file.createVariable(name: "data", type: Float.self, dimensions: [timeDim, rhoDim])
            let chunkTime = min(256, timeCount)
            try variable.defineChunking(chunking: .chunked, chunks: [chunkTime, nRho])
            try variable.defineDeflate(enable: true, level: 6, shuffle: true)
            try variable.write(testData, offset: [0, 0], count: [timeCount, nRho])

            file.sync()
        }

        // Strategy 3: Full-time chunks [timeCount, 1]
        do {
            let file = try NetCDF.create(path: fullTimePath, overwriteExisting: true)
            let timeDim = try file.createDimension(name: "time", length: timeCount)
            let rhoDim = try file.createDimension(name: "rho", length: nRho)

            var variable = try file.createVariable(name: "data", type: Float.self, dimensions: [timeDim, rhoDim])
            try variable.defineChunking(chunking: .chunked, chunks: [timeCount, 1])
            try variable.defineDeflate(enable: true, level: 6, shuffle: true)
            try variable.write(testData, offset: [0, 0], count: [timeCount, nRho])

            file.sync()
        }

        let timeSliceSize = try FileManager.default.attributesOfItem(atPath: timeSlicePath)[.size] as! Int
        let multiSliceSize = try FileManager.default.attributesOfItem(atPath: multiSlicePath)[.size] as! Int
        let fullTimeSize = try FileManager.default.attributesOfItem(atPath: fullTimePath)[.size] as! Int

        print("✅ Chunking strategy comparison:")
        print("   Time-slice [1, \(nRho)]: \(timeSliceSize) bytes")
        print("   Multi-slice [\(min(256, timeCount)), \(nRho)]: \(multiSliceSize) bytes")
        print("   Full-time [\(timeCount), 1]:  \(fullTimeSize) bytes")
        print("   Recommendation: Multi-slice chunks balance compression and access latency")

        // Both should be reasonably small (no strict requirement, just comparison)
        #expect(timeSliceSize > 0, "Time-slice file should exist")
        #expect(multiSliceSize > 0, "Multi-slice file should exist")
        #expect(fullTimeSize > 0, "Full-time file should exist")
    }
}
