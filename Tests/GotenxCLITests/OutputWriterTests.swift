// OutputWriterTests.swift
// Tests for OutputWriter NetCDF functionality

import Testing
import Foundation
@testable import GotenxCLI
import GotenxCore

@Suite("OutputWriter Tests")
struct OutputWriterTests {

    @Test("NetCDF writer creates valid file with final profiles only")
    func testNetCDFWriterFinalProfiles() throws {
        // Create test data
        let cellCount = 10
        let finalProfiles = SerializableProfiles(
            ionTemperature: (0..<cellCount).map { Float($0) * 100.0 },
            electronTemperature: (0..<cellCount).map { Float($0) * 90.0 },
            electronDensity: (0..<cellCount).map { Float($0) * 1e19 },
            poloidalFlux: (0..<cellCount).map { Float($0) * 0.1 }
        )

        let statistics = SimulationStatistics(
            totalIterations: 100,
            totalSteps: 50,
            converged: true,
            maximumResidualNorm: 1e-6,
            wallTime: 12.5
        )

        let result = SimulationResult(
            finalProfiles: finalProfiles,
            statistics: statistics,
            timeSeries: nil  // No time series
        )

        // Create output writer
        let writer = OutputWriter(format: .netcdf)

        // Write to temporary file
        let tempDir = FileManager.default.temporaryDirectory
        let outputURL = tempDir.appendingPathComponent("test_output.nc")

        // Clean up any existing file
        removeTestItemIfExists(at: outputURL)

        // Write NetCDF
        try writer.write(result, to: outputURL)

        // Verify file exists
        #expect(FileManager.default.fileExists(atPath: outputURL.path))

        // Clean up
        removeTestItemIfExists(at: outputURL)
    }

    @Test("NetCDF writer creates valid file with time series")
    func testNetCDFWriterTimeSeries() throws {
        // Create test data with time series
        let cellCount = 10
        let timeCount = 5

        var timeSeries: [TimePoint] = []
        for t in 0..<timeCount {
            let profiles = SerializableProfiles(
                ionTemperature: (0..<cellCount).map { Float($0 + t) * 100.0 },
                electronTemperature: (0..<cellCount).map { Float($0 + t) * 90.0 },
                electronDensity: (0..<cellCount).map { Float($0 + t) * 1e19 },
                poloidalFlux: (0..<cellCount).map { Float($0 + t) * 0.1 }
            )
            timeSeries.append(TimePoint(time: Float(t) * 0.1, profiles: profiles))
        }

        let finalProfiles = timeSeries.last!.profiles
        let statistics = SimulationStatistics(
            totalIterations: 100,
            totalSteps: 50,
            converged: true,
            maximumResidualNorm: 1e-6,
            wallTime: 12.5
        )

        let result = SimulationResult(
            finalProfiles: finalProfiles,
            statistics: statistics,
            timeSeries: timeSeries
        )

        // Create output writer
        let writer = OutputWriter(format: .netcdf)

        // Write to known location for manual inspection
        let outputURL = URL(fileURLWithPath: "/tmp/gotenx_test_timeseries.nc")

        // Clean up any existing file
        removeTestItemIfExists(at: outputURL)

        // Write NetCDF
        try writer.write(result, to: outputURL)

        // Verify file exists
        #expect(FileManager.default.fileExists(atPath: outputURL.path))

        print("NetCDF file created at: \(outputURL.path)")
        print("Inspect with: ncdump -h \(outputURL.path)")

        // Don't clean up - leave for inspection
        // removeTestItemIfExists(at: outputURL)
    }

    @Test("JSON writer still works")
    func testJSONWriter() throws {
        // Create test data
        let cellCount = 10
        let finalProfiles = SerializableProfiles(
            ionTemperature: (0..<cellCount).map { Float($0) * 100.0 },
            electronTemperature: (0..<cellCount).map { Float($0) * 90.0 },
            electronDensity: (0..<cellCount).map { Float($0) * 1e19 },
            poloidalFlux: (0..<cellCount).map { Float($0) * 0.1 }
        )

        let statistics = SimulationStatistics(
            totalIterations: 100,
            totalSteps: 50,
            converged: true,
            maximumResidualNorm: 1e-6,
            wallTime: 12.5
        )

        let result = SimulationResult(
            finalProfiles: finalProfiles,
            statistics: statistics,
            timeSeries: nil
        )

        // Create output writer
        let writer = OutputWriter(format: .json)

        // Write to temporary file
        let tempDir = FileManager.default.temporaryDirectory
        let outputURL = tempDir.appendingPathComponent("test_output.json")

        // Clean up any existing file
        removeTestItemIfExists(at: outputURL)

        // Write JSON
        try writer.write(result, to: outputURL)

        // Verify file exists
        #expect(FileManager.default.fileExists(atPath: outputURL.path))

        // Clean up
        removeTestItemIfExists(at: outputURL)
    }

    @Test("NetCDF handles single cell edge case")
    func testNetCDFSingleCell() throws {
        // Create test data with cellCount = 1
        let cellCount = 1
        let finalProfiles = SerializableProfiles(
            ionTemperature: [1000.0],
            electronTemperature: [900.0],
            electronDensity: [5e19],
            poloidalFlux: [0.5]
        )

        let statistics = SimulationStatistics(
            totalIterations: 10,
            totalSteps: 5,
            converged: true,
            maximumResidualNorm: 1e-7,
            wallTime: 1.0
        )

        let result = SimulationResult(
            finalProfiles: finalProfiles,
            statistics: statistics,
            timeSeries: nil
        )

        // Create output writer
        let writer = OutputWriter(format: .netcdf)

        // Write to temporary file
        let outputURL = URL(fileURLWithPath: "/tmp/gotenx_single_cell.nc")

        // Clean up any existing file
        removeTestItemIfExists(at: outputURL)

        // Write NetCDF - should not crash with division by zero
        try writer.write(result, to: outputURL)

        // Verify file exists
        #expect(FileManager.default.fileExists(atPath: outputURL.path))

        print("Single cell NetCDF created at: \(outputURL.path)")

        // Don't clean up - leave for inspection
    }

    @Test("NetCDF detects inconsistent array lengths")
    func testNetCDFInconsistentLengths() throws {
        // Create invalid data with mismatched array lengths
        let finalProfiles = SerializableProfiles(
            ionTemperature: [1000.0, 2000.0],  // 2 elements
            electronTemperature: [900.0],       // 1 element - MISMATCH
            electronDensity: [5e19, 6e19],
            poloidalFlux: [0.5, 0.6]
        )

        let statistics = SimulationStatistics()

        let result = SimulationResult(
            finalProfiles: finalProfiles,
            statistics: statistics,
            timeSeries: nil
        )

        let writer = OutputWriter(format: .netcdf)
        let outputURL = URL(fileURLWithPath: "/tmp/gotenx_invalid.nc")

        // Should throw error due to mismatched lengths
        #expect(throws: Error.self) {
            try writer.write(result, to: outputURL)
        }
    }

    @Test("NetCDF compression ratio via OutputWriter")
    func testNetCDFCompressionRatio() throws {
        // Generate highly redundant time-series data (nearly static profiles)
        // tuned to yield ~20–25× compression with current chunking policy
        let cellCount = 128
        let timeCount = 512
        let profileAmplitude: Float = 15_000
        let unit: Float = 1.0
        let electronBaseScale: Float = 0.95
        let baseProfile: [Float] = (0..<cellCount).map { idx in
            let rho = Float(idx) / Float(max(1, cellCount - 1))
            return profileAmplitude * (unit - rho * rho)
        }

        let baseDensity: Float = 9.5e19
        let fluxScale: Float = Float(1e-4)
        let deltaTime: Float = 0.01
        let wallTimeScale: Float = Float(1e-4)

        var timeSeries: [TimePoint] = []
        for step in 0..<timeCount {
            // Minimal temporal variation to mirror equilibrium phases
            let epsilon: Float = 1e-3 * Float(step % 8)
            let ionScale: Float = unit + epsilon
            let elecScale: Float = electronBaseScale + epsilon
            let densityScale: Float = unit + epsilon

            let ionTemp = baseProfile.map { $0 * ionScale }
            let elecTemp = baseProfile.map { $0 * elecScale }
            let density = Array(repeating: baseDensity * densityScale, count: cellCount)
            let flux = baseProfile.map { $0 * fluxScale }

            let profiles = SerializableProfiles(
                ionTemperature: ionTemp,
                electronTemperature: elecTemp,
                electronDensity: density,
                poloidalFlux: flux
            )

            let timeValue = Float(step) * deltaTime
            timeSeries.append(TimePoint(time: timeValue, profiles: profiles))
        }

        let finalProfiles = timeSeries.last!.profiles
        let statistics = SimulationStatistics(
            totalIterations: timeCount,
            totalSteps: timeCount,
            converged: true,
            maximumResidualNorm: Float(1e-7),
            wallTime: Float(timeCount) * wallTimeScale
        )

        let result = SimulationResult(
            finalProfiles: finalProfiles,
            statistics: statistics,
            timeSeries: timeSeries
        )

        let writer = OutputWriter(format: .netcdf)
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("gotenx_compression_ratio.nc")
        removeTestItemIfExists(at: outputURL)

        try writer.write(result, to: outputURL)

        let attributes = try FileManager.default.attributesOfItem(atPath: outputURL.path)
        let compressedSize = attributes[.size] as! Int

        // 4 variables × timeCount × cellCount × sizeof(Float)
        let variableCount = 4
        let uncompressedPayload = variableCount * timeCount * cellCount * MemoryLayout<Float>.size
        let compressionRatio = Double(uncompressedPayload) / Double(compressedSize)

        print("✅ OutputWriter compression ratio results:")
        print("   Uncompressed payload: \(uncompressedPayload) bytes")
        print("   Compressed file:      \(compressedSize) bytes")
        print("   Compression ratio:    \(String(format: "%.2f", compressionRatio))×")

        #expect(compressionRatio > 8.0, "Compression ratio should exceed 8× (got \(String(format: "%.2f", compressionRatio)))")

        removeTestItemIfExists(at: outputURL)
    }
}
