// NetCDFPoCTests.swift
// Week 1 PoC: Minimal NetCDF write test with CF-1.8 metadata

import Testing
import Foundation
import SwiftNetCDF

@Suite("NetCDF Proof of Concept")
struct NetCDFPoCTests {

    /// Week 1 PoC: Write single variable (electron temperature) to NetCDF file
    ///
    /// Success criteria:
    /// - ✅ File creation succeeds
    /// - ✅ CF-1.8 metadata is correctly written
    /// - ✅ Data is retrievable via ncdump
    @Test("PoC: Write single variable to NetCDF")
    func testMinimalNetCDFWrite() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("poc_test.nc").path

        // Clean up any existing file
        removeTestItemIfExists(atPath: filePath)

        // Create NetCDF file
        let file = try NetCDF.create(path: filePath, overwriteExisting: true)

        // Define dimensions
        let timeDim = try file.createDimension(name: "time", length: 5)
        let rhoDim = try file.createDimension(name: "rho_tor_norm", length: 10)

        // Create variable: electron temperature
        var teVar = try file.createVariable(
            name: "electrons_temperature",
            type: Float.self,
            dimensions: [timeDim, rhoDim]
        )

        // CF-1.8 metadata
        try teVar.setAttribute("long_name", "Electron temperature")
        try teVar.setAttribute("units", "eV")
        try teVar.setAttribute("standard_name", "electron_temperature")
        try teVar.setAttribute("coordinates", "time rho_tor_norm")
        try teVar.setAttribute("valid_min", Float(0.0))
        try teVar.setAttribute("valid_max", Float(100000.0))

        // Write test data (5 time points × 10 radial points)
        let testData: [Float] = (0..<50).map { i in
            let rho = Float(i % 10) / 10.0  // 0-0.9
            // Parabolic profile: T = 10000 * (1 - rho^2) eV
            return 10000.0 * (1.0 - rho * rho)
        }

        try teVar.write(testData, offset: [0, 0], count: [5, 10])

        // Sync to disk
        file.sync()

        // Verify file exists
        #expect(FileManager.default.fileExists(atPath: filePath), "NetCDF file should exist")

        // Verify file can be opened
        guard let readFile = try NetCDF.open(path: filePath, allowUpdate: false) else {
            Issue.record("Failed to open NetCDF file")
            return
        }

        // Verify variable exists
        guard let readVar = readFile.getVariable(name: "electrons_temperature") else {
            Issue.record("Variable 'electrons_temperature' not found")
            return
        }

        // Verify dimensions
        #expect(readVar.dimensionsFlat.count == 2, "Should have 2 dimensions")

        // Verify metadata
        let units: String? = try readVar.getAttribute("units")?.read()
        #expect(units == "eV", "Units should be 'eV'")

        let longName: String? = try readVar.getAttribute("long_name")?.read()
        #expect(longName == "Electron temperature", "long_name mismatch")

        // Read data back
        let readVarTyped = readVar.asType(Float.self)!
        let readData = try readVarTyped.read(offset: [0, 0], count: [5, 10])

        // Verify data integrity (first and last points)
        #expect(abs(readData[0] - testData[0]) < 1e-5, "First data point should match")
        #expect(abs(readData[49] - testData[49]) < 1e-5, "Last data point should match")

        print("✅ Week 1 PoC: Minimal NetCDF write test passed")
        print("   File: \(filePath)")
        print("   Size: \(try FileManager.default.attributesOfItem(atPath: filePath)[.size] as! Int) bytes")
    }

    /// Week 1 PoC: Verify ncdump can read the file
    @Test("PoC: ncdump verification")
    func testNcdumpVerification() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("poc_test_ncdump.nc").path

        // Create simple file in a scope to ensure it's closed
        do {
            let file = try NetCDF.create(path: filePath, overwriteExisting: true)
            let timeDim = try file.createDimension(name: "time", length: 3)
            var timeVar = try file.createVariable(name: "time", type: Float.self, dimensions: [timeDim])
            try timeVar.setAttribute("units", "seconds since 1970-01-01")
            try timeVar.write([0.0, 1.0, 2.0], offset: [0], count: [3])
            file.sync()
            // File closes when exiting scope
        }

        // Run ncdump (header only)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/ncdump")
        process.arguments = ["-h", filePath]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        #expect(process.terminationStatus == 0, "ncdump should succeed")

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""

        // Verify output contains expected elements
        #expect(output.contains("dimensions:"), "Should contain dimensions section")
        #expect(output.contains("time = 3"), "Should declare time dimension")
        #expect(output.contains("variables:"), "Should contain variables section")
        #expect(output.contains("float time(time)"), "Should declare time variable")
        #expect(output.contains("units"), "Should contain units attribute")

        print("✅ ncdump verification passed")
        print("Output preview:")
        print(output.prefix(300))
    }

    /// Week 1 PoC: Test CF-1.8 compliance with standard_name
    @Test("PoC: CF-1.8 standard_name compliance")
    func testCFStandardNames() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("poc_test_cf.nc").path

        let file = try NetCDF.create(path: filePath, overwriteExisting: true)

        // Create dimensions
        let rhoDim = try file.createDimension(name: "rho_tor_norm", length: 50)

        // Test CF-1.8 standard names for plasma physics variables
        let cfVariables: [(name: String, standardName: String, units: String)] = [
            ("ion_temperature", "ion_temperature", "eV"),
            ("electron_temperature", "electron_temperature", "eV"),
            ("electron_density", "electron_number_density", "m-3"),
            ("toroidal_field", "magnetic_flux_density", "T")
        ]

        for varSpec in cfVariables {
            var variable = try file.createVariable(
                name: varSpec.name,
                type: Float.self,
                dimensions: [rhoDim]
            )

            try variable.setAttribute("standard_name", varSpec.standardName)
            try variable.setAttribute("units", varSpec.units)
            try variable.setAttribute("long_name", varSpec.name.replacingOccurrences(of: "_", with: " ").capitalized)

            // Write dummy data
            let dummyData = Array(repeating: Float(1.0), count: 50)
            try variable.write(dummyData, offset: [0], count: [50])
        }

        file.sync()

        // Verify all variables were created with correct metadata
        guard let readFile = try NetCDF.open(path: filePath, allowUpdate: false) else {
            Issue.record("Failed to open file")
            return
        }

        for varSpec in cfVariables {
            guard let variable = readFile.getVariable(name: varSpec.name) else {
                Issue.record("Variable \(varSpec.name) not found")
                continue
            }

            let standardName: String? = try variable.getAttribute("standard_name")?.read()
            #expect(standardName == varSpec.standardName, "\(varSpec.name): standard_name mismatch")

            let units: String? = try variable.getAttribute("units")?.read()
            #expect(units == varSpec.units, "\(varSpec.name): units mismatch")
        }

        print("✅ CF-1.8 standard_name compliance test passed")
        print("   Verified \(cfVariables.count) variables")
    }
}
