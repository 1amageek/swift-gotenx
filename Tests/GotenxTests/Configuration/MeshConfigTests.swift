// MeshConfigTests.swift
// Tests for MeshConfig

import Testing
import Foundation
@testable import GotenxCore

@Suite("MeshConfig Tests")
struct MeshConfigTests {

    @Test("MeshConfig initialization")
    func testInitialization() {
        let mesh = MeshConfig(
            cellCount: 100,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        #expect(mesh.cellCount == 100)
        #expect(mesh.majorRadius == 3.0)
        #expect(mesh.minorRadius == 1.0)
        #expect(mesh.toroidalField == 2.5)
    }

    @Test("MeshConfig derived properties")
    func testDerivedProperties() {
        let mesh = MeshConfig(
            cellCount: 100,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        #expect(mesh.radialSpacing == 0.01)  // 1.0 / 100
        #expect(mesh.aspectRatio == 3.0)  // 3.0 / 1.0
    }

    @Test("MeshConfig valid configuration")
    func testValidConfiguration() throws {
        let mesh = MeshConfig(
            cellCount: 100,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        // Should not throw
        try mesh.validate()
    }

    @Test("MeshConfig invalid cellCount (zero)")
    func testInvalidNCellsZero() {
        let mesh = MeshConfig(
            cellCount: 0,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        #expect(throws: ConfigurationError.self) {
            try mesh.validate()
        }
    }

    @Test("MeshConfig invalid cellCount (negative)")
    func testInvalidNCellsNegative() {
        let mesh = MeshConfig(
            cellCount: -10,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        #expect(throws: ConfigurationError.self) {
            try mesh.validate()
        }
    }

    @Test("MeshConfig warning for few cells")
    func testWarningFewCells() {
        let mesh = MeshConfig(
            cellCount: 5,  // Less than 10
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        #expect(throws: ConfigurationError.self) {
            try mesh.validate()
        }
    }

    @Test("MeshConfig invalid radius (negative)")
    func testInvalidRadius() {
        let mesh = MeshConfig(
            cellCount: 100,
            majorRadius: -3.0,  // Negative
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        #expect(throws: ConfigurationError.self) {
            try mesh.validate()
        }
    }

    @Test("MeshConfig low aspect ratio warning")
    func testLowAspectRatioWarning() {
        let mesh = MeshConfig(
            cellCount: 100,
            majorRadius: 1.2,  // Aspect ratio = 1.2 < 1.5
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        #expect(throws: ConfigurationError.self) {
            try mesh.validate()
        }
    }

    @Test("MeshConfig invalid toroidal field")
    func testInvalidToroidalField() {
        let mesh = MeshConfig(
            cellCount: 100,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: -2.5  // Negative
        )

        #expect(throws: ConfigurationError.self) {
            try mesh.validate()
        }
    }

    @Test("MeshConfig Codable")
    func testCodable() throws {
        let mesh = MeshConfig(
            cellCount: 100,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(mesh)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(MeshConfig.self, from: data)

        #expect(mesh == decoded)
    }
}
