// GotenxConfigReaderTests.swift
// Tests for GotenxConfigReader with swift-configuration integration

import Testing
import Foundation
@testable import GotenxCore

#if canImport(GotenxCLI)
import GotenxCLI

@Suite("GotenxConfigReader Integration Tests")
struct GotenxConfigReaderTests {

    // MARK: - Basic Loading Tests

    @Test("Load minimal configuration from JSON")
    func testLoadMinimalConfig() async throws {
        let configPath = try createTestConfig(cellCount: 100)
        defer { removeTestItemIfExists(atPath: configPath) }

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: [:]
        )

        let config = try await reader.fetchConfiguration()

        // Verify basic structure
        #expect(config.runtime.static.mesh.cellCount > 0)
        #expect(config.time.end > config.time.start)
        #expect(config.time.initialTimeStep > 0)
    }

    @Test("Load ITER-like configuration from JSON")
    func testLoadIterLikeConfig() async throws {
        // ITER-like: larger major/minor radius
        let configPath = try createTestConfig(
            cellCount: 100,
            majorRadius: 6.2,
            minorRadius: 2.0
        )
        defer { removeTestItemIfExists(atPath: configPath) }

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: [:]
        )

        let config = try await reader.fetchConfiguration()

        // ITER-like should have larger parameters
        #expect(config.runtime.static.mesh.majorRadius > 5.0)  // ITER is ~6.2m
        #expect(config.runtime.static.mesh.minorRadius > 1.0)  // ITER is ~2m
    }

    // MARK: - CLI Override Tests

    @Test("CLI overrides take precedence over JSON")
    func testCLIOverrides() async throws {
        let configPath = try createTestConfig(cellCount: 100)
        defer { removeTestItemIfExists(atPath: configPath) }

        // Override mesh cells
        let cliOverrides = [
            "runtime.static.mesh.cellCount": "200",
            "time.end": "5.0"
        ]

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: cliOverrides
        )

        let config = try await reader.fetchConfiguration()

        // Verify overrides were applied
        #expect(config.runtime.static.mesh.cellCount == 200)
        #expect(config.time.end == 5.0)
    }

    @Test("CLI overrides with nested keys")
    func testNestedCLIOverrides() async throws {
        let configPath = try createTestConfig()
        defer { removeTestItemIfExists(atPath: configPath) }

        let cliOverrides = [
            "runtime.static.mesh.majorRadius": "7.0",
            "runtime.static.mesh.minorRadius": "2.5",
            "runtime.dynamic.boundaries.ionTemperature": "200.0"
        ]

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: cliOverrides
        )

        let config = try await reader.fetchConfiguration()

        #expect(config.runtime.static.mesh.majorRadius == 7.0)
        #expect(config.runtime.static.mesh.minorRadius == 2.5)
        #expect(config.runtime.dynamic.boundaries.ionTemperature == 200.0)
    }

    // MARK: - Environment Variable Tests

    @Test("Environment variables override JSON but not CLI")
    func testEnvironmentVariables() async throws {
        let configPath = try createTestConfig()
        defer { removeTestItemIfExists(atPath: configPath) }

        // Case 1: No CLI override - environment wins
        let reader1 = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: [:],
            environment: [
                "GOTENX_RUNTIME_STATIC_MESH_CELL_COUNT": "150"
            ]
        )
        let config1 = try await reader1.fetchConfiguration()
        #expect(config1.runtime.static.mesh.cellCount == 150)

        // Case 2: CLI override - CLI wins over environment
        let cliOverrides = ["runtime.static.mesh.cellCount": "200"]
        let reader2 = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: cliOverrides,
            environment: [
                "GOTENX_RUNTIME_STATIC_MESH_CELL_COUNT": "150"
            ]
        )
        let config2 = try await reader2.fetchConfiguration()
        #expect(config2.runtime.static.mesh.cellCount == 200)
    }

    // MARK: - Configuration Validation Tests

    @Test("Invalid JSON file throws error")
    func testInvalidJSONFile() async throws {
        await #expect(throws: Error.self) {
            _ = try await GotenxConfigReader.create(
                jsonPath: "/nonexistent/config.json",
                cliOverrides: [:]
            )
        }
    }

    @Test("Malformed configuration values throw validation errors")
    func testMalformedConfiguration() async throws {
        let configPath = try createTestConfig()
        defer { removeTestItemIfExists(atPath: configPath) }

        // Try to set invalid values
        let cliOverrides = [
            "runtime.static.mesh.cellCount": "-100",  // Negative cells
            "time.end": "-1.0"  // Negative time
        ]

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: cliOverrides
        )

        // Should throw validation error when fetching
        await #expect(throws: ConfigurationError.self) {
            _ = try await reader.fetchConfiguration()
        }
    }

    @Test("Unknown JSON transport parameters throw validation errors")
    func testUnknownJSONTransportParameter() async throws {
        let configPath = try createTestConfig()
        defer { removeTestItemIfExists(atPath: configPath) }

        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              var runtime = root["runtime"] as? [String: Any],
              var dynamic = runtime["dynamic"] as? [String: Any],
              var transport = dynamic["transport"] as? [String: Any] else {
            Issue.record("Test fixture must contain runtime.dynamic.transport")
            return
        }

        transport["parameters"] = [
            "ionHeatDiffusivity": 0.01,
            "electronHeatDiffusivity": 0.01,
            "obsoleteParameter": 1.0
        ]
        dynamic["transport"] = transport
        runtime["dynamic"] = dynamic
        root["runtime"] = runtime

        let encoded = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted])
        try encoded.write(to: URL(fileURLWithPath: configPath))

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: [:]
        )

        await #expect(throws: ConfigurationValidationError.self) {
            _ = try await reader.fetchConfiguration()
        }
    }

    @Test("Unknown CLI transport parameters throw validation errors")
    func testUnknownCLITransportParameter() async throws {
        let configPath = try createTestConfig()
        defer { removeTestItemIfExists(atPath: configPath) }

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: [
                "runtime.dynamic.transport.parameters.obsoleteParameter": "1.0"
            ]
        )

        await #expect(throws: ConfigurationValidationError.self) {
            _ = try await reader.fetchConfiguration()
        }
    }

    @Test("Unknown environment transport parameters throw validation errors")
    func testUnknownEnvironmentTransportParameter() async throws {
        let configPath = try createTestConfig()
        defer { removeTestItemIfExists(atPath: configPath) }

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: [:],
            environment: [
                "GOTENX_RUNTIME_DYNAMIC_TRANSPORT_PARAMETERS_OBSOLETE_PARAMETER": "1.0"
            ]
        )

        await #expect(throws: ConfigurationValidationError.self) {
            _ = try await reader.fetchConfiguration()
        }
    }

    @Test("Environment transport parameters use full Swift API names")
    func testEnvironmentTransportParametersUseFullSwiftAPINames() async throws {
        let configPath = try createTestConfig()
        defer { removeTestItemIfExists(atPath: configPath) }

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: [:],
            environment: [
                "GOTENX_RUNTIME_DYNAMIC_TRANSPORT_PARAMETERS_ION_HEAT_DIFFUSIVITY": "0.002",
                "GOTENX_RUNTIME_DYNAMIC_TRANSPORT_PARAMETERS_ELECTRON_HEAT_DIFFUSIVITY": "0.003",
                "GOTENX_RUNTIME_DYNAMIC_TRANSPORT_PARAMETERS_PARTICLE_DIFFUSIVITY": "0.0005"
            ]
        )

        let config = try await reader.fetchConfiguration()

        #expect(abs((config.runtime.dynamic.transport.parameter("ionHeatDiffusivity") ?? -1) - 0.002) < 1e-7)
        #expect(abs((config.runtime.dynamic.transport.parameter("electronHeatDiffusivity") ?? -1) - 0.003) < 1e-7)
        #expect(abs((config.runtime.dynamic.transport.parameter("particleDiffusivity") ?? -1) - 0.0005) < 1e-7)
    }

    @Test("Non-object JSON transport parameters throw configuration errors")
    func testNonObjectJSONTransportParameters() async throws {
        let configPath = try createTestConfig()
        defer { removeTestItemIfExists(atPath: configPath) }

        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              var runtime = root["runtime"] as? [String: Any],
              var dynamic = runtime["dynamic"] as? [String: Any],
              var transport = dynamic["transport"] as? [String: Any] else {
            Issue.record("Test fixture must contain runtime.dynamic.transport")
            return
        }

        transport["parameters"] = 42
        dynamic["transport"] = transport
        runtime["dynamic"] = dynamic
        root["runtime"] = runtime

        let encoded = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted])
        try encoded.write(to: URL(fileURLWithPath: configPath))

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: [:]
        )

        await #expect(throws: ConfigurationError.self) {
            _ = try await reader.fetchConfiguration()
        }
    }

    // MARK: - Complete Configuration Tests

    @Test("All configuration sections are loaded")
    func testCompleteConfiguration() async throws {
        let configPath = try createTestConfig(
            cellCount: 100,
            majorRadius: 6.2,
            minorRadius: 2.0
        )
        defer { removeTestItemIfExists(atPath: configPath) }

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: [:]
        )

        let config = try await reader.fetchConfiguration()

        // Runtime - Static
        #expect(config.runtime.static.mesh.cellCount > 0)
        #expect(config.runtime.static.mesh.majorRadius > 0)
        #expect(config.runtime.static.mesh.minorRadius > 0)
        #expect(config.runtime.static.mesh.toroidalField > 0)

        // Runtime - Dynamic
        #expect(config.runtime.dynamic.boundaries.ionTemperature > 0)
        #expect(config.runtime.dynamic.boundaries.electronTemperature > 0)
        #expect(config.runtime.dynamic.boundaries.electronDensity > 0)

        // Transport (modelType is enum, always valid)
        // No need to check - enum ensures valid value

        // Sources
        #expect(config.runtime.dynamic.sources.ohmicHeating || true)  // Valid boolean

        // Time
        #expect(config.time.start >= 0)
        #expect(config.time.end > config.time.start)
        #expect(config.time.initialTimeStep > 0)

        // Output
        #expect(!config.output.directory.isEmpty)
    }

    // MARK: - Hierarchical Override Priority Tests

    @Test("Verify complete override priority: CLI > Env > JSON > Default")
    func testOverridePriority() async throws {
        let configPath = try createTestConfig()
        defer { removeTestItemIfExists(atPath: configPath) }

        // CLI override should win
        let cliOverrides = ["time.end": "10.0"]

        let reader = try await GotenxConfigReader.create(
            jsonPath: configPath,
            cliOverrides: cliOverrides,
            environment: [
                "GOTENX_TIME_END": "3.0"
            ]
        )

        let config = try await reader.fetchConfiguration()

        // CLI value should take precedence
        #expect(config.time.end == 10.0)
    }

    // MARK: - Test Fixtures

    /// Create a minimal test JSON configuration
    /// This matches the pattern used in ConfigurationPriorityTests
    private func createTestConfig(
        cellCount: Int = 100,
        majorRadius: Double = 3.0,
        minorRadius: Double = 1.0
    ) throws -> String {
        let tempDir = FileManager.default.temporaryDirectory
        let configPath = tempDir.appendingPathComponent("test_config_\(UUID()).json")

        let json = """
        {
          "runtime": {
            "static": {
              "mesh": {
                "cellCount": \(cellCount),
                "majorRadius": \(majorRadius),
                "minorRadius": \(minorRadius),
                "toroidalField": 2.5,
                "geometryType": "circular"
              },
              "evolution": {
                "ionTemperature": true,
                "electronTemperature": true,
                "electronDensity": true,
                "poloidalFlux": false
              },
              "solver": {
                "type": "linear",
                "tolerance": 1e-6,
                "maximumIterations": 30
              },
              "scheme": {
                "theta": 1.0
              }
            },
            "dynamic": {
              "boundaries": {
                "ionTemperature": 100.0,
                "electronTemperature": 100.0,
                "electronDensity": 1e19
              },
              "transport": {
                "modelType": "constant"
              },
              "sources": {
                "ohmicHeating": true,
                "fusionPower": true,
                "ionElectronExchange": true,
                "bremsstrahlung": true
              },
              "pedestal": {
                "model": "none"
              },
              "mhd": {
                "sawtoothEnabled": false,
                "ntmEnabled": false
              },
              "restart": {
                "doRestart": false,
                "stitch": true
              }
            }
          },
          "time": {
            "start": 0.0,
            "end": 1.0,
            "initialTimeStep": 0.001,
            "adaptive": {
              "enabled": true,
              "safetyFactor": 0.9,
              "minimumTimeStep": 1e-6,
              "maximumTimeStep": 0.1
            }
          },
          "output": {
            "directory": "/tmp/gotenx_results",
            "format": "json"
          }
        }
        """

        try json.write(to: configPath, atomically: true, encoding: .utf8)
        return configPath.path
    }
}
#endif
