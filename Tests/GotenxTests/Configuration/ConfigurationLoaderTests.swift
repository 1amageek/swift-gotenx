// ConfigurationLoaderTests.swift
// Tests for ConfigurationLoader and providers

import Testing
import Foundation
@testable import GotenxCore

@Suite("ConfigurationLoader Tests")
struct ConfigurationLoaderTests {

    @Test("DefaultConfigurationProvider loads valid config")
    func testDefaultProvider() async throws {
        let provider = DefaultConfigurationProvider()
        let config = try await provider.load()

        #expect(config != nil)
        #expect(config?.runtime.static.mesh.cellCount == 100)
        #expect(config?.runtime.static.mesh.majorRadius == 6.2)
    }

    @Test("JSONConfigurationProvider loads from file")
    func testJSONProvider() async throws {
        // Create temporary JSON file
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_config.json")

        let testConfig = SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: MeshConfig(
                        cellCount: 75,
                        majorRadius: 4.0,
                        minorRadius: 1.5,
                        toroidalField: 3.0
                    )
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: 120.0,
                        electronTemperature: 120.0,
                        electronDensity: 2e19
                    ),
                    transport: try TransportConfig(
                        modelType: .constant,
                        parameters: [
                            "ionHeatDiffusivity": 0.01,
                            "electronHeatDiffusivity": 0.01,
                            "particleDiffusivity": 0.005
                        ]
                    )
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 1.5,
                initialTimeStep: 1e-6
            )
        )

        // Write config to file
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(testConfig)
        try data.write(to: tempFile)

        // Load via provider
        let provider = JSONConfigurationProvider(filePath: tempFile.path)
        let loaded = try await provider.load()

        #expect(loaded != nil)
        #expect(loaded?.runtime.static.mesh.cellCount == 75)
        #expect(loaded?.runtime.static.mesh.majorRadius == 4.0)
        #expect(loaded?.time.end == 1.5)

        // Clean up
        removeTestItemIfExists(at: tempFile)
    }

    @Test("JSONConfigurationProvider returns nil for missing file")
    func testJSONProviderMissingFile() async throws {
        let provider = JSONConfigurationProvider(filePath: "/nonexistent/path/config.json")
        let loaded = try await provider.load()

        #expect(loaded == nil)
    }

    @Test("ConfigurationLoader with default provider")
    func testLoaderWithDefaults() async throws {
        let loader = ConfigurationLoader(providers: [DefaultConfigurationProvider()])
        let config = try await loader.load()

        #expect(config.runtime.static.mesh.cellCount == 100)
        #expect(config.runtime.static.mesh.majorRadius == 6.2)
    }

    @Test("ConfigurationLoader.standard applies CLI overrides")
    func testStandardLoaderAppliesCLIOverrides() async throws {
        let loader = ConfigurationLoader.standard(
            cliArguments: [
                "mesh-cell-count": "250",
                "time-end": "4.0",
                "output-dir": "/tmp/gotenx_cli_override"
            ]
        )

        let config = try await loader.load()

        #expect(config.runtime.static.mesh.cellCount == 250)
        #expect(config.time.end == 4.0)
        #expect(config.output.directory == "/tmp/gotenx_cli_override")
    }

    @Test("ConfigurationLoader hierarchical override")
    func testLoaderHierarchicalOverride() async throws {
        // Create temp JSON file with custom config
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_override.json")

        let customConfig = SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: MeshConfig(
                        cellCount: 200,
                        majorRadius: 5.0,
                        minorRadius: 1.8,
                        toroidalField: 4.0
                    )
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: 150.0,
                        electronTemperature: 150.0,
                        electronDensity: 1.5e19
                    ),
                    transport: try TransportConfig(
                        modelType: .constant,
                        parameters: [
                            "ionHeatDiffusivity": 0.01,
                            "electronHeatDiffusivity": 0.01,
                            "particleDiffusivity": 0.005
                        ]
                    )
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 3.0,
                initialTimeStep: 1e-6
            )
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(customConfig)
        try data.write(to: tempFile)

        // Load with JSON provider (should override defaults)
        let loader = ConfigurationLoader(providers: [
            JSONConfigurationProvider(filePath: tempFile.path),
            DefaultConfigurationProvider()
        ])

        let config = try await loader.load()

        // Should use JSON config, not defaults
        #expect(config.runtime.static.mesh.cellCount == 200)
        #expect(config.runtime.static.mesh.majorRadius == 5.0)
        #expect(config.time.end == 3.0)

        // Clean up
        removeTestItemIfExists(at: tempFile)
    }

    @Test("ConfigurationLoader.loadFromJSON")
    func testLoadFromJSON() async throws {
        // Create temp JSON file
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_direct.json")

        let testConfig = SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: MeshConfig(
                        cellCount: 150,
                        majorRadius: 7.0,
                        minorRadius: 2.5,
                        toroidalField: 6.0
                    )
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: 200.0,
                        electronTemperature: 200.0,
                        electronDensity: 3e19
                    ),
                    transport: try TransportConfig(
                        modelType: .constant,
                        parameters: [
                            "ionHeatDiffusivity": 0.01,
                            "electronHeatDiffusivity": 0.01,
                            "particleDiffusivity": 0.005
                        ]
                    )
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 5.0,
                initialTimeStep: 1e-6
            )
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(testConfig)
        try data.write(to: tempFile)

        // Load directly
        let config = try await ConfigurationLoader.loadFromJSON(tempFile.path)

        #expect(config.runtime.static.mesh.cellCount == 150)
        #expect(config.runtime.static.mesh.majorRadius == 7.0)

        // Clean up
        removeTestItemIfExists(at: tempFile)
    }

    @Test("ConfigurationOverrides from CLI")
    func testOverridesFromCLI() {
        let args: [String: String] = [
            "mesh-cell-count": "250",
            "mesh-major-radius": "8.0",
            "time-end": "10.0",
            "initial-time-step": "0.002",
            "output-dir": "/custom/output"
        ]

        let overrides = ConfigurationOverrides.fromCLI(args)

        #expect(overrides.meshCellCount == 250)
        #expect(overrides.meshMajorRadius == 8.0)
        #expect(overrides.timeEnd == 10.0)
        #expect(overrides.initialTimeStep == 0.002)
        #expect(overrides.outputDirectory == "/custom/output")
    }

    @Test("ConfigurationLoader.loadWithOverrides")
    func testLoadWithOverrides() throws {
        let baseConfig = SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: MeshConfig(
                        cellCount: 100,
                        majorRadius: 6.0,
                        minorRadius: 2.0,
                        toroidalField: 5.0
                    )
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: 100.0,
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: try TransportConfig(
                        modelType: .constant,
                        parameters: [
                            "ionHeatDiffusivity": 0.01,
                            "electronHeatDiffusivity": 0.01,
                            "particleDiffusivity": 0.005
                        ]
                    )
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 2.0,
                initialTimeStep: 1e-6
            )
        )

        let overrides = ConfigurationOverrides(
            meshCellCount: 300,
            timeEnd: 5.0,
            outputDirectory: "/new/output"
        )

        let config = try ConfigurationLoader.loadWithOverrides(
            baseConfig: baseConfig,
            overrides: overrides
        )

        // Overrides should be applied
        #expect(config.runtime.static.mesh.cellCount == 300)
        #expect(config.time.end == 5.0)
        #expect(config.output.directory == "/new/output")

        // Non-overridden values should remain
        #expect(config.runtime.static.mesh.majorRadius == 6.0)
        #expect(config.time.start == 0.0)
    }

    @Test("ConfigurationLoader validates loaded config")
    func testLoaderValidatesConfig() async throws {
        // Create invalid config (negative temperature)
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_invalid.json")

        let invalidConfig = SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: MeshConfig(
                        cellCount: 100,
                        majorRadius: 3.0,
                        minorRadius: 1.0,
                        toroidalField: 2.5
                    )
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: -100.0,  // Invalid!
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: try TransportConfig(
                        modelType: .constant,
                        parameters: [
                            "ionHeatDiffusivity": 0.01,
                            "electronHeatDiffusivity": 0.01,
                            "particleDiffusivity": 0.005
                        ]
                    )
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 1.0,
                initialTimeStep: 1e-6
            )
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(invalidConfig)
        try data.write(to: tempFile)

        // Should throw validation error
        await #expect(throws: ConfigurationError.self) {
            try await ConfigurationLoader.loadFromJSON(tempFile.path)
        }

        // Clean up
        removeTestItemIfExists(at: tempFile)
    }
}
