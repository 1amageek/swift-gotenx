// SimulationConfigurationTests.swift
// Tests for SimulationConfiguration

import Testing
import Foundation
@testable import GotenxCore

@Suite("SimulationConfiguration Tests")
struct SimulationConfigurationTests {

    @Test("SimulationConfiguration initialization")
    func testInitialization() {
        let config = SimulationConfiguration(
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
                        ionTemperature: 100.0,
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: .defaultConstant
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 1.0,
                initialTimeStep: 1e-3
            )
        )

        #expect(config.runtime.static.mesh.cellCount == 100)
        #expect(config.time.end == 1.0)
    }

    @Test("SimulationConfiguration builder pattern")
    func testBuilderPattern() {
        let config = SimulationConfiguration.build { builder in
            builder.runtime.static.mesh.cellCount = 150
            builder.runtime.static.mesh.majorRadius = 3.5
            builder.runtime.dynamic.boundaries = BoundaryConfig(
                ionTemperature: 150.0,
                electronTemperature: 150.0,
                electronDensity: 1e19
            )
            builder.time.end = 2.0
        }

        #expect(config.runtime.static.mesh.cellCount == 150)
        #expect(config.runtime.static.mesh.majorRadius == 3.5)
        #expect(config.runtime.dynamic.boundaries.ionTemperature == 150.0)
        #expect(config.time.end == 2.0)
    }

    @Test("SimulationConfiguration Codable (JSON round-trip)")
    func testJSONRoundTrip() throws {
        let original = SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: MeshConfig(
                        cellCount: 100,
                        majorRadius: 3.0,
                        minorRadius: 1.0,
                        toroidalField: 2.5
                    ),
                    evolution: .default,
                    solver: .default,
                    scheme: .default
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: 100.0,
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: .defaultConstant,
                    sources: .default
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 1.0,
                initialTimeStep: 1e-3
            ),
            output: .default
        )

        // Encode to JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(original)

        // Decode from JSON
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(SimulationConfiguration.self, from: data)

        // Verify equality
        #expect(original == decoded)
    }

    @Test("SimulationConfiguration validation (valid)")
    func testValidationValid() throws {
        let config = SimulationConfiguration(
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
                end: 1.0,
                initialTimeStep: 1e-6  // Below CFL estimate of 1e-5
            )
        )

        // Should not throw
        try ConfigurationValidator.validate(config)
    }

    @Test("SimulationConfiguration validation (CFL timestep warning)")
    func testValidationCFLTimestep() throws {
        // CFL check is now a warning, not an error
        let config = SimulationConfiguration(
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
                end: 1.0,
                initialTimeStep: 1e-3  // Much larger than CFL estimate
            )
        )

        // Should not throw - only prints warning
        try ConfigurationValidator.validate(config)
    }

    @Test("SimulationConfiguration validation (invalid time range)")
    func testValidationInvalidTimeRange() {
        let config = SimulationConfiguration(
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
                        ionTemperature: 100.0,
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: .defaultConstant
                )
            ),
            time: TimeConfiguration(
                start: 1.0,
                end: 0.5,  // End < Start
                initialTimeStep: 1e-3
            )
        )

        #expect(throws: ConfigurationError.self) {
            try ConfigurationValidator.validate(config)
        }
    }

    @Test("SimulationConfiguration validation (invalid boundary)")
    func testValidationInvalidBoundary() {
        let config = SimulationConfiguration(
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
                        ionTemperature: -100.0,  // Negative
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: .defaultConstant
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 1.0,
                initialTimeStep: 1e-3
            )
        )

        #expect(throws: ConfigurationError.self) {
            try ConfigurationValidator.validate(config)
        }
    }

    @Test("SimulationConfiguration validation (fusion fractions)")
    func testValidationInvalidFusionFractions() {
        let config = SimulationConfiguration(
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
                        ionTemperature: 100.0,
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: .defaultConstant,
                    sources: SourcesConfig(
                        fusionConfig: FusionConfig(
                            deuteriumFraction: 0.7,  // Sum != 1.0
                            tritiumFraction: 0.5,
                            dilution: 0.9
                        )
                    )
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 1.0,
                initialTimeStep: 1e-3
            )
        )

        #expect(throws: ConfigurationError.self) {
            try ConfigurationValidator.validate(config)
        }
    }

    @Test("SimulationConfiguration static/dynamic equality")
    func testStaticDynamicEquality() {
        let config1 = SimulationConfiguration(
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
                        ionTemperature: 100.0,
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: .defaultConstant
                )
            ),
            time: TimeConfiguration(end: 1.0)
        )

        let config2 = SimulationConfiguration(
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
                        ionTemperature: 150.0,  // Different dynamic param
                        electronTemperature: 100.0,
                        electronDensity: 1e19
                    ),
                    transport: .defaultConstant
                )
            ),
            time: TimeConfiguration(end: 1.0)
        )

        // Static configs should be equal
        #expect(config1.runtime.static == config2.runtime.static)

        // Dynamic configs should be different
        #expect(config1.runtime.dynamic != config2.runtime.dynamic)

        // Overall configs should be different
        #expect(config1 != config2)
    }
}
