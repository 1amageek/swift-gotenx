// ConfigurationConverterTests.swift
// Tests for configuration conversion to runtime parameters

import Testing
import Foundation
@testable import GotenxCore

@Suite("Configuration Converter Tests")
struct ConfigurationConverterTests {

    @Test("StaticConfig to StaticRuntimeParameters conversion")
    func testStaticConfigConversion() throws {
        let staticConfig = StaticConfig(
            mesh: MeshConfig(
                cellCount: 100,
                majorRadius: 6.2,
                minorRadius: 2.0,
                toroidalField: 5.3
            ),
            evolution: EvolutionConfig(
                ionHeat: true,
                electronHeat: true,
                electronDensity: false,
                poloidalFlux: true
            ),
            solver: SolverConfig(
                type: "newton",
                tolerance: 1e-6,
                maximumIterations: 30
            ),
            scheme: SchemeConfig(
                theta: 0.5,
                usePereverzev: true
            )
        )

        let runtimeParams = try staticConfig.runtimeParameters()

        // Verify mesh is preserved
        #expect(runtimeParams.mesh.cellCount == 100)
        #expect(runtimeParams.mesh.majorRadius == 6.2)
        #expect(runtimeParams.mesh.minorRadius == 2.0)
        #expect(runtimeParams.mesh.toroidalField == 5.3)

        // Verify evolution flags are converted correctly
        #expect(runtimeParams.evolveIonHeat == true)
        #expect(runtimeParams.evolveElectronHeat == true)
        #expect(runtimeParams.evolveElectronDensity == false)
        #expect(runtimeParams.evolvePoloidalFlux == true)

        // Verify solver parameters
        #expect(runtimeParams.solverType == .newtonRaphson)
        #expect(runtimeParams.solverTolerance == 1e-6)
        #expect(runtimeParams.solverMaximumIterations == 30)

        // Verify scheme
        #expect(runtimeParams.theta == 0.5)
    }

    @Test("StaticConfig conversion with linear solver")
    func testLinearSolverConversion() throws {
        let staticConfig = StaticConfig(
            mesh: MeshConfig(
                cellCount: 50,
                majorRadius: 3.0,
                minorRadius: 1.0,
                toroidalField: 2.5
            ),
            solver: SolverConfig(
                type: "linear",
                tolerance: 1e-4,
                maximumIterations: 20
            )
        )

        let runtimeParams = try staticConfig.runtimeParameters()

        #expect(runtimeParams.solverType == .linear)
        #expect(runtimeParams.solverTolerance == 1e-4)
        #expect(runtimeParams.solverMaximumIterations == 20)
    }

    @Test("StaticConfig conversion with default evolution")
    func testDefaultEvolutionConversion() throws {
        let staticConfig = StaticConfig(
            mesh: MeshConfig(
                cellCount: 100,
                majorRadius: 6.2,
                minorRadius: 2.0,
                toroidalField: 5.3
            )
            // Uses default evolution: all true except current
        )

        let runtimeParams = try staticConfig.runtimeParameters()

        #expect(runtimeParams.evolveIonHeat == true)
        #expect(runtimeParams.evolveElectronHeat == true)
        #expect(runtimeParams.evolveElectronDensity == true)
        #expect(runtimeParams.evolvePoloidalFlux == false)
    }

    @Test("StaticConfig conversion throws on invalid solver type")
    func testInvalidSolverTypeConversion() {
        let staticConfig = StaticConfig(
            mesh: MeshConfig(
                cellCount: 100,
                majorRadius: 6.2,
                minorRadius: 2.0,
                toroidalField: 5.3
            ),
            solver: SolverConfig(
                type: "unknown",  // Invalid solver type
                tolerance: 1e-6,
                maximumIterations: 30
            )
        )

        // Should throw ConfigurationError.invalidValue
        #expect(throws: ConfigurationError.self) {
            try staticConfig.runtimeParameters()
        }
    }

    @Test("JSON with ionTemperature/electronTemperature decodes correctly")
    func testJSONProfileEvolutionDecoding() throws {
        let json = """
        {
            "mesh": {
                "cellCount": 75,
                "majorRadius": 4.0,
                "minorRadius": 1.5,
                "toroidalField": 3.5,
                "geometryType": "circular"
            },
            "evolution": {
                "ionTemperature": true,
                "electronTemperature": false,
                "electronDensity": true,
                "poloidalFlux": false
            },
            "solver": {
                "type": "linear",
                "tolerance": 1e-5,
                "maximumIterations": 25
            },
            "scheme": {
                "theta": 1.0,
                "usePereverzev": false
            }
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let staticConfig = try decoder.decode(StaticConfig.self, from: json)

        // Verify JSON fields are mapped correctly
        #expect(staticConfig.evolution.ionHeat == true)
        #expect(staticConfig.evolution.electronHeat == false)
        #expect(staticConfig.evolution.electronDensity == true)
        #expect(staticConfig.evolution.poloidalFlux == false)

        // Verify conversion works
        let runtimeParams = try staticConfig.runtimeParameters()
        #expect(runtimeParams.evolveIonHeat == true)
        #expect(runtimeParams.evolveElectronHeat == false)
    }
}
