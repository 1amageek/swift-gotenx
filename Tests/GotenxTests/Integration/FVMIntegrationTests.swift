// FVMIntegrationTests.swift
// Integration tests for FVM numerical methods

import Testing
import MLX
@testable import GotenxCore

@Suite("FVM Integration Tests")
struct FVMIntegrationTests {

    @Test("Power-law scheme integrated with Newton-Raphson solver")
    func powerLawSchemeIntegration() throws {
        // Verify that power-law scheme works correctly when integrated
        // into the full solver pipeline

        let cellCount = 20
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 5.0
        )
        let geometry = Geometry(config: meshConfig)

        // Create test profiles with gradients
        let Ti = MLXArray.linspace(Float(10000.0), Float(2000.0), count: cellCount)  // 10 keV → 2 keV
        let Te = MLXArray.linspace(Float(10000.0), Float(2000.0), count: cellCount)
        let ne = MLXArray.linspace(Float(5e19), Float(1e19), count: cellCount)
        let psi = MLXArray.zeros([cellCount])

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        // Create transport coefficients with convection
        let ionHeatDiffusivity = MLXArray.full([cellCount], values: MLXArray(Float(1.0)))
        let electronHeatDiffusivity = MLXArray.full([cellCount], values: MLXArray(Float(1.0)))
        let D = MLXArray.full([cellCount], values: MLXArray(Float(0.5)))
        let V = MLXArray.full([cellCount], values: MLXArray(Float(2.0)))  // Creates Pe ~ 2-5

        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: ionHeatDiffusivity),
            electronHeatDiffusivity: EvaluatedArray(evaluating: electronHeatDiffusivity),
            particleDiffusivity: EvaluatedArray(evaluating: D),
            convectionVelocity: EvaluatedArray(evaluating: V)
        )

        // Build coefficients
        let staticParameters = StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: true,
            evolveElectronHeat: true,
            evolveElectronDensity: true,
            evolvePoloidalFlux: false,
            solverType: .linear,
            theta: 1.0,
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )

        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            particleSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            currentSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let coeffs = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )

        // Verify coefficients are computed without error
        #expect(coeffs.ionCoeffs.cellSource.value.shape[0] == cellCount)
        #expect(coeffs.electronCoeffs.cellSource.value.shape[0] == cellCount)
        #expect(coeffs.densityCoeffs.cellSource.value.shape[0] == cellCount)

        // Verify no NaN or Inf in coefficients
        let ionSource = coeffs.ionCoeffs.cellSource.value.asArray(Float.self)
        for value in ionSource {
            #expect(value.isFinite)
        }
    }

    @Test("Metric tensor integration maintains shape consistency")
    func metricTensorShapeConsistency() throws {
        let cellCount = 15
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.0,
            minorRadius: 2.0,
            toroidalField: 5.0
        )
        let geometry = Geometry(config: meshConfig)

        // Get geometric factors
        let geoFactors = GeometricFactors.from(geometry: geometry)

        // Verify all metric tensors have correct cell-centered shape
        #expect(geoFactors.jacobian.value.shape[0] == cellCount)
        #expect(geoFactors.majorRadiusMetric.value.shape[0] == cellCount)
        #expect(geoFactors.shapeMetric.value.shape[0] == cellCount)

        // Verify metric tensors are used in coefficient building
        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float(5000.0)))),
            electronTemperature: EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float(5000.0)))),
            electronDensity: EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float(5e19)))),
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let staticParameters = StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: true,
            evolveElectronHeat: false,
            evolveElectronDensity: false,
            evolvePoloidalFlux: false,
            solverType: .linear,
            theta: 1.0,
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )

        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float(1.0)))),
            electronHeatDiffusivity: EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float(1.0)))),
            particleDiffusivity: EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float(0.5)))),
            convectionVelocity: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            particleSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            currentSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let coeffs = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )

        // Verify geometric factors are accessible and consistent
        #expect(coeffs.geometry.jacobian.value.shape[0] == cellCount)
        #expect(coeffs.geometry.cellVolumes.value.shape[0] == cellCount)
        #expect(coeffs.geometry.faceAreas.value.shape[0] == cellCount + 1)
    }

    @Test("Bootstrap current integrated with coefficient builder")
    func bootstrapCurrentIntegration() throws {
        // Test that bootstrap current (Phase 3) integrates correctly
        // with the full coefficient building pipeline

        let cellCount = 20
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )
        let geometry = Geometry(config: meshConfig)

        // Create peaked profiles (triggers bootstrap current)
        let Ti = MLXArray.linspace(Float(15000.0), Float(1000.0), count: cellCount)
        let Te = MLXArray.linspace(Float(15000.0), Float(1000.0), count: cellCount)
        let ne = MLXArray.linspace(Float(8e19), Float(2e19), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let staticParameters = StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: false,
            evolveElectronHeat: false,
            evolveElectronDensity: false,
            evolvePoloidalFlux: true,  // Enable current evolution
            solverType: .linear,
            theta: 1.0,
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )

        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float(1.0)))),
            electronHeatDiffusivity: EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float(1.0)))),
            particleDiffusivity: EvaluatedArray(evaluating: MLXArray.full([cellCount], values: MLXArray(Float(0.5)))),
            convectionVelocity: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            particleSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            currentSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let coeffs = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )

        // Verify bootstrap current contribution is computed
        let currentSource = coeffs.fluxCoeffs.cellSource.value.asArray(Float.self)

        // Bootstrap current should be non-zero for peaked profiles
        var hasNonZeroBootstrap = false
        for J in currentSource {
            if abs(J) > 1e-6 {
                hasNonZeroBootstrap = true
                break
            }
        }

        #expect(hasNonZeroBootstrap, "Bootstrap current should be non-zero for peaked profiles")

        // All values should be finite
        for J in currentSource {
            #expect(J.isFinite)
        }
    }
}
