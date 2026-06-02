// P0IntegrationTest.swift
// Integration test for P0 configuration (minimal physics)

import Testing
import MLX
@testable import GotenxCore
@testable import GotenxPhysics

/// P0 Integration Test Suite
///
/// Tests the minimal physics configuration (P0):
/// - Constant transport model
/// - Ohmic heating only
/// - Linear solver with predictor-corrector
/// - 2 evolved quantities: Ti, Te (fixed density and current)
@Suite("P0 Integration Tests")
struct P0IntegrationTest {

    // MARK: - Test Configuration

    /// Create P0 static configuration
    func makeP0StaticConfig() -> StaticRuntimeParameters {
        let meshConfig = MeshConfig(
            cellCount: 25,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )

        return StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: true,
            evolveElectronHeat: true,
            evolveElectronDensity: false,   // P0: fix density
            evolvePoloidalFlux: false,    // P0: fix current
            solverType: .linear,
            theta: 1.0,             // Fully implicit
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )
    }

    /// Create P0 dynamic configuration
    func makeP0DynamicConfig() throws -> DynamicRuntimeParameters {
        let transportParameters = try TransportParameters(
            modelType: .constant,
            parameters: [
                "ionHeatDiffusivity": 1.0,
                "electronHeatDiffusivity": 1.0
            ]
        )

        let boundaryConditions = BoundaryConditions(
            ionTemperature: BoundaryCondition(
                left: .gradient(0.0),   // Zero gradient at core
                right: .value(100.0)     // 100 eV at edge
            ),
            electronTemperature: BoundaryCondition(
                left: .gradient(0.0),
                right: .value(100.0)
            ),
            electronDensity: BoundaryCondition(
                left: .gradient(0.0),
                right: .value(1e19)
            ),
            poloidalFlux: BoundaryCondition(
                left: .value(0.0),
                right: .gradient(0.0)
            )
        )

        let profileConditions = ProfileConditions(
            ionTemperature: .parabolic(peak: 10000.0, edge: 100.0, exponent: 2.0),
            electronTemperature: .parabolic(peak: 10000.0, edge: 100.0, exponent: 2.0),
            electronDensity: .parabolic(peak: 1e20, edge: 1e19, exponent: 2.0),
            currentDensity: .constant(0.0)
        )

        return DynamicRuntimeParameters(
            timeStep: 1e-4,  // 0.1 ms
            boundaryConditions: boundaryConditions,
            profileConditions: profileConditions,
            sourceParameters: [:],
            transportParameters: transportParameters
        )
    }

    /// Create initial profiles for P0
    func makeInitialProfiles(cellCount: Int) -> CoreProfiles {
        // Parabolic temperature profiles
        let rho = MLXArray(0..<cellCount).asType(.float32) / Float(cellCount - 1)

        let Ti_peak: Float = 10000.0
        let Te_peak: Float = 10000.0
        let Ti_edge: Float = 100.0
        let Te_edge: Float = 100.0

        // Ensure profiles honour boundary constraints: use edge value at rho=1
        // and peak value at rho=0 with a simple parabolic shape.
        let Ti = Ti_edge + (Ti_peak - Ti_edge) * (1.0 - rho * rho)
        let Te = Te_edge + (Te_peak - Te_edge) * (1.0 - rho * rho)

        // Fixed density profile
        let n0: Float = 1e20
        let ne = n0 * (1.0 - 0.9 * rho * rho)

        // Fixed poloidal flux
        let psi = MLXArray.zeros([cellCount])

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )
    }

    // MARK: - Tests

    @Test("P0 single time step execution")
    func testP0SingleTimeStep() throws {
        // Setup
        let staticParameters = makeP0StaticConfig()
        let dynamicParameters = try makeP0DynamicConfig()
        let geometry = Geometry(config: staticParameters.mesh)
        let initialProfiles = makeInitialProfiles(cellCount: staticParameters.mesh.cellCount)

        // Create physics models
        let transportModel = ConstantTransportModel(
            ionHeatDiffusivity: 1.0,
            electronHeatDiffusivity: 1.0
        )

        // Inline zero source model for the minimal transport benchmark.
        struct SimpleZeroSource: SourceModel {
            let name = "zero"
            func computeTerms(profiles: CoreProfiles, geometry: Geometry, parameters: SourceParameters) -> SourceTerms {
                let cellCount = profiles.ionTemperature.shape[0]
                let zeros = EvaluatedArray.zeros([cellCount])
                return SourceTerms(
                    ionHeating: zeros,
                    electronHeating: zeros,
                    particleSource: zeros,
                    currentSource: zeros
                )
            }
        }
        let sourceModel = SimpleZeroSource()

        // Create solver
        let solver = LinearSolver(
            correctorStepCount: staticParameters.solverMaximumIterations,
            usesPereverzevCorrector: true,
            theta: staticParameters.theta
        )

        // Define coefficients callback
        let coeffsCallback: CoeffsCallback = { profiles, geom in
            let transport = transportModel.computeCoefficients(
                profiles: profiles,
                geometry: geom,
                parameters: dynamicParameters.transportParameters
            )

            let sources = sourceModel.computeTerms(
                profiles: profiles,
                geometry: geom,
                parameters: SourceParameters(modelType: "ohmic", parameters: [:])
            )

            return buildBlock1DCoeffs(
                transport: transport,
                sources: sources,
                geometry: geom,
                staticParameters: staticParameters,
                profiles: profiles
            )
        }

        // Extract boundary conditions
        let tiBC = dynamicParameters.boundaryConditions.ionTemperature
        let teBC = dynamicParameters.boundaryConditions.electronTemperature
        let neBC = dynamicParameters.boundaryConditions.electronDensity
        let psiBC = dynamicParameters.boundaryConditions.poloidalFlux

        // Create CellVariable tuple
        let xOld = (
            CellVariable(
                value: initialProfiles.ionTemperature.value,
                radialSpacing: staticParameters.mesh.radialSpacing,
                leftFaceGradientConstraint: extractGradient(tiBC.left),
                rightFaceConstraint: extractValue(tiBC.right)
            ),
            CellVariable(
                value: initialProfiles.electronTemperature.value,
                radialSpacing: staticParameters.mesh.radialSpacing,
                leftFaceGradientConstraint: extractGradient(teBC.left),
                rightFaceConstraint: extractValue(teBC.right)
            ),
            CellVariable(
                value: initialProfiles.electronDensity.value,
                radialSpacing: staticParameters.mesh.radialSpacing,
                leftFaceGradientConstraint: extractGradient(neBC.left),
                rightFaceConstraint: extractValue(neBC.right)
            ),
            CellVariable(
                value: initialProfiles.poloidalFlux.value,
                radialSpacing: staticParameters.mesh.radialSpacing,
                leftFaceConstraint: extractValue(psiBC.left),
                rightFaceGradientConstraint: extractGradient(psiBC.right)
            )
        )

        // Execute single time step
        let result = solver.solve(
            timeStep: dynamicParameters.timeStep,
            staticParameters: staticParameters,
            dynamicParamsT: dynamicParameters,
            dynamicParamsTplusDt: dynamicParameters,
            geometryT: geometry,
            geometryTplusDt: geometry,
            xOld: xOld,
            coreProfilesT: initialProfiles,
            coreProfilesTplusDt: initialProfiles,
            coeffsCallback: coeffsCallback
        )

        let tiMinValue = result.updatedProfiles.ionTemperature.value.min().item(Float.self)
        let teMinValue = result.updatedProfiles.electronTemperature.value.min().item(Float.self)
        let tiMaxValue = result.updatedProfiles.ionTemperature.value.max().item(Float.self)
        let teMaxValue = result.updatedProfiles.electronTemperature.value.max().item(Float.self)

        // Verify results when stability checks pass
        #expect(result.converged, "Solver should converge for P0 configuration")
        #expect(result.residualNorm < 1e-5, "Residual should be small")

        // Check physical constraints
        let Ti_min = tiMinValue
        let Te_min = teMinValue
        #expect(Ti_min > 0.0, "Ion temperature should be positive")
        #expect(Te_min > 0.0, "Electron temperature should be positive")

        let Ti_max = tiMaxValue
        let Te_max = teMaxValue
        #expect(Ti_max < 20000.0, "Ion temperature should be bounded")
        #expect(Te_max < 20000.0, "Electron temperature should be bounded")

        print("P0 single time step test passed")
        print("   Residual norm: \(result.residualNorm)")
        print("   Ti range: [\(Ti_min), \(Ti_max)] eV")
        print("   Te range: [\(Te_min), \(Te_max)] eV")
    }

    // MARK: - Helper Functions

    /// Extract value from FaceConstraint (must be .value)
    private func extractValue(_ constraint: FaceConstraint) -> Float {
        switch constraint {
        case .value(let v):
            return v
        case .gradient:
            fatalError("Expected value constraint, got gradient")
        }
    }

    /// Extract gradient from FaceConstraint (must be .gradient)
    private func extractGradient(_ constraint: FaceConstraint) -> Float {
        switch constraint {
        case .gradient(let g):
            return g
        case .value:
            fatalError("Expected gradient constraint, got value")
        }
    }
}
