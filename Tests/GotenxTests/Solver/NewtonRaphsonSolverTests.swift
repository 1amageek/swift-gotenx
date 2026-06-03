import MLX
import Testing
@testable import GotenxCore

@Suite("Newton-Raphson Solver")
struct NewtonRaphsonSolverTests {
    @Test("Backward Euler skips unused old-time coefficients", .timeLimit(.minutes(1)))
    func backwardEulerSkipsUnusedOldTimeCoefficients() throws {
        let context = try TestContext(theta: 1.0)
        let result = context.solveWithInvalidOldGeometryCoefficients()

        #expect(result.converged)
        #expect(result.metadata["failure_type"] != 5.0)
    }

    @Test("Theta methods validate required old-time coefficients", .timeLimit(.minutes(1)))
    func thetaMethodsValidateRequiredOldTimeCoefficients() throws {
        let context = try TestContext(theta: 0.5)
        let result = context.solveWithInvalidOldGeometryCoefficients()

        #expect(!result.converged)
        #expect(result.iterations == 0)
        #expect(result.metadata["failure_type"] == 5.0)
    }
}

private struct TestContext {
    let cellCount = 8
    let oldGeometry: Geometry
    let newGeometry: Geometry
    let staticParameters: StaticRuntimeParameters
    let dynamicParameters: DynamicRuntimeParameters
    let profiles: CoreProfiles

    init(theta: Float) throws {
        let oldMesh = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        let newMesh = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.4,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        self.oldGeometry = Geometry(config: oldMesh)
        self.newGeometry = Geometry(config: newMesh)
        self.staticParameters = StaticRuntimeParameters(
            mesh: newMesh,
            evolveIonHeat: false,
            evolveElectronHeat: false,
            evolveElectronDensity: false,
            evolvePoloidalFlux: false,
            solverType: .newtonRaphson,
            theta: theta,
            solverTolerance: 1e-5,
            solverMaximumIterations: 2
        )

        let boundaryConditions = BoundaryConditions(
            ionTemperature: BoundaryCondition(left: .gradient(0.0), right: .value(100.0)),
            electronTemperature: BoundaryCondition(left: .gradient(0.0), right: .value(100.0)),
            electronDensity: BoundaryCondition(left: .gradient(0.0), right: .value(1e19)),
            poloidalFlux: BoundaryCondition(left: .value(0.0), right: .gradient(0.0))
        )
        self.dynamicParameters = DynamicRuntimeParameters(
            timeStep: 1e-4,
            boundaryConditions: boundaryConditions,
            profileConditions: ProfileConditions(
                ionTemperature: .constant(1_000.0),
                electronTemperature: .constant(1_000.0),
                electronDensity: .constant(1e20),
                currentDensity: .constant(0.0)
            ),
            transportParameters: try TransportParameters(
                modelType: .constant,
                parameters: [
                    "ionHeatDiffusivity": 0.0,
                    "electronHeatDiffusivity": 0.0,
                    "particleDiffusivity": 0.0,
                    "convectionVelocity": 0.0
                ]
            )
        )
        self.profiles = CoreProfiles(
            ionTemperature: .full([cellCount], value: 1_000.0),
            electronTemperature: .full([cellCount], value: 1_000.0),
            electronDensity: .full([cellCount], value: 1e20),
            poloidalFlux: .full([cellCount], value: 0.0)
        )
    }

    func solveWithInvalidOldGeometryCoefficients() -> SolverResult {
        let solver = NewtonRaphsonSolver(
            tolerance: staticParameters.solverTolerance,
            maximumIterations: staticParameters.solverMaximumIterations,
            theta: staticParameters.theta
        )
        let oldMajorRadius = oldGeometry.majorRadius
        let callback: CoeffsCallback = { profiles, geometry in
            if abs(geometry.majorRadius - oldMajorRadius) < 1e-6 {
                return Self.invalidCoefficients(cellCount: cellCount, geometry: geometry)
            }
            return Self.validCoefficients(
                profiles: profiles,
                geometry: geometry,
                staticParameters: staticParameters
            )
        }

        return solver.solve(
            timeStep: dynamicParameters.timeStep,
            staticParameters: staticParameters,
            dynamicParamsT: dynamicParameters,
            dynamicParamsTplusDt: dynamicParameters,
            geometryT: oldGeometry,
            geometryTplusDt: newGeometry,
            xOld: profiles.asTuple(
                radialSpacing: staticParameters.mesh.radialSpacing,
                boundaryConditions: dynamicParameters.boundaryConditions
            ),
            coreProfilesT: profiles,
            coreProfilesTplusDt: profiles,
            coeffsCallback: callback
        )
    }

    private static func validCoefficients(
        profiles: CoreProfiles,
        geometry: Geometry,
        staticParameters: StaticRuntimeParameters
    ) -> Block1DCoeffs {
        let cellCount = profiles.ionTemperature.shape[0]
        let transport = TransportCoefficients(
            ionHeatDiffusivity: .full([cellCount], value: 0.0),
            electronHeatDiffusivity: .full([cellCount], value: 0.0),
            particleDiffusivity: .full([cellCount], value: 0.0),
            convectionVelocity: .full([cellCount], value: 0.0)
        )
        let sources = SourceTerms.zero(
            cellCount: cellCount,
            evaluationMode: .deferred,
            metadata: nil,
            validateDebugUnits: false
        )
        return buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )
    }

    private static func invalidCoefficients(cellCount: Int, geometry: Geometry) -> Block1DCoeffs {
        let faceCount = cellCount + 1
        let invalidEquation = EquationCoeffs(
            faceDiffusionCoefficient: MLXArray.full([faceCount], values: MLXArray(Float.nan)),
            faceConvectionVelocity: MLXArray.zeros([faceCount]),
            cellSource: MLXArray.zeros([cellCount]),
            cellSourceMatrixCoefficient: MLXArray.zeros([cellCount]),
            transientCoefficient: MLXArray.ones([cellCount])
        )
        return Block1DCoeffs(
            ionCoeffs: invalidEquation,
            electronCoeffs: invalidEquation,
            densityCoeffs: invalidEquation,
            fluxCoeffs: invalidEquation,
            geometry: GeometricFactors.from(geometry: geometry)
        )
    }
}
