import Testing
import MLX
@testable import GotenxCore

@Suite("SimulationRunner Energy Diagnostics Experiment")
struct SimulationRunnerEnergyDiagnosticsExperimentTests {

    @Test("Runner energy diagnostics are consistent with final-state source metadata", .timeLimit(.minutes(1)))
    func runnerEnergyDiagnosticsUseFinalStateSources() async throws {
        let config = Self.makeConfig()
        let transport = ConstantTransportModel(
            chiIon: 1.0,
            chiElectron: 1.0,
            particleDiffusivity: 0.0
        )
        let source = TemperatureScaledDiagnosticSource()

        let runner = SimulationRunner(config: config)
        try await runner.initialize(transportModel: transport, sourceModels: [source], mhdModels: [])
        let result = try await runner.run()

        #expect(result.statistics.converged)

        let finalTimePoint = try #require(result.timeSeries?.last)
        let runnerDerived = try #require(finalTimePoint.derived)

        #expect(abs(finalTimePoint.time - config.time.end) < 1e-8)
        #expect(runnerDerived.W_thermal > 0)
        #expect(runnerDerived.W_ion > 0)
        #expect(runnerDerived.W_electron > 0)
        #expect(runnerDerived.P_auxiliary > 0)
        #expect(runnerDerived.P_ohmic > 0)
        #expect(runnerDerived.tau_E > 0)

        let finalProfiles = CoreProfiles(from: result.finalProfiles)
        let geometry = Geometry(config: config.runtime.static.mesh)
        let finalSources = source.computeTerms(
            profiles: finalProfiles,
            geometry: geometry,
            params: SourceParameters(modelType: "composite")
        )
        let recomputed = DerivedQuantitiesComputer.compute(
            profiles: finalProfiles,
            geometry: geometry,
            sources: finalSources
        )

        #expect(Self.relativeDifference(runnerDerived.W_thermal, recomputed.W_thermal) < 1e-5)
        #expect(Self.relativeDifference(runnerDerived.P_auxiliary, recomputed.P_auxiliary) < 1e-5)
        #expect(Self.relativeDifference(runnerDerived.P_ohmic, recomputed.P_ohmic) < 1e-5)
        #expect(Self.relativeDifference(runnerDerived.tau_E, recomputed.tau_E) < 1e-5)

        let heatingPower = runnerDerived.P_auxiliary + runnerDerived.P_ohmic + runnerDerived.P_alpha
        let expectedTauE = runnerDerived.W_thermal / heatingPower
        #expect(Self.relativeDifference(runnerDerived.tau_E, expectedTauE) < 1e-5)

        print(
            """
            ENERGY DIAGNOSTICS EXPERIMENT:
              W_thermal=\(runnerDerived.W_thermal) MJ
              P_auxiliary=\(runnerDerived.P_auxiliary) MW
              P_ohmic=\(runnerDerived.P_ohmic) MW
              tau_E=\(runnerDerived.tau_E) s
              final_time=\(finalTimePoint.time) s
              steps=\(result.statistics.totalSteps)
            """
        )
    }

    private static func makeConfig() -> SimulationConfiguration {
        let mesh = MeshConfig(
            nCells: 30,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5,
            geometryType: .circular
        )

        return SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: mesh,
                    evolution: EvolutionConfig(
                        ionHeat: true,
                        electronHeat: true,
                        density: false,
                        current: false
                    ),
                    solver: SolverConfig(
                        type: "newton",
                        tolerance: 1e-6,
                        tolerances: nil,
                        physicalThresholds: .default,
                        maxIterations: 30
                    ),
                    scheme: SchemeConfig(theta: 1.0, usePereverzev: false)
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: 1000,
                        electronTemperature: 1000,
                        density: 1e19,
                        type: .dirichlet
                    ),
                    transport: TransportConfig(
                        modelType: .constant,
                        parameters: [
                            "chi_ion": 1.0,
                            "chi_electron": 1.0,
                            "particle_diffusivity": 0.0
                        ]
                    ),
                    sources: SourcesConfig(
                        ohmicHeating: false,
                        fusionPower: false,
                        ionElectronExchange: false,
                        bremsstrahlung: false,
                        fusionConfig: nil
                    ),
                    initialProfile: .flat
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 5e-4,
                initialDt: 1e-4,
                adaptive: AdaptiveTimestepConfig(
                    minDt: 1e-6,
                    minDtFraction: nil,
                    maxDt: 1e-4,
                    safetyFactor: 0.9,
                    maxTimestepGrowth: 1.0
                )
            )
        )
    }

    private static func relativeDifference(_ lhs: Float, _ rhs: Float) -> Float {
        abs(lhs - rhs) / max(abs(rhs), 1e-12)
    }
}

private struct TemperatureScaledDiagnosticSource: SourceModel {
    let name = "composite"

    func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        let terms = makeTerms(profiles: profiles)
        let volumes = GeometricFactors.from(geometry: geometry).cellVolumes.value

        let ionPower = (terms.ionHeating.value * volumes).sum() * 1e6
        let electronPower = (terms.electronHeating.value * volumes).sum() * 1e6
        eval(ionPower, electronPower)

        let metadata = SourceMetadataCollection(entries: [
            SourceMetadata(
                modelName: "diagnostic_auxiliary",
                category: .auxiliary,
                ionPower: ionPower.item(Float.self),
                electronPower: 0
            ),
            SourceMetadata(
                modelName: "diagnostic_ohmic",
                category: .ohmic,
                ionPower: 0,
                electronPower: electronPower.item(Float.self)
            )
        ])

        return SourceTerms(
            ionHeating: terms.ionHeating,
            electronHeating: terms.electronHeating,
            particleSource: terms.particleSource,
            currentSource: terms.currentSource,
            metadata: metadata
        )
    }

    func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        makeTerms(profiles: profiles)
    }

    private func makeTerms(profiles: CoreProfiles) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        let ionHeating = profiles.ionTemperature.value * 0.0002
        let electronHeating = profiles.electronTemperature.value * 0.0004
        let evaluated = EvaluatedArray.evaluatingBatch([
            ionHeating,
            electronHeating,
            MLXArray.zeros([nCells]),
            MLXArray.zeros([nCells])
        ])

        return SourceTerms(
            ionHeating: evaluated[0],
            electronHeating: evaluated[1],
            particleSource: evaluated[2],
            currentSource: evaluated[3],
            metadata: nil,
            validateDebugUnits: false
        )
    }
}
