import Testing
import MLX
@testable import GotenxCore

@Suite("SimulationRunner Energy Diagnostics Experiment")
struct SimulationRunnerEnergyDiagnosticsExperimentTests {

    @Test("Runner energy diagnostics are consistent with final-state source metadata", .timeLimit(.minutes(1)))
    func runnerEnergyDiagnosticsUseFinalStateSources() async throws {
        let config = try Self.makeConfig()
        let transport = ConstantTransportModel(
            ionHeatDiffusivity: 1.0,
            electronHeatDiffusivity: 1.0,
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
        #expect(runnerDerived.thermalEnergy > 0)
        #expect(runnerDerived.ionThermalEnergy > 0)
        #expect(runnerDerived.electronThermalEnergy > 0)
        #expect(runnerDerived.auxiliaryPower > 0)
        #expect(runnerDerived.ohmicPower > 0)
        #expect(runnerDerived.energyConfinementTime > 0)

        let finalProfiles = CoreProfiles(from: result.finalProfiles)
        let geometry = Geometry(config: config.runtime.static.mesh)
        let finalSources = source.computeTerms(
            profiles: finalProfiles,
            geometry: geometry,
            parameters: SourceParameters(modelType: "composite")
        )
        let recomputed = DerivedQuantitiesComputer.compute(
            profiles: finalProfiles,
            geometry: geometry,
            sources: finalSources
        )

        #expect(Self.relativeDifference(runnerDerived.thermalEnergy, recomputed.thermalEnergy) < 1e-5)
        #expect(Self.relativeDifference(runnerDerived.auxiliaryPower, recomputed.auxiliaryPower) < 1e-5)
        #expect(Self.relativeDifference(runnerDerived.ohmicPower, recomputed.ohmicPower) < 1e-5)
        #expect(Self.relativeDifference(runnerDerived.energyConfinementTime, recomputed.energyConfinementTime) < 1e-5)

        let heatingPower = runnerDerived.auxiliaryPower + runnerDerived.ohmicPower + runnerDerived.alphaPower
        let expectedTauE = runnerDerived.thermalEnergy / heatingPower
        #expect(Self.relativeDifference(runnerDerived.energyConfinementTime, expectedTauE) < 1e-5)

        print(
            """
            ENERGY DIAGNOSTICS EXPERIMENT:
              thermalEnergy=\(runnerDerived.thermalEnergy) MJ
              auxiliaryPower=\(runnerDerived.auxiliaryPower) MW
              ohmicPower=\(runnerDerived.ohmicPower) MW
              energyConfinementTime=\(runnerDerived.energyConfinementTime) s
              final_time=\(finalTimePoint.time) s
              steps=\(result.statistics.totalSteps)
            """
        )
    }

    private static func makeConfig() throws -> SimulationConfiguration {
        let mesh = MeshConfig(
            cellCount: 30,
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
                        electronDensity: false,
                        poloidalFlux: false
                    ),
                    solver: SolverConfig(
                        type: "newton",
                        tolerance: 1e-6,
                        tolerances: nil,
                        physicalThresholds: .default,
                        maximumIterations: 30
                    ),
                    scheme: SchemeConfig(theta: 1.0, usePereverzev: false)
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: 1000,
                        electronTemperature: 1000,
                        electronDensity: 1e19,
                        type: .dirichlet
                    ),
                    transport: try TransportConfig(
                        modelType: .constant,
                        parameters: [
                            "ionHeatDiffusivity": 1.0,
                            "electronHeatDiffusivity": 1.0,
                            "particleDiffusivity": 0.0
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
                initialTimeStep: 1e-4,
                adaptive: AdaptiveTimestepConfig(
                    minimumTimeStep: 1e-6,
                    minimumTimeStepFraction: nil,
                    maximumTimeStep: 1e-4,
                    safetyFactor: 0.9,
                    maximumTimeStepGrowth: 1.0
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

    func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        let terms = makeTerms(
            profiles: context.profiles,
            evaluationMode: context.evaluationMode
        )

        guard context.includesMetadata else {
            return terms
        }

        let volumes = context.geometricFactors.cellVolumes.value
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

        return terms.replacingMetadata(
            metadata,
            validateDebugUnits: context.validatesDebugUnits
        )
    }

    func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) -> SourceTerms {
        let context = SourceEvaluationContext(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters,
            purpose: .diagnostic
        )
        let terms = makeTerms(
            profiles: profiles,
            evaluationMode: context.evaluationMode
        )
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

    private func makeTerms(
        profiles: CoreProfiles,
        evaluationMode: MLXEvaluationMode
    ) -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let ionHeating = profiles.ionTemperature.value * 0.0002
        let electronHeating = profiles.electronTemperature.value * 0.0004
        let evaluated = evaluationMode.wrapBatch([
            ionHeating,
            electronHeating,
            MLXArray.zeros([cellCount]),
            MLXArray.zeros([cellCount])
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
