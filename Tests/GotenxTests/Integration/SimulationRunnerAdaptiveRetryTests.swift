import Testing
@testable import GotenxCore
import GotenxPhysics

@Suite("SimulationRunner Adaptive Retry")
struct SimulationRunnerAdaptiveRetryTests {

    @Test("Bohm-GyroBohm app preset can retry below adaptive minimum timestep", .timeLimit(.minutes(1)))
    func bohmGyroBohmPresetRetriesBelowAdaptiveMinimumTimestep() async throws {
        let config = try Self.makeBohmGyroBohmAppPresetConfig()
        let transportModel = try TransportModelFactory.create(config: config.runtime.dynamic.transport)
        let sourceModel = try SourceModelFactory.create(config: config.runtime.dynamic.sources)
        let mhdModels = MHDModelFactory.createAllModels(config: config.runtime.dynamic.mhd)

        let runner = SimulationRunner(config: config)
        try await runner.initialize(
            transportModel: transportModel,
            sourceModels: [sourceModel],
            mhdModels: mhdModels
        )

        let result = try await runner.run()

        #expect(result.statistics.converged)
        let edgeIonTemperature = try #require(result.finalProfiles.ionTemperature.last)
        let edgeElectronTemperature = try #require(result.finalProfiles.electronTemperature.last)
        let finalTimePoint = try #require(result.timeSeries?.last)
        #expect(abs(edgeIonTemperature - 1000.0) < 1.0)
        #expect(abs(edgeElectronTemperature - 1000.0) < 1.0)
        #expect(result.statistics.totalSteps >= 2)
        #expect(abs(finalTimePoint.time - config.time.end) < 1e-8)
    }

    private static func makeBohmGyroBohmAppPresetConfig() throws -> SimulationConfiguration {
        try SimulationConfiguration.build { builder in
            builder.time.start = 0.0
            builder.time.end = 2.5e-4
            builder.time.initialTimeStep = 1.5e-4

            builder.runtime.static.mesh.cellCount = 100
            builder.runtime.static.mesh.majorRadius = 6.2
            builder.runtime.static.mesh.minorRadius = 2.0
            builder.runtime.static.mesh.toroidalField = 5.3

            builder.runtime.dynamic.transport = try TransportConfig(
                modelType: .bohmGyrobohm,
                parameters: [
                    "bohmCoefficient": 0.5,
                    "gyroBohmCoefficient": 0.5,
                    "ionMassNumber": 2.0
                ]
            )

            builder.runtime.dynamic.boundaries = BoundaryConfig(
                ionTemperature: 1000.0,
                electronTemperature: 1000.0,
                electronDensity: 2.0e19
            )

            builder.runtime.dynamic.initialProfile = InitialProfileConfig.flat
            builder.output.saveInterval = 0.1
            builder.output.directory = "/tmp/gotenx_results"
        }
    }
}
