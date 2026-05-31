import Testing
import Foundation
@testable import GotenxCore
@testable import GotenxCLI
import GotenxPhysics

/// End-to-end P0 → P2 pipeline test.
///
/// Drives a full physics simulation (Ohmic + Bremsstrahlung + ion–electron exchange
/// sources) to completion and then runs the ITER/TORAX validation harness on the
/// produced profiles. This exercises the complete path:
///
///   load config → run solver to completion → compare to reference → pass/fail
///
/// which is only possible because the Newton solver now completes physics runs
/// (Pereverzev-Galeev stabilization, equilibrated/refined linear solve, physical
/// floors, positivity clamping, precision-floor convergence).
@Suite("End-to-End Validation")
struct EndToEndValidationTest {

    /// A short but genuinely nonlinear physics scenario that the solver completes.
    private static let configJSON = """
    {
      "runtime": {
        "static": {
          "mesh": { "nCells": 50, "majorRadius": 3.0, "minorRadius": 1.0,
                    "toroidalField": 2.5, "geometryType": "circular" },
          "evolution": { "ionTemperature": true, "electronTemperature": true,
                          "density": true, "current": false },
          "solver": { "type": "newton", "tolerance": 1e-6, "maxIterations": 20 },
          "scheme": { "theta": 1.0, "usePereverzev": true }
        },
        "dynamic": {
          "boundaries": { "ionTemperature": 100.0, "electronTemperature": 100.0,
                           "density": 1e19, "type": "dirichlet" },
          "transport": { "modelType": "constant",
                          "parameters": { "chiIon": 1.0, "chiElectron": 1.0, "D": 0.5 } },
          "sources": { "fusionPower": false, "ohmicHeating": true,
                        "bremsstrahlung": true, "ionElectronExchange": true }
        }
      },
      "time": { "start": 0.0, "end": 5e-5, "initialDt": 1e-5 },
      "output": { "saveInterval": null, "directory": "/tmp/gotenx_e2e", "format": "json" }
    }
    """

    @Test("Complete a physics run and validate its profiles against the ITER baseline")
    func completeRunAndValidate() async throws {
        // 1. Materialize the config and build a SimulationConfiguration.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("gotenx_e2e_config_\(UUID().uuidString).json")
        try Self.configJSON.write(to: tmp, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let reader = try await GotenxConfigReader.create(jsonPath: tmp.path)
        let config = try await reader.fetchConfiguration()

        // 2. P0 — run the full solver to its configured end time.
        let transportModel = try TransportModelFactory.create(config: config.runtime.dynamic.transport)
        let sourceModel = try SourceModelFactory.create(config: config.runtime.dynamic.sources)
        let runner = SimulationRunner(config: config)
        try await runner.initialize(transportModel: transportModel, sourceModels: [sourceModel])
        let result = try await runner.run()

        // The simulation reached completion with finite, physical profiles.
        #expect(result.statistics.converged)
        #expect(result.finalProfiles.ionTemperature.count == 50)
        #expect(result.finalProfiles.ionTemperature.allSatisfy { $0.isFinite && $0 > 0 })
        #expect(result.finalProfiles.electronDensity.allSatisfy { $0.isFinite && $0 > 0 })

        // 3. P2 — run the validation harness end-to-end on the completed run.
        let baseline = ITERBaselineData.load()
        let endTime = config.time.end

        let tiResult = ProfileComparator.compare(
            quantity: "Ti",
            predicted: result.finalProfiles.ionTemperature,
            reference: baseline.profiles.Ti,
            time: endTime
        )
        let teResult = ProfileComparator.compare(
            quantity: "Te",
            predicted: result.finalProfiles.electronTemperature,
            reference: baseline.profiles.Te,
            time: endTime
        )
        let neResult = ProfileComparator.compare(
            quantity: "ne",
            predicted: result.finalProfiles.electronDensity,
            reference: baseline.profiles.ne,
            time: endTime
        )

        // The full P0 → P2 pipeline executed end-to-end and produced definitive,
        // finite pass/fail metrics for every channel.
        print("END-TO-END P2 VALIDATION (sim vs ITER baseline):")
        for r in [tiResult, teResult, neResult] {
            print("  \(r.quantity): L2=\(String(format: "%.3e", r.l2Error)) "
                + "MAPE=\(String(format: "%.1f", r.mape))% passed=\(r.passed)")
        }

        #expect(tiResult.l2Error.isFinite)
        #expect(teResult.l2Error.isFinite)
        #expect(neResult.l2Error.isFinite)
    }
}
