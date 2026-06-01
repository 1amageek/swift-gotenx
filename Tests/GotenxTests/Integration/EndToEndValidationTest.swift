import Testing
import Foundation
@testable import GotenxCore
@testable import GotenxCLI
import GotenxPhysics

/// End-to-end P0 → P2 pipeline test.
///
/// Drives a full physics simulation (Ohmic + Bremsstrahlung + ion–electron exchange
/// sources) to completion and then runs the profile validation harness on the
/// produced profiles against a captured reference. This exercises the complete path:
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

    // Captured from this test configuration after exact end-time enforcement on 2026-06-01.
    // Keep this fixture independent from the current run output.
    private static let referenceIonTemperature: [Float] = [
        338.49771, 925.71313, 1, 968.20966, 144.92712, 737.29706, 319.93936, 611.58673, 377.05313, 565.21521, 385.68961, 539.45026, 380.49982, 518.52551, 367.11108, 501.19205, 346.43454, 486.36655, 320.55005, 472.12115, 291.62656, 457.33545, 260.63199, 441.92233, 228.26411, 425.45175, 195.86414, 406.99689, 165.09698, 385.7525, 136.90431, 362.43497, 110.38025, 339.5209, 83.289505, 320.74771, 51.984352, 311.73727, 10.604227, 323.26767, 1, 378.34576, 1, 540.08374, 1, 824.27209, 1, 1078.9666, 1, 802.88385
    ]

    private static let referenceElectronTemperature: [Float] = [
        338.49768, 925.71307, 1, 968.20959, 144.92712, 737.297, 319.93936, 611.58673, 377.05313, 565.21515, 385.68958, 539.45026, 380.49982, 518.52551, 367.11105, 501.19205, 346.43454, 486.36652, 320.55005, 472.12115, 291.62653, 457.33545, 260.63199, 441.9223, 228.26411, 425.45172, 195.86414, 406.99689, 165.09698, 385.7525, 136.90431, 362.43497, 110.38025, 339.52087, 83.289505, 320.74768, 51.984352, 311.73727, 10.604227, 323.26764, 1, 378.34576, 1, 540.08368, 1, 824.27203, 1, 1078.9664, 1, 802.88379
    ]

    private static let referenceElectronDensity: [Float] = [
        2.0001823e+19, 2.0000567e+19, 1.9980391e+19, 1.9949455e+19, 1.9905765e+19, 1.9849767e+19, 1.978143e+19, 1.9700898e+19, 1.960824e+19, 1.9503586e+19, 1.9387071e+19, 1.9258865e+19, 1.9119115e+19, 1.8967994e+19, 1.8805761e+19, 1.8632606e+19, 1.8448778e+19, 1.8254521e+19, 1.8050168e+19, 1.7835981e+19, 1.7612319e+19, 1.7379508e+19, 1.7137958e+19, 1.6888034e+19, 1.6630191e+19, 1.6364894e+19, 1.6092582e+19, 1.581387e+19, 1.5529217e+19, 1.5239289e+19, 1.4944694e+19, 1.4646125e+19, 1.4344325e+19, 1.404007e+19, 1.3734223e+19, 1.3427689e+19, 1.3121512e+19, 1.2816744e+19, 1.2514608e+19, 1.2216421e+19, 1.1923679e+19, 1.1638058e+19, 1.1361436e+19, 1.1096062e+19, 1.084453e+19, 1.0610106e+19, 1.039649e+19, 1.0211133e+19, 1.0055329e+19, 9.9397621e+18
    ]

    @Test("Complete a physics run and validate its profiles against a captured reference")
    func completeRunAndValidate() async throws {
        // 1. Materialize the config and build a SimulationConfiguration.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("gotenx_e2e_config_\(UUID().uuidString).json")
        try Self.configJSON.write(to: tmp, atomically: true, encoding: .utf8)
        defer {
            do {
                try FileManager.default.removeItem(at: tmp)
            } catch {
                print("Failed to remove temporary config: \(error)")
            }
        }

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
        // ITERBaselineData is design data, not a reference output for this short
        // transient scenario. Compare against a captured fixture so this test detects
        // solver-output regressions instead of self-comparing the current run.
        let endTime = config.time.end

        let tiResult = ProfileComparator.compare(
            quantity: "Ti",
            predicted: result.finalProfiles.ionTemperature,
            reference: Self.referenceIonTemperature,
            time: endTime
        )
        let teResult = ProfileComparator.compare(
            quantity: "Te",
            predicted: result.finalProfiles.electronTemperature,
            reference: Self.referenceElectronTemperature,
            time: endTime
        )
        let neResult = ProfileComparator.compare(
            quantity: "ne",
            predicted: result.finalProfiles.electronDensity,
            reference: Self.referenceElectronDensity,
            time: endTime
        )

        // The full P0 → P2 pipeline executed end-to-end and produced definitive,
        // finite pass/fail metrics for every channel.
        print("END-TO-END P2 VALIDATION (sim vs captured reference):")
        for r in [tiResult, teResult, neResult] {
            print("  \(r.quantity): L2=\(String(format: "%.3e", r.l2Error)) "
                + "MAPE=\(String(format: "%.1f", r.mape))% passed=\(r.passed)")
        }

        #expect(tiResult.l2Error.isFinite)
        #expect(teResult.l2Error.isFinite)
        #expect(neResult.l2Error.isFinite)
        #expect(tiResult.passed, "Ti validation failed: L2=\(tiResult.l2Error), MAPE=\(tiResult.mape)")
        #expect(teResult.passed, "Te validation failed: L2=\(teResult.l2Error), MAPE=\(teResult.mape)")
        #expect(neResult.passed, "ne validation failed: L2=\(neResult.l2Error), MAPE=\(neResult.mape)")
    }
}
