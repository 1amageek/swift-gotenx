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
          "mesh": { "cellCount": 50, "majorRadius": 3.0, "minorRadius": 1.0,
                    "toroidalField": 2.5, "geometryType": "circular" },
          "evolution": { "ionTemperature": true, "electronTemperature": true,
                          "electronDensity": true, "poloidalFlux": false },
          "solver": { "type": "newton", "tolerance": 1e-6, "maximumIterations": 20 },
          "scheme": { "theta": 1.0, "usePereverzev": true }
        },
        "dynamic": {
          "boundaries": { "ionTemperature": 100.0, "electronTemperature": 100.0,
                           "electronDensity": 1e19, "type": "dirichlet" },
          "transport": { "modelType": "constant",
                          "parameters": { "ionHeatDiffusivity": 1.0, "electronHeatDiffusivity": 1.0, "particleDiffusivity": 0.5 } },
          "sources": { "fusionPower": false, "ohmicHeating": true,
                        "bremsstrahlung": true, "ionElectronExchange": true }
        }
      },
      "time": { "start": 0.0, "end": 5e-5, "initialTimeStep": 1e-5 },
      "output": { "saveInterval": null, "directory": "/tmp/gotenx_e2e", "format": "json" }
    }
    """

    // Captured from this test configuration after nested transport parameters were
    // normalized by GotenxConfigReader on 2026-06-01.
    // Keep this fixture independent from the current run output.
    private static let referenceIonTemperature: [Float] = [
        499.98096, 499.69922, 498.69473, 497.03314, 494.71198, 491.73782, 488.11792, 483.8615, 478.9794, 473.48413, 467.38977, 460.71228, 453.469, 445.67917, 437.36353, 428.54468, 419.2467, 409.4954, 399.31827, 388.74454, 377.8049, 366.53183, 354.95966, 343.12393, 331.06235, 318.81396, 306.41962, 293.92175, 281.3644, 268.7937, 256.25677, 243.803, 231.48314, 219.3496, 207.45648, 195.85976, 184.61678, 173.78676, 163.43042, 153.6103, 144.39043, 135.83667, 128.01645, 120.9988, 114.85454, 109.65587, 105.47685, 102.39402, 100.47248, 99.901825
    ]

    private static let referenceElectronTemperature: [Float] = [
        499.98096, 499.6992, 498.6947, 497.0331, 494.71198, 491.7378, 488.1179, 483.86148, 478.97937, 473.48413, 467.38974, 460.71225, 453.469, 445.67917, 437.3635, 428.54468, 419.24667, 409.49536, 399.31824, 388.7445, 377.8049, 366.53183, 354.95963, 343.12393, 331.06232, 318.81396, 306.41962, 293.92175, 281.3644, 268.79367, 256.25677, 243.80298, 231.48314, 219.3496, 207.45648, 195.85976, 184.61678, 173.78674, 163.43042, 153.6103, 144.39043, 135.83665, 128.01645, 120.9988, 114.85454, 109.65586, 105.47685, 102.39402, 100.47248, 99.901825
    ]

    private static let referenceElectronDensity: [Float] = [
        2.000037e+19, 1.9994555e+19, 1.9975795e+19, 1.9944604e+19, 1.990098e+19, 1.9844993e+19, 1.9776702e+19, 1.9696203e+19, 1.9603593e+19, 1.9498994e+19, 1.9382545e+19, 1.925439e+19, 1.9114715e+19, 1.8963686e+19, 1.8801534e+19, 1.8628467e+19, 1.844474e+19, 1.8250608e+19, 1.8046361e+19, 1.7832308e+19, 1.7608779e+19, 1.7376121e+19, 1.7134716e+19, 1.6884965e+19, 1.66273e+19, 1.6362179e+19, 1.6090095e+19, 1.5811573e+19, 1.5527168e+19, 1.5237485e+19, 1.4943158e+19, 1.4644878e+19, 1.4343379e+19, 1.4039458e+19, 1.3733966e+19, 1.3427834e+19, 1.3122068e+19, 1.2817769e+19, 1.251615e+19, 1.2218546e+19, 1.1926457e+19, 1.1641571e+19, 1.1365823e+19, 1.1101471e+19, 1.0851209e+19, 1.0618359e+19, 1.0407223e+19, 1.0223815e+19, 1.0077207e+19, 9.994543e+18
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
