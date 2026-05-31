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

    // Captured from this test configuration after the solver robustness fixes on 2026-05-31.
    // Keep this fixture independent from the current run output.
    private static let referenceIonTemperature: [Float] = [
        310.05658, 1008.45624, 1, 1068.2285, 70.91734, 788.4127, 285.63812, 636.6705, 357.0921, 582.4845, 370.0659, 554.078, 366.60342, 532.05334, 353.75394, 514.69556, 332.637, 500.67984, 305.66217, 487.70676, 275.3359, 474.4123, 242.7652, 460.63345, 208.72473, 445.8161, 174.7362, 428.80606, 142.70897, 408.58734, 113.69109, 385.94482, 86.51644, 363.80814, 58.320766, 346.78995, 24.282736, 342.28152, 1, 358.48618, 1, 415.7596, 1, 589.85205, 1, 902.5817, 1, 1184.7896, 1, 886.544
    ]

    private static let referenceElectronTemperature: [Float] = [
        310.05658, 1008.4561, 1, 1068.2284, 70.91734, 788.4127, 285.63812, 636.6704, 357.09207, 582.4845, 370.06586, 554.07794, 366.60342, 532.0533, 353.7539, 514.69556, 332.637, 500.67978, 305.66217, 487.70676, 275.33588, 474.41226, 242.7652, 460.63342, 208.72472, 445.81607, 174.7362, 428.80606, 142.70897, 408.5873, 113.69109, 385.94482, 86.51644, 363.80814, 58.320766, 346.78992, 24.282736, 342.28152, 1, 358.48615, 1, 415.7596, 1, 589.852, 1, 902.58167, 1, 1184.7894, 1, 886.54395
    ]

    private static let referenceElectronDensity: [Float] = [
        2.0001832E+19, 2.000072E+19, 1.9980477E+19, 1.9949557E+19, 1.9905862E+19, 1.9849868E+19, 1.9781527E+19, 1.9700992E+19, 1.9608334E+19, 1.950368E+19, 1.9387165E+19, 1.9258958E+19, 1.9119205E+19, 1.8968084E+19, 1.880585E+19, 1.8632691E+19, 1.8448862E+19, 1.8254602E+19, 1.8050245E+19, 1.7836057E+19, 1.7612392E+19, 1.7379578E+19, 1.7138027E+19, 1.6888096E+19, 1.6630251E+19, 1.6364951E+19, 1.6092635E+19, 1.5813917E+19, 1.5529259E+19, 1.5239327E+19, 1.4944725E+19, 1.464615E+19, 1.4344345E+19, 1.4040083E+19, 1.373423E+19, 1.3427687E+19, 1.3121502E+19, 1.2816721E+19, 1.2514578E+19, 1.2216378E+19, 1.1923622E+19, 1.1637985E+19, 1.1361346E+19, 1.1095951E+19, 1.0844394E+19, 1.060994E+19, 1.0396264E+19, 1.0210897E+19, 1.0055016E+19, 9.938086E+18
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
