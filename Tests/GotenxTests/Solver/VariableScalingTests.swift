import Testing
import MLX
import Foundation
@testable import GotenxCore

/// Tests for GPU-based variable scaling in FlattenedState
@Suite("Variable Scaling Tests")
struct VariableScalingTests {

    // MARK: - Round-Trip Tests

    @Test("Round-trip: scaled then unscaled returns original")
    func testRoundTrip() throws {
        let nCells = 25

        // Create test profiles with realistic ITER-like values
        let Ti = MLXArray(Array(repeating: Float(10000.0), count: nCells))  // 10 keV
        let Te = MLXArray(Array(repeating: Float(10000.0), count: nCells))  // 10 keV
        let ne = MLXArray(Array(repeating: Float(1e20), count: nCells))     // 10^20 m^-3
        let psi = MLXArray(Array(repeating: Float(0.0), count: nCells))     // Dummy

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let state = try FlattenedState(profiles: profiles)
        let reference = state.asScalingReference(minScale: 1e-10)

        // Scale then unscale
        let scaled = state.scaled(by: reference)
        let unscaled = scaled.unscaled(by: reference)

        // Verify round-trip accuracy
        eval(state.values.value, unscaled.values.value)
        let original = state.values.value.asArray(Float.self)
        let recovered = unscaled.values.value.asArray(Float.self)

        for i in 0..<original.count {
            let relativeError = abs(recovered[i] - original[i]) / (abs(original[i]) + 1e-10)
            #expect(relativeError < 1e-6, "Round-trip error at index \(i): \(relativeError)")
        }
    }

    @Test("Scaled values are O(1)")
    func testScaledMagnitude() throws {
        let nCells = 25

        // Create profiles with vastly different magnitudes
        let Ti = MLXArray(Array(repeating: Float(15000.0), count: nCells))  // 15 keV
        let Te = MLXArray(Array(repeating: Float(12000.0), count: nCells))  // 12 keV
        let ne = MLXArray(Array(repeating: Float(1.2e20), count: nCells))   // 1.2×10^20 m^-3
        let psi = MLXArray(Array(repeating: Float(5.0), count: nCells))     // 5.0 Wb

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let state = try FlattenedState(profiles: profiles)
        let reference = state.asScalingReference(minScale: 1e-10)
        let scaled = state.scaled(by: reference)

        // Verify scaled values are O(1)
        eval(scaled.values.value)
        let scaledArray = scaled.values.value.asArray(Float.self)

        for (i, value) in scaledArray.enumerated() {
            #expect(abs(value) >= 0.1 && abs(value) <= 10.0,
                   "Scaled value at \(i) not O(1): \(value)")
        }
    }

    @Test("Scaling reference uses absolute values")
    func testScalingReferenceAbsolute() throws {
        let nCells = 10

        // Create profiles with negative values (e.g., poloidal flux can be negative)
        let Ti = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let Te = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let ne = MLXArray(Array(repeating: Float(1e20), count: nCells))
        let psi = MLXArray(Array(repeating: Float(-5.0), count: nCells))  // Negative!

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let state = try FlattenedState(profiles: profiles)
        let reference = state.asScalingReference(minScale: 1e-10)

        // Reference should use absolute values (all positive)
        eval(reference.values.value)
        let refArray = reference.values.value.asArray(Float.self)

        for (i, value) in refArray.enumerated() {
            #expect(value > 0.0, "Reference at \(i) should be positive: \(value)")
        }
    }

    @Test("Minimum scaling floor prevents division by zero")
    func testMinimumScalingFloor() throws {
        let nCells = 10

        // Create profiles with very small values
        let Ti = MLXArray(Array(repeating: Float(1e-15), count: nCells))  // Very small
        let Te = MLXArray(Array(repeating: Float(1e-15), count: nCells))
        let ne = MLXArray(Array(repeating: Float(1e-15), count: nCells))
        let psi = MLXArray(Array(repeating: Float(1e-15), count: nCells))

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let state = try FlattenedState(profiles: profiles)
        let minScale: Float = 1e-10
        let reference = state.asScalingReference(minScale: minScale)

        // Reference should be at least minScale
        eval(reference.values.value)
        let refArray = reference.values.value.asArray(Float.self)

        for (i, value) in refArray.enumerated() {
            #expect(value >= minScale, "Reference at \(i) below minScale: \(value)")
        }
    }

    // MARK: - Gradient Profile Tests

    @Test("Scaling preserves profile shape")
    func testScalingPreservesShape() throws {
        let nCells = 25

        // Create profiles with gradients (linear decay)
        var Ti_values: [Float] = []
        var Te_values: [Float] = []
        var ne_values: [Float] = []

        for i in 0..<nCells {
            let rho = Float(i) / Float(nCells - 1)  // Normalized radius 0 → 1
            Ti_values.append(15000.0 * (1.0 - 0.9 * rho))  // 15 keV → 1.5 keV
            Te_values.append(15000.0 * (1.0 - 0.9 * rho))  // 15 keV → 1.5 keV
            ne_values.append(1.2e20 * (1.0 - 0.8 * rho))   // 1.2e20 → 0.24e20
        }

        let Ti = MLXArray(Ti_values, [nCells])
        let Te = MLXArray(Te_values, [nCells])
        let ne = MLXArray(ne_values, [nCells])
        let psi = MLXArray(Array(repeating: Float(0.0), count: nCells))

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let state = try FlattenedState(profiles: profiles)
        let reference = state.asScalingReference(minScale: 1e-10)
        let scaled = state.scaled(by: reference)

        // Unscale back
        let unscaled = scaled.unscaled(by: reference)
        let recovered = unscaled.toCoreProfiles()

        // Verify Ti profile shape is preserved
        eval(recovered.ionTemperature.value)
        let originalTi = Ti.asArray(Float.self)
        let recoveredTi = recovered.ionTemperature.value.asArray(Float.self)

        for i in 0..<nCells {
            let relativeError = abs(recoveredTi[i] - originalTi[i]) / (originalTi[i] + 1e-10)
            #expect(relativeError < 1e-5, "Ti profile error at \(i): \(relativeError)")
        }
    }

    // MARK: - Jacobian Consistency Tests

    @Test("VJP Jacobian preserves non-symmetric orientation")
    func testJacobianOrientationForNonSymmetricResidual() throws {
        let coefficients = MLXArray(
            [
                Float(1.0), Float(2.0), Float(3.0),
                Float(0.5), Float(-1.0), Float(4.0),
                Float(7.0), Float(0.0), Float(2.0)
            ],
            [3, 3]
        )
        let x = MLXArray([Float(0.25), Float(-1.0), Float(2.0)])

        let residualFn: (MLXArray) -> MLXArray = { input in
            matmul(coefficients, reshaped(input, [3, 1]))[0..., 0]
        }

        let jacobian = computeJacobianViaVJP(residualFn, x)
        eval(jacobian)

        let expected: [[Float]] = [
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 4.0],
            [7.0, 0.0, 2.0]
        ]

        for row in 0..<3 {
            for column in 0..<3 {
                let actual = jacobian[row, column].item(Float.self)
                #expect(
                    abs(actual - expected[row][column]) < 1e-5,
                    "Jacobian[\(row), \(column)] = \(actual), expected \(expected[row][column])"
                )
            }
        }
    }

    @Test("Non-symmetric Newton directions use merit descent")
    func testNonSymmetricNewtonDirectionUsesMeritDescent() throws {
        let jacobian = MLXArray(
            [
                Float(1.0), Float(10.0),
                Float(0.0), Float(1.0)
            ],
            [2, 2]
        )
        let residual = MLXArray([Float(1.0), Float(1.0)])

        let delta = MLX.solve(jacobian, -residual, stream: .cpu)
        let alignment = (delta * (-residual)).sum().item(Float.self)
        let meritDescent = -(residual * jacobian.matmul(delta)).sum().item(Float.self)

        #expect(alignment < 0.0, "Legacy alignment check should reject this valid direction")
        #expect(meritDescent > 0.0, "Merit descent should accept the Newton direction")
        #expect(abs(meritDescent - 2.0) < 1e-5, "Merit descent should equal ||R||^2")
    }

    @Test("Scaling does not change Jacobian structure")
    func testJacobianConsistency() throws {
        let nCells = 5  // Small for faster test

        // Create simple test profiles
        let Ti = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let Te = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let ne = MLXArray(Array(repeating: Float(1e20), count: nCells))
        let psi = MLXArray(Array(repeating: Float(0.0), count: nCells))

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let state = try FlattenedState(profiles: profiles)
        let layout = state.layout
        let reference = state.asScalingReference(minScale: 1e-10)

        // Simple residual function: R(x) = x - x0
        let residualFnPhysical: (MLXArray) -> MLXArray = { x in
            return x - state.values.value
        }

        // Scaled residual function
        let residualFnScaled: (MLXArray) -> MLXArray = { xScaled in
            let xScaledState = FlattenedState(values: EvaluatedArray(evaluating: xScaled), layout: layout)
            let xPhysical = xScaledState.unscaled(by: reference)
            let residualPhysical = residualFnPhysical(xPhysical.values.value)
            let residualState = FlattenedState(values: EvaluatedArray(evaluating: residualPhysical), layout: layout)
            let residualScaled = residualState.scaled(by: reference)
            return residualScaled.values.value
        }

        // Compute Jacobians (should be identity matrix for R(x) = x - x0)
        let jacobianPhysical = computeJacobianViaVJP(residualFnPhysical, state.values.value)

        let scaled = state.scaled(by: reference)
        let jacobianScaled = computeJacobianViaVJP(residualFnScaled, scaled.values.value)

        eval(jacobianPhysical, jacobianScaled)

        // For R(x) = x - x0, Jacobian should be identity
        // Verify diagonal elements are ~1
        for i in 0..<nCells * 4 {
            let physDiag = jacobianPhysical[i, i].item(Float.self)
            let scaledDiag = jacobianScaled[i, i].item(Float.self)

            #expect(abs(physDiag - 1.0) < 0.1, "Physical Jacobian diagonal[\(i)] not ~1: \(physDiag)")
            #expect(abs(scaledDiag - 1.0) < 0.1, "Scaled Jacobian diagonal[\(i)] not ~1: \(scaledDiag)")
        }
    }

    // MARK: - Performance Tests

    @Test("Scaling operations are GPU-only (no CPU transfer)")
    func testGPUOnly() throws {
        let nCells = 100

        // Create large profiles
        let Ti = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let Te = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let ne = MLXArray(Array(repeating: Float(1e20), count: nCells))
        let psi = MLXArray(Array(repeating: Float(0.0), count: nCells))

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let state = try FlattenedState(profiles: profiles)
        let reference = state.asScalingReference(minScale: 1e-10)

        // Warm-up run to initialize GPU/Metal (avoid JIT compilation overhead)
        let warmup_state = try FlattenedState(profiles: profiles)
        let warmup_ref = warmup_state.asScalingReference(minScale: 1e-10)
        let warmup_scaled = warmup_state.scaled(by: warmup_ref)
        let warmup_unscaled = warmup_scaled.unscaled(by: warmup_ref)
        eval(warmup_unscaled.values.value)

        // Measure scaling performance (should be very fast ~100μs after warm-up)
        let start = Date()
        let scaled = state.scaled(by: reference)
        let unscaled = scaled.unscaled(by: reference)
        eval(unscaled.values.value)
        let elapsed = Date().timeIntervalSince(start)

        // Scaling + unscaling should be < 20ms (includes system variability)
        // Note: First run may be slower due to GPU initialization (10-50ms)
        // After warm-up, typical time is 0.5-2ms
        #expect(elapsed < 0.02, "Scaling too slow: \(elapsed * 1000)ms")

        print("[VariableScalingTests] GPU scaling + unscaling: \(String(format: "%.3f", elapsed * 1000))ms")
    }

    // MARK: - Edge Cases

    @Test("Scaling with zero values")
    func testScalingWithZeros() throws {
        let nCells = 10

        // Create profiles with some zero values
        let Ti = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let Te = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let ne = MLXArray(Array(repeating: Float(1e20), count: nCells))
        let psi = MLXArray(Array(repeating: Float(0.0), count: nCells))  // All zeros

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let state = try FlattenedState(profiles: profiles)
        let reference = state.asScalingReference(minScale: 1e-10)

        // Should not crash with zeros
        let scaled = state.scaled(by: reference)
        let unscaled = scaled.unscaled(by: reference)

        eval(unscaled.values.value)
        let recovered = unscaled.values.value.asArray(Float.self)

        // Verify all values are finite
        for (i, value) in recovered.enumerated() {
            #expect(value.isFinite, "Non-finite value at \(i): \(value)")
        }
    }

    @Test("Scaling with mixed positive/negative values")
    func testScalingMixedSigns() throws {
        let nCells = 10

        // Create profiles with mixed signs
        var psi_values: [Float] = []
        for i in 0..<nCells {
            psi_values.append(Float(i - nCells/2) * 0.5)  // -2.5 to +2.0
        }

        let Ti = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let Te = MLXArray(Array(repeating: Float(10000.0), count: nCells))
        let ne = MLXArray(Array(repeating: Float(1e20), count: nCells))
        let psi = MLXArray(psi_values, [nCells])

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let state = try FlattenedState(profiles: profiles)
        let reference = state.asScalingReference(minScale: 1e-10)
        let scaled = state.scaled(by: reference)
        let unscaled = scaled.unscaled(by: reference)

        // Verify round-trip preserves signs
        eval(psi, unscaled.values.value)
        let original = psi.asArray(Float.self)
        let layout = state.layout
        let recovered = unscaled.values.value[layout.psiRange].asArray(Float.self)

        for i in 0..<nCells {
            let relativeError = abs(recovered[i] - original[i]) / (abs(original[i]) + 1e-10)
            #expect(relativeError < 1e-5, "Psi round-trip error at \(i): \(relativeError)")

            // Verify sign is preserved
            if abs(original[i]) > 1e-10 {
                #expect(recovered[i] * original[i] > 0, "Sign flipped at \(i)")
            }
        }
    }
}
