import Testing
import MLX
@testable import GotenxCore

/// Unit tests for ValidatedProfiles validation.
///
/// Tests cover:
/// 1. Valid profiles (should pass)
/// 2. NaN detection (should fail)
/// 3. Inf detection (should fail)
/// 4. Negative temperature detection (should fail)
/// 5. Zero temperature detection (should fail)
/// 6. Zero density detection (should fail)
@Suite("ValidatedProfiles Tests")
struct ValidatedProfilesTests {

    // MARK: - Test Helpers

    /// Create valid test profiles
    func createValidProfiles(cellCount: Int = 100) -> CoreProfiles {
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(2e19)))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )
    }

    // MARK: - Valid Profiles Tests

    @Test("ValidatedProfiles accepts valid profiles")
    func testValidProfiles() throws {
        let profiles = createValidProfiles()

        let validated = try ValidatedProfiles.validate(profiles)

        // Check values are preserved (with Float32 tolerance)
        let Ti_mean = validated.ionTemperature.value.mean().item(Float.self)
        let Te_mean = validated.electronTemperature.value.mean().item(Float.self)
        let ne_mean = validated.electronDensity.value.mean().item(Float.self)

        #expect(abs(Ti_mean - 1000.0) / 1000.0 < 1e-5)  // Relative error < 0.001%
        #expect(abs(Te_mean - 1000.0) / 1000.0 < 1e-5)
        #expect(abs(ne_mean - 2e19) / 2e19 < 1e-5)
    }

    @Test("ValidatedProfiles converts back to CoreProfiles")
    func testToCoreProfiles() throws {
        let profiles = createValidProfiles()
        let validated = try ValidatedProfiles.validate(profiles)

        let converted = validated.toCoreProfiles()

        // Check with Float32 tolerance
        let Ti_mean = converted.ionTemperature.value.mean().item(Float.self)
        let Te_mean = converted.electronTemperature.value.mean().item(Float.self)

        #expect(abs(Ti_mean - 1000.0) / 1000.0 < 1e-5)
        #expect(abs(Te_mean - 1000.0) / 1000.0 < 1e-5)
    }

    // MARK: - NaN Detection Tests

    @Test("ValidatedProfiles rejects NaN in ionTemperature")
    func testRejectsNaNIonTemperature() {
        let cellCount = 100
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float.nan))  // ❌ NaN
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(2e19)))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        #expect(throws: NumericalValidationError.self) {
            try ValidatedProfiles.validate(profiles)
        }
    }

    @Test("ValidatedProfiles rejects NaN in electronTemperature")
    func testRejectsNaNElectronTemperature() {
        let cellCount = 100
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let Te = MLXArray.full([cellCount], values: MLXArray(Float.nan))  // ❌ NaN
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(2e19)))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        #expect(throws: NumericalValidationError.self) {
            try ValidatedProfiles.validate(profiles)
        }
    }

    @Test("ValidatedProfiles rejects NaN in electronDensity")
    func testRejectsNaNElectronDensity() {
        let cellCount = 100
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let ne = MLXArray.full([cellCount], values: MLXArray(Float.nan))  // ❌ NaN
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        #expect(throws: NumericalValidationError.self) {
            try ValidatedProfiles.validate(profiles)
        }
    }

    // MARK: - Inf Detection Tests

    @Test("ValidatedProfiles rejects Inf in ionTemperature")
    func testRejectsInfIonTemperature() {
        let cellCount = 100
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float.infinity))  // ❌ Inf
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(2e19)))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        #expect(throws: NumericalValidationError.self) {
            try ValidatedProfiles.validate(profiles)
        }
    }

    // MARK: - Negative/Zero Temperature Tests

    @Test("ValidatedProfiles rejects zero ionTemperature")
    func testRejectsZeroIonTemperature() {
        let cellCount = 100
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float(0.0)))  // ❌ Zero
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(2e19)))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        #expect(throws: NumericalValidationError.self) {
            try ValidatedProfiles.validate(profiles)
        }
    }

    @Test("ValidatedProfiles rejects negative electronTemperature")
    func testRejectsNegativeElectronTemperature() {
        let cellCount = 100
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(-100.0)))  // ❌ Negative
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(2e19)))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        #expect(throws: NumericalValidationError.self) {
            try ValidatedProfiles.validate(profiles)
        }
    }

    @Test("ValidatedProfiles rejects zero electronDensity")
    func testRejectsZeroElectronDensity() {
        let cellCount = 100
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(0.0)))  // ❌ Zero
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        #expect(throws: NumericalValidationError.self) {
            try ValidatedProfiles.validate(profiles)
        }
    }

    // MARK: - Edge Cases

    @Test("ValidatedProfiles accepts very small positive temperature")
    func testAcceptsSmallPositiveTemperature() throws {
        let cellCount = 100
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float(0.01)))  // ✅ Very small but positive
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(0.01)))
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(1e17)))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let validated = try ValidatedProfiles.validate(profiles)
        #expect(validated.ionTemperature.shape == [cellCount])
    }

    @Test("ValidatedProfiles accepts very large temperature")
    func testAcceptsLargeTemperature() throws {
        let cellCount = 100
        let Ti = MLXArray.full([cellCount], values: MLXArray(Float(1e6)))  // ✅ Very large but finite
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(1e6)))
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(2e19)))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let validated = try ValidatedProfiles.validate(profiles)
        #expect(validated.ionTemperature.shape == [cellCount])
    }

    @Test("ValidatedProfiles handles mixed valid/invalid cells")
    func testRejectsMixedValidInvalid() {
        let cellCount = 100
        var Ti_array = [Float](repeating: 1000.0, count: cellCount)
        Ti_array[50] = Float.nan  // One NaN cell

        let Ti = MLXArray(Ti_array)
        let Te = MLXArray.full([cellCount], values: MLXArray(Float(1000.0)))
        let ne = MLXArray.full([cellCount], values: MLXArray(Float(2e19)))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        #expect(throws: NumericalValidationError.self) {
            try ValidatedProfiles.validate(profiles)
        }
    }
}
