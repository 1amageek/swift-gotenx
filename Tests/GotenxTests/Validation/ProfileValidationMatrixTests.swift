import Testing
@testable import GotenxCore

@Suite("Profile Validation Matrix Tests")
struct ProfileValidationMatrixTests {
    @Test("ITER baseline self-comparison passes all profile channels")
    func iterBaselineSelfComparisonPasses() throws {
        let baseline = ITERBaselineData.load()

        let matrix = try ProfileValidationMatrix.compare(
            predicted: baseline.profiles,
            reference: baseline.profiles,
            sourceName: "iter-baseline"
        )

        #expect(matrix.passed)
        #expect(matrix.results.count == 3)
        #expect(matrix.failedResults.isEmpty)
        #expect(matrix.maximumL2Error < 1e-6)
        #expect(matrix.failureSummary.contains("all 3 profile comparisons passed"))
    }

    @Test("Degraded ion temperature is reported as a failed profile channel")
    func degradedProfileFailsWithSummary() throws {
        let baseline = ITERBaselineData.load()
        let degraded = ReferenceProfiles(
            normalizedRadius: baseline.profiles.normalizedRadius,
            ionTemperature: baseline.profiles.ionTemperature.map { $0 * 1.5 },
            electronTemperature: baseline.profiles.electronTemperature,
            electronDensity: baseline.profiles.electronDensity,
            time: baseline.profiles.time
        )

        let matrix = try ProfileValidationMatrix.compare(
            predicted: degraded,
            reference: baseline.profiles,
            sourceName: "degraded"
        )

        #expect(!matrix.passed)
        #expect(matrix.failedResults.count == 1)
        #expect(matrix.failedResults[0].quantity == "ion_temperature")
        #expect(matrix.failureSummary.contains("ion_temperature"))
    }

    @Test("Profile matrix rejects inconsistent profile lengths")
    func rejectsInconsistentProfileLengths() throws {
        let baseline = ITERBaselineData.load()
        let invalid = ReferenceProfiles(
            normalizedRadius: baseline.profiles.normalizedRadius,
            ionTemperature: Array(baseline.profiles.ionTemperature.dropLast()),
            electronTemperature: baseline.profiles.electronTemperature,
            electronDensity: baseline.profiles.electronDensity,
            time: baseline.profiles.time
        )

        #expect(throws: ProfileValidationMatrixError.self) {
            try ProfileValidationMatrix.compare(
                predicted: invalid,
                reference: baseline.profiles
            )
        }
    }

    @Test("Profile matrix rejects shifted radius grids")
    func rejectsShiftedRadiusGrid() throws {
        let baseline = ITERBaselineData.load()
        var shiftedRadius = baseline.profiles.normalizedRadius
        shiftedRadius[shiftedRadius.count / 2] += 1e-3

        let shifted = ReferenceProfiles(
            normalizedRadius: shiftedRadius,
            ionTemperature: baseline.profiles.ionTemperature,
            electronTemperature: baseline.profiles.electronTemperature,
            electronDensity: baseline.profiles.electronDensity,
            time: baseline.profiles.time
        )

        #expect(throws: ProfileValidationMatrixError.self) {
            try ProfileValidationMatrix.compare(
                predicted: shifted,
                reference: baseline.profiles
            )
        }
    }

    @Test("TORAX time-series self-comparison returns one result per quantity and time")
    func toraxTimeSeriesSelfComparisonPasses() throws {
        let baseline = ITERBaselineData.load()
        let reference = TORAXReferenceData(
            time: [0.0, baseline.profiles.time],
            normalizedRadius: baseline.profiles.normalizedRadius,
            ionTemperature: [
                baseline.profiles.ionTemperature,
                baseline.profiles.ionTemperature
            ],
            electronTemperature: [
                baseline.profiles.electronTemperature,
                baseline.profiles.electronTemperature
            ],
            electronDensity: [
                baseline.profiles.electronDensity,
                baseline.profiles.electronDensity
            ]
        )

        let matrix = try ProfileValidationMatrix.compareTimeSeries(
            predicted: reference,
            reference: reference
        )

        #expect(matrix.passed)
        #expect(matrix.results.count == 6)
        #expect(matrix.failedResults.isEmpty)
    }

    @Test("ValidationConfigMatcher exposes aggregate TORAX comparison matrix")
    func validationConfigMatcherReturnsMatrix() throws {
        let baseline = ITERBaselineData.load()
        let reference = TORAXReferenceData(
            time: [baseline.profiles.time],
            normalizedRadius: baseline.profiles.normalizedRadius,
            ionTemperature: [baseline.profiles.ionTemperature],
            electronTemperature: [baseline.profiles.electronTemperature],
            electronDensity: [baseline.profiles.electronDensity]
        )

        let matrix = try ValidationConfigMatcher.compareWithToraxMatrix(
            gotenx: reference,
            torax: reference
        )

        #expect(matrix.passed)
        #expect(matrix.results.count == 3)
    }
}
