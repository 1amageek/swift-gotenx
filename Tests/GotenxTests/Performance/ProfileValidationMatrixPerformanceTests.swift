import Testing
@testable import GotenxCore

@Suite("Profile Validation Matrix Performance")
struct ProfileValidationMatrixPerformanceTests {
    @Test("Reference matrix comparison stays within smoke budget", .timeLimit(.minutes(1)))
    func referenceMatrixComparisonSmokeBudget() throws {
        let baseline = ITERBaselineData.load()
        let iterationCount = 2_000
        let clock = ContinuousClock()

        let elapsed = try clock.measure {
            for _ in 0..<iterationCount {
                let matrix = try ProfileValidationMatrix.compare(
                    predicted: baseline.profiles,
                    reference: baseline.profiles,
                    sourceName: "performance-smoke"
                )
                #expect(matrix.passed)
            }
        }

        #expect(elapsed < .seconds(5))
    }
}
