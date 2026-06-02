import Testing
import MLX
@testable import GotenxCore

@Suite("NumericalValidation Tests")
struct NumericalValidationTests {
    @Test("CoreProfiles validation rejects shape mismatch")
    func coreProfilesRejectShapeMismatch() {
        let profiles = CoreProfiles(
            ionTemperature: .full([8], value: 1_000),
            electronTemperature: .full([7], value: 1_000),
            electronDensity: .full([8], value: 1e19),
            poloidalFlux: .full([8], value: 0)
        )

        #expect(throws: NumericalValidationError.self) {
            try profiles.validateNumerics()
        }
    }

    @Test("CoreProfiles validation rejects non-positive evolved variables")
    func coreProfilesRejectNonPositiveVariables() {
        let profiles = CoreProfiles(
            ionTemperature: .full([8], value: 1_000),
            electronTemperature: .full([8], value: 0),
            electronDensity: .full([8], value: 1e19),
            poloidalFlux: .full([8], value: 0)
        )

        #expect(throws: NumericalValidationError.self) {
            try profiles.validateNumerics()
        }
    }

    @Test("SourceTerms validation requires metadata on diagnostic path")
    func sourceTermsRejectMissingDiagnosticMetadata() {
        let sources = SourceTerms.zero(cellCount: 8, metadata: nil)

        #expect(throws: NumericalValidationError.self) {
            try sources.validateNumerics(expectedCellCount: 8, requiresMetadata: true)
        }
    }

    @Test("SourceTerms validation rejects non-finite source arrays")
    func sourceTermsRejectNonFiniteValues() {
        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.full([8], values: MLXArray(Float.nan))),
            electronHeating: .zeros([8]),
            particleSource: .zeros([8]),
            currentSource: .zeros([8]),
            metadata: SourceMetadataCollection.empty,
            validateDebugUnits: false
        )

        #expect(throws: NumericalValidationError.self) {
            try sources.validateNumerics(expectedCellCount: 8, requiresMetadata: true)
        }
    }

    @Test("TransportCoefficients validation rejects negative diffusivity")
    func transportRejectsNegativeDiffusivity() {
        let transport = TransportCoefficients(
            ionHeatDiffusivity: .full([8], value: -1),
            electronHeatDiffusivity: .full([8], value: 1),
            particleDiffusivity: .full([8], value: 1),
            convectionVelocity: .full([8], value: 0)
        )

        #expect(throws: NumericalValidationError.self) {
            try transport.validateNumerics(expectedCellCount: 8)
        }
    }
}
