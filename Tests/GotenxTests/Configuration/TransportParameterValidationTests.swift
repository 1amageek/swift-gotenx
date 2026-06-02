import Foundation
import Testing
@testable import GotenxCore

@Suite("Transport Parameter Validation")
struct TransportParameterValidationTests {

    @Test("Default constant transport configuration is valid")
    func defaultConstantTransportConfigurationIsValid() throws {
        try TransportConfig.defaultConstant.validate()
    }

    @Test("TransportConfig decoding rejects obsolete parameter keys")
    func decodingRejectsObsoleteParameterKeys() throws {
        let json = """
        {
          "modelType": "bohmGyrobohm",
          "parameters": {
            "obsoleteParameter": 0.5
          }
        }
        """

        #expect(throws: ConfigurationValidationError.self) {
            _ = try JSONDecoder().decode(TransportConfig.self, from: Data(json.utf8))
        }
    }

    @Test("TransportConfig initializer rejects unknown transport parameters")
    func transportConfigInitializerRejectsUnknownTransportParameters() {
        #expect(throws: ConfigurationValidationError.self) {
            _ = try TransportConfig(
                modelType: .bohmGyrobohm,
                parameters: [
                    "bohmCoefficient": 0.5,
                    "unusedCoefficient": 2.0
                ]
            )
        }
    }

    @Test("TransportConfig initializer rejects missing constant transport parameters")
    func transportConfigInitializerRejectsMissingConstantTransportParameters() {
        #expect(throws: ConfigurationValidationError.self) {
            _ = try TransportConfig(modelType: .constant)
        }
    }

    @Test("TransportConfig initializer rejects parameters for another transport model")
    func transportConfigInitializerRejectsParametersForAnotherTransportModel() {
        #expect(throws: ConfigurationValidationError.self) {
            _ = try TransportConfig(
                modelType: .densityTransition,
                parameters: [
                    "transitionDensity": 2.5e19,
                    "transitionWidth": 0.5e19,
                    "bohmCoefficient": 0.5
                ]
            )
        }
    }

    @Test("TransportParameters initializer rejects obsolete transport parameters")
    func transportParametersInitializerRejectsObsoleteTransportParameters() {
        #expect(throws: ConfigurationValidationError.self) {
            _ = try TransportParameters(
                modelType: .constant,
                parameters: [
                    "obsoleteParameter": 1.0,
                    "electronHeatDiffusivity": 1.0
                ]
            )
        }
    }

    @Test("TransportParameters initializer rejects non-finite transport parameters")
    func transportParametersInitializerRejectsNonFiniteTransportParameters() {
        #expect(throws: ConfigurationValidationError.self) {
            _ = try TransportParameters(
                modelType: .constant,
                parameters: [
                    "ionHeatDiffusivity": .infinity,
                    "electronHeatDiffusivity": 1.0
                ]
            )
        }
    }

    @Test("Direct parameter initializer rejects parameters for a different model")
    func directInitializerRejectsParametersForDifferentModel() throws {
        let parameters = try TransportParameters(
            modelType: .bohmGyrobohm,
            parameters: [:]
        )

        #expect(throws: ConfigurationValidationError.self) {
            _ = try ConstantTransportModel(parameters: parameters)
        }
    }

    @Test("Constant transport model initializer rejects missing required coefficients")
    func constantInitializerRejectsMissingRequiredCoefficients() throws {
        let parameters = try TransportParameters(
            modelType: .constant,
            parameters: [
                "ionHeatDiffusivity": 1.0
            ]
        )

        #expect(throws: ConfigurationValidationError.self) {
            _ = try ConstantTransportModel(parameters: parameters)
        }
    }
}
