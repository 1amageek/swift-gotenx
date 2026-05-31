// TransportModelFactory.swift
// Factory for creating transport models from configuration

import Foundation

/// Factory for creating transport models from configuration
public struct TransportModelFactory {
    /// Create a transport model from configuration
    ///
    /// - Parameter config: Transport model configuration
    /// - Returns: Instantiated transport model
    /// - Throws: ConfigurationError if model type is invalid or not implemented
    public static func create(config: TransportConfig) throws -> any TransportModel {
        let params = config.toTransportParameters()

        // config.modelType is already TransportModelType enum
        switch config.modelType {
        case .constant:
            return ConstantTransportModel(params: params)

        case .bohmGyrobohm:
            return BohmGyroBohmTransportModel(params: params)

        case .qlknn:
            return try QLKNNTransportModel(params: params)

        case .densityTransition:
            // Extract parameters with defaults
            let riCoefficient = params.params["ri_coefficient"] ?? 0.5
            let transitionDensity = params.params["transition_density"] ?? 2.5e19
            let transitionWidth = params.params["transition_width"] ?? 0.5e19
            let ionMassNumber = params.params["ion_mass_number"] ?? 2.0

            // Create ITG model (default: Bohm-GyroBohm)
            let itgModel = BohmGyroBohmTransportModel()

            // Create RI model
            let riModel = ResistiveInterchangeModel(
                coefficientRI: riCoefficient,
                ionMassNumber: ionMassNumber
            )

            // Create density transition model
            return DensityTransitionModel(
                itgModel: itgModel,
                riModel: riModel,
                transitionDensity: transitionDensity,
                transitionWidth: transitionWidth,
                ionMassNumber: ionMassNumber
            )
        }
    }

    /// Create a transport model with default parameters
    ///
    /// - Parameter modelType: Transport model type
    /// - Returns: Instantiated transport model with default parameters
    /// - Throws: ConfigurationError if model type is not implemented
    public static func createDefault(_ modelType: TransportModelType) throws -> any TransportModel {
        let config = TransportConfig(modelType: modelType)
        return try create(config: config)
    }
}

// MARK: - Configuration Error Extensions

extension ConfigurationError {
    /// Feature not yet implemented
    static func notImplemented(feature: String) -> ConfigurationError {
        .invalidValue(
            key: "feature",
            value: feature,
            reason: "This feature is not yet implemented"
        )
    }
}
