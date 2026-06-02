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
        try config.validateParameterKeys()
        let parameters = config.transportParameters()

        // config.modelType is already TransportModelType enum
        switch config.modelType {
        case .constant:
            return try ConstantTransportModel(parameters: parameters)

        case .bohmGyrobohm:
            return try BohmGyroBohmTransportModel(parameters: parameters)

        case .qlknn:
            return try QLKNNTransportModel(parameters: parameters)

        case .densityTransition:
            // Extract parameters with defaults
            let riCoefficient = parameters.parameters["riCoefficient"] ?? 0.5
            let transitionDensity = parameters.parameters["transitionDensity"] ?? 2.5e19
            let transitionWidth = parameters.parameters["transitionWidth"] ?? 0.5e19
            let ionMassNumber = parameters.parameters["ionMassNumber"] ?? 2.0

            // Create ITG model (default: Bohm-GyroBohm)
            let itgModel = BohmGyroBohmTransportModel()

            // Create RI model
            let riModel = ResistiveInterchangeModel(
                riCoefficient: riCoefficient,
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
        switch modelType {
        case .constant:
            return ConstantTransportModel(
                ionHeatDiffusivity: 1.0,
                electronHeatDiffusivity: 1.0
            )

        case .bohmGyrobohm:
            return BohmGyroBohmTransportModel()

        case .qlknn:
            return try QLKNNTransportModel()

        case .densityTransition:
            return DensityTransitionModel.createDefault()
        }
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
