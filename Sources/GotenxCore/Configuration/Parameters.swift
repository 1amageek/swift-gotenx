import Foundation

// MARK: - Transport Parameters

/// Transport model parameters
public struct TransportParameters: Sendable, Codable, Equatable {
    /// Transport model type (enum for type safety)
    public var modelType: TransportModelType

    /// Model-specific parameters
    public var parameters: [String: Float]

    public init(modelType: TransportModelType, parameters: [String: Float] = [:]) throws {
        self.modelType = modelType
        self.parameters = parameters
        try validateParameterKeys()
        try validateParameterValues()
    }

    enum CodingKeys: String, CodingKey {
        case modelType
        case parameters
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(TransportModelType.self, forKey: .modelType)
        self.parameters = try container.decodeIfPresent([String: Float].self, forKey: .parameters) ?? [:]
        try validateParameterKeys()
        try validateParameterValues()
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(parameters, forKey: .parameters)
    }

    /// Validate that every parameter key is supported by the selected model.
    public func validateParameterKeys() throws {
        try validateParameterKeys(for: modelType)
    }

    /// Validate that every parameter key is supported by the expected model.
    public func validateParameterKeys(for expectedModelType: TransportModelType) throws {
        guard modelType == expectedModelType else {
            throw ConfigurationValidationError.transportModelMismatch(
                expected: expectedModelType,
                actual: modelType
            )
        }

        try validateParameterKeys(
            allowedKeys: expectedModelType.allowedParameterKeys,
            errorModelType: expectedModelType
        )
    }

    /// Read a required model parameter.
    public func requireParameter(
        _ key: String,
        modelType expectedModelType: TransportModelType,
        suggestion: String
    ) throws -> Float {
        try validateParameterKeys(for: expectedModelType)

        guard let value = parameters[key] else {
            throw ConfigurationValidationError.missingRequiredParameter(
                parameter: key,
                modelType: expectedModelType,
                suggestion: suggestion
            )
        }

        return value
    }

    /// Validate against an explicit key set.
    public func validateParameterKeys(
        allowedKeys: Set<String>,
        errorModelType: TransportModelType
    ) throws {
        for key in parameters.keys where !allowedKeys.contains(key) {
            throw ConfigurationValidationError.unknownTransportParameter(
                parameter: key,
                modelType: errorModelType,
                allowed: allowedKeys.sorted()
            )
        }
    }

    /// Validate that every configured transport parameter is numerically usable.
    public func validateParameterValues() throws {
        for (key, value) in parameters where !value.isFinite {
            throw ConfigurationValidationError.invalidParameter(
                parameter: key,
                value: value,
                reason: "Transport parameter values must be finite"
            )
        }
    }
}

// MARK: - Source Parameters

/// Source model parameters
public struct SourceParameters: Sendable, Codable, Equatable {
    /// Source model type
    public var modelType: String

    /// Model-specific parameters
    public var parameters: [String: Float]

    /// Time-dependent scaling factor
    public var timeDependent: Bool

    public init(
        modelType: String,
        parameters: [String: Float] = [:],
        timeDependent: Bool = false
    ) {
        self.modelType = modelType
        self.parameters = parameters
        self.timeDependent = timeDependent
    }
}
