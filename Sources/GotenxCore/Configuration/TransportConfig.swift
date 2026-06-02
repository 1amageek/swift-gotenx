// TransportConfig.swift
// Transport model configuration

import Foundation

/// Transport model configuration
public struct TransportConfig: Codable, Sendable, Equatable {
    /// Transport model type (enum for type safety)
    public let modelType: TransportModelType

    /// Model-specific parameters
    public let parameters: [String: Float]

    public init(modelType: TransportModelType, parameters: [String: Float] = [:]) throws {
        self.modelType = modelType
        self.parameters = parameters
        try validate()
    }

    private init(uncheckedModelType modelType: TransportModelType, parameters: [String: Float]) {
        self.modelType = modelType
        self.parameters = parameters
    }

    // MARK: - Codable Support

    enum CodingKeys: String, CodingKey {
        case modelType
        case parameters
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(TransportModelType.self, forKey: .modelType)
        self.parameters = try container.decodeIfPresent([String: Float].self, forKey: .parameters) ?? [:]
        try validate()
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(parameters, forKey: .parameters)
    }

}

/// Transport model types
public enum TransportModelType: String, Codable, Sendable, CaseIterable {
    case constant
    case bohmGyrobohm
    case qlknn
    case densityTransition

    /// Parameter keys consumed by each transport model.
    public var allowedParameterKeys: Set<String> {
        switch self {
        case .constant:
            return [
                "ionHeatDiffusivity",
                "electronHeatDiffusivity",
                "particleDiffusivity",
                "convectionVelocity"
            ]
        case .bohmGyrobohm:
            return [
                "bohmCoefficient",
                "gyroBohmCoefficient",
                "ionMassNumber"
            ]
        case .qlknn:
            return [
                "effectiveCharge",
                "minimumHeatDiffusivity"
            ]
        case .densityTransition:
            return [
                "riCoefficient",
                "transitionDensity",
                "transitionWidth",
                "ionMassNumber"
            ]
        }
    }
}

// MARK: - Parameter Access

extension TransportConfig {
    /// Default constant transport configuration with explicit required coefficients.
    public static let defaultConstant = TransportConfig(
        uncheckedModelType: .constant,
        parameters: [
            "ionHeatDiffusivity": 1.0,
            "electronHeatDiffusivity": 1.0,
            "particleDiffusivity": 0.0,
            "convectionVelocity": 0.0
        ]
    )

    /// Validate model-specific parameter names and required parameters.
    public func validate() throws {
        try validateParameterKeys()
        try transportParameters().validateParameterValues()
        try validateRequiredParameters()
    }

    /// Validate that all parameter keys are consumed by the selected model.
    public func validateParameterKeys() throws {
        try transportParameters().validateParameterKeys()
    }

    /// Validate required parameters for models that cannot infer physical defaults.
    public func validateRequiredParameters() throws {
        guard modelType == .constant else {
            return
        }

        _ = try requireParameter(
            "ionHeatDiffusivity",
            suggestion: "Specify ionHeatDiffusivity in transport.parameters"
        )
        _ = try requireParameter(
            "electronHeatDiffusivity",
            suggestion: "Specify electronHeatDiffusivity in transport.parameters"
        )
    }

    /// Get parameter value (returns nil if missing)
    ///
    /// Use this when you need to handle missing values explicitly.
    ///
    /// - Parameter key: Parameter key (for example, `ionHeatDiffusivity`)
    /// - Returns: Parameter value or nil if not found
    ///
    /// Example:
    /// ```swift
    /// if let ionHeatDiffusivity = transport.parameter("ionHeatDiffusivity") {
    ///     print("ionHeatDiffusivity = \(ionHeatDiffusivity) m²/s")
    /// } else {
    ///     print("ionHeatDiffusivity not specified")
    /// }
    /// ```
    public func parameter(_ key: String) -> Float? {
        parameters[key]
    }

    /// Get required parameter (throws if missing)
    ///
    /// Use this for parameters that are mandatory for the model.
    ///
    /// - Parameter key: Parameter key (for example, `ionHeatDiffusivity`)
    /// - Returns: Parameter value
    /// - Throws: ConfigurationError.missingRequired if parameter not found
    ///
    /// Example:
    /// ```swift
    /// let ionHeatDiffusivity = try transport.requireParameter("ionHeatDiffusivity")
    /// ```
    public func requireParameter(_ key: String) throws -> Float {
        guard let value = parameters[key] else {
            throw ConfigurationError.missingRequired(
                key: "transport.parameters.\(key) for model \(modelType)"
            )
        }
        return value
    }

    /// Get required model parameter with a model-specific recovery suggestion.
    public func requireParameter(_ key: String, suggestion: String) throws -> Float {
        guard let value = parameters[key] else {
            throw ConfigurationValidationError.missingRequiredParameter(
                parameter: key,
                modelType: modelType,
                suggestion: suggestion
            )
        }

        return value
    }

    /// Get parameter with explicit default
    ///
    /// Use this when you have a context-independent fallback value.
    ///
    /// - Parameters:
    ///   - key: Parameter key (for example, `ionHeatDiffusivity`)
    ///   - defaultValue: Fallback value
    /// - Returns: Parameter value or default
    ///
    /// Example:
    /// ```swift
    /// let particleDiffusivity = transport.parameter("particleDiffusivity", default: 0.0)
    /// ```
    public func parameter(_ key: String, default defaultValue: Float) -> Float {
        parameters[key] ?? defaultValue
    }
}

// MARK: - Conversion to Runtime Parameters

extension TransportConfig {
    /// Convert to TransportParameters for runtime
    public func transportParameters() -> TransportParameters {
        TransportParameters(
            uncheckedModelType: modelType,
            parameters: parameters
        )
    }
}

extension TransportParameters {
    fileprivate init(uncheckedModelType modelType: TransportModelType, parameters: [String: Float]) {
        self.modelType = modelType
        self.parameters = parameters
    }
}
