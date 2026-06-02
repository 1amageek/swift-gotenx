// ValidationError.swift
// Configuration validation errors

import Foundation

/// Critical configuration validation errors that prevent simulation
public enum ConfigurationValidationError: Error, LocalizedError {
    case missingRequiredParameter(parameter: String, modelType: TransportModelType, suggestion: String)
    case unknownTransportParameter(parameter: String, modelType: TransportModelType, allowed: [String])
    case transportModelMismatch(expected: TransportModelType, actual: TransportModelType)
    case invalidParameter(parameter: String, value: Float, reason: String)
    case unstableTimestep(parameter: String, changeRatio: Float, suggestion: String)
    case cflViolation(parameter: String, cfl: Float, limit: Float, suggestion: String)
    case insufficientResolution(parameter: String, value: Float, minimum: Float, suggestion: String)
    case outOfPhysicalRange(parameter: String, value: Float, range: (Float, Float), unit: String)
    case inconsistentBoundary(parameter: String, coreValue: Float, boundaryValue: Float, suggestion: String)
    case invalidGeometry(parameter: String, value: Float, limit: Float, suggestion: String)
    case negativeTransportCoefficient(parameter: String, value: Float)
    case invalidFuelMix(dFraction: Float, tFraction: Float, suggestion: String)
    case timestepTooLarge(timeStep: Float, timeScale: Float, suggestion: String)
    case insufficientMeshResolution(cellCount: Int, minimum: Int, suggestion: String)

    public var errorDescription: String? {
        switch self {
        case .missingRequiredParameter(let param, let modelType, let suggestion):
            return """
            ERROR: Missing required parameter - \(param)
              Model type: \(modelType)
              Suggestion: \(suggestion)
            """

        case .unknownTransportParameter(let param, let modelType, let allowed):
            return """
            ERROR: Unknown transport parameter - \(param)
              Model type: \(modelType)
              Allowed parameters: \(allowed.joined(separator: ", "))
            """

        case .transportModelMismatch(let expected, let actual):
            return """
            ERROR: Transport model mismatch
              Expected model type: \(expected)
              Actual model type: \(actual)
            """

        case .invalidParameter(let param, let value, let reason):
            return """
            ERROR: Invalid parameter - \(param)
              Value: \(String(format: "%.3e", value))
              Reason: \(reason)
            """

        case .unstableTimestep(let param, let ratio, let suggestion):
            return """
            ERROR: Unstable timestep for \(param)
              Change per timestep: \(Int(ratio * 100))% (limit: 50%)
              Suggestion: \(suggestion)
            """

        case .cflViolation(let param, let cfl, let limit, let suggestion):
            return """
            ERROR: CFL condition violated for \(param)
              CFL = \(String(format: "%.2f", cfl)) (limit: \(String(format: "%.2f", limit)))
              Suggestion: \(suggestion)
            """

        case .insufficientResolution(let param, let value, let minimum, let suggestion):
            return """
            ERROR: Insufficient resolution for \(param)
              Current: \(String(format: "%.3f", value)), Minimum: \(String(format: "%.3f", minimum))
              Suggestion: \(suggestion)
            """

        case .outOfPhysicalRange(let param, let value, let range, let unit):
            return """
            ERROR: \(param) out of physical range
              Value: \(String(format: "%.2e", value)) \(unit)
              Valid range: \(String(format: "%.2e", range.0)) - \(String(format: "%.2e", range.1)) \(unit)
            """

        case .inconsistentBoundary(let param, let coreValue, let boundaryValue, let suggestion):
            return """
            ERROR: Inconsistent boundary condition for \(param)
              Core value: \(String(format: "%.2f", coreValue))
              Boundary value: \(String(format: "%.2f", boundaryValue))
              Suggestion: \(suggestion)
            """

        case .invalidGeometry(let param, let value, let limit, let suggestion):
            return """
            ERROR: Invalid geometry - \(param)
              Value: \(String(format: "%.3f", value)) (limit: \(String(format: "%.3f", limit)))
              Suggestion: \(suggestion)
            """

        case .negativeTransportCoefficient(let param, let value):
            return """
            ERROR: Negative transport coefficient - \(param)
              Value: \(String(format: "%.3e", value))
              Transport coefficients must be positive.
            """

        case .invalidFuelMix(let dFraction, let tFraction, let suggestion):
            return """
            ERROR: Invalid fuel mix
              Deuterium fraction: \(String(format: "%.3f", dFraction))
              Tritium fraction: \(String(format: "%.3f", tFraction))
              Suggestion: \(suggestion)
            """

        case .timestepTooLarge(let timeStep, let timeScale, let suggestion):
            return """
            ERROR: Timestep too large
              Current timeStep: \(String(format: "%.2e", timeStep)) s
              Physics time scale: \(String(format: "%.2e", timeScale)) s
              Suggestion: \(suggestion)
            """

        case .insufficientMeshResolution(let cellCount, let minimum, let suggestion):
            return """
            ERROR: Insufficient mesh resolution
              Current cellCount: \(cellCount)
              Minimum: \(minimum)
              Suggestion: \(suggestion)
            """
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .missingRequiredParameter(_, _, let suggestion),
             .unstableTimestep(_, _, let suggestion),
             .cflViolation(_, _, _, let suggestion),
             .insufficientResolution(_, _, _, let suggestion),
             .inconsistentBoundary(_, _, _, let suggestion),
             .invalidGeometry(_, _, _, let suggestion),
             .invalidFuelMix(_, _, let suggestion),
             .timestepTooLarge(_, _, let suggestion),
             .insufficientMeshResolution(_, _, let suggestion):
            return suggestion

        case .unknownTransportParameter(_, _, let allowed):
            return "Use one of the model-specific parameters: \(allowed.joined(separator: ", "))"

        case .transportModelMismatch(let expected, _):
            return "Use transport parameters whose modelType is \(expected)"

        case .invalidParameter(_, _, let reason):
            return reason

        case .outOfPhysicalRange:
            return "Adjust parameter to be within valid physical range"

        case .negativeTransportCoefficient:
            return "Set transport coefficient to a positive value"
        }
    }
}
