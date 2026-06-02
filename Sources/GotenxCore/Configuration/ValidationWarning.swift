// ValidationWarning.swift
// Configuration validation warnings

import Foundation

/// Non-critical configuration validation warnings
public enum ConfigurationValidationWarning: Error, LocalizedError {
    case highPowerDensity(value: Float, limit: Float, suggestion: String)
    case highCurrentDensity(value: Float, limit: Float, suggestion: String)
    case lowTemperatureForOhmic(value: Float, limit: Float, suggestion: String)
    case highPuffRate(value: Float, limit: Float, suggestion: String)
    case flatProfile(parameter: String, coreFactor: Float, suggestion: String)
    case timestepTooSmall(timeStep: Float, timeScale: Float, suggestion: String)
    case poorTimeResolution(parameter: String, timeStep: Float, timeScale: Float, suggestion: String)
    case excessiveMeshResolution(cellCount: Int, maximum: Int, suggestion: String)
    case insufficientGradientResolution(cellCount: Int, recommended: Int, profileExponent: Float, suggestion: String)
    case outsideTrainingRange(model: String, parameter: String, value: Float, range: (Float, Float), suggestion: String)
    case negligibleFusionPower(temperature: Float, threshold: Float, suggestion: String)

    public var errorDescription: String? {
        switch self {
        case .highPowerDensity(let value, let limit, let suggestion):
            return """
            WARNING: High ECRH power density detected
              Peak power density: \(String(format: "%.1f", value)) MW/m³
              Typical limit: \(String(format: "%.1f", limit)) MW/m³
              Suggestion: \(suggestion)
            """

        case .highCurrentDensity(let value, let limit, let suggestion):
            return """
            WARNING: High plasma current density
              Current density: \(String(format: "%.1f", value)) MA/m²
              Typical limit: \(String(format: "%.1f", limit)) MA/m²
              Suggestion: \(suggestion)
            """

        case .lowTemperatureForOhmic(let value, let limit, let suggestion):
            return """
            WARNING: Low temperature for Ohmic heating model
              Temperature: \(String(format: "%.1f", value)) eV
              Model accurate above: \(String(format: "%.1f", limit)) eV
              Suggestion: \(suggestion)
            """

        case .highPuffRate(let value, let limit, let suggestion):
            return """
            WARNING: High gas puff rate
              Puff rate: \(String(format: "%.2e", value)) particles/s
              Typical limit: \(String(format: "%.2e", limit)) particles/s
              Suggestion: \(suggestion)
            """

        case .flatProfile(let param, let coreFactor, let suggestion):
            return """
            WARNING: Profile may be too flat - \(param)
              Core factor: \(String(format: "%.2f", coreFactor))
              Suggestion: \(suggestion)
            """

        case .timestepTooSmall(let timeStep, let timeScale, let suggestion):
            return """
            WARNING: Timestep may be unnecessarily small
              Current timeStep: \(String(format: "%.2e", timeStep)) s
              Physics time scale: \(String(format: "%.2e", timeScale)) s
              Suggestion: \(suggestion)
            """

        case .poorTimeResolution(let param, let timeStep, let timeScale, let suggestion):
            return """
            WARNING: Poor time resolution for \(param)
              Current timeStep: \(String(format: "%.2e", timeStep)) s
              Time scale: \(String(format: "%.2e", timeScale)) s
              Suggestion: \(suggestion)
            """

        case .excessiveMeshResolution(let cellCount, let maximum, let suggestion):
            return """
            WARNING: Excessive mesh resolution
              Current cellCount: \(cellCount)
              Recommended maximum: \(maximum)
              Suggestion: \(suggestion)
            """

        case .insufficientGradientResolution(let cellCount, let recommended, let exponent, let suggestion):
            return """
            WARNING: Insufficient gradient resolution
              Current cellCount: \(cellCount)
              Recommended: \(recommended) (for profile exponent \(String(format: "%.1f", exponent)))
              Suggestion: \(suggestion)
            """

        case .outsideTrainingRange(let model, let param, let value, let range, let suggestion):
            return """
            WARNING: \(param) outside \(model) training range
              Value: \(String(format: "%.2e", value))
              Training range: \(String(format: "%.2e", range.0)) - \(String(format: "%.2e", range.1))
              Suggestion: \(suggestion)
            """

        case .negligibleFusionPower(let temperature, let threshold, let suggestion):
            return """
            WARNING: Fusion power will be negligible
              Ion temperature: \(String(format: "%.1f", temperature)) eV
              Fusion threshold: ~\(String(format: "%.1f", threshold)) eV
              Suggestion: \(suggestion)
            """
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .highPowerDensity(_, _, let suggestion),
             .highCurrentDensity(_, _, let suggestion),
             .lowTemperatureForOhmic(_, _, let suggestion),
             .highPuffRate(_, _, let suggestion),
             .flatProfile(_, _, let suggestion),
             .timestepTooSmall(_, _, let suggestion),
             .poorTimeResolution(_, _, _, let suggestion),
             .excessiveMeshResolution(_, _, let suggestion),
             .insufficientGradientResolution(_, _, _, let suggestion),
             .outsideTrainingRange(_, _, _, _, let suggestion),
             .negligibleFusionPower(_, _, let suggestion):
            return suggestion
        }
    }
}
