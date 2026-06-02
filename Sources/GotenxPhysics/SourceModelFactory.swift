// SourceModelFactory.swift
// Factory for creating source models from configuration

import Foundation
import GotenxCore

/// Factory for creating source models from configuration
public struct SourceModelFactory {
    /// Create source models from configuration
    ///
    /// Creates instances of enabled physics sources based on configuration.
    /// Returns a composite model that combines all enabled sources.
    ///
    /// - Parameter config: Sources configuration
    /// - Returns: Composite source model containing all enabled sources
    /// - Throws: ConfigurationError if source initialization fails
    public static func create(config: SourcesConfig) throws -> any SourceModel {
        var sources: [String: any SourceModel] = [:]

        // Add Ohmic heating if enabled
        if config.ohmicHeating {
            sources["ohmic"] = OhmicHeatingSource()
        }

        // Add fusion power if enabled
        if config.fusionPower {
            if let fusionConfig = config.fusionConfig {
                // Create with specific fuel parameters
                let parameters = SourceParameters(
                    modelType: "fusion",
                    parameters: [
                        "deuteriumFraction": fusionConfig.deuteriumFraction,
                        "tritiumFraction": fusionConfig.tritiumFraction,
                        "dilution": fusionConfig.dilution
                    ]
                )
                sources["fusion"] = try FusionPowerSource(parameters: parameters)
            } else {
                // Use defaults
                sources["fusion"] = FusionPowerSource()
            }
        }

        // Add ion-electron exchange if enabled
        if config.ionElectronExchange {
            sources["ionElectronExchange"] = IonElectronExchangeSource()
        }

        // Add Bremsstrahlung radiation if enabled
        if config.bremsstrahlung {
            sources["bremsstrahlung"] = BremsstrahlungSource()
        }

        // Add ECRH if enabled
        if let ecrhConfig = config.ecrh {
            let parameters = SourceParameters(
                modelType: "ecrh",
                parameters: [
                    "total_power": ecrhConfig.totalPower,
                    "deposition_rho": ecrhConfig.normalizedDepositionRadius,
                    "deposition_width": ecrhConfig.depositionWidth,
                    "launch_angle": ecrhConfig.launchAngle ?? 0.0,
                    "frequency": ecrhConfig.frequency ?? 0.0,
                    "current_drive": ecrhConfig.currentDriveEnabled ? 1.0 : 0.0
                ]
            )
            sources["ecrh"] = try ECRHSource(parameters: parameters)
        }

        // Add Gas Puff if enabled
        if let gasPuffConfig = config.gasPuff {
            let parameters = SourceParameters(
                modelType: "gasPuff",
                parameters: [
                    "puff_rate": gasPuffConfig.puffRate,
                    "penetration_depth": gasPuffConfig.penetrationDepth
                ]
            )
            sources["gasPuff"] = try GasPuffSource(parameters: parameters)
        }

        // Add Impurity Radiation if enabled
        if let impurityConfig = config.impurityRadiation {
            // Encode species as atomic number
            let atomicNumber: Float
            switch impurityConfig.species.lowercased() {
            case "carbon": atomicNumber = 6
            case "neon": atomicNumber = 10
            case "argon": atomicNumber = 18
            case "tungsten": atomicNumber = 74
            default: atomicNumber = 18
            }

            let parameters = SourceParameters(
                modelType: "impurityRadiation",
                parameters: [
                    "impurity_fraction": impurityConfig.impurityFraction,
                    "atomic_number": atomicNumber
                ]
            )
            sources["impurityRadiation"] = try ImpurityRadiationSource(parameters: parameters)
        }

        // Return composite model combining all sources
        return CompositeSourceModel(sources: sources)
    }

    /// Create source models from a dictionary of source parameters
    ///
    /// - Parameter sourceParameters: Dictionary of source parameters by name
    /// - Returns: Composite source model
    /// - Throws: ConfigurationError if source type is unknown
    public static func create(from sourceParameters: [String: SourceParameters]) throws -> any SourceModel {
        var sources: [String: any SourceModel] = [:]

        for (name, parameters) in sourceParameters {
            switch parameters.modelType {
            case "ohmic":
                sources[name] = OhmicHeatingSource()

            case "fusion":
                sources[name] = try FusionPowerSource(parameters: parameters)

            case "ionElectronExchange":
                sources[name] = IonElectronExchangeSource(parameters: parameters)

            case "bremsstrahlung":
                sources[name] = BremsstrahlungSource(parameters: parameters)

            case "ecrh":
                sources[name] = try ECRHSource(parameters: parameters)

            case "gasPuff":
                sources[name] = try GasPuffSource(parameters: parameters)

            case "impurityRadiation":
                sources[name] = try ImpurityRadiationSource(parameters: parameters)

            default:
                throw ConfigurationError.invalidValue(
                    key: "source.modelType",
                    value: parameters.modelType,
                    reason: "Unknown source model type. Valid types: ohmic, fusion, ionElectronExchange, bremsstrahlung, ecrh, gasPuff, impurityRadiation"
                )
            }
        }

        return CompositeSourceModel(sources: sources)
    }

    /// Create a single source model by name
    ///
    /// - Parameters:
    ///   - name: Source model name
    ///   - parameters: Optional source parameters
    /// - Returns: Source model instance
    /// - Throws: ConfigurationError if source name is unknown
    public static func createSingle(name: String, parameters: SourceParameters? = nil) throws -> any SourceModel {
        switch name {
        case "ohmic":
            return OhmicHeatingSource()

        case "fusion":
            if let parameters = parameters {
                return try FusionPowerSource(parameters: parameters)
            } else {
                return FusionPowerSource()
            }

        case "ionElectronExchange":
            return IonElectronExchangeSource()

        case "bremsstrahlung":
            return BremsstrahlungSource()

        case "ecrh":
            if let parameters = parameters {
                return try ECRHSource(parameters: parameters)
            } else {
                // Use default ECRH configuration
                let defaultParams = SourceParameters(
                    modelType: "ecrh",
                    parameters: [
                        "total_power": 20e6,
                        "deposition_rho": 0.5,
                        "deposition_width": 0.1
                    ]
                )
                return try ECRHSource(parameters: defaultParams)
            }

        case "gasPuff":
            if let parameters = parameters {
                return try GasPuffSource(parameters: parameters)
            } else {
                // Use default Gas Puff configuration
                let defaultParams = SourceParameters(
                    modelType: "gasPuff",
                    parameters: [
                        "puff_rate": 1e21,
                        "penetration_depth": 0.1
                    ]
                )
                return try GasPuffSource(parameters: defaultParams)
            }

        case "impurityRadiation":
            if let parameters = parameters {
                return try ImpurityRadiationSource(parameters: parameters)
            } else {
                // Use default Impurity Radiation configuration
                let defaultParams = SourceParameters(
                    modelType: "impurityRadiation",
                    parameters: [
                        "impurity_fraction": 0.001,
                        "atomic_number": 18  // Argon
                    ]
                )
                return try ImpurityRadiationSource(parameters: defaultParams)
            }

        default:
            throw ConfigurationError.invalidValue(
                key: "sourceName",
                value: name,
                reason: "Unknown source model name. Valid names: ohmic, fusion, ionElectronExchange, bremsstrahlung, ecrh, gasPuff, impurityRadiation"
            )
        }
    }
}
