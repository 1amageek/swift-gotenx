// SourceModelAdapters.swift
// Adapters to make physics sources conform to SourceModel protocol

import Foundation
import MLX
import GotenxCore

// MARK: - Ohmic Heating Source

/// Ohmic heating source model adapter
public struct OhmicHeatingSource: SourceModel {
    public let name: String = "ohmic"
    private let model: OhmicHeating

    public init() {
        self.model = OhmicHeating()
    }

    public init(effectiveCharge: Float = 1.5, coulombLogarithm: Float = 17.0, useNeoclassical: Bool = true) {
        self.model = OhmicHeating(effectiveCharge: effectiveCharge, coulombLogarithm: coulombLogarithm, useNeoclassical: useNeoclassical)
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([cellCount])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty
        )

        return try model.applyToSources(
            emptySourceTerms,
            profiles: profiles,
            geometry: geometry,
            plasmaCurrentDensity: nil
        )
    }

    public func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let emptySourceTerms = SourceTerms.zero(
            cellCount: cellCount,
            metadata: nil,
            validateDebugUnits: false
        )

        do {
            return try model.applyToSourcesForSolver(
                emptySourceTerms,
                profiles: profiles,
                geometry: geometry,
                plasmaCurrentDensity: nil
            )
        } catch {
            return SourceTerms.invalidNumerics(cellCount: cellCount)
        }
    }
}

// MARK: - Fusion Power Source

/// Fusion power source model adapter
public struct FusionPowerSource: SourceModel {
    public let name: String = "fusion"
    private let model: FusionPower

    public init() {
        do {
            self.model = try FusionPower(fuelMixture: .equalDT, fuelDilution: 0.9)
        } catch {
            preconditionFailure("Default fusion power parameters must be valid: \(error)")
        }
    }

    public init(parameters: SourceParameters) throws {
        let dFraction = parameters.parameters["deuteriumFraction"] ?? 0.5
        let tFraction = parameters.parameters["tritiumFraction"] ?? 0.5
        let dilution = parameters.parameters["dilution"] ?? 0.9

        // Create fuel mix from fractions
        let fuelMixture = FusionPower.FuelMixture.custom(
            deuteriumFraction: dFraction,
            tritiumFraction: tFraction
        )

        self.model = try FusionPower(fuelMixture: fuelMixture, fuelDilution: dilution)
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([cellCount])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty
        )

        let sourceTerms = try model.applyToSources(
            emptySourceTerms,
            profiles: profiles
        )

        let metadata = try model.computeMetadata(
            profiles: profiles,
            geometry: geometry
        )

        return SourceTerms(
            ionHeating: sourceTerms.ionHeating,
            electronHeating: sourceTerms.electronHeating,
            particleSource: sourceTerms.particleSource,
            currentSource: sourceTerms.currentSource,
            metadata: SourceMetadataCollection(entries: [metadata])
        )
    }

}

// MARK: - Ion-Electron Exchange Source

/// Ion-electron heat exchange source model adapter
public struct IonElectronExchangeSource: SourceModel {
    public let name: String = "ionElectronExchange"
    private let model: IonElectronExchange

    public init() {
        self.model = IonElectronExchange()
    }

    public init(parameters: SourceParameters) {
        self.model = IonElectronExchange()
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([cellCount])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty
        )

        let safeProfiles = profiles.withElectronDensityClamped()

        return try model.applyToSources(
            emptySourceTerms,
            profiles: safeProfiles,
            geometry: geometry
        )
    }

    public func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let emptySourceTerms = SourceTerms.zero(
            cellCount: cellCount,
            metadata: nil,
            validateDebugUnits: false
        )
        let safeProfiles = profiles.withElectronDensityClamped()

        do {
            return try model.applyToSourcesForSolver(
                emptySourceTerms,
                profiles: safeProfiles
            )
        } catch {
            return SourceTerms.invalidNumerics(cellCount: cellCount)
        }
    }
}

// MARK: - Bremsstrahlung Source

/// Bremsstrahlung radiation source model adapter
public struct BremsstrahlungSource: SourceModel {
    public let name: String = "bremsstrahlung"
    private let model: Bremsstrahlung

    public init() {
        self.model = Bremsstrahlung()
    }

    public init(parameters: SourceParameters) {
        self.model = Bremsstrahlung()
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([cellCount])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty
        )

        let safeProfiles = profiles.withElectronDensityClamped()

        return try model.applyToSources(
            emptySourceTerms,
            profiles: safeProfiles,
            geometry: geometry
        )
    }

    public func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let emptySourceTerms = SourceTerms.zero(
            cellCount: cellCount,
            metadata: nil,
            validateDebugUnits: false
        )
        let safeProfiles = profiles.withElectronDensityClamped()

        do {
            return try model.applyToSourcesForSolver(
                emptySourceTerms,
                profiles: safeProfiles
            )
        } catch {
            return SourceTerms.invalidNumerics(cellCount: cellCount)
        }
    }
}

// MARK: - ECRH Source

/// ECRH (Electron Cyclotron Resonance Heating) source model adapter
public struct ECRHSource: SourceModel {
    public let name: String = "ecrh"
    private let model: ECRHModel

    public init() {
        // Default: 20 MW at ρ=0.5, width=0.1
        self.model = ECRHModel(
            totalPower: 20e6,
            normalizedDepositionRadius: 0.5,
            depositionWidth: 0.1
        )
    }

    public init(parameters: SourceParameters) throws {
        guard let totalPower = parameters.parameters["total_power"], totalPower >= 0 else {
            throw ECRHError.negativePower(parameters.parameters["total_power"] ?? -1)
        }

        let normalizedDepositionRadius = parameters.parameters["deposition_rho"] ?? 0.5
        guard normalizedDepositionRadius >= 0 && normalizedDepositionRadius <= 1.0 else {
            throw ECRHError.invalidDepositionLocation(normalizedDepositionRadius)
        }

        let depositionWidth = parameters.parameters["deposition_width"] ?? 0.1
        guard depositionWidth > 0 && depositionWidth < 0.5 else {
            throw ECRHError.invalidDepositionWidth(depositionWidth)
        }

        self.model = ECRHModel(
            totalPower: totalPower,
            normalizedDepositionRadius: normalizedDepositionRadius,
            depositionWidth: depositionWidth,
            launchAngle: parameters.parameters["launch_angle"],
            frequency: parameters.parameters["frequency"],
            enableCurrentDrive: (parameters.parameters["current_drive"] ?? 0.0) > 0.5
        )
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([cellCount])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty
        )

        let sourceTerms = try model.applyToSources(
            emptySourceTerms,
            profiles: profiles,
            geometry: geometry
        )

        let metadata = model.computeMetadata(geometry: geometry)

        return SourceTerms(
            ionHeating: sourceTerms.ionHeating,
            electronHeating: sourceTerms.electronHeating,
            particleSource: sourceTerms.particleSource,
            currentSource: sourceTerms.currentSource,
            metadata: SourceMetadataCollection(entries: [metadata])
        )
    }
}

// MARK: - Gas Puff Source

/// Gas puff particle source model adapter
public struct GasPuffSource: SourceModel {
    public let name: String = "gasPuff"
    private let model: GasPuffModel

    public init() {
        // Default: 1e21 particles/s with moderate penetration
        self.model = GasPuffModel(
            puffRate: 1e21,
            penetrationDepth: 0.1
        )
    }

    public init(parameters: SourceParameters) throws {
        guard let puffRate = parameters.parameters["puff_rate"], puffRate >= 0 else {
            throw GasPuffError.negativePuffRate(parameters.parameters["puff_rate"] ?? -1)
        }

        let penetrationDepth = parameters.parameters["penetration_depth"] ?? 0.1
        guard penetrationDepth > 0 && penetrationDepth <= 1.0 else {
            throw GasPuffError.invalidPenetrationDepth(penetrationDepth)
        }

        self.model = GasPuffModel(
            puffRate: puffRate,
            penetrationDepth: penetrationDepth
        )
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms {
        let sourceTerms = model.applyToSources(
            SourceTerms(
                ionHeating: EvaluatedArray.zeros([profiles.ionTemperature.shape[0]]),
                electronHeating: EvaluatedArray.zeros([profiles.ionTemperature.shape[0]]),
                particleSource: EvaluatedArray.zeros([profiles.ionTemperature.shape[0]]),
                currentSource: EvaluatedArray.zeros([profiles.ionTemperature.shape[0]]),
                metadata: SourceMetadataCollection.empty
            ),
            geometry: geometry
        )

        let metadata = model.computeMetadata(geometry: geometry)

        return SourceTerms(
            ionHeating: sourceTerms.ionHeating,
            electronHeating: sourceTerms.electronHeating,
            particleSource: sourceTerms.particleSource,
            currentSource: sourceTerms.currentSource,
            metadata: SourceMetadataCollection(entries: [metadata])
        )
    }
}

// MARK: - Impurity Radiation Source

/// Impurity radiation source model adapter
public struct ImpurityRadiationSource: SourceModel {
    public let name: String = "impurityRadiation"
    private let model: ImpurityRadiationModel

    public init() {
        // Default: Argon with 0.1% impurity fraction
        self.model = ImpurityRadiationModel(
            impurityFraction: 0.001,
            species: .argon
        )
    }

    public init(parameters: SourceParameters) throws {
        let impurityFraction = parameters.parameters["impurity_fraction"] ?? 0.001

        guard impurityFraction >= 0 else {
            throw ImpurityRadiationError.negativeImpurityFraction(impurityFraction)
        }

        guard impurityFraction < 0.1 else {
            throw ImpurityRadiationError.excessiveImpurityFraction(impurityFraction)
        }

        // Parse species from atomic number
        let atomicNumber = parameters.parameters["atomic_number"] ?? 18  // Default: Argon
        let species: ImpurityRadiationModel.ImpuritySpecies
        switch Int(atomicNumber) {
        case 6:
            species = .carbon
        case 10:
            species = .neon
        case 18:
            species = .argon
        case 74:
            species = .tungsten
        default:
            throw ImpurityRadiationError.unknownSpecies("Z=\(Int(atomicNumber))")
        }

        self.model = ImpurityRadiationModel(
            impurityFraction: impurityFraction,
            species: species
        )
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([cellCount])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty
        )

        let sourceTerms = try model.applyToSources(
            emptySourceTerms,
            profiles: profiles
        )

        let metadata = model.computeMetadata(
            profiles: profiles,
            geometry: geometry
        )

        return SourceTerms(
            ionHeating: sourceTerms.ionHeating,
            electronHeating: sourceTerms.electronHeating,
            particleSource: sourceTerms.particleSource,
            currentSource: sourceTerms.currentSource,
            metadata: SourceMetadataCollection(entries: [metadata])
        )
    }
}

// MARK: - Composite Source Model

/// Composite source model that combines multiple sources
public struct CompositeSourceModel: SourceModel {
    public let name: String = "composite"
    private let sources: [String: any SourceModel]

    public init(sources: [String: any SourceModel]) {
        self.sources = sources
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms {
        let cellCount = profiles.ionTemperature.shape[0]
        var totalIonHeating = MLXArray.zeros([cellCount])
        var totalElectronHeating = MLXArray.zeros([cellCount])
        var totalParticleSource = MLXArray.zeros([cellCount])
        var totalCurrentSource = MLXArray.zeros([cellCount])

        // Collect metadata from all sources
        var allMetadata: [SourceMetadata] = []

        // Accumulate contributions from all sources
        for (_, source) in sources {
            let terms = try source.computeTerms(
                profiles: profiles,
                geometry: geometry,
                parameters: parameters
            )

            totalIonHeating = totalIonHeating + terms.ionHeating.value
            totalElectronHeating = totalElectronHeating + terms.electronHeating.value
            totalParticleSource = totalParticleSource + terms.particleSource.value
            totalCurrentSource = totalCurrentSource + terms.currentSource.value

            // Collect metadata if present
            if let metadata = terms.metadata {
                allMetadata.append(contentsOf: metadata.entries)
            }
        }

        // Evaluate all arrays in batch
        let evaluated = EvaluatedArray.evaluatingBatch([
            totalIonHeating,
            totalElectronHeating,
            totalParticleSource,
            totalCurrentSource
        ])

        // Create aggregated metadata collection
        // Always return a metadata collection (empty if no sources provide metadata)
        // to prevent crashes in DerivedQuantitiesComputer when checking metadata != nil
        let metadata = allMetadata.isEmpty ? SourceMetadataCollection.empty : SourceMetadataCollection(entries: allMetadata)

        return SourceTerms(
            ionHeating: evaluated[0],
            electronHeating: evaluated[1],
            particleSource: evaluated[2],
            currentSource: evaluated[3],
            metadata: metadata
        )
    }
}
