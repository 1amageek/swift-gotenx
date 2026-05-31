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

    public init(Zeff: Float = 1.5, lnLambda: Float = 17.0, useNeoclassical: Bool = true) {
        self.model = OhmicHeating(Zeff: Zeff, lnLambda: lnLambda, useNeoclassical: useNeoclassical)
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        // Start with zero sources
        let nCells = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([nCells])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty  // Always provide metadata (empty on error)
        )

        // Apply Ohmic heating (needs geometry)
        do {
            let sourceTerms = try model.applyToSources(
                emptySourceTerms,
                profiles: profiles,
                geometry: geometry,
                plasmaCurrentDensity: nil
            )

            // applyToSources now includes metadata, return directly
            return sourceTerms
        } catch {
            // If computation fails, return zeros with warning
            print("⚠️  Warning: Ohmic heating computation failed: \(error)")
            return emptySourceTerms
        }
    }

    public func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        let emptySourceTerms = SourceTerms.zero(
            nCells: nCells,
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
            print("Warning: Ohmic heating solver computation failed: \(error)")
            return emptySourceTerms
        }
    }
}

// MARK: - Fusion Power Source

/// Fusion power source model adapter
public struct FusionPowerSource: SourceModel {
    public let name: String = "fusion"
    private let model: FusionPower

    public init() {
        // Use default equal D-T mix
        self.model = try! FusionPower(fuelMix: .equalDT, fuelDilution: 0.9)
    }

    public init(params: SourceParameters) {
        let dFraction = params.params["deuteriumFraction"] ?? 0.5
        let tFraction = params.params["tritiumFraction"] ?? 0.5
        let dilution = params.params["dilution"] ?? 0.9

        // Create fuel mix from fractions
        let fuelMix = FusionPower.FuelMixture.custom(nD_frac: dFraction, nT_frac: tFraction)

        self.model = try! FusionPower(fuelMix: fuelMix, fuelDilution: dilution)
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([nCells])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty  // Always provide metadata (empty on error)
        )

        // FusionPower.applyToSources does NOT take geometry parameter
        do {
            let sourceTerms = try model.applyToSources(
                emptySourceTerms,
                profiles: profiles
            )

            // Compute metadata
            let metadata = try model.computeMetadata(
                profiles: profiles,
                geometry: geometry
            )

            // Return with metadata
            return SourceTerms(
                ionHeating: sourceTerms.ionHeating,
                electronHeating: sourceTerms.electronHeating,
                particleSource: sourceTerms.particleSource,
                currentSource: sourceTerms.currentSource,
                metadata: SourceMetadataCollection(entries: [metadata])
            )
        } catch {
            print("⚠️  Warning: Fusion power computation failed: \(error)")
            return emptySourceTerms
        }
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

    public init(params: SourceParameters) {
        self.model = IonElectronExchange()
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([nCells])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty  // Always provide metadata (empty on error)
        )

        // IonElectronExchange.applyToSources now takes geometry parameter
        let safeProfiles = profiles.withElectronDensityClamped()

        do {
            let sourceTerms = try model.applyToSources(
                emptySourceTerms,
                profiles: safeProfiles,
                geometry: geometry
            )

            // applyToSources now includes metadata, return directly
            return sourceTerms
        } catch {
            print("⚠️  Warning: Ion-electron exchange computation failed: \(error)")
            return emptySourceTerms
        }
    }

    public func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        let emptySourceTerms = SourceTerms.zero(
            nCells: nCells,
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
            print("Warning: Ion-electron exchange solver computation failed: \(error)")
            return emptySourceTerms
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

    public init(params: SourceParameters) {
        self.model = Bremsstrahlung()
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([nCells])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty  // Always provide metadata (empty on error)
        )

        // Bremsstrahlung.applyToSources now takes geometry parameter
        let safeProfiles = profiles.withElectronDensityClamped()

        do {
            let sourceTerms = try model.applyToSources(
                emptySourceTerms,
                profiles: safeProfiles,
                geometry: geometry
            )

            // applyToSources now includes metadata, return directly
            return sourceTerms
        } catch {
            print("⚠️  Warning: Bremsstrahlung computation failed: \(error)")
            return emptySourceTerms
        }
    }

    public func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        let emptySourceTerms = SourceTerms.zero(
            nCells: nCells,
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
            print("Warning: Bremsstrahlung solver computation failed: \(error)")
            return emptySourceTerms
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
            depositionRho: 0.5,
            depositionWidth: 0.1
        )
    }

    public init(params: SourceParameters) throws {
        guard let totalPower = params.params["total_power"], totalPower >= 0 else {
            throw ECRHError.negativePower(params.params["total_power"] ?? -1)
        }

        let depositionRho = params.params["deposition_rho"] ?? 0.5
        guard depositionRho >= 0 && depositionRho <= 1.0 else {
            throw ECRHError.invalidDepositionLocation(depositionRho)
        }

        let depositionWidth = params.params["deposition_width"] ?? 0.1
        guard depositionWidth > 0 && depositionWidth < 0.5 else {
            throw ECRHError.invalidDepositionWidth(depositionWidth)
        }

        self.model = ECRHModel(
            totalPower: totalPower,
            depositionRho: depositionRho,
            depositionWidth: depositionWidth,
            launchAngle: params.params["launch_angle"],
            frequency: params.params["frequency"],
            enableCurrentDrive: (params.params["current_drive"] ?? 0.0) > 0.5
        )
    }

    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([nCells])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty  // Always provide metadata (empty on error)
        )

        // Apply ECRH heating
        do {
            let sourceTerms = try model.applyToSources(
                emptySourceTerms,
                profiles: profiles,
                geometry: geometry
            )

            // Compute metadata
            let metadata = model.computeMetadata(geometry: geometry)

            // Return with metadata
            return SourceTerms(
                ionHeating: sourceTerms.ionHeating,
                electronHeating: sourceTerms.electronHeating,
                particleSource: sourceTerms.particleSource,
                currentSource: sourceTerms.currentSource,
                metadata: SourceMetadataCollection(entries: [metadata])
            )
        } catch {
            print("⚠️  Warning: ECRH computation failed: \(error)")
            return emptySourceTerms
        }
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

    public init(params: SourceParameters) throws {
        guard let puffRate = params.params["puff_rate"], puffRate >= 0 else {
            throw GasPuffError.negativePuffRate(params.params["puff_rate"] ?? -1)
        }

        let penetrationDepth = params.params["penetration_depth"] ?? 0.1
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
        params: SourceParameters
    ) -> SourceTerms {
        // Apply gas puff particle source
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

        // Compute metadata
        let metadata = model.computeMetadata(geometry: geometry)

        // Return with metadata
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

    public init(params: SourceParameters) throws {
        let impurityFraction = params.params["impurity_fraction"] ?? 0.001

        guard impurityFraction >= 0 else {
            throw ImpurityRadiationError.negativeImpurityFraction(impurityFraction)
        }

        guard impurityFraction < 0.1 else {
            throw ImpurityRadiationError.excessiveImpurityFraction(impurityFraction)
        }

        // Parse species from atomic number
        let atomicNumber = params.params["atomic_number"] ?? 18  // Default: Argon
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
        params: SourceParameters
    ) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        let zeros = EvaluatedArray.zeros([nCells])

        let emptySourceTerms = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty  // Always provide metadata (empty on error)
        )

        // Apply impurity radiation (negative electron heating)
        do {
            let sourceTerms = try model.applyToSources(
                emptySourceTerms,
                profiles: profiles
            )

            // Compute metadata
            let metadata = model.computeMetadata(
                profiles: profiles,
                geometry: geometry
            )

            // Return with metadata
            return SourceTerms(
                ionHeating: sourceTerms.ionHeating,
                electronHeating: sourceTerms.electronHeating,
                particleSource: sourceTerms.particleSource,
                currentSource: sourceTerms.currentSource,
                metadata: SourceMetadataCollection(entries: [metadata])
            )
        } catch {
            print("⚠️  Warning: Impurity radiation computation failed: \(error)")
            return emptySourceTerms
        }
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
        params: SourceParameters
    ) -> SourceTerms {
        let nCells = profiles.ionTemperature.shape[0]
        var totalIonHeating = MLXArray.zeros([nCells])
        var totalElectronHeating = MLXArray.zeros([nCells])
        var totalParticleSource = MLXArray.zeros([nCells])
        var totalCurrentSource = MLXArray.zeros([nCells])

        // Collect metadata from all sources
        var allMetadata: [SourceMetadata] = []

        // Accumulate contributions from all sources
        for (_, source) in sources {
            let terms = source.computeTerms(
                profiles: profiles,
                geometry: geometry,
                params: params
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
