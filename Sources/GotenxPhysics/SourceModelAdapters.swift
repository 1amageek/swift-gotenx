// SourceModelAdapters.swift
// Adapters to make physics sources conform to SourceModel protocol

import Foundation
import MLX
import GotenxCore

private extension SourceEvaluationContext {
    func sourceTermsWithMetadata(
        _ terms: SourceTerms,
        metadata: () -> SourceMetadata
    ) -> SourceTerms {
        guard includesMetadata else {
            return terms
        }
        return terms.replacingMetadata(
            SourceMetadataCollection(entries: [metadata()]),
            validateDebugUnits: validatesDebugUnits
        )
    }
}

// MARK: - Ohmic Heating Source

/// Ohmic heating source model adapter
public struct OhmicHeatingSource: SourceModel {
    public let name: String = "ohmic"
    private let model: OhmicHeating

    public init() {
        self.model = OhmicHeating()
    }

    public init(effectiveCharge: Float = 1.5, coulombLogarithm: Float = 17.0, useNeoclassical: Bool = true) {
        self.model = OhmicHeating(
            effectiveCharge: effectiveCharge,
            coulombLogarithm: coulombLogarithm,
            useNeoclassical: useNeoclassical
        )
    }

    public func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        try model.applyToSources(
            context.zeroSourceTerms(),
            profiles: context.profiles,
            context: context,
            plasmaCurrentDensity: nil
        )
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
        let fuelMixture = FusionPower.FuelMixture.custom(
            deuteriumFraction: dFraction,
            tritiumFraction: tFraction
        )

        self.model = try FusionPower(fuelMixture: fuelMixture, fuelDilution: dilution)
    }

    public func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        try model.applyToSources(
            context.zeroSourceTerms(),
            profiles: context.profiles,
            context: context
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

    public func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        let safeProfiles = context.profiles.withElectronDensityClamped(
            evaluationMode: context.evaluationMode
        )

        return try model.applyToSources(
            context.zeroSourceTerms(),
            profiles: safeProfiles,
            context: context
        )
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

    public func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        let safeProfiles = context.profiles.withElectronDensityClamped(
            evaluationMode: context.evaluationMode
        )

        return try model.applyToSources(
            context.zeroSourceTerms(),
            profiles: safeProfiles,
            context: context
        )
    }
}

// MARK: - ECRH Source

/// ECRH (Electron Cyclotron Resonance Heating) source model adapter
public struct ECRHSource: SourceModel {
    public let name: String = "ecrh"
    private let model: ECRHModel

    public init() {
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

    public func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        let terms = model.applyToSources(
            context.zeroSourceTerms(),
            profiles: context.profiles,
            context: context
        )
        return context.sourceTermsWithMetadata(
            terms,
            metadata: { model.computeMetadata(geometry: context.geometry) }
        )
    }
}

// MARK: - Gas Puff Source

/// Gas puff particle source model adapter
public struct GasPuffSource: SourceModel {
    public let name: String = "gasPuff"
    private let model: GasPuffModel

    public init() {
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

    public func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        let terms = model.applyToSources(
            context.zeroSourceTerms(),
            context: context
        )
        return context.sourceTermsWithMetadata(
            terms,
            metadata: { model.computeMetadata(geometry: context.geometry) }
        )
    }
}

// MARK: - Impurity Radiation Source

/// Impurity radiation source model adapter
public struct ImpurityRadiationSource: SourceModel {
    public let name: String = "impurityRadiation"
    private let model: ImpurityRadiationModel

    public init() {
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

        let atomicNumber = parameters.parameters["atomic_number"] ?? 18
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

    public func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        model.applyToSources(
            context.zeroSourceTerms(),
            profiles: context.profiles,
            context: context
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

    public func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        guard !sources.isEmpty else {
            return context.zeroSourceTerms()
        }

        guard context.includesMetadata else {
            return try computeSolverTerms(in: context)
        }

        var ionHeating: [MLXArray] = []
        var electronHeating: [MLXArray] = []
        var particleSource: [MLXArray] = []
        var currentSource: [MLXArray] = []
        var metadataEntries: [SourceMetadata] = []

        ionHeating.reserveCapacity(sources.count)
        electronHeating.reserveCapacity(sources.count)
        particleSource.reserveCapacity(sources.count)
        currentSource.reserveCapacity(sources.count)
        metadataEntries.reserveCapacity(sources.count)

        for (_, source) in sources {
            let terms = try source.computeTerms(in: context)
            ionHeating.append(terms.ionHeating.value)
            electronHeating.append(terms.electronHeating.value)
            particleSource.append(terms.particleSource.value)
            currentSource.append(terms.currentSource.value)

            if context.includesMetadata, let metadata = terms.metadata {
                metadataEntries.append(contentsOf: metadata.entries)
            }
        }

        let metadata = context.includesMetadata
            ? SourceMetadataCollection(entries: metadataEntries)
            : nil

        return SourceTerms(
            ionHeating: context.evaluationMode.wrap(Self.sumSourceField(ionHeating)),
            electronHeating: context.evaluationMode.wrap(Self.sumSourceField(electronHeating)),
            particleSource: context.evaluationMode.wrap(Self.sumSourceField(particleSource)),
            currentSource: context.evaluationMode.wrap(Self.sumSourceField(currentSource)),
            metadata: metadata,
            validateDebugUnits: context.validatesDebugUnits
        )
    }

    private func computeSolverTerms(in context: SourceEvaluationContext) throws -> SourceTerms {
        var total = context.zeroSourceTerms()

        for (_, source) in sources {
            let terms = try source.computeTerms(in: context)
            total = total.adding(
                terms,
                evaluationMode: context.evaluationMode,
                metadata: nil,
                validateDebugUnits: false
            )
        }

        return total
    }

    private static func sumSourceField(_ arrays: [MLXArray]) -> MLXArray {
        switch arrays.count {
        case 0:
            preconditionFailure("Composite source reduction requires at least one array")
        case 1:
            return arrays[0]
        default:
            return sum(stacked(arrays, axis: 0), axis: 0)
        }
    }
}
