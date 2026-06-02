// ConfigurationValidator.swift
// Configuration validator with physics-aware checks

import Foundation

/// Configuration validator with physics-aware checks
public struct ConfigurationValidator {

    /// Validate complete configuration
    public static func validate(_ config: SimulationConfiguration) throws {
        // Validate individual components
        // Catch physics warnings (advisory only) but propagate hard errors
        do {
            try config.runtime.static.mesh.validate()
        } catch ConfigurationError.physicsWarning(let key, let value, let reason) {
            print("⚠️  Warning for '\(key)': \(value). \(reason)")
        } catch {
            throw error  // Re-throw hard errors
        }

        do {
            try config.runtime.static.scheme.validate()
        } catch ConfigurationError.physicsWarning(let key, let value, let reason) {
            print("⚠️  Warning for '\(key)': \(value). \(reason)")
        } catch {
            throw error  // Re-throw hard errors
        }

        // These validations have hard errors only (no warnings)
        try validateTimeRange(config.time)
        try validateBoundaries(config.runtime.dynamic.boundaries)
        try config.runtime.dynamic.transport.validateParameterKeys()
        try validateSources(config.runtime.dynamic.sources)

        // Cross-component validation
        try validateConsistency(config)

        // Phase 1: Physical range validation
        try validatePhysicalRanges(config)

        // Phase 2: Numerical stability validation
        try validateNumericalStability(config)

        // Phase 3: Model-specific validation
        try validateModelConstraints(config)
    }

    /// Validate and collect all warnings (non-throwing)
    ///
    /// - Parameter config: Simulation configuration to validate
    /// - Returns: Array of validation warnings
    public static func collectWarnings(_ config: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        var warnings: [ConfigurationValidationWarning] = []

        // Collect warnings from each validation category
        warnings.append(contentsOf: collectSourceWarnings(config))
        warnings.append(contentsOf: collectTransportWarnings(config))
        warnings.append(contentsOf: collectBoundaryWarnings(config))
        warnings.append(contentsOf: collectTimestepWarnings(config))
        warnings.append(contentsOf: collectMeshWarnings(config))
        warnings.append(contentsOf: collectModelWarnings(config))

        return warnings
    }

    /// Validate time range
    private static func validateTimeRange(_ time: TimeConfiguration) throws {
        guard time.end > time.start else {
            throw ConfigurationError.invalidValue(
                key: "time.end",
                value: "\(time.end)",
                reason: "End time must be greater than start time"
            )
        }

        guard time.initialTimeStep > 0 else {
            throw ConfigurationError.invalidValue(
                key: "time.initialTimeStep",
                value: "\(time.initialTimeStep)",
                reason: "Initial timestep must be positive"
            )
        }

        if let adaptive = time.adaptive {
            // effectiveMinimumTimeStep must be positive
            guard adaptive.effectiveMinimumTimeStep > 0 else {
                throw ConfigurationError.invalidValue(
                    key: "time.adaptive.effectiveMinimumTimeStep",
                    value: "\(adaptive.effectiveMinimumTimeStep)",
                    reason: "Min timestep must be positive"
                )
            }

            // maximumTimeStep must be greater than effectiveMinimumTimeStep
            guard adaptive.effectiveMinimumTimeStep < adaptive.maximumTimeStep else {
                throw ConfigurationError.invalidValue(
                    key: "time.adaptive",
                    value: "min=\(adaptive.effectiveMinimumTimeStep), max=\(adaptive.maximumTimeStep)",
                    reason: "Min timestep must be less than max timestep"
                )
            }

            // safetyFactor must be in (0, 1]
            guard adaptive.safetyFactor > 0 && adaptive.safetyFactor <= 1.0 else {
                throw ConfigurationError.invalidValue(
                    key: "time.adaptive.safetyFactor",
                    value: "\(adaptive.safetyFactor)",
                    reason: "Safety factor must be in (0, 1]"
                )
            }

            // Warning: initialTimeStep should be within adaptive range
            if time.initialTimeStep < adaptive.effectiveMinimumTimeStep || time.initialTimeStep > adaptive.maximumTimeStep {
                print("⚠️  Warning: initialTimeStep (\(time.initialTimeStep)s) is outside adaptive range")
                print("   Adaptive range: [\(adaptive.effectiveMinimumTimeStep), \(adaptive.maximumTimeStep)]s")
                print("   Timestep will be clamped to this range")
            }
        }
    }

    /// Validate boundary conditions
    private static func validateBoundaries(_ boundaries: BoundaryConfig) throws {
        guard boundaries.ionTemperature > 0 else {
            throw ConfigurationError.invalidValue(
                key: "boundaries.ionTemperature",
                value: "\(boundaries.ionTemperature)",
                reason: "Temperature must be positive"
            )
        }

        guard boundaries.electronTemperature > 0 else {
            throw ConfigurationError.invalidValue(
                key: "boundaries.electronTemperature",
                value: "\(boundaries.electronTemperature)",
                reason: "Temperature must be positive"
            )
        }

        guard boundaries.electronDensity > 0 else {
            throw ConfigurationError.invalidValue(
                key: "boundaries.electronDensity",
                value: "\(boundaries.electronDensity)",
                reason: "Density must be positive"
            )
        }
    }

    /// Validate source configuration
    private static func validateSources(_ sources: SourcesConfig) throws {
        if let fusion = sources.fusionConfig {
            let totalFraction = fusion.deuteriumFraction + fusion.tritiumFraction
            // Use physical threshold for fuel fraction validation (1e-4, not hardcoded 1e-6)
            guard abs(totalFraction - 1.0) < PhysicalThresholds.default.fuelFractionTolerance else {
                throw ConfigurationError.invalidValue(
                    key: "sources.fusionConfig.fractions",
                    value: "D=\(fusion.deuteriumFraction), T=\(fusion.tritiumFraction)",
                    reason: "Fuel fractions must sum to 1.0"
                )
            }

            guard fusion.dilution > 0 && fusion.dilution <= 1.0 else {
                throw ConfigurationError.invalidValue(
                    key: "sources.fusionConfig.dilution",
                    value: "\(fusion.dilution)",
                    reason: "Dilution must be in (0, 1]"
                )
            }
        }
    }

    /// Cross-component consistency checks
    private static func validateConsistency(_ config: SimulationConfiguration) throws {
        // Check: If current evolution is enabled, appropriate sources must be configured
        if config.runtime.static.evolution.poloidalFlux {
            guard config.runtime.dynamic.sources.ohmicHeating else {
                throw ConfigurationError.inconsistency(
                    reason: "Current evolution requires Ohmic heating to be enabled"
                )
            }
        }

        // Note: CFL and timestep stability are now validated in Phase 2
        // (validateNumericalStability) using actual transport coefficients
    }

    // MARK: - Phase 1: Physical Range Validation

    private static func validatePhysicalRanges(_ config: SimulationConfiguration) throws {
        let boundary = config.runtime.dynamic.boundaries
        let mesh = config.runtime.static.mesh

        // Temperature range
        try validateTemperatureRange(
            ionTemp: boundary.ionTemperature,
            electronTemp: boundary.electronTemperature
        )

        // Density range
        try validateDensityRange(density: boundary.electronDensity)

        // Magnetic field range
        try validateMagneticFieldRange(toroidalField: mesh.toroidalField)

        // Geometry range
        try validateGeometryRange(
            majorRadius: mesh.majorRadius,
            minorRadius: mesh.minorRadius
        )
    }

    private static func validateTemperatureRange(ionTemp: Float, electronTemp: Float) throws {
        if ionTemp < 1.0 || ionTemp > 100_000 {
            throw ConfigurationValidationError.outOfPhysicalRange(
                parameter: "ionTemperature",
                value: ionTemp,
                range: (1.0, 100_000),
                unit: "eV"
            )
        }

        if electronTemp < 1.0 || electronTemp > 100_000 {
            throw ConfigurationValidationError.outOfPhysicalRange(
                parameter: "electronTemperature",
                value: electronTemp,
                range: (1.0, 100_000),
                unit: "eV"
            )
        }
    }

    private static func validateDensityRange(density: Float) throws {
        if density < 1e17 || density > 1e21 {
            throw ConfigurationValidationError.outOfPhysicalRange(
                parameter: "density",
                value: density,
                range: (1e17, 1e21),
                unit: "m⁻³"
            )
        }
    }

    private static func validateMagneticFieldRange(toroidalField: Float) throws {
        if toroidalField < 0.5 || toroidalField > 15.0 {
            throw ConfigurationValidationError.outOfPhysicalRange(
                parameter: "toroidalField",
                value: toroidalField,
                range: (0.5, 15.0),
                unit: "T"
            )
        }
    }

    private static func validateGeometryRange(majorRadius: Float, minorRadius: Float) throws {
        if majorRadius < 0.5 || majorRadius > 10.0 {
            throw ConfigurationValidationError.outOfPhysicalRange(
                parameter: "majorRadius",
                value: majorRadius,
                range: (0.5, 10.0),
                unit: "m"
            )
        }

        if minorRadius < 0.2 || minorRadius > 3.0 {
            throw ConfigurationValidationError.outOfPhysicalRange(
                parameter: "minorRadius",
                value: minorRadius,
                range: (0.2, 3.0),
                unit: "m"
            )
        }

        let aspectRatio = minorRadius / majorRadius
        if aspectRatio > 0.5 {
            throw ConfigurationValidationError.invalidGeometry(
                parameter: "aspectRatio",
                value: aspectRatio,
                limit: 0.5,
                suggestion: "Reduce minorRadius or increase majorRadius"
            )
        }
    }

    // MARK: - Phase 2: Numerical Stability Validation

    private static func validateNumericalStability(_ config: SimulationConfiguration) throws {
        let mesh = config.runtime.static.mesh
        let transport = config.runtime.dynamic.transport
        let sources = config.runtime.dynamic.sources
        let boundary = config.runtime.dynamic.boundaries
        let timeStep = config.time.initialTimeStep

        // Calculate derived quantities
        let cellSpacing = mesh.minorRadius / Float(mesh.cellCount)
        let volume = 2.0 * Float.pi * Float.pi * mesh.majorRadius * mesh.minorRadius * mesh.minorRadius

        // CFL condition for transport
        try validateCFLCondition(
            transport: transport,
            timeStep: timeStep,
            cellSpacing: cellSpacing
        )

        // Source term stability
        if let ecrh = sources.ecrh {
            try validateECRHStability(
                ecrh: ecrh,
                initialTemp: boundary.electronTemperature,
                density: boundary.electronDensity,
                volume: volume,
                minorRadius: mesh.minorRadius,
                timeStep: timeStep,
                cellSpacing: cellSpacing
            )
        }

        if let gasPuff = sources.gasPuff {
            try validateGasPuffStability(
                gasPuff: gasPuff,
                initialDensity: boundary.electronDensity,
                volume: volume,
                timeStep: timeStep
            )
        }

        // Timestep validity
        try validateDiffusionTimeScale(
            transport: transport,
            timeStep: timeStep,
            minorRadius: mesh.minorRadius
        )

        // Mesh resolution
        try validateMeshResolution(
            cellCount: mesh.cellCount,
            initialProfile: config.runtime.dynamic.initialProfile
        )

        // Boundary consistency
        try validateTemperatureBoundaryConsistency(
            boundary: boundary,
            initialProfile: config.runtime.dynamic.initialProfile
        )
    }

    private static func validateCFLCondition(
        transport: TransportConfig,
        timeStep: Float,
        cellSpacing: Float
    ) throws {
        // Only the constant-transport model carries heat diffusivity as explicit configuration
        // parameters. Self-computing models (Bohm-GyroBohm, QLKNN, density-transition)
        // derive transport coefficients at runtime, so a static CFL check from config
        // parameters does not apply — runtime adaptive timestepping handles stability.
        guard transport.modelType == .constant else {
            return
        }

        // Use optional API - explicit missing value handling
        guard let ionHeatDiffusivity = transport.parameter("ionHeatDiffusivity") else {
            throw ConfigurationValidationError.missingRequiredParameter(
                parameter: "ionHeatDiffusivity",
                modelType: transport.modelType,
                suggestion: "Specify ionHeatDiffusivity in transport.parameters or use a model that computes it (e.g., bohmGyrobohm, qlknn)"
            )
        }

        guard let electronHeatDiffusivity = transport.parameter("electronHeatDiffusivity") else {
            throw ConfigurationValidationError.missingRequiredParameter(
                parameter: "electronHeatDiffusivity",
                modelType: transport.modelType,
                suggestion: "Specify electronHeatDiffusivity in transport.parameters or use a model that computes it (e.g., bohmGyrobohm, qlknn)"
            )
        }

        let particleDiffusivity = transport.parameter("particleDiffusivity", default: 0.0)

        // Validation only - no default provisioning
        if ionHeatDiffusivity <= 0 {
            throw ConfigurationValidationError.invalidParameter(
                parameter: "ionHeatDiffusivity",
                value: ionHeatDiffusivity,
                reason: "Must be positive"
            )
        }

        if electronHeatDiffusivity <= 0 {
            throw ConfigurationValidationError.invalidParameter(
                parameter: "electronHeatDiffusivity",
                value: electronHeatDiffusivity,
                reason: "Must be positive"
            )
        }

        if particleDiffusivity < 0 {
            throw ConfigurationValidationError.invalidParameter(
                parameter: "particleDiffusivity",
                value: particleDiffusivity,
                reason: "Must be non-negative"
            )
        }

        // Compute CFL numbers
        let ionCFL = ionHeatDiffusivity * timeStep / (cellSpacing * cellSpacing)
        let electronCFL = electronHeatDiffusivity * timeStep / (cellSpacing * cellSpacing)
        let particleCFL = particleDiffusivity * timeStep / (cellSpacing * cellSpacing)

        if ionCFL > 0.5 {
            throw ConfigurationValidationError.cflViolation(
                parameter: "ionHeatDiffusivity",
                cfl: ionCFL,
                limit: 0.5,
                suggestion: "Reduce ionHeatDiffusivity to \(ionHeatDiffusivity * 0.5 / ionCFL) m²/s or decrease timeStep to \(timeStep * 0.5 / ionCFL) s"
            )
        }

        if electronCFL > 0.5 {
            throw ConfigurationValidationError.cflViolation(
                parameter: "electronHeatDiffusivity",
                cfl: electronCFL,
                limit: 0.5,
                suggestion: "Reduce electronHeatDiffusivity to \(electronHeatDiffusivity * 0.5 / electronCFL) m²/s or decrease timeStep to \(timeStep * 0.5 / electronCFL) s"
            )
        }

        if particleCFL > 0.5 {
            throw ConfigurationValidationError.cflViolation(
                parameter: "particleDiffusivity",
                cfl: particleCFL,
                limit: 0.5,
                suggestion: "Reduce particleDiffusivity to \(particleDiffusivity * 0.5 / particleCFL) m²/s or decrease timeStep to \(timeStep * 0.5 / particleCFL) s"
            )
        }
    }

    private static func validateECRHStability(
        ecrh: ECRHConfig,
        initialTemp: Float,
        density: Float,
        volume: Float,
        minorRadius: Float,
        timeStep: Float,
        cellSpacing: Float
    ) throws {
        // Estimate peak power density (Gaussian profile)
        let sigma = ecrh.depositionWidth / 3.0
        let rho_dep = ecrh.normalizedDepositionRadius
        let peakRadiusFraction = sigma / minorRadius
        let peakVolumeFraction = max(0.1, 2.0 * rho_dep * peakRadiusFraction)
        let peakPowerDensity = ecrh.totalPower / (volume * peakVolumeFraction)

        // Estimate temperature change per timestep
        // Energy equation: (3/2) n_e dT/timeStep = Q/e → dT/timeStep = (2/3) Q/(n_e e)
        let elementaryCharge: Float = 1.602e-19
        let tempChangeRate_eV = (2.0/3.0) * peakPowerDensity / (density * elementaryCharge)
        let tempChange = tempChangeRate_eV * timeStep
        let changeRatio = tempChange / initialTemp

        if changeRatio > 0.5 {
            throw ConfigurationValidationError.unstableTimestep(
                parameter: "ECRH heating",
                changeRatio: changeRatio,
                suggestion: "Reduce ECRH totalPower to \(ecrh.totalPower * 0.5 / changeRatio) W or decrease timeStep to \(timeStep * 0.5 / changeRatio) s"
            )
        }

        // Check deposition width vs mesh resolution
        let minWidthForResolution = 3.0 * cellSpacing
        if ecrh.depositionWidth < minWidthForResolution {
            throw ConfigurationValidationError.insufficientResolution(
                parameter: "ECRH depositionWidth",
                value: ecrh.depositionWidth,
                minimum: minWidthForResolution,
                suggestion: "Increase depositionWidth to \(minWidthForResolution) or increase cellCount to \(Int(3.0 * minorRadius / ecrh.depositionWidth))"
            )
        }
    }

    private static func validateGasPuffStability(
        gasPuff: GasPuffConfig,
        initialDensity: Float,
        volume: Float,
        timeStep: Float
    ) throws {
        // Estimate density change per timestep
        let particlesAdded = gasPuff.puffRate * timeStep
        let densityChange = particlesAdded / volume
        let changeRatio = densityChange / initialDensity

        if changeRatio > 0.2 {
            throw ConfigurationValidationError.unstableTimestep(
                parameter: "Gas puff",
                changeRatio: changeRatio,
                suggestion: "Reduce puffRate to \(gasPuff.puffRate * 0.2 / changeRatio) particles/s or decrease timeStep to \(timeStep * 0.2 / changeRatio) s"
            )
        }
    }

    private static func validateDiffusionTimeScale(
        transport: TransportConfig,
        timeStep: Float,
        minorRadius: Float
    ) throws {
        guard transport.modelType == .constant else {
            return
        }

        let maximumHeatDiffusivity = max(
            try transport.requireParameter("ionHeatDiffusivity"),
            try transport.requireParameter("electronHeatDiffusivity")
        )

        let diffusionTimeScale = minorRadius * minorRadius / maximumHeatDiffusivity

        if timeStep > diffusionTimeScale {
            throw ConfigurationValidationError.timestepTooLarge(
                timeStep: timeStep,
                timeScale: diffusionTimeScale,
                suggestion: "Decrease timeStep to \(diffusionTimeScale / 10) s"
            )
        }
    }

    private static func validateMeshResolution(
        cellCount: Int,
        initialProfile: InitialProfileConfig
    ) throws {
        if cellCount < 50 {
            throw ConfigurationValidationError.insufficientMeshResolution(
                cellCount: cellCount,
                minimum: 50,
                suggestion: "Increase cellCount to at least 50"
            )
        }
    }

    private static func validateTemperatureBoundaryConsistency(
        boundary: BoundaryConfig,
        initialProfile: InitialProfileConfig
    ) throws {
        // Check if profile is peaked (ratio > 1.0)
        if initialProfile.temperaturePeakRatio > 1.0 {
            let T_core_ion = boundary.ionTemperature * initialProfile.temperaturePeakRatio
            let T_core_electron = boundary.electronTemperature * initialProfile.temperaturePeakRatio

            // Core temperature should be higher than boundary temperature when peaked
            if T_core_ion < boundary.ionTemperature {
                throw ConfigurationValidationError.inconsistentBoundary(
                    parameter: "ionTemperature",
                    coreValue: T_core_ion,
                    boundaryValue: boundary.ionTemperature,
                    suggestion: "Increase temperaturePeakRatio to > 1.0 or use flat initial profile"
                )
            }

            if T_core_electron < boundary.electronTemperature {
                throw ConfigurationValidationError.inconsistentBoundary(
                    parameter: "electronTemperature",
                    coreValue: T_core_electron,
                    boundaryValue: boundary.electronTemperature,
                    suggestion: "Increase temperaturePeakRatio to > 1.0 or use flat initial profile"
                )
            }
        }
    }

    // MARK: - Phase 3: Model-Specific Validation

    private static func validateModelConstraints(_ config: SimulationConfiguration) throws {
        // QLKNN training range
        if config.runtime.dynamic.transport.modelType == .qlknn {
            try validateQLKNNRange(
                electronTemp: config.runtime.dynamic.boundaries.electronTemperature,
                density: config.runtime.dynamic.boundaries.electronDensity
            )
        }

        // Fusion power conditions
        if config.runtime.dynamic.sources.fusionPower,
           let fusionConfig = config.runtime.dynamic.sources.fusionConfig {
            try validateFusionConditions(
                ionTemp: config.runtime.dynamic.boundaries.ionTemperature,
                fusionConfig: fusionConfig
            )
        }
    }

    private static func validateQLKNNRange(electronTemp: Float, density: Float) throws {
        if electronTemp < 500.0 {
            throw ConfigurationValidationError.outOfPhysicalRange(
                parameter: "electronTemperature for QLKNN",
                value: electronTemp,
                range: (500.0, 20_000),
                unit: "eV"
            )
        }

        if density < 1e19 || density > 1e20 {
            throw ConfigurationValidationError.outOfPhysicalRange(
                parameter: "density for QLKNN",
                value: density,
                range: (1e19, 1e20),
                unit: "m⁻³"
            )
        }
    }

    private static func validateFusionConditions(
        ionTemp: Float,
        fusionConfig: FusionConfig
    ) throws {
        let totalFuelFraction = fusionConfig.deuteriumFraction + fusionConfig.tritiumFraction
        if abs(totalFuelFraction - 1.0) > 0.01 {
            throw ConfigurationValidationError.invalidFuelMix(
                dFraction: fusionConfig.deuteriumFraction,
                tFraction: fusionConfig.tritiumFraction,
                suggestion: "D+T fractions must sum to 1.0"
            )
        }
    }

    // MARK: - Warning Collection

    private static func collectSourceWarnings(_ config: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        var warnings: [ConfigurationValidationWarning] = []
        let sources = config.runtime.dynamic.sources
        let boundary = config.runtime.dynamic.boundaries
        let mesh = config.runtime.static.mesh

        // ECRH power density warning
        if let ecrh = sources.ecrh {
            let volume = 2.0 * Float.pi * Float.pi * mesh.majorRadius * mesh.minorRadius * mesh.minorRadius
            let minorRadius = mesh.minorRadius
            let sigma = ecrh.depositionWidth / 3.0
            let rho_dep = ecrh.normalizedDepositionRadius
            let peakRadiusFraction = sigma / minorRadius
            let peakVolumeFraction = max(0.1, 2.0 * rho_dep * peakRadiusFraction)
            let peakPowerDensity = ecrh.totalPower / (volume * peakVolumeFraction)
            let peakPowerDensity_MW = peakPowerDensity / 1e6

            if peakPowerDensity_MW > 100.0 {
                warnings.append(.highPowerDensity(
                    value: peakPowerDensity_MW,
                    limit: 100.0,
                    suggestion: "Reduce ECRH totalPower to \(ecrh.totalPower * 100.0 / peakPowerDensity_MW) W"
                ))
            }
        }

        // Gas puff rate warning
        if let gasPuff = sources.gasPuff {
            if gasPuff.puffRate > 1e22 {
                warnings.append(.highPuffRate(
                    value: gasPuff.puffRate,
                    limit: 1e22,
                    suggestion: "Review gas puff configuration"
                ))
            }
        }

        // Fusion power warning
        if sources.fusionPower {
            if boundary.ionTemperature < 1000.0 {
                warnings.append(.negligibleFusionPower(
                    temperature: boundary.ionTemperature,
                    threshold: 1000.0,
                    suggestion: "Fusion power is negligible below 1 keV. Consider disabling fusion source."
                ))
            }
        }

        return warnings
    }

    private static func collectTransportWarnings(_: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        // Transport-related warnings (none defined yet)
        return []
    }

    private static func collectBoundaryWarnings(_ config: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        var warnings: [ConfigurationValidationWarning] = []
        let initialProfile = config.runtime.dynamic.initialProfile

        // Flat profile warning
        if initialProfile.temperaturePeakRatio < 1.2 && initialProfile.temperaturePeakRatio > 1.0 {
            warnings.append(.flatProfile(
                parameter: "temperature",
                coreFactor: initialProfile.temperaturePeakRatio,
                suggestion: "Consider increasing temperaturePeakRatio to > 1.5 for more realistic profile"
            ))
        }

        return warnings
    }

    private static func collectTimestepWarnings(_ config: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        var warnings: [ConfigurationValidationWarning] = []
        let timeStep = config.time.initialTimeStep
        let transport = config.runtime.dynamic.transport
        let mesh = config.runtime.static.mesh

        guard transport.modelType == .constant,
              let ionHeatDiffusivity = transport.parameter("ionHeatDiffusivity"),
              let electronHeatDiffusivity = transport.parameter("electronHeatDiffusivity") else {
            return warnings
        }

        // Calculate CFL-limited maximum timestep
        let maximumHeatDiffusivity = max(ionHeatDiffusivity, electronHeatDiffusivity)
        let cellSpacing = mesh.minorRadius / Float(mesh.cellCount)
        let maximumCFLTimeStep = 0.5 * cellSpacing * cellSpacing / maximumHeatDiffusivity

        // Calculate diffusion time scale
        let diffusionTimeScale = mesh.minorRadius * mesh.minorRadius / maximumHeatDiffusivity

        // Only warn about small timestep if it's much smaller than CFL limit
        // (i.e., not limited by CFL condition)
        if timeStep < maximumCFLTimeStep / 5.0 && timeStep < diffusionTimeScale / 200 {
            warnings.append(.timestepTooSmall(
                timeStep: timeStep,
                timeScale: diffusionTimeScale,
                suggestion: "Consider increasing timeStep to \(min(maximumCFLTimeStep * 0.9, diffusionTimeScale / 10)) s for better efficiency (CFL limit: \(maximumCFLTimeStep) s)"
            ))
        }

        // Warn about extremely small timesteps (< 1 μs)
        if timeStep < 1e-6 {
            warnings.append(.timestepTooSmall(
                timeStep: timeStep,
                timeScale: 1e-6,
                suggestion: "Timestep < 1 μs may cause excessive computation time"
            ))
        }

        return warnings
    }

    private static func collectMeshWarnings(_ config: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        var warnings: [ConfigurationValidationWarning] = []
        let mesh = config.runtime.static.mesh
        let initialProfile = config.runtime.dynamic.initialProfile

        // Excessive mesh resolution
        if mesh.cellCount > 500 {
            warnings.append(.excessiveMeshResolution(
                cellCount: mesh.cellCount,
                maximum: 500,
                suggestion: "Consider reducing cellCount to ~200 for better performance"
            ))
        }

        // Gradient resolution warning
        // Use temperature exponent as the profile steepness indicator
        let exponent = initialProfile.temperatureExponent
        if exponent > 1.0 {
            let recommendedCells = max(50, Int(3.0 * exponent))
            if mesh.cellCount < recommendedCells {
                warnings.append(.insufficientGradientResolution(
                    cellCount: mesh.cellCount,
                    recommended: recommendedCells,
                    profileExponent: exponent,
                    suggestion: "Increase cellCount to \(recommendedCells) to resolve gradient scale length L_T ~ a/\(Int(exponent))"
                ))
            }
        }

        return warnings
    }

    private static func collectModelWarnings(_: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        // Model-specific warnings can be added here
        return []
    }
}
