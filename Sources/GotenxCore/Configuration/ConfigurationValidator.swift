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

        guard time.initialDt > 0 else {
            throw ConfigurationError.invalidValue(
                key: "time.initialDt",
                value: "\(time.initialDt)",
                reason: "Initial timestep must be positive"
            )
        }

        if let adaptive = time.adaptive {
            // effectiveMinDt must be positive
            guard adaptive.effectiveMinDt > 0 else {
                throw ConfigurationError.invalidValue(
                    key: "time.adaptive.effectiveMinDt",
                    value: "\(adaptive.effectiveMinDt)",
                    reason: "Min timestep must be positive"
                )
            }

            // maxDt must be greater than effectiveMinDt
            guard adaptive.effectiveMinDt < adaptive.maxDt else {
                throw ConfigurationError.invalidValue(
                    key: "time.adaptive",
                    value: "min=\(adaptive.effectiveMinDt), max=\(adaptive.maxDt)",
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

            // Warning: initialDt should be within adaptive range
            if time.initialDt < adaptive.effectiveMinDt || time.initialDt > adaptive.maxDt {
                print("⚠️  Warning: initialDt (\(time.initialDt)s) is outside adaptive range")
                print("   Adaptive range: [\(adaptive.effectiveMinDt), \(adaptive.maxDt)]s")
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

        guard boundaries.density > 0 else {
            throw ConfigurationError.invalidValue(
                key: "boundaries.density",
                value: "\(boundaries.density)",
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
        if config.runtime.static.evolution.current {
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
        try validateDensityRange(density: boundary.density)

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
        let dt = config.time.initialDt

        // Calculate derived quantities
        let cellSpacing = mesh.minorRadius / Float(mesh.nCells)
        let volume = 2.0 * Float.pi * Float.pi * mesh.majorRadius * mesh.minorRadius * mesh.minorRadius

        // CFL condition for transport
        try validateCFLCondition(
            transport: transport,
            dt: dt,
            cellSpacing: cellSpacing
        )

        // Source term stability
        if let ecrh = sources.ecrh {
            try validateECRHStability(
                ecrh: ecrh,
                initialTemp: boundary.electronTemperature,
                density: boundary.density,
                volume: volume,
                minorRadius: mesh.minorRadius,
                dt: dt,
                cellSpacing: cellSpacing
            )
        }

        if let gasPuff = sources.gasPuff {
            try validateGasPuffStability(
                gasPuff: gasPuff,
                initialDensity: boundary.density,
                volume: volume,
                dt: dt
            )
        }

        // Timestep validity
        try validateDiffusionTimeScale(
            transport: transport,
            dt: dt,
            minorRadius: mesh.minorRadius
        )

        // Mesh resolution
        try validateMeshResolution(
            nCells: mesh.nCells,
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
        dt: Float,
        cellSpacing: Float
    ) throws {
        // Only the constant-transport model carries chi as explicit configuration
        // parameters. Self-computing models (Bohm-GyroBohm, QLKNN, density-transition)
        // derive transport coefficients at runtime, so a static CFL check from config
        // parameters does not apply — runtime adaptive timestepping handles stability.
        // (Previously this validation unconditionally required chi_ion/chi_electron,
        // contradicting its own suggestion to "use a model that computes it".)
        guard transport.modelType == .constant else {
            return
        }

        // Use optional API - explicit missing value handling
        guard let chiIon = transport.parameter("chi_ion") else {
            throw ConfigurationValidationError.missingRequiredParameter(
                parameter: "chi_ion",
                modelType: transport.modelType,
                suggestion: "Specify chi_ion in transport.parameters or use a model that computes it (e.g., bohmGyrobohm, qlknn)"
            )
        }

        guard let chiElectron = transport.parameter("chi_electron") else {
            throw ConfigurationValidationError.missingRequiredParameter(
                parameter: "chi_electron",
                modelType: transport.modelType,
                suggestion: "Specify chi_electron in transport.parameters or use a model that computes it (e.g., bohmGyrobohm, qlknn)"
            )
        }

        // particle_diffusivity is optional for some models
        let particleDiff = transport.parameter("particle_diffusivity", default: 0.0)

        // Validation only - no default provisioning
        if chiIon <= 0 {
            throw ConfigurationValidationError.invalidParameter(
                parameter: "chi_ion",
                value: chiIon,
                reason: "Must be positive"
            )
        }

        if chiElectron <= 0 {
            throw ConfigurationValidationError.invalidParameter(
                parameter: "chi_electron",
                value: chiElectron,
                reason: "Must be positive"
            )
        }

        if particleDiff < 0 {
            throw ConfigurationValidationError.invalidParameter(
                parameter: "particle_diffusivity",
                value: particleDiff,
                reason: "Must be non-negative"
            )
        }

        // Compute CFL numbers
        let CFL_ion = chiIon * dt / (cellSpacing * cellSpacing)
        let CFL_electron = chiElectron * dt / (cellSpacing * cellSpacing)
        let CFL_particle = particleDiff * dt / (cellSpacing * cellSpacing)

        if CFL_ion > 0.5 {
            throw ConfigurationValidationError.cflViolation(
                parameter: "chi_ion",
                cfl: CFL_ion,
                limit: 0.5,
                suggestion: "Reduce chi_ion to \(chiIon * 0.5 / CFL_ion) m²/s or decrease dt to \(dt * 0.5 / CFL_ion) s"
            )
        }

        if CFL_electron > 0.5 {
            throw ConfigurationValidationError.cflViolation(
                parameter: "chi_electron",
                cfl: CFL_electron,
                limit: 0.5,
                suggestion: "Reduce chi_electron to \(chiElectron * 0.5 / CFL_electron) m²/s or decrease dt to \(dt * 0.5 / CFL_electron) s"
            )
        }

        if CFL_particle > 0.5 {
            throw ConfigurationValidationError.cflViolation(
                parameter: "particle_diffusivity",
                cfl: CFL_particle,
                limit: 0.5,
                suggestion: "Reduce particle_diffusivity to \(particleDiff * 0.5 / CFL_particle) m²/s or decrease dt to \(dt * 0.5 / CFL_particle) s"
            )
        }
    }

    private static func validateECRHStability(
        ecrh: ECRHConfig,
        initialTemp: Float,
        density: Float,
        volume: Float,
        minorRadius: Float,
        dt: Float,
        cellSpacing: Float
    ) throws {
        // Estimate peak power density (Gaussian profile)
        let sigma = ecrh.depositionWidth / 3.0
        let rho_dep = ecrh.depositionRho
        let peakRadiusFraction = sigma / minorRadius
        let peakVolumeFraction = max(0.1, 2.0 * rho_dep * peakRadiusFraction)
        let peakPowerDensity = ecrh.totalPower / (volume * peakVolumeFraction)

        // Estimate temperature change per timestep
        // Energy equation: (3/2) n_e dT/dt = Q/e → dT/dt = (2/3) Q/(n_e e)
        let elementaryCharge: Float = 1.602e-19
        let tempChangeRate_eV = (2.0/3.0) * peakPowerDensity / (density * elementaryCharge)
        let tempChange = tempChangeRate_eV * dt
        let changeRatio = tempChange / initialTemp

        if changeRatio > 0.5 {
            throw ConfigurationValidationError.unstableTimestep(
                parameter: "ECRH heating",
                changeRatio: changeRatio,
                suggestion: "Reduce ECRH totalPower to \(ecrh.totalPower * 0.5 / changeRatio) W or decrease dt to \(dt * 0.5 / changeRatio) s"
            )
        }

        // Check deposition width vs mesh resolution
        let minWidthForResolution = 3.0 * cellSpacing
        if ecrh.depositionWidth < minWidthForResolution {
            throw ConfigurationValidationError.insufficientResolution(
                parameter: "ECRH depositionWidth",
                value: ecrh.depositionWidth,
                minimum: minWidthForResolution,
                suggestion: "Increase depositionWidth to \(minWidthForResolution) or increase nCells to \(Int(3.0 * minorRadius / ecrh.depositionWidth))"
            )
        }
    }

    private static func validateGasPuffStability(
        gasPuff: GasPuffConfig,
        initialDensity: Float,
        volume: Float,
        dt: Float
    ) throws {
        // Estimate density change per timestep
        let particlesAdded = gasPuff.puffRate * dt
        let densityChange = particlesAdded / volume
        let changeRatio = densityChange / initialDensity

        if changeRatio > 0.2 {
            throw ConfigurationValidationError.unstableTimestep(
                parameter: "Gas puff",
                changeRatio: changeRatio,
                suggestion: "Reduce puffRate to \(gasPuff.puffRate * 0.2 / changeRatio) particles/s or decrease dt to \(dt * 0.2 / changeRatio) s"
            )
        }
    }

    private static func validateDiffusionTimeScale(
        transport: TransportConfig,
        dt: Float,
        minorRadius: Float
    ) throws {
        let chi_max = max(
            transport.parameters["chi_ion"] ?? 1.0,
            transport.parameters["chi_electron"] ?? 1.0
        )

        let tau_diffusion = minorRadius * minorRadius / chi_max

        if dt > tau_diffusion {
            throw ConfigurationValidationError.timestepTooLarge(
                dt: dt,
                timeScale: tau_diffusion,
                suggestion: "Decrease dt to \(tau_diffusion / 10) s"
            )
        }
    }

    private static func validateMeshResolution(
        nCells: Int,
        initialProfile: InitialProfileConfig
    ) throws {
        if nCells < 50 {
            throw ConfigurationValidationError.insufficientMeshResolution(
                nCells: nCells,
                minimum: 50,
                suggestion: "Increase nCells to at least 50"
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
                density: config.runtime.dynamic.boundaries.density
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
            let rho_dep = ecrh.depositionRho
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

    private static func collectTransportWarnings(_ config: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        var warnings: [ConfigurationValidationWarning] = []
        // Transport-related warnings (none defined yet)
        return warnings
    }

    private static func collectBoundaryWarnings(_ config: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        var warnings: [ConfigurationValidationWarning] = []
        let boundary = config.runtime.dynamic.boundaries
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
        let dt = config.time.initialDt
        let transport = config.runtime.dynamic.transport
        let mesh = config.runtime.static.mesh

        // Calculate CFL-limited maximum timestep
        let chi_max = max(
            transport.parameters["chi_ion"] ?? 1.0,
            transport.parameters["chi_electron"] ?? 1.0
        )
        let cellSpacing = mesh.minorRadius / Float(mesh.nCells)
        let dt_cfl_max = 0.5 * cellSpacing * cellSpacing / chi_max

        // Calculate diffusion time scale
        let tau_diffusion = mesh.minorRadius * mesh.minorRadius / chi_max

        // Only warn about small timestep if it's much smaller than CFL limit
        // (i.e., not limited by CFL condition)
        if dt < dt_cfl_max / 5.0 && dt < tau_diffusion / 200 {
            warnings.append(.timestepTooSmall(
                dt: dt,
                timeScale: tau_diffusion,
                suggestion: "Consider increasing dt to \(min(dt_cfl_max * 0.9, tau_diffusion / 10)) s for better efficiency (CFL limit: \(dt_cfl_max) s)"
            ))
        }

        // Warn about extremely small timesteps (< 1 μs)
        if dt < 1e-6 {
            warnings.append(.timestepTooSmall(
                dt: dt,
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
        if mesh.nCells > 500 {
            warnings.append(.excessiveMeshResolution(
                nCells: mesh.nCells,
                maximum: 500,
                suggestion: "Consider reducing nCells to ~200 for better performance"
            ))
        }

        // Gradient resolution warning
        // Use temperature exponent as the profile steepness indicator
        let exponent = initialProfile.temperatureExponent
        if exponent > 1.0 {
            let recommendedCells = max(50, Int(3.0 * exponent))
            if mesh.nCells < recommendedCells {
                warnings.append(.insufficientGradientResolution(
                    nCells: mesh.nCells,
                    recommended: recommendedCells,
                    profileExponent: exponent,
                    suggestion: "Increase nCells to \(recommendedCells) to resolve gradient scale length L_T ~ a/\(Int(exponent))"
                ))
            }
        }

        return warnings
    }

    private static func collectModelWarnings(_ config: SimulationConfiguration) -> [ConfigurationValidationWarning] {
        var warnings: [ConfigurationValidationWarning] = []
        // Model-specific warnings can be added here
        return warnings
    }
}
