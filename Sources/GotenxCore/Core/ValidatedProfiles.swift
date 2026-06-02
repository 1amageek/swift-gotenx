import MLX

/// Validated wrapper for CoreProfiles.
///
/// Ensures all physics models receive valid input data with critical checks:
/// - All values finite (no NaN, no Inf)
/// - Temperatures positive (T > 0 eV)
/// - Density positive (n > 0 m⁻³)
/// - Profile shapes consistent across evolved variables
public struct ValidatedProfiles {
    /// Validated ion temperature [eV]
    public let ionTemperature: EvaluatedArray

    /// Validated electron temperature [eV]
    public let electronTemperature: EvaluatedArray

    /// Validated electron density [m⁻³]
    public let electronDensity: EvaluatedArray

    /// Validated normalized poloidal flux [0, 1]
    public let poloidalFlux: EvaluatedArray

    /// Private initializer - only accessible after validation.
    private init(
        ionTemperature: EvaluatedArray,
        electronTemperature: EvaluatedArray,
        electronDensity: EvaluatedArray,
        poloidalFlux: EvaluatedArray
    ) {
        self.ionTemperature = ionTemperature
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity
        self.poloidalFlux = poloidalFlux
    }

    /// Validate profiles and return a typed wrapper.
    ///
    /// - Throws: `NumericalValidationError` when shape, finite, or positivity checks fail.
    public static func validate(_ profiles: CoreProfiles) throws -> ValidatedProfiles {
        try profiles.validateNumerics()
        return ValidatedProfiles(
            ionTemperature: profiles.ionTemperature,
            electronTemperature: profiles.electronTemperature,
            electronDensity: profiles.electronDensity,
            poloidalFlux: profiles.poloidalFlux
        )
    }

    /// Convert back to CoreProfiles (for solver interface compatibility)
    ///
    /// This allows validated profiles to be passed to interfaces expecting CoreProfiles.
    ///
    /// - Returns: CoreProfiles with same data
    public func toCoreProfiles() -> CoreProfiles {
        return CoreProfiles(
            ionTemperature: ionTemperature,
            electronTemperature: electronTemperature,
            electronDensity: electronDensity,
            poloidalFlux: poloidalFlux
        )
    }
}
