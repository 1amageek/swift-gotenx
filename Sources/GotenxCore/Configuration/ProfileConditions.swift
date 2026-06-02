import Foundation

// MARK: - Profile Specification

/// Profile specification
public enum ProfileSpec: Sendable, Codable, Equatable {
    /// Constant value
    case constant(Float)

    /// Linear profile (core, edge)
    case linear(core: Float, edge: Float)

    /// Parabolic profile (peak, edge, exponent)
    case parabolic(peak: Float, edge: Float, exponent: Float)

    /// Custom array of values
    case array([Float])

    /// Evaluate profile at normalized radial coordinate
    ///
    /// - Parameter normalizedRadius: Normalized radial coordinate [0, 1] (0=core, 1=edge)
    /// - Returns: Profile value at normalized radius
    public func evaluate(at normalizedRadius: Float) -> Float {
        // Clamp input to valid range
        let clampedRadius = max(0.0, min(1.0, normalizedRadius))

        switch self {
        case .constant(let value):
            return value

        case .linear(let core, let edge):
            // Linear interpolation: f(r) = core + (edge - core) * r
            return core + (edge - core) * clampedRadius

        case .parabolic(let peak, let edge, let exponent):
            // Parabolic profile: f(r) = edge + (peak - edge) * (1 - r^2)^exponent
            let factor = pow(1.0 - clampedRadius * clampedRadius, exponent)
            return edge + (peak - edge) * factor

        case .array(let values):
            // Linear interpolation between array values
            guard !values.isEmpty else { return 0.0 }
            guard values.count > 1 else { return values[0] }

            let n = values.count - 1
            let idx = clampedRadius * Float(n)
            let i0 = Int(floor(idx))
            let i1 = min(i0 + 1, n)
            let frac = idx - Float(i0)

            return values[i0] * (1.0 - frac) + values[i1] * frac
        }
    }
}

// MARK: - Profile Conditions

/// Initial and prescribed profile conditions
///
/// **Units**: eV for temperature, m^-3 for density
///
/// ProfileConditions uses the same units as CoreProfiles and BoundaryConditions:
/// - Temperature: eV (electron volts)
/// - Density: m^-3 (particles per cubic meter)
///
/// This ensures consistency throughout the runtime system and eliminates
/// the need for unit conversions when materializing profiles.
///
/// **Note**: While tokamak literature often uses keV and 10^20 m^-3,
/// this implementation uses eV and m^-3 for consistency with physics models
/// and to match the original Gotenx (Python) design.
public struct ProfileConditions: Sendable, Codable, Equatable {
    /// Ion temperature profile [eV]
    public var ionTemperature: ProfileSpec

    /// Electron temperature profile [eV]
    public var electronTemperature: ProfileSpec

    /// Electron density profile [m^-3]
    public var electronDensity: ProfileSpec

    /// Current profile [MA/m^2]
    public var currentDensity: ProfileSpec

    public init(
        ionTemperature: ProfileSpec,
        electronTemperature: ProfileSpec,
        electronDensity: ProfileSpec,
        currentDensity: ProfileSpec
    ) {
        self.ionTemperature = ionTemperature
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity
        self.currentDensity = currentDensity
    }
}
