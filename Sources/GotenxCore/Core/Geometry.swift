import Foundation

// MARK: - Geometry Type

/// Geometry type enumeration
public enum GeometryType: String, Sendable, Codable, CaseIterable {
    case circular
    case chease
    case eqdsk
}

// MARK: - Geometry

/// Geometric configuration of the tokamak
public struct Geometry: Sendable, Equatable {
    /// Major radius [m]
    public let majorRadius: Float

    /// Minor radius [m]
    public let minorRadius: Float

    /// Toroidal magnetic field [T]
    public let toroidalField: Float

    /// Plasma volume [m^3]
    public let volume: EvaluatedArray

    /// Geometric coefficient fluxSurfaceMetric (for FVM)
    public let fluxSurfaceMetric: EvaluatedArray

    /// Geometric coefficient majorRadiusMetric (for FVM)
    public let majorRadiusMetric: EvaluatedArray

    /// Geometric coefficient shapeMetric (for FVM)
    public let shapeMetric: EvaluatedArray

    /// Geometric coefficient minorRadiusMetric (for FVM)
    public let minorRadiusMetric: EvaluatedArray

    /// Radial coordinates at cell centers [m]
    public let radii: EvaluatedArray

    /// Safety factor profile q(r)
    public let safetyFactor: EvaluatedArray

    /// Poloidal magnetic field profile Bp(r) [T] (optional)
    public let poloidalField: EvaluatedArray?

    /// Current density profile j(r) [MA/m^2] (optional)
    public let currentDensity: EvaluatedArray?

    /// Geometry type
    public let type: GeometryType

    public init(
        majorRadius: Float,
        minorRadius: Float,
        toroidalField: Float,
        volume: EvaluatedArray,
        fluxSurfaceMetric: EvaluatedArray,
        majorRadiusMetric: EvaluatedArray,
        shapeMetric: EvaluatedArray,
        minorRadiusMetric: EvaluatedArray,
        radii: EvaluatedArray,
        safetyFactor: EvaluatedArray,
        poloidalField: EvaluatedArray? = nil,
        currentDensity: EvaluatedArray? = nil,
        type: GeometryType
    ) {
        self.majorRadius = majorRadius
        self.minorRadius = minorRadius
        self.toroidalField = toroidalField
        self.volume = volume
        self.fluxSurfaceMetric = fluxSurfaceMetric
        self.majorRadiusMetric = majorRadiusMetric
        self.shapeMetric = shapeMetric
        self.minorRadiusMetric = minorRadiusMetric
        self.radii = radii
        self.safetyFactor = safetyFactor
        self.poloidalField = poloidalField
        self.currentDensity = currentDensity
        self.type = type
    }
}
