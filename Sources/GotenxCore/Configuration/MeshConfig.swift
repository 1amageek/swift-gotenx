import Foundation

// MARK: - Mesh Configuration

/// Mesh configuration for spatial discretization
public struct MeshConfig: Sendable, Codable, Equatable, Hashable {
    /// Number of cells in radial direction
    public let cellCount: Int

    /// Major radius [m]
    public let majorRadius: Float

    /// Minor radius [m]
    public let minorRadius: Float

    /// Toroidal magnetic field at major radius [T]
    public let toroidalField: Float

    /// Geometry type
    public let geometryType: GeometryType

    public init(
        cellCount: Int,
        majorRadius: Float,
        minorRadius: Float,
        toroidalField: Float,
        geometryType: GeometryType = .circular
    ) {
        // NOTE: we intentionally avoid preconditions here so configuration
        // values coming from files can be validated explicitly via
        // `validate()`.  This keeps construction lightweight even for
        // obviously invalid inputs, which mirrors the configuration
        // loading flow exercised by the tests.
        self.cellCount = cellCount
        self.majorRadius = majorRadius
        self.minorRadius = minorRadius
        self.toroidalField = toroidalField
        self.geometryType = geometryType
    }

    /// Grid spacing [m]
    public var radialSpacing: Float {
        guard cellCount > 0 else { return .infinity }
        return minorRadius / Float(cellCount)
    }

    /// Aspect ratio (R/a)
    public var aspectRatio: Float {
        guard minorRadius != 0 else { return .infinity }
        return majorRadius / minorRadius
    }

    private enum CodingKeys: String, CodingKey {
        case cellCount
        case majorRadius
        case minorRadius
        case toroidalField
        case geometryType
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.init(
            cellCount: try container.decode(Int.self, forKey: .cellCount),
            majorRadius: try container.decode(Float.self, forKey: .majorRadius),
            minorRadius: try container.decode(Float.self, forKey: .minorRadius),
            toroidalField: try container.decode(Float.self, forKey: .toroidalField),
            geometryType: try container.decodeIfPresent(GeometryType.self, forKey: .geometryType) ?? .circular
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(cellCount, forKey: .cellCount)
        try container.encode(majorRadius, forKey: .majorRadius)
        try container.encode(minorRadius, forKey: .minorRadius)
        try container.encode(toroidalField, forKey: .toroidalField)
        try container.encode(geometryType, forKey: .geometryType)
    }
}

// MARK: - Physics Validation

extension MeshConfig {
    /// Validate physics constraints
    public func validate() throws {
        guard cellCount > 0 else {
            throw ConfigurationError.invalidValue(
                key: "mesh.cellCount",
                value: "\(cellCount)",
                reason: "Must be positive"
            )
        }

        guard cellCount >= 10 else {
            throw ConfigurationError.physicsWarning(
                key: "mesh.cellCount",
                value: "\(cellCount)",
                reason: "Fewer than 10 cells may produce inaccurate results"
            )
        }

        guard majorRadius > 0, minorRadius > 0 else {
            throw ConfigurationError.invalidValue(
                key: "mesh.radius",
                value: "R=\(majorRadius), a=\(minorRadius)",
                reason: "Radii must be positive"
            )
        }

        let aspectRatio = aspectRatio
        guard aspectRatio >= 1.5 else {
            throw ConfigurationError.physicsWarning(
                key: "mesh.aspectRatio",
                value: "\(aspectRatio)",
                reason: "Aspect ratio < 1.5 is unrealistic for tokamaks"
            )
        }

        guard toroidalField > 0 else {
            throw ConfigurationError.invalidValue(
                key: "mesh.toroidalField",
                value: "\(toroidalField)",
                reason: "Magnetic field must be positive"
            )
        }
    }
}
