// BoundaryConfig.swift
// Boundary conditions configuration

import Foundation

/// Boundary conditions configuration
public struct BoundaryConfig: Codable, Sendable, Equatable {
    /// Ion temperature at edge [eV]
    public let ionTemperature: Float

    /// Electron temperature at edge [eV]
    public let electronTemperature: Float

    /// Electron density at edge [m^-3]
    public let electronDensity: Float

    /// Boundary condition type
    public let type: BoundaryType

    public init(
        ionTemperature: Float,
        electronTemperature: Float,
        electronDensity: Float,
        type: BoundaryType = .dirichlet
    ) {
        self.ionTemperature = ionTemperature
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity
        self.type = type
    }

    private enum CodingKeys: String, CodingKey {
        case ionTemperature
        case electronTemperature
        case electronDensity
        case type
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.init(
            ionTemperature: try container.decode(Float.self, forKey: .ionTemperature),
            electronTemperature: try container.decode(Float.self, forKey: .electronTemperature),
            electronDensity: try container.decode(Float.self, forKey: .electronDensity),
            type: try container.decodeIfPresent(BoundaryType.self, forKey: .type) ?? .dirichlet
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(ionTemperature, forKey: .ionTemperature)
        try container.encode(electronTemperature, forKey: .electronTemperature)
        try container.encode(electronDensity, forKey: .electronDensity)
        try container.encode(type, forKey: .type)
    }
}

/// Boundary condition type
public enum BoundaryType: String, Codable, Sendable {
    case dirichlet  // Fixed value
    case neumann    // Fixed gradient
}

// MARK: - Conversion to Runtime Parameters

extension BoundaryConfig {
    /// Convert to BoundaryConditions for runtime
    ///
    /// **Units (no conversion):**
    /// - Temperature: eV
    /// - Density: m^-3
    ///
    /// **Assumptions:**
    /// - Left boundary (r=0, magnetic axis): symmetric, gradient = 0
    /// - Right boundary (r=a, edge): uses configured value/type
    public func toBoundaryConditions() -> BoundaryConditions {
        let rightConstraint: (Float, BoundaryType) -> FaceConstraint = { value, type in
            switch type {
            case .dirichlet:
                return .value(value)
            case .neumann:
                return .gradient(0.0)  // Zero gradient at edge
            }
        }

        return BoundaryConditions(
            ionTemperature: BoundaryCondition(
                left: .gradient(0.0),  // Symmetric at axis
                right: rightConstraint(ionTemperature, type)
            ),
            electronTemperature: BoundaryCondition(
                left: .gradient(0.0),
                right: rightConstraint(electronTemperature, type)
            ),
            electronDensity: BoundaryCondition(
                left: .gradient(0.0),
                right: rightConstraint(electronDensity, type)
            ),
            poloidalFlux: BoundaryCondition(
                left: .value(0.0),      // Flux = 0 at magnetic axis
                right: .gradient(0.0)   // Free boundary at edge
            )
        )
    }
}
