// EvolutionConfig.swift
// Evolution configuration (which equations to solve)

import Foundation

/// Evolution configuration (which equations to solve)
public struct EvolutionConfig: Codable, Sendable, Equatable, Hashable {
    /// Evolve ion heat transport equation
    public let ionHeat: Bool

    /// Evolve electron heat transport equation
    public let electronHeat: Bool

    /// Evolve electron density equation
    public let electronDensity: Bool

    /// Evolve poloidal flux (current diffusion) equation
    public let poloidalFlux: Bool

    public static let `default` = EvolutionConfig(
        ionHeat: true,
        electronHeat: true,
        electronDensity: true,
        poloidalFlux: false  // Often disabled for computational efficiency
    )

    public init(
        ionHeat: Bool = true,
        electronHeat: Bool = true,
        electronDensity: Bool = true,
        poloidalFlux: Bool = false
    ) {
        self.ionHeat = ionHeat
        self.electronHeat = electronHeat
        self.electronDensity = electronDensity
        self.poloidalFlux = poloidalFlux
    }

    /// Number of evolved equations
    public var count: Int {
        [ionHeat, electronHeat, electronDensity, poloidalFlux]
            .filter { $0 }
            .count
    }

    enum CodingKeys: String, CodingKey {
        case ionHeat = "ionTemperature"
        case electronHeat = "electronTemperature"
        case electronDensity
        case poloidalFlux
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.init(
            ionHeat: try container.decodeIfPresent(Bool.self, forKey: .ionHeat) ?? true,
            electronHeat: try container.decodeIfPresent(Bool.self, forKey: .electronHeat) ?? true,
            electronDensity: try container.decodeIfPresent(Bool.self, forKey: .electronDensity) ?? true,
            poloidalFlux: try container.decodeIfPresent(Bool.self, forKey: .poloidalFlux) ?? false
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(ionHeat, forKey: .ionHeat)
        try container.encode(electronHeat, forKey: .electronHeat)
        try container.encode(electronDensity, forKey: .electronDensity)
        try container.encode(poloidalFlux, forKey: .poloidalFlux)
    }
}
