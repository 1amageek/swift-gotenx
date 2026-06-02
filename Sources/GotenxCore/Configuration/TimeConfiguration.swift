// TimeConfiguration.swift
// Time configuration

import Foundation

/// Time configuration
public struct TimeConfiguration: Codable, Sendable, Equatable {
    /// Initial time [s]
    public let start: Float

    /// Final time [s]
    public let end: Float

    /// Initial timestep [s]
    public let initialTimeStep: Float

    /// Adaptive timestepping
    public let adaptive: AdaptiveTimestepConfig?

    public init(
        start: Float = 0.0,
        end: Float,
        initialTimeStep: Float = 1e-3,
        adaptive: AdaptiveTimestepConfig? = .default
    ) {
        self.start = start
        self.end = end
        self.initialTimeStep = initialTimeStep
        self.adaptive = adaptive
    }

    private enum CodingKeys: String, CodingKey {
        case start
        case end
        case initialTimeStep
        case adaptive
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.init(
            start: try container.decodeIfPresent(Float.self, forKey: .start) ?? 0.0,
            end: try container.decode(Float.self, forKey: .end),
            initialTimeStep: try container.decodeIfPresent(Float.self, forKey: .initialTimeStep) ?? 1e-3,
            adaptive: try container.decodeIfPresent(AdaptiveTimestepConfig.self, forKey: .adaptive)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(start, forKey: .start)
        try container.encode(end, forKey: .end)
        try container.encode(initialTimeStep, forKey: .initialTimeStep)
        try container.encodeIfPresent(adaptive, forKey: .adaptive)
    }
}

/// Adaptive timestep configuration (EXTENDED for better scalability)
public struct AdaptiveTimestepConfig: Codable, Sendable, Equatable {
    /// Minimum timestep [s] (absolute)
    /// If set, takes precedence over minimumTimeStepFraction
    public let minimumTimeStep: Float?

    /// Minimum timestep fraction of maximumTimeStep (default: 0.001)
    /// Ignored if minimumTimeStep is explicitly set
    /// Recommended approach: minimumTimeStep = maximumTimeStep * minimumTimeStepFraction
    public let minimumTimeStepFraction: Float?

    /// Maximum timestep [s]
    public let maximumTimeStep: Float

    /// CFL safety factor (< 1.0)
    public let safetyFactor: Float

    /// Maximum timestep growth rate per step (default: 1.2)
    /// Limits how quickly timestep can increase: dt_new ≤ dt_old * maximumTimeStepGrowth
    public let maximumTimeStepGrowth: Float

    /// Computed minimum timestep (adaptive)
    /// Priority: explicit minimumTimeStep > minimumTimeStepFraction > default (maximumTimeStep * 0.001)
    public var effectiveMinimumTimeStep: Float {
        if let minimumTimeStep = minimumTimeStep {
            return minimumTimeStep  // Explicit value takes precedence
        } else if let fraction = minimumTimeStepFraction {
            return maximumTimeStep * fraction
        } else {
            return maximumTimeStep * 0.001  // Default fallback: maximumTimeStep / 1000
        }
    }

    public static let `default` = AdaptiveTimestepConfig(
        minimumTimeStep: nil,              // Use fraction instead
        minimumTimeStepFraction: 0.001,    // maximumTimeStep / 1000
        maximumTimeStep: 1e-1,
        safetyFactor: 0.9,
        maximumTimeStepGrowth: 1.2
    )

    public init(
        minimumTimeStep: Float? = nil,
        minimumTimeStepFraction: Float? = 0.001,
        maximumTimeStep: Float,
        safetyFactor: Float,
        maximumTimeStepGrowth: Float = 1.2
    ) {
        self.minimumTimeStep = minimumTimeStep
        self.minimumTimeStepFraction = minimumTimeStepFraction
        self.maximumTimeStep = maximumTimeStep
        self.safetyFactor = safetyFactor
        self.maximumTimeStepGrowth = maximumTimeStepGrowth
    }

    private enum CodingKeys: String, CodingKey {
        case minimumTimeStep
        case minimumTimeStepFraction
        case maximumTimeStep
        case safetyFactor
        case maximumTimeStepGrowth
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.init(
            minimumTimeStep: try container.decodeIfPresent(Float.self, forKey: .minimumTimeStep),
            minimumTimeStepFraction: try container.decodeIfPresent(Float.self, forKey: .minimumTimeStepFraction) ?? 0.001,
            maximumTimeStep: try container.decodeIfPresent(Float.self, forKey: .maximumTimeStep) ?? 1e-1,
            safetyFactor: try container.decodeIfPresent(Float.self, forKey: .safetyFactor) ?? 0.9,
            maximumTimeStepGrowth: try container.decodeIfPresent(Float.self, forKey: .maximumTimeStepGrowth) ?? 1.2
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(minimumTimeStep, forKey: .minimumTimeStep)
        try container.encodeIfPresent(minimumTimeStepFraction, forKey: .minimumTimeStepFraction)
        try container.encode(maximumTimeStep, forKey: .maximumTimeStep)
        try container.encode(safetyFactor, forKey: .safetyFactor)
        try container.encode(maximumTimeStepGrowth, forKey: .maximumTimeStepGrowth)
    }
}
