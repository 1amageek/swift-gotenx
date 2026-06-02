// SolverConfig.swift
// Solver configuration

import Foundation

/// Solver configuration
public struct SolverConfig: Codable, Sendable, Equatable, Hashable {
    /// Solver type (using existing SolverType from RuntimeParams)
    public let type: String

    /// Scalar convergence tolerance fallback when per-equation tolerances are not supplied
    public let tolerance: Float?

    /// Per-equation numerical tolerances (recommended)
    /// If nil, falls back to legacy tolerance
    public let tolerances: NumericalTolerances?

    /// Physical thresholds for diagnostics
    /// Optional physical thresholds
    public let physicalThresholds: PhysicalThresholds?

    /// Maximum iterations
    public let maximumIterations: Int

    /// Line search enabled (default: true)
    public let lineSearchEnabled: Bool

    /// Maximum line search alpha (default: 1.0)
    public let lineSearchMaxAlpha: Float

    /// Effective physical thresholds (with fallback)
    public var effectiveThresholds: PhysicalThresholds {
        return physicalThresholds ?? .default
    }

    /// Computed tolerances (prioritizes new over legacy)
    public var effectiveTolerances: NumericalTolerances {
        if let tolerances = tolerances {
            return tolerances
        } else if let legacyTol = tolerance {
            return NumericalTolerances.fromLegacy(tolerance: legacyTol)
        } else {
            return .iterScale
        }
    }

    public static let `default` = SolverConfig(
        type: "newtonRaphson",
        tolerance: nil,
        tolerances: .iterScale,
        physicalThresholds: .default,
        maximumIterations: 30,
        lineSearchEnabled: true,
        lineSearchMaxAlpha: 1.0
    )

    public init(
        type: String = "newtonRaphson",
        tolerance: Float? = nil,
        tolerances: NumericalTolerances? = .iterScale,
        physicalThresholds: PhysicalThresholds? = .default,
        maximumIterations: Int = 100,
        lineSearchEnabled: Bool = true,
        lineSearchMaxAlpha: Float = 1.0
    ) {
        self.type = type
        self.tolerance = tolerance
        self.tolerances = tolerances
        self.physicalThresholds = physicalThresholds
        self.maximumIterations = maximumIterations
        self.lineSearchEnabled = lineSearchEnabled
        self.lineSearchMaxAlpha = lineSearchMaxAlpha
    }

    enum CodingKeys: String, CodingKey {
        case type
        case tolerance
        case tolerances
        case physicalThresholds
        case maximumIterations
        case lineSearchEnabled
        case lineSearchMaxAlpha
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        type = try container.decodeIfPresent(String.self, forKey: .type) ?? "newtonRaphson"
        maximumIterations = try container.decodeIfPresent(Int.self, forKey: .maximumIterations) ?? 30

        tolerance = try container.decodeIfPresent(Float.self, forKey: .tolerance)

        tolerances = try container.decodeIfPresent(NumericalTolerances.self, forKey: .tolerances)
        physicalThresholds = try container.decodeIfPresent(PhysicalThresholds.self, forKey: .physicalThresholds)
        lineSearchEnabled = try container.decodeIfPresent(Bool.self, forKey: .lineSearchEnabled) ?? true
        lineSearchMaxAlpha = try container.decodeIfPresent(Float.self, forKey: .lineSearchMaxAlpha) ?? 1.0
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(type, forKey: .type)
        try container.encodeIfPresent(tolerance, forKey: .tolerance)
        try container.encodeIfPresent(tolerances, forKey: .tolerances)
        try container.encodeIfPresent(physicalThresholds, forKey: .physicalThresholds)
        try container.encode(maximumIterations, forKey: .maximumIterations)
        try container.encode(lineSearchEnabled, forKey: .lineSearchEnabled)
        try container.encode(lineSearchMaxAlpha, forKey: .lineSearchMaxAlpha)
    }
}
