// MHDConfig.swift
// Configuration for MHD models (sawteeth, NTMs, etc.)

import Foundation

/// Configuration for MHD models
public struct MHDConfig: Codable, Sendable, Equatable {
    /// Enable/disable sawtooth model
    public var sawtoothEnabled: Bool

    /// Sawtooth model parameters
    public var sawtoothParameters: SawtoothParameters

    /// Enable/disable neoclassical tearing modes (future)
    public var ntmEnabled: Bool

    public init(
        sawtoothEnabled: Bool = false,
        sawtoothParameters: SawtoothParameters = SawtoothParameters(),
        ntmEnabled: Bool = false
    ) {
        self.sawtoothEnabled = sawtoothEnabled
        self.sawtoothParameters = sawtoothParameters
        self.ntmEnabled = ntmEnabled
    }

    /// Default configuration (all MHD disabled)
    public static let `default` = MHDConfig()
}

/// Parameters for sawtooth crash model
///
/// Sawteeth are m=1, n=1 kink instabilities that flatten the central core profiles
/// when the safety factor q(0) drops below 1.
///
/// **Implementation**: Based on TORAX simple trigger + simple redistribution models
public struct SawtoothParameters: Codable, Sendable, Equatable {
    // MARK: - Trigger Parameters

    /// Minimum normalized radius for q=1 surface
    ///
    /// Prevents crashes when q=1 surface is too close to magnetic axis.
    /// Crash occurs only if rho_norm_q1 > minimumRadius.
    ///
    /// **Typical value**: 0.2 (20% of minor radius)
    public var minimumRadius: Float

    /// Critical magnetic shear threshold
    ///
    /// Crash occurs when shear s = (r/q)(dq/radialSpacing) at q=1 surface exceeds this value.
    ///
    /// **Typical value**: 0.2
    public var sCritical: Float

    /// Minimum time between crashes (seconds)
    ///
    /// Prevents unphysically rapid crash sequences.
    ///
    /// **Typical value**: 0.01 s (10 ms)
    public var minimumCrashInterval: Float

    // MARK: - Redistribution Parameters

    /// Profile flattening factor
    ///
    /// Controls how flat the profile becomes at r=0 relative to r=rho_q1.
    /// - 1.0: Perfect flattening (T(0) = T(rho_q1))
    /// - 1.01: Slight gradient (T(0) = 1.01 × T(rho_q1))
    ///
    /// **Typical value**: 1.01
    public var flatteningFactor: Float

    /// Mixing radius multiplier
    ///
    /// Defines the extent of profile redistribution:
    /// rho_mix = mixingRadiusMultiplier × rho_q1
    ///
    /// **Typical value**: 1.5 (mixing extends 50% beyond q=1 surface)
    public var mixingRadiusMultiplier: Float

    /// Duration of crash event (seconds)
    ///
    /// Time step duration during which crash occurs.
    /// During crash, PDE solver is bypassed and time advances by this amount.
    ///
    /// **Typical value**: 1e-3 s (1 ms, fast MHD timescale)
    public var crashStepDuration: Float

    public init(
        minimumRadius: Float = 0.2,
        sCritical: Float = 0.2,
        minimumCrashInterval: Float = 0.01,
        flatteningFactor: Float = 1.01,
        mixingRadiusMultiplier: Float = 1.5,
        crashStepDuration: Float = 1e-3
    ) {
        self.minimumRadius = minimumRadius
        self.sCritical = sCritical
        self.minimumCrashInterval = minimumCrashInterval
        self.flatteningFactor = flatteningFactor
        self.mixingRadiusMultiplier = mixingRadiusMultiplier
        self.crashStepDuration = crashStepDuration
    }
}

/// MHD model protocol for different instability types
public protocol MHDModel: Sendable {
    /// Apply MHD effects to core profiles
    /// - Parameters:
    ///   - profiles: Current core profiles
    ///   - geometry: Simulation geometry
    ///   - time: Current simulation time
    ///   - timeStep: Timestep
    /// - Returns: Modified profiles after MHD effects
    func apply(
        to profiles: CoreProfiles,
        geometry: Geometry,
        time: Float,
        timeStep: Float
    ) -> CoreProfiles
}

/// Factory for creating MHD models from configuration
public struct MHDModelFactory {
    /// Create sawtooth model if enabled
    public static func createSawtoothModel(
        config: MHDConfig
    ) -> (any MHDModel)? {
        guard config.sawtoothEnabled else {
            return nil
        }

        return SawtoothModel(parameters: config.sawtoothParameters)
    }

    /// Create all enabled MHD models
    public static func createAllModels(
        config: MHDConfig
    ) -> [any MHDModel] {
        var models: [any MHDModel] = []

        if let sawtoothModel = createSawtoothModel(config: config) {
            models.append(sawtoothModel)
        }

        // Future: NTM models, etc.

        return models
    }
}
