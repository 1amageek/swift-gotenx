// SawtoothModel.swift
// Sawtooth crash model for tokamak MHD instabilities

import Foundation
import MLX

/// Sawtooth crash model with physically-based trigger and conservation enforcement
///
/// Sawteeth are m=1, n=1 kink instabilities that occur when the safety factor q(0) < 1.
/// They cause rapid redistribution (flattening) of core profiles within the inversion radius.
///
/// **Implementation**:
/// - Uses `SimpleSawtoothTrigger` for q-profile based triggering
/// - Uses `SimpleSawtoothRedistribution` for profile flattening with conservation
///
/// **References**:
/// - Kadomtsev reconnection model (1975)
/// - Porcelli model for sawtooth trigger (1996)
/// - TORAX: arXiv:2406.06718v2
public struct SawtoothModel: MHDModel {
    public let parameters: SawtoothParameters

    /// Trigger model (detects when crash should occur)
    private let trigger: SimpleSawtoothTrigger

    /// Redistribution model (applies profile flattening with conservation)
    private let redistribution: SimpleSawtoothRedistribution

    public init(parameters: SawtoothParameters) {
        self.parameters = parameters

        // Create trigger model from parameters
        self.trigger = SimpleSawtoothTrigger(
            minimumRadius: parameters.minimumRadius,
            sCritical: parameters.sCritical,
            minimumCrashInterval: parameters.minimumCrashInterval
        )

        // Create redistribution model from parameters
        self.redistribution = SimpleSawtoothRedistribution(
            flatteningFactor: parameters.flatteningFactor,
            mixingRadiusMultiplier: parameters.mixingRadiusMultiplier
        )
    }

    public func apply(
        to profiles: CoreProfiles,
        geometry: Geometry,
        time: Float,
        timeStep: Float
    ) -> CoreProfiles {
        // Check if sawtooth crash should occur
        let (triggered, rhoQ1) = trigger.shouldTrigger(
            profiles: profiles,
            geometry: geometry,
            timeStep: timeStep
        )

        guard triggered, let rhoQ1 = rhoQ1 else {
            // No crash: return profiles unchanged
            return profiles
        }

        // Apply crash with conservation enforcement
        return redistribution.redistribute(
            profiles: profiles,
            geometry: geometry,
            rhoQ1: rhoQ1
        )
    }
}

// MARK: - Helper Functions

extension SawtoothModel {
    /// Check if sawtooth model is active
    public var isActive: Bool {
        return true  // Always active if model is created
    }
}
