// SawtoothTrigger.swift
// Sawtooth crash trigger model based on safety factor q and magnetic shear

import Foundation
import MLX

/// Sawtooth crash trigger based on q-profile and magnetic shear
///
/// Implements the "simple trigger" model from TORAX:
/// - Detects q=1 surface from safety factor profile
/// - Checks minimum radius condition (prevents crashes too close to axis)
/// - Checks critical magnetic shear condition
///
/// **Physical Basis**:
/// Sawteeth are m=1, n=1 kink instabilities that occur when:
/// 1. Safety factor q(0) < 1 (q=1 surface exists inside plasma)
/// 2. Magnetic shear at q=1 surface exceeds critical value
/// 3. q=1 surface is not too close to magnetic axis (stability)
///
/// **References**:
/// - Porcelli et al., "Model for the sawtooth period and amplitude" (1996)
/// - Kadomtsev, "Disruptive instability in tokamaks" (1975)
/// - TORAX: arXiv:2406.06718v2
public struct SimpleSawtoothTrigger: Sendable {
    /// Minimum normalized radius for q=1 surface
    ///
    /// Prevents crashes when q=1 surface is too close to magnetic axis.
    /// **Typical value**: 0.2 (20% of minor radius)
    public let minimumRadius: Float

    /// Critical magnetic shear threshold
    ///
    /// Crash occurs when shear s = (r/q)(dq/radialSpacing) at q=1 surface exceeds this value.
    /// **Typical value**: 0.2
    public let sCritical: Float

    /// Minimum time between crashes (seconds)
    ///
    /// Prevents unphysically rapid crash sequences.
    /// **Typical value**: 0.01 s (10 ms)
    public let minimumCrashInterval: Float

    public init(
        minimumRadius: Float = 0.2,
        sCritical: Float = 0.2,
        minimumCrashInterval: Float = 0.01
    ) {
        self.minimumRadius = minimumRadius
        self.sCritical = sCritical
        self.minimumCrashInterval = minimumCrashInterval
    }

    /// Determine if sawtooth crash should occur
    ///
    /// **Trigger Conditions**:
    /// 1. q=1 surface exists (q(0) < 1)
    /// 2. rho_norm_q1 > minimumRadius (not too close to axis)
    /// 3. s_q1 > sCritical (sufficient magnetic shear)
    /// 4. timeStep >= minimumCrashInterval (rate limiting)
    ///
    /// **Parameters**:
    /// - profiles: Current core profiles
    /// - geometry: Tokamak geometry
    /// - timeStep: Current timestep [s]
    ///
    /// **Returns**: Tuple of (triggered, rhoQ1)
    ///   - triggered: Whether crash should occur
    ///   - rhoQ1: Normalized radius of q=1 surface (nil if no q=1 surface)
    public func shouldTrigger(
        profiles: CoreProfiles,
        geometry: Geometry,
        timeStep: Float
    ) -> (triggered: Bool, rhoQ1: Float?) {
        // Rate limiting: only crash if timestep resolves the crash interval
        // This prevents unphysical rapid crashes when timeStep << minimumCrashInterval
        guard timeStep >= minimumCrashInterval else {
            return (false, nil)
        }

        // Compute safety factor profile from poloidal flux
        let q = profiles.safetyFactor(geometry: geometry)

        // Check if q=1 surface exists (q(0) < 1)
        let qOnAxis = q[0].item(Float.self)
        guard qOnAxis < 1.0 else {
            return (false, nil)
        }

        // Find q=1 surface location
        guard let (rhoQ1, indexQ1) = findQ1Surface(q: q, geometry: geometry) else {
            return (false, nil)
        }

        // Check minimum radius condition
        guard rhoQ1 > minimumRadius else {
            return (false, rhoQ1)
        }

        // Compute magnetic shear at q=1 surface
        // Use interpolation for more accurate shear at exact q=1 location
        let shear = profiles.magneticShear(geometry: geometry)
        let shearQ1 = interpolateShearAtQ1(
            shear: shear,
            q: q,
            indexQ1: indexQ1,
            rhoQ1: rhoQ1,
            geometry: geometry
        )

        // Check critical magnetic shear condition
        guard shearQ1 > sCritical else {
            return (false, rhoQ1)
        }

        // All conditions met: trigger crash
        return (true, rhoQ1)
    }

    /// Find q=1 surface location in the plasma
    ///
    /// **Algorithm**:
    /// - Search for first cell where q >= 1
    /// - Interpolate to find exact rho_norm where q = 1
    ///
    /// **Parameters**:
    /// - q: Safety factor profile [cellCount]
    /// - geometry: Tokamak geometry
    ///
    /// **Returns**: Tuple of (rho_norm_q1, index) or nil if no q=1 surface found
    private func findQ1Surface(
        q: MLXArray,
        geometry: Geometry
    ) -> (rhoNorm: Float, index: Int)? {
        let qArray = q.asArray(Float.self)
        let radii = geometry.radii.value.asArray(Float.self)
        let minorRadius = geometry.minorRadius

        // Search for first cell where q >= 1
        for i in 0..<(qArray.count - 1) {
            let q_i = qArray[i]
            let q_next = qArray[i + 1]

            // Check if q crosses 1.0 between i and i+1
            if q_i < 1.0 && q_next >= 1.0 {
                // Linear interpolation to find exact location
                let r_i = radii[i]
                let r_next = radii[i + 1]

                // Interpolate: r_q1 = r_i + (r_next - r_i) * (1 - q_i) / (q_next - q_i)
                let fraction = (1.0 - q_i) / (q_next - q_i + 1e-10)
                let rQ1 = r_i + (r_next - r_i) * fraction
                let rhoNormQ1 = rQ1 / minorRadius

                return (rhoNormQ1, i)
            }
        }

        return nil
    }

    /// Interpolate magnetic shear at q=1 surface
    ///
    /// Since q=1 surface is between grid points i and i+1, we interpolate
    /// the shear value to get more accurate shear at exact q=1 location.
    ///
    /// **Parameters**:
    /// - shear: Magnetic shear profile [cellCount]
    /// - q: Safety factor profile [cellCount]
    /// - indexQ1: Grid index where q crosses 1 (q[i] < 1, q[i+1] >= 1)
    /// - rhoQ1: Normalized radius of q=1 surface
    /// - geometry: Tokamak geometry
    ///
    /// **Returns**: Interpolated shear at q=1 surface
    private func interpolateShearAtQ1(
        shear: MLXArray,
        q: MLXArray,
        indexQ1: Int,
        rhoQ1: Float,
        geometry: Geometry
    ) -> Float {
        let shearArray = shear.asArray(Float.self)
        let qArray = q.asArray(Float.self)

        // Safety check: ensure we have valid indices
        guard indexQ1 < shearArray.count - 1 else {
            // Fallback: use boundary value
            return shearArray[indexQ1]
        }

        // Get shear values at adjacent grid points
        let shear_i = shearArray[indexQ1]
        let shear_next = shearArray[indexQ1 + 1]

        // Get q values at adjacent grid points
        let q_i = qArray[indexQ1]
        let q_next = qArray[indexQ1 + 1]

        // Linear interpolation weight based on q values
        // w = (1 - q_i) / (q_next - q_i)
        let weight = (1.0 - q_i) / (q_next - q_i + 1e-10)

        // Interpolate shear: s_q1 = s_i + weight * (s_next - s_i)
        let shearQ1 = shear_i + weight * (shear_next - shear_i)

        return shearQ1
    }
}
