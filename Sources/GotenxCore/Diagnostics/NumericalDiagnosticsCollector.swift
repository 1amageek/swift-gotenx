// NumericalDiagnosticsCollector.swift
// Collects numerical diagnostics from solver and simulation state
//
// Phase 2 Implementation: Basic solver diagnostics
// Phase 3 Implementation: Conservation monitoring, advanced metrics

import Foundation
import MLX

/// Collects numerical diagnostics from solver results and simulation state
///
/// **Design Philosophy**:
/// - Minimal overhead: Only collect what's already computed
/// - Incremental approach: Phase 2 focuses on solver diagnostics
/// - Future-proof: Structure ready for Phase 3 conservation monitoring
public enum NumericalDiagnosticsCollector {

    // MARK: - Phase 2: Basic Solver Diagnostics

    /// Create diagnostics from solver result
    ///
    /// **Phase 2**: Captures solver convergence metrics only
    /// **Phase 3**: Will add conservation drift monitoring
    ///
    /// - Parameters:
    ///   - solverResult: Result from Newton-Raphson or linear solver
    ///   - timeStep: Current timestep [s]
    ///   - wallTime: Wall clock time for this step [s] (optional)
    ///   - cflNumber: CFL number (optional, Phase 3)
    /// - Returns: Numerical diagnostics
    public static func collect(
        from solverResult: SolverResult,
        timeStep: Float,
        wallTime: Float = 0.0,
        cflNumber: Float = 0.0
    ) -> NumericalDiagnostics {

        // Extract solver metrics
        let residualNorm = solverResult.residualNorm
        let newtonIterations = solverResult.iterations
        let converged = solverResult.converged

        // Linear iterations (from metadata if available)
        let linearIterations = Int(solverResult.metadata["linear_iterations"] ?? 0)

        // Evaluation count (from metadata if available)
        let evalCount = Int(solverResult.metadata["eval_count"] ?? 0)

        // Phase 2: Conservation drifts are zero (not yet monitored)
        let particleDrift: Float = 0.0
        let energyDrift: Float = 0.0
        let currentDrift: Float = 0.0

        return NumericalDiagnostics(
            residualNorm: residualNorm,
            newtonIterations: newtonIterations,
            linearIterations: linearIterations,
            converged: converged,
            particleDrift: particleDrift,
            energyDrift: energyDrift,
            currentDrift: currentDrift,
            wallTime: wallTime,
            evaluationCount: evalCount,
            timeStep: timeStep,
            cflNumber: cflNumber
        )
    }

    // MARK: - Phase 3: Conservation Monitoring

    /// Compute conservation drifts relative to initial state
    ///
    /// **Phase 3**: Computes actual conservation violations
    ///
    /// Conservation laws tested:
    /// 1. **Particle conservation**: N = ∫ ne dV = constant
    /// 2. **Energy conservation**: W = ∫ (3/2 * ne * (Ti + Te)) dV = constant
    /// 3. **Current conservation**: I ≈ ∫ ψ dV = constant (simplified)
    ///
    /// - Parameters:
    ///   - current: Current simulation state
    ///   - initial: Initial simulation state
    ///   - geometry: Tokamak geometry
    /// - Returns: Conservation drifts (particle, energy, current)
    ///
    /// ## Interpretation
    ///
    /// Drift values represent relative change from initial state:
    /// - drift = 0: Perfect conservation (numerical artifact only)
    /// - |drift| < 0.001: Excellent (< 0.1% drift)
    /// - |drift| < 0.01: Good (< 1% drift, typical for Gotenx)
    /// - |drift| > 0.05: Warning (> 5% drift, check timestep/resolution)
    /// - |drift| > 0.1: Critical (> 10% drift, simulation may be unstable)
    public static func computeConservationDrifts(
        current: CoreProfiles,
        initial: CoreProfiles,
        geometry: Geometry
    ) -> (particle: Float, energy: Float, current: Float) {

        // Extract geometry for volume integration
        let geometricFactors = GeometricFactors.from(geometry: geometry)
        let volumes = geometricFactors.cellVolumes.value

        // 1. Particle conservation: N = ∫ ne dV
        let N0 = (initial.electronDensity.value * volumes).sum()
        let N = (current.electronDensity.value * volumes).sum()
        eval(N0, N)

        let particleDrift = ((N - N0) / (N0 + 1e-20)).item(Float.self)

        // 2. Energy conservation: W = ∫ (3/2 * ne * (Ti + Te)) dV
        //    Thermal energy density in eV/m^3
        let W0_density = 1.5 * initial.electronDensity.value * (
            initial.ionTemperature.value + initial.electronTemperature.value
        )
        let W_density = 1.5 * current.electronDensity.value * (
            current.ionTemperature.value + current.electronTemperature.value
        )

        let W0 = (W0_density * volumes).sum()
        let W = (W_density * volumes).sum()
        eval(W0, W)

        let energyDrift = ((W - W0) / (W0 + 1e-20)).item(Float.self)

        // 3. Current conservation: I ≈ ∫ ψ dV (simplified approximation)
        //    Note: True current conservation requires ∫ j_parallel dV
        //    Using poloidal flux as proxy for current profile
        let I0 = (initial.poloidalFlux.value * volumes).sum()
        let I = (current.poloidalFlux.value * volumes).sum()
        eval(I0, I)

        let currentDrift = ((I - I0) / (abs(I0) + 1e-20)).item(Float.self)

        return (particleDrift, energyDrift, currentDrift)
    }

    /// Enhanced collection with optional conservation monitoring
    ///
    /// **Phase 3**: Adds conservation drift computation when initial state is provided
    ///
    /// - Parameters:
    ///   - solverResult: Result from Newton-Raphson or linear solver
    ///   - timeStep: Current timestep [s]
    ///   - wallTime: Wall clock time for this step [s]
    ///   - cflNumber: CFL number
    ///   - currentProfiles: Current state profiles (optional, for conservation)
    ///   - initialProfiles: Initial state profiles (optional, for conservation)
    ///   - geometry: Tokamak geometry (optional, for conservation)
    /// - Returns: Numerical diagnostics with conservation monitoring
    public static func collectWithConservation(
        from solverResult: SolverResult,
        timeStep: Float,
        wallTime: Float = 0.0,
        cflNumber: Float = 0.0,
        currentProfiles: CoreProfiles? = nil,
        initialProfiles: CoreProfiles? = nil,
        geometry: Geometry? = nil
    ) -> NumericalDiagnostics {

        // Extract solver metrics
        let residualNorm = solverResult.residualNorm
        let newtonIterations = solverResult.iterations
        let converged = solverResult.converged

        // Linear iterations (from metadata if available)
        let linearIterations = Int(solverResult.metadata["linear_iterations"] ?? 0)

        // Evaluation count (from metadata if available)
        let evalCount = Int(solverResult.metadata["eval_count"] ?? 0)

        // Compute conservation drifts if data is available
        var particleDrift: Float = 0.0
        var energyDrift: Float = 0.0
        var currentDrift: Float = 0.0

        if let current = currentProfiles,
           let initial = initialProfiles,
           let geo = geometry {
            let drifts = computeConservationDrifts(
                current: current,
                initial: initial,
                geometry: geo
            )
            particleDrift = drifts.particle
            energyDrift = drifts.energy
            currentDrift = drifts.current
        }

        return NumericalDiagnostics(
            residualNorm: residualNorm,
            newtonIterations: newtonIterations,
            linearIterations: linearIterations,
            converged: converged,
            particleDrift: particleDrift,
            energyDrift: energyDrift,
            currentDrift: currentDrift,
            wallTime: wallTime,
            evaluationCount: evalCount,
            timeStep: timeStep,
            cflNumber: cflNumber
        )
    }
}
