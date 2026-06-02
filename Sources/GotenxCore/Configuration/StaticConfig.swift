// StaticConfig.swift
// Static configuration (affects compilation)

import Foundation

/// Static configuration (affects compilation)
///
/// Changes to these parameters require MLX recompilation:
/// - Mesh resolution
/// - Which equations to evolve
/// - Solver type
/// - Numerical scheme
public struct StaticConfig: Codable, Sendable, Equatable, Hashable {
    /// Mesh configuration
    public let mesh: MeshConfig

    /// Evolution flags (which PDEs to solve)
    public let evolution: EvolutionConfig

    /// Solver configuration
    public let solver: SolverConfig

    /// Numerical scheme
    public let scheme: SchemeConfig

    public init(
        mesh: MeshConfig,
        evolution: EvolutionConfig = .default,
        solver: SolverConfig = .default,
        scheme: SchemeConfig = .default
    ) {
        self.mesh = mesh
        self.evolution = evolution
        self.solver = solver
        self.scheme = scheme
    }
}

// MARK: - Conversion to Runtime Parameters

extension StaticConfig {
    /// Convert to StaticRuntimeParameters for simulation execution
    ///
    /// This adapter bridges the configuration system with the runtime execution.
    /// Used in Phase 4 to initialize SimulationOrchestrator.
    ///
    /// - Throws: `ConfigurationError.invalidValue` if solver type is invalid
    public func runtimeParameters() throws -> StaticRuntimeParameters {
        let normalizedSolverType: String
        switch solver.type {
        case "newton":
            normalizedSolverType = "newtonRaphson"
        default:
            normalizedSolverType = solver.type
        }

        guard let solverType = SolverType(rawValue: normalizedSolverType) else {
            throw ConfigurationError.invalidValue(
                key: "solver.type",
                value: solver.type,
                reason: "Invalid solver type. Valid options: linear, newtonRaphson, optimizer"
            )
        }

        return StaticRuntimeParameters(
            mesh: mesh,
            evolveIonHeat: evolution.ionHeat,
            evolveElectronHeat: evolution.electronHeat,
            evolveElectronDensity: evolution.electronDensity,
            evolvePoloidalFlux: evolution.poloidalFlux,
            solverType: solverType,
            theta: scheme.theta,
            solverTolerance: solver.tolerance ?? 1e-6,
            solverMaximumIterations: solver.maximumIterations
        )
    }
}
