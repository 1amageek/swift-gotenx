// NumericalDiagnostics.swift
// Numerical solver diagnostics for monitoring convergence and conservation
//
// Phase 1 Implementation: Minimal structure with default values
// Phase 2 Implementation: Actual diagnostics from solver state

import Foundation

/// Numerical solver diagnostics for monitoring simulation health
///
/// **Purpose**:
/// - Track convergence behavior of Newton-Raphson solver
/// - Monitor conservation laws (particles, energy, current)
/// - Detect numerical instabilities early
///
/// **Implementation Phases**:
/// - Phase 1 (Current): Returns default values
/// - Phase 2: Captures actual solver diagnostics
/// - Phase 3: Adds conservation monitoring
public struct NumericalDiagnostics: Sendable, Codable, Equatable {
    // MARK: - Convergence Metrics

    /// L2 norm of residual ||R|| at current timestep
    public let residualNorm: Float

    /// Number of Newton-Raphson iterations taken
    public let newtonIterations: Int

    /// Number of linear solver iterations
    public let linearIterations: Int

    /// Convergence flag (true if residual < tolerance)
    public let converged: Bool

    // MARK: - Conservation Metrics

    /// Particle conservation drift: (N - N_0) / N_0
    ///
    /// **Acceptance Criteria**: |drift| < 0.01 (1%)
    public let particleDrift: Float

    /// Energy conservation drift: (W - W_0) / W_0
    ///
    /// **Acceptance Criteria**: |drift| < 0.01 (1%)
    public let energyDrift: Float

    /// Current conservation drift: (I - I_0) / I_0
    ///
    /// **Acceptance Criteria**: |drift| < 0.01 (1%)
    public let currentDrift: Float

    // MARK: - Performance Metrics

    /// Wall clock time for this timestep [s]
    public let wallTime: Float

    /// Number of residual function evaluations
    public let evaluationCount: Int

    // MARK: - Timestep Control

    /// Adaptive timestep size [s]
    public let timeStep: Float

    /// CFL number (Courant-Friedrichs-Lewy condition)
    public let cflNumber: Float

    // MARK: - Initialization

    public init(
        residualNorm: Float,
        newtonIterations: Int,
        linearIterations: Int,
        converged: Bool,
        particleDrift: Float,
        energyDrift: Float,
        currentDrift: Float,
        wallTime: Float,
        evaluationCount: Int,
        timeStep: Float,
        cflNumber: Float
    ) {
        self.residualNorm = residualNorm
        self.newtonIterations = newtonIterations
        self.linearIterations = linearIterations
        self.converged = converged
        self.particleDrift = particleDrift
        self.energyDrift = energyDrift
        self.currentDrift = currentDrift
        self.wallTime = wallTime
        self.evaluationCount = evaluationCount
        self.timeStep = timeStep
        self.cflNumber = cflNumber
    }
}

// MARK: - Phase 1: Default Values

extension NumericalDiagnostics {
    /// Phase 1 implementation: Return sensible defaults
    ///
    /// **Rationale**: Allows compilation and testing without breaking existing code.
    /// Actual diagnostics will be captured in Phase 2.
    public static let `default` = NumericalDiagnostics(
        residualNorm: 0,
        newtonIterations: 0,
        linearIterations: 0,
        converged: true,  // Assume convergence by default
        particleDrift: 0,
        energyDrift: 0,
        currentDrift: 0,
        wallTime: 0,
        evaluationCount: 0,
        timeStep: 1e-4,  // Default timestep
        cflNumber: 0
    )
}

// MARK: - Validation

extension NumericalDiagnostics {
    /// Check if diagnostics indicate healthy simulation
    public var isHealthy: Bool {
        // Convergence check
        guard converged else { return false }

        // Conservation checks (within 1% tolerance)
        guard abs(particleDrift) < 0.01 else { return false }
        guard abs(energyDrift) < 0.01 else { return false }
        guard abs(currentDrift) < 0.01 else { return false }

        return true
    }

    /// Warning level (0 = healthy, 1 = warning, 2 = critical)
    public var warningLevel: Int {
        if !converged { return 2 }

        let maximumDrift = max(
            abs(particleDrift),
            abs(energyDrift),
            abs(currentDrift)
        )

        if maximumDrift > 0.05 { return 2 }  // > 5% drift
        if maximumDrift > 0.01 { return 1 }  // > 1% drift
        return 0  // Healthy
    }
}
