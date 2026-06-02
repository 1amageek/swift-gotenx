import MLX
import Foundation

// MARK: - Solver Result

/// Result from PDE solver
public struct SolverResult: Sendable {
    /// Updated profiles after solving
    public let updatedProfiles: CoreProfiles

    /// Number of iterations taken
    public let iterations: Int

    /// Final residual norm
    public let residualNorm: Float

    /// Whether the solver converged
    public let converged: Bool

    /// Solver-specific metadata
    public let metadata: [String: Float]

    public init(
        updatedProfiles: CoreProfiles,
        iterations: Int,
        residualNorm: Float,
        converged: Bool,
        metadata: [String: Float] = [:]
    ) {
        self.updatedProfiles = updatedProfiles
        self.iterations = iterations
        self.residualNorm = residualNorm
        self.converged = converged
        self.metadata = metadata
    }
}

// MARK: - Solver Error

/// Errors that can occur during PDE solving
public enum SolverError: Error, Sendable {
    case convergenceFailure(iterations: Int, residualNorm: Float)
    case singularMatrix
    case invalidInput(String)
    case numericalInstability
    case maximumIterationsExceeded
}
