import Foundation
import MLX
import Logging

// MARK: - Hybrid Linear Solver

// Logger for the hybrid linear solver
private let logger = Logger(label: "com.gotenx.core.linear")

private struct LinearSolveAttempt {
    let y: MLXArray
    let relativeResidual: Float
    let accepted: Bool
}

/// Dense linear solver with a bounded Metal iterative candidate and CPU LU fallback
///
/// **Design Policy**: For Newton-Raphson Jacobian matrices (typical size 400×400):
/// - First equilibrate rows and columns to keep the Float32 system well scaled.
/// - Try a bounded Metal-backed CGNR solve for systems where the candidate is
///   operationally cheaper than direct factorization.
/// - Use MLX CPU LU for larger dense systems until MLX exposes Metal LU
///   factorization.
///
/// **Performance**:
/// - CGNR uses only MLX matrix-vector products and reductions on Metal.
/// - The CPU LU path remains explicit and isolated because MLX's LU
///   factorization is not implemented on the Metal backend.
public struct HybridLinearSolver: Sendable {
    /// Maximum CGNR iterations for the Metal-backed solve.
    public let gpuMaxIterations: Int

    /// Relative residual threshold for accepting the Metal-backed solve.
    public let gpuResidualTolerance: Float

    /// Largest dense system size that should attempt the Metal CGNR candidate.
    public let gpuDimensionLimit: Int

    /// Create a dense linear solver.
    ///
    /// - Parameters:
    ///   - gpuMaxIterations: Maximum Metal CGNR iterations.
    ///   - gpuResidualTolerance: Relative residual threshold for accepting the Metal solve.
    ///   - gpuDimensionLimit: Largest dimension that should attempt Metal CGNR.
    public init(
        gpuMaxIterations: Int = 24,
        gpuResidualTolerance: Float = 1e-4,
        gpuDimensionLimit: Int = 64
    ) {
        self.gpuMaxIterations = gpuMaxIterations
        self.gpuResidualTolerance = gpuResidualTolerance
        self.gpuDimensionLimit = gpuDimensionLimit
    }

    /// Solve linear system Ax = b using a bounded Metal candidate with CPU LU fallback.
    ///
    /// **Strategy**: Try Metal CGNR only for bounded-size systems, then fall back
    /// to CPU LU if the candidate cannot meet the configured residual target.
    ///
    /// - Parameters:
    ///   - matrix: System matrix [n, n]
    ///   - rightHandSide: Right-hand side [n]
    /// - Returns: Solution x [n]
    /// - Throws: SolverError if solution fails to converge
    public func solve(_ matrix: MLXArray, rightHandSide: MLXArray) throws -> MLXArray {
        // Two-sided equilibration before the linear solve.
        //
        // The coupled-transport Jacobian becomes badly scaled and ill-conditioned
        // once stiff source-term derivatives appear (single entries reaching ~10¹¹,
        // κ ≈ 10⁵). In Float32, solving the raw system loses useful digits and
        // produces a poor Newton direction.
        //
        // Symmetric equilibration `Aeq = Dr·A·Dc` (Dr=1/sqrt(rowNorm),
        // Dc=1/sqrt(colNorm)) brings rows and columns near unit norm. The scaling
        // norms are floored relative to the largest norm so genuinely decoupled
        // near-zero rows or columns are not amplified.
        let n = matrix.shape[0]
        let rowNorms = MLX.norm(matrix, ord: 2, axis: 1, keepDims: false)
        let colNorms = MLX.norm(matrix, ord: 2, axis: 0, keepDims: false)
        let rowFloor = rowNorms.max() * 1e-6
        let colFloor = colNorms.max() * 1e-6
        let rowScaling = 1.0 / sqrt(maximum(rowNorms, rowFloor))
        let columnScaling = 1.0 / sqrt(maximum(colNorms, colFloor))

        let equilibratedMatrix = rowScaling.reshaped([n, 1]) * matrix * columnScaling.reshaped([1, n])
        let equilibratedRightHandSide = rowScaling * rightHandSide

        var fallbackResidual = Float.infinity
        if shouldAttemptMetalCGNR(dimension: n) {
            let gpuAttempt = solveEquilibratedWithMetalCGNR(equilibratedMatrix, equilibratedRightHandSide)
            let xGPU = columnScaling * gpuAttempt.y
            let gpuRelativeResidual = relativeResidual(A: matrix, x: xGPU, b: rightHandSide)
            fallbackResidual = gpuRelativeResidual
            if gpuAttempt.accepted && gpuRelativeResidual <= gpuResidualTolerance {
                logger.debug("Metal CGNR linear solve accepted", metadata: [
                    "relativeResidual": "\(String(format: "%.2e", gpuRelativeResidual))"
                ])
                return xGPU
            }

            logger.debug("Metal CGNR linear solve requires CPU LU fallback", metadata: [
                "equilibratedResidual": "\(String(format: "%.2e", gpuAttempt.relativeResidual))",
                "relativeResidual": "\(String(format: "%.2e", gpuRelativeResidual))",
                "threshold": "\(String(format: "%.2e", gpuResidualTolerance))"
            ])
        } else {
            logger.debug("Metal CGNR linear solve skipped for dense system", metadata: [
                "dimension": "\(n)",
                "limit": "\(gpuDimensionLimit)"
            ])
        }

        let yCPU = solveEquilibratedWithCPULU(equilibratedMatrix, equilibratedRightHandSide)
        let xCPU = columnScaling * yCPU
        let range = MLX.stacked([
            xCPU.min(keepDims: false),
            xCPU.max(keepDims: false)
        ], axis: 0).asArray(Float.self)

        if range[0].isFinite && range[1].isFinite {
            return xCPU
        }

        throw SolverError.convergenceFailure(
            iterations: gpuMaxIterations,
            residualNorm: fallbackResidual
        )
    }

    // MARK: - Metal CGNR Solve

    private func shouldAttemptMetalCGNR(dimension: Int) -> Bool {
        gpuMaxIterations > 0 && dimension <= gpuDimensionLimit
    }

    /// Solve the equilibrated system with conjugate gradients on the normal equations.
    ///
    /// The routine uses only MLX matrix-vector products and reductions, so it executes
    /// on the default MLX Metal stream. It performs a single host read at the end to
    /// decide whether the result is accurate enough for the inexact Newton step.
    private func solveEquilibratedWithMetalCGNR(_ A: MLXArray, _ b: MLXArray) -> LinearSolveAttempt {
        let n = A.shape[0]
        let aTranspose = A.T

        var y = MLXArray.zeros([n])
        var residual = b
        var gradient = aTranspose.matmul(residual)
        var direction = gradient
        var gradientNormSquared = (gradient * gradient).sum()

        for iteration in 0..<gpuMaxIterations {
            let aDirection = A.matmul(direction)
            let denominator = (aDirection * aDirection).sum() + 1e-20
            let alpha = gradientNormSquared / denominator

            y = y + alpha * direction
            residual = residual - alpha * aDirection

            let nextGradient = aTranspose.matmul(residual)
            let nextGradientNormSquared = (nextGradient * nextGradient).sum()
            let beta = nextGradientNormSquared / (gradientNormSquared + 1e-20)

            direction = nextGradient + beta * direction
            gradient = nextGradient
            gradientNormSquared = nextGradientNormSquared

            if (iteration + 1).isMultiple(of: 16) {
                eval(y, residual, gradient, direction, gradientNormSquared)
            }
        }

        let finalResidual = A.matmul(y) - b
        let normB = maximum(MLX.norm(b), MLXArray(1e-20))
        let metrics = MLX.stacked([
            MLX.norm(finalResidual) / normB,
            y.min(keepDims: false),
            y.max(keepDims: false)
        ], axis: 0).asArray(Float.self)

        let relativeResidual = metrics[0]
        let accepted = relativeResidual.isFinite
            && metrics[1].isFinite
            && metrics[2].isFinite
            && relativeResidual <= gpuResidualTolerance

        return LinearSolveAttempt(
            y: y,
            relativeResidual: relativeResidual,
            accepted: accepted
        )
    }

    private func relativeResidual(A: MLXArray, x: MLXArray, b: MLXArray) -> Float {
        let residual = A.matmul(x) - b
        let normB = maximum(MLX.norm(b), MLXArray(1e-20))
        let metrics = MLX.stacked([
            MLX.norm(residual) / normB,
            x.min(keepDims: false),
            x.max(keepDims: false)
        ], axis: 0).asArray(Float.self)

        guard metrics[0].isFinite, metrics[1].isFinite, metrics[2].isFinite else {
            return .infinity
        }
        return metrics[0]
    }

    // MARK: - CPU LU Fallback

    /// Solve the equilibrated system with MLX's CPU LU implementation.
    ///
    /// This path is deliberately isolated: MLX 0.31.3 does not implement LU
    /// factorization on the Metal backend, so `stream: .cpu` is the only direct
    /// dense solve available for the rare cases where CGNR is not accurate enough.
    private func solveEquilibratedWithCPULU(_ A: MLXArray, _ b: MLXArray) -> MLXArray {
        var y = MLX.solve(A, b, stream: .cpu)
        for _ in 0..<2 {
            let residual = b - A.matmul(y)
            y = y + MLX.solve(A, residual, stream: .cpu)
        }
        return y
    }
}
