import Foundation
import MLX
import Logging

// MARK: - Hybrid Linear Solver

// Logger for the hybrid linear solver
private let logger = Logger(label: "com.gotenx.core.linear")

/// Iterative linear solver with row normalization preconditioning
///
/// **Design Policy**: For Newton-Raphson Jacobian matrices (typical size 400×400):
/// - Always use iterative SOR solver with row normalization preconditioning
/// - Jacobians are inherently ill-conditioned → condition number estimation is unnecessary
/// - Row normalization is more stable than diagonal preconditioning for Float32
///
/// **Why not estimate condition number?**
/// - Estimation requires ~100 SOR iterations on 400×400 matrix → takes minutes
/// - Newton-Raphson Jacobians are always ill-conditioned (κ >> 1e10)
/// - Direct solver (MLX.solve) would fail anyway for ill-conditioned systems
///
/// **Performance**:
/// - Iterative solver with preconditioning: converges in seconds
/// - Typical convergence: 10-100 iterations for Newton-Raphson Jacobians
public struct HybridLinearSolver: Sendable {

    /// Iterative solver tolerance
    ///
    /// Stopping criterion: ||x_new - x_old|| / ||x_new|| < tolerance
    public let iterativeTolerance: Float

    /// Maximum iterations for iterative solver
    public let maxIterations: Int

    /// SOR relaxation parameter (ω ∈ [1, 2])
    ///
    /// - ω = 1.0: Gauss-Seidel
    /// - ω ∈ (1, 2): Over-relaxation (faster convergence for some systems)
    /// - ω = 1.5: Typical default (good for many PDEs)
    public let omega: Float

    /// Create iterative linear solver
    ///
    /// - Parameters:
    ///   - iterativeTolerance: Convergence tolerance (default: 1e-8)
    ///   - maxIterations: Maximum iterations (default: 10000)
    ///   - omega: SOR relaxation parameter (default: 1.5)
    public init(
        iterativeTolerance: Float = 1e-8,
        maxIterations: Int = 10000,
        omega: Float = 1.5
    ) {
        self.iterativeTolerance = iterativeTolerance
        self.maxIterations = maxIterations
        self.omega = omega
    }

    /// Solve linear system Ax = b using direct solver (MLX.solve on CPU) or iterative SOR
    ///
    /// **Strategy**: Try direct solver first (fast, accurate), fall back to iterative if needed.
    ///
    /// - Parameters:
    ///   - A: System matrix [n, n]
    ///   - b: Right-hand side [n]
    ///   - usePreconditioner: Apply row normalization preconditioning for iterative solver (default: true)
    /// - Returns: Solution x [n]
    /// - Throws: SolverError if solution fails to converge
    public func solve(_ A: MLXArray, _ b: MLXArray, usePreconditioner: Bool = true) throws -> MLXArray {
        // Direct solve with two-sided equilibration + iterative refinement.
        //
        // The coupled-transport Jacobian becomes badly scaled and ill-conditioned
        // once stiff source-term derivatives appear (single entries reaching ~10¹¹,
        // κ ≈ 10⁵). In Float32 a raw LU solve then loses ~2 digits, leaving
        // ||A·x − b||/||b|| of a few percent and an unusable Newton direction.
        //
        // 1. Symmetric equilibration `Â = Dr·A·Dc` (Dr=1/√rowNorm, Dc=1/√colNorm)
        //    brings every row and column to ~unit norm, collapsing κ to O(1). The
        //    scaling norms are floored relative to the largest norm so genuinely
        //    decoupled (near-zero) rows/columns — e.g. a non-evolved channel — are
        //    never amplified.
        // 2. Iterative refinement on the *equilibrated* system drives Â·y → Dr·b to
        //    near machine precision; because x = Dc·y, the recovered x solves the
        //    original A·x = b accurately. For an already well-conditioned system this
        //    is a no-op, so it cannot perturb steps that were converging.
        let n = A.shape[0]
        let rowNorms = MLX.norm(A, ord: 2, axis: 1, keepDims: false)
        let colNorms = MLX.norm(A, ord: 2, axis: 0, keepDims: false)
        let rowFloor = rowNorms.max() * 1e-6
        let colFloor = colNorms.max() * 1e-6
        let dr = 1.0 / sqrt(maximum(rowNorms, rowFloor))
        let dc = 1.0 / sqrt(maximum(colNorms, colFloor))

        let aEq = dr.reshaped([n, 1]) * A * dc.reshaped([1, n])
        let bEq = dr * b

        var y = MLX.solve(aEq, bEq, stream: .cpu)
        for _ in 0..<2 {
            let residual = bEq - aEq.matmul(y)
            y = y + MLX.solve(aEq, residual, stream: .cpu)
        }
        let x = dc * y

        // Verify solution is valid (.item() forces evaluation of x).
        let xMin = x.min(keepDims: false).item(Float.self)
        let xMax = x.max(keepDims: false).item(Float.self)

        if xMin.isFinite && xMax.isFinite {
            return x
        }
        logger.warning("Direct solver returned inf/nan, falling back to iterative")

        // Fallback: iterative solver
        logger.debug("Using iterative solver", metadata: [
            "preconditioner": "\(usePreconditioner ? "row-normalization" : "none")"
        ])

        if usePreconditioner {
            return try solveWithPreconditioning(A, b)
        } else {
            return try solveIterative(A, b)
        }
    }

    // MARK: - Row Normalization Preconditioning

    /// Solve preconditioned system using row normalization
    ///
    /// **GPU-First Design**: All operations use MLXArray element-wise arithmetic.
    ///
    /// **Purpose**: Transform ill-conditioned system into better-conditioned one:
    /// ```
    /// Original:  Ax = b
    /// Preconditioned: S·Ax = S·b  where S[i] = 1/||A[i,:]||
    /// ```
    ///
    /// **Row Normalization vs Diagonal Preconditioning**:
    /// - Diagonal: Uses only diagonal elements → unstable when diag has extreme range
    /// - Row Norm: Uses entire row magnitude → more stable for Float32
    ///
    /// **Benefits**:
    /// - More stable than diagonal preconditioning for ill-conditioned matrices
    /// - Avoids Float32 precision loss when diagonal has extreme range (e.g., [1e3, 1e8])
    /// - Pure GPU operations (no CPU transfers)
    ///
    /// **Example**:
    /// ```
    /// // Diagonal range [1e3, 4.74e8] → κ_D ≈ 4.8e5
    /// // Diagonal preconditioning: 1/4.74e8 = 2.1e-9 → underflow in Float32
    /// // Row normalization: ||row|| typically within [1e3, 1e6] → 1e-6 is safe
    /// ```
    ///
    /// - Parameters:
    ///   - A: Original system matrix [n, n]
    ///   - b: Original right-hand side [n]
    /// - Returns: Solution x [n]
    /// - Throws: SolverError if solution fails to converge
    private func solveWithPreconditioning(_ A: MLXArray, _ b: MLXArray) throws -> MLXArray {
        let n = A.shape[0]

        // Compute row norms: ||A[i,:]|| for each row
        let rowNorms = MLX.norm(A, ord: 2, axis: 1, keepDims: false)  // [n]
        eval(rowNorms)

        // Check conditioning
        let minNorm = rowNorms.min(keepDims: false)
        let maxNorm = rowNorms.max(keepDims: false)
        let normB = MLX.norm(b)
        eval(minNorm, maxNorm, normB)

        let minNormValue = minNorm.item(Float.self)
        let maxNormValue = maxNorm.item(Float.self)
        let normBValue = normB.item(Float.self)

        print("[DEBUG-LINEAR] Row norm range: [\(String(format: "%.2e", minNormValue)), \(String(format: "%.2e", maxNormValue))]")
        print("[DEBUG-LINEAR] RHS norm: \(String(format: "%.2e", normBValue))")

        let rowCondition = maxNormValue / (minNormValue + 1e-30)
        print("[DEBUG-LINEAR] Row norm condition: \(String(format: "%.2e", rowCondition))")

        if minNormValue < 1e-12 {
            print("[HybridLinearSolver] Warning: Some rows have very small norm (min=\(String(format: "%.2e", minNormValue)))")
        }

        // Compute scaling factors: S[i] = 1/||A[i,:]||
        let rowNormsSafe = maximum(rowNorms, MLXArray(1e-10))
        let S = 1.0 / rowNormsSafe  // [n]
        eval(S)

        // Check if S has reasonable values (avoid Float32 underflow)
        let minS = S.min(keepDims: false).item(Float.self)
        let maxS = S.max(keepDims: false).item(Float.self)
        print("[DEBUG-LINEAR] Scaling factor range: [\(String(format: "%.2e", minS)), \(String(format: "%.2e", maxS))]")

        // Precondition system: A' = S·A, b' = S·b
        // Use broadcasting to apply row-wise scaling
        let S_2d = S.reshaped([n, 1])  // [n, 1] for broadcasting
        let A_precond = S_2d * A  // [n, 1] * [n, n] → [n, n] (row-wise scaling)
        let b_precond = S * b  // [n] * [n] → [n] (element-wise)
        eval(A_precond, b_precond)

        // Solve preconditioned system: S·Ax = S·b
        print("[HybridLinearSolver] Applying row normalization preconditioning")
        let x = try solveIterative(A_precond, b_precond)

        // Note: The solution x from the preconditioned system S·Ax = S·b
        // is mathematically identical to the solution of Ax = b (left preconditioning).
        // No back-transformation is needed.

        return x
    }

    // MARK: - Iterative Solver (SOR)

    /// Solve using Successive Over-Relaxation (SOR)
    ///
    /// SOR iteration: x_new[i] = (1-ω)x[i] + (ω/a_ii)(b[i] - Σ a_ij x_j)
    ///
    /// - Parameters:
    ///   - A: System matrix [n, n]
    ///   - b: Right-hand side [n]
    /// - Returns: Solution x [n]
    /// - Throws: SolverError.convergenceFailure if not converged
    private func solveIterative(_ A: MLXArray, _ b: MLXArray) throws -> MLXArray {
        let n = A.shape[0]
        var x = MLXArray.zeros([n])

        for iter in 0..<maxIterations {
            let xOld = x
            x = sorIteration(A, b, x: x, omega: omega, iterations: 1)

            // Check convergence
            let diff = x - xOld

            // CRITICAL: Force evaluation before calling .item()
            // MLX.norm() returns lazy MLXArray
            let normDiffArray = MLX.norm(diff)
            let normXArray = MLX.norm(x)
            eval(normDiffArray, normXArray)

            let relativeChange = normDiffArray.item(Float.self) / (normXArray.item(Float.self) + 1e-10)

            // Debug first 3 iterations
            if iter < 3 {
                let xMinMax = (x.min(keepDims: false), x.max(keepDims: false))
                eval(xMinMax.0, xMinMax.1)
                print("[DEBUG-SOR] iter=\(iter): x range=[\(String(format: "%.2e", xMinMax.0.item(Float.self))), \(String(format: "%.2e", xMinMax.1.item(Float.self)))], rel_change=\(String(format: "%.2e", relativeChange))")
            }

            if relativeChange < iterativeTolerance {
                print("[HybridLinearSolver] Converged in \(iter + 1) iterations (rel_change=\(String(format: "%.2e", relativeChange)))")
                return x
            }

            // Early stopping if diverging
            if relativeChange > 1e6 || !relativeChange.isFinite {
                print("[HybridLinearSolver] SOR diverging (rel_change=\(String(format: "%.2e", relativeChange)))")
                throw SolverError.convergenceFailure(
                    iterations: iter + 1,
                    residualNorm: relativeChange
                )
            }
        }

        print("[HybridLinearSolver] SOR did not converge in \(maxIterations) iterations")
        throw SolverError.convergenceFailure(
            iterations: maxIterations,
            residualNorm: Float.infinity
        )
    }

    /// True SOR iteration (HIGH #4 FIX)
    ///
    /// Implements proper Successive Over-Relaxation with forward sweep.
    /// Uses updated values immediately (Gauss-Seidel style + over-relaxation).
    ///
    /// Algorithm: x_new[i] = (1-ω)x[i] + (ω/A[i,i]) * (b[i] - Σ(j≠i) A[i,j]*x[j])
    ///
    /// Note: This requires a sequential forward sweep, so it's less vectorized
    /// than the previous Jacobi-style implementation. However, it converges
    /// 2-10x faster for many PDE systems.
    ///
    /// - Parameters:
    ///   - A: System matrix [n, n]
    ///   - b: Right-hand side [n]
    ///   - x: Current solution [n]
    ///   - omega: Relaxation parameter (1.0 = Gauss-Seidel, >1.0 = over-relaxation)
    ///   - iterations: Number of SOR sweeps
    /// - Returns: Updated solution [n]
    private func sorIteration(
        _ A: MLXArray,
        _ b: MLXArray,
        x: MLXArray,
        omega: Float,
        iterations: Int
    ) -> MLXArray {
        var xCurrent = x
        let n = A.shape[0]

        // Extract diagonal once (vectorized)
        let diag = extractDiagonal(A, n: n)  // [n]

        for sweepIter in 0..<iterations {
            // True SOR: Forward sweep with immediate updates
            // Note: This loop is necessary for Gauss-Seidel convergence properties
            for i in 0..<n {
                // Compute: b[i] - Σ(j≠i) A[i,j] * x[j]
                // Use row slice and dot product for efficiency
                let A_row = A[i, 0..<n]  // Get entire row [n]

                // CRITICAL: Force evaluation before calling .item()
                // All of these are lazy MLXArrays (slices and arithmetic results)
                let axArray = (A_row * xCurrent).sum()
                let b_i_array = b[i]
                let a_ii_array = diag[i]
                let x_i_old_array = xCurrent[i]
                eval(axArray, b_i_array, a_ii_array, x_i_old_array)

                let ax = axArray.item(Float.self)  // A[i,:] · x
                let b_i = b_i_array.item(Float.self)
                let a_ii = a_ii_array.item(Float.self)
                let x_i_old = x_i_old_array.item(Float.self)

                // SOR update: x_new[i] = (1-ω)x[i] + (ω/a_ii)(b[i] - Σ(j≠i) a_ij*x_j)
                // Note: ax includes a_ii*x_i, so we subtract it
                let residual_i = b_i - (ax - a_ii * x_i_old)
                let x_i_new = (1.0 - omega) * x_i_old + (omega / (a_ii + 1e-10)) * residual_i

                // Debug first element on first sweep
                if sweepIter == 0 && i == 0 {
                    print("[DEBUG-SOR-DETAIL] i=0: b_i=\(String(format: "%.2e", b_i)), a_ii=\(String(format: "%.2e", a_ii)), ax=\(String(format: "%.2e", ax)), x_old=\(String(format: "%.2e", x_i_old)), x_new=\(String(format: "%.2e", x_i_new))")
                }

                // Update x[i] immediately (key difference from Jacobi)
                xCurrent[i] = MLXArray(x_i_new)
            }

            // Ensure computation is complete before next iteration
            eval(xCurrent)
        }

        return xCurrent
    }

    /// Extract diagonal from matrix (VECTORIZED - MEDIUM #6 FIX)
    ///
    /// - Parameters:
    ///   - A: Matrix [n, n]
    ///   - n: Matrix size
    /// - Returns: Diagonal elements [n]
    private func extractDiagonal(_ A: MLXArray, n: Int) -> MLXArray {
        // MEDIUM #6 FIX: Vectorized diagonal extraction
        // Create index array [0, 1, 2, ..., n-1]
        let indices = MLXArray(0..<n)

        // Extract diagonal using advanced indexing: A[indices, indices]
        // This extracts A[0,0], A[1,1], A[2,2], ..., A[n-1,n-1] in one operation
        let diag = A[indices, indices]

        return diag
    }
}
