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

private struct BlockTriDiagonalSolveResult {
    let values: [Float]
    let relativeResidual: Float
    let accepted: Bool
}

/// Dense linear solver with a block-tridiagonal candidate, direct MLX LU fallback, and an opt-in Metal iterative candidate
///
/// **Design Policy**: For Newton-Raphson Jacobian matrices (typical size 400×400):
/// - Try a cell-wise block-tridiagonal solve in the original scale first. The
///   coupled 1D PDE residual is normally local in radius, so the Jacobian is block
///   tridiagonal after grouping `[Ti, Te, ne, psi]` by cell.
/// - Accept the block candidate only when the full dense residual confirms it. If
///   source/transport physics introduces wider coupling or the raw scale is too
///   ill-conditioned, retry after equilibration and then fall back to direct LU.
/// - Equilibrate rows and columns before dense/iterative fallback to keep Float32
///   systems well scaled.
/// - Keep the Metal-backed CGNR path opt-in for small or well-conditioned systems
///   where the iterative candidate can meet the configured residual target.
///
/// **Performance**:
/// - CGNR uses only MLX matrix-vector products and reductions on Metal.
/// - The CPU LU path remains explicit and isolated because MLX's LU
///   factorization is not implemented on the Metal backend.
public struct HybridLinearSolver: Sendable {
    /// Maximum CGNR iterations for the Metal-backed solve.
    ///
    /// `0` disables the iterative candidate and uses direct MLX LU after equilibration.
    public let gpuMaxIterations: Int

    /// Relative residual threshold for accepting the Metal-backed solve.
    public let gpuResidualTolerance: Float

    /// Largest dense system size that should attempt the Metal CGNR candidate.
    public let gpuDimensionLimit: Int

    /// Number of iterative-refinement corrections after the direct LU solve.
    ///
    /// `0` uses the equilibrated LU solution directly. Newton still verifies the
    /// resulting direction with its linear residual and merit-descent checks.
    public let cpuRefinementIterations: Int

    /// Whether to try the cell-wise block-tridiagonal solve before direct LU.
    public let usesBlockTridiagonalCandidate: Bool

    /// Relative residual threshold for accepting the block-tridiagonal candidate.
    public let blockTridiagonalResidualTolerance: Float

    /// Create a dense linear solver.
    ///
    /// - Parameters:
    ///   - gpuMaxIterations: Maximum Metal CGNR iterations. Use `0` for direct LU.
    ///   - gpuResidualTolerance: Relative residual threshold for accepting the Metal solve.
    ///   - gpuDimensionLimit: Largest dimension that should attempt Metal CGNR.
    ///   - cpuRefinementIterations: Number of direct-LU refinement corrections.
    ///   - usesBlockTridiagonalCandidate: Whether to try the block-tridiagonal solve before direct LU.
    ///   - blockTridiagonalResidualTolerance: Relative residual threshold for accepting the block candidate.
    public init(
        gpuMaxIterations: Int = 0,
        gpuResidualTolerance: Float = 1e-4,
        gpuDimensionLimit: Int = 64,
        cpuRefinementIterations: Int = 0,
        usesBlockTridiagonalCandidate: Bool = true,
        blockTridiagonalResidualTolerance: Float = 5e-2
    ) {
        precondition(cpuRefinementIterations >= 0, "CPU refinement iterations must be non-negative")
        precondition(blockTridiagonalResidualTolerance > 0, "Block-tridiagonal residual tolerance must be positive")
        self.gpuMaxIterations = gpuMaxIterations
        self.gpuResidualTolerance = gpuResidualTolerance
        self.gpuDimensionLimit = gpuDimensionLimit
        self.cpuRefinementIterations = cpuRefinementIterations
        self.usesBlockTridiagonalCandidate = usesBlockTridiagonalCandidate
        self.blockTridiagonalResidualTolerance = blockTridiagonalResidualTolerance
    }

    /// Solve linear system Ax = b using a verified block solve, equilibration, and MLX LU fallback.
    ///
    /// **Strategy**: Try the block-tridiagonal candidate in the original scale first
    /// for the expected `[Ti, Te, ne, psi] × radial-cell` structure. If that cannot
    /// meet the configured residual target, equilibrate and retry bounded candidates.
    /// Fall back to direct CPU LU whenever no candidate is accurate enough.
    ///
    /// - Parameters:
    ///   - matrix: System matrix [n, n]
    ///   - rightHandSide: Right-hand side [n]
    /// - Returns: Solution x [n]
    /// - Throws: SolverError if solution fails to converge
    public func solve(_ matrix: MLXArray, rightHandSide: MLXArray) throws -> MLXArray {
        let n = matrix.shape[0]
        var fallbackResidual = Float.infinity

        if shouldAttemptBlockTridiagonal(dimension: n) {
            let rawBlockAttempt = solveWithBlockTridiagonal(matrix, rightHandSide)
            fallbackResidual = rawBlockAttempt.relativeResidual
            if rawBlockAttempt.accepted && rawBlockAttempt.relativeResidual <= blockTridiagonalResidualTolerance {
                let yBlock = MLXArray(rawBlockAttempt.values)
                logger.debug("Raw block-tridiagonal linear solve accepted", metadata: [
                    "relativeResidual": "\(String(format: "%.2e", rawBlockAttempt.relativeResidual))"
                ])
                return yBlock
            }

            logger.debug("Raw block-tridiagonal linear solve requires equilibrated fallback", metadata: [
                "relativeResidual": "\(String(format: "%.2e", rawBlockAttempt.relativeResidual))",
                "threshold": "\(String(format: "%.2e", blockTridiagonalResidualTolerance))"
            ])
        }

        // Two-sided equilibration before fallback linear solves.
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
        let rowNorms = MLX.norm(matrix, ord: 2, axis: 1, keepDims: false)
        let colNorms = MLX.norm(matrix, ord: 2, axis: 0, keepDims: false)
        let rowFloor = rowNorms.max() * 1e-6
        let colFloor = colNorms.max() * 1e-6
        let rowScaling = 1.0 / sqrt(maximum(rowNorms, rowFloor))
        let columnScaling = 1.0 / sqrt(maximum(colNorms, colFloor))

        let equilibratedMatrix = rowScaling.reshaped([n, 1]) * matrix * columnScaling.reshaped([1, n])
        let equilibratedRightHandSide = rowScaling * rightHandSide

        if shouldAttemptBlockTridiagonal(dimension: n) {
            let blockAttempt = solveWithBlockTridiagonal(
                equilibratedMatrix,
                equilibratedRightHandSide,
                rowScaling: rowScaling
            )
            let blockRelativeResidual = blockAttempt.relativeResidual
            fallbackResidual = blockRelativeResidual
            if blockAttempt.accepted && blockRelativeResidual <= blockTridiagonalResidualTolerance {
                let xBlock = columnScaling * MLXArray(blockAttempt.values)
                logger.debug("Block-tridiagonal linear solve accepted", metadata: [
                    "relativeResidual": "\(String(format: "%.2e", blockRelativeResidual))"
                ])
                return xBlock
            }

            logger.debug("Block-tridiagonal linear solve requires direct LU fallback", metadata: [
                "relativeResidual": "\(String(format: "%.2e", blockRelativeResidual))",
                "threshold": "\(String(format: "%.2e", blockTridiagonalResidualTolerance))"
            ])
        }

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
            logger.debug("Metal CGNR linear solve skipped", metadata: [
                "dimension": "\(n)",
                "limit": "\(gpuDimensionLimit)",
                "reason": "\(gpuMaxIterations <= 0 ? "disabled" : "dimension_limit")"
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

    private func shouldAttemptBlockTridiagonal(dimension: Int) -> Bool {
        usesBlockTridiagonalCandidate && dimension > 0 && dimension.isMultiple(of: 4)
    }

    private func shouldAttemptMetalCGNR(dimension: Int) -> Bool {
        gpuMaxIterations > 0 && dimension <= gpuDimensionLimit
    }

    // MARK: - Block-Tridiagonal Solve

    private func solveWithBlockTridiagonal(
        _ A: MLXArray,
        _ b: MLXArray,
        rowScaling: MLXArray? = nil
    ) -> BlockTriDiagonalSolveResult {
        let n = A.shape[0]
        let cellCount = n / 4
        let matrix = A.asArray(Float.self)
        let rightHandSide = b.asArray(Float.self)
        let rowScalingValues = rowScaling?.asArray(Float.self)

        return solveCellMajorBlockTridiagonal(
            matrix: matrix,
            rightHandSide: rightHandSide,
            rowScaling: rowScalingValues,
            cellCount: cellCount,
            dimension: n
        )
    }

    private func solveCellMajorBlockTridiagonal(
        matrix: [Float],
        rightHandSide: [Float],
        rowScaling: [Float]?,
        cellCount: Int,
        dimension: Int
    ) -> BlockTriDiagonalSolveResult {
        var lowerBlocks = [Float](repeating: 0, count: cellCount * 16)
        var diagonalBlocks = [Float](repeating: 0, count: cellCount * 16)
        var upperBlocks = [Float](repeating: 0, count: cellCount * 16)
        var rhsBlocks = [Float](repeating: 0, count: cellCount * 4)

        for cell in 0..<cellCount {
            let currentBlockOffset = blockOffset(cell)
            let currentVectorOffset = vectorOffset(cell)
            for rowComponent in 0..<4 {
                let row = variableMajorIndex(cell: cell, component: rowComponent, cellCount: cellCount)
                rhsBlocks[currentVectorOffset + rowComponent] = rightHandSide[row]

                for columnComponent in 0..<4 {
                    let diagonalColumn = variableMajorIndex(cell: cell, component: columnComponent, cellCount: cellCount)
                    diagonalBlocks[currentBlockOffset + blockIndex(rowComponent, columnComponent)] =
                        matrix[row * dimension + diagonalColumn]

                    if cell > 0 {
                        let lowerColumn = variableMajorIndex(cell: cell - 1, component: columnComponent, cellCount: cellCount)
                        lowerBlocks[currentBlockOffset + blockIndex(rowComponent, columnComponent)] =
                            matrix[row * dimension + lowerColumn]
                    }

                    if cell + 1 < cellCount {
                        let upperColumn = variableMajorIndex(cell: cell + 1, component: columnComponent, cellCount: cellCount)
                        upperBlocks[currentBlockOffset + blockIndex(rowComponent, columnComponent)] =
                            matrix[row * dimension + upperColumn]
                    }
                }
            }
        }

        var modifiedUpper = [Float](repeating: 0, count: cellCount * 16)
        var modifiedRHS = [Float](repeating: 0, count: cellCount * 4)
        var scratchMatrix = [Float](repeating: 0, count: 16)
        var scratchColumn = [Float](repeating: 0, count: 4)
        var scratchSolution = [Float](repeating: 0, count: 4)
        var eliminatedDiagonal = [Float](repeating: 0, count: 16)
        var eliminatedRHS = [Float](repeating: 0, count: 4)
        var solutionScratch = [Float](repeating: 0, count: 4)

        guard solveBlockMatrix(
            diagonalBlocks,
            matrixOffset: 0,
            upperBlocks,
            rightHandSideOffset: 0,
            into: &modifiedUpper,
            resultOffset: 0,
            scratchMatrix: &scratchMatrix,
            scratchColumn: &scratchColumn,
            scratchSolution: &scratchSolution
        ),
            solveBlockVector(
                diagonalBlocks,
                matrixOffset: 0,
                rhsBlocks,
                rightHandSideOffset: 0,
                into: &modifiedRHS,
                vectorResultOffset: 0,
                scratchMatrix: &scratchMatrix
            )
        else {
            return BlockTriDiagonalSolveResult(
                values: [Float](repeating: .nan, count: dimension),
                relativeResidual: .infinity,
                accepted: false
            )
        }

        if cellCount > 1 {
            for cell in 1..<cellCount {
                let currentBlockOffset = blockOffset(cell)
                let currentVectorOffset = vectorOffset(cell)
                let previousBlockOffset = blockOffset(cell - 1)
                let previousVectorOffset = vectorOffset(cell - 1)

                subtractBlockProduct(
                    diagonalBlocks,
                    diagonalOffset: currentBlockOffset,
                    lowerBlocks,
                    lowerOffset: currentBlockOffset,
                    modifiedUpper,
                    upperOffset: previousBlockOffset,
                    into: &eliminatedDiagonal
                )
                subtractBlockVectorProduct(
                    rhsBlocks,
                    rightHandSideOffset: currentVectorOffset,
                    lowerBlocks,
                    lowerOffset: currentBlockOffset,
                    modifiedRHS,
                    vectorOffset: previousVectorOffset,
                    into: &eliminatedRHS
                )

                guard solveBlockMatrix(
                    eliminatedDiagonal,
                    matrixOffset: 0,
                    upperBlocks,
                    rightHandSideOffset: currentBlockOffset,
                    into: &modifiedUpper,
                    resultOffset: currentBlockOffset,
                    scratchMatrix: &scratchMatrix,
                    scratchColumn: &scratchColumn,
                    scratchSolution: &scratchSolution
                ),
                    solveBlockVector(
                        eliminatedDiagonal,
                        matrixOffset: 0,
                        eliminatedRHS,
                        rightHandSideOffset: 0,
                        into: &modifiedRHS,
                        vectorResultOffset: currentVectorOffset,
                        scratchMatrix: &scratchMatrix
                    )
                else {
                    return BlockTriDiagonalSolveResult(
                        values: [Float](repeating: .nan, count: dimension),
                        relativeResidual: .infinity,
                        accepted: false
                    )
                }
            }
        }

        var solutionBlocks = [Float](repeating: 0, count: cellCount * 4)
        let lastVectorOffset = vectorOffset(cellCount - 1)
        for component in 0..<4 {
            solutionBlocks[lastVectorOffset + component] = modifiedRHS[lastVectorOffset + component]
        }

        if cellCount > 1 {
            for cell in stride(from: cellCount - 2, through: 0, by: -1) {
                let currentBlockOffset = blockOffset(cell)
                let currentVectorOffset = vectorOffset(cell)
                let nextVectorOffset = vectorOffset(cell + 1)
                subtractBlockVectorProduct(
                    modifiedRHS,
                    rightHandSideOffset: currentVectorOffset,
                    modifiedUpper,
                    lowerOffset: currentBlockOffset,
                    solutionBlocks,
                    vectorOffset: nextVectorOffset,
                    into: &solutionScratch
                )
                for component in 0..<4 {
                    solutionBlocks[currentVectorOffset + component] = solutionScratch[component]
                }
            }
        }

        var solution = [Float](repeating: 0, count: dimension)
        for cell in 0..<cellCount {
            let currentVectorOffset = vectorOffset(cell)
            for component in 0..<4 {
                let index = variableMajorIndex(cell: cell, component: component, cellCount: cellCount)
                let value = solutionBlocks[currentVectorOffset + component]
                guard value.isFinite else {
                    return BlockTriDiagonalSolveResult(
                        values: solution,
                        relativeResidual: .infinity,
                        accepted: false
                    )
                }
                solution[index] = value
            }
        }

        let residual = originalScaleRelativeResidual(
            matrix: matrix,
            rightHandSide: rightHandSide,
            solution: solution,
            rowScaling: rowScaling,
            dimension: dimension
        )
        return BlockTriDiagonalSolveResult(
            values: solution,
            relativeResidual: residual,
            accepted: residual.isFinite
        )
    }

    private func originalScaleRelativeResidual(
        matrix: [Float],
        rightHandSide: [Float],
        solution: [Float],
        rowScaling: [Float]?,
        dimension: Int
    ) -> Float {
        var residualNormSquared: Double = 0
        var rightHandSideNormSquared: Double = 0

        for row in 0..<dimension {
            let scale = rowScaling?[row] ?? 1.0
            guard scale.isFinite, abs(scale) > 0 else {
                return .infinity
            }

            var ax: Double = 0
            let rowOffset = row * dimension
            for column in 0..<dimension {
                ax += Double(matrix[rowOffset + column]) * Double(solution[column])
            }

            let residual = (ax - Double(rightHandSide[row])) / Double(scale)
            let rhs = Double(rightHandSide[row]) / Double(scale)
            guard residual.isFinite, rhs.isFinite else {
                return .infinity
            }
            residualNormSquared += residual * residual
            rightHandSideNormSquared += rhs * rhs
        }

        let denominator = max(sqrt(rightHandSideNormSquared), 1e-20)
        let relativeResidual = sqrt(residualNormSquared) / denominator
        guard relativeResidual.isFinite else {
            return .infinity
        }
        return Float(relativeResidual)
    }

    private func variableMajorIndex(cell: Int, component: Int, cellCount: Int) -> Int {
        component * cellCount + cell
    }

    private func blockOffset(_ cell: Int) -> Int {
        cell * 16
    }

    private func vectorOffset(_ cell: Int) -> Int {
        cell * 4
    }

    private func blockIndex(_ row: Int, _ column: Int) -> Int {
        row * 4 + column
    }

    private func subtractBlockProduct(
        _ diagonal: [Float],
        diagonalOffset: Int,
        _ lower: [Float],
        lowerOffset: Int,
        _ upper: [Float],
        upperOffset: Int,
        into result: inout [Float]
    ) {
        for row in 0..<4 {
            for column in 0..<4 {
                var sum: Float = 0
                for k in 0..<4 {
                    sum += lower[lowerOffset + blockIndex(row, k)] * upper[upperOffset + blockIndex(k, column)]
                }
                result[blockIndex(row, column)] = diagonal[diagonalOffset + blockIndex(row, column)] - sum
            }
        }
    }

    private func subtractBlockVectorProduct(
        _ rightHandSide: [Float],
        rightHandSideOffset: Int,
        _ matrix: [Float],
        lowerOffset: Int,
        _ vector: [Float],
        vectorOffset: Int,
        into result: inout [Float]
    ) {
        for row in 0..<4 {
            var sum: Float = 0
            for column in 0..<4 {
                sum += matrix[lowerOffset + blockIndex(row, column)] * vector[vectorOffset + column]
            }
            result[row] = rightHandSide[rightHandSideOffset + row] - sum
        }
    }

    private func solveBlockMatrix(
        _ matrix: [Float],
        matrixOffset: Int,
        _ rightHandSide: [Float],
        rightHandSideOffset: Int,
        into result: inout [Float],
        resultOffset: Int,
        scratchMatrix: inout [Float],
        scratchColumn: inout [Float],
        scratchSolution: inout [Float]
    ) -> Bool {
        for column in 0..<4 {
            for row in 0..<4 {
                scratchColumn[row] = rightHandSide[rightHandSideOffset + blockIndex(row, column)]
            }

            guard solveBlockVector(
                matrix,
                matrixOffset: matrixOffset,
                scratchColumn,
                rightHandSideOffset: 0,
                into: &scratchSolution,
                vectorResultOffset: 0,
                scratchMatrix: &scratchMatrix
            ) else {
                return false
            }

            for row in 0..<4 {
                result[resultOffset + blockIndex(row, column)] = scratchSolution[row]
            }
        }

        return true
    }

    private func solveBlockVector(
        _ matrix: [Float],
        matrixOffset: Int,
        _ rightHandSide: [Float],
        rightHandSideOffset: Int,
        into result: inout [Float],
        vectorResultOffset: Int,
        scratchMatrix: inout [Float]
    ) -> Bool {
        let pivotTolerance: Float = 1e-20

        for index in 0..<16 {
            scratchMatrix[index] = matrix[matrixOffset + index]
        }
        for row in 0..<4 {
            result[vectorResultOffset + row] = rightHandSide[rightHandSideOffset + row]
        }

        for column in 0..<4 {
            var pivotRow = column
            var pivotMagnitude = abs(scratchMatrix[blockIndex(column, column)])

            if column + 1 < 4 {
                for row in (column + 1)..<4 {
                    let magnitude = abs(scratchMatrix[blockIndex(row, column)])
                    if magnitude > pivotMagnitude {
                        pivotMagnitude = magnitude
                        pivotRow = row
                    }
                }
            }

            guard pivotMagnitude.isFinite, pivotMagnitude > pivotTolerance else {
                return false
            }

            if pivotRow != column {
                for entryColumn in 0..<4 {
                    scratchMatrix.swapAt(blockIndex(column, entryColumn), blockIndex(pivotRow, entryColumn))
                }
                result.swapAt(vectorResultOffset + column, vectorResultOffset + pivotRow)
            }

            let pivot = scratchMatrix[blockIndex(column, column)]
            if column + 1 < 4 {
                for row in (column + 1)..<4 {
                    let factor = scratchMatrix[blockIndex(row, column)] / pivot
                    scratchMatrix[blockIndex(row, column)] = 0
                    for entryColumn in (column + 1)..<4 {
                        scratchMatrix[blockIndex(row, entryColumn)] -= factor * scratchMatrix[blockIndex(column, entryColumn)]
                    }
                    result[vectorResultOffset + row] -= factor * result[vectorResultOffset + column]
                }
            }
        }

        for row in stride(from: 3, through: 0, by: -1) {
            var value = result[vectorResultOffset + row]
            if row + 1 < 4 {
                for column in (row + 1)..<4 {
                    value -= scratchMatrix[blockIndex(row, column)] * result[vectorResultOffset + column]
                }
            }

            let diagonal = scratchMatrix[blockIndex(row, row)]
            guard diagonal.isFinite, abs(diagonal) > pivotTolerance else {
                return false
            }

            let solvedValue = value / diagonal
            guard solvedValue.isFinite else {
                return false
            }
            result[vectorResultOffset + row] = solvedValue
        }

        return true
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
        for _ in 0..<cpuRefinementIterations {
            let residual = b - A.matmul(y)
            y = y + MLX.solve(A, residual, stream: .cpu)
        }
        return y
    }
}
