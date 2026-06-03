import MLX
import Testing
@testable import GotenxCore

@Suite("Hybrid Linear Solver")
struct HybridLinearSolverTests {
    @Test("Small diagonal systems solve through the Metal candidate path")
    func smallDiagonalSystemSolves() throws {
        let diagonal = MLXArray([Float(2.0), Float(4.0), Float(8.0)])
        let matrix = MLXArray.eye(3) * diagonal.reshaped([3, 1])
        let rhs = MLXArray([Float(2.0), Float(8.0), Float(24.0)])
        let solver = HybridLinearSolver(
            gpuMaxIterations: 16,
            gpuResidualTolerance: 1e-5,
            gpuDimensionLimit: 3
        )

        let solution = try solver.solve(matrix, rightHandSide: rhs).asArray(Float.self)

        #expect(abs(solution[0] - 1.0) < 1e-4)
        #expect(abs(solution[1] - 2.0) < 1e-4)
        #expect(abs(solution[2] - 3.0) < 1e-4)
    }

    @Test("Large dense systems bypass Metal CGNR and remain accurate")
    func largeDenseSystemUsesDirectFallback() throws {
        let dimension = 80
        let diagonalValues = (0..<dimension).map { Float($0 + 2) }
        let expected = (0..<dimension).map { Float($0 + 1) }
        let diagonal = MLXArray(diagonalValues)
        let matrix = MLXArray.eye(dimension) * diagonal.reshaped([dimension, 1])
        let rhs = diagonal * MLXArray(expected)
        let solver = HybridLinearSolver(
            gpuMaxIterations: 16,
            gpuResidualTolerance: 1e-5,
            gpuDimensionLimit: 16
        )

        let solution = try solver.solve(matrix, rightHandSide: rhs).asArray(Float.self)
        let maxError = zip(solution, expected)
            .map { abs($0 - $1) }
            .max() ?? .infinity

        #expect(maxError < 1e-4)
    }

    @Test("Nonlocal dense coupling rejects raw block candidate and falls back accurately")
    func nonlocalDenseCouplingFallsBackAccurately() throws {
        let dimension = 12
        var matrixValues = [Float](repeating: 0, count: dimension * dimension)
        for index in 0..<dimension {
            matrixValues[index * dimension + index] = 1.0
        }
        matrixValues[0 * dimension + 8] = 10.0

        let matrix = MLXArray(matrixValues).reshaped([dimension, dimension])
        let expected = MLXArray([Float](repeating: 1.0, count: dimension))
        let rhs = matrix.matmul(expected)
        let solver = HybridLinearSolver(gpuMaxIterations: 0)

        let solution = try solver.solve(matrix, rightHandSide: rhs).asArray(Float.self)
        let maxError = solution
            .map { abs($0 - 1.0) }
            .max() ?? .infinity

        #expect(maxError < 1e-4)
    }
}
