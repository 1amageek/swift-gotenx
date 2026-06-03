import MLX
import Testing
@testable import GotenxCore

@Suite("Banded Jacobian")
struct BandedJacobianTests {
    @Test("Colored VJP matches dense VJP for local residuals", .timeLimit(.minutes(1)))
    func coloredVJPMatchesDenseVJPForLocalResiduals() throws {
        let layout = try FlattenedState.StateLayout(cellCount: 8)
        let cellCount = layout.cellCount
        let x = MLXArray.linspace(Float(0.1), Float(1.0), count: layout.totalSize)

        let residualFn: (MLXArray) -> MLXArray = { values in
            let ti = values[layout.tiRange]
            let te = values[layout.teRange]
            let ne = values[layout.neRange]
            let psi = values[layout.psiRange]

            let tiLeft = shiftedRight(ti, count: cellCount)
            let tiRight = shiftedLeft(ti, count: cellCount)
            let teLeft = shiftedRight(te, count: cellCount)
            let teRight = shiftedLeft(te, count: cellCount)
            let neLeft = shiftedRight(ne, count: cellCount)
            let neRight = shiftedLeft(ne, count: cellCount)
            let psiLeft = shiftedRight(psi, count: cellCount)
            let psiRight = shiftedLeft(psi, count: cellCount)

            return concatenated([
                2.0 * ti - 0.25 * tiLeft - 0.50 * tiRight + 0.10 * te,
                3.0 * te - 0.10 * teLeft - 0.20 * teRight + 0.20 * ti,
                1.5 * ne - 0.15 * neLeft - 0.05 * neRight + 0.30 * psi,
                1.2 * psi - 0.05 * psiLeft - 0.10 * psiRight + 0.40 * ne
            ], axis: 0)
        }

        let denseJacobian = computeJacobianViaVJP(residualFn, x)
        let bandedJacobian = computeBlockTriDiagonalJacobianViaColoredVJP(
            residualFn,
            x,
            layout: layout
        )
        let denseValues = denseJacobian.asArray(Float.self)
        let bandedValues = bandedJacobian.asArray(Float.self)
        let maxError = zip(denseValues, bandedValues)
            .map { abs($0 - $1) }
            .max() ?? .infinity

        #expect(maxError < 1e-4)
    }

}

private func shiftedLeft(_ values: MLXArray, count: Int) -> MLXArray {
    concatenated([
        values[1..<count],
        MLXArray.zeros([1])
    ], axis: 0)
}

private func shiftedRight(_ values: MLXArray, count: Int) -> MLXArray {
    concatenated([
        MLXArray.zeros([1]),
        values[0..<(count - 1)]
    ], axis: 0)
}
