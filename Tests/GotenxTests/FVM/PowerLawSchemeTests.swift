// PowerLawSchemeTests.swift
// Tests for Patankar power-law scheme

import Testing
import MLX
@testable import GotenxCore

@Suite("Power-Law Scheme Tests")
struct PowerLawSchemeTests {

    @Test("Péclet number calculation")
    func pecletNumber() {
        let faceConvectionVelocity = MLXArray([Float(0.0), Float(1.0), Float(10.0), Float(100.0)])
        let faceDiffusionCoefficient = MLXArray([Float(1.0), Float(1.0), Float(1.0), Float(1.0)])
        let cellSpacing = MLXArray(Float(1.0))

        let peclet = PowerLawScheme.computePecletNumber(
            faceConvectionVelocity: faceConvectionVelocity,
            faceDiffusionCoefficient: faceDiffusionCoefficient,
            cellSpacing: cellSpacing
        )
        eval(peclet)

        let result = peclet.asArray(Float.self)
        #expect(abs(result[0] - 0.0) < 1e-6)
        #expect(abs(result[1] - 1.0) < 1e-6)
        #expect(abs(result[2] - 10.0) < 1e-6)
        #expect(abs(result[3] - 100.0) < 1e-6)
    }

    @Test("Péclet number with non-uniform diffusion")
    func pecletNumberNonUniformDiffusion() {
        let faceConvectionVelocity = MLXArray([Float(10.0), Float(10.0), Float(10.0), Float(10.0)])
        let faceDiffusionCoefficient = MLXArray([Float(1.0), Float(2.0), Float(5.0), Float(10.0)])
        let cellSpacing = MLXArray(Float(1.0))

        let peclet = PowerLawScheme.computePecletNumber(
            faceConvectionVelocity: faceConvectionVelocity,
            faceDiffusionCoefficient: faceDiffusionCoefficient,
            cellSpacing: cellSpacing
        )
        eval(peclet)

        let result = peclet.asArray(Float.self)
        #expect(abs(result[0] - 10.0) < 1e-5)
        #expect(abs(result[1] - 5.0) < 1e-5)
        #expect(abs(result[2] - 2.0) < 1e-5)
        #expect(abs(result[3] - 1.0) < 1e-5)
    }

    @Test("Power-law weighting for different Péclet numbers")
    func weightingFactor() {
        let peclet = MLXArray([Float(0.0), Float(1.0), Float(5.0), Float(10.0), Float(50.0), Float(-10.0)])
        let alpha = PowerLawScheme.computeWeightingFactor(peclet: peclet)
        eval(alpha)

        let result = alpha.asArray(Float.self)

        // Pe = 0: central differencing → α ≈ 1
        #expect(abs(result[0] - 1.0) < 1e-5)

        // Pe = 1: α = (1 - 0.1)^5 = 0.59049
        #expect(abs(result[1] - 0.59049) < 1e-4)

        // Pe = 5: α = (1 - 0.5)^5 = 0.03125
        #expect(abs(result[2] - 0.03125) < 1e-4)

        // Pe = 10: α = (1 - 1.0)^5 = 0
        #expect(abs(result[3]) < 1e-5)

        // |Pe| > 10: full upwinding → α = 0
        #expect(abs(result[4]) < 1e-5)
        #expect(abs(result[5]) < 1e-5)
    }

    @Test("Weighting factor is bounded in [0, 1]")
    func weightingFactorBounds() {
        // Test over wide range of Péclet numbers
        let pecletRange = MLXArray.linspace(Float(-100.0), Float(100.0), count: 201)
        let alpha = PowerLawScheme.computeWeightingFactor(peclet: pecletRange)
        eval(alpha)

        let result = alpha.asArray(Float.self)

        for value in result {
            #expect(value >= 0.0)
            #expect(value <= 1.0)
        }
    }

    @Test("Upwind selection for positive Péclet")
    func upwindSelectionPositive() {
        let cellValues = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0)])

        // Positive Pe: upwind from left (|Pe| > 10 → α=0 → full upwinding)
        let pecletPos = MLXArray([Float(0.0), Float(15.0), Float(15.0), Float(15.0), Float(0.0)])
        let facePos = PowerLawScheme.interpolateToFaces(
            cellValues: cellValues,
            peclet: pecletPos
        )
        eval(facePos)

        let resultPos = facePos.asArray(Float.self)

        // Boundary faces use adjacent cell
        #expect(abs(resultPos[0] - 1.0) < 1e-5)
        #expect(abs(resultPos[4] - 4.0) < 1e-5)

        // Interior faces: full upwinding (α=0, Pe>10)
        // face = α * central + (1-α) * upwind = 0 * central + 1 * upwind
        // For positive Pe: upwind = left cell
        #expect(abs(resultPos[1] - 1.0) < 1e-5)  // upwind from cell[0]
        #expect(abs(resultPos[2] - 2.0) < 1e-5)  // upwind from cell[1]
        #expect(abs(resultPos[3] - 3.0) < 1e-5)  // upwind from cell[2]
    }

    @Test("Upwind selection for negative Péclet")
    func upwindSelectionNegative() {
        let cellValues = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0)])

        // Negative Pe: upwind from right (|Pe| > 10 → α=0 → full upwinding)
        let pecletNeg = MLXArray([Float(0.0), Float(-15.0), Float(-15.0), Float(-15.0), Float(0.0)])
        let faceNeg = PowerLawScheme.interpolateToFaces(
            cellValues: cellValues,
            peclet: pecletNeg
        )
        eval(faceNeg)

        let resultNeg = faceNeg.asArray(Float.self)

        // Boundary faces use adjacent cell
        #expect(abs(resultNeg[0] - 1.0) < 1e-5)
        #expect(abs(resultNeg[4] - 4.0) < 1e-5)

        // Interior faces: full upwinding (α=0, Pe<-10)
        // face = α * central + (1-α) * upwind = 0 * central + 1 * upwind
        // For negative Pe: upwind = right cell
        #expect(abs(resultNeg[1] - 2.0) < 1e-5)  // upwind from cell[1]
        #expect(abs(resultNeg[2] - 3.0) < 1e-5)  // upwind from cell[2]
        #expect(abs(resultNeg[3] - 4.0) < 1e-5)  // upwind from cell[3]
    }

    @Test("Central differencing for low Péclet")
    func centralDifferencingLowPeclet() {
        let cellValues = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0)])

        // Very small Pe: should behave like central differencing
        // α ≈ (1 - 0.1*0.01)^5 ≈ 0.995 (nearly 1)
        let pecletLow = MLXArray([Float(0.0), Float(0.01), Float(0.01), Float(0.01), Float(0.0)])
        let faceLow = PowerLawScheme.interpolateToFaces(
            cellValues: cellValues,
            peclet: pecletLow
        )
        eval(faceLow)

        let resultLow = faceLow.asArray(Float.self)

        // Interior faces: approximately central difference
        // face = α * central + (1-α) * upwind
        // α ≈ 0.995 → face ≈ central = 0.5*(left + right)
        #expect(abs(resultLow[1] - 1.5) < 0.01)  // ≈ 0.5*(1+2)
        #expect(abs(resultLow[2] - 2.5) < 0.01)  // ≈ 0.5*(2+3)
        #expect(abs(resultLow[3] - 3.5) < 0.01)  // ≈ 0.5*(3+4)
    }

    @Test("Smooth transition across Péclet range")
    func smoothTransition() {
        let cellValues = MLXArray([Float(1.0), Float(2.0)])

        // Test transition from central (Pe=0) to upwind (Pe=20)
        let pecletRange: [Float] = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

        var faceValues: [Float] = []
        for pe in pecletRange {
            let peclet = MLXArray([Float(0.0), Float(pe), Float(0.0)])
            let face = PowerLawScheme.interpolateToFaces(
                cellValues: cellValues,
                peclet: peclet
            )
            eval(face)
            faceValues.append(face[1].item(Float.self))
        }

        // Should monotonically decrease from ~1.5 (central) to ~1.0 (upwind left)
        // face = α * central + (1-α) * upwind
        // Pe=0: α=1 → face = central = 1.5
        // Pe>10: α=0 → face = upwind = 1.0 (left cell for positive Pe)
        for i in 0..<(faceValues.count - 1) {
            #expect(faceValues[i] >= faceValues[i+1] - 1e-4)  // Monotonic decrease
        }

        // Check endpoints
        #expect(abs(faceValues[0] - 1.5) < 0.01)  // Pe=0: central = 0.5*(1+2) = 1.5
        #expect(abs(faceValues[6] - 1.0) < 0.01)  // Pe=20: upwind left = 1.0
    }
}
