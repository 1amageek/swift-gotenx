import Testing
import MLX
@testable import GotenxCore

struct CellVariableTests {

    // MARK: - Initialization Tests

    @Test("CellVariable initialization with valid parameters")
    func testInitialization() throws {
        let values = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)])
        let radialSpacing: Float = 0.1

        let cellVar = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceConstraint: Float(0.5),
            rightFaceConstraint: Float(5.5)
        )

        #expect(cellVar.cellCount == 5)
        #expect(cellVar.faceCount == 6)
        #expect(cellVar.radialSpacing == 0.1)
    }

    @Test("CellVariable requires 1D array")
    func testRequires1DArray() {
        // CellVariable requires 1D arrays
        // 2D arrays will trigger precondition failure
        // In Swift Testing, we can't easily test precondition failures
        // This is more of a documentation of the requirement
    }

    @Test("CellVariable requires positive radialSpacing")
    func testRequiresPositiveDr() {
        // CellVariable requires radialSpacing > 0
        // radialSpacing <= 0 will trigger precondition failure
        // Documenting the requirement
    }

    @Test("CellVariable requires exactly one left boundary condition")
    func testRequiresOneLeftBC() {
        let values = MLXArray([Float(1.0), Float(2.0), Float(3.0)])
        let radialSpacing: Float = 0.1

        // Valid: value constraint only
        let _ = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceConstraint: Float(0.5),
            rightFaceConstraint: Float(3.5)
        )

        // Valid: gradient constraint only
        let _ = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceGradientConstraint: Float(1.0),
            rightFaceConstraint: Float(3.5)
        )

        // Invalid cases would trigger precondition failures
    }

    // MARK: - Face Value Tests

    @Test("Face values with value constraints on both boundaries")
    func testFaceValueWithValueConstraints() {
        // Create a simple linear profile: [1, 2, 3, 4, 5]
        let values = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)])
        let radialSpacing: Float = 0.1

        let cellVar = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceConstraint: Float(0.5),  // Left boundary
            rightFaceConstraint: Float(5.5)   // Right boundary
        )

        let faceVals = cellVar.faceValues()
        eval(faceVals)

        // Expected: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        // Left: 0.5 (constraint)
        // Inner: averages of neighbors
        // Right: 5.5 (constraint)

        #expect(faceVals.shape == [6])

        let expected = MLXArray([Float(0.5), Float(1.5), Float(2.5), Float(3.5), Float(4.5), Float(5.5)])
        let diff = abs(faceVals - expected)
        let maxDiff = diff.max().item(Float.self)
        #expect(maxDiff < 1e-5)
    }

    @Test("Face values with gradient constraint on right boundary")
    func testFaceValueWithGradConstraint() {
        let values = MLXArray([Float(1.0), Float(2.0), Float(3.0)])
        let radialSpacing: Float = 0.2

        let cellVar = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceConstraint: Float(0.5),
            rightFaceGradientConstraint: Float(10.0)  // Gradient constraint
        )

        let faceVals = cellVar.faceValues()
        eval(faceVals)

        #expect(faceVals.shape == [4])

        // Left: 0.5 (constraint)
        // Inner: [1.5, 2.5] (averages)
        // Right: 3.0 + 10.0 * 0.2/2 = 3.0 + 1.0 = 4.0

        let faceArray = faceVals.asArray(Float.self)
        #expect(abs(faceArray[0] - 0.5) < 1e-5)
        #expect(abs(faceArray[1] - 1.5) < 1e-5)
        #expect(abs(faceArray[2] - 2.5) < 1e-5)
        #expect(abs(faceArray[3] - 4.0) < 1e-5)
    }

    @Test("Face values for uniform profile")
    func testFaceValueUniform() {
        // Uniform profile should have uniform face values
        let values = MLXArray([Float(2.0), Float(2.0), Float(2.0), Float(2.0)])
        let radialSpacing: Float = 0.1

        let cellVar = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceConstraint: Float(2.0),
            rightFaceConstraint: Float(2.0)
        )

        let faceVals = cellVar.faceValues()
        eval(faceVals)

        let expected = MLXArray([Float(2.0), Float(2.0), Float(2.0), Float(2.0), Float(2.0)])
        let diff = abs(faceVals - expected)
        let maxDiff = diff.max().item(Float.self)
        #expect(maxDiff < 1e-5)
    }

    // MARK: - Face Gradient Tests

    @Test("Face gradients with gradient constraints")
    func testFaceGradWithGradConstraints() {
        // Linear profile: gradient should be constant
        let values = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)])
        let radialSpacing: Float = 0.1

        let cellVar = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceGradientConstraint: Float(10.0),   // 1.0 / 0.1 = 10
            rightFaceGradientConstraint: Float(10.0)
        )

        let faceGrads = cellVar.faceGradients()
        eval(faceGrads)

        #expect(faceGrads.shape == [6])

        // All gradients should be 10.0 for this linear profile
        let expected = MLXArray([Float(10.0), Float(10.0), Float(10.0), Float(10.0), Float(10.0), Float(10.0)])
        let diff = abs(faceGrads - expected)
        let maxDiff = diff.max().item(Float.self)
        #expect(maxDiff < 1e-4)
    }

    @Test("Face gradients with value constraints")
    func testFaceGradWithValueConstraints() {
        let values = MLXArray([Float(2.0), Float(3.0), Float(4.0)])
        let radialSpacing: Float = 0.2

        let cellVar = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceConstraint: Float(1.5),
            rightFaceConstraint: Float(4.5)
        )

        let faceGrads = cellVar.faceGradients()
        eval(faceGrads)

        #expect(faceGrads.shape == [4])

        // Left: (2.0 - 1.5) / (0.2/2) = 0.5 / 0.1 = 5.0
        // Inner: [5.0, 5.0] (forward differences)
        // Right: (4.5 - 4.0) / (0.2/2) = 0.5 / 0.1 = 5.0

        let gradsArray = faceGrads.asArray(Float.self)
        for grad in gradsArray {
            #expect(abs(grad - 5.0) < 1e-4)
        }
    }

    @Test("Face gradients for constant profile")
    func testFaceGradConstantProfile() {
        // Constant profile should have zero gradients
        let values = MLXArray([Float(3.0), Float(3.0), Float(3.0), Float(3.0)])
        let radialSpacing: Float = 0.1

        let cellVar = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceGradientConstraint: Float(0.0),
            rightFaceGradientConstraint: Float(0.0)
        )

        let faceGrads = cellVar.faceGradients()
        eval(faceGrads)

        let maxGrad = abs(faceGrads).max().item(Float.self)
        #expect(maxGrad < 1e-5)
    }

    // MARK: - Cell Gradient Tests

    @Test("Cell gradients from face values")
    func testCellGrad() {
        let values = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)])
        let radialSpacing: Float = 0.1

        let cellVar = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceConstraint: Float(0.5),
            rightFaceConstraint: Float(5.5)
        )

        let cellGrads = cellVar.gradients()
        eval(cellGrads)

        #expect(cellGrads.shape == [5])

        // Face values: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        // Cell gradients: diff / radialSpacing = [1.0, 1.0, 1.0, 1.0, 1.0] / 0.1 = [10, 10, 10, 10, 10]

        let expected = MLXArray([Float(10.0), Float(10.0), Float(10.0), Float(10.0), Float(10.0)])
        let diff = abs(cellGrads - expected)
        let maxDiff = diff.max().item(Float.self)
        #expect(maxDiff < 1e-4)
    }

    // MARK: - Integration Tests

    @Test("Round-trip: construct profile from boundary conditions")
    func testBoundaryConditionIntegration() {
        // Test that boundary conditions are properly enforced
        let values = MLXArray([Float(1.0), Float(2.0), Float(3.0)])
        let radialSpacing: Float = 0.1
        let leftBC: Float = 0.0
        let rightBC: Float = 4.0

        let cellVar = CellVariable(
            value: values,
            radialSpacing: radialSpacing,
            leftFaceConstraint: leftBC,
            rightFaceConstraint: rightBC
        )

        let faceVals = cellVar.faceValues()
        eval(faceVals)

        let faceArray = faceVals.asArray(Float.self)

        // Check boundary values
        #expect(abs(faceArray[0] - leftBC) < 1e-5)
        #expect(abs(faceArray[3] - rightBC) < 1e-5)
    }

    @Test("Physical plasma profile example")
    func testPhysicalPlasmaProfile() {
        // Simulate a typical temperature profile: peaked at center, lower at edge
        // T(rho) ~ T0 * (1 - rho^2)
        let radialSpacing: Float = 0.1
        let rho = MLXArray(stride(from: 0.05, to: 1.0, by: 0.1).map { Float($0) })
        let T0: Float = 10.0  // keV

        let temperatures = T0 * (1.0 - rho * rho)

        let cellVar = CellVariable(
            value: temperatures,
            radialSpacing: radialSpacing,
            leftFaceGradientConstraint: Float(0.0),     // Zero gradient at center (symmetry)
            rightFaceConstraint: Float(0.1)         // Edge temperature (boundary)
        )

        let faceVals = cellVar.faceValues()
        let faceGrads = cellVar.faceGradients()
        eval(faceVals, faceGrads)

        #expect(faceVals.shape == [11])
        #expect(faceGrads.shape == [11])

        // Gradients should be negative (temperature decreases with radius)
        let innerGrads = faceGrads[1..<10]
        let meanGrad = innerGrads.mean().item(Float.self)
        #expect(meanGrad < 0.0)
    }
}
