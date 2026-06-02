import MLX
import Foundation

// MARK: - MLX Utilities

/// Compute forward difference along axis
private func diff(_ array: MLXArray, axis: Int = 0) -> MLXArray {
    let shape = array.shape
    guard axis < shape.count else {
        fatalError("Axis \(axis) out of bounds for array with ndim=\(shape.count)")
    }

    // For 1D array, compute array[1:] - array[:-1]
    if axis == 0 {
        let left = array[0..<(shape[0] - 1)]
        let right = array[1..<shape[0]]
        return right - left
    }

    fatalError("diff() currently only supports axis=0")
}

// MARK: - Cell Variable

/// Grid variable with boundary conditions for 1D finite volume method
///
/// `CellVariable` represents values discretized on a 1D uniform grid.
/// It stores values at cell centers and handles boundary conditions
/// at the leftmost and rightmost faces.
///
/// Note: This type now uses EvaluatedArray to ensure type safety.
/// It is pure Sendable since all fields are Sendable.
public struct CellVariable: Sendable {
    // MARK: - Properties

    /// Values at cell centers (shape: [cellCount])
    public let value: EvaluatedArray

    /// Distance between cell centers
    public let radialSpacing: Float

    /// Optional value constraint for the leftmost face
    public let leftFaceConstraint: Float?

    /// Optional gradient constraint for the leftmost face
    public let leftFaceGradientConstraint: Float?

    /// Optional value constraint for the rightmost face
    public let rightFaceConstraint: Float?

    /// Optional gradient constraint for the rightmost face
    public let rightFaceGradientConstraint: Float?

    // MARK: - Initialization

    /// Create a cell variable with optional boundary conditions
    ///
    /// - Parameters:
    ///   - value: Values at cell centers (will be evaluated)
    ///   - radialSpacing: Distance between cell centers
    ///   - leftFaceConstraint: Optional value constraint for left boundary
    ///   - leftFaceGradientConstraint: Optional gradient constraint for left boundary
    ///   - rightFaceConstraint: Optional value constraint for right boundary
    ///   - rightFaceGradientConstraint: Optional gradient constraint for right boundary
    ///
    /// - Note: Exactly one of (leftFaceConstraint, leftFaceGradientConstraint) must be non-nil,
    ///         and exactly one of (rightFaceConstraint, rightFaceGradientConstraint) must be non-nil
    public init(
        value: MLXArray,
        radialSpacing: Float,
        leftFaceConstraint: Float? = nil,
        leftFaceGradientConstraint: Float? = nil,
        rightFaceConstraint: Float? = nil,
        rightFaceGradientConstraint: Float? = nil
    ) {
        precondition(value.ndim == 1, "CellVariable value must be 1D array")
        precondition(radialSpacing > 0, "radialSpacing must be positive")

        // Validate left boundary condition
        let hasLeftValue = leftFaceConstraint != nil
        let hasLeftGrad = leftFaceGradientConstraint != nil
        precondition(
            hasLeftValue != hasLeftGrad,
            "Exactly one of leftFaceConstraint or leftFaceGradientConstraint must be set"
        )

        // Validate right boundary condition
        let hasRightValue = rightFaceConstraint != nil
        let hasRightGrad = rightFaceGradientConstraint != nil
        precondition(
            hasRightValue != hasRightGrad,
            "Exactly one of rightFaceConstraint or rightFaceGradientConstraint must be set"
        )

        self.value = EvaluatedArray(evaluating: value)
        self.radialSpacing = radialSpacing
        self.leftFaceConstraint = leftFaceConstraint
        self.leftFaceGradientConstraint = leftFaceGradientConstraint
        self.rightFaceConstraint = rightFaceConstraint
        self.rightFaceGradientConstraint = rightFaceGradientConstraint
    }

    // MARK: - Computed Properties

    /// Number of cells
    public var cellCount: Int {
        value.shape[0]
    }

    /// Number of faces (cellCount + 1)
    public var faceCount: Int {
        cellCount + 1
    }

    // MARK: - Face Value Calculation

    /// Calculate values at faces
    ///
    /// Inner faces are calculated as the average of neighboring cell values.
    /// Boundary faces use the specified constraints.
    ///
    /// - Returns: Array of face values (shape: [faceCount])
    public func faceValues() -> MLXArray {
        // Extract underlying MLXArray for computation
        let cellValues = value.value

        // Left face value (reshape to [1] for concatenation)
        let leftValue: MLXArray
        if let constraint = leftFaceConstraint {
            leftValue = MLXArray([constraint])
        } else if let gradConstraint = leftFaceGradientConstraint {
            // Linear extrapolation: x_face = x_cell0 - (radialSpacing/2) * gradient
            let firstCell = cellValues[0..<1]
            leftValue = firstCell - MLXArray(gradConstraint * radialSpacing / 2.0)
        } else {
            fatalError("Left boundary condition not properly set")
        }

        // Inner face values (average of neighbors)
        let leftCells = cellValues[0..<(cellCount - 1)]
        let rightCells = cellValues[1..<cellCount]
        let innerValues = (leftCells + rightCells) / 2.0

        // Right face value (reshape to [1] for concatenation)
        let rightValue: MLXArray
        if let constraint = rightFaceConstraint {
            rightValue = MLXArray([constraint])
        } else if let gradConstraint = rightFaceGradientConstraint {
            // Calculate from gradient constraint: value[end] + grad * radialSpacing/2
            let lastCell = cellValues[(cellCount - 1)..<cellCount]
            rightValue = lastCell + MLXArray(gradConstraint * radialSpacing / 2.0)
        } else {
            fatalError("Right boundary condition not properly set")
        }

        // Concatenate: [left, inner..., right]
        return concatenated([leftValue, innerValues, rightValue], axis: 0)
    }

    // MARK: - Face Gradient Calculation

    /// Calculate gradients at faces
    ///
    /// Gradients are computed using forward differences between cells,
    /// with boundary gradients determined by the specified constraints.
    ///
    /// - Parameter x: Optional coordinate array for non-uniform grids
    /// - Returns: Array of face gradients (shape: [faceCount])
    public func faceGradients(x: MLXArray? = nil) -> MLXArray {
        // Extract underlying MLXArray for computation
        let cellValues = value.value

        // Forward difference for inner faces
        let difference = diff(cellValues, axis: 0)
        let dx = x != nil ? diff(x!, axis: 0) : MLXArray(radialSpacing)
        let forwardDiff = difference / dx

        // Left gradient (reshape to [1] for concatenation)
        let leftGrad: MLXArray
        if let gradConstraint = leftFaceGradientConstraint {
            leftGrad = MLXArray([gradConstraint])
        } else if let valueConstraint = leftFaceConstraint {
            // Calculate from value constraint: (value[0] - constraint) / (radialSpacing/2)
            let firstCell = cellValues[0..<1]
            leftGrad = (firstCell - MLXArray(valueConstraint)) / MLXArray(radialSpacing / 2.0)
        } else {
            fatalError("Left boundary condition not properly set")
        }

        // Right gradient (reshape to [1] for concatenation)
        let rightGrad: MLXArray
        if let gradConstraint = rightFaceGradientConstraint {
            rightGrad = MLXArray([gradConstraint])
        } else if let valueConstraint = rightFaceConstraint {
            // Calculate from value constraint: (constraint - value[end]) / (radialSpacing/2)
            let lastCell = cellValues[(cellCount - 1)..<cellCount]
            rightGrad = (MLXArray(valueConstraint) - lastCell) / MLXArray(radialSpacing / 2.0)
        } else {
            fatalError("Right boundary condition not properly set")
        }

        // Concatenate: [left, forward_diff..., right]
        return concatenated([leftGrad, forwardDiff, rightGrad], axis: 0)
    }

    // MARK: - Cell Gradient

    /// Calculate gradients at cell centers
    ///
    /// This is computed as the difference of face values divided by radialSpacing.
    ///
    /// - Returns: Array of cell gradients (shape: [cellCount])
    public func gradients() -> MLXArray {
        let faceVals = faceValues()
        let difference = diff(faceVals, axis: 0)
        return difference / MLXArray(radialSpacing)
    }
}

// MARK: - Equatable Conformance

extension CellVariable: Equatable {
    public static func == (lhs: CellVariable, rhs: CellVariable) -> Bool {
        // Compare all properties
        guard lhs.radialSpacing == rhs.radialSpacing,
              lhs.leftFaceConstraint == rhs.leftFaceConstraint,
              lhs.leftFaceGradientConstraint == rhs.leftFaceGradientConstraint,
              lhs.rightFaceConstraint == rhs.rightFaceConstraint,
              lhs.rightFaceGradientConstraint == rhs.rightFaceGradientConstraint else {
            return false
        }

        // Compare evaluated arrays
        return lhs.value == rhs.value
    }
}
