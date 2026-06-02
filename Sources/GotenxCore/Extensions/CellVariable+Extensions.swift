import MLX
import Foundation

// MARK: - CellVariable Extension

extension CellVariable {
    /// Convert to tuple for solver interface
    public func asTuple() -> (MLXArray, Float, Float?, Float?, Float?, Float?) {
        (value.value, radialSpacing, leftFaceConstraint, leftFaceGradientConstraint, rightFaceConstraint, rightFaceGradientConstraint)
    }
}
