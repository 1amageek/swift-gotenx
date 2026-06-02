// PowerLawScheme.swift
// Patankar power-law scheme for convection-diffusion face weighting

import Foundation
import MLX

/// Patankar power-law scheme for convection-diffusion face weighting
///
/// **Physics**: High Péclet number (Pe = V·Δx/D >> 1) causes numerical oscillations
/// with central differencing. Power-law scheme provides smooth transition:
///
/// - Pe < 0.1: Central differencing (2nd order accurate)
/// - 0.1 ≤ Pe ≤ 10: Power-law interpolation
/// - Pe > 10: First-order upwinding (stable but diffusive)
///
/// **References**:
/// - Patankar, S.V. (1980). "Numerical Heat Transfer and Fluid Flow"
/// - TORAX: arXiv:2406.06718v2, Section 2.2.3
public struct PowerLawScheme {

    /// Compute Péclet number at faces
    ///
    /// Pe = V·Δx / D
    ///
    /// - Parameters:
    ///   - faceConvectionVelocity: Convection velocity at faces [m/s], shape [faceCount]
    ///   - faceDiffusionCoefficient: Diffusion coefficient at faces [m²/s], shape [faceCount]
    ///   - cellSpacing: Cell spacing [m], shape [faceCount-1] or scalar
    /// - Returns: Péclet number [dimensionless], shape [faceCount]
    public static func computePecletNumber(
        faceConvectionVelocity: MLXArray,
        faceDiffusionCoefficient: MLXArray,
        cellSpacing: MLXArray
    ) -> MLXArray {
        // Prevent division by zero
        let safeFaceDiffusionCoefficient = faceDiffusionCoefficient + 1e-30

        // Broadcast cellSpacing to [faceCount] if needed
        let broadcastCellSpacing: MLXArray
        if cellSpacing.ndim == 0 {
            // Scalar: create full array
            broadcastCellSpacing = MLXArray.full([faceConvectionVelocity.shape[0]], values: cellSpacing)
        } else if cellSpacing.shape[0] == faceConvectionVelocity.shape[0] - 2 {
            // cellSpacing is [faceCount-1] (interior only): pad boundaries
            let leftSpacing = cellSpacing[0..<1]
            let rightSpacing = cellSpacing[(cellSpacing.shape[0]-1)..<cellSpacing.shape[0]]
            broadcastCellSpacing = concatenated([leftSpacing, cellSpacing, rightSpacing], axis: 0)
        } else {
            // Already correct size
            broadcastCellSpacing = cellSpacing
        }

        return faceConvectionVelocity * broadcastCellSpacing / safeFaceDiffusionCoefficient
    }

    /// Compute power-law weighting factor α for face interpolation
    ///
    /// **Correct formula**:
    /// Face value: x_face = α·x_central + (1-α)·x_upwind
    ///
    /// **Patankar formula**:
    /// ```
    /// α(Pe) = max(0, (1 - 0.1·|Pe|)^5)  for |Pe| ≤ 10
    /// α(Pe) = 0 (full upwinding)        for |Pe| > 10
    /// ```
    ///
    /// **Behavior**:
    /// - Pe = 0: α = 1 → central differencing (2nd order accurate)
    /// - Pe = 10: α = 0 → full upwinding (1st order, stable)
    ///
    /// - Parameter peclet: Péclet number [dimensionless], shape [faceCount]
    /// - Returns: Weighting factor α ∈ [0,1], shape [faceCount]
    public static func computeWeightingFactor(peclet: MLXArray) -> MLXArray {
        let absPe = abs(peclet)

        // Power-law formula: (1 - 0.1*|Pe|)^5
        let clamped = maximum(0.0, 1.0 - 0.1 * absPe)
        let powerLaw = pow(clamped, 5.0)

        // For |Pe| > 10: full upwinding (α = 0)
        // Use where() for conditional selection
        // Note: MLX uses .> for element-wise comparison
        let alpha = `where`(absPe .> 10.0, MLXArray(0.0), powerLaw)

        return alpha
    }

    /// Compute face values using power-law weighting
    ///
    /// - Parameters:
    ///   - cellValues: Values at cell centers [cellCount]
    ///   - peclet: Péclet number at faces [faceCount]
    /// - Returns: Weighted face values [faceCount]
    public static func interpolateToFaces(
        cellValues: MLXArray,
        peclet: MLXArray
    ) -> MLXArray {
        let cellCount = cellValues.shape[0]

        // Interior faces: power-law weighted
        let leftCells = cellValues[0..<(cellCount-1)]
        let rightCells = cellValues[1..<cellCount]
        let pecletInterior = peclet[1..<(peclet.shape[0]-1)]

        let alpha = computeWeightingFactor(peclet: pecletInterior)

        // Central differencing (second-order accurate)
        let centralDiff = 0.5 * (leftCells + rightCells)

        // Upwind selection based on flow direction
        // Use where() for conditional selection
        // Note: MLX uses .> for element-wise comparison
        let upwindValues = `where`(
            pecletInterior .> 0,  // condition: Pe > 0
            leftCells,           // if true: upwind = left cell
            rightCells          // if false: upwind = right cell
        )

        // Power-law weighted average: blend central diff and upwind
        // α = 1 (Pe=0): pure central differencing
        // α = 0 (Pe>10): pure upwinding
        let faceInterior = alpha * centralDiff + (1.0 - alpha) * upwindValues

        // Boundary faces: use adjacent cell value
        let faceLeft = cellValues[0..<1]
        let faceRight = cellValues[(cellCount-1)..<cellCount]

        return concatenated([faceLeft, faceInterior, faceRight], axis: 0)
    }
}
