import MLX
import Foundation

// MARK: - Shared 1D Finite-Volume Spatial Operator

/// Single source of truth for the 1D finite-volume transport operator used by
/// every PDE solver (Newton-Raphson and the linear solver).
///
/// Computes `F(u) = ∇·(−D∇u + v·u) + S + Sᴹ·u` for one transport channel, with:
/// - **Dirichlet/Neumann boundary conditions** applied at the domain edges
///   (a Dirichlet edge value enters through the boundary-face gradient), and
/// - a **metric-Jacobian (√g) weighted flux divergence**, so fluxes are area-weighted
///   consistently with the cell volumes (`∇·F = (1/√g)·∂(√g·F)/∂ψ`).
///
/// The result is the spatial rate term `F` in physical units; it is NOT divided by
/// the transient coefficient — callers that need `∂u/∂t = F / transientCoefficient`
/// (e.g. an explicit update) perform that normalization themselves, while the
/// theta-method residual keeps `F` as-is.
///
/// Keeping this in one place ensures the linear solver and Newton-Raphson solver
/// discretize the equations identically (previously the linear solver used a
/// separate operator that ignored boundary conditions and mis-weighted the
/// divergence, causing unconditional divergence).
func applySpatialOperator1D(
    u: MLXArray,
    coeffs: EquationCoeffs,
    geometry: GeometricFactors,
    boundaryCondition: BoundaryCondition
) -> MLXArray {
    let cellCount = u.shape[0]

    // 1. Gradient at interior faces: ∇u = (u[i+1] - u[i]) / cellSpacing
    let u_right = u[1..<cellCount]
    let u_left = u[0..<(cellCount - 1)]
    let cellSpacing = geometry.cellDistances.value  // [cellCount-1]

    let gradFace_interior = (u_right - u_left) / (cellSpacing + 1e-10)

    // Boundary-face gradients from the boundary conditions.
    let gradFace_left: MLXArray
    switch boundaryCondition.left {
    case .value(let val):
        // Dirichlet: gradient from the prescribed edge value.
        let u_boundary = MLXArray(val)
        let dx_left = cellSpacing[0..<1]
        gradFace_left = (u[0..<1] - u_boundary) / (dx_left + 1e-10)
    case .gradient(let grad):
        // Neumann: use the prescribed gradient directly.
        gradFace_left = MLXArray([grad])
    }

    let gradFace_right: MLXArray
    switch boundaryCondition.right {
    case .value(let val):
        let u_boundary = MLXArray(val)
        let dx_right = cellSpacing[(cellCount - 2)..<(cellCount - 1)]
        gradFace_right = (u_boundary - u[(cellCount - 1)..<cellCount]) / (dx_right + 1e-10)
    case .gradient(let grad):
        gradFace_right = MLXArray([grad])
    }

    let gradFace = concatenated([gradFace_left, gradFace_interior, gradFace_right], axis: 0)

    // 2. Diffusive flux: F_diff = -D·∇u
    let faceDiffusionCoefficient = coeffs.faceDiffusionCoefficient.value
    let diffusiveFlux = -faceDiffusionCoefficient * gradFace

    // 3. Convective flux: F_conv = v·faceValues (power-law interpolation for stability)
    let faceConvectionVelocity = coeffs.faceConvectionVelocity.value
    let faceValues = interpolateToFacesPowerLaw(u, faceConvectionVelocity: faceConvectionVelocity, faceDiffusionCoefficient: faceDiffusionCoefficient, cellSpacing: cellSpacing)
    let convectiveFlux = faceConvectionVelocity * faceValues

    // 4. Total flux at faces
    let totalFlux = diffusiveFlux + convectiveFlux

    // 5. Metric-Jacobian weighted flux divergence: ∇·F = (1/√g)·∂(√g·F)/∂ψ
    let jacobianCells = geometry.jacobian.value
    let jacobianFaces_interior = 0.5 * (jacobianCells[0..<(cellCount - 1)] + jacobianCells[1..<cellCount])
    let jacobianFaces = concatenated([
        jacobianCells[0..<1],
        jacobianFaces_interior,
        jacobianCells[(cellCount - 1)..<cellCount]
    ], axis: 0)

    let weightedFlux = jacobianFaces * totalFlux

    let flux_right = weightedFlux[1..<(cellCount + 1)]
    let flux_left = weightedFlux[0..<cellCount]
    let cellDistances = geometry.cellDistances.value

    // Per-cell characteristic length (map [cellCount-1] face distances to [cellCount]).
    let dx_padded = concatenated([
        cellDistances,
        cellDistances[(cellDistances.shape[0] - 1)..<cellDistances.shape[0]]
    ], axis: 0)

    let fluxDivergence = (flux_right - flux_left) / ((jacobianCells * dx_padded) + 1e-10)

    // 6. Source terms
    let source = coeffs.cellSource.value
    let sourceMatrix = coeffs.cellSourceMatrixCoefficient.value

    // 7. Total spatial operator
    return fluxDivergence + source + sourceMatrix * u
}

/// Interpolate cell values to faces using the Patankar power-law scheme.
///
/// Provides convection-diffusion stability across the Péclet range:
/// central differencing at low Pe, first-order upwinding at high Pe.
func interpolateToFacesPowerLaw(
    _ u: MLXArray,
    faceConvectionVelocity: MLXArray,
    faceDiffusionCoefficient: MLXArray,
    cellSpacing: MLXArray
) -> MLXArray {
    let peclet = PowerLawScheme.computePecletNumber(faceConvectionVelocity: faceConvectionVelocity, faceDiffusionCoefficient: faceDiffusionCoefficient, cellSpacing: cellSpacing)
    return PowerLawScheme.interpolateToFaces(cellValues: u, peclet: peclet)
}
