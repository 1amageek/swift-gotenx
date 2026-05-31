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
/// the transient coefficient — callers that need `∂u/∂t = F / transientCoeff`
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
    let nCells = u.shape[0]

    // 1. Gradient at interior faces: ∇u = (u[i+1] - u[i]) / dx
    let u_right = u[1..<nCells]
    let u_left = u[0..<(nCells - 1)]
    let dx = geometry.cellDistances.value  // [nCells-1]

    let gradFace_interior = (u_right - u_left) / (dx + 1e-10)

    // Boundary-face gradients from the boundary conditions.
    let gradFace_left: MLXArray
    switch boundaryCondition.left {
    case .value(let val):
        // Dirichlet: gradient from the prescribed edge value.
        let u_boundary = MLXArray(val)
        let dx_left = dx[0..<1]
        gradFace_left = (u[0..<1] - u_boundary) / (dx_left + 1e-10)
    case .gradient(let grad):
        // Neumann: use the prescribed gradient directly.
        gradFace_left = MLXArray([grad])
    }

    let gradFace_right: MLXArray
    switch boundaryCondition.right {
    case .value(let val):
        let u_boundary = MLXArray(val)
        let dx_right = dx[(nCells - 2)..<(nCells - 1)]
        gradFace_right = (u_boundary - u[(nCells - 1)..<nCells]) / (dx_right + 1e-10)
    case .gradient(let grad):
        gradFace_right = MLXArray([grad])
    }

    let gradFace = concatenated([gradFace_left, gradFace_interior, gradFace_right], axis: 0)

    // 2. Diffusive flux: F_diff = -D·∇u
    let dFace = coeffs.dFace.value
    let diffusiveFlux = -dFace * gradFace

    // 3. Convective flux: F_conv = v·u_face (power-law interpolation for stability)
    let vFace = coeffs.vFace.value
    let u_face = interpolateToFacesPowerLaw(u, vFace: vFace, dFace: dFace, dx: dx)
    let convectiveFlux = vFace * u_face

    // 4. Total flux at faces
    let totalFlux = diffusiveFlux + convectiveFlux

    // 5. Metric-Jacobian weighted flux divergence: ∇·F = (1/√g)·∂(√g·F)/∂ψ
    let jacobianCells = geometry.jacobian.value
    let jacobianFaces_interior = 0.5 * (jacobianCells[0..<(nCells - 1)] + jacobianCells[1..<nCells])
    let jacobianFaces = concatenated([
        jacobianCells[0..<1],
        jacobianFaces_interior,
        jacobianCells[(nCells - 1)..<nCells]
    ], axis: 0)

    let weightedFlux = jacobianFaces * totalFlux

    let flux_right = weightedFlux[1..<(nCells + 1)]
    let flux_left = weightedFlux[0..<nCells]
    let cellDistances = geometry.cellDistances.value

    // Per-cell characteristic length (map [nCells-1] face distances to [nCells]).
    let dx_padded = concatenated([
        cellDistances,
        cellDistances[(cellDistances.shape[0] - 1)..<cellDistances.shape[0]]
    ], axis: 0)

    let fluxDivergence = (flux_right - flux_left) / ((jacobianCells * dx_padded) + 1e-10)

    // 6. Source terms
    let source = coeffs.sourceCell.value
    let sourceMatrix = coeffs.sourceMatCell.value

    // 7. Total spatial operator
    return fluxDivergence + source + sourceMatrix * u
}

/// Interpolate cell values to faces using the Patankar power-law scheme.
///
/// Provides convection-diffusion stability across the Péclet range:
/// central differencing at low Pe, first-order upwinding at high Pe.
func interpolateToFacesPowerLaw(
    _ u: MLXArray,
    vFace: MLXArray,
    dFace: MLXArray,
    dx: MLXArray
) -> MLXArray {
    let peclet = PowerLawScheme.computePecletNumber(vFace: vFace, dFace: dFace, dx: dx)
    return PowerLawScheme.interpolateToFaces(cellValues: u, peclet: peclet)
}
