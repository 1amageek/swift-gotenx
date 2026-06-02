import MLX
import Foundation

// MARK: - Linear Solver

/// Linear (predictor–corrector) solver for the implicit theta-method transport
/// equations.
///
/// Solves one timestep with an explicit predictor followed by theta-method
/// corrector sweeps:
/// 1. Predictor: `x* = xⁿ + timeStep · f(xⁿ)`
/// 2. Corrector: `x^{k+1} = xⁿ + timeStep · [θ·f(x^k) + (1−θ)·f(xⁿ)]` (optionally Pereverzev-damped)
///
/// where `f(x) = F(x) / transientCoefficient` is the per-cell rate of change. Unlike the
/// Newton-Raphson solver this keeps every operation differentiable (no iterative
/// inner solve, no line search), which is why the differentiable simulation path
/// uses it.
///
/// The spatial operator `F` is the **shared** finite-volume operator
/// (`applySpatialOperator1D`) — identical to the one the Newton solver uses — so the
/// linear solver now applies Dirichlet/Neumann boundary conditions and an
/// area-weighted (metric-Jacobian) flux divergence. The previous implementation used
/// a private operator that ignored boundary conditions and mis-weighted the
/// divergence, which made the scheme unconditionally unstable (temperatures ran
/// negative and the residual oscillated) even for constant transport.
public struct LinearSolver: PDESolver {
    // MARK: - Properties

    public let solverType: SolverType = .linear

    /// Number of corrector steps
    public let correctorStepCount: Int

    /// Use Pereverzev corrector (improves convergence)
    public let usesPereverzevCorrector: Bool

    /// Theta parameter for time discretization
    public let theta: Float

    // MARK: - Initialization

    public init(
        correctorStepCount: Int = 3,
        usesPereverzevCorrector: Bool = true,
        theta: Float = 1.0
    ) {
        precondition(correctorStepCount >= 1, "Must have at least 1 corrector step")
        precondition(theta >= 0.0 && theta <= 1.0, "Theta must be in [0, 1]")
        self.correctorStepCount = correctorStepCount
        self.usesPereverzevCorrector = usesPereverzevCorrector
        self.theta = theta
    }

    // MARK: - PDESolver Protocol

    public func solve(
        timeStep: Float,
        staticParameters: StaticRuntimeParameters,
        dynamicParamsT: DynamicRuntimeParameters,
        dynamicParamsTplusDt: DynamicRuntimeParameters,
        geometryT: Geometry,
        geometryTplusDt: Geometry,
        xOld: (CellVariable, CellVariable, CellVariable, CellVariable),
        coreProfilesT: CoreProfiles,
        coreProfilesTplusDt: CoreProfiles,
        coeffsCallback: @escaping CoeffsCallback
    ) -> SolverResult {
        do {
            try coreProfilesT.validateNumerics(expectedCellCount: staticParameters.mesh.cellCount)
            try coreProfilesTplusDt.validateNumerics(expectedCellCount: staticParameters.mesh.cellCount)
        } catch {
            return SolverResult(
                updatedProfiles: coreProfilesTplusDt,
                iterations: 0,
                residualNorm: Float.greatestFiniteMagnitude,
                converged: false,
                metadata: [
                    "theta": theta,
                    "dt": timeStep,
                    "failure_type": 4.0
                ]
            )
        }

        // Boundary conditions for the implicit step.
        let boundary = dynamicParamsTplusDt.boundaryConditions

        // Get coefficients at old time
        let coeffsOld = coeffsCallback(coreProfilesT, geometryT)
        do {
            try coeffsOld.validateNumerics()
        } catch {
            return SolverResult(
                updatedProfiles: coreProfilesTplusDt,
                iterations: 0,
                residualNorm: Float.greatestFiniteMagnitude,
                converged: false,
                metadata: [
                    "theta": theta,
                    "dt": timeStep,
                    "failure_type": 5.0
                ]
            )
        }

        // Predictor step: explicit Euler from x^n
        var xNew = predictorStep(
            xOld: coreProfilesT,
            coeffsOld: coeffsOld,
            timeStep: timeStep,
            staticParameters: staticParameters,
            boundary: boundary
        )

        // Corrector steps: theta-method fixed-point iteration
        var residualNorm: Float = 0.0
        var actualIterations = 0

        for _ in 0..<correctorStepCount {
            actualIterations += 1
            let xPrev = xNew

            // Coefficients at the current iterate (new time)
            let coeffsNew = coeffsCallback(xNew, geometryTplusDt)
            do {
                try coeffsNew.validateNumerics()
            } catch {
                return SolverResult(
                    updatedProfiles: xNew,
                    iterations: actualIterations,
                    residualNorm: Float.greatestFiniteMagnitude,
                    converged: false,
                    metadata: [
                        "theta": theta,
                        "dt": timeStep,
                        "failure_type": 5.0
                    ]
                )
            }

            xNew = correctorStep(
                xOld: coreProfilesT,
                xPrev: xPrev,
                coeffsOld: coeffsOld,
                coeffsNew: coeffsNew,
                timeStep: timeStep,
                theta: theta,
                usesPereverzev: usesPereverzevCorrector,
                staticParameters: staticParameters,
                boundary: boundary
            )

            residualNorm = computeResidualNorm(xNew: xNew, xPrev: xPrev)

            if residualNorm < staticParameters.solverTolerance {
                break
            }
        }

        do {
            try xNew.validateNumerics(expectedCellCount: staticParameters.mesh.cellCount)
        } catch {
            return SolverResult(
                updatedProfiles: xNew,
                iterations: actualIterations,
                residualNorm: Float.greatestFiniteMagnitude,
                converged: false,
                metadata: [
                    "theta": theta,
                    "dt": timeStep,
                    "failure_type": 6.0
                ]
            )
        }

        return SolverResult(
            updatedProfiles: xNew,
            iterations: actualIterations,
            residualNorm: residualNorm,
            converged: residualNorm < staticParameters.solverTolerance,
            metadata: [
                "theta": theta,
                "dt": timeStep,
                "corrector_steps": Float(correctorStepCount)
            ]
        )
    }

    // MARK: - Predictor Step

    /// Predictor step: explicit Euler `x* = xⁿ + timeStep · f(xⁿ)` (evolved variables only).
    private func predictorStep(
        xOld: CoreProfiles,
        coeffsOld: Block1DCoeffs,
        timeStep: Float,
        staticParameters: StaticRuntimeParameters,
        boundary: BoundaryConditions
    ) -> CoreProfiles {
        let fOld = spatialRates(profiles: xOld, coeffs: coeffsOld, boundary: boundary)

        let tiNew = staticParameters.evolveIonHeat
            ? xOld.ionTemperature.value + timeStep * fOld.0
            : xOld.ionTemperature.value
        let teNew = staticParameters.evolveElectronHeat
            ? xOld.electronTemperature.value + timeStep * fOld.1
            : xOld.electronTemperature.value
        let neNew = staticParameters.evolveElectronDensity
            ? xOld.electronDensity.value + timeStep * fOld.2
            : xOld.electronDensity.value
        let psiNew = staticParameters.evolvePoloidalFlux
            ? xOld.poloidalFlux.value + timeStep * fOld.3
            : xOld.poloidalFlux.value

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: tiNew),
            electronTemperature: EvaluatedArray(evaluating: teNew),
            electronDensity: EvaluatedArray(evaluating: neNew),
            poloidalFlux: EvaluatedArray(evaluating: psiNew)
        )
    }

    // MARK: - Corrector Step

    /// Corrector step: `x^{k+1} = xⁿ + timeStep · [θ·f(x^k) + (1−θ)·f(xⁿ)]` (evolved variables only).
    private func correctorStep(
        xOld: CoreProfiles,
        xPrev: CoreProfiles,
        coeffsOld: Block1DCoeffs,
        coeffsNew: Block1DCoeffs,
        timeStep: Float,
        theta: Float,
        usesPereverzev: Bool,
        staticParameters: StaticRuntimeParameters,
        boundary: BoundaryConditions
    ) -> CoreProfiles {
        let fOld = spatialRates(profiles: xOld, coeffs: coeffsOld, boundary: boundary)
        let fNew = spatialRates(profiles: xPrev, coeffs: coeffsNew, boundary: boundary)

        let dtTheta = timeStep * theta
        let dtOneMinusTheta = timeStep * (1.0 - theta)

        var tiNew = xOld.ionTemperature.value
        var teNew = xOld.electronTemperature.value
        var neNew = xOld.electronDensity.value
        var psiNew = xOld.poloidalFlux.value

        if staticParameters.evolveIonHeat {
            tiNew = xOld.ionTemperature.value + dtTheta * fNew.0 + dtOneMinusTheta * fOld.0
        }
        if staticParameters.evolveElectronHeat {
            teNew = xOld.electronTemperature.value + dtTheta * fNew.1 + dtOneMinusTheta * fOld.1
        }
        if staticParameters.evolveElectronDensity {
            neNew = xOld.electronDensity.value + dtTheta * fNew.2 + dtOneMinusTheta * fOld.2
        }
        if staticParameters.evolvePoloidalFlux {
            psiNew = xOld.poloidalFlux.value + dtTheta * fNew.3 + dtOneMinusTheta * fOld.3
        }

        // Pereverzev damping (blend with previous iterate) for evolved variables.
        if usesPereverzev {
            let alpha: Float = 0.5
            if staticParameters.evolveIonHeat {
                tiNew = alpha * tiNew + (1.0 - alpha) * xPrev.ionTemperature.value
            }
            if staticParameters.evolveElectronHeat {
                teNew = alpha * teNew + (1.0 - alpha) * xPrev.electronTemperature.value
            }
            if staticParameters.evolveElectronDensity {
                neNew = alpha * neNew + (1.0 - alpha) * xPrev.electronDensity.value
            }
            if staticParameters.evolvePoloidalFlux {
                psiNew = alpha * psiNew + (1.0 - alpha) * xPrev.poloidalFlux.value
            }
        }

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: tiNew),
            electronTemperature: EvaluatedArray(evaluating: teNew),
            electronDensity: EvaluatedArray(evaluating: neNew),
            poloidalFlux: EvaluatedArray(evaluating: psiNew)
        )
    }

    // MARK: - Spatial Rates

    /// Per-cell rates of change `∂x/∂t = F(x) / transientCoefficient` for all four channels,
    /// using the shared finite-volume operator (boundary conditions + area-weighted
    /// divergence) so the discretization matches the Newton-Raphson solver exactly.
    private func spatialRates(
        profiles: CoreProfiles,
        coeffs: Block1DCoeffs,
        boundary: BoundaryConditions
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let geometry = coeffs.geometry

        let fTi = rate(
            u: profiles.ionTemperature.value,
            eqCoeffs: coeffs.ionCoeffs,
            geometry: geometry,
            boundaryCondition: boundary.ionTemperature
        )
        let fTe = rate(
            u: profiles.electronTemperature.value,
            eqCoeffs: coeffs.electronCoeffs,
            geometry: geometry,
            boundaryCondition: boundary.electronTemperature
        )
        let fNe = rate(
            u: profiles.electronDensity.value,
            eqCoeffs: coeffs.densityCoeffs,
            geometry: geometry,
            boundaryCondition: boundary.electronDensity
        )
        let fPsi = rate(
            u: profiles.poloidalFlux.value,
            eqCoeffs: coeffs.fluxCoeffs,
            geometry: geometry,
            boundaryCondition: boundary.poloidalFlux
        )

        return (fTi, fTe, fNe, fPsi)
    }

    /// Rate of change for a single channel: `F(x) / transientCoefficient`.
    ///
    /// The transient coefficient (e.g. `n_e` for the temperature equations) is floored
    /// to a physical minimum density to avoid division by zero.
    private func rate(
        u: MLXArray,
        eqCoeffs: EquationCoeffs,
        geometry: GeometricFactors,
        boundaryCondition: BoundaryCondition
    ) -> MLXArray {
        let F = applySpatialOperator1D(
            u: u,
            coeffs: eqCoeffs,
            geometry: geometry,
            boundaryCondition: boundaryCondition
        )

        // Physical density floor [m⁻³] guards the non-conservation-form division.
        let safetyFloor: Float = 1e18
        let transientCoefficient = eqCoeffs.transientCoefficient.value
        return F / maximum(transientCoefficient, MLXArray(safetyFloor))
    }

    // MARK: - Convergence Check

    /// Residual norm between successive corrector iterates.
    private func computeResidualNorm(xNew: CoreProfiles, xPrev: CoreProfiles) -> Float {
        let diffTi = xNew.ionTemperature.value - xPrev.ionTemperature.value
        let diffTe = xNew.electronTemperature.value - xPrev.electronTemperature.value
        let diffNe = xNew.electronDensity.value - xPrev.electronDensity.value
        let diffPsi = xNew.poloidalFlux.value - xPrev.poloidalFlux.value

        let norm = sqrt(
            (diffTi * diffTi).mean() +
            (diffTe * diffTe).mean() +
            (diffNe * diffNe).mean() +
            (diffPsi * diffPsi).mean()
        )

        return norm.item(Float.self)
    }
}
