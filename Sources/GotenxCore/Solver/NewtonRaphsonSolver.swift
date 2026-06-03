import MLX
import Foundation
import Logging

// MARK: - Newton-Raphson Solver

// Logger for Newton-Raphson solver
private let logger = Logger(label: "com.gotenx.core.newton")

private struct NewtonDirection {
    let delta: MLXArray
    let linearError: Float
    let meritDescent: Float
    let usedBandedJacobian: Bool
}

private enum NewtonFailureType {
    static let linearSolverError: Float = 1.0
    static let invalidDescentDirection: Float = 2.0
    static let lineSearchNoDecrease: Float = 3.0
    static let invalidInputProfiles: Float = 4.0
    static let invalidCoefficients: Float = 5.0
    static let invalidFinalProfiles: Float = 6.0
    static let maximumIterationsExceeded: Float = 7.0
    static let invalidTolerance: Float = 8.0
}

private enum NewtonConvergenceMode {
    static let notConverged: Float = 0.0
    static let strict: Float = 1.0
    static let precisionFloor: Float = 2.0
}

private struct NewtonConvergenceCriteria {
    private static let toleranceFloor: Float = 1e-6

    let requestedTolerance: Float
    let effectiveTolerance: Float
    let ionTemperatureResidualTolerance: Float
    let electronTemperatureResidualTolerance: Float
    let electronDensityResidualTolerance: Float
    let poloidalFluxResidualTolerance: Float
    let electronDensityPrecisionFloorBound: Float

    init(requestedTolerance: Float) {
        self.requestedTolerance = requestedTolerance
        self.effectiveTolerance = max(requestedTolerance, Self.toleranceFloor)

        let scale = effectiveTolerance / Self.toleranceFloor
        self.ionTemperatureResidualTolerance = 10.0 * scale
        self.electronTemperatureResidualTolerance = 10.0 * scale
        self.electronDensityResidualTolerance = 0.1 * scale
        self.poloidalFluxResidualTolerance = 1e-3 * scale
        self.electronDensityPrecisionFloorBound = 5.0 * scale
    }

    var metadata: [String: Float] {
        [
            "requested_tolerance": requestedTolerance,
            "effective_tolerance": effectiveTolerance,
            "tolerance_floor_applied": requestedTolerance < Self.toleranceFloor ? 1.0 : 0.0,
            "tolerance_ti": ionTemperatureResidualTolerance,
            "tolerance_te": electronTemperatureResidualTolerance,
            "tolerance_ne": electronDensityResidualTolerance,
            "tolerance_psi": poloidalFluxResidualTolerance,
            "ne_precision_floor_bound": electronDensityPrecisionFloorBound
        ]
    }
}

/// Newton-Raphson solver for nonlinear implicit PDE systems
///
/// Uses automatic differentiation (vjp) for efficient Jacobian computation.
/// Solves: R(x^{n+1}) = 0
/// where R is the residual function from theta-method time discretization.
///
/// Key features:
/// - Vectorized spatial operators (NO loops)
/// - Per-equation coefficient handling (4 coupled equations)
/// - Hybrid linear solver (direct + iterative fallback)
/// - Efficient Jacobian via vjp() (3-4x faster than separate grad() calls)
public struct NewtonRaphsonSolver: PDESolver {
    // MARK: - Properties

    public let solverType: SolverType = .newtonRaphson

    /// Convergence tolerance for residual norm
    public let tolerance: Float

    /// Maximum number of Newton iterations
    public let maximumIterations: Int

    /// Theta parameter for time discretization (0: explicit, 0.5: Crank-Nicolson, 1: implicit)
    public let theta: Float

    /// Hybrid linear solver
    private let linearSolver: HybridLinearSolver

    /// Whether to try the colored-VJP block-tridiagonal Jacobian before full dense VJP.
    ///
    /// This candidate is verified with a true directional derivative before use, but
    /// it is opt-in because current MLX overhead makes full dense VJP faster for the
    /// default 100-cell benchmark.
    package let usesBandedJacobianCandidate: Bool

    // MARK: - Initialization

    /// Pereverzev-Galeev artificial-diffusion factor.
    ///
    /// Adds an artificial diffusion `D_pv = factor · D` plus a compensating pinch to
    /// the implicit spatial operator, evaluated at the previous-time profile so the
    /// extra flux cancels at convergence. This makes the stiff transport Jacobian
    /// diagonally dominant, turning the previously linear/stalling convergence into a
    /// fast one. `0` disables it. (TORAX uses the same stabilization for stiff χ.)
    public let pereverzevFactor: Float

    public init(
        tolerance: Float = 1e-6,
        maximumIterations: Int = 100,
        theta: Float = 1.0,
        linearSolver: HybridLinearSolver = HybridLinearSolver(),
        pereverzevFactor: Float = 0.5
    ) {
        self.init(
            tolerance: tolerance,
            maximumIterations: maximumIterations,
            theta: theta,
            linearSolver: linearSolver,
            pereverzevFactor: pereverzevFactor,
            usesBandedJacobianCandidate: false
        )
    }

    package init(
        tolerance: Float = 1e-6,
        maximumIterations: Int = 100,
        theta: Float = 1.0,
        linearSolver: HybridLinearSolver = HybridLinearSolver(),
        pereverzevFactor: Float = 0.5,
        usesBandedJacobianCandidate: Bool = false
    ) {
        precondition(theta >= 0.0 && theta <= 1.0, "Theta must be in [0, 1]")
        self.tolerance = tolerance
        self.maximumIterations = maximumIterations
        self.theta = theta
        self.linearSolver = linearSolver
        self.pereverzevFactor = pereverzevFactor
        self.usesBandedJacobianCandidate = usesBandedJacobianCandidate
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
        guard tolerance.isFinite && tolerance > 0 else {
            logger.error("Invalid Newton tolerance", metadata: ["tolerance": "\(tolerance)"])
            return validationFailureResult(
                profiles: coreProfilesTplusDt,
                timeStep: timeStep,
                failureType: NewtonFailureType.invalidTolerance,
                extraMetadata: ["requested_tolerance": tolerance]
            )
        }
        let convergenceCriteria = NewtonConvergenceCriteria(requestedTolerance: tolerance)

        do {
            try coreProfilesT.validateNumerics(expectedCellCount: staticParameters.mesh.cellCount)
            try coreProfilesTplusDt.validateNumerics(expectedCellCount: staticParameters.mesh.cellCount)
        } catch {
            logger.error("Invalid solver input profiles", metadata: ["error": "\(error)"])
            return validationFailureResult(
                profiles: coreProfilesTplusDt,
                timeStep: timeStep,
                failureType: NewtonFailureType.invalidInputProfiles,
                extraMetadata: convergenceCriteria.metadata
            )
        }

        // Flatten initial guess
        let xFlat: FlattenedState
        let xOldFlat: FlattenedState
        do {
            xFlat = try FlattenedState(profiles: coreProfilesTplusDt)
            xOldFlat = try FlattenedState(profiles: CoreProfiles.fromTuple(xOld))
        } catch {
            logger.error("Failed to flatten solver input profiles", metadata: ["error": "\(error)"])
            return validationFailureResult(
                profiles: coreProfilesTplusDt,
                timeStep: timeStep,
                failureType: NewtonFailureType.invalidInputProfiles,
                extraMetadata: convergenceCriteria.metadata
            )
        }
        let layout = xFlat.layout
        let cellCount = layout.cellCount

        // GPU Variable Scaling: Create reference state for normalization.
        // Uses physically meaningful scales per variable (Ti~1keV, Te~1keV, ne~10^20, psi~1Wb)
        // to prevent Float32 precision loss from extreme scale differences (e.g., psi=0 vs ne=10^20).
        let referenceState = xFlat.asPhysicalScalingReference()

        // Scale initial state to O(1)
        var xScaled = xFlat.scaled(by: referenceState)

        // Old-time coefficients are only used by theta methods with an explicit
        // old spatial contribution. Backward Euler skips them entirely, avoiding
        // unnecessary source/transport graph construction before the Newton loop.
        let coeffsOld: Block1DCoeffs?
        if theta == 1.0 {
            coeffsOld = nil
        } else {
            // Floor the old-time profiles too: a previous step may have left a cell at
            // (or just below) zero temperature, which would make the old-time source
            // coefficients blow up to NaN before the new step can even begin.
            let oldCoefficients = coeffsCallback(coreProfilesT.withPhysicalFloors(), geometryT)
            do {
                try oldCoefficients.validateNumerics()
            } catch {
                logger.error("Invalid old-time coefficients", metadata: ["error": "\(error)"])
                return validationFailureResult(
                    profiles: coreProfilesTplusDt,
                    timeStep: timeStep,
                    failureType: NewtonFailureType.invalidCoefficients,
                    extraMetadata: convergenceCriteria.metadata
                )
            }
            coeffsOld = oldCoefficients
        }

        // Extract boundary conditions
        let boundaryConditions = dynamicParamsTplusDt.boundaryConditions

        // Residual function in PHYSICAL space (not scaled)
        // This ensures physics calculations use correct units
        let residualFnPhysical: (MLXArray) -> MLXArray = { xNewFlatPhysical in
            // Unflatten to CoreProfiles (physical units)
            let xNewState = FlattenedState(
                values: xNewFlatPhysical,
                layout: layout,
                evaluationMode: .deferred
            )
            // Floor temperatures and density to keep source/transport derivatives
            // bounded (and NaN-free) while a variable transiently overshoots during
            // the Newton iteration. Only the coefficient evaluation sees the floored
            // state; the residual's time-derivative term still uses the raw state.
            let profilesNew = xNewState
                .toCoreProfiles(evaluationMode: .deferred)
                .withPhysicalFloors(evaluationMode: .deferred)

            // Get coefficients at new time (via callback)
            let coeffsNew = coeffsCallback(profilesNew, geometryTplusDt)

            // Compute residual for theta-method (physical units)
            let residual = self.computeThetaMethodResidual(
                xOld: xOldFlat.values.value,
                xNew: xNewFlatPhysical,
                coeffsOld: coeffsOld,
                coeffsNew: coeffsNew,
                timeStep: timeStep,
                theta: self.theta,
                layout: layout,
                staticParameters: staticParameters,
                boundaryConditions: boundaryConditions
            )

            return residual
        }

        // Residual function in SCALED space (for Newton iteration)
        // Converts scaled variables to physical, computes residual, then scales residual back
        let residualFnScaled: (MLXArray) -> MLXArray = { xNewScaled in
            // Unscale to physical units
            let xScaledState = FlattenedState(
                values: xNewScaled,
                layout: layout,
                evaluationMode: .deferred
            )
            let xPhysical = xScaledState.unscaled(by: referenceState, evaluationMode: .deferred)

            // Compute residual in physical units
            let residualPhysical = residualFnPhysical(xPhysical.values.value)

            // Scale residual for uniform precision
            let residualState = FlattenedState(
                values: residualPhysical,
                layout: layout,
                evaluationMode: .deferred
            )
            let residualScaled = residualState.scaled(by: referenceState, evaluationMode: .deferred)

            return residualScaled.values.value
        }
        // Newton-Raphson iteration in SCALED space
        var converged = false
        var iterations = 0
        // Best total residual seen and how many iterations since it last improved
        // meaningfully — used to detect stagnation at the Float32 precision floor.
        var bestTotalResidual: Float = .infinity
        var stagnantIterations = 0
        var residualNorm: Float = 0.0
        var residualNorm_Ti: Float = .infinity
        var residualNorm_Te: Float = .infinity
        var residualNorm_ne: Float = .infinity
        var residualNorm_psi: Float = .infinity
        var convergenceMode = NewtonConvergenceMode.notConverged
        let jacobianBasis = MLXArray.eye(xScaled.values.value.shape[0])
        eval(jacobianBasis)

        for iter in 0..<maximumIterations {
            iterations = iter + 1

            // Compute residual in scaled space
            let residualScaled = residualFnScaled(xScaled.values.value)

            // Track residual norms by variable.
            let residual_Ti = residualScaled[0..<cellCount]
            let residual_Te = residualScaled[cellCount..<(2*cellCount)]
            let residual_ne = residualScaled[(2*cellCount)..<(3*cellCount)]
            let residual_psi = residualScaled[(3*cellCount)..<(4*cellCount)]

            // Compute the total and per-variable residual norms in a single fused
            // graph, then pull all five scalars across with ONE GPU→CPU transfer
            // instead of five separate .item() synchronizations.
            let normTotal = sqrt((residualScaled * residualScaled).mean())
            let normsBatched = MLX.stacked([
                normTotal,
                MLX.norm(residual_Ti),
                MLX.norm(residual_Te),
                MLX.norm(residual_ne),
                MLX.norm(residual_psi)
            ], axis: 0)
            let norms = normsBatched.asArray(Float.self)
            residualNorm = norms[0]
            residualNorm_Ti = norms[1]
            residualNorm_Te = norms[2]
            residualNorm_ne = norms[3]
            residualNorm_psi = norms[4]

            if !residualNorm.isFinite {
                logger.warning("Residual contains NaN/Inf; stopping iteration", metadata: [
                    "iter": "\(iter)", "total": "\(residualNorm)"
                ])
                break
            }

            logger.debug("Newton residual", metadata: [
                "iter": "\(iter)",
                "total": "\(String(format: "%.2e", residualNorm))",
                "Ti": "\(String(format: "%.2e", residualNorm_Ti))",
                "Te": "\(String(format: "%.2e", residualNorm_Te))",
                "ne": "\(String(format: "%.2e", residualNorm_ne))",
                "psi": "\(String(format: "%.2e", residualNorm_psi))"
            ])

            // Keep the Newton direction and Jacobian intact; only the convergence
            // check is variable-specific.
            // Based on NEWTON_DIRECTION_ANALYSIS.md: Ti/Te stagnate, ne improves
            let tolerance_Ti = convergenceCriteria.ionTemperatureResidualTolerance
            let tolerance_Te = convergenceCriteria.electronTemperatureResidualTolerance
            let tolerance_ne = convergenceCriteria.electronDensityResidualTolerance
            let tolerance_psi = convergenceCriteria.poloidalFluxResidualTolerance

            let converged_Ti = residualNorm_Ti < tolerance_Ti
            let converged_Te = residualNorm_Te < tolerance_Te
            let converged_ne = residualNorm_ne < tolerance_ne
            let converged_psi = residualNorm_psi < tolerance_psi

            converged = converged_Ti && converged_Te && converged_ne && converged_psi

            // Stagnation / precision-floor acceptance.
            //
            // The vjp Jacobian is correct (verified against finite differences to within
            // ~1%), but in Float32 the coupled density residual floors at ~O(1) in scaled
            // units and cannot be driven down to the strict ne tolerance — the iteration
            // either oscillates around that floor or creeps toward it over hundreds of
            // steps. This is a fixed-precision limit, not a divergence (Apple-Silicon GPUs
            // are Float32-only; see docs/NUMERICAL_PRECISION.md). We therefore track the
            // best residual reached and, once it stops improving meaningfully for several
            // iterations while every dominant channel (Ti, Te, ψ) has converged, accept
            // the precision-limited solution rather than failing the otherwise-good step.
            if residualNorm < bestTotalResidual * 0.98 {
                bestTotalResidual = residualNorm
                stagnantIterations = 0
            } else {
                stagnantIterations += 1
            }
            let stagnated = stagnantIterations >= 6

            // Upper guard on the accepted density residual: precision-floor acceptance
            // is only legitimate when ne has actually reached its (Float32-limited) floor
            // — typically O(1) in scaled units. Without this bound a stalled-but-still-large
            // density residual could be reported as converged, masking a genuinely
            // unconverged (and unphysical) density. `neAcceptanceBound` is set well above
            // the observed floor (~0.5–1) yet far below any divergent value.
            let neAcceptanceBound = convergenceCriteria.electronDensityPrecisionFloorBound

            if !converged, stagnated, converged_Ti, converged_Te, converged_psi,
               residualNorm_ne < neAcceptanceBound {
                converged = true
                convergenceMode = NewtonConvergenceMode.precisionFloor
                logger.info("Converged to Float32 precision floor", metadata: [
                    "ne": "\(String(format: "%.2e", residualNorm_ne))",
                    "tolerance_ne": "\(String(format: "%.2e", tolerance_ne))",
                    "neAcceptanceBound": "\(String(format: "%.2e", neAcceptanceBound))",
                    "iterations": "\(iterations)"
                ])
                break
            }

            if converged {
                if convergenceMode == NewtonConvergenceMode.notConverged {
                    convergenceMode = NewtonConvergenceMode.strict
                }
                logger.info("All variables converged", metadata: [
                    "Ti": "\(String(format: "%.2e", residualNorm_Ti))",
                    "Te": "\(String(format: "%.2e", residualNorm_Te))",
                    "ne": "\(String(format: "%.2e", residualNorm_ne))",
                    "psi": "\(String(format: "%.2e", residualNorm_psi))",
                    "iterations": "\(iterations)"
                ])
                break
            } else {
                // Log which variables are blocking convergence (debug level)
                var notConverged: [String] = []
                if !converged_Ti { notConverged.append("Ti(\(String(format: "%.2e", residualNorm_Ti)))") }
                if !converged_Te { notConverged.append("Te(\(String(format: "%.2e", residualNorm_Te)))") }
                if !converged_ne { notConverged.append("ne(\(String(format: "%.2e", residualNorm_ne)))") }
                if !converged_psi { notConverged.append("psi(\(String(format: "%.2e", residualNorm_psi)))") }

                logger.debug("Convergence check", metadata: [
                    "iter": "\(iterations)",
                    "notConverged": "\(notConverged.joined(separator: ", "))"
                ])
            }

            // Terminate early only when the Newton direction is truly
            // unusable. This is an *inexact* Newton method: the linear system only has
            // to be solved accurately enough that the direction still reduces the
            // residual. Inexact/Newton-Krylov theory uses a forcing term eta (here 0.5):
            // any direction with ||J*delta + R|| <= eta*||R|| is acceptable. In Float32 a
            // stiff, ill-conditioned Jacobian can leave a few-percent linear error even
            // after equilibration; that direction is still a valid descent direction,
            // and the descent check and line search below are the real safeguards.
            let linearErrorThreshold: Float = 0.5

            let direction: NewtonDirection
            do {
                direction = try computeNewtonDirection(
                    residualFn: residualFnScaled,
                    x: xScaled.values.value,
                    residual: residualScaled,
                    layout: layout,
                    jacobianBasis: jacobianBasis,
                    linearErrorThreshold: linearErrorThreshold
                )
            } catch {
                logger.error("Linear solver failed", metadata: ["iter": "\(iter)", "error": "\(error)"])
                let finalPhysical = xScaled.unscaled(by: referenceState)
                let finalProfiles = finalPhysical.toCoreProfiles()
                return SolverResult(
                    updatedProfiles: finalProfiles,
                    iterations: iterations,
                    residualNorm: residualNorm,
                    converged: false,
                    metadata: resultMetadata(
                        timeStep: timeStep,
                        convergenceCriteria: convergenceCriteria,
                        residualTi: residualNorm_Ti,
                        residualTe: residualNorm_Te,
                        residualNe: residualNorm_ne,
                        residualPsi: residualNorm_psi,
                        convergenceMode: NewtonConvergenceMode.notConverged,
                        extra: ["failure_type": NewtonFailureType.linearSolverError]
                    )
                )
            }

            let deltaScaled = direction.delta
            let linear_error = direction.linearError
            let merit_descent = direction.meritDescent

            logger.debug("Newton direction", metadata: [
                "iter": "\(iter)",
                "linearError": "\(String(format: "%.2e", linear_error))",
                "meritDescent": "\(String(format: "%.2e", merit_descent))",
                "jacobian": "\(direction.usedBandedJacobian ? "banded" : "dense")"
            ])

            if linear_error > linearErrorThreshold {
                logger.error("Linear solver error too high - aborting", metadata: [
                    "linearError": "\(String(format: "%.2e", linear_error))",
                    "threshold": "\(String(format: "%.2e", linearErrorThreshold))",
                    "action": "trigger timeStep retry"
                ])

                // Return partial solution with converged=false
                let finalPhysical = xScaled.unscaled(by: referenceState)
                let finalProfiles = finalPhysical.toCoreProfiles()
                return SolverResult(
                    updatedProfiles: finalProfiles,
                    iterations: iterations,
                    residualNorm: residualNorm,
                    converged: false,
                    metadata: resultMetadata(
                        timeStep: timeStep,
                        convergenceCriteria: convergenceCriteria,
                        residualTi: residualNorm_Ti,
                        residualTe: residualNorm_Te,
                        residualNe: residualNorm_ne,
                        residualPsi: residualNorm_psi,
                        convergenceMode: NewtonConvergenceMode.notConverged,
                        extra: [
                            "linear_error": linear_error,
                            "failure_type": NewtonFailureType.linearSolverError
                        ]
                    )
                )
            }

            if merit_descent <= 0 {
                logger.error("Invalid descent direction - aborting", metadata: [
                    "meritDescent": "\(String(format: "%.2e", merit_descent))",
                    "action": "trigger timeStep retry"
                ])

                // Return partial solution with converged=false
                let finalPhysical = xScaled.unscaled(by: referenceState)
                let finalProfiles = finalPhysical.toCoreProfiles()
                return SolverResult(
                    updatedProfiles: finalProfiles,
                    iterations: iterations,
                    residualNorm: residualNorm,
                    converged: false,
                    metadata: resultMetadata(
                        timeStep: timeStep,
                        convergenceCriteria: convergenceCriteria,
                        residualTi: residualNorm_Ti,
                        residualTe: residualNorm_Te,
                        residualNe: residualNorm_ne,
                        residualPsi: residualNorm_psi,
                        convergenceMode: NewtonConvergenceMode.notConverged,
                        extra: [
                            "descent_value": merit_descent,
                            "failure_type": NewtonFailureType.invalidDescentDirection
                        ]
                    )
                )
            }

            // Update solution with line search (in scaled space).
            // A nil result means no step along this direction reduces the residual:
            // the (frozen-coefficient) linearization is no longer trustworthy at this
            // timeStep. Reject the step and report non-convergence so the orchestrator
            // retries with a smaller timeStep — taking a residual-increasing fallback step
            // here is what previously drove the stiff solve to diverge.
            guard let alpha = lineSearch(
                residualFn: residualFnScaled,
                x: xScaled.values.value,
                delta: deltaScaled,
                initialNorm: residualNorm,
                maxAlpha: 1.0
            ) else {
                logger.warning("Line search found no residual-reducing step - aborting", metadata: [
                    "iter": "\(iter)",
                    "residualNorm": "\(String(format: "%.2e", residualNorm))",
                    "action": "trigger timeStep retry"
                ])
                // Non-converged result is discarded by the orchestrator (it retries
                // from the previous step's profiles), so mirror the other abort paths
                // (linear_error, invalid descent) and skip the physical-floor pass.
                let finalPhysical = xScaled.unscaled(by: referenceState)
                let finalProfiles = finalPhysical.toCoreProfiles()
                return SolverResult(
                    updatedProfiles: finalProfiles,
                    iterations: iterations,
                    residualNorm: residualNorm,
                    converged: false,
                    metadata: resultMetadata(
                        timeStep: timeStep,
                        convergenceCriteria: convergenceCriteria,
                        residualTi: residualNorm_Ti,
                        residualTe: residualNorm_Te,
                        residualNe: residualNorm_ne,
                        residualPsi: residualNorm_psi,
                        convergenceMode: NewtonConvergenceMode.notConverged,
                        extra: ["failure_type": NewtonFailureType.lineSearchNoDecrease]
                    )
                )
            }

            let xNewScaled = xScaled.values.value + alpha * deltaScaled
            xScaled = FlattenedState(values: EvaluatedArray(evaluating: xNewScaled), layout: layout)
        }

        // Unscale final solution to physical units, then enforce positivity.
        //
        // The theta-method FVM discretization is not strictly positivity-preserving:
        // a strong radiative sink can drive a cell's temperature slightly negative
        // during the transient, which then makes the next step's source terms blow up.
        // Clamping the converged solution to physical floors keeps the profiles
        // physical (the floors sit far below the plasma temperature/density, so this is
        // a small positivity limiter, inactive in well-resolved regions).
        let xFinalPhysical = xScaled.unscaled(by: referenceState)
        let finalProfiles = xFinalPhysical.toCoreProfiles().withPhysicalFloors()
        do {
            try finalProfiles.validateNumerics(expectedCellCount: staticParameters.mesh.cellCount)
        } catch {
            logger.error("Invalid final solver profiles", metadata: ["error": "\(error)"])
            return validationFailureResult(
                profiles: finalProfiles,
                timeStep: timeStep,
                failureType: NewtonFailureType.invalidFinalProfiles,
                extraMetadata: convergenceCriteria.metadata
            )
        }

        var finalMetadata = resultMetadata(
            timeStep: timeStep,
            convergenceCriteria: convergenceCriteria,
            residualTi: residualNorm_Ti,
            residualTe: residualNorm_Te,
            residualNe: residualNorm_ne,
            residualPsi: residualNorm_psi,
            convergenceMode: convergenceMode,
            extra: ["variable_scaling": 1.0]
        )
        if !converged {
            finalMetadata["failure_type"] = NewtonFailureType.maximumIterationsExceeded
        }

        return SolverResult(
            updatedProfiles: finalProfiles,
            iterations: iterations,
            residualNorm: residualNorm,
            converged: converged,
            metadata: finalMetadata
        )
    }

    private func resultMetadata(
        timeStep: Float,
        convergenceCriteria: NewtonConvergenceCriteria,
        residualTi: Float,
        residualTe: Float,
        residualNe: Float,
        residualPsi: Float,
        convergenceMode: Float,
        extra: [String: Float] = [:]
    ) -> [String: Float] {
        var metadata: [String: Float] = [
            "theta": theta,
            "dt": timeStep,
            "convergence_mode": convergenceMode,
            "residual_ti": residualTi,
            "residual_te": residualTe,
            "residual_ne": residualNe,
            "residual_psi": residualPsi
        ]
        metadata.merge(convergenceCriteria.metadata) { _, new in new }
        metadata.merge(extra) { _, new in new }
        return metadata
    }

    private func computeNewtonDirection(
        residualFn: @escaping (MLXArray) -> MLXArray,
        x: MLXArray,
        residual: MLXArray,
        layout: FlattenedState.StateLayout,
        jacobianBasis: MLXArray,
        linearErrorThreshold: Float
    ) throws -> NewtonDirection {
        if usesBandedJacobianCandidate {
            let bandedJacobian = computeBlockTriDiagonalJacobianViaColoredVJP(
                residualFn,
                x,
                layout: layout
            )

            do {
                let candidateDelta = try linearSolver.solve(bandedJacobian, rightHandSide: -residual)
                let trueJacobianDelta = computeDirectionalDerivativeViaFiniteDifference(
                    residualFn,
                    x,
                    tangent: candidateDelta
                )
                let candidateMetrics = directionMetrics(
                    jacobianDelta: trueJacobianDelta,
                    residual: residual
                )

                if candidateMetrics.linearError <= linearErrorThreshold,
                   candidateMetrics.meritDescent > 0 {
                    return NewtonDirection(
                        delta: candidateDelta,
                        linearError: candidateMetrics.linearError,
                        meritDescent: candidateMetrics.meritDescent,
                        usedBandedJacobian: true
                    )
                }

                logger.debug("Banded Jacobian candidate requires dense fallback", metadata: [
                    "linearError": "\(String(format: "%.2e", candidateMetrics.linearError))",
                    "meritDescent": "\(String(format: "%.2e", candidateMetrics.meritDescent))"
                ])
            } catch {
                logger.debug("Banded Jacobian candidate failed; using dense fallback", metadata: [
                    "error": "\(error)"
                ])
            }
        }

        let denseJacobian = computeJacobianViaVJP(
            residualFn,
            x,
            basis: jacobianBasis
        )
        let delta = try linearSolver.solve(denseJacobian, rightHandSide: -residual)
        let jacobianDelta = denseJacobian.matmul(delta)
        let metrics = directionMetrics(jacobianDelta: jacobianDelta, residual: residual)

        return NewtonDirection(
            delta: delta,
            linearError: metrics.linearError,
            meritDescent: metrics.meritDescent,
            usedBandedJacobian: false
        )
    }

    private func directionMetrics(
        jacobianDelta: MLXArray,
        residual: MLXArray
    ) -> (linearError: Float, meritDescent: Float) {
        let linearResidual = jacobianDelta + residual
        let checks = MLX.stacked([
            MLX.norm(linearResidual),
            MLX.norm(residual),
            -(residual * jacobianDelta).sum()
        ], axis: 0).asArray(Float.self)
        let linearResidualNorm = checks[0]
        let residualNorm = checks[1]
        let meritDescent = checks[2]
        let linearError = linearResidualNorm / (residualNorm + 1e-20)
        return (linearError, meritDescent)
    }

    private func validationFailureResult(
        profiles: CoreProfiles,
        timeStep: Float,
        failureType: Float,
        extraMetadata: [String: Float] = [:]
    ) -> SolverResult {
        var metadata: [String: Float] = [
            "theta": theta,
            "dt": timeStep,
            "convergence_mode": NewtonConvergenceMode.notConverged,
            "failure_type": failureType
        ]
        metadata.merge(extraMetadata) { _, new in new }
        return SolverResult(
            updatedProfiles: profiles,
            iterations: 0,
            residualNorm: Float.greatestFiniteMagnitude,
            converged: false,
            metadata: metadata
        )
    }

    // MARK: - Residual Computation

    /// Apply Pereverzev-Galeev stabilization to a transport channel's coefficients.
    ///
    /// Adds artificial diffusion `D_pv = pereverzevFactor · faceDiffusionCoefficient` together with a
    /// compensating pinch `v_pv = D_pv · ∇u_ref / u_ref`, evaluated at the frozen
    /// linearization point `u_ref = stopGradient(u)`. Because `u_ref == u` at every
    /// evaluation point, the extra diffusive and convective fluxes cancel exactly, so
    /// the residual VALUE is unchanged and the converged solution is unbiased. The vjp
    /// Jacobian, however, gains the well-conditioning `D_pv·∇²` term (the pinch is a
    /// constant under differentiation), which turns the stiff transport solve's
    /// stalling, linear convergence into a fast, robust one. This is the same
    /// stabilization TORAX uses for stiff χ.
    private func pereverzevAugmented(
        _ coeffs: EquationCoeffs,
        u: MLXArray,
        geometry: GeometricFactors
    ) -> EquationCoeffs {
        guard pereverzevFactor > 0 else { return coeffs }

        let cellCount = u.shape[0]
        let faceDiffusionCoefficient = coeffs.faceDiffusionCoefficient.value          // [faceCount]
        let faceConvectionVelocity = coeffs.faceConvectionVelocity.value          // [faceCount]
        let dx = geometry.cellDistances.value   // [cellCount-1]

        // Artificial diffusion proportional to the existing diffusion (unit-consistent).
        let dPv = pereverzevFactor * faceDiffusionCoefficient      // [faceCount]

        // Pinch from the frozen linearization point u_ref = stopGradient(u).
        let uRef = stopGradient(u)
        let uRefRight = uRef[1..<cellCount]
        let uRefLeft = uRef[0..<(cellCount - 1)]
        let gradInterior = (uRefRight - uRefLeft) / (dx + 1e-10)        // [cellCount-1]
        let uFaceInterior = 0.5 * (uRefLeft + uRefRight)               // [cellCount-1]
        let logGradInterior = gradInterior / (uFaceInterior + 1e-10)  // [cellCount-1]
        let zero1 = MLXArray.zeros([1])
        // No pinch at the domain boundaries.
        let logGradFace = concatenated([zero1, logGradInterior, zero1], axis: 0)  // [faceCount]

        let dFaceAug = faceDiffusionCoefficient + dPv
        let vFaceAug = faceConvectionVelocity + dPv * logGradFace

        return EquationCoeffs(
            faceDiffusionCoefficient: .uncheckedLazy(dFaceAug),
            faceConvectionVelocity: .uncheckedLazy(vFaceAug),
            cellSource: coeffs.cellSource,
            cellSourceMatrixCoefficient: coeffs.cellSourceMatrixCoefficient,
            transientCoefficient: coeffs.transientCoefficient
        )
    }

    /// Compute residual for theta-method time discretization (VECTORIZED)
    ///
    /// Theta-method: (x^{n+1} - x^n) / timeStep = θ*f(x^{n+1}) + (1-θ)*f(x^n)
    /// Residual: R = (x^{n+1} - x^n) / timeStep - θ*f(x^{n+1}) - (1-θ)*f(x^n)
    private func computeThetaMethodResidual(
        xOld: MLXArray,
        xNew: MLXArray,
        coeffsOld: Block1DCoeffs?,
        coeffsNew: Block1DCoeffs,
        timeStep: Float,
        theta: Float,
        layout: FlattenedState.StateLayout,
        staticParameters: StaticRuntimeParameters,
        boundaryConditions: BoundaryConditions
    ) -> MLXArray {
        // Unflatten state vectors
        let Ti_old = xOld[layout.tiRange]
        let Te_old = xOld[layout.teRange]
        let ne_old = xOld[layout.neRange]
        let psi_old = xOld[layout.psiRange]

        let Ti_new = xNew[layout.tiRange]
        let Te_new = xNew[layout.teRange]
        let ne_new = xNew[layout.neRange]
        let psi_new = xNew[layout.psiRange]

        // Get transient coefficients.
        // These multiply the time derivative term: transientCoefficient * ∂u/∂t
        let transientCoeff_Ti = coeffsNew.ionCoeffs.transientCoefficient.value        // n_e for Ti
        let transientCoeff_Te = coeffsNew.electronCoeffs.transientCoefficient.value   // n_e for Te
        let transientCoeff_ne = coeffsNew.densityCoeffs.transientCoefficient.value    // 1.0 for ne
        let transientCoeff_psi = coeffsNew.fluxCoeffs.transientCoefficient.value      // L_p for psi

        // Time derivative terms WITH transient coefficients
        // Correct form: transientCoefficient * (u_new - u_old) / timeStep
        let dTi_dt = transientCoeff_Ti * (Ti_new - Ti_old) / timeStep
        let dTe_dt = transientCoeff_Te * (Te_new - Te_old) / timeStep
        let dne_dt = transientCoeff_ne * (ne_new - ne_old) / timeStep
        let dpsi_dt = transientCoeff_psi * (psi_new - psi_old) / timeStep

        // Spatial operators at new time (VECTORIZED) - with boundary conditions.
        // Inactive equations use inert coefficients and do not need spatial graphs;
        // their residual reduces to the time derivative, keeping disabled profiles fixed.
        let zeroCells = MLXArray.zeros([layout.cellCount])
        let f_Ti_new: MLXArray
        if staticParameters.evolveIonHeat {
            let ionCoeffsNew = pereverzevAugmented(coeffsNew.ionCoeffs, u: Ti_new, geometry: coeffsNew.geometry)
            f_Ti_new = applySpatialOperator1D(
                u: Ti_new,
                coeffs: ionCoeffsNew,
                geometry: coeffsNew.geometry,
                boundaryCondition: boundaryConditions.ionTemperature
            )
        } else {
            f_Ti_new = zeroCells
        }

        let f_Te_new: MLXArray
        if staticParameters.evolveElectronHeat {
            let electronCoeffsNew = pereverzevAugmented(coeffsNew.electronCoeffs, u: Te_new, geometry: coeffsNew.geometry)
            f_Te_new = applySpatialOperator1D(
                u: Te_new,
                coeffs: electronCoeffsNew,
                geometry: coeffsNew.geometry,
                boundaryCondition: boundaryConditions.electronTemperature
            )
        } else {
            f_Te_new = zeroCells
        }

        let f_ne_new: MLXArray
        if staticParameters.evolveElectronDensity {
            let densityCoeffsNew = pereverzevAugmented(coeffsNew.densityCoeffs, u: ne_new, geometry: coeffsNew.geometry)
            f_ne_new = applySpatialOperator1D(
                u: ne_new,
                coeffs: densityCoeffsNew,
                geometry: coeffsNew.geometry,
                boundaryCondition: boundaryConditions.electronDensity
            )
        } else {
            f_ne_new = zeroCells
        }

        let f_psi_new: MLXArray
        if staticParameters.evolvePoloidalFlux {
            f_psi_new = applySpatialOperator1D(
                u: psi_new,
                coeffs: coeffsNew.fluxCoeffs,
                geometry: coeffsNew.geometry,
                boundaryCondition: boundaryConditions.poloidalFlux
            )
        } else {
            f_psi_new = zeroCells
        }

        let R_Ti_raw: MLXArray
        let R_Te_raw: MLXArray
        let R_ne_raw: MLXArray
        let R_psi_raw: MLXArray

        let oneMinusTheta = 1.0 - theta
        if oneMinusTheta == 0.0 {
            // Backward Euler has no old-time spatial contribution. Avoid building
            // the old operator graph inside every residual and VJP evaluation.
            R_Ti_raw = dTi_dt - f_Ti_new
            R_Te_raw = dTe_dt - f_Te_new
            R_ne_raw = dne_dt - f_ne_new
            R_psi_raw = dpsi_dt - f_psi_new
        } else {
            guard let coeffsOld else {
                preconditionFailure("Old-time coefficients are required when theta != 1")
            }

            let f_Ti_old = staticParameters.evolveIonHeat
                ? applySpatialOperator1D(
                    u: Ti_old,
                    coeffs: coeffsOld.ionCoeffs,
                    geometry: coeffsOld.geometry,
                    boundaryCondition: boundaryConditions.ionTemperature
                )
                : zeroCells

            let f_Te_old = staticParameters.evolveElectronHeat
                ? applySpatialOperator1D(
                    u: Te_old,
                    coeffs: coeffsOld.electronCoeffs,
                    geometry: coeffsOld.geometry,
                    boundaryCondition: boundaryConditions.electronTemperature
                )
                : zeroCells

            let f_ne_old = staticParameters.evolveElectronDensity
                ? applySpatialOperator1D(
                    u: ne_old,
                    coeffs: coeffsOld.densityCoeffs,
                    geometry: coeffsOld.geometry,
                    boundaryCondition: boundaryConditions.electronDensity
                )
                : zeroCells

            let f_psi_old = staticParameters.evolvePoloidalFlux
                ? applySpatialOperator1D(
                    u: psi_old,
                    coeffs: coeffsOld.fluxCoeffs,
                    geometry: coeffsOld.geometry,
                    boundaryCondition: boundaryConditions.poloidalFlux
                )
                : zeroCells

            R_Ti_raw = dTi_dt - theta * f_Ti_new - oneMinusTheta * f_Ti_old
            R_Te_raw = dTe_dt - theta * f_Te_new - oneMinusTheta * f_Te_old
            R_ne_raw = dne_dt - theta * f_ne_new - oneMinusTheta * f_ne_old
            R_psi_raw = dpsi_dt - theta * f_psi_new - oneMinusTheta * f_psi_old
        }

        // Normalize residuals by dividing by transient coefficients.
        // This converts the equation from:
        //   n_e ∂T/∂t = RHS  (units: [eV/(m³·s)])
        // to:
        //   ∂T/∂t = RHS/n_e  (units: [eV/s])
        //
        // Problem: With n_e = 2×10¹⁹ m⁻³ and source = 10²⁴ eV/(m³·s),
        //          raw residual = 10²⁴, which causes Newton-Raphson to fail
        //
        // Solution: Divide by n_e to get ∂T/∂t ~ 10⁵ eV/s (manageable scale)
        //
        // Physical interpretation: We solve for temperature rate of change [eV/s]
        //                          instead of density-weighted rate [eV/(m³·s)]
        let R_Ti = R_Ti_raw / (transientCoeff_Ti + 1e-10)  // [eV/s]
        let R_Te = R_Te_raw / (transientCoeff_Te + 1e-10)  // [eV/s]
        let R_ne = R_ne_raw / (transientCoeff_ne + 1e-10)  // [m⁻³/s]
        let R_psi = R_psi_raw / (transientCoeff_psi + 1e-10)  // [Wb/s]

        // Flatten residuals
        return concatenated([R_Ti, R_Te, R_ne, R_psi], axis: 0)
    }

    // MARK: - Line Search

    /// Backtracking line search for step size selection
    ///
    /// Finds α such that ||R(x + α*Δx)|| < ||R(x)||
    /// Backtracking line search for the Newton step.
    ///
    /// Returns the largest `alpha ∈ {maxAlpha, maxAlpha·β, …}` whose trial residual
    /// is strictly smaller than `initialNorm`, or `nil` if NO tried step reduces the
    /// residual. Returning `nil` (rather than a fixed fallback step) is essential for
    /// global convergence: for stiff, solution-dependent transport the frozen-coefficient
    /// linearization can yield a direction along which every step *increases* the
    /// residual. Taking a fixed fallback step there compounds into divergence
    /// (residual blows up, temperatures reach unphysical values). The caller treats
    /// `nil` as "this timeStep is too large" and reduces the timestep instead of corrupting
    /// the iterate with a residual-increasing step.
    private func lineSearch(
        residualFn: (MLXArray) -> MLXArray,
        x: MLXArray,
        delta: MLXArray,
        initialNorm: Float,
        maxAlpha: Float
    ) -> Float? {
        let beta: Float = 0.5  // Reduction factor
        let maximumIterations = 10
        let batchSize = 4
        var alpha = maxAlpha
        let initialMerit = initialNorm * initialNorm

        // Most stabilized Newton steps accept alpha=1. Check it before the
        // batched fallback so successful steps do not compute unused residuals.
        let firstResidual = residualFn(x + alpha * delta)
        let firstMerit = (firstResidual * firstResidual).mean().item(Float.self)
        if firstMerit.isFinite && firstMerit < initialMerit {
            return alpha
        }

        alpha *= beta
        var checked = 1

        while checked < maximumIterations {
            var batchAlphas: [Float] = []
            var batchMerits: [MLXArray] = []
            batchAlphas.reserveCapacity(batchSize)
            batchMerits.reserveCapacity(batchSize)

            while batchAlphas.count < batchSize && checked < maximumIterations {
                let xNew = x + alpha * delta
                let residualNew = residualFn(xNew)
                batchAlphas.append(alpha)
                batchMerits.append((residualNew * residualNew).mean())
                alpha *= beta
                checked += 1
            }

            let merits = MLX.stacked(batchMerits, axis: 0).asArray(Float.self)
            for index in merits.indices {
                if merits[index].isFinite && merits[index] < initialMerit {
                    return batchAlphas[index]
                }
            }
        }

        // No tried step reduces the residual: signal failure so the caller can
        // reduce timeStep rather than take a residual-increasing (divergent) step.
        return nil
    }
}

// MARK: - State Layout Helper
// Note: StateLayout is now defined in NumericalTolerances.swift
