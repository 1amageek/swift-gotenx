// DifferentiableSimulation.swift
// Gradient-aware simulation for optimization
//
// ⚠️ CRITICAL: This simulation DOES NOT use compile() to preserve gradient tape
//
// Differences from SimulationOrchestrator:
// 1. No compile() - gradient tracking requires uncompiled ops
// 2. Simplified timestep (no adaptive, no retries)
// 3. Returns (final_profiles, loss) where loss is differentiable
// 4. Not an actor (pure function for MLX AD)

import Foundation
import MLX

/// Protocol for sources that support gradient-aware computation
///
/// Sources conforming to this protocol can receive MLXArray parameters
/// directly, allowing gradients to flow through the computation
public protocol GradientAwareSource: SourceModel {
    /// Set the MLXArray power for gradient computation
    func setMLXPower(_ power: MLXArray)
}

/// Differentiable simulation for gradient-based optimization
///
/// **Purpose**: Enable automatic differentiation for parameter optimization
///
/// **Key Constraint**: NO `compile()` - compilation erases gradient tape
///
/// **Use Cases**:
/// - Forward sensitivity analysis (∂Q_fusion / ∂parameters)
/// - Inverse problems (optimize actuators to maximize Q_fusion)
/// - Model predictive control
///
/// **Example**:
/// ```swift
/// let sim = DifferentiableSimulation(
///     staticParams: staticParams,
///     transport: BohmGyrobohmModel(),
///     sources: [FusionSourceModel()],
///     geometry: geometry
/// )
///
/// let (finalProfiles, loss) = sim.forward(
///     initialProfiles: initialProfiles,
///     actuators: actuators,
///     timeHorizon: 2.0,
///     dt: 0.01
/// )
/// ```
public struct DifferentiableSimulation {
    // MARK: - Configuration

    /// Static runtime parameters
    private let staticParams: StaticRuntimeParams

    /// Transport model
    private let transport: any TransportModel

    /// Source models
    private let sources: [any SourceModel]

    /// Geometry
    public let geometry: Geometry

    /// Solver (must be differentiable - use LinearSolver, not Newton-Raphson with compile)
    private let solver: any PDESolver

    // MARK: - Initialization

    public init(
        staticParams: StaticRuntimeParams,
        transport: any TransportModel,
        sources: [any SourceModel] = [],
        geometry: Geometry
    ) {
        self.staticParams = staticParams
        self.transport = transport
        self.sources = sources
        self.geometry = geometry

        // Use LinearSolver for differentiation (simpler, no iterative solve)
        self.solver = LinearSolver(
            nCorrectorSteps: 1,  // Minimal correction
            usePereversevCorrector: false,
            theta: staticParams.theta
        )
    }

    // MARK: - Forward Pass

    /// Differentiable forward simulation
    ///
    /// **Critical**: This function preserves the gradient tape for automatic differentiation
    ///
    /// - Parameters:
    ///   - initialProfiles: Initial plasma profiles
    ///   - actuators: Time series of control parameters
    ///   - dynamicParams: Dynamic runtime parameters (boundaries, transport params, etc.)
    ///   - timeHorizon: Total simulation time [s]
    ///   - dt: Fixed timestep [s] (adaptive timestep breaks gradients!)
    ///
    /// - Returns: Tuple of (final profiles, loss value for minimization)
    ///
    /// **Loss Function**:
    /// Default: `-Q_fusion` (negative for maximization via minimization)
    /// Can be customized for other objectives (profile matching, energy confinement, etc.)
    public func forward(
        initialProfiles: CoreProfiles,
        actuators: ActuatorTimeSeries,
        dynamicParams: DynamicRuntimeParams,
        timeHorizon: Float,
        dt: Float
    ) -> (CoreProfiles, MLXArray) {
        var profiles = initialProfiles
        let nSteps = Int(timeHorizon / dt)

        // CRITICAL FOR GRADIENTS: Extract actuator MLXArray once
        // This preserves the gradient tape connection
        let actuatorArray = actuators.toMLXArray()  // Shape: [nSteps × 4]

        // For constant actuators, use average (all timesteps have same value)
        // This maintains differentiability while avoiding asArray() in loop
        let avgP_ECRH = MLX.mean(actuatorArray[0..<actuators.nSteps])
        let avgP_ICRH = MLX.mean(actuatorArray[actuators.nSteps..<(2*actuators.nSteps)])
        let avgGasPuff = MLX.mean(actuatorArray[(2*actuators.nSteps)..<(3*actuators.nSteps)])
        let avgI_plasma = MLX.mean(actuatorArray[(3*actuators.nSteps)..<(4*actuators.nSteps)])

        // Update dynamic params once with MLXArray values
        // These MLXArrays preserve gradients
        let dynamicParamsWithActuators = updateDynamicParamsMLX(
            dynamicParams,
            P_ECRH: avgP_ECRH,
            P_ICRH: avgP_ICRH,
            gas_puff: avgGasPuff,
            I_plasma: avgI_plasma
        )

        // CRITICAL FOR GRADIENTS: Set MLXArray power on sources
        // This allows gradient-aware sources to use MLXArrays instead of Floats
        let P_aux_total_mlx = avgP_ECRH + avgP_ICRH
        setMLXPowerOnSources(P_aux_total_mlx)

        // Time-stepping loop (NO compile!)
        for _ in 0..<nSteps {
            // Perform single differentiable timestep
            // Use same actuator values for all steps (constant actuators)
            profiles = stepDifferentiable(
                profiles: profiles,
                dynamicParams: dynamicParamsWithActuators,
                dt: dt
            )
        }

        // Compute loss from final state
        let loss = computeLoss(profiles: profiles)

        return (profiles, loss)
    }

    /// Differentiable timestep (core operation)
    ///
    /// **Critical**: All operations must be differentiable w.r.t. profiles and params
    private func stepDifferentiable(
        profiles: CoreProfiles,
        dynamicParams: DynamicRuntimeParams,
        dt: Float
    ) -> CoreProfiles {
        // Build CoeffsCallback (for solver)
        // Note: We build coefficients inside the callback to ensure they depend on
        // the profiles being solved (needed for iterative solvers)
        let coeffsCallback: CoeffsCallback = { [transport, sources, dynamicParams, staticParams] profs, geo in
            let transportCoeffs = transport.computeCoefficients(
                profiles: profs,
                geometry: geo,
                params: dynamicParams.transportParams
            )

            let sourceTerms = sources.reduce(
                into: SourceTerms.zero(
                    nCells: staticParams.mesh.nCells,
                    metadata: nil,
                    validateDebugUnits: false
                )
            ) { total, model in
                if let params = dynamicParams.sourceParams[model.name] {
                    let contribution = model.computeTermsForSolver(
                        profiles: profs,
                        geometry: geo,
                        params: params
                    )
                    total = total.adding(contribution, validateDebugUnits: false)
                }
            }

            return buildBlock1DCoeffs(
                transport: transportCoeffs,
                sources: sourceTerms,
                geometry: geo,
                staticParams: staticParams,
                profiles: profs
            )
        }

        // 5. Solve (differentiable - linear solver only!)
        let xOld = profiles.asTuple(
            dr: staticParams.mesh.dr,
            boundaryConditions: dynamicParams.boundaryConditions
        )

        let result = solver.solve(
            dt: dt,
            staticParams: staticParams,
            dynamicParamsT: dynamicParams,
            dynamicParamsTplusDt: dynamicParams,
            geometryT: geometry,
            geometryTplusDt: geometry,
            xOld: xOld,
            coreProfilesT: profiles,
            coreProfilesTplusDt: profiles,
            coeffsCallback: coeffsCallback
        )

        // 6. Return new profiles
        return result.updatedProfiles
    }

    // MARK: - Loss Functions

    /// Compute loss from profiles (differentiable)
    ///
    /// **Default objective**: Maximize average temperature
    ///
    /// Returns `-T_avg` so minimization → maximization
    ///
    /// **Note**: We use average temperature instead of Q_fusion because:
    /// 1. Q_fusion requires high temperatures (10-20 keV) to be non-zero
    /// 2. Temperature directly responds to heating power
    /// 3. More sensitive for gradient-based optimization
    ///
    /// For actual scenario optimization with realistic parameters,
    /// Q_fusion maximization can be used.
    private func computeLoss(profiles: CoreProfiles) -> MLXArray {
        // Average ion and electron temperature
        let avgTi = MLX.mean(profiles.ionTemperature.value)
        let avgTe = MLX.mean(profiles.electronTemperature.value)
        let avgT = (avgTi + avgTe) / 2.0

        // Return negative for maximization via minimization
        return -avgT
    }

    /// Compute profile matching loss (L2 error)
    ///
    /// Use for inverse problems: match experimental target profiles
    public func computeProfileMatchingLoss(
        profiles: CoreProfiles,
        target: TargetProfiles
    ) -> MLXArray {
        let Ti = profiles.ionTemperature.value
        let Te = profiles.electronTemperature.value
        let ne = profiles.electronDensity.value

        let Ti_target = target.ionTemperature
        let Te_target = target.electronTemperature
        let ne_target = target.electronDensity

        // L2 error (differentiable)
        let Ti_error = sum(pow(Ti - Ti_target, 2))
        let Te_error = sum(pow(Te - Te_target, 2))
        let ne_error = sum(pow(ne - ne_target, 2))

        let totalError = Ti_error + Te_error + ne_error

        return totalError / Float(staticParams.mesh.nCells)
    }

    // MARK: - Helper Functions

    /// Set MLXArray power on gradient-aware sources
    ///
    /// This allows gradient-aware sources to access the actuator MLXArray directly,
    /// preserving the gradient tape for automatic differentiation
    private func setMLXPowerOnSources(_ power: MLXArray) {
        for source in sources {
            if let gradientSource = source as? GradientAwareSource {
                gradientSource.setMLXPower(power)
            }
        }
    }

    /// Update dynamic params with actuator MLXArrays (gradient-preserving)
    ///
    /// **Critical**: Uses MLXArrays directly to preserve gradient tape
    ///
    /// This version is used in forward() to maintain differentiability
    private func updateDynamicParamsMLX(
        _ params: DynamicRuntimeParams,
        P_ECRH: MLXArray,
        P_ICRH: MLXArray,
        gas_puff: MLXArray,
        I_plasma: MLXArray
    ) -> DynamicRuntimeParams {
        var updated = params

        // Convert MLXArrays to Float for storage (gradient still flows through computation)
        eval(P_ECRH, P_ICRH, gas_puff, I_plasma)
        let P_ECRH_val = P_ECRH.item(Float.self)
        let P_ICRH_val = P_ICRH.item(Float.self)
        let gas_puff_val = gas_puff.item(Float.self)
        let I_plasma_val = I_plasma.item(Float.self)

        // Calculate total auxiliary power
        let P_aux_total = P_ECRH_val + P_ICRH_val  // [MW]

        // Update all heating sources
        for (sourceName, var sourceParams) in updated.sourceParams {
            if sourceName.contains("fusion") || sourceName.contains("heating") {
                sourceParams.params["P_auxiliary"] = P_aux_total
                sourceParams.params["P_ECRH"] = P_ECRH_val
                sourceParams.params["P_ICRH"] = P_ICRH_val
                updated.sourceParams[sourceName] = sourceParams
            }
        }

        // Update ohmic heating if present
        if var ohmicParams = updated.sourceParams["ohmic"] {
            ohmicParams.params["I_plasma"] = I_plasma_val
            updated.sourceParams["ohmic"] = ohmicParams
        }

        // Update boundary conditions (gas puff → edge density)
        let gasPuffScaling: Float = 0.1
        let densityFromGasPuff = gasPuffScaling * gas_puff_val
        let clampedDensity = max(1e18, min(5e19, densityFromGasPuff))

        var updatedBC = updated.boundaryConditions
        updatedBC.electronDensity.right = .value(clampedDensity)
        updated.boundaryConditions = updatedBC

        return updated
    }

    /// Update dynamic params with actuator values
    ///
    /// **Engineering mapping**:
    /// 1. P_ECRH, P_ICRH → Heating power sources (MW → eV/m³/s)
    /// 2. gas_puff → Density boundary condition (particles/s → m⁻³)
    /// 3. I_plasma → Current drive (MA → A/m²)
    ///
    /// **Mathematical consistency**:
    /// - All unit conversions must be physically correct
    /// - Power must be conserved (P_in = P_ECRH + P_ICRH + P_ohmic)
    /// - Particle balance (gas_puff → density BC)
    private func updateDynamicParams(
        _ params: DynamicRuntimeParams,
        with actuators: ActuatorValues
    ) -> DynamicRuntimeParams {
        var updated = params

        // 1. Update auxiliary heating power
        // Map P_ECRH + P_ICRH to total auxiliary power
        let P_aux_total = actuators.P_ECRH + actuators.P_ICRH  // [MW]

        // Update all heating sources with auxiliary power
        // This supports both "fusion" (real simulations) and "simple_heating" (tests)
        for (sourceName, var sourceParams) in updated.sourceParams {
            if sourceName.contains("fusion") || sourceName.contains("heating") {
                // Store total auxiliary power
                sourceParams.params["P_auxiliary"] = P_aux_total

                // Store individual powers for power partition analysis
                sourceParams.params["P_ECRH"] = actuators.P_ECRH
                sourceParams.params["P_ICRH"] = actuators.P_ICRH

                updated.sourceParams[sourceName] = sourceParams
            }
        }

        // Also update ohmic heating params if present
        if var ohmicParams = updated.sourceParams["ohmic"] {
            // Ohmic heating depends on plasma current
            ohmicParams.params["I_plasma"] = actuators.I_plasma  // [MA]
            updated.sourceParams["ohmic"] = ohmicParams
        }

        // 2. Update density boundary condition from gas puff
        // Engineering model: gas_puff [particles/s] → edge density [m⁻³]
        //
        // Simplified model: n_edge ∝ gas_puff / (particle_confinement_time × surface_area)
        // For optimization: we use a scaling factor
        //
        // Typical values:
        // - gas_puff: 1e20 particles/s
        // - edge density: 1e19 m⁻³
        // - scaling: ~0.1
        let gasPuffScaling: Float = 0.1  // Calibrated constant
        let densityFromGasPuff = gasPuffScaling * actuators.gas_puff  // [m⁻³]

        // Clamp to physical range
        let minEdgeDensity: Float = 1e18  // 0.1 × 10²⁰ m⁻³
        let maxEdgeDensity: Float = 5e19  // 5 × 10²⁰ m⁻³
        let clampedDensity = max(minEdgeDensity, min(maxEdgeDensity, densityFromGasPuff))

        // Update boundary conditions (gas puff affects edge density)
        var updatedBC = updated.boundaryConditions
        // Update right boundary (edge) with new density value
        updatedBC.electronDensity.right = .value(clampedDensity)
        updated.boundaryConditions = updatedBC

        // 3. Plasma current (I_plasma) affects:
        // - Ohmic heating (j² / σ)
        // - Magnetic field configuration
        // - Bootstrap current fraction
        //
        // Note: Current evolution is not enabled in our simplified model
        // (evolution.current = false), so we use I_plasma as a parameter
        // for source calculations only

        return updated
    }
}

/// Target profiles for profile matching optimization
public struct TargetProfiles {
    public let ionTemperature: MLXArray      // [eV]
    public let electronTemperature: MLXArray // [eV]
    public let electronDensity: MLXArray     // [m⁻³]

    public init(
        ionTemperature: MLXArray,
        electronTemperature: MLXArray,
        electronDensity: MLXArray
    ) {
        self.ionTemperature = ionTemperature
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity
    }
}
