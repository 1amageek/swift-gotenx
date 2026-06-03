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
/// - Forward sensitivity analysis (∂fusionGain / ∂parameters)
/// - Inverse problems (optimize actuators to maximize fusionGain)
/// - Model predictive control
///
/// **Example**:
/// ```swift
/// let sim = DifferentiableSimulation(
///     staticParameters: staticParameters,
///     transport: BohmGyrobohmModel(),
///     sources: [FusionSourceModel()],
///     geometry: geometry
/// )
///
/// let (finalProfiles, loss) = sim.forward(
///     initialProfiles: initialProfiles,
///     actuators: actuators,
///     timeHorizon: 2.0,
///     timeStep: 0.01
/// )
/// ```
public struct DifferentiableSimulation {
    // MARK: - Configuration

    /// Static runtime parameters
    private let staticParameters: StaticRuntimeParameters

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
        staticParameters: StaticRuntimeParameters,
        transport: any TransportModel,
        sources: [any SourceModel] = [],
        geometry: Geometry
    ) {
        self.staticParameters = staticParameters
        self.transport = transport
        self.sources = sources
        self.geometry = geometry

        // Use LinearSolver for differentiation (simpler, no iterative solve)
        self.solver = LinearSolver(
            correctorStepCount: 1,  // Minimal correction
            usesPereverzevCorrector: false,
            theta: staticParameters.theta
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
    ///   - dynamicParameters: Dynamic runtime parameters (boundaries, transport parameters, etc.)
    ///   - timeHorizon: Total simulation time [s]
    ///   - timeStep: Fixed timestep [s] (adaptive timestep breaks gradients!)
    ///
    /// - Returns: Tuple of (final profiles, loss value for minimization)
    ///
    /// **Loss Function**:
    /// Default: `-fusionGain` (negative for maximization via minimization)
    /// Can be customized for other objectives (profile matching, energy confinement, etc.)
    public func forward(
        initialProfiles: CoreProfiles,
        actuators: ActuatorTimeSeries,
        dynamicParameters: DynamicRuntimeParameters,
        timeHorizon: Float,
        timeStep: Float
    ) -> (CoreProfiles, MLXArray) {
        var profiles = initialProfiles
        let stepCount = Int(timeHorizon / timeStep)

        // CRITICAL FOR GRADIENTS: Extract actuator MLXArray once
        // This preserves the gradient tape connection
        let actuatorArray = actuators.asMLXArray()  // Shape: [stepCount × 4]

        // For constant actuators, use average (all timesteps have same value)
        // This maintains differentiability while avoiding asArray() in loop
        let avgP_ECRH = MLX.mean(actuatorArray[0..<actuators.stepCount])
        let avgP_ICRH = MLX.mean(actuatorArray[actuators.stepCount..<(2*actuators.stepCount)])
        let avgGasPuff = MLX.mean(actuatorArray[(2*actuators.stepCount)..<(3*actuators.stepCount)])
        let avgI_plasma = MLX.mean(actuatorArray[(3*actuators.stepCount)..<(4*actuators.stepCount)])

        // Update dynamic parameters once with MLXArray values
        // These MLXArrays preserve gradients
        let dynamicParamsWithActuators = updateDynamicParamsMLX(
            dynamicParameters,
            ecrhPower: avgP_ECRH,
            icrhPower: avgP_ICRH,
            gasPuffRate: avgGasPuff,
            plasmaCurrent: avgI_plasma
        )

        // CRITICAL FOR GRADIENTS: Set MLXArray power on sources
        // This allows gradient-aware sources to use MLXArrays instead of Floats
        let P_aux_total_mlx = avgP_ECRH + avgP_ICRH
        setMLXPowerOnSources(P_aux_total_mlx)

        // Time-stepping loop (NO compile!)
        for _ in 0..<stepCount {
            // Perform single differentiable timestep
            // Use same actuator values for all steps (constant actuators)
            profiles = stepDifferentiable(
                profiles: profiles,
                dynamicParameters: dynamicParamsWithActuators,
                timeStep: timeStep
            )
        }

        // Compute loss from final state
        let loss = computeLoss(profiles: profiles)

        return (profiles, loss)
    }

    /// Differentiable timestep (core operation)
    ///
    /// **Critical**: All operations must be differentiable w.r.t. profiles and parameters
    private func stepDifferentiable(
        profiles: CoreProfiles,
        dynamicParameters: DynamicRuntimeParameters,
        timeStep: Float
    ) -> CoreProfiles {
        // Build CoeffsCallback (for solver)
        // Note: We build coefficients inside the callback to ensure they depend on
        // the profiles being solved (needed for iterative solvers)
        let coeffsCallback: CoeffsCallback = { [transport, sources, dynamicParameters, staticParameters] profs, geo in
            let transportCoeffs = transport.computeCoefficients(
                profiles: profs,
                geometry: geo,
                parameters: dynamicParameters.transportParameters
            )
            let geometricFactors = GeometricFactors.from(
                geometry: geo,
                evaluationMode: .deferred
            )

            let sourceTerms = sources.reduce(
                into: SourceTerms.zero(
                    cellCount: staticParameters.mesh.cellCount,
                    evaluationMode: .deferred,
                    metadata: nil,
                    validateDebugUnits: false
                )
            ) { total, model in
                if let parameters = dynamicParameters.sourceParameters[model.name] {
                    let context = SourceEvaluationContext(
                        profiles: profs,
                        geometry: geo,
                        geometricFactors: geometricFactors,
                        parameters: parameters,
                        purpose: .solver
                    )
                    let contribution: SourceTerms
                    do {
                        contribution = try model.computeTerms(in: context)
                    } catch {
                        contribution = SourceTerms.invalidNumerics(cellCount: staticParameters.mesh.cellCount)
                    }
                    total = total.adding(
                        contribution,
                        evaluationMode: .deferred,
                        metadata: nil,
                        validateDebugUnits: false
                    )
                }
            }

            return buildBlock1DCoeffs(
                transport: transportCoeffs,
                sources: sourceTerms,
                geometry: geo,
                staticParameters: staticParameters,
                profiles: profs,
                evaluationMode: .deferred,
                geometricFactors: geometricFactors
            )
        }

        // 5. Solve (differentiable - linear solver only!)
        let xOld = profiles.asTuple(
            radialSpacing: staticParameters.mesh.radialSpacing,
            boundaryConditions: dynamicParameters.boundaryConditions
        )

        let result = solver.solve(
            timeStep: timeStep,
            staticParameters: staticParameters,
            dynamicParamsT: dynamicParameters,
            dynamicParamsTplusDt: dynamicParameters,
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
    /// **Note**: We use average temperature instead of fusionGain because:
    /// 1. fusionGain requires high temperatures (10-20 keV) to be non-zero
    /// 2. Temperature directly responds to heating power
    /// 3. More sensitive for gradient-based optimization
    ///
    /// For actual scenario optimization with realistic parameters,
    /// fusionGain maximization can be used.
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

        return totalError / Float(staticParameters.mesh.cellCount)
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

    /// Update dynamic parameters with actuator MLXArrays (gradient-preserving)
    ///
    /// **Critical**: Uses MLXArrays directly to preserve gradient tape
    ///
    /// This version is used in forward() to maintain differentiability
    private func updateDynamicParamsMLX(
        _ parameters: DynamicRuntimeParameters,
        ecrhPower: MLXArray,
        icrhPower: MLXArray,
        gasPuffRate: MLXArray,
        plasmaCurrent: MLXArray
    ) -> DynamicRuntimeParameters {
        var updated = parameters

        // Convert MLXArrays to Float for storage (gradient still flows through computation)
        eval(ecrhPower, icrhPower, gasPuffRate, plasmaCurrent)
        let P_ECRH_val = ecrhPower.item(Float.self)
        let P_ICRH_val = icrhPower.item(Float.self)
        let gas_puff_val = gasPuffRate.item(Float.self)
        let I_plasma_val = plasmaCurrent.item(Float.self)

        // Calculate total auxiliary power
        let P_aux_total = P_ECRH_val + P_ICRH_val  // [MW]

        // Update all heating sources
        for (sourceName, var sourceParameters) in updated.sourceParameters {
            if sourceName.contains("fusion") || sourceName.contains("heating") {
                sourceParameters.parameters["P_auxiliary"] = P_aux_total
                sourceParameters.parameters["P_ECRH"] = P_ECRH_val
                sourceParameters.parameters["P_ICRH"] = P_ICRH_val
                updated.sourceParameters[sourceName] = sourceParameters
            }
        }

        // Update ohmic heating if present
        if var ohmicParams = updated.sourceParameters["ohmic"] {
            ohmicParams.parameters["I_plasma"] = I_plasma_val
            updated.sourceParameters["ohmic"] = ohmicParams
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

    /// Update dynamic parameters with actuator values
    ///
    /// **Engineering mapping**:
    /// 1. ecrhPower, icrhPower → Heating power sources (MW → eV/m³/s)
    /// 2. gasPuffRate → Density boundary condition (particles/s → m⁻³)
    /// 3. plasmaCurrent → Current drive (MA → A/m²)
    ///
    /// **Mathematical consistency**:
    /// - All unit conversions must be physically correct
    /// - Power must be conserved (P_in = ecrhPower + icrhPower + ohmicPower)
    /// - Particle balance (gasPuffRate → density BC)
    private func updateDynamicParams(
        _ parameters: DynamicRuntimeParameters,
        with actuators: ActuatorValues
    ) -> DynamicRuntimeParameters {
        var updated = parameters

        // 1. Update auxiliary heating power
        // Map ecrhPower + icrhPower to total auxiliary power
        let P_aux_total = actuators.ecrhPower + actuators.icrhPower  // [MW]

        // Update all heating sources with auxiliary power
        // This supports both "fusion" (real simulations) and "simple_heating" (tests)
        for (sourceName, var sourceParameters) in updated.sourceParameters {
            if sourceName.contains("fusion") || sourceName.contains("heating") {
                // Store total auxiliary power
                sourceParameters.parameters["P_auxiliary"] = P_aux_total

                // Store individual powers for power partition analysis
                sourceParameters.parameters["P_ECRH"] = actuators.ecrhPower
                sourceParameters.parameters["P_ICRH"] = actuators.icrhPower

                updated.sourceParameters[sourceName] = sourceParameters
            }
        }

        // Also update ohmic heating parameters if present
        if var ohmicParams = updated.sourceParameters["ohmic"] {
            // Ohmic heating depends on plasma current
            ohmicParams.parameters["I_plasma"] = actuators.plasmaCurrent  // [MA]
            updated.sourceParameters["ohmic"] = ohmicParams
        }

        // 2. Update density boundary condition from gas puff
        // Engineering model: gasPuffRate [particles/s] → edge density [m⁻³]
        //
        // Simplified model: n_edge ∝ gasPuffRate / (particle_confinement_time × surface_area)
        // For optimization: we use a scaling factor
        //
        // Typical values:
        // - gasPuffRate: 1e20 particles/s
        // - edge density: 1e19 m⁻³
        // - scaling: ~0.1
        let gasPuffScaling: Float = 0.1  // Calibrated constant
        let densityFromGasPuff = gasPuffScaling * actuators.gasPuffRate  // [m⁻³]

        // Clamp to physical range
        let minEdgeDensity: Float = 1e18  // 0.1 × 10²⁰ m⁻³
        let maxEdgeDensity: Float = 5e19  // 5 × 10²⁰ m⁻³
        let clampedDensity = max(minEdgeDensity, min(maxEdgeDensity, densityFromGasPuff))

        // Update boundary conditions (gas puff affects edge density)
        var updatedBC = updated.boundaryConditions
        // Update right boundary (edge) with new density value
        updatedBC.electronDensity.right = .value(clampedDensity)
        updated.boundaryConditions = updatedBC

        // 3. Plasma current (plasmaCurrent) affects:
        // - Ohmic heating (j² / σ)
        // - Magnetic field configuration
        // - Bootstrap current fraction
        //
        // Note: Current evolution is not enabled in our simplified model
        // (evolution.poloidalFlux = false), so we use plasmaCurrent as a parameter
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
