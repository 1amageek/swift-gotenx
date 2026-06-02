import MLX
import Foundation

// MARK: - CoeffsCallback

/// Coefficient calculation callback (synchronous, thread-safe)
///
/// The callback accepts only (CoreProfiles, Geometry) as parameters.
/// Additional context (dynamicParameters, staticParameters, etc.) is provided via closure capture.
public typealias CoeffsCallback = @Sendable (CoreProfiles, Geometry) -> Block1DCoeffs

// MARK: - PDE Solver Protocol

/// PDE solver protocol for solving the transport equations
public protocol PDESolver {
    /// Solver type
    var solverType: SolverType { get }

    /// Solve PDE system for one timestep
    ///
    /// - Parameters:
    ///   - timeStep: Time step [s]
    ///   - staticParameters: Static runtime parameters
    ///   - dynamicParamsT: Dynamic parameters at time t
    ///   - dynamicParamsTplusDt: Dynamic parameters at time t+timeStep
    ///   - geometryT: Geometry at time t
    ///   - geometryTplusDt: Geometry at time t+timeStep
    ///   - xOld: Old state (Ti, Te, ne, psi) as CellVariable tuple
    ///   - coreProfilesT: Core profiles at time t
    ///   - coreProfilesTplusDt: Core profiles at time t+timeStep (initial guess)
    ///   - coeffsCallback: Callback for computing coefficients
    /// - Returns: Solver result with updated profiles
    func solve(
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
    ) -> SolverResult
}
