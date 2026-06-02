import Foundation

// MARK: - Solver Type

/// Solver type enumeration
public enum SolverType: String, Sendable, Codable {
    case linear
    case newtonRaphson
    case optimizer
}

// MARK: - Static Runtime Parameters

/// Static runtime parameters (trigger recompilation when changed)
public struct StaticRuntimeParameters: Sendable, Codable, Equatable {
    /// Mesh configuration
    public let mesh: MeshConfig

    /// Evolve ion heat transport equation
    public let evolveIonHeat: Bool

    /// Evolve electron heat transport equation
    public let evolveElectronHeat: Bool

    /// Evolve electron density equation
    public let evolveElectronDensity: Bool

    /// Evolve current diffusion equation
    public let evolvePoloidalFlux: Bool

    /// Solver type
    public let solverType: SolverType

    /// Theta parameter for time discretization (0: explicit, 0.5: Crank-Nicolson, 1: implicit)
    public let theta: Float

    /// Solver tolerance
    public let solverTolerance: Float

    /// Maximum solver iterations
    public let solverMaximumIterations: Int

    public init(
        mesh: MeshConfig,
        evolveIonHeat: Bool = true,
        evolveElectronHeat: Bool = true,
        evolveElectronDensity: Bool = true,
        evolvePoloidalFlux: Bool = true,
        solverType: SolverType = .newtonRaphson,
        theta: Float = 0.5,
        solverTolerance: Float = 1e-6,
        solverMaximumIterations: Int = 30
    ) {
        self.mesh = mesh
        self.evolveIonHeat = evolveIonHeat
        self.evolveElectronHeat = evolveElectronHeat
        self.evolveElectronDensity = evolveElectronDensity
        self.evolvePoloidalFlux = evolvePoloidalFlux
        self.solverType = solverType
        self.theta = theta
        self.solverTolerance = solverTolerance
        self.solverMaximumIterations = solverMaximumIterations
    }
}

// MARK: - Dynamic Runtime Parameters

/// Dynamic runtime parameters (can change without recompilation)
public struct DynamicRuntimeParameters: Sendable, Codable, Equatable {
    /// Time step [s]
    public var timeStep: Float

    /// Boundary conditions
    public var boundaryConditions: BoundaryConditions

    /// Profile conditions
    public var profileConditions: ProfileConditions

    /// Source parameters by source name
    public var sourceParameters: [String: SourceParameters]

    /// Transport parameters
    public var transportParameters: TransportParameters

    public init(
        timeStep: Float,
        boundaryConditions: BoundaryConditions,
        profileConditions: ProfileConditions,
        sourceParameters: [String: SourceParameters] = [:],
        transportParameters: TransportParameters
    ) {
        self.timeStep = timeStep
        self.boundaryConditions = boundaryConditions
        self.profileConditions = profileConditions
        self.sourceParameters = sourceParameters
        self.transportParameters = transportParameters
    }
}
