import Foundation

// MARK: - Pedestal Output

/// Pedestal model output
///
/// **Units**: eV for temperature, m^-3 for density
///
/// Pedestal parameters use the same units as CoreProfiles and BoundaryConditions:
/// - Temperature: eV (electron volts)
/// - Density: m^-3 (particles per cubic meter)
/// - Width: m (meters)
///
/// This ensures consistency throughout the runtime system.
public struct PedestalOutput: Sendable, Equatable {
    /// Pedestal temperature [eV]
    public let temperature: Float

    /// Pedestal density [m^-3]
    public let density: Float

    /// Pedestal width [m]
    public let width: Float

    public init(temperature: Float, density: Float, width: Float) {
        self.temperature = temperature
        self.density = density
        self.width = width
    }
}

// MARK: - Pedestal Model Protocol

/// Pedestal model protocol for computing edge boundary conditions
public protocol PedestalModel: PhysicsComponent {
    /// Compute pedestal boundary conditions
    ///
    /// - Parameters:
    ///   - profiles: Current core profiles
    ///   - geometry: Tokamak geometry
    ///   - parameters: Pedestal model parameters
    /// - Returns: Pedestal output (boundary conditions)
    func computePedestal(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: [String: Float]
    ) -> PedestalOutput
}
