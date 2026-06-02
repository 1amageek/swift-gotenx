import Foundation

// MARK: - Transport Model Protocol

/// Transport model protocol for computing heat and particle transport coefficients
public protocol TransportModel: PhysicsComponent, Sendable {
    /// Compute transport coefficients
    ///
    /// - Parameters:
    ///   - profiles: Current core profiles
    ///   - geometry: Tokamak geometry
    ///   - parameters: Transport model parameters
    /// - Returns: Transport coefficients (chi, D, V)
    func computeCoefficients(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: TransportParameters
    ) -> TransportCoefficients
}
