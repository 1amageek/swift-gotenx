import Foundation

// MARK: - Source Model Protocol

/// Source model protocol for computing heating, particle, and current sources
///
/// Diagnostic source terms must include metadata so power accounting is explicit.
/// Solver source terms may omit metadata because they are evaluated repeatedly inside
/// Newton iterations and automatic differentiation transforms.
public protocol SourceModel: PhysicsComponent, Sendable {
    /// Compute source terms for diagnostics and time-series capture.
    ///
    /// - Parameters:
    ///   - profiles: Current core profiles
    ///   - geometry: Tokamak geometry
    ///   - parameters: Source model parameters
    /// - Returns: Source terms with source metadata
    func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms

    /// Compute source terms for solver residual evaluation.
    ///
    /// This path is called repeatedly inside Newton iterations and AD transforms.
    /// Implementations should return differentiable arrays only and avoid metadata
    /// integration or host-side scalar reads.
    func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) -> SourceTerms
}

extension SourceModel {
    public func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) -> SourceTerms {
        do {
            return try computeTerms(profiles: profiles, geometry: geometry, parameters: parameters)
        } catch {
            let cellCount = profiles.ionTemperature.shape.first ?? 0
            return SourceTerms.invalidNumerics(cellCount: cellCount)
        }
    }
}
