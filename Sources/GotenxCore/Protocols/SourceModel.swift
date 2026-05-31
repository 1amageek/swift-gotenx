import Foundation

// MARK: - Source Model Protocol

/// Source model protocol for computing heating, particle, and current sources
///
/// Phase 4a: Added optional metadata computation for power balance tracking.
/// Models can opt-in by implementing `computeTermsWithMetadata()`.
public protocol SourceModel: PhysicsComponent, Sendable {
    /// Compute source terms (Phase 3 compatibility)
    ///
    /// - Parameters:
    ///   - profiles: Current core profiles
    ///   - geometry: Tokamak geometry
    ///   - params: Source model parameters
    /// - Returns: Source terms (heating, particles, current)
    func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms

    /// Compute source terms for solver residual evaluation.
    ///
    /// This path is called repeatedly inside Newton iterations and AD transforms.
    /// Implementations should return differentiable arrays only and avoid metadata
    /// integration or host-side scalar reads.
    func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms

    /// Phase 4a: Compute source terms with metadata (optional)
    ///
    /// Models that implement this method enable accurate power balance tracking.
    /// Default implementation falls back to `computeTerms()` without metadata.
    ///
    /// - Parameters:
    ///   - profiles: Current core profiles
    ///   - geometry: Tokamak geometry
    ///   - params: Source model parameters
    /// - Returns: Source terms with metadata
    func computeTermsWithMetadata(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms
}

// MARK: - Default Implementation (Phase 3 Compatibility)

extension SourceModel {
    public func computeTermsForSolver(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        computeTerms(profiles: profiles, geometry: geometry, params: params)
    }

    /// Default implementation: calls `computeTerms()` without metadata
    ///
    /// Phase 3 models automatically get this fallback behavior.
    public func computeTermsWithMetadata(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: SourceParameters
    ) -> SourceTerms {
        // Fall back to Phase 3 implementation (no metadata)
        return computeTerms(profiles: profiles, geometry: geometry, params: params)
    }
}
