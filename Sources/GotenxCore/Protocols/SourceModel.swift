import Foundation

// MARK: - Source Model Protocol

/// Source model protocol for computing heating, particle, and current sources
///
/// Diagnostic source terms must include metadata so power accounting is explicit.
/// Solver source terms may omit metadata because they are evaluated repeatedly inside
/// Newton iterations and automatic differentiation transforms.
public protocol SourceModel: PhysicsComponent, Sendable {
    /// Compute source terms from a complete evaluation context.
    ///
    /// Internal solver paths pass `.solver`, which keeps arrays lazy and omits
    /// metadata. Diagnostic paths pass `.diagnostic`, which evaluates arrays and
    /// includes metadata for power accounting.
    func computeTerms(in context: SourceEvaluationContext) throws -> SourceTerms
}

extension SourceModel {
    /// Compute source terms for diagnostics and time-series capture.
    ///
    /// This convenience entry point builds a diagnostic context. Solver and
    /// differentiation paths should pass a full `SourceEvaluationContext` so the
    /// caller controls eager vs deferred MLX evaluation.
    public func computeTerms(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: SourceParameters
    ) throws -> SourceTerms {
        try computeTerms(
            in: SourceEvaluationContext(
                profiles: profiles,
                geometry: geometry,
                parameters: parameters,
                purpose: .diagnostic
            )
        )
    }
}
