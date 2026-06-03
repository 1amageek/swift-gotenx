import Foundation

/// Complete input for source-term evaluation.
///
/// The context carries both the physical data and the evaluation purpose so callers
/// do not have to coordinate separate lazy/eager and metadata decisions.
public struct SourceEvaluationContext: Sendable {
    public enum Purpose: Sendable {
        case diagnostic
        case solver
    }

    public let profiles: CoreProfiles
    public let geometry: Geometry
    public let geometricFactors: GeometricFactors
    public let parameters: SourceParameters
    public let purpose: Purpose

    public init(
        profiles: CoreProfiles,
        geometry: Geometry,
        geometricFactors: GeometricFactors? = nil,
        parameters: SourceParameters,
        purpose: Purpose
    ) {
        self.profiles = profiles
        self.geometry = geometry
        self.parameters = parameters
        self.purpose = purpose
        self.geometricFactors = geometricFactors ?? GeometricFactors.from(
            geometry: geometry,
            evaluationMode: purpose.evaluationMode
        )
    }
}

extension SourceEvaluationContext {
    package var cellCount: Int {
        profiles.ionTemperature.shape[0]
    }

    package var evaluationMode: MLXEvaluationMode {
        switch purpose {
        case .diagnostic:
            .eager
        case .solver:
            .deferred
        }
    }

    package var includesMetadata: Bool {
        purpose == .diagnostic
    }

    package var validatesDebugUnits: Bool {
        purpose == .diagnostic
    }

    package var initialMetadata: SourceMetadataCollection? {
        includesMetadata ? SourceMetadataCollection.empty : nil
    }

    package func zeroSourceTerms() -> SourceTerms {
        SourceTerms.zero(
            cellCount: cellCount,
            evaluationMode: evaluationMode,
            metadata: initialMetadata,
            validateDebugUnits: validatesDebugUnits
        )
    }
}

extension SourceEvaluationContext.Purpose {
    package var evaluationMode: MLXEvaluationMode {
        switch self {
        case .diagnostic:
            .eager
        case .solver:
            .deferred
        }
    }
}
