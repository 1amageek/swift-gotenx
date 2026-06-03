import MLX

/// Evaluation policy for MLX arrays at API boundaries.
///
/// Solver paths keep arrays deferred so MLX can fuse operations and avoid host
/// synchronization. Diagnostic and persistence paths use eager evaluation before
/// storing arrays in Sendable value types or reading scalar metadata.
package enum MLXEvaluationMode: Sendable, Equatable {
    case eager
    case deferred

    package func wrap(_ array: MLXArray) -> EvaluatedArray {
        switch self {
        case .eager:
            EvaluatedArray(evaluating: array)
        case .deferred:
            .uncheckedLazy(array)
        }
    }

    package func wrapBatch(_ arrays: [MLXArray]) -> [EvaluatedArray] {
        switch self {
        case .eager:
            EvaluatedArray.evaluatingBatch(arrays)
        case .deferred:
            arrays.map { .uncheckedLazy($0) }
        }
    }
}
