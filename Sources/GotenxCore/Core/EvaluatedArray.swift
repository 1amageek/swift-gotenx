import MLX
import Foundation

// MARK: - EvaluatedArray

/// Type-safe wrapper ensuring MLXArray evaluation (ONLY type marked @unchecked Sendable)
///
/// This is the foundation of the type system for ensuring evaluation safety.
/// All MLXArray computations must be wrapped in EvaluatedArray before crossing
/// actor boundaries or being stored in Sendable data structures.
public struct EvaluatedArray: @unchecked Sendable {
    private let array: MLXArray

    /// Create an evaluated array by forcing evaluation of the input
    ///
    /// - Parameter array: Lazy MLXArray to evaluate
    public init(evaluating array: MLXArray) {
        eval(array)  // Force evaluation at construction
        self.array = array
    }

    /// Batch evaluation for efficiency
    ///
    /// Evaluates all arrays in a single pass before wrapping them.
    ///
    /// - Parameter arrays: Array of lazy MLXArrays to evaluate
    /// - Returns: Array of EvaluatedArrays
    public static func evaluatingBatch(_ arrays: [MLXArray]) -> [EvaluatedArray] {
        // Force all arrays in one MLX evaluation pass.
        eval(arrays)
        return arrays.map { EvaluatedArray(uncheckedLazy: $0) }
    }

    /// Wrap an array without forcing evaluation.
    ///
    /// This is intended for pure MLX graph composition in solver internals where
    /// eager evaluation would break fusion and add synchronization points. Callers
    /// must force evaluation before storing the value across actor or task
    /// boundaries.
    package static func uncheckedLazy(_ array: MLXArray) -> EvaluatedArray {
        EvaluatedArray(uncheckedLazy: array)
    }

    /// Internal initializer for arrays whose evaluation is managed by the caller.
    private init(uncheckedLazy array: MLXArray) {
        self.array = array
    }

    /// Read-only access to the evaluated MLXArray
    public var value: MLXArray { array }

    // MARK: - Convenience Accessors

    /// Shape of the evaluated array
    public var shape: [Int] { array.shape }

    /// Number of dimensions
    public var ndim: Int { array.ndim }

    /// Data type
    public var dtype: DType { array.dtype }
}

// MARK: - EvaluatedArray Equatable

extension EvaluatedArray: Equatable {
    public static func == (lhs: EvaluatedArray, rhs: EvaluatedArray) -> Bool {
        // Use MLX's array equality
        allClose(lhs.array, rhs.array).item(Bool.self)
    }
}

// MARK: - EvaluatedArray Convenience Constructors

extension EvaluatedArray {
    /// Create zero array
    public static func zeros(_ shape: [Int], dtype: DType = .float32) -> EvaluatedArray {
        EvaluatedArray(evaluating: MLXArray.zeros(shape, dtype: dtype))
    }

    /// Create ones array
    public static func ones(_ shape: [Int], dtype: DType = .float32) -> EvaluatedArray {
        EvaluatedArray(evaluating: MLXArray.ones(shape, dtype: dtype))
    }

    /// Create array filled with a constant value
    public static func full(_ shape: [Int], value: Float, dtype: DType = .float32) -> EvaluatedArray {
        EvaluatedArray(evaluating: MLXArray.full(shape, values: MLXArray(value), dtype: dtype))
    }
}
