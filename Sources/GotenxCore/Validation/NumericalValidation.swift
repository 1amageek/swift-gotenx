import MLX

public enum NumericalValidation {
    public static func validateShape(
        _ array: MLXArray,
        field: String,
        expected: [Int]
    ) throws {
        guard array.shape == expected else {
            throw NumericalValidationError.invalidShape(
                field: field,
                expected: expected,
                actual: array.shape
            )
        }

        guard expected.allSatisfy({ $0 > 0 }) else {
            throw NumericalValidationError.invalidValue(
                field: field,
                reason: "shape dimensions must be positive"
            )
        }
    }

    public static func validateFinite(_ array: MLXArray, field: String) throws {
        guard array.shape.allSatisfy({ $0 > 0 }) else {
            throw NumericalValidationError.invalidValue(
                field: field,
                reason: "shape dimensions must be positive"
            )
        }

        let range = MLX.stacked([
            array.min(keepDims: false),
            array.max(keepDims: false)
        ], axis: 0).asArray(Float.self)

        guard range.count == 2 else {
            throw NumericalValidationError.invalidValue(
                field: field,
                reason: "range computation returned \(range.count) values"
            )
        }

        guard range[0].isFinite, range[1].isFinite else {
            throw NumericalValidationError.nonFinite(
                field: field,
                minimum: range[0],
                maximum: range[1]
            )
        }
    }

    public static func validatePositive(_ array: MLXArray, field: String) throws {
        try validateFinite(array, field: field)
        let minimum = array.min(keepDims: false).item(Float.self)
        guard minimum > 0 else {
            throw NumericalValidationError.nonPositive(field: field, minimum: minimum)
        }
    }

    public static func validateNonNegative(_ array: MLXArray, field: String) throws {
        try validateFinite(array, field: field)
        let minimum = array.min(keepDims: false).item(Float.self)
        guard minimum >= 0 else {
            throw NumericalValidationError.negativeValue(field: field, minimum: minimum)
        }
    }
}
