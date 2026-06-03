import MLX

public enum NumericalValidation {
    public enum Requirement {
        case finite
        case positive
        case nonNegative
    }

    public struct Field {
        public let array: MLXArray
        public let field: String
        public let requirement: Requirement

        public init(_ array: MLXArray, field: String, requirement: Requirement) {
            self.array = array
            self.field = field
            self.requirement = requirement
        }

        public static func finite(_ array: MLXArray, field: String) -> Field {
            Field(array, field: field, requirement: .finite)
        }

        public static func positive(_ array: MLXArray, field: String) -> Field {
            Field(array, field: field, requirement: .positive)
        }

        public static func nonNegative(_ array: MLXArray, field: String) -> Field {
            Field(array, field: field, requirement: .nonNegative)
        }
    }

    private struct RangeSummary {
        let minimum: Float
        let maximum: Float
    }

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
        _ = try validatedRange(array, field: field)
    }

    public static func validatePositive(_ array: MLXArray, field: String) throws {
        let range = try validatedRange(array, field: field)
        guard range.minimum > 0 else {
            throw NumericalValidationError.nonPositive(field: field, minimum: range.minimum)
        }
    }

    public static func validateNonNegative(_ array: MLXArray, field: String) throws {
        let range = try validatedRange(array, field: field)
        guard range.minimum >= 0 else {
            throw NumericalValidationError.negativeValue(field: field, minimum: range.minimum)
        }
    }

    public static func validate(_ fields: [Field]) throws {
        guard !fields.isEmpty else {
            return
        }

        for field in fields {
            try validateNonEmptyShape(field.array, field: field.field)
        }

        let reductions = fields.flatMap { field in
            [
                field.array.min(keepDims: false),
                field.array.max(keepDims: false)
            ]
        }
        let ranges = MLX.stacked(reductions, axis: 0).asArray(Float.self)
        let expectedCount = fields.count * 2

        guard ranges.count == expectedCount else {
            throw NumericalValidationError.invalidValue(
                field: "NumericalValidation.fields",
                reason: "range computation returned \(ranges.count) values"
            )
        }

        for index in fields.indices {
            let field = fields[index]
            let range = RangeSummary(
                minimum: ranges[index * 2],
                maximum: ranges[index * 2 + 1]
            )
            try validate(range: range, field: field.field, requirement: field.requirement)
        }
    }

    private static func validatedRange(_ array: MLXArray, field: String) throws -> RangeSummary {
        try validateNonEmptyShape(array, field: field)

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

        let summary = RangeSummary(minimum: range[0], maximum: range[1])
        try validate(range: summary, field: field, requirement: .finite)
        return summary
    }

    private static func validateNonEmptyShape(_ array: MLXArray, field: String) throws {
        guard array.shape.allSatisfy({ $0 > 0 }) else {
            throw NumericalValidationError.invalidValue(
                field: field,
                reason: "shape dimensions must be positive"
            )
        }
    }

    private static func validate(
        range: RangeSummary,
        field: String,
        requirement: Requirement
    ) throws {
        guard range.minimum.isFinite, range.maximum.isFinite else {
            throw NumericalValidationError.nonFinite(
                field: field,
                minimum: range.minimum,
                maximum: range.maximum
            )
        }

        switch requirement {
        case .finite:
            return
        case .positive:
            guard range.minimum > 0 else {
                throw NumericalValidationError.nonPositive(field: field, minimum: range.minimum)
            }
        case .nonNegative:
            guard range.minimum >= 0 else {
                throw NumericalValidationError.negativeValue(field: field, minimum: range.minimum)
            }
        }
    }
}
