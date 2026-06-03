import Foundation

public enum ProfileValidationMatrixError: Error, Sendable, Equatable, CustomStringConvertible {
    case emptyProfileSet(label: String)
    case emptyTimeSeries(label: String)
    case inconsistentProfileLength(label: String, quantity: String, expected: Int, actual: Int)
    case nonFiniteValue(label: String, quantity: String, index: Int, value: Float)
    case nonPositiveValue(label: String, quantity: String, index: Int, value: Float)
    case nonMonotonicRadius(label: String, index: Int, previous: Float, current: Float)
    case radiusOutOfRange(label: String, index: Int, value: Float, tolerance: Float)
    case radiusMismatch(index: Int, predicted: Float, reference: Float, tolerance: Float)
    case timeMismatch(predicted: Float, reference: Float, tolerance: Float)
    case timeCountMismatch(predicted: Int, reference: Int)

    public var description: String {
        switch self {
        case .emptyProfileSet(let label):
            return "\(label) profile set is empty"
        case .emptyTimeSeries(let label):
            return "\(label) time series is empty"
        case .inconsistentProfileLength(let label, let quantity, let expected, let actual):
            return "\(label).\(quantity) has \(actual) values, expected \(expected)"
        case .nonFiniteValue(let label, let quantity, let index, let value):
            return "\(label).\(quantity)[\(index)] is non-finite: \(value)"
        case .nonPositiveValue(let label, let quantity, let index, let value):
            return "\(label).\(quantity)[\(index)] must be positive: \(value)"
        case .nonMonotonicRadius(let label, let index, let previous, let current):
            return "\(label).normalizedRadius is not strictly increasing at \(index): previous=\(previous), current=\(current)"
        case .radiusOutOfRange(let label, let index, let value, let tolerance):
            return "\(label).normalizedRadius[\(index)] must be within [0, 1] ± \(tolerance): \(value)"
        case .radiusMismatch(let index, let predicted, let reference, let tolerance):
            return "normalizedRadius[\(index)] mismatch: predicted=\(predicted), reference=\(reference), tolerance=\(tolerance)"
        case .timeMismatch(let predicted, let reference, let tolerance):
            return "profile time mismatch: predicted=\(predicted), reference=\(reference), tolerance=\(tolerance)"
        case .timeCountMismatch(let predicted, let reference):
            return "time count mismatch: predicted=\(predicted), reference=\(reference)"
        }
    }
}
