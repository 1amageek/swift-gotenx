import Foundation

public enum NumericalValidationError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalidShape(field: String, expected: [Int], actual: [Int])
    case invalidValue(field: String, reason: String)
    case nonFinite(field: String, minimum: Float, maximum: Float)
    case nonPositive(field: String, minimum: Float)
    case negativeValue(field: String, minimum: Float)
    case missingMetadata(field: String)

    public var description: String {
        switch self {
        case .invalidShape(let field, let expected, let actual):
            return "\(field) has shape \(actual), expected \(expected)"
        case .invalidValue(let field, let reason):
            return "\(field) is invalid: \(reason)"
        case .nonFinite(let field, let minimum, let maximum):
            return "\(field) contains non-finite values: min=\(minimum), max=\(maximum)"
        case .nonPositive(let field, let minimum):
            return "\(field) must be positive: min=\(minimum)"
        case .negativeValue(let field, let minimum):
            return "\(field) must be non-negative: min=\(minimum)"
        case .missingMetadata(let field):
            return "\(field) requires source metadata"
        }
    }
}
