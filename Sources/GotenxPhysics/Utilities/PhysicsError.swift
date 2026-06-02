import Foundation
import MLX

/// Errors that can occur in physics model computations
public enum PhysicsError: Error, CustomStringConvertible {
    /// Invalid temperature (negative or zero)
    case invalidTemperature(String)

    /// Invalid density (negative or zero)
    case invalidDensity(String)

    /// Non-finite values detected (NaN or Inf)
    case nonFiniteValues(String)

    /// Array shape mismatch
    case shapeMismatch(String)

    /// Physics parameter out of valid range
    case parameterOutOfRange(String)

    public var description: String {
        switch self {
        case .invalidTemperature(let msg):
            return "Invalid temperature: \(msg)"
        case .invalidDensity(let msg):
            return "Invalid density: \(msg)"
        case .nonFiniteValues(let msg):
            return "Non-finite values: \(msg)"
        case .shapeMismatch(let msg):
            return "Shape mismatch: \(msg)"
        case .parameterOutOfRange(let msg):
            return "Parameter out of range: \(msg)"
        }
    }
}

/// Utilities for validating physics inputs
public enum PhysicsValidation {

    /// Validate that temperature is positive and finite
    ///
    /// - Parameters:
    ///   - T: Temperature array [eV]
    ///   - name: Name of temperature for error messages
    /// - Throws: PhysicsError if validation fails
    public static func validateTemperature(_ T: MLXArray, name: String = "T") throws {
        // Check for positive values
        let result = MLX.all(MLX.greater(T, Float(0.0)))
        eval(result)  // Explicitly evaluate before reading
        let isPositive = result.item(Bool.self)
        guard isPositive else {
            throw PhysicsError.invalidTemperature("\(name) must be positive everywhere")
        }

        // Check for finite values
        try validateFinite(T, name: name)
    }

    /// Validate that density is positive and finite
    ///
    /// - Parameters:
    ///   - n: Density array [m⁻³]
    ///   - name: Name of density for error messages
    /// - Throws: PhysicsError if validation fails
    public static func validateDensity(_ n: MLXArray, name: String = "n") throws {
        // Check for positive values
        let result = MLX.all(MLX.greater(n, Float(0.0)))
        eval(result)  // Explicitly evaluate before reading
        let isPositive = result.item(Bool.self)
        guard isPositive else {
            throw PhysicsError.invalidDensity("\(name) must be positive everywhere")
        }

        // Check for finite values
        try validateFinite(n, name: name)
    }

    /// Validate that array contains only finite values (no NaN or Inf)
    ///
    /// - Parameters:
    ///   - array: Array to validate
    ///   - name: Name for error messages
    /// - Throws: PhysicsError if non-finite values detected
    public static func validateFinite(_ array: MLXArray, name: String) throws {
        // Check if min and max are finite
        // NaN and Inf will show up as non-finite in min/max
        let minArray = MLX.min(array)
        let maxArray = MLX.max(array)
        eval(minArray, maxArray)  // Explicitly evaluate before reading
        let minVal = minArray.item(Float.self)
        let maxVal = maxArray.item(Float.self)

        guard minVal.isFinite && maxVal.isFinite else {
            throw PhysicsError.nonFiniteValues("\(name) contains non-finite values (NaN or Inf)")
        }
    }

    /// Validate that arrays have compatible shapes
    ///
    /// - Parameters:
    ///   - arrays: Arrays to check
    ///   - names: Names of arrays for error messages
    /// - Throws: PhysicsError if shapes don't match
    public static func validateShapes(_ arrays: [MLXArray], names: [String]) throws {
        guard arrays.count == names.count else {
            throw PhysicsError.shapeMismatch("Number of arrays and names don't match")
        }

        guard !arrays.isEmpty else { return }

        let referenceShape = arrays[0].shape

        for (array, name) in zip(arrays.dropFirst(), names.dropFirst()) {
            guard array.shape == referenceShape else {
                throw PhysicsError.shapeMismatch(
                    "\(names[0]) has shape \(referenceShape) but \(name) has shape \(array.shape)"
                )
            }
        }
    }

    /// Validate that a parameter is within a valid range
    ///
    /// - Parameters:
    ///   - value: Value to check
    ///   - range: Valid range (closed)
    ///   - name: Parameter name for error messages
    /// - Throws: PhysicsError if out of range
    public static func validateRange(_ value: Float, range: ClosedRange<Float>, name: String) throws {
        guard range.contains(value) else {
            throw PhysicsError.parameterOutOfRange(
                "\(name) = \(value) is outside valid range [\(range.lowerBound), \(range.upperBound)]"
            )
        }
    }

    /// Clamp Coulomb logarithm to physically reasonable bounds
    ///
    /// - Parameter coulombLogarithm: Raw Coulomb logarithm
    /// - Returns: Bounded Coulomb logarithm [5, 25]
    public static func clampCoulombLog(_ coulombLogarithm: MLXArray) -> MLXArray {
        // Physical bounds: ln(Λ) ∈ [5, 25] for most plasmas
        return MLX.clip(coulombLogarithm, min: Float(5.0), max: Float(25.0))
    }
}
