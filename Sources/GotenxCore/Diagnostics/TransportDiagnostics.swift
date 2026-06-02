import MLX
import Foundation

// MARK: - Transport Diagnostics

/// Diagnostics for transport coefficients validation
///
/// Checks transport coefficients (χᵢ, χₑ, D, v) for:
/// - Large variations (> 10⁴ range)
/// - Negative values (unphysical)
/// - NaN/Inf values (numerical breakdown)
///
/// ## Example Usage
///
/// ```swift
/// let results = TransportDiagnostics.diagnose(
///     coefficients: transportCoeffs,
///     step: step,
///     time: time
/// )
///
/// for result in results {
///     if result.severity == .error || result.severity == .critical {
///         print(result.formatted())
///     }
/// }
/// ```
public struct TransportDiagnostics {

    /// Diagnose transport coefficients
    ///
    /// Performs multiple checks on transport coefficients and returns
    /// an array of diagnostic results (one per issue found).
    ///
    /// - Parameters:
    ///   - coefficients: Transport coefficients to check
    ///   - step: Current timestep
    ///   - time: Current simulation time [s]
    /// - Returns: Array of diagnostic results
    ///
    /// ## Checks Performed
    ///
    /// 1. **Range check**: Detect large variations (χ_max / χ_min > 10⁴)
    /// 2. **Negativity check**: Diffusivity must be ≥ 0
    /// 3. **NaN/Inf check**: Detect numerical breakdown
    public static func diagnose(
        coefficients: TransportCoefficients,
        step: Int,
        time: Float
    ) -> [DiagnosticResult] {
        var results: [DiagnosticResult] = []

        // Check ion diffusivity
        results.append(contentsOf: checkDiffusivity(
            array: coefficients.ionHeatDiffusivity.value,
            name: "χᵢ (ion)",
            step: step,
            time: time
        ))

        // Check electron diffusivity
        results.append(contentsOf: checkDiffusivity(
            array: coefficients.electronHeatDiffusivity.value,
            name: "χₑ (electron)",
            step: step,
            time: time
        ))

        // Check particle diffusivity
        results.append(contentsOf: checkDiffusivity(
            array: coefficients.particleDiffusivity.value,
            name: "D (particle)",
            step: step,
            time: time
        ))

        // Check convection velocity (can be negative, so only check finiteness)
        results.append(contentsOf: checkFiniteness(
            array: coefficients.convectionVelocity.value,
            name: "v (convection)",
            step: step,
            time: time
        ))

        return results
    }

    // MARK: - Private Helpers

    /// Check diffusivity coefficient (must be non-negative and finite)
    private static func checkDiffusivity(
        array: MLXArray,
        name: String,
        step: Int,
        time: Float
    ) -> [DiagnosticResult] {
        var results: [DiagnosticResult] = []

        eval(array)

        // Extract min/max values
        let minValue = array.min().item(Float.self)
        let maxValue = array.max().item(Float.self)

        // 1. Check for NaN/Inf
        if !minValue.isFinite || !maxValue.isFinite {
            results.append(DiagnosticResult(
                name: "TransportCoefficients",
                severity: .critical,
                message: "NaN or Inf detected in \(name)",
                value: nil,
                threshold: nil,
                time: time,
                step: step
            ))
            return results  // Stop checking if NaN/Inf found
        }

        // 2. Check for negative values
        if minValue < 0 {
            results.append(DiagnosticResult(
                name: "TransportCoefficients",
                severity: .error,
                message: "Negative diffusivity in \(name): min = \(String(format: "%.2e", minValue))",
                value: minValue,
                threshold: 0.0,
                time: time,
                step: step
            ))
        }

        // 3. Check for large range (indicates potential numerical issues)
        if maxValue > 0 && minValue > 0 {
            let range = maxValue / minValue

            if range > 1e4 {
                results.append(DiagnosticResult(
                    name: "TransportCoefficients",
                    severity: .warning,
                    message: "Large variation in \(name): \(String(format: "%.2e", range))× range",
                    value: range,
                    threshold: 1e4,
                    time: time,
                    step: step
                ))
            }
        }

        // 4. Info: Report normal range
        if results.isEmpty {
            results.append(DiagnosticResult(
                name: "TransportCoefficients",
                severity: .info,
                message: "\(name) in normal range: [\(String(format: "%.2e", minValue)), \(String(format: "%.2e", maxValue))]",
                value: maxValue / minValue,
                threshold: nil,
                time: time,
                step: step
            ))
        }

        return results
    }

    /// Check finiteness only (for quantities that can be negative)
    private static func checkFiniteness(
        array: MLXArray,
        name: String,
        step: Int,
        time: Float
    ) -> [DiagnosticResult] {
        var results: [DiagnosticResult] = []

        eval(array)

        let minValue = array.min().item(Float.self)
        let maxValue = array.max().item(Float.self)

        // Check for NaN/Inf
        if !minValue.isFinite || !maxValue.isFinite {
            results.append(DiagnosticResult(
                name: "TransportCoefficients",
                severity: .critical,
                message: "NaN or Inf detected in \(name)",
                value: nil,
                threshold: nil,
                time: time,
                step: step
            ))
        } else {
            results.append(DiagnosticResult(
                name: "TransportCoefficients",
                severity: .info,
                message: "\(name) in range: [\(String(format: "%.2e", minValue)), \(String(format: "%.2e", maxValue))]",
                value: nil,
                threshold: nil,
                time: time,
                step: step
            ))
        }

        return results
    }

    /// Check specific coefficient value at index
    ///
    /// Useful for debugging specific cells.
    ///
    /// - Parameters:
    ///   - array: Coefficient array
    ///   - index: Cell index to check
    /// - Returns: Value at index
    public static func checkValueAt(array: MLXArray, index: Int) -> Float {
        eval(array)
        return array[index].item(Float.self)
    }
}
