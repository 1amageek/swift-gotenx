import Foundation

// MARK: - Conservation Diagnostics

/// Diagnostics for conservation law drift monitoring
///
/// Passively monitors conservation drift without enforcement.
/// Use when you want to track drift but not apply corrections.
///
/// ## Use Case
///
/// When `ConservationEnforcer` is disabled, this diagnostic provides
/// passive monitoring of conservation drift for analysis.
///
/// ## Example Usage
///
/// ```swift
/// // Get enforcement results from ConservationEnforcer
/// let (_, conservationResults) = enforcer.enforce(...)
///
/// // Convert to diagnostic results for reporting
/// let diagnostics = ConservationDiagnostics.diagnose(
///     results: conservationResults
/// )
///
/// for diag in diagnostics {
///     if diag.severity == .error {
///         print(diag.formatted())
///     }
/// }
/// ```
public struct ConservationDiagnostics {

    /// Diagnose conservation drift
    ///
    /// Converts `ConservationResult` from enforcement into `DiagnosticResult`
    /// with appropriate severity levels based on drift magnitude.
    ///
    /// - Parameter results: Conservation results from enforcer
    /// - Returns: Array of diagnostic results
    ///
    /// ## Severity Mapping
    ///
    /// - drift < 0.1%: `.info` (excellent)
    /// - 0.1% ≤ drift < 0.5%: `.info` (acceptable)
    /// - 0.5% ≤ drift < 1%: `.warning` (needs monitoring)
    /// - 1% ≤ drift < 5%: `.error` (investigate)
    /// - drift ≥ 5%: `.critical` (unphysical)
    public static func diagnose(
        results: [ConservationResult]
    ) -> [DiagnosticResult] {
        return results.map { result in
            // Map drift to severity
            let severity: DiagnosticResult.Severity
            let statusMessage: String

            if result.relativeDrift < 0.001 {
                severity = .info
                statusMessage = "Excellent conservation"
            } else if result.relativeDrift < 0.005 {
                severity = .info
                statusMessage = "Acceptable drift"
            } else if result.relativeDrift < 0.01 {
                severity = .warning
                statusMessage = "Drift needs monitoring"
            } else if result.relativeDrift < 0.05 {
                severity = .error
                statusMessage = "Significant drift, investigate numerical scheme"
            } else {
                severity = .critical
                statusMessage = "Critical drift, unphysical state"
            }

            // Format drift message
            let driftPercent = result.relativeDrift * 100
            let correctionStatus = result.corrected ? "✓ Corrected" : "⚠️ Monitored only"

            let message = """
            \(result.lawName): \(statusMessage)
              Drift: \(String(format: "%.3f", driftPercent))%
              Status: \(correctionStatus)
              Reference: \(String(format: "%.3e", result.referenceQuantity))
              Current: \(String(format: "%.3e", result.currentQuantity))
            """

            return DiagnosticResult(
                name: result.lawName,
                severity: severity,
                message: message,
                value: result.relativeDrift,
                threshold: nil,
                time: result.time,
                step: result.step
            )
        }
    }

    /// Check if drift exceeds threshold
    ///
    /// Quick check for alerting without full diagnostic generation.
    ///
    /// - Parameters:
    ///   - results: Conservation results
    ///   - threshold: Relative drift threshold (default: 0.01 = 1%)
    /// - Returns: True if any law exceeds threshold
    public static func hasCriticalDrift(
        results: [ConservationResult],
        threshold: Float = 0.01
    ) -> Bool {
        return results.contains { $0.relativeDrift > threshold }
    }

    /// Get maximum drift across all laws
    ///
    /// - Parameter results: Conservation results
    /// - Returns: Maximum relative drift
    public static func maximumDrift(results: [ConservationResult]) -> Float {
        return results.map { $0.relativeDrift }.max() ?? 0.0
    }

    /// Format drift summary
    ///
    /// Concise summary of all conservation drifts for logging.
    ///
    /// - Parameter results: Conservation results
    /// - Returns: Summary string
    ///
    /// ## Example Output
    ///
    /// ```
    /// Conservation Drift: Particle=0.3%, Energy=0.8%
    /// ```
    public static func formatDriftSummary(results: [ConservationResult]) -> String {
        let driftStrings = results.map { result in
            let driftPercent = result.relativeDrift * 100
            let lawShort = result.lawName.replacingOccurrences(of: "Conservation", with: "")
            return "\(lawShort)=\(String(format: "%.1f", driftPercent))%"
        }

        return "Conservation Drift: " + driftStrings.joined(separator: ", ")
    }
}
