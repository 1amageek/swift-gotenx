import Foundation

// MARK: - Diagnostics Report

/// Comprehensive diagnostics report for post-simulation analysis
///
/// Aggregates all diagnostic results and conservation enforcement results
/// into a single report for analysis and debugging.
///
/// ## Example Usage
///
/// ```swift
/// // At end of simulation
/// let report = DiagnosticsReport(
///     results: diagnosticResults,
///     conservationResults: conservationResults,
///     startTime: 0.0,
///     endTime: 2.0,
///     totalSteps: 20000
/// )
///
/// // Print summary
/// print(report.summary())
///
/// // Export to JSON
/// try report.exportJSON(to: "diagnostics_report.json")
/// ```
public struct DiagnosticsReport: Sendable, Codable {
    /// All diagnostic results
    public let results: [DiagnosticResult]

    /// Conservation enforcement results
    public let conservationResults: [ConservationResult]

    /// Simulation start time [s]
    public let startTime: Float

    /// Simulation end time [s]
    public let endTime: Float

    /// Total number of timesteps
    public let totalSteps: Int

    public init(
        results: [DiagnosticResult],
        conservationResults: [ConservationResult],
        startTime: Float,
        endTime: Float,
        totalSteps: Int
    ) {
        self.results = results
        self.conservationResults = conservationResults
        self.startTime = startTime
        self.endTime = endTime
        self.totalSteps = totalSteps
    }

    // MARK: - Summary

    /// Generate human-readable summary
    ///
    /// - Returns: Formatted summary string
    ///
    /// ## Example Output
    ///
    /// ```
    /// ═══════════════════════════════════════
    /// SIMULATION DIAGNOSTICS REPORT
    /// ═══════════════════════════════════════
    /// Time range: [0.0, 2.0] s
    /// Total steps: 20000
    ///
    /// Summary:
    ///   ❌ Errors: 2
    ///   ⚠️  Warnings: 5
    ///   ℹ️  Info: 18
    ///
    /// Conservation Enforcement:
    ///   • ParticleConservation: drift = 0.3% (corrected)
    ///   • EnergyConservation: drift = 0.8% (corrected)
    ///
    /// Critical Issues:
    ///   ❌ [JacobianConditioning] Ill-conditioned: κ = 2.3e6
    ///   ❌ [TransportDiagnostics] Negative diffusivity
    /// ═══════════════════════════════════════
    /// ```
    public func summary() -> String {
        var output = """
        ═══════════════════════════════════════
        SIMULATION DIAGNOSTICS REPORT
        ═══════════════════════════════════════
        Time range: [\(String(format: "%.1f", startTime)), \(String(format: "%.1f", endTime))] s
        Total steps: \(totalSteps)

        """

        // Count by severity
        let infoCount = results.filter { $0.severity == .info }.count
        let warningCount = results.filter { $0.severity == .warning }.count
        let errorCount = results.filter { $0.severity == .error }.count
        let criticalCount = results.filter { $0.severity == .critical }.count

        output += """
        Summary:
          ℹ️  Info: \(infoCount)
          ⚠️  Warnings: \(warningCount)
          ❌ Errors: \(errorCount)
          🔥 Critical: \(criticalCount)

        """

        // Conservation summary
        if !conservationResults.isEmpty {
            output += "Conservation Enforcement:\n"
            for result in conservationResults {
                let driftPercent = result.relativeDrift * 100
                let status = result.corrected ? "corrected" : "monitored"
                output += "  • \(result.lawName): drift = \(String(format: "%.1f", driftPercent))% (\(status))\n"
            }
            output += "\n"
        }

        // Critical issues
        let criticalResults = results.filter { $0.severity == .error || $0.severity == .critical }
        if !criticalResults.isEmpty {
            output += "Critical Issues:\n"
            for result in criticalResults.prefix(10) {  // Limit to first 10
                output += "  \(result.severity.rawValue) [\(result.name)] \(result.message)\n"
            }
            if criticalResults.count > 10 {
                output += "  ... and \(criticalResults.count - 10) more\n"
            }
            output += "\n"
        }

        output += "═══════════════════════════════════════"

        return output
    }

    // MARK: - JSON Export

    /// Export report to JSON
    ///
    /// - Returns: JSON data
    /// - Throws: Encoding errors
    ///
    /// ## JSON Schema
    ///
    /// ```json
    /// {
    ///   "results": [
    ///     {
    ///       "name": "JacobianConditioning",
    ///       "severity": "warning",
    ///       "message": "...",
    ///       "value": 520000.0,
    ///       "threshold": 1000000.0,
    ///       "time": 1.234,
    ///       "step": 12340
    ///     }
    ///   ],
    ///   "conservationResults": [...],
    ///   "startTime": 0.0,
    ///   "endTime": 2.0,
    ///   "totalSteps": 20000
    /// }
    /// ```
    public func exportJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    /// Export to JSON file
    ///
    /// - Parameter path: File path
    /// - Throws: Encoding or file writing errors
    public func exportJSON(to path: String) throws {
        let data = try exportJSON()
        let url = URL(fileURLWithPath: path)
        try data.write(to: url)
    }

    // MARK: - Filtering

    /// Filter results by severity
    ///
    /// - Parameter severity: Severity level to filter
    /// - Returns: Filtered results
    public func filterBySeverity(_ severity: DiagnosticResult.Severity) -> [DiagnosticResult] {
        return results.filter { $0.severity == severity }
    }

    /// Filter results by name
    ///
    /// - Parameter name: Diagnostic name
    /// - Returns: Filtered results
    public func filterByName(_ name: String) -> [DiagnosticResult] {
        return results.filter { $0.name == name }
    }

    /// Get conservation results by law name
    ///
    /// - Parameter lawName: Conservation law name
    /// - Returns: Filtered conservation results
    public func conservationResultsFor(lawName: String) -> [ConservationResult] {
        return conservationResults.filter { $0.lawName == lawName }
    }

    // MARK: - Statistics

    /// Check if simulation has critical issues
    ///
    /// - Returns: True if any critical or error results exist
    public func hasCriticalIssues() -> Bool {
        return results.contains { $0.severity == .critical || $0.severity == .error }
    }

    /// Get maximum conservation drift
    ///
    /// - Returns: Maximum relative drift across all laws
    public func maximumConservationDrift() -> Float {
        return conservationResults.map { $0.relativeDrift }.max() ?? 0.0
    }

    /// Generate statistics dictionary
    ///
    /// - Returns: Statistics for analysis
    public func statistics() -> [String: Any] {
        return [
            "totalDiagnostics": results.count,
            "infoCount": results.filter { $0.severity == .info }.count,
            "warningCount": results.filter { $0.severity == .warning }.count,
            "errorCount": results.filter { $0.severity == .error }.count,
            "criticalCount": results.filter { $0.severity == .critical }.count,
            "conservationLaws": conservationResults.map { $0.lawName }.uniqued(),
            "maximumConservationDrift": maximumConservationDrift(),
            "hasCriticalIssues": hasCriticalIssues()
        ]
    }
}

// MARK: - Array Extension

extension Array where Element: Hashable {
    /// Return unique elements preserving order
    fileprivate func uniqued() -> [Element] {
        var seen = Set<Element>()
        return filter { seen.insert($0).inserted }
    }
}
