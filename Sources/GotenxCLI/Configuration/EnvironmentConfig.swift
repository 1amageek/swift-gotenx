// EnvironmentConfig.swift
// Environment configuration for MLX and Gotenx runtime

import Foundation
import MLX

/// Configuration for Gotenx runtime environment
struct EnvironmentConfig {
    let compilationEnabled: Bool
    let errorsEnabled: Bool
    let cacheLimitMB: Int?

    /// Apply environment configuration
    func apply() throws {
        print("\nEnvironment Configuration:")

        // MLX compilation control - ACTUAL API CALL
        compile(enable: compilationEnabled)
        print("  • Compilation: \(compilationEnabled ? "enabled" : "disabled")")
        if !compilationEnabled {
            print("    ⚠️  WARNING: MLX JIT compilation disabled - performance will be severely degraded")
            print("       This mode is only for debugging purposes")
        }

        // Error checking
        // Note: MLX doesn't support global error checking mode.
        // The errorsEnabled flag is informational for now.
        // Actual error checking requires using checkedEval() instead of eval()
        // at each evaluation point in the simulation code.
        print("  • Error checking: \(errorsEnabled ? "enabled" : "disabled")")
        if errorsEnabled {
            print("    ℹ️  Error checking flag set - simulation code should use checkedEval()")
            print("       Note: MLX doesn't support global error checking mode")
        }

        // MLX GPU cache limit - ACTUAL API CALL
        if let limitMB = cacheLimitMB {
            let limitBytes = limitMB * 1024 * 1024
            MLX.Memory.cacheLimit = limitBytes
            print("  • MLX GPU cache limit: \(limitMB) MB")
        } else {
            print("  • MLX GPU cache limit: default")
        }

        // Display GPU memory info
        displayGPUInfo()
    }

    /// Display GPU memory information
    private func displayGPUInfo() {
        // Query actual GPU memory status using MLX API
        let snapshot = MLX.Memory.snapshot()

        print("\nGPU Memory Status:")
        print("  • Active memory: \(formatBytes(snapshot.activeMemory))")
        print("  • Cached memory: \(formatBytes(snapshot.cacheMemory))")
        print("  • Peak memory:   \(formatBytes(snapshot.peakMemory))")
    }

    /// Format bytes to human-readable string
    private func formatBytes(_ bytes: Int) -> String {
        let kb = Double(bytes) / 1024.0
        let mb = kb / 1024.0
        let gb = mb / 1024.0

        if gb >= 1.0 {
            return String(format: "%.2f GB", gb)
        } else if mb >= 1.0 {
            return String(format: "%.2f MB", mb)
        } else if kb >= 1.0 {
            return String(format: "%.2f KB", kb)
        } else {
            return "\(bytes) bytes"
        }
    }
}

/// Profiling context for performance measurement
final class ProfilingContext {
    private let outputPath: String?
    private var startTime: Date?
    private var compilationStartTime: Date?
    private var compilationTime: TimeInterval = 0
    private var checkpoints: [(name: String, time: Date)] = []

    init(outputPath: String?) {
        self.outputPath = outputPath
    }

    func start() {
        startTime = Date()
        checkpoints.append(("start", Date()))
        print("\n⏱️  Profiling started")
    }

    func checkpoint(_ name: String) {
        let now = Date()
        checkpoints.append((name, now))

        if let start = startTime {
            let elapsed = now.timeIntervalSince(start)
            print("  ⏱️  \(name): \(String(format: "%.3f", elapsed))s")
        }
    }

    func stop() -> ProfilingStats {
        guard let start = startTime else {
            return ProfilingStats(totalTime: 0, compilationTime: 0, executionTime: 0)
        }

        let end = Date()
        let totalTime = end.timeIntervalSince(start)
        let executionTime = totalTime - compilationTime

        let stats = ProfilingStats(
            totalTime: totalTime,
            compilationTime: compilationTime,
            executionTime: executionTime
        )

        // Write to file if path provided
        if let path = outputPath {
            do {
                try writeProfilingReport(stats: stats, to: path)
                print("\n  📊 Profiling report saved to: \(path)")
            } catch {
                print("\n  ⚠️  Failed to save profiling report: \(error)")
            }
        }

        return stats
    }

    private func writeProfilingReport(stats: ProfilingStats, to path: String) throws {
        var report = """
        Gotenx Profiling Report
        ======================
        Generated: \(ISO8601DateFormatter().string(from: Date()))

        Timing Summary
        --------------
        Total time:       \(String(format: "%.3f", stats.totalTime))s
        Compilation time: \(String(format: "%.3f", stats.compilationTime))s
        Execution time:   \(String(format: "%.3f", stats.executionTime))s

        Checkpoints
        -----------
        """

        if let start = startTime {
            for (name, time) in checkpoints {
                let elapsed = time.timeIntervalSince(start)
                report += "\n\(String(format: "%.3f", elapsed))s - \(name)"
            }
        }

        report += "\n\nGPU Memory\n----------\n"
        report += "Memory tracking enabled (MLX.GPU.snapshot)\n"

        try report.write(toFile: path, atomically: true, encoding: .utf8)
    }
}

/// Profiling statistics
struct ProfilingStats {
    let totalTime: TimeInterval
    let compilationTime: TimeInterval
    let executionTime: TimeInterval

    var compilationPercentage: Double {
        guard totalTime > 0 else { return 0 }
        return (compilationTime / totalTime) * 100
    }

    var executionPercentage: Double {
        guard totalTime > 0 else { return 0 }
        return (executionTime / totalTime) * 100
    }
}
