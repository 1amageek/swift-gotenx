// LoggingBootstrap.swift
// One-time initialization of the swift-log backend for the Gotenx CLI.

import Foundation
import Logging

/// Configures the process-wide `LoggingSystem` exactly once.
///
/// Without an explicit bootstrap, swift-log installs a default handler whose
/// level is fixed at `.info` and cannot be changed, which silently hides every
/// `.debug`/`.trace` diagnostic the simulation emits. This bootstrap routes all
/// loggers through a `StreamLogHandler` whose level is controlled by the
/// `GOTENX_LOG_LEVEL` environment variable (trace, debug, info, notice, warning,
/// error, critical). The default is `.info`.
enum GotenxLogging {
    private static let bootstrapOnce: Void = {
        let level: Logger.Level
        if let raw = ProcessInfo.processInfo.environment["GOTENX_LOG_LEVEL"]?.lowercased(),
           let parsed = Logger.Level(rawValue: raw) {
            level = parsed
        } else {
            level = .info
        }

        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardError(label: label)
            handler.logLevel = level
            return handler
        }
    }()

    /// Bootstraps the logging backend. Safe to call multiple times; only the
    /// first call has any effect (`LoggingSystem.bootstrap` may run only once).
    static func bootstrap() {
        _ = bootstrapOnce
    }
}
