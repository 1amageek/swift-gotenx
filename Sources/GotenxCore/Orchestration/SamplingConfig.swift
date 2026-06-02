// SamplingConfig.swift
// Configuration for 3-tier data sampling strategy
//
// Controls memory/performance tradeoff by selectively capturing simulation data:
// - Tier 1: 0D scalars (always, ~8 MB for 20k steps)
// - Tier 2: 1D profiles (adaptive, ~2.4 MB for 200 snapshots)
// - Tier 3: 2D/3D data (on-demand, 0 MB stored)
//
// Design based on VISUALIZATION_DESIGN.md Section 3.5

import Foundation

/// Configuration for simulation data sampling and storage
///
/// **Memory Budget Analysis** (100 cells, 20,000 steps, 2s simulation):
/// - Full capture: 2.4 GB (infeasible)
/// - Tier 1 only: 2.7 MB (minimal, scalars only)
/// - Tier 1+2 (100 steps): 314 KB profiles + 2.7 MB scalars = 3.0 MB (recommended)
/// - Tier 1+2 (10 steps): 3.1 MB profiles + 2.7 MB scalars = 5.8 MB (high detail)
///
/// **Usage**:
/// ```swift
/// // Recommended: Balanced memory/detail (3.0 MB for 20k steps)
/// let config = SamplingConfig(
///     profileSamplingInterval: 100,  // Every 100 steps (201 snapshots)
///     enableDerivedQuantities: true,
///     enableDiagnostics: true
/// )
///
/// // Memory-constrained: Scalars only (2.7 MB for 20k steps)
/// let minimal = SamplingConfig(
///     profileSamplingInterval: nil,  // No profile snapshots
///     enableDerivedQuantities: true,
///     enableDiagnostics: false
/// )
///
/// // Maximum detail: High-frequency sampling (5.8 MB for 20k steps)
/// let detailed = SamplingConfig(
///     profileSamplingInterval: 10,   // Every 10 steps (2000 snapshots)
///     enableDerivedQuantities: true,
///     enableDiagnostics: true
/// )
/// ```
public struct SamplingConfig: Sendable, Codable, Equatable {
    // MARK: - Tier 1: Scalar Time Series (Always Captured)

    /// Enable derived quantities (τE, Q, βN, etc.)
    ///
    /// **Cost**: 24 scalars × 4 bytes × stepCount = ~1.9 MB for 20k steps
    ///
    /// **Phase 1**: Always false (DerivedQuantities returns .zero)
    /// **Phase 2+**: Recommended true
    public let enableDerivedQuantities: Bool

    /// Enable numerical diagnostics (residuals, conservation, iterations)
    ///
    /// **Cost**: 11 scalars × 4 bytes × stepCount = ~0.9 MB for 20k steps
    ///
    /// **Phase 1**: Always false (NumericalDiagnostics returns .default)
    /// **Phase 2+**: Recommended true for debugging
    public let enableDiagnostics: Bool

    // MARK: - Tier 2: Profile Snapshots (Adaptive Sampling)

    /// Sampling interval for profile snapshots (nil = disabled)
    ///
    /// **Cost per snapshot**: 4 profiles × 100 cells × 4 bytes = 1.6 KB
    /// **Total cost**: (stepCount / interval) × 1.6 KB
    ///
    /// **Examples**:
    /// - `nil`: No profile snapshots (0 MB)
    /// - `100`: Every 100 steps → 200 snapshots → 320 KB
    /// - `10`: Every 10 steps → 2000 snapshots → 3.2 MB
    ///
    /// **Recommended**: 50-100 for 20k-step simulations
    public let profileSamplingInterval: Int?

    /// Enable transport coefficient capture in profile snapshots
    ///
    /// **Cost per snapshot**: 4 coeffs × 100 cells × 4 bytes = 1.6 KB
    ///
    /// **Phase 1**: Always false (not implemented)
    /// **Phase 3+**: Enable for transport analysis
    public let enableTransportCapture: Bool

    /// Enable source term capture in profile snapshots
    ///
    /// **Cost per snapshot**: 4 sources × 100 cells × 4 bytes = 1.6 KB
    ///
    /// **Phase 1**: Always false (not implemented)
    /// **Phase 3+**: Enable for power balance analysis
    public let enableSourceCapture: Bool

    /// Enable live plotting (includes profiles in ProgressInfo)
    ///
    /// **Cost**: Profile serialization ~100μs @ 100 cells, throttled by SimulationRunner (100ms)
    ///
    /// **App Integration**: When enabled, `ProgressInfo` includes current profiles and derived quantities
    /// for real-time visualization. Disable for batch simulations to minimize overhead.
    ///
    /// **Performance**: Negligible impact (~0.1% overhead) due to throttling
    public let enableLivePlotting: Bool

    // MARK: - Tier 3: 2D/3D Data (On-Demand Reconstruction)

    /// Enable 2D poloidal cross-section reconstruction
    ///
    /// **Cost**: 0 MB (reconstructed from 1D profiles on-demand)
    ///
    /// **Phase 1**: Always false (not implemented)
    /// **Phase 2+**: Enable for visualization
    public let enable2DReconstruction: Bool

    /// Enable 3D volumetric reconstruction
    ///
    /// **Cost**: 0 MB (reconstructed from 1D profiles on-demand)
    ///
    /// **Phase 1**: Always false (not implemented)
    /// **Phase 3+**: Enable for advanced visualization
    public let enable3DReconstruction: Bool

    // MARK: - Initialization

    public init(
        profileSamplingInterval: Int? = 100,
        enableDerivedQuantities: Bool = true,
        enableDiagnostics: Bool = true,
        enableTransportCapture: Bool = false,
        enableSourceCapture: Bool = false,
        enableLivePlotting: Bool = false,
        enable2DReconstruction: Bool = false,
        enable3DReconstruction: Bool = false
    ) {
        self.profileSamplingInterval = profileSamplingInterval
        self.enableDerivedQuantities = enableDerivedQuantities
        self.enableDiagnostics = enableDiagnostics
        self.enableTransportCapture = enableTransportCapture
        self.enableSourceCapture = enableSourceCapture
        self.enableLivePlotting = enableLivePlotting
        self.enable2DReconstruction = enable2DReconstruction
        self.enable3DReconstruction = enable3DReconstruction
    }
}

// MARK: - Presets

extension SamplingConfig {
    /// Minimal memory configuration: Scalars only, no profile snapshots
    ///
    /// **Memory**: ~2.7 MB for 20k steps (scalars only)
    ///
    /// **Use case**: Long-duration runs, memory-constrained devices
    public static let minimal = SamplingConfig(
        profileSamplingInterval: nil,
        enableDerivedQuantities: true,
        enableDiagnostics: false,
        enableTransportCapture: false,
        enableSourceCapture: false,
        enableLivePlotting: false,
        enable2DReconstruction: false,
        enable3DReconstruction: false
    )

    /// Balanced configuration: Scalars + moderate profile sampling
    ///
    /// **Memory**: ~3.0 MB for 20k steps (201 profile snapshots)
    ///
    /// **Use case**: Standard production runs (recommended default)
    public static let balanced = SamplingConfig(
        profileSamplingInterval: 100,
        enableDerivedQuantities: true,
        enableDiagnostics: true,
        enableTransportCapture: false,
        enableSourceCapture: false,
        enableLivePlotting: false,
        enable2DReconstruction: false,
        enable3DReconstruction: false
    )

    /// Detailed configuration: High-frequency profile sampling
    ///
    /// **Memory**: ~5.8 MB for 20k steps (2001 profile snapshots)
    ///
    /// **Use case**: Debugging, detailed physics analysis
    public static let detailed = SamplingConfig(
        profileSamplingInterval: 10,
        enableDerivedQuantities: true,
        enableDiagnostics: true,
        enableTransportCapture: false,
        enableSourceCapture: false,
        enableLivePlotting: false,
        enable2DReconstruction: false,
        enable3DReconstruction: false
    )

    /// Full physics configuration: All data capture enabled
    ///
    /// **Memory**: ~7 MB for 20k steps (Phase 3+)
    ///
    /// **Use case**: Research, transport analysis, power balance studies
    public static let fullPhysics = SamplingConfig(
        profileSamplingInterval: 100,
        enableDerivedQuantities: true,
        enableDiagnostics: true,
        enableTransportCapture: true,
        enableSourceCapture: true,
        enableLivePlotting: false,
        enable2DReconstruction: true,
        enable3DReconstruction: false
    )

    /// Real-time plotting configuration: High-frequency sampling with live updates
    ///
    /// **Memory**: ~5.8 MB for 20k steps (2001 profile snapshots)
    ///
    /// **Use case**: Interactive app with live visualization (Gotenx app)
    public static let realTimePlotting = SamplingConfig(
        profileSamplingInterval: 10,
        enableDerivedQuantities: true,
        enableDiagnostics: false,
        enableTransportCapture: false,
        enableSourceCapture: false,
        enableLivePlotting: true,  // ← Enable for live UI updates
        enable2DReconstruction: false,
        enable3DReconstruction: false
    )
}

// MARK: - Sampling Logic

extension SamplingConfig {
    /// Check if profile snapshot should be captured at given step
    ///
    /// **Logic**:
    /// - Always capture step 0 (initial condition)
    /// - Capture every `profileSamplingInterval` steps
    /// - Always capture final step (handled by caller)
    ///
    /// - Parameter step: Current simulation step
    /// - Returns: true if profile snapshot should be captured
    public func shouldCaptureProfile(at step: Int) -> Bool {
        guard let interval = profileSamplingInterval else {
            // Profile sampling disabled
            return false
        }

        // Always capture initial condition
        if step == 0 {
            return true
        }

        // Capture at intervals
        return step % interval == 0
    }

    /// Estimate total memory usage for given simulation
    ///
    /// - Parameters:
    ///   - stepCount: Total number of timesteps
    ///   - cellCount: Number of radial cells
    /// - Returns: Estimated memory usage in bytes
    public func estimateMemoryUsage(stepCount: Int, cellCount: Int) -> Int {
        var totalBytes = 0

        // Tier 1: Scalar time series
        if enableDerivedQuantities {
            totalBytes += 24 * 4 * stepCount  // 24 scalars in DerivedQuantities
        }

        if enableDiagnostics {
            totalBytes += 11 * 4 * stepCount  // 11 scalars in NumericalDiagnostics
        }

        // Tier 2: Profile snapshots
        if let interval = profileSamplingInterval {
            let nSnapshots = (stepCount / interval) + 1  // +1 for initial condition

            // Core profiles (4 arrays)
            totalBytes += 4 * cellCount * 4 * nSnapshots

            // Transport coefficients (4 arrays)
            if enableTransportCapture {
                totalBytes += 4 * cellCount * 4 * nSnapshots
            }

            // Source terms (4 arrays)
            if enableSourceCapture {
                totalBytes += 4 * cellCount * 4 * nSnapshots
            }
        }

        // Tier 3: No storage (on-demand reconstruction)

        return totalBytes
    }

    /// Human-readable memory estimate
    ///
    /// - Parameters:
    ///   - stepCount: Total number of timesteps
    ///   - cellCount: Number of radial cells
    /// - Returns: Memory estimate as string (e.g., "10.5 MB")
    public func memoryEstimateString(stepCount: Int, cellCount: Int) -> String {
        let bytes = estimateMemoryUsage(stepCount: stepCount, cellCount: cellCount)
        let mb = Double(bytes) / (1024 * 1024)

        if mb < 1.0 {
            let kb = Double(bytes) / 1024
            return String(format: "%.1f KB", kb)
        } else {
            return String(format: "%.1f MB", mb)
        }
    }
}
