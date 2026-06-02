import Foundation

// MARK: - Validation Config Matcher

/// Utilities for matching swift-Gotenx configuration to reference data
///
/// Ensures consistent comparison by aligning:
/// - Mesh resolution
/// - Boundary conditions
/// - Time range and sampling
/// - Physical parameters
///
/// ## Usage
///
/// ```swift
/// // Load TORAX reference data
/// let toraxData = try TORAXReferenceData.load(
///     from: "reference_data/torax_iter_baseline.nc"
/// )
///
/// // Generate matching swift-Gotenx configuration
/// let config = try ValidationConfigMatcher.matchToTorax(toraxData)
///
/// // Run simulation with matched config
/// let orchestrator = try await SimulationOrchestrator(configuration: config)
/// try await orchestrator.run()
///
/// // Compare outputs
/// let gotenxOutput = await orchestrator.getOutputData()
/// let comparison = ValidationConfigMatcher.compareWithTorax(
///     gotenx: gotenxOutput,
///     torax: toraxData
/// )
/// ```
public struct ValidationConfigMatcher {
    // MARK: - Configuration Matching

    /// Generate swift-Gotenx configuration matching TORAX reference data
    ///
    /// Aligns the following parameters:
    /// - Mesh: cellCount = rho.count
    /// - Time: start/end/saveInterval matched to TORAX time array
    /// - Geometry: R₀, a, B₀ from ITER Baseline
    /// - Transport: Bohm-GyroBohm (same as TORAX)
    /// - Sources: Same models as TORAX
    ///
    /// - Parameter toraxData: TORAX reference data
    /// - Returns: Swift-Gotenx configuration
    /// - Throws: ValidationError if TORAX data is incompatible
    public static func matchToTorax(
        _ toraxData: TORAXReferenceData
    ) throws -> SimulationConfiguration {
        // Extract mesh size from TORAX data
        let cellCount = toraxData.normalizedRadius.count
        guard cellCount >= 10 && cellCount <= 200 else {
            throw ValidationConfigError.invalidMeshSize(cellCount)
        }

        // Extract time range from TORAX data
        guard let tStart = toraxData.time.first,
              let tEnd = toraxData.time.last else {
            throw ValidationConfigError.emptyTimeArray
        }

        let nTimePoints = toraxData.time.count
        let saveInterval = (tEnd - tStart) / Float(nTimePoints - 1)

        // ITER Baseline geometry (matches TORAX settings)
        let geometry = MeshConfig(
            cellCount: cellCount,              // Match TORAX mesh
            majorRadius: 6.2,            // [m] - ITER
            minorRadius: 2.0,            // [m] - ITER
            toroidalField: 5.3           // [T] - ITER
        )

        // Boundary conditions (typical ITER edge values)
        // Extract from TORAX data at edge (rho ≈ 1.0)
        // Find edge index dynamically (maximum rho value)
        let edgeIdx = toraxData.normalizedRadius.enumerated().max(by: { $0.element < $1.element })!.offset

        // Verify edge rho is close to 1.0
        let edgeRho = toraxData.normalizedRadius[edgeIdx]
        guard abs(edgeRho - 1.0) < 0.05 else {
            throw ValidationConfigError.incompatibleTimeRanges(
                "Edge rho = \(edgeRho), expected ~1.0. TORAX data may have unexpected rho range."
            )
        }

        let Ti_edge = toraxData.ionTemperature[0][edgeIdx]  // Initial edge temperature
        let Te_edge = toraxData.electronTemperature[0][edgeIdx]
        let ne_edge = toraxData.electronDensity[0][edgeIdx]

        let boundaries = BoundaryConfig(
            ionTemperature: Ti_edge,
            electronTemperature: Te_edge,
            electronDensity: ne_edge
        )

        // Transport model: Bohm-GyroBohm (same as TORAX)
        let transport = try TransportConfig(
            modelType: .bohmGyrobohm,
            parameters: [:]
        )

        // Source models: Same as TORAX
        let sources = SourcesConfig(
            ohmicHeating: true,
            fusionPower: true,
            ionElectronExchange: true,
            bremsstrahlung: true
        )

        // Build complete configuration
        let config = SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: geometry,
                    evolution: EvolutionConfig(
                        ionHeat: true,
                        electronHeat: true,
                        electronDensity: true,
                        poloidalFlux: false  // Match TORAX (no current evolution)
                    ),
                    solver: SolverConfig(
                        type: "newton_raphson",
                        tolerance: 1e-6,
                        maximumIterations: 30
                    ),
                    scheme: SchemeConfig(theta: 1.0)  // Implicit Euler (TORAX default)
                ),
                dynamic: DynamicConfig(
                    boundaries: boundaries,
                    transport: transport,
                    sources: sources,
                    pedestal: nil,
                    mhd: MHDConfig(
                        sawtoothEnabled: false,
                        sawtoothParameters: SawtoothParameters(),
                        ntmEnabled: false
                    ),
                    restart: RestartConfig(doRestart: false)
                )
            ),
            time: TimeConfiguration(
                start: tStart,
                end: tEnd,
                initialTimeStep: 1e-3,  // [s]
                adaptive: AdaptiveTimestepConfig(
                    minimumTimeStep: 1e-6,
                    maximumTimeStep: 1e-1,
                    safetyFactor: 0.9
                )
            ),
            output: OutputConfiguration(
                saveInterval: saveInterval,
                directory: "/tmp/gotenx_validation",
                format: .netcdf
            )
        )

        return config
    }

    /// Generate configuration matching ITER Baseline
    ///
    /// Uses design parameters from ITER Physics Basis.
    ///
    /// - Returns: Swift-Gotenx configuration for ITER Baseline
    public static func matchToITERBaseline() throws -> SimulationConfiguration {
        let baseline = ITERBaselineData.load()

        let config = SimulationConfiguration(
            runtime: RuntimeConfiguration(
                static: StaticConfig(
                    mesh: MeshConfig(
                        cellCount: 50,  // Standard resolution
                        majorRadius: baseline.geometry.majorRadius,
                        minorRadius: baseline.geometry.minorRadius,
                        toroidalField: baseline.geometry.toroidalField
                    ),
                    evolution: EvolutionConfig(
                        ionHeat: true,
                        electronHeat: true,
                        electronDensity: true,
                        poloidalFlux: false
                    ),
                    solver: SolverConfig(
                        type: "newton_raphson",
                        tolerance: 1e-6,
                        maximumIterations: 30
                    ),
                    scheme: SchemeConfig(theta: 1.0)
                ),
                dynamic: DynamicConfig(
                    boundaries: BoundaryConfig(
                        ionTemperature: 100.0,    // [eV]
                        electronTemperature: 100.0,
                        electronDensity: 2.0e19           // [m⁻³]
                    ),
                    transport: try TransportConfig(
                        modelType: .bohmGyrobohm,
                        parameters: [:]
                    ),
                    sources: SourcesConfig(
                        ohmicHeating: true,
                        fusionPower: true,
                        ionElectronExchange: true,
                        bremsstrahlung: true
                    ),
                    pedestal: nil,
                    mhd: MHDConfig(
                        sawtoothEnabled: false,
                        sawtoothParameters: SawtoothParameters(),
                        ntmEnabled: false
                    ),
                    restart: RestartConfig(doRestart: false)
                )
            ),
            time: TimeConfiguration(
                start: 0.0,
                end: 2.0,
                initialTimeStep: 1e-3,
                adaptive: AdaptiveTimestepConfig(
                    minimumTimeStep: 1e-6,
                    maximumTimeStep: 1e-1,
                    safetyFactor: 0.9
                )
            ),
            output: OutputConfiguration(
                saveInterval: 0.04,  // 50 points in 2s
                directory: "/tmp/gotenx_iter_baseline",
                format: .netcdf
            )
        )

        return config
    }

    // MARK: - Comparison

    /// Compare swift-Gotenx output with TORAX reference data
    ///
    /// Returns comparison results for all time points and quantities.
    ///
    /// - Parameters:
    ///   - gotenx: Swift-Gotenx output (time-series)
    ///   - torax: TORAX reference data
    ///   - thresholds: Validation thresholds
    /// - Returns: Array of comparison results
    ///
    /// ## Example
    ///
    /// ```swift
    /// let results = ValidationConfigMatcher.compareWithTorax(
    ///     gotenx: gotenxOutput,
    ///     torax: toraxData
    /// )
    ///
    /// let passedAll = results.allSatisfy { $0.passed }
    /// if passedAll {
    ///     print("✅ Validation passed for all quantities")
    /// }
    /// ```
    public static func compareWithTorax(
        gotenx: TORAXReferenceData,
        torax: TORAXReferenceData,
        thresholds: ValidationThresholds = .torax
    ) -> [ComparisonResult] {
        var results: [ComparisonResult] = []

        // Compare at each time point
        for i in 0..<min(gotenx.time.count, torax.time.count) {
            let time = torax.time[i]

            // Compare Ti
            let tiResult = ProfileComparator.compare(
                quantity: "ion_temperature",
                predicted: gotenx.ionTemperature[i],
                reference: torax.ionTemperature[i],
                time: time,
                thresholds: thresholds
            )
            results.append(tiResult)

            // Compare Te
            let teResult = ProfileComparator.compare(
                quantity: "electron_temperature",
                predicted: gotenx.electronTemperature[i],
                reference: torax.electronTemperature[i],
                time: time,
                thresholds: thresholds
            )
            results.append(teResult)

            // Compare ne
            let neResult = ProfileComparator.compare(
                quantity: "electron_density",
                predicted: gotenx.electronDensity[i],
                reference: torax.electronDensity[i],
                time: time,
                thresholds: thresholds
            )
            results.append(neResult)
        }

        return results
    }

    /// Compare swift-Gotenx output with TORAX reference data and return an aggregate matrix.
    ///
    /// This is the preferred validation entry point for quality gates because it
    /// validates compatible time/radius grids before computing metrics and keeps a
    /// failure summary for review logs.
    public static func compareWithToraxMatrix(
        gotenx: TORAXReferenceData,
        torax: TORAXReferenceData,
        thresholds: ValidationThresholds = .torax
    ) throws -> ProfileValidationMatrix {
        try ProfileValidationMatrix.compareTimeSeries(
            predicted: gotenx,
            reference: torax,
            sourceName: "torax",
            thresholds: thresholds
        )
    }
}

// MARK: - Validation Config Errors

/// Errors that can occur during validation configuration matching
public enum ValidationConfigError: Error, CustomStringConvertible {
    case invalidMeshSize(Int)
    case emptyTimeArray
    case incompatibleTimeRanges(String)

    public var description: String {
        switch self {
        case .invalidMeshSize(let n):
            return "Invalid mesh size: \(n) (must be 10-200)"
        case .emptyTimeArray:
            return "Time array is empty"
        case .incompatibleTimeRanges(let message):
            return "Incompatible time ranges: \(message)"
        }
    }
}
