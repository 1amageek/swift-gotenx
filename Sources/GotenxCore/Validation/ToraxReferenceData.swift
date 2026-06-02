import Foundation

// MARK: - TORAX Reference Data

/// TORAX Python implementation reference data
///
/// Loads time-series data from TORAX NetCDF outputs for detailed validation.
///
/// ## Purpose
///
/// Primary validation data source for swift-Gotenx:
/// - Same physics models (Bohm-GyroBohm, Bremsstrahlung, etc.)
/// - Detailed time-evolution profiles
/// - NetCDF-4 format (Phase 5 compatible)
///
/// ## Usage
///
/// ```swift
/// // Load TORAX reference data
/// let toraxData = try TORAXReferenceData.load(
///     from: "reference_data/torax_iter_baseline.nc"
/// )
///
/// // Compare at specific time
/// let timeIndex = toraxData.findTimeIndex(closestTo: 2.0)
/// let ionTemperatureReference = toraxData.ionTemperature[timeIndex]
/// let ionTemperatureGotenx = // ... from simulation
///
/// let result = ProfileComparator.compare(
///     quantity: "ion_temperature",
///     predicted: ionTemperatureGotenx,
///     reference: ionTemperatureReference,
///     time: toraxData.time[timeIndex],
///     thresholds: .torax
/// )
/// ```
///
/// ## Data Structure
///
/// TORAX NetCDF output structure:
/// ```
/// dimensions:
///   time = 100
///   rho_tor_norm = 100
/// variables:
///   float time(time)                           // Time points [s]
///   float rho_tor_norm(rho_tor_norm)          // Normalized flux
///   float ion_temperature(time, rho_tor_norm) // [eV]
///   float electron_temperature(time, rho_tor_norm)
///   float electron_density(time, rho_tor_norm)
///   float poloidal_flux(time, rho_tor_norm)
/// ```
public struct TORAXReferenceData: Sendable {
    /// Time points [s]
    public let time: [Float]

    /// Normalized toroidal flux coordinate [dimensionless]
    public let normalizedRadius: [Float]

    /// Ion temperature [eV] - shape: [timeCount, radiusCount]
    public let ionTemperature: [[Float]]

    /// Electron temperature [eV] - shape: [timeCount, radiusCount]
    public let electronTemperature: [[Float]]

    /// Electron density [m⁻³] - shape: [timeCount, radiusCount]
    public let electronDensity: [[Float]]

    /// Poloidal flux [Wb] - shape: [timeCount, radiusCount] (optional)
    public let poloidalFlux: [[Float]]?

    public init(
        time: [Float],
        normalizedRadius: [Float],
        ionTemperature: [[Float]],
        electronTemperature: [[Float]],
        electronDensity: [[Float]],
        poloidalFlux: [[Float]]? = nil
    ) {
        self.time = time
        self.normalizedRadius = normalizedRadius
        self.ionTemperature = ionTemperature
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity
        self.poloidalFlux = poloidalFlux

        // Validate shapes
        precondition(ionTemperature.count == time.count, "ionTemperature time dimension mismatch")
        precondition(electronTemperature.count == time.count, "electronTemperature time dimension mismatch")
        precondition(electronDensity.count == time.count, "electronDensity time dimension mismatch")
        if let poloidalFlux = poloidalFlux {
            precondition(poloidalFlux.count == time.count, "poloidalFlux time dimension mismatch")
        }

        for i in 0..<time.count {
            precondition(ionTemperature[i].count == normalizedRadius.count, "ionTemperature spatial dimension mismatch at time \(i)")
            precondition(electronTemperature[i].count == normalizedRadius.count, "electronTemperature spatial dimension mismatch at time \(i)")
            precondition(electronDensity[i].count == normalizedRadius.count, "electronDensity spatial dimension mismatch at time \(i)")
        }
    }

    // MARK: - Loading (Placeholder)

    /// Load TORAX reference data from NetCDF file
    ///
    /// **Note**: This is a placeholder implementation.
    /// Full NetCDF reading requires Phase 5 IMAS I/O implementation.
    ///
    /// For now, this function will:
    /// 1. Check if file exists
    /// 2. Return error if NetCDF reader not available
    ///
    /// - Parameter path: Path to TORAX NetCDF output file
    /// - Returns: Loaded reference data
    /// - Throws: TORAXDataError if file not found or reader unavailable
    public static func load(from path: String) throws -> TORAXReferenceData {
        // Check if file exists
        guard FileManager.default.fileExists(atPath: path) else {
            throw TORAXDataError.fileNotFound(path)
        }

        // TODO: Implement NetCDF reading in Phase 5
        // For now, return a placeholder error
        throw TORAXDataError.netCDFReaderUnavailable(
            "NetCDF reader will be implemented in Phase 5 (IMAS I/O)"
        )

        // Future implementation will use IMASReader:
        // let reader = try IMASReader(filePath: path)
        // let time = try reader.readVariable(name: "time", type: Float.self)
        // let rho = try reader.readVariable(name: "rho_tor_norm", type: Float.self)
        // let Ti = try reader.read2DVariable(name: "ion_temperature")
        // ...
    }

    // MARK: - Time Utilities

    /// Find time index closest to target time
    ///
    /// - Parameter targetTime: Target time [s]
    /// - Returns: Index of closest time point
    ///
    /// ## Example
    ///
    /// ```swift
    /// let idx = toraxData.findTimeIndex(closestTo: 2.0)
    /// let ionTemperatureAtTwoSeconds = toraxData.ionTemperature[idx]
    /// ```
    public func findTimeIndex(closestTo targetTime: Float) -> Int {
        var minDiff: Float = Float.infinity
        var closestIndex = 0

        for (i, t) in time.enumerated() {
            let diff = abs(t - targetTime)
            if diff < minDiff {
                minDiff = diff
                closestIndex = i
            }
        }

        return closestIndex
    }

    /// Get profiles at specific time index
    ///
    /// - Parameter timeIndex: Index in time array
    /// - Returns: Reference profiles at that time
    public func profiles(at timeIndex: Int) -> ReferenceProfiles {
        precondition(timeIndex >= 0 && timeIndex < time.count, "Time index out of bounds")

        return ReferenceProfiles(
            normalizedRadius: normalizedRadius,
            ionTemperature: ionTemperature[timeIndex],
            electronTemperature: electronTemperature[timeIndex],
            electronDensity: electronDensity[timeIndex],
            time: time[timeIndex]
        )
    }

    /// Get profiles closest to target time
    ///
    /// - Parameter targetTime: Target time [s]
    /// - Returns: Reference profiles at closest time point
    public func profiles(closestTo targetTime: Float) -> ReferenceProfiles {
        let idx = findTimeIndex(closestTo: targetTime)
        return profiles(at: idx)
    }
}

// MARK: - Errors

/// Errors that can occur when loading TORAX reference data
public enum TORAXDataError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case fileOpenFailed(String)
    case netCDFReaderUnavailable(String)
    case invalidDataShape(String)
    case missingVariable(String)
    case variableNotFound(String)
    case invalidDimensions(String)
    case invalidData(String)

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return "TORAX reference data file not found: \(path)"
        case .fileOpenFailed(let path):
            return "Failed to open NetCDF file: \(path)"
        case .netCDFReaderUnavailable(let message):
            return "NetCDF reader not available: \(message)"
        case .invalidDataShape(let message):
            return "Invalid data shape: \(message)"
        case .missingVariable(let name):
            return "Required variable '\(name)' not found in NetCDF file"
        case .variableNotFound(let name):
            return "Variable '\(name)' not found in NetCDF file"
        case .invalidDimensions(let message):
            return "Invalid dimensions: \(message)"
        case .invalidData(let message):
            return "Invalid data: \(message)"
        }
    }
}
