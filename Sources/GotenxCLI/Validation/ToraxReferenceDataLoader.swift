// TORAXReferenceDataLoader.swift
// NetCDF I/O for TORAX reference data

import Foundation
import SwiftNetCDF
import GotenxCore

// MARK: - NetCDF Loading Extension

extension TORAXReferenceData {

    /// Load TORAX reference data from NetCDF file
    ///
    /// - Parameter path: Path to TORAX NetCDF output file
    /// - Returns: TORAXReferenceData with all profiles
    /// - Throws: TORAXDataError if file cannot be read
    ///
    /// ## Expected NetCDF structure
    ///
    /// ```
    /// dimensions:
    ///   time = 100
    ///   rho_tor_norm = 100
    /// variables:
    ///   float time(time)
    ///     long_name = "simulation time"
    ///     units = "s"
    ///   float rho_tor_norm(rho_tor_norm)
    ///     long_name = "normalized toroidal flux coordinate"
    ///     units = "1"
    ///   float ion_temperature(time, rho_tor_norm)
    ///     long_name = "ion temperature"
    ///     units = "eV"
    ///   float electron_temperature(time, rho_tor_norm)
    ///   float electron_density(time, rho_tor_norm)
    ///   float poloidal_flux(time, rho_tor_norm)  [optional]
    /// ```
    ///
    /// ## Example
    ///
    /// ```swift
    /// let toraxData = try TORAXReferenceData.loadFromNetCDF(
    ///     path: "Tests/GotenxTests/Validation/ReferenceData/torax_iter_baseline.nc"
    /// )
    /// print("Loaded \(toraxData.time.count) time points")
    /// print("Grid size: \(toraxData.normalizedRadius.count) cells")
    /// ```
    public static func loadFromNetCDF(path: String) throws -> TORAXReferenceData {
        // Check file exists
        guard FileManager.default.fileExists(atPath: path) else {
            throw TORAXDataError.fileNotFound(path)
        }

        // Open NetCDF file
        guard let file = try NetCDF.open(path: path, allowUpdate: false) else {
            throw TORAXDataError.fileOpenFailed(path)
        }

        // Read coordinate variables with fallback names
        let timeCandidates = ["time", "t"]
        guard let timeVar = findVariable(file: file, candidates: timeCandidates) else {
            throw TORAXDataError.variableNotFound("time (tried: \(timeCandidates.joined(separator: ", ")))")
        }

        let rhoCandidates = ["rho_tor_norm", "rho", "rho_toroidal"]
        guard let rhoVar = findVariable(file: file, candidates: rhoCandidates) else {
            throw TORAXDataError.variableNotFound("rho_tor_norm (tried: \(rhoCandidates.joined(separator: ", ")))")
        }

        guard let timeVarTyped = timeVar.asType(Float.self) else {
            throw TORAXDataError.invalidData("time variable is not Float type")
        }
        guard let rhoVarTyped = rhoVar.asType(Float.self) else {
            throw TORAXDataError.invalidData("rho_tor_norm variable is not Float type")
        }

        let timeData: [Float] = try timeVarTyped.read()
        let rhoData: [Float] = try rhoVarTyped.read()

        let timeCount = timeData.count
        let nRho = rhoData.count

        // Validate dimensions
        guard timeCount > 0 else {
            throw TORAXDataError.invalidDimensions("time dimension is empty")
        }
        guard nRho >= 10 && nRho <= 200 else {
            throw TORAXDataError.invalidDimensions("rho_tor_norm must be 10-200, got \(nRho)")
        }

        // Verify rho is in ascending order (0 → 1)
        if rhoData.first! > rhoData.last! {
            throw TORAXDataError.invalidData("rho_tor_norm must be in ascending order (0 → 1), got descending")
        }

        // Read profile variables with fallback names
        let tiCandidates = ["ion_temperature", "temp_ion", "Ti", "ti"]
        let ionTemperature = try read2DProfileWithFallback(file: file, candidates: tiCandidates, timeCount: timeCount, nRho: nRho)

        let teCandidates = ["electron_temperature", "temp_electron", "Te", "te", "temp_el"]
        let electronTemperature = try read2DProfileWithFallback(file: file, candidates: teCandidates, timeCount: timeCount, nRho: nRho)

        let neCandidates = ["electron_density", "ne", "n_e", "dens_electron"]
        let electronDensity = try read2DProfileWithFallback(file: file, candidates: neCandidates, timeCount: timeCount, nRho: nRho)

        // Poloidal flux is optional
        let psiCandidates = ["poloidal_flux", "psi", "flux_pol"]
        let poloidalFlux: [[Float]]?
        if findVariable(file: file, candidates: psiCandidates) != nil {
            poloidalFlux = try read2DProfileWithFallback(file: file, candidates: psiCandidates, timeCount: timeCount, nRho: nRho)
        } else {
            poloidalFlux = nil
        }

        return TORAXReferenceData(
            time: timeData,
            normalizedRadius: rhoData,
            ionTemperature: ionTemperature,
            electronTemperature: electronTemperature,
            electronDensity: electronDensity,
            poloidalFlux: poloidalFlux
        )
    }

    /// Find variable in NetCDF file with fallback names
    ///
    /// - Parameters:
    ///   - file: NetCDF file group
    ///   - candidates: List of candidate variable names
    /// - Returns: Variable if found, nil otherwise
    private static func findVariable(file: Group, candidates: [String]) -> Variable? {
        for name in candidates {
            if let variable = file.getVariable(name: name) {
                return variable
            }
        }
        return nil
    }

    /// Read 2D profile with fallback variable names
    ///
    /// - Parameters:
    ///   - file: NetCDF file group
    ///   - candidates: List of candidate variable names
    ///   - timeCount: Expected time dimension length
    ///   - nRho: Expected rho dimension length
    /// - Returns: 2D array [timeCount][nRho]
    private static func read2DProfileWithFallback(
        file: Group,
        candidates: [String],
        timeCount: Int,
        nRho: Int
    ) throws -> [[Float]] {
        guard let variable = findVariable(file: file, candidates: candidates) else {
            throw TORAXDataError.variableNotFound("\(candidates[0]) (tried: \(candidates.joined(separator: ", ")))")
        }

        return try read2DProfile(variable: variable, timeCount: timeCount, nRho: nRho)
    }

    /// Read 2D profile variable from NetCDF file
    ///
    /// - Parameters:
    ///   - variable: NetCDF variable
    ///   - timeCount: Expected time dimension length
    ///   - nRho: Expected rho dimension length
    /// - Returns: 2D array [timeCount][nRho]
    private static func read2DProfile(
        variable: Variable,
        timeCount: Int,
        nRho: Int
    ) throws -> [[Float]] {
        let name = variable.name

        // Verify dimensions
        let dims = variable.dimensions
        guard dims.count == 2 else {
            throw TORAXDataError.invalidDimensions("\(name) must be 2D, got \(dims.count)D")
        }

        // Verify dimension order: expect [time, rho]
        let dimNames = dims.map { $0.name }
        let isTimeFirst = dimNames[0].contains("time") || dimNames[0] == "t"
        let isRhoSecond = dimNames[1].contains("rho")

        guard isTimeFirst && isRhoSecond else {
            throw TORAXDataError.invalidDimensions(
                "\(name) dimensions: expected [time, rho_*], got [\(dimNames[0]), \(dimNames[1])]"
            )
        }

        // Verify dimension sizes match
        let actualNTime = dims[0].length
        let actualNRho = dims[1].length

        guard actualNTime == timeCount else {
            throw TORAXDataError.invalidDimensions(
                "\(name) time dimension mismatch: expected \(timeCount), got \(actualNTime)"
            )
        }
        guard actualNRho == nRho else {
            throw TORAXDataError.invalidDimensions(
                "\(name) rho dimension mismatch: expected \(nRho), got \(actualNRho)"
            )
        }

        // Get typed variable
        guard let typedVar = variable.asType(Float.self) else {
            throw TORAXDataError.invalidData("\(name) is not Float type")
        }

        // Read flat data (row-major: [T0R0, T0R1, ..., T0Rn, T1R0, T1R1, ...])
        let flatData: [Float] = try typedVar.read(offset: [0, 0], count: [timeCount, nRho])

        // Verify data size
        guard flatData.count == timeCount * nRho else {
            throw TORAXDataError.invalidData("\(name) size mismatch: expected \(timeCount * nRho), got \(flatData.count)")
        }

        // Reshape to [[Float]] (time-series of profiles)
        let profiles: [[Float]] = (0..<timeCount).map { t in
            let start = t * nRho
            let end = start + nRho
            return Array(flatData[start..<end])
        }

        return profiles
    }
}
