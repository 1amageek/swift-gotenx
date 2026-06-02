// PlotData3D.swift
// 3D volumetric simulation data for Chart3D visualization
//
// Converts 1D radial profiles to 3D cylindrical coordinates assuming:
// 1. Toroidal symmetry (no φ dependence)
// 2. Circular poloidal cross-section
// 3. Up-down symmetry

import Foundation

/// 3D volumetric simulation data for Chart3D visualization
public struct PlotData3D: Sendable {
    // MARK: - Cylindrical Coordinates

    /// Major radius R [m] - flattened from (ρ, θ) grid
    public let majorRadiusCoordinates: [Float]

    /// Height Z [m] - flattened from (ρ, θ) grid
    public let heightCoordinates: [Float]

    /// Toroidal angle φ [rad] [toroidalAngleCount]
    public let toroidalAngles: [Float]

    /// Time [s] [timeCount]
    public let time: [Float]

    // MARK: - 4D Volumetric Data [timeCount, poloidalPointCount, toroidalAngleCount]
    // Note: poloidalPointCount = radialPointCount * poloidalAngleCount

    /// Temperature [keV]
    public let temperature: [[[Float]]]

    /// Density [10^20 m^-3]
    public let density: [[[Float]]]

    /// Pressure [kPa]
    public let pressure: [[[Float]]]

    // MARK: - Grid Dimensions

    /// Number of radial points (ρ direction)
    public let radialPointCount: Int

    /// Number of poloidal angle points (θ direction)
    public let poloidalAngleCount: Int

    /// Total poloidal cross-section points
    public var poloidalPointCount: Int { majorRadiusCoordinates.count }

    /// Number of toroidal angle points (φ direction)
    public var toroidalAngleCount: Int { toroidalAngles.count }

    /// Number of time points
    public var timeCount: Int { time.count }

    // MARK: - Initialization

    public init(
        majorRadiusCoordinates: [Float],
        heightCoordinates: [Float],
        toroidalAngles: [Float],
        time: [Float],
        temperature: [[[Float]]],
        density: [[[Float]]],
        pressure: [[[Float]]],
        radialPointCount: Int,
        poloidalAngleCount: Int
    ) {
        self.majorRadiusCoordinates = majorRadiusCoordinates
        self.heightCoordinates = heightCoordinates
        self.toroidalAngles = toroidalAngles
        self.time = time
        self.temperature = temperature
        self.density = density
        self.pressure = pressure
        self.radialPointCount = radialPointCount
        self.poloidalAngleCount = poloidalAngleCount
    }

    /// Generate 3D volumetric data from 1D radial profile
    ///
    /// **Physical Model**:
    /// - Circular poloidal cross-section: R(ρ,θ) = R₀ + ρ·a·cos(θ), Z(ρ,θ) = ρ·a·sin(θ)
    /// - Toroidal symmetry: All quantities independent of φ
    /// - Pressure: P = n·k_B·T (ideal gas law)
    ///
    /// **Coordinate System**:
    /// - ρ ∈ [0,1]: Normalized minor radius
    /// - θ ∈ [0,2π): Poloidal angle (θ=0 is outboard midplane)
    /// - φ ∈ [0,2π): Toroidal angle
    ///
    /// - Parameters:
    ///   - profile: 1D radial profiles (ionTemperature, electronTemperature, electronDensity vs ρ)
    ///   - poloidalAngleCount: Number of poloidal angle points (default: 16)
    ///   - toroidalAngleCount: Number of toroidal angle points (default: 16)
    ///   - geometry: Tokamak geometry (R₀, a)
    public init(
        from profile: PlotData,
        poloidalAngleCount: Int = 16,
        toroidalAngleCount: Int = 16,
        geometry: GeometryParameters
    ) {
        let cellCount = profile.cellCount

        // Generate poloidal cross-section grid (ρ, θ) → (R, Z)
        var rPoints: [Float] = []
        var zPoints: [Float] = []

        for i in 0..<cellCount {
            let normalizedRadius = profile.normalizedRadius[i]

            for j in 0..<poloidalAngleCount {
                let theta = Float(j) * 2.0 * Float.pi / Float(poloidalAngleCount)

                // Circular cross-section in (R, Z) coordinates
                let r = geometry.majorRadius + normalizedRadius * geometry.minorRadius * cos(theta)
                let z = normalizedRadius * geometry.minorRadius * sin(theta)

                rPoints.append(r)
                zPoints.append(z)
            }
        }

        self.majorRadiusCoordinates = rPoints
        self.heightCoordinates = zPoints
        self.radialPointCount = cellCount
        self.poloidalAngleCount = poloidalAngleCount

        // Generate toroidal angle grid
        self.toroidalAngles = (0..<toroidalAngleCount).map { i in
            Float(i) * 2.0 * Float.pi / Float(toroidalAngleCount)
        }

        self.time = profile.time

        // Convert 1D radial profiles to 3D assuming toroidal symmetry
        // Data structure: [timeCount][radialPointCount * poloidalAngleCount][toroidalAngleCount]
        // Each (ρ, θ) point has the same value for all φ

        self.temperature = profile.ionTemperature.map { tiAtTime in
            (0..<cellCount).flatMap { iRho in
                (0..<poloidalAngleCount).map { _ in
                    (0..<toroidalAngleCount).map { _ in
                        tiAtTime[iRho]  // Toroidal symmetry: T(ρ,θ,φ) = T(ρ)
                    }
                }
            }
        }

        self.density = profile.electronDensity.map { neAtTime in
            (0..<cellCount).flatMap { iRho in
                (0..<poloidalAngleCount).map { _ in
                    (0..<toroidalAngleCount).map { _ in
                        neAtTime[iRho]  // Toroidal symmetry: n(ρ,θ,φ) = n(ρ)
                    }
                }
            }
        }

        // Pressure from ideal gas law: P = n * k_B * T
        // Input: electronDensity [10^20 m^-3], electronTemperature [keV]
        // Output: P [kPa]
        //
        // P [Pa] = n [m^-3] * k_B [J/K] * T [K]
        //        = n [10^20 m^-3] * 10^20 * k_B [eV/K] * T [keV] * 1000 * eV_to_J [J/eV]
        //        = n * 10^20 * 8.617e-5 * T * 1000 * 1.602e-19
        //        = n * T * 0.1380649  [Pa]
        // P [kPa] = n * T * 1.380649e-4
        let pressureConversion: Float = 1.380649e-4  // (10^20 m^-3 * keV) → kPa

        self.pressure = zip(profile.electronDensity, profile.electronTemperature).map { electronDensity, te in
            (0..<cellCount).flatMap { iRho in
                (0..<poloidalAngleCount).map { _ in
                    (0..<toroidalAngleCount).map { _ in
                        electronDensity[iRho] * te[iRho] * pressureConversion
                    }
                }
            }
        }
    }

    // MARK: - 3D Point Extraction

    /// Extract 3D points for PointMark3D at given time index
    ///
    /// Returns flattened array of all (R, Z, φ) points at the specified time.
    /// Points are ordered as: [(ρ₀,θ₀,φ₀), (ρ₀,θ₀,φ₁), ..., (ρₙ,θₘ,φₖ)]
    ///
    /// - Parameter timeIndex: Time index (0..<timeCount)
    /// - Returns: Array of volumetric points with physical quantities
    public func volumetricPoints(timeIndex: Int) -> [VolumetricPoint] {
        guard timeIndex < timeCount else { return [] }

        let poloidalPointCount = radialPointCount * poloidalAngleCount

        var points: [VolumetricPoint] = []
        points.reserveCapacity(poloidalPointCount * toroidalAngleCount)

        for iPoloidal in 0..<poloidalPointCount {
            for iPhi in 0..<toroidalAngleCount {
                let point = VolumetricPoint(
                    majorRadius: majorRadiusCoordinates[iPoloidal],
                    height: heightCoordinates[iPoloidal],
                    toroidalAngle: toroidalAngles[iPhi],
                    temperature: temperature[timeIndex][iPoloidal][iPhi],
                    density: density[timeIndex][iPoloidal][iPhi],
                    pressure: pressure[timeIndex][iPoloidal][iPhi]
                )
                points.append(point)
            }
        }

        return points
    }

    /// Temperature range for color mapping
    public var temperatureRange: ClosedRange<Float> {
        let allValues = temperature.flatMap { $0.flatMap { $0 } }
        let min = allValues.min() ?? 0
        let max = allValues.max() ?? 1
        return min...max
    }

    /// Density range for color mapping
    public var densityRange: ClosedRange<Float> {
        let allValues = density.flatMap { $0.flatMap { $0 } }
        let min = allValues.min() ?? 0
        let max = allValues.max() ?? 1
        return min...max
    }

    /// Pressure range for color mapping
    public var pressureRange: ClosedRange<Float> {
        let allValues = pressure.flatMap { $0.flatMap { $0 } }
        let min = allValues.min() ?? 0
        let max = allValues.max() ?? 1
        return min...max
    }
}

// MARK: - Supporting Types

/// Geometry parameters for 3D reconstruction
public struct GeometryParameters: Sendable {
    /// Major radius R₀ [m]
    public let majorRadius: Float

    /// Minor radius a [m]
    public let minorRadius: Float

    public init(majorRadius: Float, minorRadius: Float) {
        self.majorRadius = majorRadius
        self.minorRadius = minorRadius
    }

    /// Default ITER-like geometry (R₀=6.2m, a=2.0m)
    public static let iterLike = GeometryParameters(majorRadius: 6.2, minorRadius: 2.0)

    /// Aspect ratio R₀/a
    public var aspectRatio: Float {
        majorRadius / minorRadius
    }

    /// Plasma volume [m³] for circular cross-section
    /// V = 2π²·R₀·a²
    public var volume: Float {
        2.0 * Float.pi * Float.pi * majorRadius * minorRadius * minorRadius
    }
}

/// Single point in 3D volumetric space
public struct VolumetricPoint: Identifiable, Sendable {
    public let id = UUID()

    /// Major radius R [m]
    public let majorRadius: Float

    /// Height Z [m]
    public let height: Float

    /// Toroidal angle φ [rad]
    public let toroidalAngle: Float

    /// Temperature [keV]
    public let temperature: Float

    /// Density [10^20 m^-3]
    public let density: Float

    /// Pressure [kPa]
    public let pressure: Float

    public init(
        majorRadius: Float,
        height: Float,
        toroidalAngle: Float,
        temperature: Float,
        density: Float,
        pressure: Float
    ) {
        self.majorRadius = majorRadius
        self.height = height
        self.toroidalAngle = toroidalAngle
        self.temperature = temperature
        self.density = density
        self.pressure = pressure
    }

    /// Distance from magnetic axis [m]
    public func minorRadius(geometry: GeometryParameters) -> Float {
        let radialOffset = majorRadius - geometry.majorRadius
        return sqrt(radialOffset * radialOffset + height * height)
    }

    /// Normalized minor radius ρ = r_minor / a
    public func normalizedRadius(geometry: GeometryParameters) -> Float {
        minorRadius(geometry: geometry) / geometry.minorRadius
    }

    /// Poloidal angle θ [rad]
    public func poloidalAngle(geometry: GeometryParameters) -> Float {
        let radialOffset = majorRadius - geometry.majorRadius
        return atan2(height, radialOffset)
    }
}
