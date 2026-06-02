// PlotData3DTests.swift
// Unit tests for PlotData3D model

import Testing
import Foundation
@testable import GotenxUI

@Suite("PlotData3D Tests")
struct PlotData3DTests {

    // MARK: - Coordinate System Tests

    @Test("Circular cross-section coordinate generation")
    func testCircularCrossSection() {
        let nTheta = 8
        let geometry = GeometryParameters(majorRadius: 6.0, minorRadius: 2.0)

        let mockPlotData = createMockPlotData(cellCount: 3, timeCount: 1)
        let plotData3D = PlotData3D(from: mockPlotData, poloidalAngleCount: nTheta, toroidalAngleCount: 4, geometry: geometry)

        // Check that we have cellCount * nTheta poloidal points
        #expect(plotData3D.poloidalPointCount == 3 * 8)
        #expect(plotData3D.majorRadiusCoordinates.count == 24)
        #expect(plotData3D.heightCoordinates.count == 24)

        // Check first point (ρ=0, θ=0) should be at (R₀, 0)
        #expect(abs(plotData3D.majorRadiusCoordinates[0] - geometry.majorRadius) < 1e-6)
        #expect(abs(plotData3D.heightCoordinates[0]) < 1e-6)

        // Check outboard midplane point (ρ=1, θ=0) should be at (R₀+a, 0)
        let lastRhoFirstTheta = (3 - 1) * nTheta  // (cellCount-1) * nTheta
        #expect(abs(plotData3D.majorRadiusCoordinates[lastRhoFirstTheta] - (geometry.majorRadius + geometry.minorRadius)) < 1e-5)
        #expect(abs(plotData3D.heightCoordinates[lastRhoFirstTheta]) < 1e-5)
    }

    @Test("Toroidal angle grid generation")
    func testToroidalGrid() {
        let nPhi = 8
        let geometry = GeometryParameters.iterLike

        let mockPlotData = createMockPlotData(cellCount: 5, timeCount: 1)
        let plotData3D = PlotData3D(from: mockPlotData, poloidalAngleCount: 4, toroidalAngleCount: nPhi, geometry: geometry)

        // Check phi coordinate (should span 0 to 2π)
        #expect(plotData3D.toroidalAngles.count == nPhi)
        #expect(plotData3D.toroidalAngles.first! == 0.0)

        let expectedLastPhi = 2.0 * Float.pi * Float(nPhi - 1) / Float(nPhi)
        #expect(abs(plotData3D.toroidalAngles.last! - expectedLastPhi) < 1e-5)
    }

    @Test("Toroidal symmetry assumption")
    func testToroidalSymmetry() {
        let nPhi = 4
        let geometry = GeometryParameters.iterLike

        let mockPlotData = createMockPlotData(cellCount: 5, timeCount: 1)
        let plotData3D = PlotData3D(from: mockPlotData, poloidalAngleCount: 4, toroidalAngleCount: nPhi, geometry: geometry)

        // Extract temperature at fixed poloidal point for all phi
        let iPoloidal = 10  // Arbitrary poloidal point

        let tempAtDifferentPhi = (0..<nPhi).map { iPhi in
            plotData3D.temperature[0][iPoloidal][iPhi]
        }

        // All phi values should be the same (toroidal symmetry)
        let referenceTemp = tempAtDifferentPhi[0]
        for temp in tempAtDifferentPhi {
            #expect(abs(temp - referenceTemp) < 1e-6)
        }
    }

    // MARK: - Physical Calculations Tests

    @Test("Pressure calculation from ideal gas law")
    func testPressureCalculation() {
        let geometry = GeometryParameters.iterLike

        // Create data with known values
        let mockPlotData = createMockPlotData(
            cellCount: 5,
            timeCount: 1,
            tempValues: [10.0, 8.0, 6.0, 4.0, 2.0],  // keV
            densityValues: [5.0, 4.0, 3.0, 2.0, 1.0]  // 10^20 m^-3
        )

        let plotData3D = PlotData3D(from: mockPlotData, poloidalAngleCount: 4, toroidalAngleCount: 2, geometry: geometry)

        // Check pressure at first radial point (ρ=0)
        // P = n * T * 1.380649e-4  [kPa]
        // P(ρ=0) = 5.0 * 10.0 * 1.380649e-4 = 0.00690 kPa
        let expectedPressure: Float = 5.0 * 10.0 * 1.380649e-4
        let actualPressure = plotData3D.pressure[0][0][0]  // First time, first point, first phi

        #expect(abs(actualPressure - expectedPressure) < 1e-5)
    }

    @Test("Geometry parameters calculations")
    func testGeometryCalculations() {
        let geometry = GeometryParameters(majorRadius: 6.0, minorRadius: 2.0)

        // Aspect ratio
        #expect(geometry.aspectRatio == 3.0)

        // Volume for circular cross-section: V = 2π²·R₀·a²
        let expectedVolume = 2.0 * Float.pi * Float.pi * 6.0 * 2.0 * 2.0
        #expect(abs(geometry.volume - expectedVolume) < 1e-4)
    }

    // MARK: - Volumetric Point Extraction Tests

    @Test("Volumetric point extraction")
    func testVolumetricPoints() {
        let nRho = 3
        let nTheta = 4
        let nPhi = 2
        let geometry = GeometryParameters(majorRadius: 6.0, minorRadius: 2.0)

        let mockPlotData = createMockPlotData(cellCount: nRho, timeCount: 1)
        let plotData3D = PlotData3D(from: mockPlotData, poloidalAngleCount: nTheta, toroidalAngleCount: nPhi, geometry: geometry)

        let points = plotData3D.volumetricPoints(timeIndex: 0)

        // Total number of points = nRho × nTheta × nPhi
        let expectedCount = nRho * nTheta * nPhi
        #expect(points.count == expectedCount)

        // Check that all points have valid coordinates
        for point in points {
            #expect(point.majorRadius > 0)
            #expect(point.toroidalAngle >= 0 && point.toroidalAngle < 2 * Float.pi)
            #expect(point.temperature >= 0)
            #expect(point.density >= 0)
            #expect(point.pressure >= 0)
        }
    }

    @Test("VolumetricPoint helper methods")
    func testVolumetricPointHelpers() {
        let geometry = GeometryParameters(majorRadius: 6.0, minorRadius: 2.0)

        // Point at outboard midplane (θ=0) at ρ=0.5
        let majorRadius: Float = 6.0 + 0.5 * 2.0 * cos(0)  // = 7.0
        let height: Float = 0.5 * 2.0 * sin(0)        // = 0.0

        let point = VolumetricPoint(
            majorRadius: majorRadius,
            height: height,
            toroidalAngle: 0,
            temperature: 10.0,
            density: 5.0,
            pressure: 0.0069
        )

        // Minor radius from axis
        let minorRad = point.minorRadius(geometry: geometry)
        #expect(abs(minorRad - 1.0) < 1e-5)  // Should be 1.0 m (0.5 * 2.0)

        // Normalized radius
        let normalizedRadius = point.normalizedRadius(geometry: geometry)
        #expect(abs(normalizedRadius - 0.5) < 1e-5)

        // Poloidal angle
        let theta = point.poloidalAngle(geometry: geometry)
        #expect(abs(theta) < 1e-5)  // Should be 0 (outboard midplane)
    }

    @Test("Range calculations for color mapping")
    func testColorMappingRanges() {
        let mockPlotData = createMockPlotData(
            cellCount: 5,
            timeCount: 2,
            tempValues: [1.0, 5.0, 10.0, 15.0, 20.0]
        )
        let plotData3D = PlotData3D(from: mockPlotData, poloidalAngleCount: 4, toroidalAngleCount: 4, geometry: .iterLike)

        // Temperature range
        let tempRange = plotData3D.temperatureRange
        #expect(tempRange.lowerBound <= 1.0)
        #expect(tempRange.upperBound >= 20.0)

        // Density range
        let densityRange = plotData3D.densityRange
        #expect(densityRange.lowerBound >= 0)
        #expect(densityRange.upperBound > 0)

        // Pressure range
        let pressureRange = plotData3D.pressureRange
        #expect(pressureRange.lowerBound >= 0)
        #expect(pressureRange.upperBound > 0)
    }

    // MARK: - Edge Case Tests

    @Test("Single radial point")
    func testSingleRadialPoint() {
        let mockPlotData = createMockPlotData(cellCount: 1, timeCount: 1)
        let plotData3D = PlotData3D(from: mockPlotData, poloidalAngleCount: 4, toroidalAngleCount: 4, geometry: .iterLike)

        #expect(plotData3D.radialPointCount == 1)
        #expect(plotData3D.poloidalAngleCount == 4)
        #expect(plotData3D.poloidalPointCount == 4)
    }

    @Test("Single poloidal point")
    func testSinglePoloidalPoint() {
        let mockPlotData = createMockPlotData(cellCount: 5, timeCount: 1)
        let plotData3D = PlotData3D(from: mockPlotData, poloidalAngleCount: 1, toroidalAngleCount: 4, geometry: .iterLike)

        #expect(plotData3D.poloidalAngleCount == 1)
        #expect(plotData3D.poloidalPointCount == 5)
    }

    @Test("Out of bounds time index returns empty points")
    func testOutOfBoundsTimeIndex() {
        let mockPlotData = createMockPlotData(cellCount: 5, timeCount: 2)
        let plotData3D = PlotData3D(from: mockPlotData, poloidalAngleCount: 4, toroidalAngleCount: 4, geometry: .iterLike)

        let points = plotData3D.volumetricPoints(timeIndex: 999)

        #expect(points.isEmpty)
    }

    // MARK: - Mock Data Helpers

    private func createMockPlotData(
        cellCount: Int,
        timeCount: Int,
        tempValues: [Float]? = nil,
        densityValues: [Float]? = nil
    ) -> PlotData {
        let normalizedRadius = (0..<cellCount).map { Float($0) / Float(max(cellCount - 1, 1)) }
        let time = (0..<timeCount).map { Float($0) * 0.01 }

        let defaultTemp = tempValues ?? Array(repeating: 10.0 as Float, count: cellCount)
        let defaultDensity = densityValues ?? Array(repeating: 5.0 as Float, count: cellCount)

        let ionTemperature: [[Float]] = Array(repeating: defaultTemp, count: timeCount)
        let electronTemperature: [[Float]] = Array(repeating: defaultTemp, count: timeCount)
        let electronDensity: [[Float]] = Array(repeating: defaultDensity, count: timeCount)

        let zeroProfile: [Float] = Array(repeating: 0.0 as Float, count: cellCount)
        let zeroProfiles: [[Float]] = Array(repeating: zeroProfile, count: timeCount)
        let zeroScalar: [Float] = Array(repeating: 0.0 as Float, count: timeCount)

        return PlotData(
            normalizedRadius: normalizedRadius,
            time: time,
            ionTemperature: ionTemperature,
            electronTemperature: electronTemperature,
            electronDensity: electronDensity,
            safetyFactor: zeroProfiles,
            magneticShear: zeroProfiles,
            poloidalFlux: zeroProfiles,
            totalIonHeatConductivity: zeroProfiles,
            totalElectronHeatConductivity: zeroProfiles,
            turbulentIonHeatConductivity: zeroProfiles,
            turbulentElectronHeatConductivity: zeroProfiles,
            particleDiffusivity: zeroProfiles,
            totalCurrentDensity: zeroProfiles,
            ohmicCurrentDensity: zeroProfiles,
            bootstrapCurrentDensity: zeroProfiles,
            ecrhCurrentDensity: zeroProfiles,
            ohmicHeatSource: zeroProfiles,
            fusionHeatSource: zeroProfiles,
            icrhIonHeatingPowerDensity: zeroProfiles,
            icrhElectronHeatingPowerDensity: zeroProfiles,
            ecrhElectronHeatingPowerDensity: zeroProfiles,
            plasmaCurrent: zeroScalar,
            bootstrapCurrent: zeroScalar,
            ecrhCurrent: zeroScalar,
            fusionGain: zeroScalar,
            auxiliaryHeatingPower: zeroScalar,
            ohmicElectronHeatingPower: zeroScalar,
            totalAlphaPower: zeroScalar,
            bremsstrahlungPower: zeroScalar,
            radiationPower: zeroScalar
        )
    }
}
