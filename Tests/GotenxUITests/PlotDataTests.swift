// PlotDataTests.swift
// Unit tests for PlotData model

import Testing
import Foundation
@testable import GotenxUI

@Suite("PlotData Tests")
struct PlotDataTests {

    // MARK: - Data Structure Tests

    @Test("PlotData initialization with valid data")
    func testPlotDataInitialization() {
        let plotData = createMockPlotData(cellCount: 10, timeCount: 5)

        #expect(plotData.cellCount == 10)
        #expect(plotData.timeCount == 5)
        #expect(plotData.normalizedRadius.count == 10)
        #expect(plotData.time.count == 5)
    }

    @Test("Rho coordinate generation")
    func testRhoCoordinate() {
        let plotData = createMockPlotData(cellCount: 10, timeCount: 1)

        // Verify normalizedRadius coordinate (normalized radius from 0 to 1)
        #expect(plotData.normalizedRadius.first! == 0.0)
        #expect(plotData.normalizedRadius.last! == 1.0)
        #expect(plotData.normalizedRadius.count == 10)

        // Check spacing
        let expectedSpacing = 1.0 / Float(10 - 1)
        for i in 0..<9 {
            let spacing = plotData.normalizedRadius[i + 1] - plotData.normalizedRadius[i]
            #expect(abs(spacing - expectedSpacing) < 1e-6)
        }
    }

    @Test("Time range calculation")
    func testTimeRange() {
        let plotData = createMockPlotData(cellCount: 5, timeCount: 10)

        let range = plotData.timeRange
        #expect(range.lowerBound == plotData.time.first!)
        #expect(range.upperBound == plotData.time.last!)
    }

    @Test("Rho range calculation")
    func testRhoRange() {
        let plotData = createMockPlotData(cellCount: 5, timeCount: 1)

        let range = plotData.normalizedRadiusRange
        #expect(range.lowerBound == 0.0)
        #expect(range.upperBound == 1.0)
    }

    // MARK: - Mock Data Helpers

    private func createMockPlotData(cellCount: Int, timeCount: Int) -> PlotData {
        let normalizedRadius = (0..<cellCount).map { Float($0) / Float(max(cellCount - 1, 1)) }
        let time = (0..<timeCount).map { Float($0) * 0.01 }

        let zeroProfile: [Float] = Array(repeating: 0.0 as Float, count: cellCount)
        let zeroProfiles: [[Float]] = Array(repeating: zeroProfile, count: timeCount)
        let zeroScalar: [Float] = Array(repeating: 0.0 as Float, count: timeCount)

        return PlotData(
            normalizedRadius: normalizedRadius,
            time: time,
            ionTemperature: zeroProfiles,
            electronTemperature: zeroProfiles,
            electronDensity: zeroProfiles,
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
