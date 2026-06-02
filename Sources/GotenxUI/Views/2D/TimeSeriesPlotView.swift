// TimeSeriesPlotView.swift
// Time series charts (t-axis plots)

import SwiftUI
import Charts

// MARK: - Plasma Current Chart

struct PlasmaCurrentChart: View {
    let data: PlotData
    let config: PlotConfiguration

    var body: some View {
        Chart {
            // Total plasma current
            ForEach(Array(data.time.enumerated()), id: \.offset) { index, time in
                LineMark(
                    x: .value("Time", time),
                    y: .value("Ip", data.plasmaCurrent[index])
                )
                .foregroundStyle(Color(hex: config.plot.colors[safe: 0] ?? "#000000"))
            }
            .accessibilityLabel("Total Plasma Current")

            // Bootstrap current
            ForEach(Array(data.time.enumerated()), id: \.offset) { index, time in
                LineMark(
                    x: .value("Time", time),
                    y: .value("Ibs", data.bootstrapCurrent[index])
                )
                .foregroundStyle(Color(hex: config.plot.colors[safe: 1] ?? "#E74C3C"))
                .lineStyle(StrokeStyle(dash: [5, 5]))
            }
            .accessibilityLabel("Bootstrap Current")

            // ECRH current
            ForEach(Array(data.time.enumerated()), id: \.offset) { index, time in
                LineMark(
                    x: .value("Time", time),
                    y: .value("ecrhCurrent", data.ecrhCurrent[index])
                )
                .foregroundStyle(Color(hex: config.plot.colors[safe: 2] ?? "#3498DB"))
                .lineStyle(StrokeStyle(dash: [5, 5]))
            }
            .accessibilityLabel("ECRH Current")
        }
        .chartLegend(config.plot.showLegend ? .visible : .hidden)
        .frame(height: config.figure.height * 0.7)
    }
}

// MARK: - Fusion Power Chart

struct FusionPowerChart: View {
    let data: PlotData
    let config: PlotConfiguration

    var body: some View {
        Chart {
            ForEach(Array(data.time.enumerated()), id: \.offset) { index, time in
                LineMark(
                    x: .value("Time", time),
                    y: .value("Q", data.fusionGain[index])
                )
                .foregroundStyle(Color(hex: config.plot.colors[safe: 0] ?? "#27AE60"))
            }
        }
        .chartLegend(.hidden)
        .frame(height: config.figure.height * 0.7)
    }
}

// MARK: - Energy Balance Chart

struct EnergyBalanceChart: View {
    let data: PlotData
    let config: PlotConfiguration

    var body: some View {
        Chart {
            // Auxiliary power
            ForEach(Array(data.time.enumerated()), id: \.offset) { index, time in
                LineMark(
                    x: .value("Time", time),
                    y: .value("P_aux", data.auxiliaryHeatingPower[index])
                )
                .foregroundStyle(Color(hex: config.plot.colors[safe: 0] ?? "#E67E22"))
            }
            .accessibilityLabel("Auxiliary Power")

            // Alpha power
            ForEach(Array(data.time.enumerated()), id: \.offset) { index, time in
                LineMark(
                    x: .value("Time", time),
                    y: .value("P_alpha", data.totalAlphaPower[index])
                )
                .foregroundStyle(Color(hex: config.plot.colors[safe: 1] ?? "#9B59B6"))
            }
            .accessibilityLabel("Alpha Power")

            // Ohmic power
            ForEach(Array(data.time.enumerated()), id: \.offset) { index, time in
                LineMark(
                    x: .value("Time", time),
                    y: .value("P_ohmic", data.ohmicElectronHeatingPower[index])
                )
                .foregroundStyle(Color(hex: config.plot.colors[safe: 2] ?? "#E74C3C"))
                .lineStyle(StrokeStyle(dash: [5, 5]))
            }
            .accessibilityLabel("Ohmic Power")

            // Radiation
            ForEach(Array(data.time.enumerated()), id: \.offset) { index, time in
                LineMark(
                    x: .value("Time", time),
                    y: .value("P_rad", data.radiationPower[index])
                )
                .foregroundStyle(Color(hex: config.plot.colors[safe: 3] ?? "#95A5A6"))
                .lineStyle(StrokeStyle(dash: [2, 2]))
            }
            .accessibilityLabel("Radiation Loss")
        }
        .chartLegend(config.plot.showLegend ? .visible : .hidden)
        .frame(height: config.figure.height * 0.7)
    }
}
