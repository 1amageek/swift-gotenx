// TimeSlider.swift
// Interactive time slider for navigating simulation time series

import SwiftUI

// MARK: - Time Slider

/// Interactive slider for selecting time index in simulation data
public struct TimeSlider: View {
    let data: PlotData
    @Binding var timeIndex: Int

    /// Display mode for time label
    public enum DisplayMode {
        case timeValue      // Show time in seconds
        case timeIndex      // Show time index
        case both           // Show both
    }

    let displayMode: DisplayMode

    public init(
        data: PlotData,
        timeIndex: Binding<Int>,
        displayMode: DisplayMode = .both
    ) {
        self.data = data
        self._timeIndex = timeIndex
        self.displayMode = displayMode
    }

    public var body: some View {
        VStack(spacing: 8) {
            // Time label
            HStack {
                Text(timeLabel)
                    .font(.headline)
                Spacer()
                Text(timeDetails)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            // Slider
            HStack(spacing: 12) {
                Button(action: previousStep) {
                    Image(systemName: "chevron.left")
                }
                .disabled(timeIndex == 0)

                Slider(
                    value: Binding(
                        get: { Double(timeIndex) },
                        set: { timeIndex = Int($0) }
                    ),
                    in: 0...Double(max(data.timeCount - 1, 0)),
                    step: 1
                )

                Button(action: nextStep) {
                    Image(systemName: "chevron.right")
                }
                .disabled(timeIndex >= data.timeCount - 1)
            }

            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    Rectangle()
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 4)

                    // Progress
                    Rectangle()
                        .fill(Color.blue)
                        .frame(width: progressWidth(geometry.size.width), height: 4)
                }
            }
            .frame(height: 4)
        }
        .padding()
        #if canImport(UIKit)
        .background(Color(uiColor: .systemBackground))
        #else
        .background(Color(nsColor: .controlBackgroundColor))
        #endif
        .cornerRadius(8)
        .shadow(radius: 2)
    }

    // MARK: - Computed Properties

    private var timeLabel: String {
        switch displayMode {
        case .timeValue:
            return String(format: "t = %.4f s", currentTime)
        case .timeIndex:
            return "Step \(timeIndex)"
        case .both:
            return String(format: "Step %d (t = %.4f s)", timeIndex, currentTime)
        }
    }

    private var timeDetails: String {
        guard data.timeCount > 1 else { return "" }
        let timeStep = timeIndex < data.timeCount - 1 ? data.time[timeIndex + 1] - data.time[timeIndex] : 0
        return String(format: "dt = %.2e s", timeStep)
    }

    private var currentTime: Float {
        guard timeIndex < data.timeCount else { return 0 }
        return data.time[timeIndex]
    }

    private func progressWidth(_ totalWidth: CGFloat) -> CGFloat {
        guard data.timeCount > 1 else { return 0 }
        let progress = CGFloat(timeIndex) / CGFloat(data.timeCount - 1)
        return totalWidth * progress
    }

    // MARK: - Actions

    private func previousStep() {
        if timeIndex > 0 {
            timeIndex -= 1
        }
    }

    private func nextStep() {
        if timeIndex < data.timeCount - 1 {
            timeIndex += 1
        }
    }
}

// MARK: - Time Range Selector

/// Range selector for filtering time series data
public struct TimeRangeSelector: View {
    let data: PlotData
    @Binding var startIndex: Int
    @Binding var endIndex: Int

    public init(
        data: PlotData,
        startIndex: Binding<Int>,
        endIndex: Binding<Int>
    ) {
        self.data = data
        self._startIndex = startIndex
        self._endIndex = endIndex
    }

    public var body: some View {
        VStack(spacing: 12) {
            // Title
            HStack {
                Text("Time Range")
                    .font(.headline)
                Spacer()
                Button("Reset") {
                    startIndex = 0
                    endIndex = data.timeCount - 1
                }
                .font(.subheadline)
            }

            // Start time
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Start")
                        .font(.subheadline)
                    Spacer()
                    Text(String(format: "%.4f s", data.time[startIndex]))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Slider(
                    value: Binding(
                        get: { Double(startIndex) },
                        set: { startIndex = min(Int($0), endIndex) }
                    ),
                    in: 0...Double(max(data.timeCount - 1, 0)),
                    step: 1
                )
            }

            // End time
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("End")
                        .font(.subheadline)
                    Spacer()
                    Text(String(format: "%.4f s", data.time[endIndex]))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Slider(
                    value: Binding(
                        get: { Double(endIndex) },
                        set: { endIndex = max(Int($0), startIndex) }
                    ),
                    in: 0...Double(max(data.timeCount - 1, 0)),
                    step: 1
                )
            }

            // Duration
            HStack {
                Text("Duration:")
                    .font(.subheadline)
                Spacer()
                Text(String(format: "%.4f s", data.time[endIndex] - data.time[startIndex]))
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        #if canImport(UIKit)
        .background(Color(uiColor: .systemBackground))
        #else
        .background(Color(nsColor: .controlBackgroundColor))
        #endif
        .cornerRadius(8)
        .shadow(radius: 2)
    }
}

// MARK: - Playback Controls

/// Playback controls for animating time series
public struct PlaybackControls: View {
    let data: PlotData
    @Binding var timeIndex: Int
    @State private var isPlaying = false
    @State private var playbackSpeed: Double = 1.0

    let timer = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    public init(
        data: PlotData,
        timeIndex: Binding<Int>
    ) {
        self.data = data
        self._timeIndex = timeIndex
    }

    public var body: some View {
        HStack(spacing: 20) {
            // Play/Pause button
            Button(action: togglePlayback) {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.title2)
            }

            // Step backward
            Button(action: stepBackward) {
                Image(systemName: "backward.frame.fill")
            }
            .disabled(timeIndex == 0)

            // Step forward
            Button(action: stepForward) {
                Image(systemName: "forward.frame.fill")
            }
            .disabled(timeIndex >= data.timeCount - 1)

            Divider()

            // Speed control
            VStack(alignment: .leading, spacing: 4) {
                Text("Speed: \(String(format: "%.1f×", playbackSpeed))")
                    .font(.caption)
                Slider(value: $playbackSpeed, in: 0.1...5.0, step: 0.1)
                    .frame(width: 150)
            }

            Spacer()

            // Reset button
            Button("Reset") {
                timeIndex = 0
                isPlaying = false
            }
        }
        .padding()
        #if canImport(UIKit)
        .background(Color(uiColor: .systemBackground))
        #else
        .background(Color(nsColor: .controlBackgroundColor))
        #endif
        .cornerRadius(8)
        .shadow(radius: 2)
        .onReceive(timer) { _ in
            if isPlaying {
                advanceFrame()
            }
        }
    }

    // MARK: - Actions

    private func togglePlayback() {
        isPlaying.toggle()
        if isPlaying && timeIndex >= data.timeCount - 1 {
            timeIndex = 0
        }
    }

    private func stepForward() {
        if timeIndex < data.timeCount - 1 {
            timeIndex += 1
        }
    }

    private func stepBackward() {
        if timeIndex > 0 {
            timeIndex -= 1
        }
    }

    private func advanceFrame() {
        let speedFactor = Int(max(1, playbackSpeed))
        timeIndex += speedFactor
        if timeIndex >= data.timeCount {
            timeIndex = data.timeCount - 1
            isPlaying = false
        }
    }
}

// MARK: - Preview

#Preview("Time Slider") {
    // @Previewable declarations MUST come first
    @Previewable @State var timeIndex = 0

    // Static data declarations come after @Previewable
    let sampleData = PlotData(
        normalizedRadius: [0.0, 0.5, 1.0],
        time: [0.0, 0.5, 1.0, 1.5, 2.0],
        ionTemperature: Array(repeating: [1.0, 2.0, 3.0], count: 5),
        electronTemperature: Array(repeating: [1.0, 2.0, 3.0], count: 5),
        electronDensity: Array(repeating: [1.0, 2.0, 3.0], count: 5),
        safetyFactor: Array(repeating: [1.0, 2.0, 3.0], count: 5),
        magneticShear: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        poloidalFlux: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        totalIonHeatConductivity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        totalElectronHeatConductivity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        turbulentIonHeatConductivity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        turbulentElectronHeatConductivity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        particleDiffusivity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        totalCurrentDensity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        ohmicCurrentDensity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        bootstrapCurrentDensity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        ecrhCurrentDensity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        ohmicHeatSource: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        fusionHeatSource: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        icrhIonHeatingPowerDensity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        icrhElectronHeatingPowerDensity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        ecrhElectronHeatingPowerDensity: Array(repeating: [0.0, 0.0, 0.0], count: 5),
        plasmaCurrent: [0.0, 0.0, 0.0, 0.0, 0.0],
        bootstrapCurrent: [0.0, 0.0, 0.0, 0.0, 0.0],
        ecrhCurrent: [0.0, 0.0, 0.0, 0.0, 0.0],
        fusionGain: [0.0, 0.0, 0.0, 0.0, 0.0],
        auxiliaryHeatingPower: [0.0, 0.0, 0.0, 0.0, 0.0],
        ohmicElectronHeatingPower: [0.0, 0.0, 0.0, 0.0, 0.0],
        totalAlphaPower: [0.0, 0.0, 0.0, 0.0, 0.0],
        bremsstrahlungPower: [0.0, 0.0, 0.0, 0.0, 0.0],
        radiationPower: [0.0, 0.0, 0.0, 0.0, 0.0]
    )

    TimeSlider(data: sampleData, timeIndex: $timeIndex)
        .padding()
}
