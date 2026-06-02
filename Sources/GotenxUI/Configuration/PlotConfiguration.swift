// PlotConfiguration.swift
// Configuration for plot properties and figure layout

import Foundation

// MARK: - Plot Type

/// Type of plot to generate
public enum PlotType: String, Sendable, Codable {
    // 2D Spatial Profiles
    case tempDensity = "temp_density"           // Temperature and density vs ρ
    case currentDensity = "current_density"     // Current densities vs ρ
    case qProfile = "q_profile"                 // Safety factor q vs ρ
    case poloidalFlux = "poloidalFlux"                            // Poloidal flux vs ρ

    // 2D Transport Coefficients
    case chiEffective = "chi_eff"               // Effective heat diffusivity
    case chiComparison = "chi_comparison"       // Compare ion/electron χ
    case particleDiffusivity = "d_face"         // Particle diffusivity

    // 2D Sources
    case heatSources = "heat_sources"           // Heating power densities
    case particleSources = "particle_sources"   // Particle sources

    // Time Series
    case plasmaCurrent = "plasma_current"       // Ip, Ibootstrap, ecrhCurrent vs time
    case fusionPower = "fusion_power"           // Fusion gain Q vs time
    case energyBalance = "energy_balance"       // Power balance vs time

    // 3D Volumetric
    case temperature3D = "temperature_3d"       // 3D temperature distribution
    case density3D = "density_3d"               // 3D density distribution
    case pressure3D = "pressure_3d"             // 3D pressure distribution
}

// MARK: - Plot Properties

/// Properties for individual plot customization
public struct PlotProperties: Sendable, Codable {
    /// Plot type
    public let type: PlotType

    /// Plot title
    public let title: String

    /// X-axis label
    public let xLabel: String

    /// Y-axis label
    public let yLabel: String

    /// Z-axis label (3D only)
    public let zLabel: String?

    /// Show legend
    public let showLegend: Bool

    /// Show grid
    public let showGrid: Bool

    /// Line colors (hex strings)
    public let colors: [String]

    /// Line styles
    public let lineStyles: [LineStyle]

    /// Time index for spatial plots (-1 = final time)
    public let timeIndex: Int

    public init(
        type: PlotType,
        title: String,
        xLabel: String,
        yLabel: String,
        zLabel: String? = nil,
        showLegend: Bool = true,
        showGrid: Bool = true,
        colors: [String] = [],
        lineStyles: [LineStyle] = [],
        timeIndex: Int = -1
    ) {
        self.type = type
        self.title = title
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.zLabel = zLabel
        self.showLegend = showLegend
        self.showGrid = showGrid
        self.colors = colors
        self.lineStyles = lineStyles
        self.timeIndex = timeIndex
    }
}

// MARK: - Line Style

/// Line style for 2D plots
public enum LineStyle: String, Sendable, Codable {
    case solid
    case dashed
    case dotted
    case dashDot = "dashdot"
}

// MARK: - Figure Properties

/// Properties for figure layout and appearance
public struct FigureProperties: Sendable, Codable {
    /// Figure width (points)
    public let width: CGFloat

    /// Figure height (points)
    public let height: CGFloat

    /// Background color (hex string)
    public let backgroundColor: String

    /// Font family
    public let fontFamily: String

    /// Base font size (points)
    public let fontSize: CGFloat

    /// DPI for raster output
    public let dpi: Int

    /// Margins (top, right, bottom, left)
    public let margins: FigureMargins

    public init(
        width: CGFloat = 800,
        height: CGFloat = 600,
        backgroundColor: String = "#FFFFFF",
        fontFamily: String = "System",
        fontSize: CGFloat = 12,
        dpi: Int = 300,
        margins: FigureMargins = FigureMargins()
    ) {
        self.width = width
        self.height = height
        self.backgroundColor = backgroundColor
        self.fontFamily = fontFamily
        self.fontSize = fontSize
        self.dpi = dpi
        self.margins = margins
    }
}

// MARK: - Figure Margins

/// Margins for figure layout
public struct FigureMargins: Sendable, Codable {
    public let top: CGFloat
    public let right: CGFloat
    public let bottom: CGFloat
    public let left: CGFloat

    public init(top: CGFloat = 40, right: CGFloat = 40, bottom: CGFloat = 60, left: CGFloat = 80) {
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left
    }
}

// MARK: - Plot Configuration

/// Complete configuration for a plot
public struct PlotConfiguration: Sendable, Codable {
    /// Plot properties
    public let plot: PlotProperties

    /// Figure properties
    public let figure: FigureProperties

    public init(plot: PlotProperties, figure: FigureProperties = FigureProperties()) {
        self.plot = plot
        self.figure = figure
    }
}

// MARK: - Default Configurations

extension PlotConfiguration {
    /// Default configuration for temperature and density profiles
    public static let tempDensityProfile = PlotConfiguration(
        plot: PlotProperties(
            type: .tempDensity,
            title: "Temperature and Density Profiles",
            xLabel: "Normalized radius ρ",
            yLabel: "Temperature [keV] / Density [10²⁰ m⁻³]",
            colors: ["#FF6B6B", "#4ECDC4", "#45B7D1"],
            lineStyles: [.solid, .solid, .dashed]
        )
    )

    /// Default configuration for current density profiles
    public static let currentDensityProfile = PlotConfiguration(
        plot: PlotProperties(
            type: .currentDensity,
            title: "Current Density Profiles",
            xLabel: "Normalized radius ρ",
            yLabel: "Current Density [MA/m²]",
            colors: ["#000000", "#FF6B6B", "#4ECDC4", "#FFA07A"],
            lineStyles: [.solid, .dashed, .dashed, .dotted]
        )
    )

    /// Default configuration for safety factor profile
    public static let qProfile = PlotConfiguration(
        plot: PlotProperties(
            type: .qProfile,
            title: "Safety Factor Profile",
            xLabel: "Normalized radius ρ",
            yLabel: "Safety factor q",
            colors: ["#9B59B6"],
            lineStyles: [.solid]
        )
    )

    /// Default configuration for transport coefficients
    public static let chiEffective = PlotConfiguration(
        plot: PlotProperties(
            type: .chiEffective,
            title: "Effective Heat Diffusivity",
            xLabel: "Normalized radius ρ",
            yLabel: "χ [m²/s]",
            colors: ["#E74C3C", "#3498DB"],
            lineStyles: [.solid, .solid]
        )
    )

    /// Default configuration for heat sources
    public static let heatSources = PlotConfiguration(
        plot: PlotProperties(
            type: .heatSources,
            title: "Heating Power Densities",
            xLabel: "Normalized radius ρ",
            yLabel: "Power Density [MW/m³]",
            colors: ["#E67E22", "#9B59B6", "#1ABC9C"],
            lineStyles: [.solid, .dashed, .dotted]
        )
    )

    /// Default configuration for plasma current time series
    public static let plasmaCurrent = PlotConfiguration(
        plot: PlotProperties(
            type: .plasmaCurrent,
            title: "Plasma Current Evolution",
            xLabel: "Time [s]",
            yLabel: "Current [MA]",
            colors: ["#000000", "#E74C3C", "#3498DB"],
            lineStyles: [.solid, .dashed, .dashed]
        )
    )

    /// Default configuration for fusion power
    public static let fusionPower = PlotConfiguration(
        plot: PlotProperties(
            type: .fusionPower,
            title: "Fusion Gain Evolution",
            xLabel: "Time [s]",
            yLabel: "Fusion Gain Q",
            colors: ["#27AE60"],
            lineStyles: [.solid]
        )
    )

    /// Default configuration for energy balance
    public static let energyBalance = PlotConfiguration(
        plot: PlotProperties(
            type: .energyBalance,
            title: "Power Balance",
            xLabel: "Time [s]",
            yLabel: "Power [MW]",
            colors: ["#E67E22", "#9B59B6", "#E74C3C", "#95A5A6"],
            lineStyles: [.solid, .solid, .dashed, .dotted]
        )
    )

    /// Default configuration for 3D temperature distribution
    public static let temperature3D = PlotConfiguration(
        plot: PlotProperties(
            type: .temperature3D,
            title: "3D Temperature Distribution",
            xLabel: "R [m]",
            yLabel: "Z [m]",
            zLabel: "Temperature [keV]",
            showLegend: false,
            colors: ["#FF0000", "#FFFF00", "#0000FF"]  // Hot to cold colormap
        )
    )

    /// Default configuration for 3D density distribution
    public static let density3D = PlotConfiguration(
        plot: PlotProperties(
            type: .density3D,
            title: "3D Density Distribution",
            xLabel: "R [m]",
            yLabel: "Z [m]",
            zLabel: "Density [10²⁰ m⁻³]",
            showLegend: false,
            colors: ["#FFFFFF", "#4ECDC4", "#0000FF"]  // White to blue
        )
    )

    /// Default configuration for 3D pressure distribution
    public static let pressure3D = PlotConfiguration(
        plot: PlotProperties(
            type: .pressure3D,
            title: "3D Pressure Distribution",
            xLabel: "R [m]",
            yLabel: "Z [m]",
            zLabel: "Pressure [kPa]",
            showLegend: false,
            colors: ["#FFFFCC", "#FFA07A", "#8B0000"]  // Yellow to dark red
        )
    )
}
