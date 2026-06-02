import Foundation

// MARK: - Validation Types

/// Geometry parameters for tokamak configuration
public struct GeometryParameters: Sendable, Codable {
    /// Major radius [m]
    public let majorRadius: Float

    /// Minor radius [m]
    public let minorRadius: Float

    /// Plasma elongation (optional)
    public let elongation: Float?

    /// Plasma triangularity (optional)
    public let triangularity: Float?

    /// Plasma current [MA]
    public let plasmaCurrent: Float

    /// Toroidal magnetic field [T]
    public let toroidalField: Float

    public init(
        majorRadius: Float,
        minorRadius: Float,
        elongation: Float? = nil,
        triangularity: Float? = nil,
        plasmaCurrent: Float,
        toroidalField: Float
    ) {
        self.majorRadius = majorRadius
        self.minorRadius = minorRadius
        self.elongation = elongation
        self.triangularity = triangularity
        self.plasmaCurrent = plasmaCurrent
        self.toroidalField = toroidalField
    }
}

/// Reference profiles at a specific time point
public struct ReferenceProfiles: Sendable, Codable {
    /// Normalized toroidal flux coordinate [dimensionless]
    public let normalizedRadius: [Float]

    /// Ion temperature [eV]
    public let ionTemperature: [Float]

    /// Electron temperature [eV]
    public let electronTemperature: [Float]

    /// Electron density [m⁻³]
    public let electronDensity: [Float]

    /// Time point [s]
    public let time: Float

    public init(
        normalizedRadius: [Float],
        ionTemperature: [Float],
        electronTemperature: [Float],
        electronDensity: [Float],
        time: Float
    ) {
        self.normalizedRadius = normalizedRadius
        self.ionTemperature = ionTemperature
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity
        self.time = time
    }
}

/// Global quantities (volume-integrated)
public struct GlobalQuantities: Sendable, Codable {
    /// Fusion power [MW]
    public let fusionPower: Float

    /// Alpha power [MW]
    public let alphaPower: Float

    /// Energy confinement time [s]
    public let energyConfinementTime: Float

    /// Normalized beta
    public let normalizedBeta: Float

    /// Fusion gain Q = fusionPower / P_input
    public let fusionGain: Float

    public init(fusionPower: Float, alphaPower: Float, energyConfinementTime: Float, normalizedBeta: Float, fusionGain: Float) {
        self.fusionPower = fusionPower
        self.alphaPower = alphaPower
        self.energyConfinementTime = energyConfinementTime
        self.normalizedBeta = normalizedBeta
        self.fusionGain = fusionGain
    }
}

/// Comparison result between predicted and reference data
public struct ComparisonResult: Sendable {
    /// Quantity being compared (e.g., "ion_temperature")
    public let quantity: String

    /// L2 relative error
    public let l2Error: Float

    /// Mean absolute percentage error (%)
    public let mape: Float

    /// Pearson correlation coefficient
    public let correlation: Float

    /// Time point [s]
    public let time: Float

    /// Pass/fail status (based on thresholds)
    public let passed: Bool

    public init(
        quantity: String,
        l2Error: Float,
        mape: Float,
        correlation: Float,
        time: Float,
        passed: Bool
    ) {
        self.quantity = quantity
        self.l2Error = l2Error
        self.mape = mape
        self.correlation = correlation
        self.time = time
        self.passed = passed
    }
}

/// Validation thresholds for comparison metrics
public struct ValidationThresholds: Sendable {
    /// Maximum acceptable L2 relative error
    public let maximumL2Error: Float

    /// Maximum acceptable MAPE (%)
    public let maximumMAPE: Float

    /// Minimum acceptable Pearson correlation
    public let minimumCorrelation: Float

    public init(
        maximumL2Error: Float = 0.1,      // 10%
        maximumMAPE: Float = 20.0,         // 20%
        minimumCorrelation: Float = 0.95   // r > 0.95
    ) {
        self.maximumL2Error = maximumL2Error
        self.maximumMAPE = maximumMAPE
        self.minimumCorrelation = minimumCorrelation
    }

    /// Standard thresholds for TORAX comparison
    public static let torax = ValidationThresholds(
        maximumL2Error: 0.1,
        maximumMAPE: 15.0,
        minimumCorrelation: 0.95
    )

    /// Relaxed thresholds for experimental data
    public static let experimental = ValidationThresholds(
        maximumL2Error: 0.2,
        maximumMAPE: 25.0,
        minimumCorrelation: 0.90
    )
}
