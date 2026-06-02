// PowerBalanceComputer.swift
// Power balance computation with source metadata
//
// Tracks individual source contributions and rejects untracked power accounting.

/// Power balance computation results
///
/// Units: All powers in [W] (Watts)
public struct PowerBalance: Sendable {
    /// Total fusion power [W]
    public let fusionPower: Float

    /// Alpha particle heating power [W]
    public let alphaPower: Float

    /// Auxiliary heating power [W]
    public let auxiliaryPower: Float

    /// Ohmic heating power [W]
    public let ohmicPower: Float

    /// Total radiation losses [W] (negative)
    public let radiationPower: Float

    /// Total heating power (input) [W]
    public var totalHeating: Float {
        fusionPower + auxiliaryPower + ohmicPower
    }

    /// Net power (heating - radiation) [W]
    public var netPower: Float {
        totalHeating + radiationPower  // radiationPower is negative
    }

    public init(
        fusionPower: Float,
        alphaPower: Float,
        auxiliaryPower: Float,
        ohmicPower: Float,
        radiationPower: Float
    ) {
        self.fusionPower = fusionPower
        self.alphaPower = alphaPower
        self.auxiliaryPower = auxiliaryPower
        self.ohmicPower = ohmicPower
        self.radiationPower = radiationPower
    }
}

public enum PowerBalanceError: Error, Equatable, CustomStringConvertible {
    case missingSourceMetadata

    public var description: String {
        switch self {
        case .missingSourceMetadata:
            return "SourceTerms.metadata is required for power balance computation"
        }
    }
}

/// Power balance computation from explicit source metadata.
public enum PowerBalanceComputer {

    /// Compute power balance from source terms
    ///
    /// - Parameters:
    ///   - sources: Source terms with metadata
    ///   - profiles: Current plasma profiles
    ///   - geometry: Geometry for volume integration
    ///
    /// - Returns: Power balance with categorized components
    /// - Throws: `PowerBalanceError.missingSourceMetadata` when source metadata is absent,
    ///   or `SourceMetadataValidationError` when metadata powers are not numerically usable.
    public static func compute(
        sources: SourceTerms,
        profiles: CoreProfiles,
        geometry: Geometry
    ) throws -> PowerBalance {
        guard let metadata = sources.metadata else {
            throw PowerBalanceError.missingSourceMetadata
        }

        try metadata.validatePowerAccounting()
        return PowerBalance(
            fusionPower: metadata.fusionPower,
            alphaPower: metadata.alphaPower,
            auxiliaryPower: metadata.auxiliaryPower,
            ohmicPower: metadata.ohmicPower,
            radiationPower: metadata.radiationPower
        )
    }
}
