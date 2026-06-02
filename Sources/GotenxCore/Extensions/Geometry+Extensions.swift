import Foundation
import MLX

// MARK: - Geometry Extensions

extension Geometry {
    /// Convenience initializer from MeshConfig
    ///
    /// Creates a Geometry instance by computing all geometric coefficients from
    /// the mesh configuration. This is a convenience wrapper around
    /// `createGeometry(from:)` from GeometryHelpers.swift.
    ///
    /// **Geometric Coefficients (for circular geometry):**
    /// - fluxSurfaceMetric = (R₀ + r)² - flux surface area metric
    /// - majorRadiusMetric = R₀ + r - major radius at flux surface
    /// - shapeMetric = 1 - shape factor (constant for circular)
    /// - minorRadiusMetric = r - minor radius coordinate
    ///
    /// - Parameters:
    ///   - config: Mesh configuration
    ///   - axisSafetyFactor: Safety factor at axis (default: 1.0)
    ///   - edgeSafetyFactor: Safety factor at edge (default: 3.5)
    public init(
        config: MeshConfig,
        axisSafetyFactor: Float = 1.0,
        edgeSafetyFactor: Float = 3.5
    ) {
        let geometry = createGeometry(from: config, axisSafetyFactor: axisSafetyFactor, edgeSafetyFactor: edgeSafetyFactor)
        self.init(
            majorRadius: geometry.majorRadius,
            minorRadius: geometry.minorRadius,
            toroidalField: geometry.toroidalField,
            volume: geometry.volume,
            fluxSurfaceMetric: geometry.fluxSurfaceMetric,
            majorRadiusMetric: geometry.majorRadiusMetric,
            shapeMetric: geometry.shapeMetric,
            minorRadiusMetric: geometry.minorRadiusMetric,
            radii: geometry.radii,
            safetyFactor: geometry.safetyFactor,
            poloidalField: geometry.poloidalField,
            currentDensity: geometry.currentDensity,
            type: geometry.type
        )
    }

    /// Number of radial cells
    ///
    /// Derived from fluxSurfaceMetric shape. fluxSurfaceMetric is defined on cell faces, so:
    /// - fluxSurfaceMetric.shape = [faceCount]
    /// - faceCount = cellCount + 1
    /// - Therefore: cellCount = fluxSurfaceMetric.shape[0] - 1
    public var cellCount: Int {
        // fluxSurfaceMetric is on faces (boundaries between cells)
        let faceCount = fluxSurfaceMetric.value.shape[0]
        return faceCount - 1
    }

    /// Radial grid spacing (for uniform grids only)
    ///
    /// **MEDIUM #7 WARNING**: This property assumes **uniform grid spacing**.
    ///
    /// - For uniform grids: Returns constant Δr = a / cellCount
    /// - For non-uniform grids: **This is incorrect** - use GeometricFactors.cellDistances instead
    ///
    /// **Recommendation**: Avoid using this property. Instead:
    /// 1. Use `GeometricFactors.cellDistances` for actual cell-to-cell distances
    /// 2. Use `GeometricFactors.cellRadii` and `GeometricFactors.faceRadii` for radial coordinates
    ///
    /// This property is kept for backward compatibility with uniform grid configurations,
    /// but should be deprecated in favor of explicit grid specification in `Geometry`.
    public var radialSpacing: Float {
        guard cellCount > 0 else { return 0.0 }
        return minorRadius / Float(cellCount)
    }
}
