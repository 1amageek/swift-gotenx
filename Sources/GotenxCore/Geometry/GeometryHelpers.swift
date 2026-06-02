import MLX
import Foundation

// MARK: - Geometry Computation Helpers

/// Compute plasma volume from mesh configuration
///
/// For circular cross-section: V = 2π²R·a²
///
/// - Parameter mesh: Mesh configuration
/// - Returns: Lazy MLXArray (caller wraps in EvaluatedArray)
public func computeVolume(_ mesh: MeshConfig) -> MLXArray {
    let rMajor = MLXArray(mesh.majorRadius)
    let rMinor = MLXArray(mesh.minorRadius)

    // V = 2π²R·a² for circular cross-section
    return 2.0 * Float.pi * Float.pi * rMajor * rMinor * rMinor
}

/// Compute geometric coefficient fluxSurfaceMetric for FVM
///
/// fluxSurfaceMetric = (R0 + r·cos(θ))² for circular geometry
///
/// - Parameter mesh: Mesh configuration
/// - Returns: Lazy MLXArray of shape [faceCount] (caller wraps in EvaluatedArray)
public func computeFluxSurfaceMetric(_ mesh: MeshConfig) -> MLXArray {
    // Grid points (face-centered)
    let r = MLXArray.linspace(0.0, mesh.minorRadius, count: mesh.cellCount + 1)

    // fluxSurfaceMetric = (R0 + r)² for circular geometry (assuming θ=0)
    let rMajor = MLXArray(mesh.majorRadius)
    return (rMajor + r) * (rMajor + r)
}

/// Compute geometric coefficient majorRadiusMetric for FVM
///
/// majorRadiusMetric = R0 + r·cos(θ) for circular geometry
///
/// - Parameter mesh: Mesh configuration
/// - Returns: Lazy MLXArray of shape [faceCount] (caller wraps in EvaluatedArray)
public func computeMajorRadiusMetric(_ mesh: MeshConfig) -> MLXArray {
    // Grid points (face-centered)
    let r = MLXArray.linspace(0.0, mesh.minorRadius, count: mesh.cellCount + 1)

    // majorRadiusMetric = R0 + r for circular geometry (assuming θ=0)
    let rMajor = MLXArray(mesh.majorRadius)
    return rMajor + r
}

/// Compute geometric coefficient shapeMetric for FVM
///
/// shapeMetric = 1 for circular geometry
///
/// - Parameter mesh: Mesh configuration
/// - Returns: Lazy MLXArray of shape [faceCount] (caller wraps in EvaluatedArray)
public func computeShapeMetric(_ mesh: MeshConfig) -> MLXArray {
    // shapeMetric = 1 for circular geometry
    return MLXArray.ones([mesh.cellCount + 1])
}

/// Compute geometric coefficient minorRadiusMetric for FVM
///
/// minorRadiusMetric = r for circular geometry
///
/// - Parameter mesh: Mesh configuration
/// - Returns: Lazy MLXArray of shape [faceCount] (caller wraps in EvaluatedArray)
public func computeMinorRadiusMetric(_ mesh: MeshConfig) -> MLXArray {
    // Grid points (face-centered)
    let r = MLXArray.linspace(0.0, mesh.minorRadius, count: mesh.cellCount + 1)

    // minorRadiusMetric = r for circular geometry
    return r
}

// MARK: - Safety Factor Computation

/// Compute safety factor profile for circular geometry
///
/// Simple parametric model: q(r) = q₀ + (q_edge - q₀) * (r/a)^α
///
/// - Parameters:
///   - mesh: Mesh configuration
///   - axisSafetyFactor: Safety factor at axis (default: 1.0)
///   - edgeSafetyFactor: Safety factor at edge (default: 3.5)
///   - alpha: Profile shape parameter (default: 2.0 for parabolic)
/// - Returns: Lazy MLXArray of shape [cellCount] (caller wraps in EvaluatedArray)
public func computeSafetyFactor(
    _ mesh: MeshConfig,
    axisSafetyFactor: Float = 1.0,
    edgeSafetyFactor: Float = 3.5,
    alpha: Float = 2.0
) -> MLXArray {
    // Cell-centered radial coordinates
    let radialSpacing = mesh.radialSpacing
    let r = (MLXArray(0..<mesh.cellCount).asType(.float32) + 0.5) * radialSpacing

    // Normalized radius
    let rNorm = r / mesh.minorRadius

    // q(r) = q₀ + (q_edge - q₀) * (r/a)^α
    return axisSafetyFactor + (edgeSafetyFactor - axisSafetyFactor) * pow(rNorm, alpha)
}

/// Compute cell-centered radial coordinates
///
/// - Parameter mesh: Mesh configuration
/// - Returns: Lazy MLXArray of shape [cellCount] (caller wraps in EvaluatedArray)
public func computeRadii(_ mesh: MeshConfig) -> MLXArray {
    let radialSpacing = mesh.radialSpacing
    return (MLXArray(0..<mesh.cellCount).asType(.float32) + 0.5) * radialSpacing
}

// MARK: - Geometry Construction

/// Construct Geometry from mesh configuration
///
/// - Parameters:
///   - mesh: Mesh configuration
///   - axisSafetyFactor: Safety factor at axis (default: 1.0)
///   - edgeSafetyFactor: Safety factor at edge (default: 3.5)
/// - Returns: Geometry with evaluated arrays
public func createGeometry(
    from mesh: MeshConfig,
    axisSafetyFactor: Float = 1.0,
    edgeSafetyFactor: Float = 3.5
) -> Geometry {
    Geometry(
        majorRadius: mesh.majorRadius,
        minorRadius: mesh.minorRadius,
        toroidalField: mesh.toroidalField,
        volume: EvaluatedArray(evaluating: computeVolume(mesh)),
        fluxSurfaceMetric: EvaluatedArray(evaluating: computeFluxSurfaceMetric(mesh)),
        majorRadiusMetric: EvaluatedArray(evaluating: computeMajorRadiusMetric(mesh)),
        shapeMetric: EvaluatedArray(evaluating: computeShapeMetric(mesh)),
        minorRadiusMetric: EvaluatedArray(evaluating: computeMinorRadiusMetric(mesh)),
        radii: EvaluatedArray(evaluating: computeRadii(mesh)),
        safetyFactor: EvaluatedArray(evaluating: computeSafetyFactor(mesh, axisSafetyFactor: axisSafetyFactor, edgeSafetyFactor: edgeSafetyFactor)),
        poloidalField: nil,
        currentDensity: nil,
        type: mesh.geometryType
    )
}

// MARK: - Geometry Provider Implementations

/// Static geometry provider (time-independent)
public struct StaticGeometryProvider: GeometryProvider {
    private let mesh: MeshConfig
    private let geometry: Geometry

    public init(mesh: MeshConfig) {
        self.mesh = mesh
        self.geometry = createGeometry(from: mesh)
    }

    public func geometry(at time: Float) -> Geometry {
        // Same geometry regardless of time
        geometry
    }
}

/// Time-evolving geometry provider
public struct TimeEvolvingGeometryProvider: GeometryProvider {
    private let baseMesh: MeshConfig
    private let scaleProfile: (Float) -> Float

    /// Initialize time-evolving geometry provider
    ///
    /// - Parameters:
    ///   - baseMesh: Base mesh configuration
    ///   - scaleProfile: Time-dependent scaling function
    public init(baseMesh: MeshConfig, scaleProfile: @escaping (Float) -> Float) {
        self.baseMesh = baseMesh
        self.scaleProfile = scaleProfile
    }

    public func geometry(at time: Float) -> Geometry {
        let scale = scaleProfile(time)

        // Scale minor radius and field with time
        let evolvedMesh = MeshConfig(
            cellCount: baseMesh.cellCount,
            majorRadius: baseMesh.majorRadius,
            minorRadius: baseMesh.minorRadius * scale,
            toroidalField: baseMesh.toroidalField / scale,  // Flux conservation
            geometryType: baseMesh.geometryType
        )

        return createGeometry(from: evolvedMesh)
    }
}
