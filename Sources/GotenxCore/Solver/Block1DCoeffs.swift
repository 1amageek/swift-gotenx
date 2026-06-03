import Foundation
import MLX

/// Block-structured coefficients for coupled 1D transport equations
///
/// Manages coefficients for 4 coupled PDEs representing tokamak core transport:
/// - ionTemperature: Ion temperature (eV)
/// - electronTemperature: Electron temperature (eV)
/// - electronDensity: Electron density (m⁻³)
/// - psi: Poloidal flux (Wb)
///
/// Each equation has its own set of coefficients (EquationCoeffs), allowing
/// for different diffusion, convection, and source terms per variable.
public struct Block1DCoeffs: Sendable {
    /// Coefficients for ion temperature equation
    ///
    /// Equation: n_e ∂T_i/∂t = ∇·(n_e χ_i ∇T_i) + ∇·(n_e V_i T_i) + Q_i - Q_exchange
    public let ionCoeffs: EquationCoeffs

    /// Coefficients for electron temperature equation
    ///
    /// Equation: n_e ∂T_e/∂t = ∇·(n_e χ_e ∇T_e) + ∇·(n_e V_e T_e) + Q_e + Q_exchange + Q_ohmic
    public let electronCoeffs: EquationCoeffs

    /// Coefficients for electron density equation
    ///
    /// Equation: ∂n_e/∂t = ∇·(D ∇n_e) + ∇·(V n_e) + S_n
    public let densityCoeffs: EquationCoeffs

    /// Coefficients for poloidal flux equation
    ///
    /// Equation: ∂ψ/∂t = η_∥ j_∥ (Ohm's law)
    public let fluxCoeffs: EquationCoeffs

    /// Geometric factors (shared across all equations)
    public let geometry: GeometricFactors

    /// Create block-structured coefficients
    ///
    /// - Parameters:
    ///   - ionCoeffs: Coefficients for Ti equation
    ///   - electronCoeffs: Coefficients for Te equation
    ///   - densityCoeffs: Coefficients for ne equation
    ///   - fluxCoeffs: Coefficients for psi equation
    ///   - geometry: Geometric factors
    public init(
        ionCoeffs: EquationCoeffs,
        electronCoeffs: EquationCoeffs,
        densityCoeffs: EquationCoeffs,
        fluxCoeffs: EquationCoeffs,
        geometry: GeometricFactors
    ) {
        self.ionCoeffs = ionCoeffs
        self.electronCoeffs = electronCoeffs
        self.densityCoeffs = densityCoeffs
        self.fluxCoeffs = fluxCoeffs
        self.geometry = geometry
    }
}

// MARK: - Geometric Factors

/// Geometric factors for finite volume discretization in 1D cylindrical geometry
///
/// Encapsulates all geometric information needed for spatial discretization:
/// - Cell volumes and face areas
/// - Distances between cell centers
/// - Radial coordinates
/// - Metric tensor components for non-uniform grids
public struct GeometricFactors: Sendable {
    /// Cell volumes [cellCount]
    ///
    /// For cylindrical geometry: V_i = 2π R₀ Δr_i
    /// where R₀ is major radius, Δr_i is radial cell width
    public let cellVolumes: EvaluatedArray

    /// Face areas [faceCount]
    ///
    /// For cylindrical geometry: A_j = 2π R₀
    /// (constant for 1D slab approximation)
    public let faceAreas: EvaluatedArray

    /// Distance between adjacent cell centers [cellCount-1]
    ///
    /// Δx_j = r_{i+1} - r_i (for face j between cells i and i+1)
    public let cellDistances: EvaluatedArray

    /// Radial coordinate at cell centers [cellCount]
    ///
    /// Normalized radial coordinate: r/a where a is minor radius
    public let cellRadii: EvaluatedArray

    /// Radial coordinate at cell faces [faceCount]
    ///
    /// Normalized radial coordinate at faces (boundaries between cells)
    public let faceRadii: EvaluatedArray

    /// Metric tensor component g₀ = √g (Jacobian of flux coordinates) [cellCount]
    ///
    /// For circular geometry: g₀ = F / B_p where F is flux function
    /// Used for flux divergence: ∇·F = (1/√g) ∂(√g·F)/∂ψ
    public let jacobian: EvaluatedArray

    /// Metric tensor component g₁ [cellCount]
    ///
    /// Geometric factor for non-uniform grids
    public let majorRadiusMetric: EvaluatedArray

    /// Metric tensor component g₂ [cellCount]
    ///
    /// Geometric factor for non-uniform grids
    public let shapeMetric: EvaluatedArray

    /// Create geometric factors
    ///
    /// - Parameters:
    ///   - cellVolumes: Cell volumes [cellCount]
    ///   - faceAreas: Face areas [faceCount]
    ///   - cellDistances: Distances between cell centers [cellCount-1]
    ///   - cellRadii: Radial coordinates at cells [cellCount]
    ///   - faceRadii: Radial coordinates at faces [faceCount]
    ///   - jacobian: Metric tensor g₀ (Jacobian) [cellCount]
    ///   - majorRadiusMetric: Metric tensor g₁ [cellCount]
    ///   - shapeMetric: Metric tensor g₂ [cellCount]
    public init(
        cellVolumes: EvaluatedArray,
        faceAreas: EvaluatedArray,
        cellDistances: EvaluatedArray,
        cellRadii: EvaluatedArray,
        faceRadii: EvaluatedArray,
        jacobian: EvaluatedArray,
        majorRadiusMetric: EvaluatedArray,
        shapeMetric: EvaluatedArray
    ) {
        self.cellVolumes = cellVolumes
        self.faceAreas = faceAreas
        self.cellDistances = cellDistances
        self.cellRadii = cellRadii
        self.faceRadii = faceRadii
        self.jacobian = jacobian
        self.majorRadiusMetric = majorRadiusMetric
        self.shapeMetric = shapeMetric
    }

    /// Create geometric factors from Geometry (UNIFORM GRID ONLY)
    ///
    /// **MEDIUM #7 WARNING**: This method assumes **uniform grid spacing**.
    ///
    /// - For uniform grids: Generates correct geometric factors with constant Δr
    /// - For non-uniform grids: **Incorrect** - need to specify actual grid in Geometry
    ///
    /// **Future improvement**: Add explicit grid arrays (faceRadii, cellRadii) to `Geometry` struct
    /// to support non-uniform grids. Current implementation is adequate for:
    /// - Initial development and testing
    /// - Uniform grid configurations
    /// - Simple tokamak geometries
    ///
    /// For production simulations with edge-refined grids, consider:
    /// 1. Adding `faceRadii: EvaluatedArray` to `Geometry` struct
    /// 2. Computing geometric factors from actual grid coordinates
    ///
    /// - Parameter geometry: Tokamak geometry
    /// - Returns: Geometric factors for finite volume discretization
    public static func from(geometry: Geometry) -> GeometricFactors {
        from(geometry: geometry, evaluationMode: .eager)
    }

    package static func from(
        geometry: Geometry,
        evaluationMode: MLXEvaluationMode
    ) -> GeometricFactors {
        let cellCount = geometry.cellCount
        let faceCount = cellCount + 1
        let radialSpacing = geometry.radialSpacing  // Assumes uniform spacing

        // Validate geometry shape consistency
        let radiiShape = geometry.radii.shape[0]
        let fluxSurfaceMetricShape = geometry.fluxSurfaceMetric.shape[0]

        guard radiiShape == cellCount else {
            fatalError("""
                GeometricFactors.from: Geometry.radii shape mismatch.
                Expected radii.shape[0] = \(cellCount) (cellCount)
                Got radii.shape[0] = \(radiiShape)
                This indicates inconsistent Geometry construction.
                """)
        }

        guard fluxSurfaceMetricShape == faceCount else {
            fatalError("""
                GeometricFactors.from: Geometry.fluxSurfaceMetric shape mismatch.
                Expected fluxSurfaceMetric.shape[0] = \(faceCount) (cellCount + 1, face-centered)
                Got fluxSurfaceMetric.shape[0] = \(fluxSurfaceMetricShape)
                This indicates incorrect Geometry construction.
                Use createGeometry(from:) or Geometry(config:) to ensure correct shapes.
                """)
        }

        // Use existing radii from geometry (ensures consistency)
        let cellRadii = geometry.radii.value  // [cellCount]

        // Face radii: uniformly spaced from 0 to minorRadius
        let faceRadii = MLXArray(0..<faceCount).asType(.float32) * radialSpacing  // [faceCount]

        // Cell volumes (2π R₀ Δr for cylindrical geometry)
        let volumeValue: Float = 2.0 * Float.pi * geometry.majorRadius * radialSpacing
        let cellVolumes = MLXArray.full([cellCount], values: MLXArray(volumeValue))

        // Face areas (2π R₀ - constant)
        let areaValue: Float = 2.0 * Float.pi * geometry.majorRadius
        let faceAreas = MLXArray.full([faceCount], values: MLXArray(areaValue))

        // Cell distances (uniform grid: all equal to radialSpacing)
        let cellDistances = MLXArray.full([cellCount - 1], values: MLXArray(radialSpacing))

        // Metric tensor components from geometry
        // ALL metric tensors (fluxSurfaceMetric, majorRadiusMetric, shapeMetric) are face-centered [faceCount]
        // Convert to cell-centered [cellCount] using arithmetic average

        // Validate shapes first
        guard geometry.fluxSurfaceMetric.value.shape[0] == faceCount else {
            fatalError("GeometricFactors.from: fluxSurfaceMetric shape mismatch. Expected \(faceCount) (faceCount), got \(geometry.fluxSurfaceMetric.value.shape[0])")
        }
        guard geometry.majorRadiusMetric.value.shape[0] == faceCount else {
            fatalError("GeometricFactors.from: majorRadiusMetric shape mismatch. Expected \(faceCount) (faceCount), got \(geometry.majorRadiusMetric.value.shape[0])")
        }
        guard geometry.shapeMetric.value.shape[0] == faceCount else {
            fatalError("GeometricFactors.from: shapeMetric shape mismatch. Expected \(faceCount) (faceCount), got \(geometry.shapeMetric.value.shape[0])")
        }

        // Use a 1D cylindrical approximation for the Jacobian.
        // In 1D cylindrical coordinates: √g = 2πR₀ (constant)
        // The original implementation used fluxSurfaceMetric = (R₀ + r)², which varies with r
        // This caused flux divergence to become O(10³⁰), leading to solver failure
        //
        // Physical justification for constant Jacobian:
        // - Large aspect ratio: R₀ >> a → R₀ + r ≈ R₀
        // - 1D reduction: Toroidal variation averaged out → √g = 2πR₀
        //
        // Alternative (if full toroidal geometry needed):
        // - Use 2D (r, θ) solver instead of 1D approximation
        let jacobianValue = 2.0 * Float.pi * geometry.majorRadius
        let jacobian = MLXArray.full([cellCount], values: MLXArray(jacobianValue))

        // Keep majorRadiusMetric, shapeMetric from geometry (not critical for 1D cylindrical)
        let majorRadiusMetricFaces = geometry.majorRadiusMetric.value
        let shapeMetricFaces = geometry.shapeMetric.value

        let majorRadiusMetric = 0.5 * (majorRadiusMetricFaces[0..<cellCount] + majorRadiusMetricFaces[1..<(cellCount+1)])
        let shapeMetric = 0.5 * (shapeMetricFaces[0..<cellCount] + shapeMetricFaces[1..<(cellCount+1)])

        let wrapped = evaluationMode.wrapBatch([
            cellVolumes,
            faceAreas,
            cellDistances,
            cellRadii,
            faceRadii,
            jacobian,
            majorRadiusMetric,
            shapeMetric
        ])

        return GeometricFactors(
            cellVolumes: wrapped[0],
            faceAreas: wrapped[1],
            cellDistances: wrapped[2],
            cellRadii: wrapped[3],
            faceRadii: wrapped[4],
            jacobian: wrapped[5],
            majorRadiusMetric: wrapped[6],
            shapeMetric: wrapped[7]
        )
    }
}

// MARK: - Validation

extension Block1DCoeffs {
    /// Validate all coefficient shapes for consistency
    ///
    /// - Throws: ValidationError if any coefficient has inconsistent shape
    public func validate() throws {
        let cellCount = geometry.cellRadii.value.shape[0]

        try ionCoeffs.validate(cellCount: cellCount)
        try electronCoeffs.validate(cellCount: cellCount)
        try densityCoeffs.validate(cellCount: cellCount)
        try fluxCoeffs.validate(cellCount: cellCount)

        // Validate geometry
        let faceCount = cellCount + 1

        guard geometry.cellVolumes.value.shape[0] == cellCount else {
            throw ValidationError.inconsistentShape(
                field: "geometry.cellVolumes",
                expected: [cellCount],
                actual: geometry.cellVolumes.value.shape
            )
        }

        guard geometry.faceAreas.value.shape[0] == faceCount else {
            throw ValidationError.inconsistentShape(
                field: "geometry.faceAreas",
                expected: [faceCount],
                actual: geometry.faceAreas.value.shape
            )
        }

        guard geometry.cellDistances.value.shape[0] == cellCount - 1 else {
            throw ValidationError.inconsistentShape(
                field: "geometry.cellDistances",
                expected: [cellCount - 1],
                actual: geometry.cellDistances.value.shape
            )
        }
    }

    public func validateNumerics() throws {
        try validate()

        let cellCount = geometry.cellRadii.value.shape[0]
        try ionCoeffs.validateNumerics(cellCount: cellCount, name: "ionCoeffs")
        try electronCoeffs.validateNumerics(cellCount: cellCount, name: "electronCoeffs")
        try densityCoeffs.validateNumerics(cellCount: cellCount, name: "densityCoeffs")
        try fluxCoeffs.validateNumerics(cellCount: cellCount, name: "fluxCoeffs")

        try NumericalValidation.validate([
            .positive(geometry.cellVolumes.value, field: "geometry.cellVolumes"),
            .positive(geometry.faceAreas.value, field: "geometry.faceAreas"),
            .positive(geometry.cellDistances.value, field: "geometry.cellDistances"),
            .finite(geometry.cellRadii.value, field: "geometry.cellRadii"),
            .finite(geometry.faceRadii.value, field: "geometry.faceRadii"),
            .positive(geometry.jacobian.value, field: "geometry.jacobian"),
            .finite(geometry.majorRadiusMetric.value, field: "geometry.majorRadiusMetric"),
            .finite(geometry.shapeMetric.value, field: "geometry.shapeMetric")
        ])
    }
}
