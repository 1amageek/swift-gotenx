import Foundation
import MLX

/// Coefficients for a single PDE equation in 1D finite volume discretization
///
/// For equation: ∂ψ/∂t = ∇·(d∇ψ) + ∇·(vψ) + s + s_mat·ψ
///
/// This structure encapsulates all spatial and source coefficients needed
/// to discretize a single transport equation on a 1D radial grid.
public struct EquationCoeffs: Sendable {
    /// Diffusion coefficient at cell faces [faceCount]
    ///
    /// Units depend on equation:
    /// - Ti, electronTemperature: m²/s (thermal diffusivity)
    /// - electronDensity: m²/s (particle diffusivity)
    /// - psi: Wb·m/s (magnetic diffusivity)
    public let faceDiffusionCoefficient: EvaluatedArray

    /// Convection velocity at cell faces [faceCount]
    ///
    /// Units depend on equation:
    /// - Ti, electronTemperature: m/s (heat convection)
    /// - electronDensity: m/s (particle convection)
    /// - psi: Wb/s (poloidal flux convection)
    public let faceConvectionVelocity: EvaluatedArray

    /// Source term in cells [cellCount]
    ///
    /// Units depend on equation:
    /// - Ti, electronTemperature: W/m³ (heating power density)
    /// - electronDensity: particles/m³/s (particle source rate)
    /// - psi: A/m² (current density)
    public let cellSource: EvaluatedArray

    /// Source matrix coefficient in cells [cellCount]
    ///
    /// Matrix term for equation coupling: s_mat·ψ
    ///
    /// Examples:
    /// - Ion-electron energy exchange: Q_exchange(Ti, Te)
    /// - Particle-energy coupling: ionization/recombination terms
    public let cellSourceMatrixCoefficient: EvaluatedArray

    /// Transient coefficient for time stepping [cellCount]
    ///
    /// Coefficient multiplying ∂ψ/∂t term.
    ///
    /// Examples:
    /// - Temperature: n_e (density)
    /// - Density: 1.0 (continuity)
    /// - Flux: L_p (poloidal inductance)
    public let transientCoefficient: EvaluatedArray

    /// Create equation coefficients
    ///
    /// - Parameters:
    ///   - faceDiffusionCoefficient: Diffusion coefficient at faces [faceCount]
    ///   - faceConvectionVelocity: Convection velocity at faces [faceCount]
    ///   - cellSource: Source term in cells [cellCount]
    ///   - cellSourceMatrixCoefficient: Source matrix coefficient in cells [cellCount]
    ///   - transientCoefficient: Transient coefficient in cells [cellCount]
    public init(
        faceDiffusionCoefficient: EvaluatedArray,
        faceConvectionVelocity: EvaluatedArray,
        cellSource: EvaluatedArray,
        cellSourceMatrixCoefficient: EvaluatedArray,
        transientCoefficient: EvaluatedArray
    ) {
        self.faceDiffusionCoefficient = faceDiffusionCoefficient
        self.faceConvectionVelocity = faceConvectionVelocity
        self.cellSource = cellSource
        self.cellSourceMatrixCoefficient = cellSourceMatrixCoefficient
        self.transientCoefficient = transientCoefficient
    }

    public init(
        faceDiffusionCoefficient: MLXArray,
        faceConvectionVelocity: MLXArray,
        cellSource: MLXArray,
        cellSourceMatrixCoefficient: MLXArray,
        transientCoefficient: MLXArray
    ) {
        self.init(
            faceDiffusionCoefficient: faceDiffusionCoefficient,
            faceConvectionVelocity: faceConvectionVelocity,
            cellSource: cellSource,
            cellSourceMatrixCoefficient: cellSourceMatrixCoefficient,
            transientCoefficient: transientCoefficient,
            evaluationMode: .eager
        )
    }

    package init(
        faceDiffusionCoefficient: MLXArray,
        faceConvectionVelocity: MLXArray,
        cellSource: MLXArray,
        cellSourceMatrixCoefficient: MLXArray,
        transientCoefficient: MLXArray,
        evaluationMode: MLXEvaluationMode
    ) {
        let evaluated = evaluationMode.wrapBatch([
            faceDiffusionCoefficient,
            faceConvectionVelocity,
            cellSource,
            cellSourceMatrixCoefficient,
            transientCoefficient
        ])
        self.init(
            faceDiffusionCoefficient: evaluated[0],
            faceConvectionVelocity: evaluated[1],
            cellSource: evaluated[2],
            cellSourceMatrixCoefficient: evaluated[3],
            transientCoefficient: evaluated[4]
        )
    }
}

// MARK: - Validation

extension EquationCoeffs {
    /// Validate coefficient shapes for consistency
    ///
    /// - Parameter cellCount: Expected number of cells
    /// - Throws: ValidationError if shapes are inconsistent
    public func validate(cellCount: Int) throws {
        let faceCount = cellCount + 1

        guard faceDiffusionCoefficient.value.shape[0] == faceCount else {
            throw ValidationError.inconsistentShape(
                field: "faceDiffusionCoefficient",
                expected: [faceCount],
                actual: faceDiffusionCoefficient.value.shape
            )
        }

        guard faceConvectionVelocity.value.shape[0] == faceCount else {
            throw ValidationError.inconsistentShape(
                field: "faceConvectionVelocity",
                expected: [faceCount],
                actual: faceConvectionVelocity.value.shape
            )
        }

        guard cellSource.value.shape[0] == cellCount else {
            throw ValidationError.inconsistentShape(
                field: "cellSource",
                expected: [cellCount],
                actual: cellSource.value.shape
            )
        }

        guard cellSourceMatrixCoefficient.value.shape[0] == cellCount else {
            throw ValidationError.inconsistentShape(
                field: "cellSourceMatrixCoefficient",
                expected: [cellCount],
                actual: cellSourceMatrixCoefficient.value.shape
            )
        }

        guard transientCoefficient.value.shape[0] == cellCount else {
            throw ValidationError.inconsistentShape(
                field: "transientCoefficient",
                expected: [cellCount],
                actual: transientCoefficient.value.shape
            )
        }
    }

    public func validateNumerics(cellCount: Int, name: String) throws {
        try validate(cellCount: cellCount)

        try NumericalValidation.validate([
            .nonNegative(faceDiffusionCoefficient.value, field: "\(name).faceDiffusionCoefficient"),
            .finite(faceConvectionVelocity.value, field: "\(name).faceConvectionVelocity"),
            .finite(cellSource.value, field: "\(name).cellSource"),
            .finite(cellSourceMatrixCoefficient.value, field: "\(name).cellSourceMatrixCoefficient"),
            .positive(transientCoefficient.value, field: "\(name).transientCoefficient")
        ])
    }
}

// MARK: - Validation Error

public enum ValidationError: Error, CustomStringConvertible {
    case inconsistentShape(field: String, expected: [Int], actual: [Int])
    case invalidValue(field: String, reason: String)

    public var description: String {
        switch self {
        case .inconsistentShape(let field, let expected, let actual):
            return "Inconsistent shape for \(field): expected \(expected), got \(actual)"
        case .invalidValue(let field, let reason):
            return "Invalid value for \(field): \(reason)"
        }
    }
}
