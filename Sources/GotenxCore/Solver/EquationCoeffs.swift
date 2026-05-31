import Foundation
import MLX

/// Coefficients for a single PDE equation in 1D finite volume discretization
///
/// For equation: ∂ψ/∂t = ∇·(d∇ψ) + ∇·(vψ) + s + s_mat·ψ
///
/// This structure encapsulates all spatial and source coefficients needed
/// to discretize a single transport equation on a 1D radial grid.
public struct EquationCoeffs: Sendable {
    /// Diffusion coefficient at cell faces [nFaces]
    ///
    /// Units depend on equation:
    /// - Ti, Te: m²/s (thermal diffusivity)
    /// - ne: m²/s (particle diffusivity)
    /// - psi: Wb·m/s (magnetic diffusivity)
    public let dFace: EvaluatedArray

    /// Convection velocity at cell faces [nFaces]
    ///
    /// Units depend on equation:
    /// - Ti, Te: m/s (heat convection)
    /// - ne: m/s (particle convection)
    /// - psi: Wb/s (poloidal flux convection)
    public let vFace: EvaluatedArray

    /// Source term in cells [nCells]
    ///
    /// Units depend on equation:
    /// - Ti, Te: W/m³ (heating power density)
    /// - ne: particles/m³/s (particle source rate)
    /// - psi: A/m² (current density)
    public let sourceCell: EvaluatedArray

    /// Source matrix coefficient in cells [nCells]
    ///
    /// Matrix term for equation coupling: s_mat·ψ
    ///
    /// Examples:
    /// - Ion-electron energy exchange: Q_exchange(Ti, Te)
    /// - Particle-energy coupling: ionization/recombination terms
    public let sourceMatCell: EvaluatedArray

    /// Transient coefficient for time stepping [nCells]
    ///
    /// Coefficient multiplying ∂ψ/∂t term.
    ///
    /// Examples:
    /// - Temperature: n_e (density)
    /// - Density: 1.0 (continuity)
    /// - Flux: L_p (poloidal inductance)
    public let transientCoeff: EvaluatedArray

    /// Create equation coefficients
    ///
    /// - Parameters:
    ///   - dFace: Diffusion coefficient at faces [nFaces]
    ///   - vFace: Convection velocity at faces [nFaces]
    ///   - sourceCell: Source term in cells [nCells]
    ///   - sourceMatCell: Source matrix coefficient in cells [nCells]
    ///   - transientCoeff: Transient coefficient in cells [nCells]
    public init(
        dFace: EvaluatedArray,
        vFace: EvaluatedArray,
        sourceCell: EvaluatedArray,
        sourceMatCell: EvaluatedArray,
        transientCoeff: EvaluatedArray
    ) {
        self.dFace = dFace
        self.vFace = vFace
        self.sourceCell = sourceCell
        self.sourceMatCell = sourceMatCell
        self.transientCoeff = transientCoeff
    }

    public init(
        evaluatingDface dFace: MLXArray,
        vFace: MLXArray,
        sourceCell: MLXArray,
        sourceMatCell: MLXArray,
        transientCoeff: MLXArray
    ) {
        let evaluated = EvaluatedArray.evaluatingBatch([
            dFace,
            vFace,
            sourceCell,
            sourceMatCell,
            transientCoeff
        ])
        self.init(
            dFace: evaluated[0],
            vFace: evaluated[1],
            sourceCell: evaluated[2],
            sourceMatCell: evaluated[3],
            transientCoeff: evaluated[4]
        )
    }
}

// MARK: - Validation

extension EquationCoeffs {
    /// Validate coefficient shapes for consistency
    ///
    /// - Parameter nCells: Expected number of cells
    /// - Throws: ValidationError if shapes are inconsistent
    public func validate(nCells: Int) throws {
        let nFaces = nCells + 1

        guard dFace.value.shape[0] == nFaces else {
            throw ValidationError.inconsistentShape(
                field: "dFace",
                expected: [nFaces],
                actual: dFace.value.shape
            )
        }

        guard vFace.value.shape[0] == nFaces else {
            throw ValidationError.inconsistentShape(
                field: "vFace",
                expected: [nFaces],
                actual: vFace.value.shape
            )
        }

        guard sourceCell.value.shape[0] == nCells else {
            throw ValidationError.inconsistentShape(
                field: "sourceCell",
                expected: [nCells],
                actual: sourceCell.value.shape
            )
        }

        guard sourceMatCell.value.shape[0] == nCells else {
            throw ValidationError.inconsistentShape(
                field: "sourceMatCell",
                expected: [nCells],
                actual: sourceMatCell.value.shape
            )
        }

        guard transientCoeff.value.shape[0] == nCells else {
            throw ValidationError.inconsistentShape(
                field: "transientCoeff",
                expected: [nCells],
                actual: transientCoeff.value.shape
            )
        }
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
