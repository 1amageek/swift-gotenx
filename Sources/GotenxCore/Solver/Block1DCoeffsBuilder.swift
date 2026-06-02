import MLX
import Foundation

// MARK: - Block 1D Coefficients Builder

/// Build block-structured coefficients from physics models
///
/// Constructs per-equation coefficients for the 4 coupled transport equations in **non-conservation form**:
///
/// - Ion temperature: n_e ∂T_i/∂t = ∇·(n_e χ_i ∇T_i) + ∇·(n_e V_i T_i) + Q_i - Q_exchange
/// - Electron temperature: n_e ∂T_e/∂t = ∇·(n_e χ_e ∇T_e) + ∇·(n_e V_e T_e) + Q_e + Q_exchange + Q_ohmic
/// - Electron density: ∂n_e/∂t = ∇·(D ∇n_e) + ∇·(V n_e) + S_n
/// - Poloidal flux: ∂ψ/∂t = η_∥ j_∥ (from Ohm's law)
///
/// **Implementation Note (Conservation Form):**
/// This implementation uses **non-conservation form** (following Python TORAX), where the time derivative
/// is `n_e ∂T_i/∂t` rather than the conservation form `∂(n_e T_i)/∂t`.
///
/// - Conservation form: ∂(n_e T_i)/∂t = ∇·(n_e χ_i ∇T_i) + Q_i
///   - Expands to: n_e ∂T_i/∂t + T_i ∂n_e/∂t = ∇·(n_e χ_i ∇T_i) + Q_i
///   - **Pro**: Better energy conservation when density changes rapidly (pellets, gas puff)
///   - **Con**: More complex to implement
///
/// - Non-conservation form: n_e ∂T_i/∂t = ∇·(n_e χ_i ∇T_i) + Q_i (current implementation)
///   - **Pro**: Simpler, matches Python TORAX, adequate for slow density evolution
///   - **Con**: May have small energy conservation errors during rapid density changes
///
/// The `transientCoefficient` field in `EquationCoeffs` contains n_e(r) to properly weight the time derivative.
///
/// - Parameters:
///   - transport: Transport coefficients (chi, D, V)
///   - sources: Source terms (heating, particles, current)
///   - geometry: Tokamak geometry
///   - staticParameters: Static runtime parameters
///   - profiles: Current core profiles used for spatial density weighting
/// - Returns: Block coefficients with per-equation structure
public func buildBlock1DCoeffs(
    transport: TransportCoefficients,
    sources: SourceTerms,
    geometry: Geometry,
    staticParameters: StaticRuntimeParameters,
    profiles: CoreProfiles
) -> Block1DCoeffs {
    // Build geometric factors (shared across equations)
    let geoFactors = GeometricFactors.from(geometry: geometry)

    // Build per-equation coefficients (with actual profiles)
    let ionCoeffs = buildIonEquationCoeffs(
        transport: transport,
        sources: sources,
        geometry: geometry,
        staticParameters: staticParameters,
        profiles: profiles
    )

    let electronCoeffs = buildElectronEquationCoeffs(
        transport: transport,
        sources: sources,
        geometry: geometry,
        staticParameters: staticParameters,
        profiles: profiles
    )

    let densityCoeffs = buildDensityEquationCoeffs(
        transport: transport,
        sources: sources,
        geometry: geometry,
        staticParameters: staticParameters,
        profiles: profiles
    )

    let fluxCoeffs = buildFluxEquationCoeffs(
        transport: transport,
        sources: sources,
        geometry: geometry,
        staticParameters: staticParameters,
        profiles: profiles
    )

    return Block1DCoeffs(
        ionCoeffs: ionCoeffs,
        electronCoeffs: electronCoeffs,
        densityCoeffs: densityCoeffs,
        fluxCoeffs: fluxCoeffs,
        geometry: geoFactors
    )
}

// MARK: - Per-Equation Coefficient Builders

/// Build coefficients for ion temperature equation
///
/// Equation: n_e ∂T_i/∂t = ∇·(n_e χ_i ∇T_i) + ∇·(n_e V_i T_i) + Q_i - Q_exchange
///
/// - Returns: Coefficients for Ti equation
private func buildIonEquationCoeffs(
    transport: TransportCoefficients,
    sources: SourceTerms,
    geometry: Geometry,
    staticParameters: StaticRuntimeParameters,
    profiles: CoreProfiles
) -> EquationCoeffs {
    let cellCount = geometry.cellCount
    let faceCount = cellCount + 1

    // Interpolate transport coefficients to faces
    let ionHeatDiffusivityFaces = interpolateToFaces(transport.ionHeatDiffusivity.value, mode: .harmonic)  // [faceCount]

    // Use the actual density profile with a physical floor.
    // The floor prevents division by zero in non-conservation form (dT/timeStep = rhs / n_e).
    let ne_floor: Float = 1e18  // [m⁻³]
    let ne_cell = maximum(profiles.electronDensity.value, MLXArray(ne_floor))  // [cellCount] - actual spatial profile with floor
    let ne_face = interpolateToFaces(ne_cell, mode: .harmonic)  // [faceCount]

    // Diffusion coefficient: d = n_e * χ_i (with spatial variation!)
    let faceDiffusionCoefficient = ionHeatDiffusivityFaces * ne_face  // [faceCount]

    // Convection velocity: v = n_e * V_i
    // For now, assume no ion heat convection (V_i = 0)
    let faceConvectionVelocity = MLXArray.zeros([faceCount])  // [faceCount]

    // Source term: Q_i - Q_exchange
    // SourceTerms provides heating in [MW/m³].
    // Temperature equation requires [eV/(m³·s)] to match left side: n_e ∂T_i/∂t [eV/(m³·s)]
    //
    // Dimensional analysis:
    //   Left side:  [m⁻³] × [eV/s] = [eV/(m³·s)]
    //   Diffusion:  ∇·([m⁻³] × [m²/s] × [eV/m]) = [eV/(m³·s)]
    //   Source:     Must be [eV/(m³·s)]
    //
    // Conversion: 1 MW/m³ = 6.2415090744×10²⁴ eV/(m³·s)
    let cellSource = UnitConversions.megawattsToElectronVoltDensity(sources.ionHeating.value)  // [eV/(m³·s)]
    // Q_exchange is implicit coupling term (handled via cellSourceMatrixCoefficient)

    // Source matrix coefficient: -Q_exchange coupling
    // For now, assume decoupled (explicit exchange in source)
    let cellSourceMatrixCoefficient = MLXArray.zeros([cellCount])  // [cellCount]

    // Transient coefficient: n_e (with spatial variation!)
    let transientCoefficient = ne_cell  // [cellCount] - actual density profile

    return EquationCoeffs(
        faceDiffusionCoefficient: faceDiffusionCoefficient,
        faceConvectionVelocity: faceConvectionVelocity,
        cellSource: cellSource,
        cellSourceMatrixCoefficient: cellSourceMatrixCoefficient,
        transientCoefficient: transientCoefficient
    )
}

/// Build coefficients for electron temperature equation
///
/// Equation: n_e ∂T_e/∂t = ∇·(n_e χ_e ∇T_e) + ∇·(n_e V_e T_e) + Q_e + Q_exchange + Q_ohmic
///
/// - Returns: Coefficients for Te equation
private func buildElectronEquationCoeffs(
    transport: TransportCoefficients,
    sources: SourceTerms,
    geometry: Geometry,
    staticParameters: StaticRuntimeParameters,
    profiles: CoreProfiles
) -> EquationCoeffs {
    let cellCount = geometry.cellCount
    let faceCount = cellCount + 1

    // Interpolate transport coefficients to faces
    let electronHeatDiffusivityFaces = interpolateToFaces(transport.electronHeatDiffusivity.value, mode: .harmonic)  // [faceCount]

    // Use the actual density profile with a physical floor.
    // The floor prevents division by zero in non-conservation form (dT/timeStep = rhs / n_e).
    let ne_floor: Float = 1e18  // [m⁻³]
    let ne_cell = maximum(profiles.electronDensity.value, MLXArray(ne_floor))  // [cellCount] - actual spatial profile with floor
    let ne_face = interpolateToFaces(ne_cell, mode: .harmonic)  // [faceCount]

    // Diffusion coefficient: d = n_e * χ_e (with spatial variation!)
    let faceDiffusionCoefficient = electronHeatDiffusivityFaces * ne_face  // [faceCount]

    // Convection velocity: v = n_e * V_e
    // For now, assume no electron heat convection (V_e = 0)
    let faceConvectionVelocity = MLXArray.zeros([faceCount])  // [faceCount]

    // Source term: Q_e + Q_ohmic (Q_exchange handled via coupling)
    // SourceTerms provides heating in [MW/m³].
    // Temperature equation requires [eV/(m³·s)] to match left side: n_e ∂T_e/∂t [eV/(m³·s)]
    //
    // Dimensional analysis:
    //   Left side:  [m⁻³] × [eV/s] = [eV/(m³·s)]
    //   Diffusion:  ∇·([m⁻³] × [m²/s] × [eV/m]) = [eV/(m³·s)]
    //   Source:     Must be [eV/(m³·s)]
    //
    // Conversion: 1 MW/m³ = 6.2415090744×10²⁴ eV/(m³·s)
    let cellSource = UnitConversions.megawattsToElectronVoltDensity(sources.electronHeating.value)  // [eV/(m³·s)]

    // Source matrix coefficient
    let cellSourceMatrixCoefficient = MLXArray.zeros([cellCount])  // [cellCount]

    // Transient coefficient: n_e (with spatial variation!)
    let transientCoefficient = ne_cell  // [cellCount] - actual density profile

    return EquationCoeffs(
        faceDiffusionCoefficient: faceDiffusionCoefficient,
        faceConvectionVelocity: faceConvectionVelocity,
        cellSource: cellSource,
        cellSourceMatrixCoefficient: cellSourceMatrixCoefficient,
        transientCoefficient: transientCoefficient
    )
}

/// Build coefficients for electron density equation
///
/// Equation: ∂n_e/∂t = ∇·(D ∇n_e) + ∇·(V n_e) + S_n
///
/// - Returns: Coefficients for ne equation
private func buildDensityEquationCoeffs(
    transport: TransportCoefficients,
    sources: SourceTerms,
    geometry: Geometry,
    staticParameters: StaticRuntimeParameters,
    profiles: CoreProfiles
) -> EquationCoeffs {
    let cellCount = geometry.cellCount

    // Interpolate particle diffusivity to faces
    let DFaces = interpolateToFaces(transport.particleDiffusivity.value, mode: .harmonic)  // [faceCount]

    // Diffusion coefficient
    let faceDiffusionCoefficient = DFaces  // [faceCount]

    // Convection velocity
    let VFaces = interpolateToFaces(transport.convectionVelocity.value, mode: .arithmetic)  // [faceCount]
    let faceConvectionVelocity = VFaces  // [faceCount]

    // Source term
    let cellSource = sources.particleSource.value  // [cellCount]

    // Source matrix coefficient
    let cellSourceMatrixCoefficient = MLXArray.zeros([cellCount])  // [cellCount]

    // Transient coefficient: 1.0 (continuity equation)
    let transientCoefficient = MLXArray.ones([cellCount])  // [cellCount]

    return EquationCoeffs(
        faceDiffusionCoefficient: faceDiffusionCoefficient,
        faceConvectionVelocity: faceConvectionVelocity,
        cellSource: cellSource,
        cellSourceMatrixCoefficient: cellSourceMatrixCoefficient,
        transientCoefficient: transientCoefficient
    )
}

/// Build coefficients for poloidal flux equation
///
/// Equation: ∂ψ/∂t = η_∥ j_∥ (from Ohm's law)
///
/// This can be rewritten as a diffusion-like equation with current sources.
///
/// **Implementation Update**: Now uses temperature-dependent Spitzer resistivity
/// and bootstrap current from pressure gradients.
///
/// - Returns: Coefficients for psi equation
private func buildFluxEquationCoeffs(
    transport: TransportCoefficients,
    sources: SourceTerms,
    geometry: Geometry,
    staticParameters: StaticRuntimeParameters,
    profiles: CoreProfiles
) -> EquationCoeffs {
    let cellCount = geometry.cellCount
    let faceCount = cellCount + 1

    // 1. Temperature-dependent resistivity (Spitzer formula with neoclassical correction)
    let eta_cell = computeSpitzerResistivity(
        electronTemperature: profiles.electronTemperature.value,
        geometry: geometry
    )
    let faceDiffusionCoefficient = interpolateToFaces(eta_cell, mode: .harmonic)  // [faceCount]

    // No convection for flux
    let faceConvectionVelocity = MLXArray.zeros([faceCount])  // [faceCount]

    // 2. Bootstrap current from pressure gradients
    let J_bootstrap = computeBootstrapCurrent(
        profiles: profiles,
        geometry: geometry
    )

    // 3. Total current source: bootstrap + external
    // Note: J_bootstrap is in A/m², J_external is in MA/m²
    // Convert J_bootstrap to MA/m² before adding
    let J_external = sources.currentSource.value  // [MA/m²]
    let cellSource = J_bootstrap / 1e6 + J_external  // [MA/m²]

    // Source matrix coefficient
    let cellSourceMatrixCoefficient = MLXArray.zeros([cellCount])  // [cellCount]

    // Transient coefficient: L_p (poloidal inductance)
    // For simplicity, use 1.0 (properly should be μ₀ R₀)
    let transientCoefficient = MLXArray.ones([cellCount])  // [cellCount]

    return EquationCoeffs(
        faceDiffusionCoefficient: faceDiffusionCoefficient,
        faceConvectionVelocity: faceConvectionVelocity,
        cellSource: cellSource,
        cellSourceMatrixCoefficient: cellSourceMatrixCoefficient,
        transientCoefficient: transientCoefficient
    )
}

// MARK: - Interpolation Helpers

/// Interpolation mode for cell-to-face conversion
///
/// **Design Decision**: Different interpolation methods for different physical quantities
///
/// - `arithmetic`: Simple average (a + b) / 2
///   - Used for: convection velocity (standard central differencing)
///   - LinearSolver uses this for variable interpolation
///
/// - `harmonic`: Harmonic mean 2ab / (a + b) = 2 / (1/a + 1/b)
///   - Used for: transport coefficients (χ, D) and electron density in coefficients
///   - Preserves flux continuity across cell boundaries
///   - Reciprocal form prevents Float32 overflow for large values (n_e ~ 1e20)
///
/// See IMPLEMENTATION_NOTES.md Section 1 for detailed rationale.
private enum InterpolationMode {
    case arithmetic  // Simple average: (a + b) / 2
    case harmonic    // Harmonic mean: 2ab / (a + b) - preserves flux continuity
}

/// Interpolate cell-centered values to faces
///
/// - Parameters:
///   - cellValues: Values at cell centers [cellCount]
///   - mode: Interpolation mode (arithmetic or harmonic)
/// - Returns: Values at cell faces [faceCount]
private func interpolateToFaces(_ cellValues: MLXArray, mode: InterpolationMode) -> MLXArray {
    let values = stopGradient(cellValues)
    let cellCount = values.shape[0]

    // Interior faces
    let leftCells = values[0..<(cellCount - 1)]   // [cellCount-1]
    let rightCells = values[1..<cellCount]        // [cellCount-1]

    let interiorFaces: MLXArray
    switch mode {
    case .arithmetic:
        // Simple average
        interiorFaces = (leftCells + rightCells) / 2.0  // [cellCount-1]

    case .harmonic:
        // Harmonic mean in reciprocal form avoids Float32 overflow when
        // densities are around 1e20 m^-3.
        let reciprocalSum = 1.0 / (leftCells + 1e-30) + 1.0 / (rightCells + 1e-30)
        interiorFaces = 2.0 / (reciprocalSum + 1e-30)  // [cellCount-1]
    }

    // Boundary faces use adjacent cell values. Keep interpolation on MLX, but freeze
    // this linearization path so coefficient assembly keeps the previous semi-implicit
    // Newton behavior without host round-trips.
    return concatenated([
        values[0..<1],
        interiorFaces,
        values[(cellCount - 1)..<cellCount]
    ], axis: 0)
}

// MARK: - Current Diffusion Helpers

/// Compute Spitzer resistivity with neoclassical correction
///
/// **Formula**:
/// ```
/// η_Spitzer = 5.2 × 10⁻⁵ * effectiveCharge * ln(Λ) / T_e^(3/2)  [Ω·m]
/// η_neo = η_Spitzer * (1 + 1.46 * √ε)  [neoclassical correction]
/// ```
///
/// **Parameters**:
/// - electronTemperature: Electron temperature [eV], shape [cellCount]
/// - geometry: Tokamak geometry
///
/// **Returns**: Resistivity [Ω·m], shape [cellCount]
///
/// **References**:
/// - Spitzer & Härm, "Transport Phenomena in a Completely Ionized Gas", Phys. Rev. 89, 977 (1953)
/// - NRL Plasma Formulary (2019)
///
/// **Implementation Note**:
/// Uses default effectiveCharge = 1.5 (typical for ITER with low-Z impurities).
/// Future enhancement: extract effectiveCharge from TransportParameters.parameters if available.
private func computeSpitzerResistivity(
    electronTemperature: MLXArray,
    geometry: Geometry
) -> MLXArray {
    // Default parameters (consistent with OhmicHeating.swift)
    let effectiveCharge: Float = 1.5        // Effective charge (deuterium + low-Z impurities)
    let coulombLog: Float = 17.0  // Coulomb logarithm (typical for tokamak core)

    // Spitzer resistivity: η = 5.2e-5 * effectiveCharge * ln(Λ) / T_e^(3/2)
    let eta_spitzer = 5.2e-5 * effectiveCharge * coulombLog / pow(electronTemperature, 1.5)

    // Neoclassical correction for trapped particles
    // Inverse aspect ratio: ε = r/R₀
    let epsilon = geometry.radii.value / geometry.majorRadius

    // Trapped particle correction factor: f_trap ≈ 1 + 1.46 * √ε
    let ft = 1.0 + 1.46 * sqrt(epsilon)

    // Neoclassical resistivity
    let eta_neo = eta_spitzer * ft

    return eta_neo
}

/// Compute bootstrap current from pressure gradients
///
/// **Physics**: Bootstrap current is self-generated current from pressure gradients
/// and trapped particle effects. It's crucial for tokamak steady-state operation.
///
/// **Full Sauter Formula**:
/// ```
/// J_BS = -C_BS(ν*, ft, ε) · (∇P / B_φ)
/// where C_BS = L₃₁·ft + L₃₂·ft·α + L₃₄·ft·α²
/// ```
///
/// **Critical**: Preserves sign (can be negative at edge for counter-current drive)
///
/// **Parameters**:
/// - profiles: Current core profiles
/// - geometry: Tokamak geometry
///
/// **Returns**: Bootstrap current density [A/m²], shape [cellCount]
///
/// **References**:
/// - Sauter et al., PoP 6, 2834 (1999), Eqs. 13-14, Table I
private func computeBootstrapCurrent(
    profiles: CoreProfiles,
    geometry: Geometry
) -> MLXArray {
    let Ti = profiles.ionTemperature.value
    let Te = profiles.electronTemperature.value
    let ne = profiles.electronDensity.value

    // 1. Total pressure: P = n_e (T_i + T_e) * e
    let P = ne * (Ti + Te) * UnitConversions.electronVolt  // [Pa]

    // 2. Pressure gradient: ∇P [Pa/m]
    let geoFactors = GeometricFactors.from(geometry: geometry)
    let gradP = computeGradient(P, cellDistances: geoFactors.cellDistances.value)

    // 3. Normalized collisionality ν*
    let nu_star = CollisionalityHelpers.computeNormalizedCollisionality(
        electronTemperature: Te,
        electronDensity: ne,
        geometry: geometry
    )

    // 4. Trapped particle fraction
    // ft = 1 - √(1 - ε) for ε < 1
    // Clamp ε to [0, 0.99] to ensure physical values
    let epsilon = geometry.radii.value / geometry.majorRadius
    let epsilon_safe = minimum(epsilon, MLXArray(0.99))  // Prevent ε ≥ 1
    let ft = 1.0 - sqrt(1.0 - epsilon_safe)

    // 5. Sauter coefficients L₃₁, L₃₂, L₃₄
    let L31 = computeSauterL31(normalizedCollisionality: nu_star, ft: ft)
    let L32 = computeSauterL32(normalizedCollisionality: nu_star, ft: ft)
    let L34 = computeSauterL34(normalizedCollisionality: nu_star, ft: ft)

    // 6. Pressure anisotropy parameter α (assume isotropic: α = 0)
    let alpha = MLXArray.zeros(like: Te)

    // 7. Bootstrap coefficient: C_BS = L₃₁·ft + L₃₂·ft·α + L₃₄·ft·α²
    let C_BS = L31 * ft + L32 * ft * alpha + L34 * ft * alpha * alpha

    // 8. Bootstrap current: J_BS = -C_BS · (∇P / B_φ)
    let J_BS = -C_BS * gradP / geometry.toroidalField

    // 9. Clamp magnitude only and preserve sign.
    // Bootstrap current can be negative at edge (counter-current drive)
    let J_BS_magnitude = abs(J_BS)
    let J_BS_clamped_magnitude = minimum(J_BS_magnitude, MLXArray(1e7))  // Max 10 MA/m²
    let J_BS_final = sign(J_BS) * J_BS_clamped_magnitude

    return J_BS_final
}

/// Compute Sauter L₃₁ coefficient (bootstrap current, main term)
///
/// **Formula** (Sauter Table I, simplified):
/// ```
/// L₃₁(ν*, ft) = ((1 + 0.15/ft) - 0.22/(1 + 0.01·ν*)) / (1 + 0.5·√ν*)
/// ```
///
/// - Parameters:
///   - normalizedCollisionality: Normalized collisionality [dimensionless]
///   - ft: Trapped fraction [dimensionless]
/// - Returns: L₃₁ coefficient [dimensionless]
private func computeSauterL31(normalizedCollisionality: MLXArray, ft: MLXArray) -> MLXArray {
    let ft_safe = ft + 1e-10
    let nu_safe = normalizedCollisionality + 1e-10

    let numerator = (1.0 + 0.15 / ft_safe) - 0.22 / (1.0 + 0.01 * nu_safe)
    let denominator = 1.0 + 0.5 * sqrt(nu_safe)

    return numerator / denominator
}

/// Compute Sauter L₃₂ coefficient (pressure anisotropy correction)
///
/// - Parameters:
///   - normalizedCollisionality: Normalized collisionality [dimensionless]
///   - ft: Trapped fraction [dimensionless]
/// - Returns: L₃₂ coefficient [dimensionless]
private func computeSauterL32(normalizedCollisionality: MLXArray, ft: MLXArray) -> MLXArray {
    // Simplified: L₃₂ ≈ 0.05
    return MLXArray.full(normalizedCollisionality.shape, values: MLXArray(0.05))
}

/// Compute Sauter L₃₄ coefficient (second-order pressure anisotropy)
///
/// - Parameters:
///   - normalizedCollisionality: Normalized collisionality [dimensionless]
///   - ft: Trapped fraction [dimensionless]
/// - Returns: L₃₄ coefficient [dimensionless]
private func computeSauterL34(normalizedCollisionality: MLXArray, ft: MLXArray) -> MLXArray {
    // Simplified: L₃₄ ≈ 0.01
    return MLXArray.full(normalizedCollisionality.shape, values: MLXArray(0.01))
}

/// Compute gradient with epsilon regularization
///
/// **Formula**: ∇f ≈ (f[i+1] - f[i]) / Δr
///
/// **Parameters**:
/// - profile: Profile values [cellCount]
/// - cellDistances: Distance between cell centers [cellCount-1]
///
/// **Returns**: Gradient at cell centers [cellCount]
///
/// **Implementation**:
/// - Interior cells: central differencing
/// - Boundary cells: one-sided differencing
/// - Epsilon regularization prevents division by zero
private func computeGradient(_ profile: MLXArray, cellDistances: MLXArray) -> MLXArray {
    let cellCount = profile.shape[0]

    // Compute differences: Δf = f[i+1] - f[i]
    let df = profile[1...] - profile[..<(cellCount - 1)]  // [cellCount-1]

    // Add epsilon to prevent division by zero
    let dr_safe = cellDistances + 1e-10  // [cellCount-1]

    // Gradient at interior faces
    let gradFaces = df / dr_safe  // [cellCount-1]

    // Interpolate to cell centers (GPU-first, no CPU transfer)
    // - Boundary cells: use nearest face value
    // - Interior cells: average of adjacent faces

    // Left boundary cell (i=0): use gradFaces[0]
    let gradCell0 = gradFaces[0..<1]  // [1]

    // Interior cells (i=1...cellCount-2): average of adjacent faces
    // gradCell[i] = (gradFaces[i-1] + gradFaces[i]) / 2
    let leftFaces = gradFaces[0..<(cellCount - 2)]   // [cellCount-2]
    let rightFaces = gradFaces[1..<(cellCount - 1)]  // [cellCount-2]
    let gradInterior = (leftFaces + rightFaces) / 2.0  // [cellCount-2]

    // Right boundary cell (i=cellCount-1): use gradFaces[cellCount-2]
    let gradCellN = gradFaces[(cellCount - 2)..<(cellCount - 1)]  // [1]

    // Concatenate: [1] + [cellCount-2] + [1] = [cellCount]
    let gradCells = concatenated([gradCell0, gradInterior, gradCellN], axis: 0)

    return gradCells
}
