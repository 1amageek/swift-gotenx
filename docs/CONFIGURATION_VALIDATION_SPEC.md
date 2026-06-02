# Configuration Validation Specification

**Version:** 1.0
**Last Updated:** 2025-10-24
**Status:** Design Document

## Overview

This document specifies pre-simulation validation checks to ensure numerical stability and physical validity before running simulations. These checks prevent common failure modes and provide actionable feedback to users.

## Design Principles

1. **Fail Fast**: Detect issues before computation starts
2. **Actionable Feedback**: Provide specific suggestions for fixing issues
3. **Layered Severity**: Distinguish between errors (must fix) and warnings (should review)
4. **Physics-Based**: Use physical time scales and constraints, not arbitrary thresholds

## Validation Categories

### 1. Source Term Stability

#### 1.1 ECRH Stability Validation

**Purpose**: Prevent timestep-scale temperature explosions from ECRH heating

**Physics Background**:
- ECRH deposits power in a localized Gaussian profile
- Peak power density can be 10-100x higher than average
- Rapid temperature changes destabilize Newton-Raphson solver

**Validation Rules**:

| Check | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Temperature change per timestep | `ΔT/T < 0.5` | ERROR | Reduce `totalPower` or decrease `dt` |
| Peak power density | `P_peak < 100 MW/m³` | WARNING | Review ECRH configuration |
| Deposition width resolution | `depositionWidth > 3×Δr` | ERROR | Increase `depositionWidth` or `cellCount` |

**Implementation**:

```swift
func validateECRHStability(
    ecrh: ECRHConfig,
    initialTemp: Float,      // [eV]
    density: Float,          // [m^-3]
    volume: Float,           // [m^3]
    minorRadius: Float,      // [m]
    dt: Float,               // [s]
    cellSpacing: Float       // [m]
) throws {
    // Estimate peak power density (Gaussian profile)
    // For Gaussian deposition, peak region (within 1σ) contains ~68% of power
    let sigma = ecrh.depositionWidth / 3.0
    let rho_dep = ecrh.depositionRho
    let peakRadiusFraction = sigma / minorRadius  // Normalized σ
    let peakVolumeFraction = max(0.1, 2.0 * rho_dep * peakRadiusFraction)  // Approximate, with 10% floor
    let peakPowerDensity = ecrh.totalPower / (volume * peakVolumeFraction)  // [W/m^3]

    // Convert to MW/m^3 for reporting
    let peakPowerDensity_MW = peakPowerDensity / 1e6

    // Check peak power density
    if peakPowerDensity_MW > 100.0 {
        throw ValidationWarning.highPowerDensity(
            value: peakPowerDensity_MW,
            limit: 100.0,
            suggestion: "Reduce ECRH totalPower to \(ecrh.totalPower * 100.0 / peakPowerDensity_MW) W"
        )
    }

    // Estimate temperature change per timestep
    // Energy equation (non-conservation form): (3/2) n_e dT/dt = Q / e
    // → dT/dt = (2/3) Q / (n_e e)
    // where e = 1.602×10⁻¹⁹ J/eV (elementary charge)
    //
    // Dimension check:
    //   (2/3) * [W/m³] / ([m⁻³] * [J/eV])
    // = (2/3) * [J/(s·m³)] / ([m⁻³] * [J/eV])
    // = (2/3) * [eV/s] ✓
    let elementaryCharge: Float = 1.602e-19  // [J/eV]
    let tempChangeRate_eV = (2.0/3.0) * peakPowerDensity / (density * elementaryCharge)  // [eV/s]
    let tempChange = tempChangeRate_eV * dt  // [eV]
    let changeRatio = tempChange / initialTemp

    if changeRatio > 0.5 {
        throw ValidationError.unstableTimestep(
            parameter: "ECRH heating",
            changeRatio: changeRatio,
            suggestion: "Reduce ECRH totalPower to \(ecrh.totalPower * 0.5 / changeRatio) W or decrease dt to \(dt * 0.5 / changeRatio) s"
        )
    }

    // Check deposition width vs mesh resolution
    let minWidthForResolution = 3.0 * cellSpacing
    if ecrh.depositionWidth < minWidthForResolution {
        throw ValidationError.insufficientResolution(
            parameter: "ECRH depositionWidth",
            value: ecrh.depositionWidth,
            minimum: minWidthForResolution,
            suggestion: "Increase depositionWidth to \(minWidthForResolution) or increase cellCount to \(Int(3.0 * minorRadius / ecrh.depositionWidth))"
        )
    }
}
```

**Example Error Messages**:

```
ERROR: Unstable timestep for ECRH heating
  Temperature change per timestep: 87% (limit: 50%)
  Suggestion: Reduce ECRH totalPower to 5.7 MW or decrease dt to 5.7e-5 s

WARNING: High ECRH power density detected
  Peak power density: 142 MW/m³ (typical limit: 100 MW/m³)
  Suggestion: Reduce ECRH totalPower to 7.0 MW

ERROR: Insufficient mesh resolution for ECRH
  depositionWidth: 0.08 m (minimum: 0.12 m for 3-cell resolution)
  Suggestion: Increase depositionWidth to 0.12 or increase cellCount to 150
```

#### 1.2 Ohmic Heating Stability

**Purpose**: Ensure Ohmic heating calculation doesn't produce unphysical results

**Validation Rules**:

| Check | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Plasma current density | `j < 10 MA/m²` | WARNING | Review plasma current |
| Temperature for resistivity | `T_e > 10 eV` | WARNING | Ohmic heating model may be inaccurate |

**Implementation**:

```swift
func validateOhmicHeating(
    plasmaCurrent: Float,    // [MA]
    minorRadius: Float,      // [m]
    temperature: Float       // [eV]
) throws {
    let crossSectionArea = Float.pi * minorRadius * minorRadius
    let currentDensity = plasmaCurrent / crossSectionArea  // [MA/m^2]

    if currentDensity > 10.0 {
        throw ValidationWarning.highCurrentDensity(
            value: currentDensity,
            limit: 10.0,
            suggestion: "Review plasma current configuration"
        )
    }

    if temperature < 10.0 {
        throw ValidationWarning.lowTemperatureForOhmic(
            value: temperature,
            limit: 10.0,
            suggestion: "Ohmic heating model may be inaccurate at T < 10 eV"
        )
    }
}
```

#### 1.3 Gas Puff Stability

**Purpose**: Prevent density explosions from particle fueling

**Validation Rules**:

| Check | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Density change per timestep | `Δn/n < 0.2` | ERROR | Reduce `puffRate` or decrease `dt` |
| Particle source rate | `S_n < 1e22 particles/s` | WARNING | Review puff rate |

**Implementation**:

```swift
func validateGasPuffStability(
    gasPuff: GasPuffConfig,
    initialDensity: Float,   // [m^-3]
    volume: Float,           // [m^3]
    dt: Float                // [s]
) throws {
    // Estimate density change per timestep
    // ΔN = S * dt (total particles added)
    // Δn = ΔN / V (density change)
    let particlesAdded = gasPuff.puffRate * dt
    let densityChange = particlesAdded / volume
    let changeRatio = densityChange / initialDensity

    if changeRatio > 0.2 {
        throw ValidationError.unstableTimestep(
            parameter: "Gas puff",
            changeRatio: changeRatio,
            suggestion: "Reduce puffRate to \(gasPuff.puffRate * 0.2 / changeRatio) particles/s or decrease dt to \(dt * 0.2 / changeRatio) s"
        )
    }

    if gasPuff.puffRate > 1e22 {
        throw ValidationWarning.highPuffRate(
            value: gasPuff.puffRate,
            limit: 1e22,
            suggestion: "Review gas puff configuration"
        )
    }
}
```

### 2. Transport Coefficient Validity

#### 2.1 CFL Condition (Thermal Diffusion)

**Purpose**: Ensure numerical stability of heat diffusion equation

**Physics Background**:
- Heat diffusion equation: `∂T/∂t = χ ∂²T/∂r²`
- Explicit schemes require CFL condition: `χ dt / (Δr)² < 0.5`
- Implicit schemes (Newton-Raphson) are stable but accuracy degrades if CFL >> 1

**Validation Rules**:

| Check | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Ion thermal diffusion CFL | `CFL_i = χ_i dt/(Δr)² < 0.5` | ERROR | Reduce `ionHeatDiffusivity` or decrease `dt` |
| Electron thermal diffusion CFL | `CFL_e = χ_e dt/(Δr)² < 0.5` | ERROR | Reduce `electronHeatDiffusivity` or decrease `dt` |
| Negative diffusivity | `χ > 0` | ERROR | Check transport model configuration |

**Implementation**:

```swift
func validateCFLCondition(
    transport: TransportConfig,
    dt: Float,
    cellSpacing: Float
) throws {
    try transport.validateParameterKeys()

    guard let ionHeatDiffusivity = transport.parameter("ionHeatDiffusivity") else {
        throw ValidationError.missingRequiredParameter(parameter: "ionHeatDiffusivity")
    }
    guard let electronHeatDiffusivity = transport.parameter("electronHeatDiffusivity") else {
        throw ValidationError.missingRequiredParameter(parameter: "electronHeatDiffusivity")
    }

    // Check positive
    if ionHeatDiffusivity <= 0 || electronHeatDiffusivity <= 0 {
        throw ValidationError.negativeTransportCoefficient(
            parameter: ionHeatDiffusivity <= 0 ? "ionHeatDiffusivity" : "electronHeatDiffusivity",
            value: ionHeatDiffusivity <= 0 ? ionHeatDiffusivity : electronHeatDiffusivity
        )
    }

    // Compute CFL numbers
    let ionCFL = ionHeatDiffusivity * dt / (cellSpacing * cellSpacing)
    let electronCFL = electronHeatDiffusivity * dt / (cellSpacing * cellSpacing)

    if ionCFL > 0.5 {
        // To achieve CFL = 0.5: χ_new = χ × 0.5/CFL or dt_new = dt × 0.5/CFL
        throw ValidationError.cflViolation(
            parameter: "ionHeatDiffusivity",
            cfl: ionCFL,
            limit: 0.5,
            suggestion: "Reduce ionHeatDiffusivity to \(ionHeatDiffusivity * 0.5 / ionCFL) m²/s or decrease dt to \(dt * 0.5 / ionCFL) s"
        )
    }

    if electronCFL > 0.5 {
        // To achieve CFL = 0.5: χ_new = χ × 0.5/CFL or dt_new = dt × 0.5/CFL
        throw ValidationError.cflViolation(
            parameter: "electronHeatDiffusivity",
            cfl: electronCFL,
            limit: 0.5,
            suggestion: "Reduce electronHeatDiffusivity to \(electronHeatDiffusivity * 0.5 / electronCFL) m²/s or decrease dt to \(dt * 0.5 / electronCFL) s"
        )
    }
}
```

**Example Error Messages**:

```
ERROR: CFL condition violated for ionHeatDiffusivity
  CFL = 0.87 (limit: 0.5)
  Suggestion: Reduce ionHeatDiffusivity to 0.57 m²/s or decrease dt to 5.7e-5 s
```

#### 2.2 Particle Diffusion CFL

**Purpose**: Ensure numerical stability of particle transport

**Validation Rules**:

| Check | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Particle diffusion CFL | `CFL_D = D dt/(Δr)² < 0.5` | ERROR | Reduce `particleDiffusivity` or decrease `dt` |

**Implementation**:

```swift
func validateParticleCFL(
    particleDiffusivity: Float,
    dt: Float,
    cellSpacing: Float
) throws {
    if particleDiffusivity <= 0 {
        throw ValidationError.negativeTransportCoefficient(
            parameter: "particleDiffusivity",
            value: particleDiffusivity
        )
    }

    let CFL_particle = particleDiffusivity * dt / (cellSpacing * cellSpacing)

    if CFL_particle > 0.5 {
        // To achieve CFL = 0.5: D_new = D × 0.5/CFL or dt_new = dt × 0.5/CFL
        throw ValidationError.cflViolation(
            parameter: "particleDiffusivity",
            cfl: CFL_particle,
            limit: 0.5,
            suggestion: "Reduce particleDiffusivity to \(particleDiffusivity * 0.5 / CFL_particle) m²/s or decrease dt to \(dt * 0.5 / CFL_particle) s"
        )
    }
}
```

### 3. Boundary Condition Consistency

#### 3.1 Temperature Boundary Consistency

**Purpose**: Ensure initial profile is consistent with boundary conditions

**Validation Rules**:

| Check | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Core > boundary (peaked) | `T_core > T_boundary` when peaked profile | ERROR | Fix boundary temperature or profile ratio |
| Minimum ratio | `T_core / T_boundary > 1.2` when peaked | WARNING | Profile may be too flat |

**Implementation**:

```swift
func validateTemperatureBoundaryConsistency(
    boundary: BoundaryConfig,
    initialProfile: InitialProfileConfig
) throws {
    switch initialProfile {
    case .flat:
        // No constraints for flat profile
        break

    case .peaked(let coreFactor, _):
        let T_core_ion = boundary.ionTemperature * coreFactor
        let T_core_electron = boundary.electronTemperature * coreFactor

        if T_core_ion < boundary.ionTemperature {
            throw ValidationError.inconsistentBoundary(
                parameter: "ionTemperature",
                coreValue: T_core_ion,
                boundaryValue: boundary.ionTemperature,
                suggestion: "Increase coreFactor to > 1.0 or use flat initial profile"
            )
        }

        if T_core_electron < boundary.electronTemperature {
            throw ValidationError.inconsistentBoundary(
                parameter: "electronTemperature",
                coreValue: T_core_electron,
                boundaryValue: boundary.electronTemperature,
                suggestion: "Increase coreFactor to > 1.0 or use flat initial profile"
            )
        }

        if coreFactor < 1.2 {
            throw ValidationWarning.flatProfile(
                parameter: "temperature",
                coreFactor: coreFactor,
                suggestion: "Consider increasing coreFactor to > 1.5 for more realistic profile"
            )
        }
    }
}
```

#### 3.2 Density Boundary Consistency

**Purpose**: Ensure density profile is consistent with boundaries

**Implementation**: Similar to temperature boundary consistency

### 4. Timestep Validity

#### 4.1 Diffusion Time Scale

**Purpose**: Ensure timestep is appropriate for transport time scales

**Physics Background**:
- Diffusion time scale: `τ_diff = a² / χ`
- Simulation should resolve time scales: `dt < τ_diff / 10`

**Validation Rules**:

| Check | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Undersampling | `dt > τ_diff` | ERROR | Decrease `dt` or check `chi` values |
| Poor resolution | `dt < τ_diff / 100` | WARNING | Consider increasing `dt` for efficiency |
| Excessive resolution | `dt < 1e-6 s` | WARNING | Timestep may be too small |

**Implementation**:

```swift
func validateDiffusionTimeScale(
    transport: TransportConfig,
    dt: Float,
    minorRadius: Float
) throws {
    guard transport.modelType == .constant else {
        return
    }

    let maximumHeatDiffusivity = max(
        try transport.requireParameter("ionHeatDiffusivity"),
        try transport.requireParameter("electronHeatDiffusivity")
    )

    let diffusionTimeScale = minorRadius * minorRadius / maximumHeatDiffusivity

    if dt > diffusionTimeScale {
        throw ValidationError.timestepTooLarge(
            dt: dt,
            timeScale: diffusionTimeScale,
            suggestion: "Decrease dt to \(diffusionTimeScale / 10) s"
        )
    }

    if dt < diffusionTimeScale / 100 {
        throw ValidationWarning.timestepTooSmall(
            dt: dt,
            timeScale: diffusionTimeScale,
            suggestion: "Consider increasing dt to \(diffusionTimeScale / 10) s for better efficiency"
        )
    }

    if dt < 1e-6 {
        throw ValidationWarning.timestepTooSmall(
            dt: dt,
            timeScale: 1e-6,
            suggestion: "Timestep < 1 μs may cause excessive computation time"
        )
    }
}
```

#### 4.2 Heating Time Scale

**Purpose**: Ensure timestep resolves heating dynamics

**Physics Background**:
- Energy balance: `dE/dt = P`
- Stored energy: `E = (3/2) n T V`
- Heating time scale: `τ_heat = E / P = (3/2) n T V / P`

**Validation Rules**:

| Check | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Heating resolution | `dt < τ_heat / 5` | WARNING | Increase `dt` or check heating power |

**Implementation**:

```swift
func validateHeatingTimeScale(
    sources: SourcesConfig,
    boundary: BoundaryConfig,
    volume: Float,
    dt: Float
) throws {
    // Estimate total heating power
    var totalPower: Float = 0.0

    if let ecrh = sources.ecrh {
        totalPower += ecrh.totalPower
    }

    if totalPower == 0.0 {
        return  // No heating sources
    }

    // Estimate stored energy: E = (3/2) n_e k_B (T_i + T_e) V
    // In eV units: E = (3/2) n_e e (T_i + T_e) V [J]
    // where e = 1.602×10⁻¹⁹ J/eV converts eV to Joules
    //
    // Dimension check:
    //   1.5 × [m⁻³] × [eV/particle] × [J/eV] × [m³]
    // = 1.5 × [particles/m³] × [J/particle] × [m³]
    // = 1.5 × [J] ✓
    let totalTemp_eV = boundary.ionTemperature + boundary.electronTemperature  // [eV] (ions + electrons)
    let elementaryCharge: Float = 1.602e-19  // [J/eV]
    let storedEnergy = 1.5 * boundary.density * totalTemp_eV * elementaryCharge * volume  // [J]

    let tau_heating = storedEnergy / totalPower  // [J] / [W] = [s]

    if dt > tau_heating / 5 {
        throw ValidationWarning.poorTimeResolution(
            parameter: "heating time scale",
            dt: dt,
            timeScale: tau_heating,
            suggestion: "Decrease dt to \(tau_heating / 5) s for better resolution"
        )
    }
}
```

### 5. Mesh Resolution

#### 5.1 Gradient Resolution

**Purpose**: Ensure mesh can resolve profile gradients

**Physics Background**:
- Gradient scale length: `L_T = T / |∇T|`
- Require at least 3 cells to resolve gradient: `Δr < L_T / 3`

**Validation Rules**:

| Check | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Minimum cells | `cellCount > 50` | ERROR | Increase `cellCount` |
| Gradient resolution | `cellCount > 3 * exponent` | WARNING | Increase `cellCount` for steep profiles |
| Excessive cells | `cellCount < 500` | WARNING | Reduce `cellCount` for efficiency |

**Implementation**:

```swift
func validateMeshResolution(
    cellCount: Int,
    initialProfile: InitialProfileConfig
) throws {
    if cellCount < 50 {
        throw ValidationError.insufficientMeshResolution(
            cellCount: cellCount,
            minimum: 50,
            suggestion: "Increase cellCount to at least 50"
        )
    }

    if cellCount > 500 {
        throw ValidationWarning.excessiveMeshResolution(
            cellCount: cellCount,
            maximum: 500,
            suggestion: "Consider reducing cellCount to ~200 for better performance"
        )
    }

    switch initialProfile {
    case .peaked(_, let exponent):
        // For parabolic profile T(ρ) ∝ (1-ρ)^n, gradient scale length L_T ~ a/n at mid-radius
        // Require 3 cells per gradient scale length: cellCount > 3n
        // Use minimum of 50 cells even for small exponents
        let recommendedCells = max(50, Int(3.0 * exponent))
        if cellCount < recommendedCells {
            throw ValidationWarning.insufficientGradientResolution(
                cellCount: cellCount,
                recommended: recommendedCells,
                profileExponent: exponent,
                suggestion: "Increase cellCount to \(recommendedCells) to resolve gradient scale length L_T ~ a/\(Int(exponent))"
            )
        }
    case .flat:
        break
    }
}
```

#### 5.2 ECRH Localization Resolution

**Purpose**: Ensure ECRH deposition is resolved by mesh

**Implementation**: See Section 1.1 (ECRH Stability Validation)

### 6. Physical Range Validation

#### 6.1 Temperature Range

**Purpose**: Ensure temperatures are in physically valid range

**Validation Rules**:

| Parameter | Minimum | Maximum | Unit |
|-----------|---------|---------|------|
| ionTemperature | 1.0 | 100,000 | eV |
| electronTemperature | 1.0 | 100,000 | eV |

**Implementation**:

```swift
func validateTemperatureRange(
    ionTemp: Float,
    electronTemp: Float
) throws {
    if ionTemp < 1.0 || ionTemp > 100_000 {
        throw ValidationError.outOfPhysicalRange(
            parameter: "ionTemperature",
            value: ionTemp,
            range: (1.0, 100_000),
            unit: "eV"
        )
    }

    if electronTemp < 1.0 || electronTemp > 100_000 {
        throw ValidationError.outOfPhysicalRange(
            parameter: "electronTemperature",
            value: electronTemp,
            range: (1.0, 100_000),
            unit: "eV"
        )
    }
}
```

#### 6.2 Density Range

**Purpose**: Ensure density is in physically valid range for tokamak plasmas

**Validation Rules**:

| Parameter | Minimum | Maximum | Unit |
|-----------|---------|---------|------|
| density | 1e17 | 1e21 | m⁻³ |

**Implementation**:

```swift
func validateDensityRange(
    density: Float
) throws {
    if density < 1e17 || density > 1e21 {
        throw ValidationError.outOfPhysicalRange(
            parameter: "density",
            value: density,
            range: (1e17, 1e21),
            unit: "m⁻³"
        )
    }
}
```

#### 6.3 Magnetic Field Range

**Purpose**: Ensure magnetic field is in realistic tokamak range

**Validation Rules**:

| Parameter | Minimum | Maximum | Unit |
|-----------|---------|---------|------|
| toroidalField | 0.5 | 15.0 | T |

**Implementation**:

```swift
func validateMagneticFieldRange(
    toroidalField: Float
) throws {
    if toroidalField < 0.5 || toroidalField > 15.0 {
        throw ValidationError.outOfPhysicalRange(
            parameter: "toroidalField",
            value: toroidalField,
            range: (0.5, 15.0),
            unit: "T"
        )
    }
}
```

#### 6.4 Geometry Range

**Purpose**: Ensure tokamak geometry is physically realistic

**Validation Rules**:

| Parameter | Minimum | Maximum | Unit | Notes |
|-----------|---------|---------|------|-------|
| majorRadius | 0.5 | 10.0 | m | |
| minorRadius | 0.2 | 3.0 | m | |
| aspectRatio (a/R) | - | 0.5 | - | Tight-aspect limit |

**Implementation**:

```swift
func validateGeometryRange(
    majorRadius: Float,
    minorRadius: Float
) throws {
    if majorRadius < 0.5 || majorRadius > 10.0 {
        throw ValidationError.outOfPhysicalRange(
            parameter: "majorRadius",
            value: majorRadius,
            range: (0.5, 10.0),
            unit: "m"
        )
    }

    if minorRadius < 0.2 || minorRadius > 3.0 {
        throw ValidationError.outOfPhysicalRange(
            parameter: "minorRadius",
            value: minorRadius,
            range: (0.2, 3.0),
            unit: "m"
        )
    }

    let aspectRatio = minorRadius / majorRadius
    if aspectRatio > 0.5 {
        throw ValidationError.invalidGeometry(
            parameter: "aspectRatio",
            value: aspectRatio,
            limit: 0.5,
            suggestion: "Reduce minorRadius or increase majorRadius"
        )
    }
}
```

### 7. Model-Specific Validation

#### 7.1 QLKNN Training Range

**Purpose**: Ensure simulation conditions are within QLKNN neural network training range

**Validation Rules** (when `transport.modelType == .qlknn`):

| Parameter | Minimum | Maximum | Unit | Reason |
|-----------|---------|---------|------|--------|
| T_e | 500 | 20,000 | eV | Training data range |
| n_e | 1e19 | 1e20 | m⁻³ | Training data range |

**Implementation**:

```swift
func validateQLKNNRange(
    electronTemp: Float,
    density: Float
) throws {
    if electronTemp < 500.0 {
        throw ValidationWarning.outsideTrainingRange(
            model: "QLKNN",
            parameter: "electronTemperature",
            value: electronTemp,
            range: (500.0, 20_000),
            suggestion: "QLKNN may be inaccurate below 500 eV. Consider using Bohm-GyroBohm model."
        )
    }

    if density < 1e19 || density > 1e20 {
        throw ValidationWarning.outsideTrainingRange(
            model: "QLKNN",
            parameter: "density",
            value: density,
            range: (1e19, 1e20),
            suggestion: "QLKNN was trained on densities 1e19-1e20 m⁻³. Results may be less accurate."
        )
    }
}
```

#### 7.2 Fusion Power Conditions

**Purpose**: Warn when fusion power will be negligible

**Validation Rules** (when `sources.fusionPower == true`):

| Parameter | Minimum | Unit | Reason |
|-----------|---------|------|--------|
| T_i | 1,000 | eV | Fusion cross-section threshold |
| fuelFraction (D+T) | 0.9 | - | Sufficient reactants |

**Implementation**:

```swift
func validateFusionConditions(
    ionTemp: Float,
    fusionConfig: FusionConfig
) throws {
    if ionTemp < 1000.0 {
        throw ValidationWarning.negligibleFusionPower(
            temperature: ionTemp,
            threshold: 1000.0,
            suggestion: "Fusion power is negligible below 1 keV. Consider disabling fusion source."
        )
    }

    let totalFuelFraction = fusionConfig.deuteriumFraction + fusionConfig.tritiumFraction
    if abs(totalFuelFraction - 1.0) > 0.01 {
        throw ValidationError.invalidFuelMix(
            dFraction: fusionConfig.deuteriumFraction,
            tFraction: fusionConfig.tritiumFraction,
            suggestion: "D+T fractions must sum to 1.0"
        )
    }
}
```

---

## Implementation Architecture

### File Structure

```
Sources/GotenxCore/Configuration/
├── ConfigurationValidator.swift      (main validator)
├── ValidationError.swift             (error types)
└── ValidationWarnings.swift          (warning types)
```

### Main Validator Interface

```swift
/// Configuration validator for pre-simulation checks
public struct ConfigurationValidator {

    /// Validate complete simulation configuration
    ///
    /// - Parameter config: Simulation configuration to validate
    /// - Throws: ValidationError for critical issues, ValidationWarning for concerns
    public static func validate(_ config: SimulationConfiguration) throws {
        // Phase 1: Physical range validation
        try validatePhysicalRanges(config)

        // Phase 2: Numerical stability validation
        try validateNumericalStability(config)

        // Phase 3: Model-specific validation
        try validateModelConstraints(config)
    }

    /// Validate and collect all warnings (non-throwing)
    ///
    /// - Parameter config: Simulation configuration to validate
    /// - Returns: Array of validation warnings
    public static func collectWarnings(_ config: SimulationConfiguration) -> [ValidationWarning] {
        var warnings: [ValidationWarning] = []

        // Collect all warnings without throwing
        // ...

        return warnings
    }
}
```

### Error Types

```swift
/// Critical validation errors that prevent simulation
public enum ValidationError: Error, LocalizedError {
    case unstableTimestep(parameter: String, changeRatio: Float, suggestion: String)
    case cflViolation(parameter: String, cfl: Float, limit: Float, suggestion: String)
    case insufficientResolution(parameter: String, value: Float, minimum: Float, suggestion: String)
    case outOfPhysicalRange(parameter: String, value: Float, range: (Float, Float), unit: String)
    case inconsistentBoundary(parameter: String, coreValue: Float, boundaryValue: Float, suggestion: String)
    case invalidGeometry(parameter: String, value: Float, limit: Float, suggestion: String)
    case negativeTransportCoefficient(parameter: String, value: Float)
    case invalidFuelMix(dFraction: Float, tFraction: Float, suggestion: String)

    public var errorDescription: String? {
        switch self {
        case .unstableTimestep(let param, let ratio, let suggestion):
            return """
            ERROR: Unstable timestep for \(param)
              Change per timestep: \(Int(ratio * 100))% (limit: 50%)
              \(suggestion)
            """
        case .cflViolation(let param, let cfl, let limit, let suggestion):
            return """
            ERROR: CFL condition violated for \(param)
              CFL = \(String(format: "%.2f", cfl)) (limit: \(String(format: "%.2f", limit)))
              \(suggestion)
            """
        // ... other cases
        }
    }
}

/// Non-critical validation warnings
public enum ValidationWarning: Error, LocalizedError {
    case highPowerDensity(value: Float, limit: Float, suggestion: String)
    case flatProfile(parameter: String, coreFactor: Float, suggestion: String)
    case timestepTooSmall(dt: Float, timeScale: Float, suggestion: String)
    case poorTimeResolution(parameter: String, dt: Float, timeScale: Float, suggestion: String)
    case outsideTrainingRange(model: String, parameter: String, value: Float, range: (Float, Float), suggestion: String)
    case negligibleFusionPower(temperature: Float, threshold: Float, suggestion: String)

    public var errorDescription: String? {
        switch self {
        case .highPowerDensity(let value, let limit, let suggestion):
            return """
            WARNING: High power density detected
              Peak power density: \(String(format: "%.1f", value)) MW/m³ (typical limit: \(String(format: "%.1f", limit)) MW/m³)
              \(suggestion)
            """
        // ... other cases
        }
    }
}
```

### Integration Point

```swift
// In SimulationRunner.swift or AppViewModel.swift

func runSimulation(config: SimulationConfiguration) async throws {
    // Validate configuration before starting
    do {
        try ConfigurationValidator.validate(config)
    } catch let error as ValidationError {
        // Critical error - abort
        throw error
    } catch let warning as ValidationWarning {
        // Non-critical warning - log and continue (or ask user)
        print("⚠️  \(warning.localizedDescription)")
    }

    // Proceed with simulation...
}
```

---

## Testing Strategy

### Unit Tests

Each validation function should have tests covering:
1. Valid configurations (should pass)
2. Boundary cases (should pass or warn)
3. Invalid configurations (should error with correct message)

**Example Test**:

```swift
func testECRHStabilityValidation() throws {
    // Valid configuration
    XCTAssertNoThrow(
        try ConfigurationValidator.validateECRHStability(
            ecrh: ECRHConfig(totalPower: 1e6, depositionRho: 0.3, depositionWidth: 0.15),
            initialTemp: 1000.0,
            density: 2e19,
            volume: 40.0,
            dt: 1e-4,
            cellSpacing: 0.02
        )
    )

    // Too high power - should throw
    XCTAssertThrowsError(
        try ConfigurationValidator.validateECRHStability(
            ecrh: ECRHConfig(totalPower: 50e6, depositionRho: 0.3, depositionWidth: 0.15),
            initialTemp: 100.0,  // Low temp + high power = unstable
            density: 2e19,
            volume: 40.0,
            dt: 1e-4,
            cellSpacing: 0.02
        )
    ) { error in
        XCTAssert(error is ValidationError)
    }
}
```

### Integration Tests

Test real-world configuration examples:
1. ITER baseline (should pass)
2. Small tokamak with high power (should warn/error)
3. Low-temperature startup (should warn about fusion)

---

## Future Enhancements

### Phase 2 Validations

1. **Profile consistency**: Check that peaked profiles are physically reasonable
2. **Power balance check**: Estimate steady-state temperature from P = n²f(T)
3. **Confinement time estimate**: Warn if simulation time >> τ_E
4. **MHD stability**: Basic q-profile check for sawteeth

### Phase 3 Validations

1. **Time-dependent validation**: Re-check CFL during simulation if transport coefficients change
2. **Adaptive suggestions**: AI-based parameter tuning suggestions
3. **Historical comparison**: Compare with previous successful runs

---

## References

1. **CFL Condition**: Courant, R., et al., "Über die partiellen Differenzengleichungen der mathematischen Physik", Mathematische Annalen, 1928
2. **ITER Physics Basis**: Nucl. Fusion 39, 2495 (1999)
3. **QLKNN Model**: van de Plassche, K.L., et al., Phys. Plasmas 27, 022310 (2020)
4. **Numerical Stability**: Press, W.H., et al., "Numerical Recipes", Cambridge University Press, 2007

---

## Changelog

### Version 1.0 (2025-10-24)
- Initial specification
- Phase 1 validations defined (ECRH, CFL, physical ranges, timestep)
- Error and warning types specified
- Implementation architecture designed
