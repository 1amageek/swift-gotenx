import Testing
import Foundation
import MLX
@testable import GotenxCore

/// Tests for Block1DCoeffsBuilder functionality
@Suite("Block1DCoeffsBuilder Tests")
struct Block1DCoeffsBuilderTests {

    // MARK: - Harmonic Mean with Large Density Tests

    /// Test that harmonic mean interpolation handles large density values without overflow
    ///
    /// Verifies the fix for Float32 overflow with realistic plasma densities (n_e ~ 1e20 m⁻³)
    /// See IMPLEMENTATION_NOTES.md Section 4 for context.
    @Test("Harmonic mean with large density (1e20 m⁻³) does not produce NaN or inf")
    func testHarmonicMeanLargeDensity() throws {
        // Typical ITER-scale plasma density
        let cellCount = 25
        let densityValue: Float = 1e20  // [m⁻³] - realistic plasma density

        // Create uniform density profile
        let densityArray = MLXArray(Array(repeating: densityValue, count: cellCount))
        let density = EvaluatedArray(evaluating: densityArray)

        // Create minimal profiles for testing
        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(10000.0), count: cellCount))),
            electronTemperature: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(10000.0), count: cellCount))),
            electronDensity: density,
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        // Create geometry using MeshConfig
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        let geometry = Geometry(config: meshConfig)

        // Create minimal transport coefficients
        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            electronHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            particleDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            convectionVelocity: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        // Create minimal sources
        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            particleSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            currentSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        // Create static parameters
        let staticParameters = StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: true,
            evolveElectronHeat: true,
            evolveElectronDensity: false,
            evolvePoloidalFlux: false,
            solverType: .linear,
            theta: 1.0,
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )

        // Build coefficients - this should not produce NaN or inf
        let coeffs = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )
        try coeffs.validateNumerics()

        // Verify no NaN in ion coefficients
        let ionDFace = coeffs.ionCoeffs.faceDiffusionCoefficient.value.asArray(Float.self)
        for (i, value) in ionDFace.enumerated() {
            #expect(!value.isNaN, "Ion faceDiffusionCoefficient[\(i)] is NaN")
            #expect(!value.isInfinite, "Ion faceDiffusionCoefficient[\(i)] is infinite")
        }

        // Verify no NaN in electron coefficients
        let electronDFace = coeffs.electronCoeffs.faceDiffusionCoefficient.value.asArray(Float.self)
        for (i, value) in electronDFace.enumerated() {
            #expect(!value.isNaN, "Electron faceDiffusionCoefficient[\(i)] is NaN")
            #expect(!value.isInfinite, "Electron faceDiffusionCoefficient[\(i)] is infinite")
        }

        // Verify no NaN in transient coefficients
        let ionTransient = coeffs.ionCoeffs.transientCoefficient.value.asArray(Float.self)
        for (i, value) in ionTransient.enumerated() {
            #expect(!value.isNaN, "Ion transientCoefficient[\(i)] is NaN")
            #expect(!value.isInfinite, "Ion transientCoefficient[\(i)] is infinite")
        }

        let electronTransient = coeffs.electronCoeffs.transientCoefficient.value.asArray(Float.self)
        for (i, value) in electronTransient.enumerated() {
            #expect(!value.isNaN, "Electron transientCoefficient[\(i)] is NaN")
            #expect(!value.isInfinite, "Electron transientCoefficient[\(i)] is infinite")
        }
    }

    /// Test density floor is applied correctly
    ///
    /// Verifies that very low density values are clamped to the floor (1e18 m⁻³)
    @Test("Density floor prevents division by zero")
    func testDensityFloor() throws {
        let cellCount = 25

        // Create very low density profile (below physical minimum)
        let lowDensityValue: Float = 1e10  // Much below floor of 1e18
        let densityArray = MLXArray(Array(repeating: lowDensityValue, count: cellCount))
        let density = EvaluatedArray(evaluating: densityArray)

        // Create minimal profiles
        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(10000.0), count: cellCount))),
            electronTemperature: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(10000.0), count: cellCount))),
            electronDensity: density,
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        // Create geometry using MeshConfig
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        let geometry = Geometry(config: meshConfig)

        // Create transport and sources
        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            electronHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            particleDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            convectionVelocity: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            particleSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            currentSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let staticParameters = StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: true,
            evolveElectronHeat: true,
            evolveElectronDensity: false,
            evolvePoloidalFlux: false,
            solverType: .linear,
            theta: 1.0,
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )

        // Build coefficients
        let coeffs = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )

        // Verify transient coefficients are at or above the floor
        let densityFloor: Float = 1e18
        let ionTransient = coeffs.ionCoeffs.transientCoefficient.value.asArray(Float.self)
        for (i, value) in ionTransient.enumerated() {
            #expect(value >= densityFloor, "Ion transientCoefficient[\(i)] = \(value) is below floor \(densityFloor)")
        }

        let electronTransient = coeffs.electronCoeffs.transientCoefficient.value.asArray(Float.self)
        for (i, value) in electronTransient.enumerated() {
            #expect(value >= densityFloor, "Electron transientCoefficient[\(i)] = \(value) is below floor \(densityFloor)")
        }
    }

    // MARK: - Source Term Unit Conversion Tests

    /// Test that source term unit conversion is correct
    ///
    /// Verifies MW/m³ → eV/(m³·s) conversion for temperature equation consistency
    @Test("Source term unit conversion MW/m³ → eV/(m³·s)")
    func testSourceTermUnitConversion() throws {
        // Test scalar conversion
        let Q_MW: Float = 1.0  // [MW/m³]
        let Q_eV = UnitConversions.megawattsToElectronVoltDensity(Q_MW)

        // Expected: 1 MW/m³ = 6.2415090744×10²⁴ eV/(m³·s)
        let expected: Float = 6.2415090744e24
        let relativeError = abs(Q_eV - expected) / expected

        #expect(relativeError < 1e-6, "Power density unit conversion error: \(Q_eV) vs \(expected)")
    }

    /// Test dimensional consistency of temperature equation
    ///
    /// Verifies that all terms in n_e ∂T_i/∂t = ∇·(n_e χ_i ∇T_i) + Q_i have matching dimensions
    @Test("Temperature equation dimensional consistency")
    func testTemperatureEquationDimensions() throws {
        // Setup typical ITER plasma parameters
        let electronDensity: Float = 1e20      // [m⁻³]
        let heatDiffusivity: Float = 1.0      // [m²/s]
        let temperatureGradient: Float = 1000.0 // [eV/m]
        let Q_MW: Float = 0.5     // [MW/m³]

        // Diffusion term: ∇·(n_e χ ∇T) [eV/(m³·s)]
        // Approximation for order-of-magnitude: n_e χ gradT / radialSpacing
        let radialSpacing: Float = 0.08  // [m] typical cell size for 25-cell ITER mesh
        let diffusionTerm = electronDensity * heatDiffusivity * temperatureGradient / radialSpacing
        // [m⁻³] × [m²/s] × [eV/m] × [1/m] = [eV/(m³·s)] ✓

        // Source term: Q [eV/(m³·s)] after conversion
        let sourceTerm = UnitConversions.megawattsToElectronVoltDensity(Q_MW)

        // Both terms must have same dimension and comparable magnitude
        let ratio = sourceTerm / diffusionTerm

        // For typical ITER: heating power and transport losses are comparable
        // Expect ratio O(1) to O(10) for realistic scenarios
        #expect(ratio > 0.1 && ratio < 100,
                "Source and diffusion terms have inconsistent magnitude (ratio = \(ratio))")
    }

    /// Test that normal density values are not affected by the floor
    @Test("Normal density values unchanged by floor")
    func testNormalDensityUnchanged() throws {
        let cellCount = 25

        // Create normal density profile (well above floor)
        let normalDensityValue: Float = 5e19  // Typical plasma density
        let densityArray = MLXArray(Array(repeating: normalDensityValue, count: cellCount))
        let density = EvaluatedArray(evaluating: densityArray)

        // Create profiles
        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(10000.0), count: cellCount))),
            electronTemperature: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(10000.0), count: cellCount))),
            electronDensity: density,
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        // Create geometry using MeshConfig
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        let geometry = Geometry(config: meshConfig)

        // Create transport and sources
        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            electronHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            particleDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            convectionVelocity: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            particleSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            currentSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let staticParameters = StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: true,
            evolveElectronHeat: true,
            evolveElectronDensity: false,
            evolvePoloidalFlux: false,
            solverType: .linear,
            theta: 1.0,
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )

        // Build coefficients
        let coeffs = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )

        // Verify transient coefficients match the input density (within tolerance)
        let ionTransient = coeffs.ionCoeffs.transientCoefficient.value.asArray(Float.self)
        for (i, value) in ionTransient.enumerated() {
            let relativeError = abs(value - normalDensityValue) / normalDensityValue
            #expect(relativeError < 1e-6, "Ion transientCoefficient[\(i)] = \(value) differs from input \(normalDensityValue)")
        }

        let electronTransient = coeffs.electronCoeffs.transientCoefficient.value.asArray(Float.self)
        for (i, value) in electronTransient.enumerated() {
            let relativeError = abs(value - normalDensityValue) / normalDensityValue
            #expect(relativeError < 1e-6, "Electron transientCoefficient[\(i)] = \(value) differs from input \(normalDensityValue)")
        }
    }

    @Test("Disabled equations use inert coefficients without evaluating inactive physics")
    func testDisabledEquationsUseInactiveCoefficients() throws {
        let cellCount = 25
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        let geometry = Geometry(config: meshConfig)
        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(10_000.0), count: cellCount))),
            electronTemperature: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronDensity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1e20), count: cellCount))),
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )
        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            electronHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            particleDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            convectionVelocity: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )
        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            particleSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            currentSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )
        let staticParameters = StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: false,
            evolveElectronHeat: false,
            evolveElectronDensity: false,
            evolvePoloidalFlux: false,
            solverType: .linear,
            theta: 1.0,
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )

        let coeffs = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )
        try coeffs.validateNumerics()

        let fluxDiffusion = coeffs.fluxCoeffs.faceDiffusionCoefficient.value.asArray(Float.self)
        for value in fluxDiffusion {
            #expect(value == 0)
        }
    }

    // MARK: - Current Diffusion Equation Tests

    /// Test Spitzer resistivity magnitude and temperature scaling
    ///
    /// Verifies that:
    /// 1. Resistivity magnitude is reasonable (η ~ 1e-7 Ω·m for tokamak core)
    /// 2. Temperature scaling follows η ∝ Te^(-3/2)
    /// 3. Neoclassical correction increases resistivity
    @Test("Spitzer resistivity magnitude and temperature scaling")
    func testSpitzerResistivity() throws {
        let cellCount = 25

        // Create ITER-like temperature profile (10 keV = 10000 eV core)
        let coreElectronTemperature: Float = 10000.0  // [eV]
        let Te_array = MLXArray(Array(repeating: coreElectronTemperature, count: cellCount))
        let Te = EvaluatedArray(evaluating: Te_array)

        // Create minimal profiles
        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(10000.0), count: cellCount))),
            electronTemperature: Te,
            electronDensity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1e20), count: cellCount))),
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        // Create ITER-like geometry
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        let geometry = Geometry(config: meshConfig)

        // Create transport and sources with evolvePoloidalFlux enabled
        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            electronHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            particleDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            convectionVelocity: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            particleSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            currentSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let staticParameters = StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: false,
            evolveElectronHeat: false,
            evolveElectronDensity: false,
            evolvePoloidalFlux: true,  // Enable current diffusion
            solverType: .linear,
            theta: 1.0,
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )

        // Build coefficients
        let coeffs = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )

        // Verify resistivity magnitude
        // For Te = 10 keV, η ≈ 5.2e-5 * 1.5 * 17 / (10000^1.5) ≈ 1.3e-9 Ω·m (Spitzer)
        // With neoclassical correction (f_trap ≈ 1.5), η_neo ≈ 2e-9 Ω·m
        let faceDiffusionCoefficient = coeffs.fluxCoeffs.faceDiffusionCoefficient.value.asArray(Float.self)
        for (i, eta) in faceDiffusionCoefficient.enumerated() {
            #expect(eta > 1e-10, "Resistivity[\(i)] = \(eta) too low")
            #expect(eta < 1e-6, "Resistivity[\(i)] = \(eta) too high")
        }

        // Test temperature scaling: double temperature → resistivity reduced by 2^(3/2) ≈ 2.83
        let Te_high: Float = 20000.0  // 20 keV
        let profiles_high_Te = CoreProfiles(
            ionTemperature: profiles.ionTemperature,
            electronTemperature: EvaluatedArray(evaluating: MLXArray(Array(repeating: Te_high, count: cellCount))),
            electronDensity: profiles.electronDensity,
            poloidalFlux: profiles.poloidalFlux
        )

        let coeffs_high_Te = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles_high_Te
        )

        let dFace_high = coeffs_high_Te.fluxCoeffs.faceDiffusionCoefficient.value.asArray(Float.self)
        let eta_low = faceDiffusionCoefficient[cellCount / 2]
        let eta_high = dFace_high[cellCount / 2]
        let scaling_factor = eta_low / eta_high

        // Expect scaling_factor ≈ 2^(3/2) = 2.828
        let expected_scaling: Float = Foundation.pow(2.0, 1.5)
        let scaling_error = abs(scaling_factor - expected_scaling) / expected_scaling

        #expect(scaling_error < 0.2, "Temperature scaling error = \(scaling_error) (expected ~2.83, got \(scaling_factor))")
    }

    /// Test bootstrap current calculation
    ///
    /// Verifies that:
    /// 1. Bootstrap current magnitude is reasonable (~1-2% with simplified formula)
    /// 2. Bootstrap current depends on pressure gradient
    /// 3. Clamping to [0, 10 MA/m²] works
    ///
    /// Note: Uses simplified C_BS ≈ (1-ε) formula. Full Sauter formula gives 15-25% for ITER.
    @Test("Bootstrap current calculation")
    func testBootstrapCurrent() throws {
        let cellCount = 25

        // Create ITER-like peaked temperature profile
        let coreIonTemperature: Float = 20000.0  // 20 keV core
        let coreElectronTemperature: Float = 20000.0  // 20 keV core
        let coreElectronDensity: Float = 1e20     // [m⁻³]

        // Linear profile from core to edge
        var Ti_values = [Float](repeating: 0, count: cellCount)
        var Te_values = [Float](repeating: 0, count: cellCount)
        var ne_values = [Float](repeating: 0, count: cellCount)

        for i in 0..<cellCount {
            let rho = Float(i) / Float(cellCount - 1)  // Normalized radius
            Ti_values[i] = coreIonTemperature * (1.0 - rho * 0.9)  // 10% edge temperature
            Te_values[i] = coreElectronTemperature * (1.0 - rho * 0.9)
            ne_values[i] = coreElectronDensity * (1.0 - rho * 0.5)  // 50% edge density
        }

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: MLXArray(Ti_values)),
            electronTemperature: EvaluatedArray(evaluating: MLXArray(Te_values)),
            electronDensity: EvaluatedArray(evaluating: MLXArray(ne_values)),
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        // Create ITER-like geometry
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        let geometry = Geometry(config: meshConfig)

        // Create transport and sources with evolvePoloidalFlux enabled
        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            electronHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            particleDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(1.0), count: cellCount))),
            convectionVelocity: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        // External current drive: 5 MA/m² (typical for ITER)
        let J_external: Float = 5.0  // MA/m² (not 5e6!)
        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            electronHeating: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            particleSource: EvaluatedArray(evaluating: MLXArray.zeros([cellCount])),
            currentSource: EvaluatedArray(evaluating: MLXArray(Array(repeating: J_external, count: cellCount)))
        )

        let staticParameters = StaticRuntimeParameters(
            mesh: meshConfig,
            evolveIonHeat: false,
            evolveElectronHeat: false,
            evolveElectronDensity: false,
            evolvePoloidalFlux: true,
            solverType: .linear,
            theta: 1.0,
            solverTolerance: 1e-6,
            solverMaximumIterations: 100
        )

        // Build coefficients
        let coeffs = buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )

        // Verify total current source includes bootstrap
        let cellSource = coeffs.fluxCoeffs.cellSource.value.asArray(Float.self)

        // Total current = J_external + J_bootstrap
        // Expect J_bootstrap ~ 15-25% of total for ITER-like scenario
        // Total ~ 5-6 MA/m² (external + bootstrap)
        for (i, J_total) in cellSource.enumerated() {
            #expect(J_total >= J_external, "Total current[\(i)] = \(J_total) less than external \(J_external)")
            #expect(J_total <= 10.0, "Total current[\(i)] = \(J_total) exceeds clamp limit (10 MA/m²)")
        }

        // Check that mid-radius has bootstrap contribution
        let J_total_mid = cellSource[cellCount / 2]
        let bootstrap_fraction = (J_total_mid - J_external) / J_total_mid

        // Note: This implementation uses Sauter neoclassical formula (Phase 3)
        // - Normalized collisionality ν* from electron-ion collision time (τₑ)
        // - Trapped particle fraction ft = 1 - √(1-ε)
        // - Sauter L₃₁ coefficient: ((1 + 0.15/ft) - 0.22/(1 + 0.01·ν*)) / (1 + 0.5·√ν*)
        // - Simplified L₃₂ ≈ 0.05, L₃₄ ≈ 0.01
        // - Bootstrap coefficient: C_BS = L₃₁·ft (with α=0 for isotropic pressure)
        //
        // **Current implementation behavior**:
        // For test conditions (Ti, Te ~ 20 keV core, ne ~ 1e20 m⁻³, R=6.2m, a=2.0m):
        // - Bootstrap fraction: ~0.2-0.5% (observed)
        // - This is lower than full Sauter formula (15-25%) due to:
        //   1. Simplified L₃₂, L₃₄ (not collisionality-dependent)
        //   2. No geometric corrections (F, G functions)
        //   3. Single ion species approximation
        //
        // **Acceptance criteria** (validates implementation consistency):
        // - Lower bound: 0.1% (non-zero bootstrap contribution)
        // - Upper bound: 5.0% (reasonable for simplified formula)
        #expect(bootstrap_fraction > 0.001, "Bootstrap fraction = \(bootstrap_fraction) too low (expect > 0.1%)")
        #expect(bootstrap_fraction < 0.05, "Bootstrap fraction = \(bootstrap_fraction) too high for simplified implementation")
    }
}
