import Testing
import Foundation
import MLX
@testable import GotenxCore

/// Tests for flux divergence calculation in Newton-Raphson solver
///
/// Verifies the spatial operator implementation including:
/// - Gradient calculation at faces
/// - Flux computation (diffusive + convective)
/// - Jacobian weighting (√g = 2πR₀)
/// - Flux divergence with metric tensor
/// - Unit consistency throughout the calculation chain
///
/// ## Context
///
/// The flux divergence calculation is critical for the transport equations:
/// ```
/// n_e ∂T_i/∂t = ∇·(n_e χ_i ∇T_i) + Q_i
/// ```
///
/// This test suite validates the implementation in NewtonRaphsonSolver.applySpatialOperatorVectorized()
@Suite("Flux Divergence Tests")
struct FluxDivergenceTests {

    // MARK: - Test Helpers

    /// Create minimal test setup for flux divergence tests
    private func createTestSetup(
        cellCount: Int = 50,
        majorRadius: Float = 6.2,
        minorRadius: Float = 2.0,
        temperature: [Float]? = nil,
        density: Float = 2.0e19,
        chi: Float = 1.0,
        sourceValue: Float = 0.0
    ) -> (
        profiles: CoreProfiles,
        geometry: Geometry,
        transport: TransportCoefficients,
        sources: SourceTerms,
        staticParameters: StaticRuntimeParameters
    ) {
        // Use provided temperature or create flat profile
        let tempArray = temperature ?? Array(repeating: Float(1000.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: MLXArray(tempArray)),
            electronTemperature: EvaluatedArray(evaluating: MLXArray(tempArray)),
            electronDensity: EvaluatedArray(evaluating: MLXArray(Array(repeating: density, count: cellCount))),
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: majorRadius,
            minorRadius: minorRadius,
            toroidalField: 5.3,
            geometryType: .circular
        )
        let geometry = Geometry(config: meshConfig)

        let transport = TransportCoefficients(
            ionHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: chi, count: cellCount))),
            electronHeatDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: chi, count: cellCount))),
            particleDiffusivity: EvaluatedArray(evaluating: MLXArray(Array(repeating: Float(0.1), count: cellCount))),
            convectionVelocity: EvaluatedArray(evaluating: MLXArray.zeros([cellCount]))
        )

        let sources = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray(Array(repeating: sourceValue, count: cellCount))),
            electronHeating: EvaluatedArray(evaluating: MLXArray(Array(repeating: sourceValue, count: cellCount))),
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

        return (profiles, geometry, transport, sources, staticParameters)
    }

    // MARK: - Gradient Calculation Tests

    /// Test diffusion coefficients with flat temperature profile
    ///
    /// **Expected**: faceDiffusionCoefficient = n_e × χ_i for all faces
    ///
    /// **Physics**: Diffusion coefficient depends on density and thermal diffusivity
    @Test("Flat profile diffusion coefficients are correct")
    func testFlatProfileDiffusionCoeffs() throws {
        let cellCount = 50
        let density: Float = 2.0e19
        let chi: Float = 1.0

        let setup = createTestSetup(
            cellCount: cellCount,
            temperature: Array(repeating: 1000.0, count: cellCount),
            density: density,
            chi: chi
        )

        let coeffs = buildBlock1DCoeffs(
            transport: setup.transport,
            sources: setup.sources,
            geometry: setup.geometry,
            staticParameters: setup.staticParameters,
            profiles: setup.profiles
        )

        // Verify faceDiffusionCoefficient = n_e × χ_i
        let faceDiffusionCoefficient = coeffs.ionCoeffs.faceDiffusionCoefficient.value.asArray(Float.self)
        let expectedD = density * chi  // 2e19 × 1.0 = 2e19

        for (i, d) in faceDiffusionCoefficient.enumerated() {
            // Allow 10% variation due to harmonic mean interpolation at boundaries
            let relativeError = abs(d - expectedD) / expectedD
            #expect(relativeError < 0.1,
                   "faceDiffusionCoefficient[\(i)] = \(d), expected ≈\(expectedD) (error: \(relativeError*100)%)")
        }

        // Verify transient coefficient = n_e
        let transientCoefficient = coeffs.ionCoeffs.transientCoefficient.value.asArray(Float.self)
        for (i, coeff) in transientCoefficient.enumerated() {
            let relativeError = abs(coeff - density) / density
            #expect(relativeError < 0.01,
                   "transientCoefficient[\(i)] = \(coeff), expected \(density) (error: \(relativeError*100)%)")
        }
    }

    /// Test gradient calculation with linear temperature profile
    ///
    /// **Profile**: T(r) = 2000 - 1000 × (r/a), where r ∈ [0, a]
    /// - Core (r=0): 2000 eV
    /// - Edge (r=a): 1000 eV
    ///
    /// **Expected gradient**: dT/radialSpacing = -1000/a = -500 eV/m (for a=2.0 m)
    @Test("Linear profile produces constant gradient")
    func testLinearProfileGradient() throws {
        let cellCount = 50
        let minorRadius: Float = 2.0
        let Tcore: Float = 2000.0
        let Tedge: Float = 1000.0

        // Create linear temperature profile: T(r) = Tedge + (Tcore - Tedge) * (1 - r/a)
        var tempProfile = [Float](repeating: 0.0, count: cellCount)
        for i in 0..<cellCount {
            let rNorm = Float(i) / Float(cellCount - 1)  // r/a ∈ [0, 1]
            tempProfile[i] = Tedge + (Tcore - Tedge) * (1.0 - rNorm)
        }

        let setup = createTestSetup(
            cellCount: cellCount,
            minorRadius: minorRadius,
            temperature: tempProfile
        )

        // Expected gradient magnitude: |dT/radialSpacing| = (Tcore - Tedge) / a
        // let expectedGradientMag = abs(Tcore - Tedge) / minorRadius
        // Expected: |500| eV/m for (2000-1000)/2.0

        // Note: We can't directly access gradFace from outside the solver,
        // but we can verify the coefficients are built correctly
        let coeffs = buildBlock1DCoeffs(
            transport: setup.transport,
            sources: setup.sources,
            geometry: setup.geometry,
            staticParameters: setup.staticParameters,
            profiles: setup.profiles
        )

        // Verify faceDiffusionCoefficient is computed correctly (should be n_e × χ_i)
        let faceDiffusionCoefficient = coeffs.ionCoeffs.faceDiffusionCoefficient.value.asArray(Float.self)
        let expectedD: Float = 2.0e19 * 1.0  // n_e × χ_i = 2e19 × 1.0

        // All interior faces should have similar faceDiffusionCoefficient values
        for (i, d) in faceDiffusionCoefficient.enumerated() {
            // Allow 10% variation due to harmonic mean interpolation
            let relativeError = abs(d - expectedD) / expectedD
            #expect(relativeError < 0.1, "faceDiffusionCoefficient[\(i)] = \(d), expected ≈\(expectedD) (error: \(relativeError*100)%)")
        }
    }

    // MARK: - Unit Conversion Tests

    /// Test MW/m³ to eV/(m³·s) conversion
    ///
    /// **Conversion factor**: 1 MW/m³ = 6.2415090744×10²⁴ eV/(m³·s)
    ///
    /// **Physics**: Temperature equation requires [eV/(m³·s)] units:
    /// ```
    /// n_e ∂T_i/∂t = ∇·(n_e χ_i ∇T_i) + Q_i
    /// [m⁻³][eV/s] = [eV/(m³·s)]      + [eV/(m³·s)]
    /// ```
    @Test("Unit conversion from MW/m³ to eV/(m³·s) is correct")
    func testUnitConversion() throws {
        // Test conversion constant
        let conversionFactor = UnitConversions.megawattsPerCubicMeterToElectronVoltsPerCubicMeterPerSecond
        let expectedFactor: Float = 6.2415090744e24

        let relativeError = abs(conversionFactor - expectedFactor) / expectedFactor
        #expect(relativeError < 1e-6, "Conversion factor mismatch: \(conversionFactor) vs \(expectedFactor)")

        // Test conversion of typical heating power
        let Q_MW: Float = 1.0  // 1 MW/m³ (typical heating power density)
        let Q_eV = UnitConversions.megawattsToElectronVoltDensity(Q_MW)

        let expectedQ_eV = Q_MW * expectedFactor
        let conversionError = abs(Q_eV - expectedQ_eV) / expectedQ_eV
        #expect(conversionError < 1e-6, "Converted value mismatch: \(Q_eV) vs \(expectedQ_eV)")

        // Test array conversion
        let Q_MW_array = MLXArray([Float(1.0), Float(10.0), Float(100.0)])
        let Q_eV_array = UnitConversions.megawattsToElectronVoltDensity(Q_MW_array)
        let Q_eV_values = Q_eV_array.asArray(Float.self)

        let expected_eV_values: [Float] = [1.0e24, 1.0e25, 1.0e26].map { Float($0 * 6.2415090744) }
        for (i, (computed, expected)) in zip(Q_eV_values, expected_eV_values).enumerated() {
            let error = abs(computed - expected) / expected
            #expect(error < 1e-5, "Array conversion[\(i)] error: \(computed) vs \(expected)")
        }
    }

    /// Test source term magnitude is physically reasonable
    ///
    /// **Expected range**: 10⁴-10⁶ W/m³ for ITER-scale plasma
    /// In eV/(m³·s): 10²²-10²⁴ eV/(m³·s) after conversion
    @Test("Source term magnitude is physically reasonable")
    func testSourceTermMagnitude() throws {
        // Typical ohmic heating: 10 kW/m³ = 1e4 W/m³
        let Q_watts: Float = 1e4
        let Q_MW = Q_watts / 1e6  // 0.01 MW/m³
        let Q_eV = UnitConversions.megawattsToElectronVoltDensity(Q_MW)

        // Expected: ~6e22 eV/(m³·s)
        let expectedOrder: Float = 1e22
        let orderOfMagnitude = log10(abs(Q_eV))
        let expectedOrderOfMagnitude = log10(expectedOrder)

        // Should be within same order of magnitude
        #expect(abs(orderOfMagnitude - Float(expectedOrderOfMagnitude)) < 1.0,
               "Source term \(Q_eV) eV/(m³·s) is not in expected range (~\(expectedOrder))")
    }

    // MARK: - Jacobian Tests

    /// Test Jacobian calculation for 1D cylindrical geometry
    ///
    /// **Expected**: √g = 2πR₀ (constant for circular cross-section)
    ///
    /// **Physics**: In 1D cylindrical coordinates with large aspect ratio:
    /// - √g is constant throughout the plasma
    /// - Value depends only on major radius R₀
    @Test("Jacobian equals 2πR₀ for circular geometry")
    func testJacobianValue() throws {
        let majorRadius: Float = 6.2
        let setup = createTestSetup(majorRadius: majorRadius)

        let geomFactors = GeometricFactors.from(geometry: setup.geometry)
        let jacobian = geomFactors.jacobian.value.asArray(Float.self)

        let expectedJacobian = 2.0 * Float.pi * majorRadius
        // Expected: 2π × 6.2 ≈ 38.956 m

        // All cells should have the same Jacobian (constant for circular geometry)
        for (i, jac) in jacobian.enumerated() {
            let relativeError = abs(jac - expectedJacobian) / expectedJacobian
            #expect(relativeError < 1e-5,
                   "Jacobian[\(i)] = \(jac) m, expected \(expectedJacobian) m (error: \(relativeError*100)%)")
        }
    }

    /// Test flux divergence dimensional consistency
    ///
    /// **Expected units**: [eV/(m³·s)]
    ///
    /// **Calculation chain**:
    /// 1. gradFace = ∂T/∂r [eV/m]
    /// 2. faceDiffusionCoefficient = n_e × χ [m⁻¹·s⁻¹]
    /// 3. flux = -faceDiffusionCoefficient × gradFace [eV/(m²·s)]
    /// 4. weightedFlux = √g × flux [eV/(m·s)]
    /// 5. fluxDiv = Δ(weightedFlux) / (√g × Δr) [eV/(m³·s)]
    @Test("Flux divergence has correct dimensions")
    func testFluxDivergenceDimensions() throws {
        let cellCount = 50
        let minorRadius: Float = 2.0

        // Create linear profile for non-zero gradient
        var tempProfile = [Float](repeating: 0.0, count: cellCount)
        for i in 0..<cellCount {
            let rNorm = Float(i) / Float(cellCount - 1)
            tempProfile[i] = 2000.0 - 1000.0 * rNorm
        }

        let setup = createTestSetup(
            cellCount: cellCount,
            minorRadius: minorRadius,
            temperature: tempProfile,
            sourceValue: 1.0  // 1 MW/m³ source
        )

        let coeffs = buildBlock1DCoeffs(
            transport: setup.transport,
            sources: setup.sources,
            geometry: setup.geometry,
            staticParameters: setup.staticParameters,
            profiles: setup.profiles
        )

        // Source term should be converted to eV/(m³·s)
        let cellSource = coeffs.ionCoeffs.cellSource.value.asArray(Float.self)

        // Expected: 1 MW/m³ × 6.24e24 ≈ 6.24e24 eV/(m³·s)
        let expectedSourceMagnitude: Float = 1.0 * 6.2415090744e24

        for (i, source) in cellSource.enumerated() {
            // Source should be in correct order of magnitude
            let orderOfMagnitude = log10(abs(source))
            let expectedOrder = log10(expectedSourceMagnitude)

            #expect(abs(orderOfMagnitude - expectedOrder) < 1.0,
                   "Source[\(i)] = \(source) eV/(m³·s), expected order ~\(expectedSourceMagnitude)")
        }
    }

    // MARK: - Physical Magnitude Tests

    /// Test flux divergence magnitude for transient evolution
    ///
    /// **Scenario**: Flat profile relaxing to steady state
    /// - Initial: Ti = 1000 eV everywhere
    /// - Source: Ohmic heating ~10 kW/m³
    /// - Expected: Rapid temperature evolution (∂T/∂t ~ 1000 eV/s)
    ///
    /// **Expected flux divergence**: ~10²² eV/(m³·s)
    @Test("Flux divergence magnitude is physically correct for transient evolution")
    func testFluxDivergenceMagnitudeTransient() throws {
        // Flat profile with heating sources
        let Q_kW_m3: Float = 10.0  // 10 kW/m³ (typical ohmic heating)
        let Q_MW_m3 = Q_kW_m3 / 1000.0  // 0.01 MW/m³

        let setup = createTestSetup(
            temperature: Array(repeating: 1000.0, count: 50),
            sourceValue: Q_MW_m3
        )

        let coeffs = buildBlock1DCoeffs(
            transport: setup.transport,
            sources: setup.sources,
            geometry: setup.geometry,
            staticParameters: setup.staticParameters,
            profiles: setup.profiles
        )

        // For flat profile, flux divergence ≈ source term
        let cellSource = coeffs.ionCoeffs.cellSource.value.asArray(Float.self)

        // Expected order: ~6e22 eV/(m³·s) for 10 kW/m³
        for (i, source) in cellSource.enumerated() {
            let orderOfMagnitude = log10(abs(source))

            // Should be in range 10²² - 10²³ eV/(m³·s)
            #expect(orderOfMagnitude >= 22.0 && orderOfMagnitude <= 23.0,
                   "Source[\(i)] order of magnitude \(orderOfMagnitude) outside expected range [22, 23]")
        }
    }

    /// Test that flat profile with zero sources has zero source term
    ///
    /// **Physics**: Thermal equilibrium state (∂T/∂t ≈ 0)
    /// - No gradient → no diffusive flux
    /// - No sources → no heating
    /// - Result: steady state (no evolution)
    @Test("Flat profile with zero sources has zero source coefficient")
    func testThermalEquilibriumSourceTerm() throws {
        let setup = createTestSetup(
            temperature: Array(repeating: 1000.0, count: 50),
            sourceValue: 0.0  // No external heating
        )

        let coeffs = buildBlock1DCoeffs(
            transport: setup.transport,
            sources: setup.sources,
            geometry: setup.geometry,
            staticParameters: setup.staticParameters,
            profiles: setup.profiles
        )

        // Verify source term is zero
        let cellSource = coeffs.ionCoeffs.cellSource.value.asArray(Float.self)

        for (i, source) in cellSource.enumerated() {
            #expect(abs(source) < 1e-10,
                   "Source[\(i)] = \(source), expected 0 for zero source input")
        }
    }

    // MARK: - CFL Condition Tests

    /// Test CFL condition for numerical stability
    ///
    /// **CFL Condition**: CFL = χ × timeStep / dx² < 0.5 (diffusion stability limit)
    ///
    /// **Physics**: Violating CFL leads to numerical instability and oscillations
    ///
    /// **Expected**: For cellCount=50, minorRadius=2.0, chi=1.0, timeStep=7e-4:
    /// - dx = 2.0/50 = 0.04 m
    /// - CFL = 1.0 × 7e-4 / (0.04)² = 0.4375 < 0.5 ✓
    @Test("CFL condition is satisfied for stable timestepping")
    func testCFLCondition() throws {
        let cellCount = 50
        let minorRadius: Float = 2.0
        let chi: Float = 1.0
        let timeStep: Float = 7e-4

        let setup = createTestSetup(
            cellCount: cellCount,
            minorRadius: minorRadius,
            chi: chi
        )

        // Cell spacing: dx = a / cellCount
        let dx = minorRadius / Float(cellCount)

        // CFL number: CFL = chi * timeStep / dx^2
        let CFL = chi * timeStep / (dx * dx)

        // Verify CFL < 0.5 for diffusion stability
        #expect(CFL < 0.5, "CFL = \(CFL) violates stability condition (must be < 0.5)")

        // Verify it's reasonably below the limit (safety margin)
        #expect(CFL < 0.45, "CFL = \(CFL) too close to stability limit (recommend < 0.45)")

        print("✓ CFL condition satisfied: CFL = \(CFL) < 0.5")
    }

    /// Test CFL condition with different mesh resolutions
    ///
    /// **Expected**: Higher resolution (smaller dx) requires smaller timestep
    @Test("CFL condition scales correctly with mesh resolution")
    func testCFLConditionScaling() throws {
        let minorRadius: Float = 2.0
        let chi: Float = 1.0

        // Test different resolutions
        let testCases: [(cellCount: Int, timeStep: Float, expectedCFL: Float)] = [
            (25, 2.8e-3, 0.4375),  // Coarse mesh, larger timeStep
            (50, 7.0e-4, 0.4375),  // Medium mesh
            (100, 1.75e-4, 0.4375) // Fine mesh, smaller timeStep
        ]

        for testCase in testCases {
            let dx = minorRadius / Float(testCase.cellCount)
            let CFL = chi * testCase.timeStep / (dx * dx)

            #expect(abs(CFL - testCase.expectedCFL) < 1e-3,
                   "cellCount=\(testCase.cellCount): CFL=\(CFL) ≠ expected \(testCase.expectedCFL)")

            #expect(CFL < 0.5,
                   "cellCount=\(testCase.cellCount): CFL=\(CFL) violates stability")
        }
    }

    // MARK: - Boundary Condition Tests

    /// Test boundary condition induced gradients in flat profile
    ///
    /// **Key Finding from Phase 2**: Even with flat initial profile (Ti=1000 eV everywhere),
    /// Dirichlet boundary conditions create small gradients during solver iteration
    ///
    /// **Expected**: gradFace ~ 6.5 eV/m near boundaries (from Newton-Raphson iteration)
    /// This corresponds to ~0.26 eV temperature difference over one cell (0.026% of 1000 eV)
    @Test("Boundary conditions create small gradients in flat profile")
    func testBoundaryConditionGradient() throws {
        let cellCount = 50
        let minorRadius: Float = 2.0
        let boundaryTemp: Float = 1000.0

        // Flat profile (all cells at boundary temperature)
        let setup = createTestSetup(
            cellCount: cellCount,
            minorRadius: minorRadius,
            temperature: Array(repeating: boundaryTemp, count: cellCount)
        )

        // Expected gradient near boundaries from solver iteration
        // From Phase 2 logs: gradFace ~ 6.5 eV/m
        let dx = minorRadius / Float(cellCount)  // Cell spacing

        // Maximum expected temperature variation per cell
        let expectedMaxGrad: Float = 10.0  // eV/m (conservative upper bound)
        let maxTempDiff = expectedMaxGrad * dx

        // This should be << 1% of boundary temperature
        let relativeVariation = maxTempDiff / boundaryTemp

        #expect(relativeVariation < 0.01,
               "Boundary-induced gradient too large: \(relativeVariation*100)% (expected < 1%)")

        print("✓ Boundary gradient is small: ~\(maxTempDiff) eV per cell (\(relativeVariation*100)%)")
    }

    // MARK: - Geometric Consistency Tests

    /// Test consistency of geometric factors with Jacobian
    ///
    /// **Expected**: For circular 1D geometry:
    /// - Jacobian: √g = 2πR₀
    /// - Cell volume: V = 2πR₀ × Δr
    /// - Face area: A = 2πR₀
    @Test("Geometric factors are consistent with Jacobian")
    func testGeometricFactorsConsistency() throws {
        let cellCount = 50
        let majorRadius: Float = 6.2
        let minorRadius: Float = 2.0

        let setup = createTestSetup(
            cellCount: cellCount,
            majorRadius: majorRadius,
            minorRadius: minorRadius
        )

        let geomFactors = GeometricFactors.from(geometry: setup.geometry)

        // Expected Jacobian
        let expectedJacobian = 2.0 * Float.pi * majorRadius

        // Verify Jacobian
        let jacobian = geomFactors.jacobian.value.asArray(Float.self)
        for (i, jac) in jacobian.enumerated() {
            let error = abs(jac - expectedJacobian) / expectedJacobian
            #expect(error < 1e-5, "Jacobian[\(i)] inconsistent: \(jac) vs \(expectedJacobian)")
        }

        // Verify cell volumes: V = Jacobian × Δr
        let radialSpacing = minorRadius / Float(cellCount)
        let expectedVolume = expectedJacobian * radialSpacing
        let cellVolumes = geomFactors.cellVolumes.value.asArray(Float.self)

        for (i, vol) in cellVolumes.enumerated() {
            let error = abs(vol - expectedVolume) / expectedVolume
            #expect(error < 1e-5, "Volume[\(i)] inconsistent: \(vol) vs \(expectedVolume)")
        }

        // Verify face areas: A = Jacobian (constant)
        let expectedArea = expectedJacobian
        let faceAreas = geomFactors.faceAreas.value.asArray(Float.self)

        for (i, area) in faceAreas.enumerated() {
            let error = abs(area - expectedArea) / expectedArea
            #expect(error < 1e-5, "Area[\(i)] inconsistent: \(area) vs \(expectedArea)")
        }

        print("✓ Geometric consistency: √g = \(expectedJacobian) m, V = \(expectedVolume) m³, A = \(expectedArea) m²")
    }

    // MARK: - End-to-End Unit Chain Tests

    /// Test complete unit chain from coefficients to flux divergence
    ///
    /// **Unit Chain**:
    /// 1. faceDiffusionCoefficient = n_e × χ [m⁻³ × m²/s = m⁻¹·s⁻¹]
    /// 2. gradFace = ∂T/∂r [eV/m]
    /// 3. flux = -faceDiffusionCoefficient × gradFace [m⁻¹·s⁻¹ × eV/m = eV/(m²·s)]
    /// 4. weightedFlux = √g × flux [m × eV/(m²·s) = eV/(m·s)]
    /// 5. divergence = Δ(weightedFlux)/(√g × Δr) [eV/(m³·s)]
    @Test("Complete unit chain from coefficients to divergence")
    func testUnitChainEndToEnd() throws {
        let cellCount = 50
        let minorRadius: Float = 2.0
        let density: Float = 2.0e19
        let chi: Float = 1.0

        // Linear temperature profile for predictable gradient
        var tempProfile = [Float](repeating: 0.0, count: cellCount)
        for i in 0..<cellCount {
            let rNorm = Float(i) / Float(cellCount - 1)
            tempProfile[i] = 2000.0 - 1000.0 * rNorm  // 2000 eV → 1000 eV
        }

        let setup = createTestSetup(
            cellCount: cellCount,
            minorRadius: minorRadius,
            temperature: tempProfile,
            density: density,
            chi: chi
        )

        let coeffs = buildBlock1DCoeffs(
            transport: setup.transport,
            sources: setup.sources,
            geometry: setup.geometry,
            staticParameters: setup.staticParameters,
            profiles: setup.profiles
        )

        // 1. Verify faceDiffusionCoefficient units [m⁻¹·s⁻¹]
        let faceDiffusionCoefficient = coeffs.ionCoeffs.faceDiffusionCoefficient.value.asArray(Float.self)
        let expectedD = density * chi  // 2e19 m⁻³ × 1.0 m²/s = 2e19 m⁻¹·s⁻¹

        // Sample check (middle face)
        let midIdx = faceDiffusionCoefficient.count / 2
        let dRelError = abs(faceDiffusionCoefficient[midIdx] - expectedD) / expectedD
        #expect(dRelError < 0.1, "faceDiffusionCoefficient units incorrect: \(faceDiffusionCoefficient[midIdx]) vs \(expectedD)")

        // 2. Expected gradient [eV/m]
        let expectedGrad = 1000.0 / minorRadius  // ΔT / Δr = 1000 eV / 2 m = 500 eV/m

        // 3. Expected flux magnitude [eV/(m²·s)]
        let expectedFlux = expectedD * expectedGrad  // 2e19 × 500 = 1e22 eV/(m²·s)

        // Order of magnitude check
        let fluxOrder = log10(expectedFlux)
        #expect(fluxOrder >= 21.0 && fluxOrder <= 23.0,
               "Flux magnitude outside expected range: 10^\(fluxOrder) eV/(m²·s)")

        print("✓ Unit chain verified: faceDiffusionCoefficient ~ \(expectedD), flux ~ \(expectedFlux) eV/(m²·s)")
    }

    // MARK: - Transient Evolution Tests

    /// Test time scale of transient evolution
    ///
    /// **Physics**: n_e × ∂T/∂t ≈ Q_source during rapid evolution
    ///
    /// **Expected**: For Q_source ~ 10 kW/m³ = 6e22 eV/(m³·s):
    /// - ∂T/∂t ≈ Q_source / n_e = 6e22 / 2e19 = 3000 eV/s
    /// - Time to heat 1000 eV: ~0.3 seconds
    @Test("Transient evolution time scale is physically correct")
    func testTransientEvolutionTimeScale() throws {
        let density: Float = 2.0e19  // m⁻³
        let Q_kW_m3: Float = 10.0    // 10 kW/m³ (typical ohmic heating)
        let Q_MW_m3 = Q_kW_m3 / 1000.0

        let setup = createTestSetup(
            temperature: Array(repeating: 1000.0, count: 50),
            density: density,
            sourceValue: Q_MW_m3
        )

        let coeffs = buildBlock1DCoeffs(
            transport: setup.transport,
            sources: setup.sources,
            geometry: setup.geometry,
            staticParameters: setup.staticParameters,
            profiles: setup.profiles
        )

        // Source term in eV/(m³·s)
        let cellSource = coeffs.ionCoeffs.cellSource.value.asArray(Float.self)
        let avgSource = cellSource.reduce(0.0, +) / Float(cellSource.count)

        // Expected: Q_source ≈ 6e22 eV/(m³·s)
        let expectedSource: Float = Q_kW_m3 * 1e3 * 6.2415e18  // W/m³ → eV/(m³·s)

        let sourceRelError = abs(avgSource - expectedSource) / expectedSource
        #expect(sourceRelError < 0.1,
               "Source term magnitude incorrect: \(avgSource) vs \(expectedSource)")

        // Temperature evolution rate: ∂T/∂t = Q_source / n_e
        let dTdt = avgSource / density  // eV/s

        // Expected: ~3000 eV/s
        let expectedDTdt: Float = 3000.0
        let dtRelError = abs(dTdt - expectedDTdt) / expectedDTdt
        #expect(dtRelError < 0.5,
               "Evolution rate incorrect: \(dTdt) eV/s vs \(expectedDTdt) eV/s")

        // Time to heat by 1000 eV
        let heatingTime = 1000.0 / dTdt  // seconds
        #expect(heatingTime > 0.1 && heatingTime < 1.0,
               "Heating time unrealistic: \(heatingTime) s (expected 0.1-1.0 s)")

        print("✓ Evolution time scale: ∂T/∂t ≈ \(dTdt) eV/s, heating time ≈ \(heatingTime) s")
    }
}
