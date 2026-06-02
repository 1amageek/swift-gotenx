// TurbulenceTransitionTests.swift
// Tests for turbulence transition models (ITG → RI)

import Testing
import MLX
import Foundation
@testable import GotenxCore

// MARK: - Test Helpers

/// Helper function to create test geometry
func createTestGeometry(cellCount: Int) -> Geometry {
    let mesh = MeshConfig(
        cellCount: cellCount,
        majorRadius: 6.2,
        minorRadius: 2.0,
        toroidalField: 5.3,
        geometryType: .circular
    )
    return createGeometry(from: mesh)
}

// MARK: - Gradient Computation Tests

@Suite("Gradient Computation Tests")
struct GradientComputationTestSuite {

    @Test("Gradient computation with linear profile")
    func testGradientComputationAccuracy() throws {
        let cellCount = 100
        let radii = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        // Linear profile: f(r) = 1000 + 500*r
        // Expected gradient: df/radialSpacing = 500 everywhere
        let linearProfile = 1000.0 + 500.0 * radii

        let gradient = GradientComputation.computeGradient(
            variable: linearProfile,
            radii: radii
        )

        let gradArray = gradient.asArray(Float.self)

        // Check interior points (central differences are accurate for linear profiles)
        // Float32 precision: expect ~0.003 error due to finite precision
        for i in 1..<(cellCount - 1) {
            #expect(abs(gradArray[i] - 500.0) < 0.01)  // 0.01 tolerance for Float32
        }
    }

    @Test("Gradient scale length computation")
    func testGradientLengthComputation() throws {
        let cellCount = 50
        let radii = MLXArray.linspace(Float(0.1), Float(1.0), count: cellCount)

        // Exponential profile: f(r) = 1000 * exp(-r/0.2)
        // L = |f| / |df/radialSpacing| = 0.2 (constant)
        let L_expected: Float = 0.2
        let profile = 1000.0 * exp(-radii / L_expected)

        let L = GradientComputation.computeGradientLength(
            variable: profile,
            radii: radii
        )

        let L_array = L.asArray(Float.self)

        // Check interior points (should be close to 0.2)
        for i in 10..<(cellCount - 10) {
            #expect(abs(L_array[i] - L_expected) < 0.01)
        }
    }

    @Test("Pressure gradient length computation")
    func testPressureGradientLength() throws {
        let cellCount = 100
        let radii = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        // Create simple profiles
        let Te = MLXArray.full([cellCount], values: MLXArray(5000.0))
        let Ti = MLXArray.full([cellCount], values: MLXArray(5000.0))
        let ne = MLXArray.full([cellCount], values: MLXArray(1e20))

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount))
        )

        let L_p = GradientComputation.computePressureGradientLength(
            profiles: profiles,
            radii: radii
        )

        let L_p_array = L_p.asArray(Float.self)
        #expect(L_p_array[cellCount / 2] > 1.0)
    }
}

// MARK: - Plasma Physics Tests

@Suite("Plasma Physics Tests")
struct PlasmaPhysicsTestSuite {

    @Test("Spitzer resistivity units and scaling")
    func testSpitzerResistivityUnits() throws {
        let Te_eV = MLXArray([Float](repeating: 5000.0, count: 10))
        let ne_m3 = MLXArray([Float](repeating: 1e20, count: 10))

        let eta = PlasmaPhysics.spitzerResistivity(
            Te_eV: Te_eV,
            ne_m3: ne_m3,
            effectiveCharge: 1.0
        )

        let etaArray = eta.asArray(Float.self)

        // Expected: η ~ 2-3×10⁻⁹ Ω·m for T=5keV, n=10²⁰ m⁻³
        #expect(etaArray[0] > 1e-9)
        #expect(etaArray[0] < 1e-8)

        // Check temperature scaling: η ∝ T^(-3/2)
        let Te_high = MLXArray([Float](repeating: 10000.0, count: 10))
        let eta_high = PlasmaPhysics.spitzerResistivity(
            Te_eV: Te_high,
            ne_m3: ne_m3,
            effectiveCharge: 1.0
        )
        let etaHighArray = eta_high.asArray(Float.self)

        let ratio = etaArray[0] / etaHighArray[0]
        // T^(-3/2) scaling: (T_high/T_low)^(3/2) = 2^(3/2) ≈ 2.83
        // Note: Coulomb logarithm also depends on T, so ratio is slightly higher
        let expectedRatio: Float = 2.828  // 2^(3/2)

        #expect(abs(ratio - expectedRatio) < 0.15)  // Wider tolerance due to ln(Λ) temperature dependence
    }

    @Test("Plasma beta calculation")
    func testPlasmaBeta() throws {
        let cellCount = 10
        let Te = MLXArray([Float](repeating: 5000.0, count: cellCount))
        let Ti = MLXArray([Float](repeating: 5000.0, count: cellCount))
        let ne = MLXArray([Float](repeating: 1e20, count: cellCount))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let B = MLXArray([Float](repeating: 5.3, count: cellCount))

        let beta = PlasmaPhysics.plasmaBeta(
            profiles: profiles,
            magneticField: B
        )

        let betaArray = beta.asArray(Float.self)

        // Expected β ~ 0.01-0.05 for tokamaks
        #expect(betaArray[0] > 0.001)
        #expect(betaArray[0] < 0.2)
    }

    @Test("Total magnetic field shape consistency")
    func testTotalMagneticFieldShape() throws {
        let cellCount = 100

        // Case 1: No poloidal field
        let B_total_no_pol = PlasmaPhysics.totalMagneticField(
            toroidalField: 5.3,
            poloidalField: nil,
            cellCount: cellCount
        )

        #expect(B_total_no_pol.shape == [cellCount])

        let B_array_no_pol = B_total_no_pol.asArray(Float.self)
        #expect(B_array_no_pol.count == cellCount)
        #expect(abs(B_array_no_pol[0] - 5.3) < 1e-6)

        // Case 2: With poloidal field
        let B_pol = MLXArray.full([cellCount], values: MLXArray(0.5))
        let B_total_with_pol = PlasmaPhysics.totalMagneticField(
            toroidalField: 5.3,
            poloidalField: B_pol,
            cellCount: cellCount
        )

        #expect(B_total_with_pol.shape == [cellCount])

        let B_array_with_pol = B_total_with_pol.asArray(Float.self)
        // √(5.3² + 0.5²) = √(28.09 + 0.25) = √28.34 ≈ 5.324
        let expected_total: Float = 5.324
        #expect(abs(B_array_with_pol[0] - expected_total) < 1e-2)
    }

    @Test("Ion sound Larmor radius and isotope scaling")
    func testIonSoundLarmorRadius() throws {
        let cellCount = 10
        let Te = MLXArray([Float](repeating: 5000.0, count: cellCount))
        let B = MLXArray([Float](repeating: 5.3, count: cellCount))

        // Hydrogen (m = 1)
        let m_H = PlasmaPhysics.ionMass(massNumber: 1.0)
        let rho_s_H = PlasmaPhysics.ionSoundLarmorRadius(
            Te_eV: Te,
            magneticField: B,
            ionMass: m_H
        )

        // Deuterium (m = 2)
        let m_D = PlasmaPhysics.ionMass(massNumber: 2.0)
        let rho_s_D = PlasmaPhysics.ionSoundLarmorRadius(
            Te_eV: Te,
            magneticField: B,
            ionMass: m_D
        )

        let rho_H_array = rho_s_H.asArray(Float.self)
        let rho_D_array = rho_s_D.asArray(Float.self)

        // ρ_s ∝ √m_i, so ρ_s(D) / ρ_s(H) = √2
        let ratio = rho_D_array[0] / rho_H_array[0]
        let sqrt2: Float = 1.414  // √2 ≈ 1.414
        #expect(abs(ratio - sqrt2) < 0.01)

        // Typical value: 1-5 mm
        #expect(rho_H_array[0] > 0.001)
        #expect(rho_H_array[0] < 0.01)
    }
}

// MARK: - RI Model Tests

@Suite("Resistive-Interchange Model Tests")
struct RIModelTestSuite {

    @Test("RI transport coefficients computation")
    func testRICoefficients() throws {
        let cellCount = 100
        let radii = MLXArray.linspace(Float(0.1), Float(1.0), count: cellCount)

        let Te = MLXArray.full([cellCount], values: MLXArray(5000.0))
        let Ti = MLXArray.full([cellCount], values: MLXArray(5000.0))
        let ne = 1e20 * exp(-radii / 0.3)
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let geometry = createTestGeometry(cellCount: cellCount)
        let parameters = try TransportParameters(modelType: .constant, parameters: [:])

        let riModel = ResistiveInterchangeModel(
            riCoefficient: 0.5,
            ionMassNumber: 2.0
        )

        let coeffs = riModel.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        let chiArray = coeffs.ionHeatDiffusivity.value.asArray(Float.self)

        // Should be in reasonable range [1e-9, 100] m²/s
        // CRITICAL: RI can be very small at moderate β due to exp(-β_crit/β) suppression
        for i in 0..<cellCount {
            #expect(chiArray[i] >= 1e-9)
            #expect(chiArray[i] <= 100.0)
        }
    }

    @Test("CRITICAL: Isotope scaling in RI model (χ_D > χ_H)")
    func testIsotopeScalingInRIModel() throws {
        let cellCount = 50

        // CRITICAL: Must create PEAKED profiles for gradient-driven RI transport!
        let geometry = createTestGeometry(cellCount: cellCount)
        let rhoNorm = geometry.radii.value / geometry.minorRadius

        // Peaked parabolic profiles at MODERATE temperature (2-3 keV)
        // PHYSICS: RI turbulence is collisional - needs moderate T for sufficient η
        // At 10 keV: η too low → τ_R huge → χ_RI negligible
        // At 2-3 keV: η adequate → τ_R reasonable → χ_RI observable
        // CRITICAL: Higher density for sufficient β (β ∝ n × T / B²)
        let Te = MLXArray(Float(2500.0)) * (MLXArray(Float(1.0)) - MLXArray(Float(0.5)) * rhoNorm * rhoNorm)
        let Ti = MLXArray(Float(2500.0)) * (MLXArray(Float(1.0)) - MLXArray(Float(0.5)) * rhoNorm * rhoNorm)
        let ne_high = MLXArray(Float(1.0e20)) * (MLXArray(Float(1.0)) - MLXArray(Float(0.3)) * rhoNorm * rhoNorm)
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne_high),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let parameters = try TransportParameters(modelType: .constant, parameters: [:])

        // Hydrogen plasma
        // CRITICAL: Very high C_RI to overcome beta suppression
        // β = 0.006 with β_crit = 0.02 → exp(-3.3) ≈ 0.037 (strong suppression)
        // Need C_RI ~ 1000 to get observable χ_RI
        let riModel_H = ResistiveInterchangeModel(
            riCoefficient: 1000.0,
            ionMassNumber: 1.0
        )
        let coeffs_H = riModel_H.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        // Deuterium plasma
        let riModel_D = ResistiveInterchangeModel(
            riCoefficient: 1000.0,
            ionMassNumber: 2.0
        )
        let coeffs_D = riModel_D.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        let chi_H_array = coeffs_H.ionHeatDiffusivity.value.asArray(Float.self)
        let chi_D_array = coeffs_D.ionHeatDiffusivity.value.asArray(Float.self)

        // CRITICAL: χ ∝ ρ_s² ∝ m_i, so χ_D / χ_H ≈ 2
        let ratio = chi_D_array[cellCount / 2] / chi_H_array[cellCount / 2]

        print("RI Isotope Scaling Test:")
        print("  χ_H = \(chi_H_array[cellCount / 2]) m²/s")
        print("  χ_D = \(chi_D_array[cellCount / 2]) m²/s")
        print("  χ_D / χ_H = \(ratio) (expected ≈ 2.0)")

        #expect(ratio > 1.5)
        #expect(ratio < 2.5)
    }
}

// MARK: - Density Transition Model Tests

@Suite("Density Transition Model Tests")
struct TurbulenceTransitionTestSuite {

    @Test("Density transition blending between ITG and RI")
    func testDensityTransitionBlending() throws {
        let cellCount = 50

        let Te = MLXArray.full([cellCount], values: MLXArray(5000.0))
        let Ti = MLXArray.full([cellCount], values: MLXArray(5000.0))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let geometry = createTestGeometry(cellCount: cellCount)
        let parameters = try TransportParameters(modelType: .densityTransition, parameters: [:])
        let model = DensityTransitionModel.createDefault(ionMassNumber: 2.0)

        // Low density (pure ITG regime)
        let ne_low = MLXArray.full([cellCount], values: MLXArray(1.0e19))
        let profiles_low = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne_low),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )
        let coeffs_low = model.computeCoefficients(
            profiles: profiles_low,
            geometry: geometry,
            parameters: parameters
        )

        // High density (pure RI regime)
        let ne_high = MLXArray.full([cellCount], values: MLXArray(4.0e19))
        let profiles_high = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne_high),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )
        let coeffs_high = model.computeCoefficients(
            profiles: profiles_high,
            geometry: geometry,
            parameters: parameters
        )

        let chi_low = coeffs_low.ionHeatDiffusivity.value.asArray(Float.self)
        let chi_high = coeffs_high.ionHeatDiffusivity.value.asArray(Float.self)

        // Coefficients should differ between regimes
        #expect(chi_low[cellCount / 2] != chi_high[cellCount / 2])

        print("Density Transition Test:")
        print("  χ(n=1e19) = \(chi_low[cellCount / 2]) m²/s (ITG regime)")
        print("  χ(n=4e19) = \(chi_high[cellCount / 2]) m²/s (RI regime)")
    }

    @Test("DEBUG: Pure GyroBohm isotope scaling")
    func testPureGyroBohmIsotopeScaling() throws {
        let cellCount = 50
        let geometry = createTestGeometry(cellCount: cellCount)
        let rhoNorm = geometry.radii.value / geometry.minorRadius

        let Te = MLXArray(Float(2500.0)) * (MLXArray(Float(1.0)) - MLXArray(Float(0.5)) * rhoNorm * rhoNorm)
        let Ti = Te
        let ne = MLXArray(Float(2.0e19)) * (MLXArray(Float(1.0)) - MLXArray(Float(0.3)) * rhoNorm * rhoNorm)
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )
        let parameters = try TransportParameters(modelType: .constant, parameters: [:])

        // Pure GyroBohm (no Bohm term)
        let model_H = BohmGyroBohmTransportModel(bohmCoefficient: 0.0, gyroBohmCoefficient: 10000.0, ionMassNumber: 1.0)
        let model_D = BohmGyroBohmTransportModel(bohmCoefficient: 0.0, gyroBohmCoefficient: 10000.0, ionMassNumber: 2.0)

        let coeffs_H = model_H.computeCoefficients(profiles: profiles, geometry: geometry, parameters: parameters)
        let coeffs_D = model_D.computeCoefficients(profiles: profiles, geometry: geometry, parameters: parameters)

        let chi_H = coeffs_H.ionHeatDiffusivity.value.asArray(Float.self)[cellCount / 2]
        let chi_D = coeffs_D.ionHeatDiffusivity.value.asArray(Float.self)[cellCount / 2]
        let ratio = chi_D / chi_H

        print("Pure GyroBohm Isotope Test:")
        print("  χ_H = \(chi_H) m²/s")
        print("  χ_D = \(chi_D) m²/s")
        print("  χ_D / χ_H = \(ratio)")
        print("  Expected: ≈ 2.0 (ρ_s² ∝ m_i)")

        #expect(ratio > 1.5)
        #expect(ratio < 2.5)
    }

    @Test("CRITICAL: Overall isotope effect at high density (χ_D > χ_H)")
    func testIsotopeEffectInTransitionModel() throws {
        let cellCount = 50

        // High-density regime (above transition)
        // CRITICAL: Use peaked profiles at MODERATE temperature for RI regime
        let geometry = createTestGeometry(cellCount: cellCount)
        let rhoNorm = geometry.radii.value / geometry.minorRadius

        // Parabolic profiles at 2-3 keV (collisional regime for RI turbulence)
        // CRITICAL: High density for sufficient β
        let Te = MLXArray(Float(2500.0)) * (MLXArray(Float(1.0)) - MLXArray(Float(0.5)) * rhoNorm * rhoNorm)
        let Ti = MLXArray(Float(2500.0)) * (MLXArray(Float(1.0)) - MLXArray(Float(0.5)) * rhoNorm * rhoNorm)
        let ne = MLXArray(Float(1.0e20)) * (MLXArray(Float(1.0)) - MLXArray(Float(0.3)) * rhoNorm * rhoNorm)
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )
        let parameters = try TransportParameters(modelType: .densityTransition, parameters: [:])

        // Hydrogen plasma
        // CRITICAL: Use pure GyroBohm for ITG to show isotope effect
        // Bohm term has NO mass dependence → must use only GyroBohm
        // GyroBohm is naturally small (~10⁻⁶ m²/s) → need large coefficient
        let itgModel_H = BohmGyroBohmTransportModel(
            bohmCoefficient: 0.0,
            gyroBohmCoefficient: 10000.0,  // Amplify to observable range
            ionMassNumber: 1.0
        )
        let riModel_H = ResistiveInterchangeModel(
            riCoefficient: 1000.0,  // Same as RI test
            ionMassNumber: 1.0
        )
        let model_H = DensityTransitionModel(
            itgModel: itgModel_H,
            riModel: riModel_H,
            transitionDensity: 2.5e19,
            transitionWidth: 0.5e19,
            ionMassNumber: 1.0
        )
        let coeffs_H = model_H.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        // Deuterium plasma
        let itgModel_D = BohmGyroBohmTransportModel(
            bohmCoefficient: 0.0,
            gyroBohmCoefficient: 10000.0,  // Same amplification
            ionMassNumber: 2.0
        )
        let riModel_D = ResistiveInterchangeModel(
            riCoefficient: 1000.0,  // Same as RI test
            ionMassNumber: 2.0
        )
        let model_D = DensityTransitionModel(
            itgModel: itgModel_D,
            riModel: riModel_D,
            transitionDensity: 2.5e19,
            transitionWidth: 0.5e19,
            ionMassNumber: 2.0
        )
        let coeffs_D = model_D.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        let chi_H_array = coeffs_H.ionHeatDiffusivity.value.asArray(Float.self)
        let chi_D_array = coeffs_D.ionHeatDiffusivity.value.asArray(Float.self)

        // At high density (RI regime), χ ∝ ρ_s² ∝ m_i
        let ratio = chi_D_array[cellCount / 2] / chi_H_array[cellCount / 2]

        print("\nTransition Model Isotope Test (High Density):")
        print("  n_e = 3.5e19 m⁻³ (above n_trans = 2.5e19)")
        print("  χ_H = \(chi_H_array[cellCount / 2]) m²/s")
        print("  χ_D = \(chi_D_array[cellCount / 2]) m²/s")
        print("  χ_D / χ_H = \(ratio)")
        print("  Expected: χ_D > χ_H (ρ_s² scaling dominates)")

        // CRITICAL: After fix, χ_D should be LARGER than χ_H
        #expect(ratio > 1.5)
        #expect(ratio < 2.5)
    }
}

// MARK: - Numerical Stability Tests

@Suite("Numerical Stability Tests")
struct NumericalStabilityTestSuite {

    @Test("Float32 stability with extreme temperatures")
    func testFloat32Stability() throws {
        let cellCount = 100

        // Extreme temperature (high)
        let Te_extreme = MLXArray.full([cellCount], values: MLXArray(50000.0))
        let Ti_extreme = MLXArray.full([cellCount], values: MLXArray(50000.0))
        let ne = MLXArray.full([cellCount], values: MLXArray(1e20))
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        let profiles = CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti_extreme),
            electronTemperature: EvaluatedArray(evaluating: Te_extreme),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )

        let geometry = createTestGeometry(cellCount: cellCount)
        let parameters = try TransportParameters(modelType: .densityTransition, parameters: [:])
        let model = DensityTransitionModel.createDefault()

        // Should not crash with extreme values
        let coeffs = model.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        let chiArray = coeffs.ionHeatDiffusivity.value.asArray(Float.self)

        // Check no NaN or Inf
        for i in 0..<cellCount {
            #expect(!chiArray[i].isNaN)
            #expect(!chiArray[i].isInfinite)
            #expect(chiArray[i] >= 1e-6)
            #expect(chiArray[i] <= 100.0)
        }
    }
}
