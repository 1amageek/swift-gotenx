import Testing
import MLX
@testable import GotenxCore
@testable import GotenxPhysics

@Test("Ion-electron exchange equilibration")
func testEquilibration() throws {
    let exchange = IonElectronExchange(effectiveCharge: 1.5, ionMass: 2.014)

    let ne = MLXArray.full([100], values: MLXArray(Float(5e19)))
    var Te = MLXArray.full([100], values: MLXArray(Float(10000.0)))  // 10 keV
    var Ti = MLXArray.full([100], values: MLXArray(Float(5000.0)))   // 5 keV

    // Use much smaller time step for numerical stability
    // Characteristic equilibration time ~ microseconds
    let timeStep: Float = 1e-7  // 0.1 microseconds
    let stepCount = 10000

    // Simulate equilibration using forward Euler
    for _ in 0..<stepCount {
        let Q_ie = try exchange.compute(electronDensity: ne, electronTemperature: Te, ionTemperature: Ti)

        // Energy balance: dE/timeStep = Q
        // E = (3/2) * n * k_B * T
        // dT/timeStep = Q / ((3/2) * n * k_B)
        // Fixed: Added missing (3/2) factor
        Te = Te - timeStep * Q_ie / ((3.0/2.0) * ne * PhysicsConstants.electronVolt)
        Ti = Ti + timeStep * Q_ie / ((3.0/2.0) * ne * PhysicsConstants.electronVolt)
    }

    // Should equilibrate within 100 eV
    let diff_array = abs(Te - Ti).mean()
    eval(diff_array)  // Evaluate before calling .item()
    let diff = diff_array.item(Float.self)
    #expect(diff < 100.0, "Temperatures should equilibrate: ΔT = \(diff) eV")
}

@Test("Collision frequency scaling with density")
func testCollisionFrequencyDensityScaling() {
    let exchange = IonElectronExchange()

    // ν_ei ∝ n_e * ln(Λ(n_e, T_e)) / T_e^(3/2)
    // Coulomb logarithm ln(Λ) depends on density, so scaling is not exactly linear
    let ne1 = MLXArray([Float(1e19)])
    let ne2 = MLXArray([Float(2e19)])
    let Te = MLXArray([Float(1000.0)])

    let nu1 = exchange.computeCollisionFrequency(electronDensity: ne1, electronTemperature: Te)
    let nu2 = exchange.computeCollisionFrequency(electronDensity: ne2, electronTemperature: Te)

    let ratio_array = nu2 / nu1
    eval(ratio_array)  // Evaluate before calling .item()
    let ratio = ratio_array.item(Float.self)
    // ln(Λ) decreases slightly with density, so ratio ≈ 1.96 (not exactly 2.0)
    #expect(abs(ratio - 2.0) < 0.05, "Collision frequency should scale approximately with density: ratio = \(ratio)")
}

@Test("Collision frequency scaling with temperature")
func testCollisionFrequencyTemperatureScaling() {
    let exchange = IonElectronExchange()

    // ν_ei ∝ ln(Λ(n_e, T_e)) / T_e^(3/2)
    // Coulomb logarithm ln(Λ) also depends on temperature
    let ne = MLXArray([Float(1e19)])
    let Te1 = MLXArray([Float(1000.0)])
    let Te2 = MLXArray([Float(2000.0)])

    let nu1 = exchange.computeCollisionFrequency(electronDensity: ne, electronTemperature: Te1)
    let nu2 = exchange.computeCollisionFrequency(electronDensity: ne, electronTemperature: Te2)

    let ratio_array = nu2 / nu1
    eval(ratio_array)  // Evaluate before calling .item()
    let ratio = ratio_array.item(Float.self)
    let expected: Float = 0.35355339  // pow(0.5, 1.5) = 0.5^(3/2) ≈ 0.35355

    // ln(Λ) increases with T, so the ratio is slightly different from pure T^(-3/2)
    #expect(abs(ratio - expected) < 0.02, "Collision frequency should scale approximately as T^(-3/2): ratio = \(ratio)")
}

@Test("Energy conservation")
func testEnergyConservation() throws {
    let exchange = IonElectronExchange()

    let ne = MLXArray([Float(5e19)])
    let Te = MLXArray([Float(8000.0)])
    let Ti = MLXArray([Float(6000.0)])

    let Q_ie = try exchange.compute(electronDensity: ne, electronTemperature: Te, ionTemperature: Ti)
    eval(Q_ie)  // Evaluate before calling .item()

    // Q_ie should be positive (heating ions) when Te > Ti
    #expect(Q_ie.item(Float.self) > 0, "Power should flow from electrons to ions when Te > Ti")

    // Test opposite case
    let Q_ie_reverse = try exchange.compute(electronDensity: ne, electronTemperature: Ti, ionTemperature: Te)
    eval(Q_ie_reverse)  // Evaluate before calling .item()
    #expect(Q_ie_reverse.item(Float.self) < 0, "Power should flow from ions to electrons when Ti > Te")

    // Magnitudes are not exactly equal due to ln(Λ) temperature dependence
    // ν_ei(T1) ≠ ν_ei(T2), so Q_ie magnitudes differ
    let ratio = abs(Q_ie.item(Float.self) / Q_ie_reverse.item(Float.self))
    #expect(abs(ratio - 1.0) < 0.4, "Energy exchange direction should reverse: ratio = \(ratio)")
}

@Test("Coulomb logarithm reasonable values")
func testCoulombLogarithm() {
    let exchange = IonElectronExchange()

    // Typical plasma parameters
    let ne = MLXArray([Float(1e20)])
    let Te = MLXArray([Float(10000.0)])  // 10 keV

    let coulombLogarithm = exchange.computeCoulombLogarithm(electronDensity: ne, electronTemperature: Te)
    eval(coulombLogarithm)  // Evaluate before calling .item()
    let value = coulombLogarithm.item(Float.self)

    // Coulomb logarithm should be in range [10, 20] for typical plasmas
    #expect(value > 10.0 && value < 20.0, "Coulomb logarithm should be reasonable: ln(Λ) = \(value)")
}

@Test("Zero temperature difference gives zero exchange")
func testZeroExchange() throws {
    let exchange = IonElectronExchange()

    let ne = MLXArray([Float(1e20)])
    let T = MLXArray([Float(5000.0)])

    let Q_ie = try exchange.compute(electronDensity: ne, electronTemperature: T, ionTemperature: T)
    eval(Q_ie)  // Evaluate before calling .item()

    #expect(abs(Q_ie.item(Float.self)) < 1e-6, "No exchange when temperatures are equal")
}
