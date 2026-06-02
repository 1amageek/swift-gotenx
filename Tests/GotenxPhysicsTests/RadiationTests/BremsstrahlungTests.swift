import Testing
import MLX
@testable import GotenxCore
@testable import GotenxPhysics

@Test("Bremsstrahlung is always negative")
func testRadiationIsLoss() throws {
    let brems = Bremsstrahlung()

    let ne = MLXArray([Float(1e20)])
    let Te = MLXArray([Float(10000.0)])

    let P = try brems.compute(electronDensity: ne, electronTemperature: Te)
    eval(P)  // Evaluate before calling .item()

    #expect(P.item(Float.self) < 0, "Bremsstrahlung should always be a loss (negative power)")
}

@Test("Bremsstrahlung scaling with density")
func testDensityScaling() throws {
    let brems = Bremsstrahlung()

    // P_brems ∝ n²
    let ne1 = MLXArray([Float(1e20)])
    let ne2 = MLXArray([Float(2e20)])
    let Te = MLXArray([Float(10000.0)])

    let P1 = try brems.compute(electronDensity: ne1, electronTemperature: Te)
    let P2 = try brems.compute(electronDensity: ne2, electronTemperature: Te)

    let ratio_array = abs(P2 / P1)
    eval(ratio_array)  // Evaluate before calling .item()
    let ratio = ratio_array.item(Float.self)

    // Should scale as n²
    #expect(abs(ratio - 4.0) < 0.1, "Bremsstrahlung should scale as n²: got ratio \(ratio)")
}

@Test("Bremsstrahlung scaling with temperature")
func testTemperatureScaling() throws {
    let brems = Bremsstrahlung(includeRelativistic: false)  // Use classical only

    // P_brems ∝ √T
    let ne = MLXArray([Float(1e20)])
    let Te1 = MLXArray([Float(10000.0)])
    let Te2 = MLXArray([Float(40000.0)])

    let P1 = try brems.compute(electronDensity: ne, electronTemperature: Te1)
    let P2 = try brems.compute(electronDensity: ne, electronTemperature: Te2)

    let ratio_array = abs(P2 / P1)
    eval(ratio_array)  // Evaluate before calling .item()
    let ratio = ratio_array.item(Float.self)
    let expected: Float = 2.0  // sqrt(40000/10000) = sqrt(4) = 2.0

    #expect(abs(ratio - expected) < 0.1, "Bremsstrahlung should scale as √T: got ratio \(ratio)")
}

@Test("Relativistic correction at high temperature")
func testRelativisticCorrection() throws {
    let brems_classical = Bremsstrahlung(includeRelativistic: false)
    let brems_relativistic = Bremsstrahlung(includeRelativistic: true)

    let ne = MLXArray([Float(1e20)])
    let Te = MLXArray([Float(100000.0)])  // 100 keV - relativistic effects matter

    let P_classical = try brems_classical.compute(electronDensity: ne, electronTemperature: Te)
    eval(P_classical)  // Evaluate before calling .item()

    let P_relativistic = try brems_relativistic.compute(electronDensity: ne, electronTemperature: Te)
    eval(P_relativistic)  // Evaluate before calling .item()

    // Relativistic correction should increase radiation
    #expect(abs(P_relativistic.item(Float.self)) > abs(P_classical.item(Float.self)),
            "Relativistic correction should increase radiation at high T")
}

@Test("Negligible relativistic correction at low temperature")
func testLowTemperatureRelativistic() throws {
    let brems_classical = Bremsstrahlung(includeRelativistic: false)
    let brems_relativistic = Bremsstrahlung(includeRelativistic: true)

    let ne = MLXArray([Float(1e20)])
    let Te = MLXArray([Float(1000.0)])  // 1 keV - relativistic effects negligible

    let P_classical = try brems_classical.compute(electronDensity: ne, electronTemperature: Te)
    eval(P_classical)  // Evaluate before calling .item()

    let P_relativistic = try brems_relativistic.compute(electronDensity: ne, electronTemperature: Te)
    eval(P_relativistic)  // Evaluate before calling .item()

    let ratio = abs(P_relativistic.item(Float.self) / P_classical.item(Float.self))

    // Should be nearly identical at low temperature
    #expect(abs(ratio - 1.0) < 0.01,
            "Relativistic correction should be negligible at 1 keV: ratio = \(ratio)")
}

@Test("Radiation increases with effective charge")
func testEffectiveChargeScaling() throws {
    let brems_low = Bremsstrahlung(effectiveCharge: 1.0)
    let brems_high = Bremsstrahlung(effectiveCharge: 2.0)

    let ne = MLXArray([Float(1e20)])
    let Te = MLXArray([Float(10000.0)])

    let P_low = try brems_low.compute(electronDensity: ne, electronTemperature: Te)
    eval(P_low)  // Evaluate before calling .item()

    let P_high = try brems_high.compute(electronDensity: ne, electronTemperature: Te)
    eval(P_high)  // Evaluate before calling .item()

    let ratio = abs(P_high.item(Float.self) / P_low.item(Float.self))

    // Should scale linearly with effectiveCharge
    #expect(abs(ratio - 2.0) < 0.01, "Bremsstrahlung should scale linearly with effective charge")
}

@Test("Relativistic correction factor")
func testRelativisticFactor() {
    let brems = Bremsstrahlung()

    // At low temperature, f_rel should be ~0
    let Te_low = MLXArray([Float(1000.0)])
    let f_rel_low = brems.computeRelativisticCorrection(electronTemperature: Te_low)
    eval(f_rel_low)  // Evaluate before calling .item()
    #expect(f_rel_low.item(Float.self) < 0.001, "Relativistic correction should be tiny at 1 keV")

    // At high temperature, f_rel should be significant
    let Te_high = MLXArray([Float(100000.0)])
    let f_rel_high = brems.computeRelativisticCorrection(electronTemperature: Te_high)
    eval(f_rel_high)  // Evaluate before calling .item()
    #expect(f_rel_high.item(Float.self) > 0.01, "Relativistic correction should be significant at 100 keV")
}

@Test("Radiation power units")
func testUnits() throws {
    let brems = Bremsstrahlung()

    let ne = MLXArray([Float(1e20)])  // m⁻³
    let Te = MLXArray([Float(10000.0)])  // eV

    let P = try brems.compute(electronDensity: ne, electronTemperature: Te)
    eval(P)  // Evaluate before calling .item()

    // Power density should be in W/m³
    // Typical values: -1e5 to -1e7 W/m³
    let value = abs(P.item(Float.self))

    #expect(value > 1e4 && value < 1e8,
            "Bremsstrahlung power density should be reasonable: \(value) W/m³")
}
