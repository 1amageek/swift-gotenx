import Testing
import MLX
@testable import GotenxCore
@testable import GotenxPhysics

@Test("Fusion reactivity peak at ~70 keV")
func testFusionPeak() throws {
    let fusion = try FusionPower()

    let ne = MLXArray([Float(1e20)])

    // Test range from 1 keV to 100 keV
    let temperatures = stride(from: 1000.0, through: 100000.0, by: 1000.0).map { Float($0) }

    var powers: [Float] = []
    for Ti in temperatures {
        let P = try fusion.compute(electronDensity: ne, ionTemperature: MLXArray([Ti]))
        eval(P)  // Evaluate before calling .item()
        powers.append(P.item(Float.self))
    }

    // Find peak
    let peakIdx = powers.enumerated().max(by: { $0.1 < $1.1 })!.0
    let peakTi_keV = temperatures[peakIdx] / 1000.0

    // Peak should be around 70 keV
    #expect(abs(peakTi_keV - 70.0) < 15.0, "Fusion reactivity peak should be near 70 keV: got \(peakTi_keV) keV")
}

@Test("Fusion power density scaling")
func testFusionScaling() throws {
    let fusion = try FusionPower()

    // fusionPower ∝ n²
    let ne1 = MLXArray([Float(1e20)])
    let ne2 = MLXArray([Float(2e20)])
    let Ti = MLXArray([Float(70000.0)])  // At peak reactivity

    let P1 = try fusion.compute(electronDensity: ne1, ionTemperature: Ti)
    let P2 = try fusion.compute(electronDensity: ne2, ionTemperature: Ti)

    let ratio_array = P2 / P1
    eval(ratio_array)  // Evaluate before calling .item()
    let ratio = ratio_array.item(Float.self)

    // Should scale as n²
    #expect(abs(ratio - 4.0) < 0.5, "Fusion power should scale as n²: got ratio \(ratio)")
}

@Test("Zero power at low temperature")
func testLowTemperature() throws {
    let fusion = try FusionPower()

    let ne = MLXArray([Float(1e20)])
    let Ti = MLXArray([Float(100.0)])  // 100 eV - too cold for fusion

    let P = try fusion.compute(electronDensity: ne, ionTemperature: Ti)
    eval(P)  // Evaluate before calling .item()

    #expect(P.item(Float.self) < 1e6, "Fusion power should be negligible at 100 eV")
}

@Test("Reactivity increases with temperature (below peak)")
func testReactivityMonotonic() throws {
    let fusion = try FusionPower()

    let Ti1_keV = MLXArray([Float(10.0)])
    let Ti2_keV = MLXArray([Float(20.0)])

    let sigma_v1 = fusion.computeReactivity(ionTemperatureKeV: Ti1_keV)
    eval(sigma_v1)  // Evaluate before calling .item()

    let sigma_v2 = fusion.computeReactivity(ionTemperatureKeV: Ti2_keV)
    eval(sigma_v2)  // Evaluate before calling .item()

    #expect(sigma_v2.item(Float.self) > sigma_v1.item(Float.self),
            "Reactivity should increase with temperature below peak")
}

@Test("Fuel mixture densities")
func testFuelMixture() throws {
    // Equal D-T mixture
    let fusionEqual = try FusionPower(fuelMixture: .equalDT)

    let ne = MLXArray([Float(2e20)])
    let Ti = MLXArray([Float(70000.0)])

    let P_equal = try fusionEqual.compute(electronDensity: ne, ionTemperature: Ti)
    eval(P_equal)  // Evaluate before calling .item()

    // Custom mixture (25-75)
    let fusionCustom = try FusionPower(fuelMixture: .custom(deuteriumFraction: 0.25, tritiumFraction: 0.75))
    let P_custom = try fusionCustom.compute(electronDensity: ne, ionTemperature: Ti)
    eval(P_custom)  // Evaluate before calling .item()

    // Equal mixture should give more power (optimal is 50-50)
    #expect(P_equal.item(Float.self) > P_custom.item(Float.self),
            "Equal D-T mixture should be more efficient than 25-75")
}

@Test("Triple product computation")
func testTripleProduct() throws {
    let fusion = try FusionPower()

    let ne = MLXArray([Float(1e20)])
    let Ti = MLXArray([Float(10000.0)])  // 10 keV
    let energyConfinementTime: Float = 1.0  // 1 second

    let nTtau = fusion.computeTripleProduct(electronDensity: ne, ionTemperature: Ti, energyConfinementTime: energyConfinementTime)
    eval(nTtau)  // Evaluate before calling .item()

    let expected: Float = 1e20 * 10000.0 * 1.0  // 1e24
    let value = nTtau.item(Float.self)

    #expect(abs(value - expected) / expected < 0.01, "Triple product should be computed correctly")
}

@Test("Fusion power density magnitude")
func testFusionPowerMagnitude() throws {
    let fusion = try FusionPower()

    let ne = MLXArray([Float(1e20)])  // m⁻³
    let Ti = MLXArray([Float(70000.0)])  // eV

    let P = try fusion.compute(electronDensity: ne, ionTemperature: Ti)
    eval(P)  // Evaluate before calling .item()

    // Power density should be in W/m³
    // Typical values at n=1e20, T=70keV: ~1×10⁶ W/m³ (= 1 MW/m³)
    let value = P.item(Float.self)

    #expect(value > 1e5 && value < 1e7,
            "Fusion power density should be reasonable: \(value) W/m³")
}
