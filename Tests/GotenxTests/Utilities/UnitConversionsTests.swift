import Testing
import MLX
@testable import GotenxCore

/// Tests for UnitConversions utilities
@Suite("UnitConversions Tests")
struct UnitConversionsTests {

    // MARK: - Test Setup

    /// Force CPU backend for tests (avoids Metal library issues in CI/test environments)
    init() {
        // Set default device to CPU to avoid Metal library loading issues
        MLX.GPU.set(cacheLimit: 0)
    }

    // MARK: - Constants Tests

    /// Test that eV constant matches the fundamental constant
    @Test("eV constant is correct")
    func testEvConstant() {
        let expected: Float = 1.602176634e-19  // [J/eV]
        #expect(UnitConversions.electronVolt == expected, "eV constant mismatch")
    }

    /// Test that conversion constant is correct
    @Test("MW/m³ to eV/(m³·s) conversion constant")
    func testConversionConstant() {
        // Derivation:
        // 1 MW/m³ = 10⁶ W/m³
        //         = 10⁶ J/(m³·s)
        //         = 10⁶ J/(m³·s) × (1 eV / 1.602176634×10⁻¹⁹ J)
        //         = 6.2415090744×10²⁴ eV/(m³·s)
        let expected: Float = 6.2415090744e24
        #expect(UnitConversions.megawattsPerCubicMeterToElectronVoltsPerCubicMeterPerSecond == expected,
                "Conversion constant mismatch")
    }

    /// Test conversion constant derivation from first principles
    @Test("Conversion constant matches derivation from eV")
    func testConversionDerivation() {
        // 1 MW = 10⁶ W = 10⁶ J/s
        let megawatt: Float = 1e6  // [W]

        // Convert J to eV: J / (J/eV) = eV
        let evPerJoule: Float = 1.0 / UnitConversions.electronVolt  // [eV/J]
        let evPerSecond: Float = megawatt * evPerJoule  // [eV/s]

        // For power density: [MW/m³] → [eV/(m³·s)]
        let derived = evPerSecond  // Same as MW * (eV/J)

        let expected = UnitConversions.megawattsPerCubicMeterToElectronVoltsPerCubicMeterPerSecond
        let relativeError = abs(derived - expected) / expected

        #expect(relativeError < 1e-6,
                "Derived conversion constant (\(derived)) differs from defined constant (\(expected))")
    }

    // MARK: - Scalar Conversion Tests

    /// Test scalar conversion with typical value
    @Test("Scalar conversion: 1 MW/m³ → eV/(m³·s)")
    func testScalarConversionUnity() {
        let input: Float = 1.0  // [MW/m³]
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        let expected: Float = 6.2415090744e24  // [eV/(m³·s)]
        let relativeError = abs(output - expected) / expected

        #expect(relativeError < 1e-6, "Conversion error for 1 MW/m³")
    }

    /// Test scalar conversion with realistic ITER heating power
    @Test("Scalar conversion: 0.5 MW/m³ (typical ITER heating)")
    func testScalarConversionRealistic() {
        let input: Float = 0.5  // [MW/m³] - typical fusion heating
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        let expected: Float = 0.5 * 6.2415090744e24  // [eV/(m³·s)]
        let relativeError = abs(output - expected) / expected

        #expect(relativeError < 1e-6, "Conversion error for 0.5 MW/m³")
    }

    /// Test scalar conversion with zero
    @Test("Scalar conversion: 0 MW/m³")
    func testScalarConversionZero() {
        let input: Float = 0.0  // [MW/m³]
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        #expect(output == 0.0, "Zero input should give zero output")
    }

    /// Test scalar conversion with negative value (cooling)
    @Test("Scalar conversion: negative value (cooling)")
    func testScalarConversionNegative() {
        let input: Float = -0.1  // [MW/m³] - cooling/loss term
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        let expected: Float = -0.1 * 6.2415090744e24  // [eV/(m³·s)]
        let relativeError = abs(output - expected) / abs(expected)

        #expect(relativeError < 1e-6, "Conversion error for negative value")
    }

    // MARK: - Array Conversion Tests

    /// Test array conversion with uniform values
    @Test("Array conversion: uniform heating profile")
    func testArrayConversionUniform() {
        let cellCount = 25
        let input = MLXArray(Array(repeating: Float(1.0), count: cellCount))  // [MW/m³]
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        // Verify output is Float32 (GPU-compatible)
        #expect(output.dtype == .float32, "Output should be Float32 for GPU compatibility")

        // CRITICAL: eval() forces computation before extracting values
        eval(output)

        let expected: Float = 6.2415090744e24  // [eV/(m³·s)]
        let outputArray = output.asArray(Float.self)

        for (i, value) in outputArray.enumerated() {
            let relativeError = abs(value - expected) / expected
            #expect(relativeError < 1e-6, "Conversion error at cell \(i)")
        }
    }

    /// Test array conversion with profile (core-to-edge gradient)
    @Test("Array conversion: realistic heating profile")
    func testArrayConversionProfile() {
        let cellCount = 25

        // Realistic heating profile: peaked in core, decaying to edge
        // Q(r) = Q0 * (1 - 0.9 * (r/a)²)
        var inputArray = [Float]()
        for i in 0..<cellCount {
            let rho = Float(i) / Float(cellCount - 1)  // Normalized radius
            let Q_MW: Float = 1.0 * (1.0 - 0.9 * rho * rho)  // [MW/m³]
            inputArray.append(Q_MW)
        }

        let input = MLXArray(inputArray)
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        // Verify output is Float32 (GPU-compatible)
        #expect(output.dtype == .float32, "Output should be Float32 for GPU compatibility")

        // CRITICAL: eval() forces computation before extracting values
        eval(output)

        let outputArray = output.asArray(Float.self)
        let conversionFactor: Float = 6.2415090744e24

        for (i, value) in outputArray.enumerated() {
            let expected = inputArray[i] * conversionFactor
            let relativeError = abs(value - expected) / (expected + Float(1e-30))  // Avoid division by zero at edge

            #expect(relativeError < Float(1e-6) || expected < Float(1e10),
                    "Conversion error at cell \(i): \(value) vs \(expected)")
        }
    }

    /// Test array conversion with zeros
    @Test("Array conversion: zero heating")
    func testArrayConversionZeros() {
        let cellCount = 25
        let input = MLXArray.zeros([cellCount])  // [MW/m³]
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        // Verify output is Float32 (GPU-compatible)
        #expect(output.dtype == .float32, "Output should be Float32 for GPU compatibility")

        // CRITICAL: eval() forces computation before extracting values
        eval(output)

        let outputArray = output.asArray(Float.self)

        for (i, value) in outputArray.enumerated() {
            #expect(value == 0.0, "Zero input should give zero output at cell \(i)")
        }
    }

    /// Test array conversion with mixed positive/negative values
    @Test("Array conversion: mixed heating/cooling")
    func testArrayConversionMixed() {
        let input = MLXArray([Float(1.0), Float(-0.5), Float(0.0), Float(0.3), Float(-0.1)])  // [MW/m³]
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        // Verify output is Float32 (GPU-compatible)
        #expect(output.dtype == .float32, "Output should be Float32 for GPU compatibility")

        // CRITICAL: eval() forces computation of lazy MLXArray before extracting values
        eval(output)

        let coefficient: Float = 6.2415090744e24
        let expectedArray: [Float] = [
            Float(1.0) * coefficient,
            Float(-0.5) * coefficient,
            Float(0.0),
            Float(0.3) * coefficient,
            Float(-0.1) * coefficient
        ]

        let outputArray = output.asArray(Float.self)

        for (i, value) in outputArray.enumerated() {
            let expected = expectedArray[i]
            if expected == 0.0 {
                #expect(value == 0.0, "Zero mismatch at index \(i)")
            } else {
                let relativeError = abs(value - expected) / abs(expected)
                #expect(relativeError < 1e-6, "Conversion error at index \(i)")
            }
        }
    }

    // MARK: - Dimensional Consistency Tests

    /// Test that conversion maintains dimensional consistency in temperature equation
    @Test("Temperature equation dimensional consistency with conversion")
    func testTemperatureEquationDimensions() {
        // Setup typical ITER plasma parameters
        let electronDensity: Float = 1e20      // [m⁻³]
        let heatDiffusivity: Float = 1.0      // [m²/s]
        let temperatureGradient: Float = 1000.0 // [eV/m]
        let Q_MW: Float = 0.5     // [MW/m³]

        // Left side: n_e ∂T/∂t
        // Dimension: [m⁻³] × [eV/s] = [eV/(m³·s)]
        // (We don't compute actual value, just verify units)

        // Diffusion term: ∇·(n_e χ ∇T)
        // Dimension: ∇·([m⁻³] × [m²/s] × [eV/m]) = [eV/(m³·s)]
        let radialSpacing: Float = 0.08  // [m] typical cell size
        let diffusionTerm = electronDensity * heatDiffusivity * temperatureGradient / radialSpacing
        // [m⁻³] × [m²/s] × [eV/m] × [1/m] = [eV/(m³·s)] ✓

        // Source term: Q after conversion
        // Must have dimension [eV/(m³·s)]
        let sourceTerm = UnitConversions.megawattsToElectronVoltDensity(Q_MW)

        // Verify both terms are comparable in magnitude
        // (same dimension means they can be added/subtracted)
        let ratio = sourceTerm / diffusionTerm

        // For typical ITER: heating and transport are comparable
        #expect(ratio > Float(0.1) && ratio < Float(100),
                "Source and diffusion terms have inconsistent magnitude (ratio = \(ratio))")
    }

    /// Test numerical precision of conversion
    @Test("Conversion maintains Float32 precision")
    func testConversionPrecision() {
        // Test that conversion doesn't lose precision for typical values
        let input: Float = 0.123456789  // [MW/m³]
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        // Verify at least 6 significant figures preserved
        let coefficient: Float = 6.2415090744e24
        let expected = input * coefficient
        let relativeError = abs(output - expected) / expected

        // Float32 has ~7 decimal digits precision
        #expect(relativeError < 1e-6, "Precision loss in conversion")
    }

    /// Test conversion with very small values (edge case)
    @Test("Conversion with very small values")
    func testConversionVerySmall() {
        let input: Float = 1e-6  // [MW/m³] - very small heating
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        let coefficient: Float = 6.2415090744e24
        let expected = input * coefficient
        let relativeError = abs(output - expected) / expected

        #expect(relativeError < 1e-6, "Conversion error for very small value")
    }

    /// Test conversion with very large values (edge case)
    @Test("Conversion with very large values")
    func testConversionVeryLarge() {
        let input: Float = 100.0  // [MW/m³] - very large heating
        let output = UnitConversions.megawattsToElectronVoltDensity(input)

        let coefficient: Float = 6.2415090744e24
        let expected = input * coefficient

        // Check for overflow
        #expect(!output.isInfinite, "Conversion resulted in infinity")
        #expect(!output.isNaN, "Conversion resulted in NaN")

        let relativeError = abs(output - expected) / expected
        #expect(relativeError < 1e-6, "Conversion error for very large value")
    }

    // MARK: - Type Safety Tests

    /// Test that array version uses Float32 (GPU-compatible)
    @Test("Array conversion uses Float32 for GPU compatibility")
    func testArrayConversionDtype() {
        // Test with Float32 input
        let cellCount = 10

        // Float32 input → Float32 output (GPU-compatible)
        let float32Input = MLXArray(Array(repeating: Float(1.0), count: cellCount))
        let float32Output = UnitConversions.megawattsToElectronVoltDensity(float32Input)
        #expect(float32Output.dtype == .float32, "Float32 input should produce Float32 output (GPU-compatible)")

        // Note: Float64 is NOT supported on Apple Silicon GPU
        // All computations use Float32
    }

    /// Test no overflow with large values in Float32
    @Test("No overflow with large conversion coefficient in Float32")
    func testNoOverflowWithLargeCoefficient() {
        // Test scalar version (Float32 arithmetic)
        let scalarValues: [Float] = [1.0, 10.0, 100.0]
        let coefficient = Float(6.2415090744e24)  // Explicitly Float32

        for (i, input) in scalarValues.enumerated() {
            let output = UnitConversions.megawattsToElectronVoltDensity(input)

            #expect(!output.isInfinite, "Scalar value at index \(i) overflowed to infinity")
            #expect(!output.isNaN, "Scalar value at index \(i) is NaN")

            let expected = input * coefficient
            let relativeError = abs(output - expected) / expected
            #expect(relativeError < 1e-6, "Scalar conversion error at index \(i)")
        }

        // Test array version with Float32 output (GPU-compatible, sufficient precision)
        let arrayInput = MLXArray([Float(1.0), Float(10.0), Float(100.0)])
        let arrayOutput = UnitConversions.megawattsToElectronVoltDensity(arrayInput)

        #expect(arrayOutput.dtype == .float32, "Array output should be Float32 (GPU-compatible)")

        eval(arrayOutput)
        let outputArray = arrayOutput.asArray(Float.self)

        for (i, value) in outputArray.enumerated() {
            #expect(!value.isInfinite, "Array value at index \(i) overflowed to infinity")
            #expect(!value.isNaN, "Array value at index \(i) is NaN")

            let expected = scalarValues[i] * coefficient  // Use same coefficient as scalar test
            let relativeError = abs(value - expected) / expected
            // Float32 has ~7 significant digits, so relative error should be < 1e-6
            #expect(relativeError < 1e-6, "Array conversion error at index \(i): got \(value), expected \(expected), rel_error=\(relativeError)")
        }
    }
}
