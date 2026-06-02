import Testing
import MLX
@testable import GotenxCore

// MARK: - EvaluatedArray Tests

@Suite("EvaluatedArray Tests")
struct EvaluatedArrayTests {

    @Test("EvaluatedArray forces evaluation at construction")
    func testEvaluationAtConstruction() {
        // Create lazy computation
        let lazy = MLXArray([Float(1.0), Float(2.0), Float(3.0)]) + MLXArray([Float(4.0), Float(5.0), Float(6.0)])

        // Wrap in EvaluatedArray
        let evaluated = EvaluatedArray(evaluating: lazy)

        // Verify shape
        #expect(evaluated.shape == [3])
        #expect(evaluated.ndim == 1)
        #expect(evaluated.dtype == .float32)
    }

    @Test("EvaluatedArray batch evaluation")
    func testBatchEvaluation() {
        let arrays = [
            MLXArray([Float(1.0), Float(2.0)]),
            MLXArray([Float(3.0), Float(4.0)]),
            MLXArray([Float(5.0), Float(6.0)])
        ]

        let evaluated = EvaluatedArray.evaluatingBatch(arrays)

        #expect(evaluated.count == 3)
        #expect(evaluated[0].shape == [2])
        #expect(evaluated[1].shape == [2])
        #expect(evaluated[2].shape == [2])
    }

    @Test("EvaluatedArray convenience constructors")
    func testConvenienceConstructors() {
        // Zeros
        let zeros = EvaluatedArray.zeros([3])
        #expect(zeros.shape == [3])
        #expect(allClose(zeros.value, MLXArray.zeros([3])).item(Bool.self))

        // Ones
        let ones = EvaluatedArray.ones([3])
        #expect(ones.shape == [3])
        #expect(allClose(ones.value, MLXArray.ones([3])).item(Bool.self))

        // Full
        let full = EvaluatedArray.full([3], value: Float(5.0))
        #expect(full.shape == [3])
        #expect(allClose(full.value, MLXArray.full([3], values: MLXArray(Float(5.0)))).item(Bool.self))
    }

    @Test("EvaluatedArray equality")
    func testEquality() {
        let array1 = EvaluatedArray(evaluating: MLXArray([Float(1.0), Float(2.0), Float(3.0)]))
        let array2 = EvaluatedArray(evaluating: MLXArray([Float(1.0), Float(2.0), Float(3.0)]))
        let array3 = EvaluatedArray(evaluating: MLXArray([Float(1.0), Float(2.0), Float(4.0)]))

        #expect(array1 == array2)
        #expect(array1 != array3)
    }
}

// MARK: - CoreProfiles Tests

@Suite("CoreProfiles Tests")
struct CoreProfilesTests {

    @Test("CoreProfiles construction")
    func testConstruction() {
        let ti = EvaluatedArray.full([10], value: Float(5.0))
        let te = EvaluatedArray.full([10], value: Float(4.0))
        let ne = EvaluatedArray.full([10], value: Float(3.0))
        let psi = EvaluatedArray.full([10], value: Float(0.5))

        let profiles = CoreProfiles(
            ionTemperature: ti,
            electronTemperature: te,
            electronDensity: ne,
            poloidalFlux: psi
        )

        #expect(profiles.ionTemperature.shape == [10])
        #expect(profiles.electronTemperature.shape == [10])
        #expect(profiles.electronDensity.shape == [10])
        #expect(profiles.poloidalFlux.shape == [10])
    }

    @Test("CoreProfiles equality")
    func testEquality() {
        let profiles1 = CoreProfiles(
            ionTemperature: .full([10], value: Float(5.0)),
            electronTemperature: .full([10], value: Float(4.0)),
            electronDensity: .full([10], value: Float(3.0)),
            poloidalFlux: .full([10], value: Float(0.5))
        )

        let profiles2 = CoreProfiles(
            ionTemperature: .full([10], value: Float(5.0)),
            electronTemperature: .full([10], value: Float(4.0)),
            electronDensity: .full([10], value: Float(3.0)),
            poloidalFlux: .full([10], value: Float(0.5))
        )

        #expect(profiles1 == profiles2)
    }
}

// MARK: - Geometry Tests

@Suite("Geometry Tests")
struct GeometryTests {

    @Test("Geometry construction from mesh config")
    func testGeometryFromMesh() {
        let mesh = MeshConfig(
            cellCount: 10,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        let geometry = createGeometry(from: mesh)

        #expect(geometry.majorRadius == 3.0)
        #expect(geometry.minorRadius == 1.0)
        #expect(geometry.toroidalField == 2.5)
        #expect(geometry.volume.shape == [])  // Scalar
        #expect(geometry.fluxSurfaceMetric.shape == [11])  // cellCount + 1 faces
    }

    @Test("Geometry volume computation")
    func testVolumeComputation() {
        let mesh = MeshConfig(
            cellCount: 10,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )

        let volume = computeVolume(mesh)
        eval(volume)

        // V = 2π²R·a² = 2 * π² * 3.0 * 1.0² ≈ 59.22
        let expected: Float = 2.0 * Float.pi * Float.pi * 3.0 * 1.0 * 1.0

        #expect(abs(volume.item(Float.self) - expected) < 0.01)
    }
}

// MARK: - Transport Coefficients Tests

@Suite("TransportCoefficients Tests")
struct TransportCoefficientsTests {

    @Test("TransportCoefficients construction")
    func testConstruction() {
        let coeffs = TransportCoefficients(
            ionHeatDiffusivity: .full([10], value: Float(1.0)),
            electronHeatDiffusivity: .full([10], value: Float(1.5)),
            particleDiffusivity: .full([10], value: Float(0.5)),
            convectionVelocity: .zeros([10])
        )

        #expect(coeffs.ionHeatDiffusivity.shape == [10])
        #expect(coeffs.electronHeatDiffusivity.shape == [10])
    }
}

// MARK: - Source Terms Tests

@Suite("SourceTerms Data Structure Tests")
struct SourceTermsDataStructureTests {

    @Test("SourceTerms zero initialization")
    func testZeroInitialization() {
        let sources = SourceTerms.zero(cellCount: 10)

        #expect(sources.ionHeating.shape == [10])
        #expect(sources.electronHeating.shape == [10])
        #expect(sources.particleSource.shape == [10])
        #expect(sources.currentSource.shape == [10])

        // Verify all zeros
        #expect(allClose(sources.ionHeating.value, MLXArray.zeros([10])).item(Bool.self))
    }
}
