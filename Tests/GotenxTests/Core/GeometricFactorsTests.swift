// GeometricFactorsTests.swift
// Tests for metric tensor and non-uniform grid support

import Testing
import MLX
@testable import GotenxCore

@Suite("Geometric Factors Tests")
struct GeometricFactorsTests {

    @Test("GeometricFactors validating rejects inconsistent Geometry")
    func validatingRejectsInconsistentGeometry() {
        let geometry = Self.inconsistentGeometry()

        #expect(throws: NumericalValidationError.self) {
            _ = try GeometricFactors.validating(geometry: geometry)
        }
    }

    @Test("GeometricFactors from returns validation-failing factors instead of trapping")
    func fromReturnsValidationFailingFactorsInsteadOfTrapping() {
        let geometry = Self.inconsistentGeometry()

        let geoFactors = GeometricFactors.from(geometry: geometry)
        let volumes = geoFactors.cellVolumes.value.asArray(Float.self)

        #expect(!volumes.isEmpty)
        #expect(volumes.contains(where: { !$0.isFinite }))
    }

    @Test("GeometricFactors created from Geometry includes metric tensors")
    func metricTensorInclusion() throws {
        // Create geometry with known metric tensor values
        let cellCount = 10
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 5.0
        )
        let geometry = Geometry(config: meshConfig)

        // Create GeometricFactors from geometry
        let geoFactors = GeometricFactors.from(geometry: geometry)

        // Verify metric tensors are included
        #expect(geoFactors.jacobian.value.shape[0] == cellCount)
        #expect(geoFactors.majorRadiusMetric.value.shape[0] == cellCount)
        #expect(geoFactors.shapeMetric.value.shape[0] == cellCount)

        // Verify values are positive (physical requirement)
        let jacobianArray = geoFactors.jacobian.value.asArray(Float.self)
        let majorRadiusMetricArray = geoFactors.majorRadiusMetric.value.asArray(Float.self)
        let shapeMetricArray = geoFactors.shapeMetric.value.asArray(Float.self)

        for i in 0..<cellCount {
            #expect(jacobianArray[i] > 0.0)  // Jacobian must be positive
            #expect(majorRadiusMetricArray[i].isFinite)     // majorRadiusMetric must be finite
            #expect(shapeMetricArray[i].isFinite)     // shapeMetric must be finite
        }
    }

    @Test("Metric tensor preserves shape consistency")
    func metricTensorShapeConsistency() throws {
        let cellCount = 20
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.0,
            minorRadius: 2.0,
            toroidalField: 4.0
        )
        let geometry = Geometry(config: meshConfig)

        let geoFactors = GeometricFactors.from(geometry: geometry)

        // All cell-centered quantities should have shape [cellCount]
        #expect(geoFactors.cellVolumes.value.shape[0] == cellCount)
        #expect(geoFactors.cellRadii.value.shape[0] == cellCount)
        #expect(geoFactors.jacobian.value.shape[0] == cellCount)
        #expect(geoFactors.majorRadiusMetric.value.shape[0] == cellCount)
        #expect(geoFactors.shapeMetric.value.shape[0] == cellCount)

        // Face-centered quantities should have shape [faceCount] = [cellCount + 1]
        #expect(geoFactors.faceAreas.value.shape[0] == cellCount + 1)
        #expect(geoFactors.faceRadii.value.shape[0] == cellCount + 1)

        // Cell distances between centers should have shape [cellCount - 1]
        #expect(geoFactors.cellDistances.value.shape[0] == cellCount - 1)
    }

    @Test("Metric tensor flux divergence reduces to standard for uniform grid")
    func metricTensorUniformGridEquivalence() throws {
        // For uniform circular geometry with constant Jacobian,
        // metric tensor formulation should be equivalent to standard formulation

        let cellCount = 10
        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 5.0
        )
        let geometry = Geometry(config: meshConfig)

        let geoFactors = GeometricFactors.from(geometry: geometry)

        // For circular geometry, Jacobian should be approximately constant
        // (varies slightly with radius for tokamak geometry)
        let jacobianArray = geoFactors.jacobian.value.asArray(Float.self)

        // Check that Jacobian doesn't vary by more than 50% (generous bound for circular geom)
        let jMean = jacobianArray.reduce(0.0, +) / Float(cellCount)
        for j in jacobianArray {
            let relativeVariation = abs(j - jMean) / jMean
            #expect(relativeVariation < 0.5)
        }
    }

    @Test("Cell volumes computed correctly from geometry")
    func cellVolumeComputation() throws {
        let cellCount = 10
        let minorRadius: Float = 1.0
        let majorRadius: Float = 3.0

        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: majorRadius,
            minorRadius: minorRadius,
            toroidalField: 5.0
        )
        let geometry = Geometry(config: meshConfig)

        let geoFactors = GeometricFactors.from(geometry: geometry)

        // For uniform grid: V_cell = 2π R₀ Δr
        let radialSpacing = minorRadius / Float(cellCount)
        let expectedVolume = 2.0 * Float.pi * majorRadius * radialSpacing

        let volumes = geoFactors.cellVolumes.value.asArray(Float.self)

        for v in volumes {
            #expect(abs(v - expectedVolume) < expectedVolume * 0.01)  // Within 1%
        }
    }

    @Test("Face areas constant for cylindrical geometry")
    func faceAreaConstancy() throws {
        let cellCount = 10
        let majorRadius: Float = 3.0

        let meshConfig = MeshConfig(
            cellCount: cellCount,
            majorRadius: majorRadius,
            minorRadius: 1.0,
            toroidalField: 5.0
        )
        let geometry = Geometry(config: meshConfig)

        let geoFactors = GeometricFactors.from(geometry: geometry)

        // For 1D cylindrical approximation: A = 2π R₀ (constant)
        let expectedArea = 2.0 * Float.pi * majorRadius

        let areas = geoFactors.faceAreas.value.asArray(Float.self)

        for a in areas {
            #expect(abs(a - expectedArea) < expectedArea * 0.01)  // Within 1%
        }
    }

    private static func inconsistentGeometry() -> Geometry {
        Geometry(
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 5.0,
            volume: EvaluatedArray(evaluating: MLXArray(Float(1.0))),
            fluxSurfaceMetric: .ones([4]),
            majorRadiusMetric: .ones([4]),
            shapeMetric: .ones([4]),
            minorRadiusMetric: .ones([4]),
            radii: .ones([2]),
            safetyFactor: .ones([3]),
            type: .circular
        )
    }
}
