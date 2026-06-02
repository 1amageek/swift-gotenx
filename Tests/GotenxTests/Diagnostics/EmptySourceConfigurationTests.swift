// EmptySourceConfigurationTests.swift
// Tests for empty source configuration (zero active sources)

import Testing
import MLX
@testable import GotenxCore
@testable import GotenxPhysics

/// Empty Source Configuration Tests
///
    /// Verifies that the metadata pipeline handles configurations with:
    /// - Zero active sources
    /// - All sources disabled
    ///
    /// Without crashing in DEBUG builds.
@Suite("Empty Source Configuration Tests")
struct EmptySourceConfigurationTests {

    // MARK: - Test Helpers

    private func createTestGeometry() -> Geometry {
        let mesh = MeshConfig(
            cellCount: 10,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        return createGeometry(from: mesh, axisSafetyFactor: 1.0, edgeSafetyFactor: 3.5)
    }

    private func createTestProfiles(cellCount: Int) -> CoreProfiles {
        let Ti = [Float](repeating: 10000, count: cellCount)
        let Te = [Float](repeating: 10000, count: cellCount)
        let ne = [Float](repeating: 1e20, count: cellCount)
        let psi = [Float](repeating: 0.0, count: cellCount)

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: MLXArray(Ti)),
            electronTemperature: EvaluatedArray(evaluating: MLXArray(Te)),
            electronDensity: EvaluatedArray(evaluating: MLXArray(ne)),
            poloidalFlux: EvaluatedArray(evaluating: MLXArray(psi))
        )
    }

    // MARK: - Empty Source Configuration Tests

    @Test("Composite source with zero sources")
    func testCompositeWithZeroSources() throws {
        let geometry = createTestGeometry()
        let profiles = createTestProfiles(cellCount: 10)

        // Create composite with empty source dict
        let composite = CompositeSourceModel(sources: [:])

        let parameters = SourceParameters(modelType: "composite", parameters: [:])
        let terms = try composite.computeTerms(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        // Verify metadata is not nil (should be .empty)
        #expect(terms.metadata != nil, "Metadata should not be nil for zero sources")

        // Verify metadata is empty
        guard let metadata = terms.metadata else {
            Issue.record("Metadata is nil!")
            return
        }

        #expect(metadata.entries.isEmpty, "Metadata entries should be empty")

        // Verify all powers are zero
        #expect(metadata.fusionPower == 0)
        #expect(metadata.ohmicPower == 0)
        #expect(metadata.auxiliaryPower == 0)
        #expect(metadata.radiationPower == 0)
        #expect(metadata.alphaPower == 0)

        print("✅ Composite with zero sources test passed")
    }

    @Test("DerivedQuantities with empty source metadata")
    func testDerivedQuantitiesWithEmptyMetadata() {
        let geometry = createTestGeometry()
        let profiles = createTestProfiles(cellCount: 10)

        // Create source terms with empty metadata
        let cellCount = 10
        let zeros = EvaluatedArray.zeros([cellCount])
        let sources = SourceTerms(
            ionHeating: zeros,
            electronHeating: zeros,
            particleSource: zeros,
            currentSource: zeros,
            metadata: SourceMetadataCollection.empty  // Empty metadata (no sources)
        )

        // This should NOT crash in DEBUG builds
        let derived = DerivedQuantitiesComputer.compute(
            profiles: profiles,
            geometry: geometry,
            transport: nil,
            sources: sources
        )

        // Verify power values are zero
        #expect(derived.fusionPower == 0, "Fusion power should be 0 with empty metadata")
        #expect(derived.ohmicPower == 0, "Ohmic power should be 0 with empty metadata")
        #expect(derived.auxiliaryPower == 0, "Auxiliary power should be 0 with empty metadata")
        #expect(derived.alphaPower == 0, "Alpha power should be 0 with empty metadata")

        // Verify fusionGain is 0 (no heating)
        #expect(derived.fusionGain == 0, "Q_fusion should be 0 with no sources")

        print("✅ DerivedQuantities with empty metadata test passed")
    }

    @Test("Source adapter returns valid metadata")
    func testSourceAdapterReturnsValidMetadata() throws {
        let cellCount = 10
        let geometry = createTestGeometry()
        let profiles = createTestProfiles(cellCount: cellCount)

        // Test with OhmicHeatingSource (which can throw errors)
        let ohmicSource = OhmicHeatingSource()
        let parameters = SourceParameters(modelType: "ohmic", parameters: [:])

        let terms = try ohmicSource.computeTerms(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        #expect(terms.metadata != nil, "Metadata should not be nil")

        // This ensures DerivedQuantitiesComputer will not crash
        let derived = DerivedQuantitiesComputer.compute(
            profiles: profiles,
            geometry: geometry,
            sources: terms
        )

        #expect(derived.ohmicPower >= 0)
    }

    @Test("Source-free simulation configuration")
    func testSourceFreeSimulation() {
        let geometry = createTestGeometry()
        let profiles = createTestProfiles(cellCount: 10)

        // Simulate a source-free run (only transport, no sources)
        // This is a valid configuration for testing transport models

        let derived = DerivedQuantitiesComputer.compute(
            profiles: profiles,
            geometry: geometry,
            transport: nil,
            sources: nil  // No sources provided
        )

        // All power values should be zero
        #expect(derived.fusionPower == 0)
        #expect(derived.ohmicPower == 0)
        #expect(derived.auxiliaryPower == 0)
        #expect(derived.alphaPower == 0)

        // fusionGain should be 0
        #expect(derived.fusionGain == 0)

        // Thermal energy should still be > 0 (from profiles)
        #expect(derived.thermalEnergy > 0)

        print("✅ Source-free simulation test passed")
    }

    @Test("Power balance with empty metadata")
    func testPowerBalanceWithEmptyMetadata() {
        let geometry = createTestGeometry()
        let profiles = createTestProfiles(cellCount: 10)

        // Create sources with empty metadata
        let sources = SourceTerms(
            ionHeating: EvaluatedArray.zeros([10]),
            electronHeating: EvaluatedArray.zeros([10]),
            particleSource: EvaluatedArray.zeros([10]),
            currentSource: EvaluatedArray.zeros([10]),
            metadata: SourceMetadataCollection.empty
        )

        // Compute derived quantities
        let derived = DerivedQuantitiesComputer.compute(
            profiles: profiles,
            geometry: geometry,
            sources: sources
        )

        // Verify all computed powers are zero
        let totalPower = derived.fusionPower + derived.auxiliaryPower + derived.ohmicPower

        #expect(totalPower == 0, "Total power should be 0 with empty metadata")
        #expect(derived.fusionGain == 0, "Q should be 0 with no sources")

        print("✅ Power balance with empty metadata test passed")
        print("   fusionPower: \(derived.fusionPower) MW")
        print("   auxiliaryPower: \(derived.auxiliaryPower) MW")
        print("   ohmicPower: \(derived.ohmicPower) MW")
        print("   fusionGain: \(derived.fusionGain)")
    }
}
