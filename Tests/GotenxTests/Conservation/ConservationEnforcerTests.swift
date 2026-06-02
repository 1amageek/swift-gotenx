import Testing
import MLX
import Foundation
@testable import GotenxCore

/// Tests for ConservationEnforcer orchestration
@Suite("Conservation Enforcer Tests")
struct ConservationEnforcerTests {

    // MARK: - Test Helpers

    /// Create test profiles
    private func createProfiles(cellCount: Int, electronTemperature: Float, ionTemperature: Float, electronDensity: Float) throws -> CoreProfiles {
        let electronTemperatureArray = MLXArray(Array(repeating: electronTemperature, count: cellCount))
        let ionTemperatureArray = MLXArray(Array(repeating: ionTemperature, count: cellCount))
        let electronDensityArray = MLXArray(Array(repeating: electronDensity, count: cellCount))
        let poloidalFlux = MLXArray(Array(repeating: Float(0.0), count: cellCount))

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: ionTemperatureArray),
            electronTemperature: EvaluatedArray(evaluating: electronTemperatureArray),
            electronDensity: EvaluatedArray(evaluating: electronDensityArray),
            poloidalFlux: EvaluatedArray(evaluating: poloidalFlux)
        )
    }

    /// Create geometry
    private func createGeometry(cellCount: Int) -> Geometry {
        let config = MeshConfig(
            cellCount: cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3
        )
        return Geometry(config: config)
    }

    // MARK: - Initialization Tests

    @Test("Initialize enforcer with single law")
    func testInitializeSingleLaw() throws {
        let cellCount = 25
        let profiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [ParticleConservation()],
            initialProfiles: profiles,
            geometry: geometry,
            verbose: false
        )

        #expect(enforcer.enforcementInterval == 1000, "Default interval should be 1000")
    }

    @Test("Initialize enforcer with multiple laws")
    func testInitializeMultipleLaws() throws {
        let cellCount = 25
        let profiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [
                ParticleConservation(driftTolerance: 0.005),
                EnergyConservation(driftTolerance: 0.01)
            ],
            initialProfiles: profiles,
            geometry: geometry,
            verbose: false
        )

        let summary = enforcer.lawsSummary()
        #expect(summary.contains("ParticleConservation"), "Summary should include ParticleConservation")
        #expect(summary.contains("EnergyConservation"), "Summary should include EnergyConservation")
    }

    // MARK: - Enforcement Tests

    @Test("Enforce with no drift (no correction needed)")
    func testEnforceNoDrift() throws {
        let cellCount = 25
        let profiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [ParticleConservation()],
            initialProfiles: profiles,
            geometry: geometry,
            verbose: false
        )

        // No drift → profiles should be unchanged
        let (corrected, results) = enforcer.enforce(
            profiles: profiles,
            geometry: geometry,
            step: 1000,
            time: 1.0
        )

        #expect(results.count == 1, "Should have 1 result")
        #expect(!results[0].corrected, "Should not have corrected")
        #expect(results[0].relativeDrift < 1e-10, "Drift should be near zero")
    }

    @Test("Enforce with small drift (correction applied)")
    func testEnforceSmallDrift() throws {
        let cellCount = 25
        let initialProfiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [ParticleConservation(driftTolerance: 0.005)],  // 0.5% tolerance
            initialProfiles: initialProfiles,
            geometry: geometry,
            verbose: false
        )

        // Simulate 1% drift
        let drifted = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 0.99e20)

        let (corrected, results) = enforcer.enforce(
            profiles: drifted,
            geometry: geometry,
            step: 1000,
            time: 1.0
        )

        #expect(results.count == 1, "Should have 1 result")
        #expect(results[0].corrected, "Should have corrected")
        #expect(results[0].relativeDrift > 0.005, "Drift should exceed tolerance")
        #expect(abs(results[0].correctionFactor - 1.0101) < 0.001, "Correction factor should be ~1/0.99")
    }

    @Test("Enforce multiple laws sequentially")
    func testEnforceMultipleLaws() throws {
        let cellCount = 25
        let initialProfiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [
                ParticleConservation(driftTolerance: 0.005),
                EnergyConservation(driftTolerance: 0.01)
            ],
            initialProfiles: initialProfiles,
            geometry: geometry,
            verbose: false
        )

        // Simulate drift in both particle and energy
        // Use 9800 eV (2% reduction) to clearly exceed 1% tolerance
        let drifted = try createProfiles(cellCount: cellCount, electronTemperature: 9800.0, ionTemperature: 9800.0, electronDensity: 0.99e20)

        let (corrected, results) = enforcer.enforce(
            profiles: drifted,
            geometry: geometry,
            step: 1000,
            time: 1.0
        )

        #expect(results.count == 2, "Should have 2 results")

        // First result: ParticleConservation
        #expect(results[0].lawName == "ParticleConservation")
        #expect(results[0].corrected, "Particle conservation should be corrected")

        // Second result: EnergyConservation
        #expect(results[1].lawName == "EnergyConservation")
        #expect(results[1].corrected, "Energy conservation should be corrected")
    }

    @Test("Sequential application: particle then energy")
    func testSequentialApplication() throws {
        let cellCount = 25
        let initialProfiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let particleLaw = ParticleConservation()
        let energyLaw = EnergyConservation()

        let enforcer = ConservationEnforcer(
            laws: [particleLaw, energyLaw],
            initialProfiles: initialProfiles,
            geometry: geometry,
            verbose: false
        )

        // Drift: density -1%, temperature -2% (to clearly exceed tolerances)
        let drifted = try createProfiles(cellCount: cellCount, electronTemperature: 9800.0, ionTemperature: 9800.0, electronDensity: 0.99e20)

        let (corrected, results) = enforcer.enforce(
            profiles: drifted,
            geometry: geometry,
            step: 1000,
            time: 1.0
        )

        // After particle correction, density should be restored
        let N_corrected = particleLaw.computeConservedQuantity(
            profiles: corrected,
            geometry: geometry
        )
        let N0 = particleLaw.computeConservedQuantity(
            profiles: initialProfiles,
            geometry: geometry
        )
        let particleError = abs(N_corrected - N0) / N0
        #expect(particleError < 0.001, "Particle conservation should be restored")

        // After energy correction, energy should be restored
        let E_corrected = energyLaw.computeConservedQuantity(
            profiles: corrected,
            geometry: geometry
        )
        let E0 = energyLaw.computeConservedQuantity(
            profiles: initialProfiles,
            geometry: geometry
        )
        let energyError = abs(E_corrected - E0) / E0
        #expect(energyError < 0.001, "Energy conservation should be restored")
    }

    // MARK: - Interval Tests

    @Test("shouldEnforce check")
    func testShouldEnforce() throws {
        let cellCount = 25
        let profiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [ParticleConservation()],
            initialProfiles: profiles,
            geometry: geometry,
            enforcementInterval: 1000,
            verbose: false
        )

        #expect(!enforcer.shouldEnforce(step: 0), "Should not enforce at step 0")
        #expect(!enforcer.shouldEnforce(step: 500), "Should not enforce at step 500")
        #expect(enforcer.shouldEnforce(step: 1000), "Should enforce at step 1000")
        #expect(!enforcer.shouldEnforce(step: 1500), "Should not enforce at step 1500")
        #expect(enforcer.shouldEnforce(step: 2000), "Should enforce at step 2000")
    }

    // MARK: - Diagnostics Tests

    @Test("Compute current drift")
    func testComputeCurrentDrift() throws {
        let cellCount = 25
        let initialProfiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [
                ParticleConservation(),
                EnergyConservation()
            ],
            initialProfiles: initialProfiles,
            geometry: geometry,
            verbose: false
        )

        // Simulate drift
        let drifted = try createProfiles(cellCount: cellCount, electronTemperature: 9900.0, ionTemperature: 9900.0, electronDensity: 0.99e20)

        let drifts = enforcer.computeCurrentDrift(profiles: drifted, geometry: geometry)

        #expect(drifts.count == 2, "Should have 2 drift measurements")
        #expect(drifts[0].lawName == "ParticleConservation")
        #expect(drifts[0].drift > 0.009, "Particle drift should be ~1%")
        #expect(drifts[1].lawName == "EnergyConservation")
        #expect(drifts[1].drift > 0.009, "Energy drift should be ~1%")
    }

    @Test("Laws summary")
    func testLawsSummary() throws {
        let cellCount = 25
        let profiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [
                ParticleConservation(driftTolerance: 0.005),
                EnergyConservation(driftTolerance: 0.01)
            ],
            initialProfiles: profiles,
            geometry: geometry,
            verbose: false
        )

        let summary = enforcer.lawsSummary()

        #expect(summary.contains("ParticleConservation"), "Summary should include ParticleConservation")
        #expect(summary.contains("0.50%"), "Summary should show 0.5% tolerance")
        #expect(summary.contains("EnergyConservation"), "Summary should include EnergyConservation")
        #expect(summary.contains("1.00%"), "Summary should show 1.0% tolerance")
    }

    // MARK: - Edge Cases

    @Test("Enforce with large drift (clamped correction)")
    func testEnforceLargeDrift() throws {
        let cellCount = 25
        let initialProfiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [ParticleConservation(driftTolerance: 0.005)],
            initialProfiles: initialProfiles,
            geometry: geometry,
            verbose: false
        )

        // Simulate 30% drift (should be clamped to 20%)
        let drifted = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 0.7e20)

        let (_, results) = enforcer.enforce(
            profiles: drifted,
            geometry: geometry,
            step: 1000,
            time: 1.0
        )

        #expect(results[0].corrected, "Should have corrected")
        #expect(abs(results[0].correctionFactor - 1.2) < 0.01, "Correction should be clamped to 1.2")
    }

    @Test("Enforce with empty laws array")
    func testEnforceEmptyLaws() throws {
        let cellCount = 25
        let profiles = try createProfiles(cellCount: cellCount, electronTemperature: 10000.0, ionTemperature: 10000.0, electronDensity: 1e20)
        let geometry = createGeometry(cellCount: cellCount)

        let enforcer = ConservationEnforcer(
            laws: [],
            initialProfiles: profiles,
            geometry: geometry,
            verbose: false
        )

        let (corrected, results) = enforcer.enforce(
            profiles: profiles,
            geometry: geometry,
            step: 1000,
            time: 1.0
        )

        #expect(results.isEmpty, "Should have no results")

        // Profiles should be unchanged
        eval(profiles.electronDensity.value, corrected.electronDensity.value)
        let ne_orig = profiles.electronDensity.value.asArray(Float.self)
        let ne_corr = corrected.electronDensity.value.asArray(Float.self)

        for (orig, corr) in zip(ne_orig, ne_corr) {
            #expect(abs(orig - corr) < 1e-10, "Profiles should be unchanged")
        }
    }

    @Test("ConservationResult summary")
    func testConservationResultSummary() {
        let result = ConservationResult(
            lawName: "ParticleConservation",
            referenceQuantity: 1.0e21,
            currentQuantity: 0.995e21,
            relativeDrift: 0.005,
            correctionFactor: 1.005,
            corrected: true,
            time: 1.0,
            step: 1000
        )

        let summary = result.summary()

        #expect(summary.contains("ParticleConservation"), "Summary should include law name")
        #expect(summary.contains("0.500%"), "Summary should show drift percentage")
        #expect(summary.contains("1.005000"), "Summary should show correction factor")
        #expect(summary.contains("✓ Corrected"), "Summary should show corrected status")
    }
}
