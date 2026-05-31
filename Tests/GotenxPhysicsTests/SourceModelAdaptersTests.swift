import Testing
import Foundation
import MLX
@testable import GotenxCore
@testable import GotenxPhysics

/// Tests for SourceModelAdapters ensuring metadata is always provided
///
/// CRITICAL: These tests prevent runtime crashes from missing metadata
@Suite("SourceModelAdapters Tests")
struct SourceModelAdaptersTests {
    
    // MARK: - Test Helpers
    
    func createTestProfiles(nCells: Int = 50) throws -> CoreProfiles {
        let Ti = MLXArray.full([nCells], values: MLXArray(Float(5000.0)))  // 5 keV
        let Te = MLXArray.full([nCells], values: MLXArray(Float(5000.0)))  // 5 keV
        let ne = MLXArray.full([nCells], values: MLXArray(Float(5e19)))    // 5e19 m^-3
        let psi = MLXArray.linspace(Float(0.0), Float(1.0), count: nCells)

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: Ti),
            electronTemperature: EvaluatedArray(evaluating: Te),
            electronDensity: EvaluatedArray(evaluating: ne),
            poloidalFlux: EvaluatedArray(evaluating: psi)
        )
    }
    
    func createTestGeometry() -> Geometry {
        let meshConfig = MeshConfig(
            nCells: 50,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5,
            geometryType: .circular
        )
        return Geometry(config: meshConfig)
    }

    func expectClose(_ lhs: MLXArray, _ rhs: MLXArray, name: String) {
        #expect(allClose(lhs, rhs).item(Bool.self), "\(name) should match")
    }
    
    // MARK: - Ohmic Heating Source Tests
    
    @Test("Ohmic heating source returns metadata")
    func testOhmicHeatingSourceReturnsMetadata() throws {
        let source = OhmicHeatingSource()
        let profiles = try createTestProfiles()
        let geometry = createTestGeometry()

        let terms = source.computeTerms(
            profiles: profiles,
            geometry: geometry,
            params: SourceParameters(modelType: "ohmic_heating")
        )
        
        // CRITICAL: metadata must not be nil
        #expect(terms.metadata != nil, "OhmicHeatingSource must provide metadata")
        
        guard let metadata = terms.metadata else {
            Issue.record("Metadata is nil")
            return
        }
        
        #expect(metadata.entries.count >= 1, "Should have at least 1 metadata entry")
        
        // Find ohmic metadata
        let ohmicMetadata = metadata.entries.first { $0.modelName == "ohmic_heating" }
        #expect(ohmicMetadata != nil, "Should contain ohmic_heating metadata")
        #expect(ohmicMetadata?.category == .ohmic, "Category should be .ohmic")
        
        // Ohmic heating goes to electrons only
        #expect(ohmicMetadata?.ionPower == 0, "Ion power should be 0")
        #expect(ohmicMetadata?.electronPower != 0, "Electron power should be non-zero")
    }

    @Test("Ohmic heating solver terms match diagnostic terms without metadata")
    func testOhmicHeatingSolverTermsMatchDiagnosticTerms() throws {
        let source = OhmicHeatingSource()
        let profiles = try createTestProfiles()
        let geometry = createTestGeometry()
        let params = SourceParameters(modelType: "ohmic_heating")

        let diagnostic = source.computeTerms(profiles: profiles, geometry: geometry, params: params)
        let solver = source.computeTermsForSolver(profiles: profiles, geometry: geometry, params: params)

        expectClose(solver.ionHeating.value, diagnostic.ionHeating.value, name: "ion heating")
        expectClose(solver.electronHeating.value, diagnostic.electronHeating.value, name: "electron heating")
        expectClose(solver.particleSource.value, diagnostic.particleSource.value, name: "particle source")
        expectClose(solver.currentSource.value, diagnostic.currentSource.value, name: "current source")
        #expect(solver.metadata == nil, "Solver terms should skip diagnostic metadata")
    }
    
    // MARK: - Bremsstrahlung Source Tests
    
    @Test("Bremsstrahlung source returns metadata")
    func testBremsstrahlungSourceReturnsMetadata() throws {
        let source = BremsstrahlungSource()
        let profiles = try createTestProfiles()
        let geometry = createTestGeometry()
        
        let terms = source.computeTerms(
            profiles: profiles,
            geometry: geometry,
            params: SourceParameters(modelType: "bremsstrahlung")
        )

        // CRITICAL: metadata must not be nil
        #expect(terms.metadata != nil, "BremsstrahlungSource must provide metadata")
        
        guard let metadata = terms.metadata else {
            Issue.record("Metadata is nil")
            return
        }
        
        #expect(metadata.entries.count >= 1)
        
        let bremsMetadata = metadata.entries.first { $0.modelName == "bremsstrahlung" }
        #expect(bremsMetadata != nil, "Should contain bremsstrahlung metadata")
        #expect(bremsMetadata?.category == .radiation, "Category should be .radiation")

        // Bremsstrahlung is a power loss (negative)
        if let electronPower = bremsMetadata?.electronPower {
            #expect(electronPower <= 0, "Electron power should be negative (loss)")
        }
    }

    @Test("Bremsstrahlung solver terms match diagnostic terms without metadata")
    func testBremsstrahlungSolverTermsMatchDiagnosticTerms() throws {
        let source = BremsstrahlungSource()
        let profiles = try createTestProfiles()
        let geometry = createTestGeometry()
        let params = SourceParameters(modelType: "bremsstrahlung")

        let diagnostic = source.computeTerms(profiles: profiles, geometry: geometry, params: params)
        let solver = source.computeTermsForSolver(profiles: profiles, geometry: geometry, params: params)

        expectClose(solver.ionHeating.value, diagnostic.ionHeating.value, name: "ion heating")
        expectClose(solver.electronHeating.value, diagnostic.electronHeating.value, name: "electron heating")
        expectClose(solver.particleSource.value, diagnostic.particleSource.value, name: "particle source")
        expectClose(solver.currentSource.value, diagnostic.currentSource.value, name: "current source")
        #expect(solver.metadata == nil, "Solver terms should skip diagnostic metadata")
    }
    
    // MARK: - Ion-Electron Exchange Source Tests
    
    @Test("Ion-electron exchange source returns metadata")
    func testIonElectronExchangeSourceReturnsMetadata() throws {
        let source = IonElectronExchangeSource()
        let profiles = try createTestProfiles()
        let geometry = createTestGeometry()
        
        let terms = source.computeTerms(
            profiles: profiles,
            geometry: geometry,
            params: SourceParameters(modelType: "ion_electron_exchange")
        )

        // CRITICAL: metadata must not be nil
        #expect(terms.metadata != nil, "IonElectronExchangeSource must provide metadata")
        
        guard let metadata = terms.metadata else {
            Issue.record("Metadata is nil")
            return
        }
        
        #expect(metadata.entries.count >= 1)
        
        let exchangeMetadata = metadata.entries.first { $0.modelName == "ion_electron_exchange" }
        #expect(exchangeMetadata != nil, "Should contain ion_electron_exchange metadata")
        #expect(exchangeMetadata?.category == .other, "Category should be .other")
        
        // Energy conservation: ion power + electron power = 0
        if let meta = exchangeMetadata {
            let totalPower = meta.ionPower + meta.electronPower
            #expect(abs(totalPower) < 1e-3, "Energy should be conserved (total ~0)")
        }
    }

    @Test("Ion-electron exchange solver terms match diagnostic terms without metadata")
    func testIonElectronExchangeSolverTermsMatchDiagnosticTerms() throws {
        let source = IonElectronExchangeSource()
        let profiles = try createTestProfiles()
        let geometry = createTestGeometry()
        let params = SourceParameters(modelType: "ion_electron_exchange")

        let diagnostic = source.computeTerms(profiles: profiles, geometry: geometry, params: params)
        let solver = source.computeTermsForSolver(profiles: profiles, geometry: geometry, params: params)

        expectClose(solver.ionHeating.value, diagnostic.ionHeating.value, name: "ion heating")
        expectClose(solver.electronHeating.value, diagnostic.electronHeating.value, name: "electron heating")
        expectClose(solver.particleSource.value, diagnostic.particleSource.value, name: "particle source")
        expectClose(solver.currentSource.value, diagnostic.currentSource.value, name: "current source")
        #expect(solver.metadata == nil, "Solver terms should skip diagnostic metadata")
    }
    
    // MARK: - Fusion Power Source Tests
    
    @Test("Fusion power source returns metadata")
    func testFusionPowerSourceReturnsMetadata() throws {
        let source = FusionPowerSource()
        let profiles = try createTestProfiles()
        let geometry = createTestGeometry()
        
        let terms = source.computeTerms(
            profiles: profiles,
            geometry: geometry,
            params: SourceParameters(modelType: "fusion")
        )

        // CRITICAL: metadata must not be nil
        #expect(terms.metadata != nil, "FusionPowerSource must provide metadata")
        
        guard let metadata = terms.metadata else {
            Issue.record("Metadata is nil")
            return
        }
        
        #expect(metadata.entries.count >= 1)
        
        let fusionMetadata = metadata.entries.first { $0.modelName.contains("fusion") }
        #expect(fusionMetadata != nil, "Should contain fusion metadata")
        #expect(fusionMetadata?.category == .fusion, "Category should be .fusion")
    }
    
    // MARK: - Composite Source Model Tests
    
    @Test("Composite source model merges metadata from all sources")
    func testCompositeSourceModelMergesMetadata() throws {
        let ohmic = OhmicHeatingSource()
        let brems = BremsstrahlungSource()
        let exchange = IonElectronExchangeSource()
        
        let composite = CompositeSourceModel(sources: [
            "ohmic": ohmic,
            "brems": brems,
            "exchange": exchange
        ])
        
        let profiles = try createTestProfiles()
        let geometry = createTestGeometry()
        
        let terms = composite.computeTerms(
            profiles: profiles,
            geometry: geometry,
            params: SourceParameters(modelType: "composite")
        )

        // CRITICAL: metadata must not be nil
        #expect(terms.metadata != nil, "CompositeSourceModel must provide metadata")

        guard let metadata = terms.metadata else {
            Issue.record("Metadata is nil")
            return
        }
        
        // Should have metadata from all 3 sources
        #expect(metadata.entries.count >= 3, "Should have metadata from all sources")
        
        // Verify each source's metadata is present
        let hasOhmic = metadata.entries.contains { $0.modelName == "ohmic_heating" }
        let hasBrems = metadata.entries.contains { $0.modelName == "bremsstrahlung" }
        let hasExchange = metadata.entries.contains { $0.modelName == "ion_electron_exchange" }
        
        #expect(hasOhmic, "Should have ohmic metadata")
        #expect(hasBrems, "Should have bremsstrahlung metadata")
        #expect(hasExchange, "Should have exchange metadata")
    }
    
    @Test("Composite source with empty sources returns empty metadata")
    func testCompositeSourceModelEmptySources() throws {
        let composite = CompositeSourceModel(sources: [:])
        
        let profiles = try createTestProfiles()
        let geometry = createTestGeometry()
        
        let terms = composite.computeTerms(
            profiles: profiles,
            geometry: geometry,
            params: SourceParameters(modelType: "composite")
        )

        // Even with no sources, metadata should not be nil
        #expect(terms.metadata != nil, "CompositeSourceModel must always provide metadata")
        
        // But it should be empty
        if let metadata = terms.metadata {
            #expect(metadata.entries.isEmpty, "Empty composite should have empty metadata")
        }
    }
}
