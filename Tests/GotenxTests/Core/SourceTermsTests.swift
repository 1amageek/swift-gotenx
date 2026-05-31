import Testing
import Foundation
import MLX
@testable import GotenxCore

/// Tests for SourceTerms metadata handling
///
/// CRITICAL: Ensures metadata is never nil to prevent runtime crashes
@Suite("SourceTerms Tests")
struct SourceTermsTests {
    
    // MARK: - Zero Source Terms Tests
    
    @Test("Zero source terms has empty metadata")
    func testZeroSourceTermsHasEmptyMetadata() {
        let zero = SourceTerms.zero(nCells: 50)
        
        // CRITICAL: metadata must not be nil
        #expect(zero.metadata != nil, "SourceTerms.zero() must provide metadata")
        
        // Should be empty but not nil
        if let metadata = zero.metadata {
            #expect(metadata.entries.isEmpty, "Zero source should have empty metadata")
        }
    }
    
    @Test("Zero source terms has correct shape")
    func testZeroSourceTermsShape() {
        let nCells = 100
        let zero = SourceTerms.zero(nCells: nCells)
        
        #expect(zero.ionHeating.shape == [nCells])
        #expect(zero.electronHeating.shape == [nCells])
        #expect(zero.particleSource.shape == [nCells])
        #expect(zero.currentSource.shape == [nCells])
    }

    @Test("Localized exchange-scale heating remains valid")
    func testLocalizedExchangeScaleHeatingRemainsValid() {
        let nCells = 10
        let localizedHeating = EvaluatedArray(evaluating: MLXArray.full([nCells], values: MLXArray(Float(2_000.0))))
        let zero = EvaluatedArray.zeros([nCells])

        let source = SourceTerms(
            ionHeating: localizedHeating,
            electronHeating: zero,
            particleSource: zero,
            currentSource: zero,
            metadata: SourceMetadataCollection.empty
        )

        #expect(source.ionHeating.shape == [nCells])
    }

    @Test("Localized exchange-scale cooling remains valid")
    func testLocalizedExchangeScaleCoolingRemainsValid() {
        let nCells = 10
        let localizedCooling = EvaluatedArray(evaluating: MLXArray.full([nCells], values: MLXArray(Float(-2_000.0))))
        let zero = EvaluatedArray.zeros([nCells])

        let source = SourceTerms(
            ionHeating: zero,
            electronHeating: localizedCooling,
            particleSource: zero,
            currentSource: zero,
            metadata: SourceMetadataCollection.empty
        )

        #expect(source.electronHeating.shape == [nCells])
    }

    // MARK: - Metadata Addition Tests

    @Test("Addition merges metadata from both sources")
    func testAdditionMergesMetadata() {
        let source1 = SourceTerms(
            ionHeating: .zeros([50]),
            electronHeating: .zeros([50]),
            particleSource: .zeros([50]),
            currentSource: .zeros([50]),
            metadata: SourceMetadataCollection(entries: [
                SourceMetadata(
                    modelName: "model_A",
                    category: .ohmic,
                    ionPower: 0,
                    electronPower: 10e6
                )
            ])
        )
        
        let source2 = SourceTerms(
            ionHeating: .zeros([50]),
            electronHeating: .zeros([50]),
            particleSource: .zeros([50]),
            currentSource: .zeros([50]),
            metadata: SourceMetadataCollection(entries: [
                SourceMetadata(
                    modelName: "model_B",
                    category: .fusion,
                    ionPower: 5e6,
                    electronPower: 5e6
                )
            ])
        )
        
        let combined = source1 + source2
        
        // CRITICAL: metadata should be merged
        #expect(combined.metadata != nil, "Combined source must have metadata")
        
        if let metadata = combined.metadata {
            #expect(metadata.entries.count == 2, "Should have 2 metadata entries")
            
            let hasModelA = metadata.entries.contains { $0.modelName == "model_A" }
            let hasModelB = metadata.entries.contains { $0.modelName == "model_B" }
            
            #expect(hasModelA, "Should contain model_A metadata")
            #expect(hasModelB, "Should contain model_B metadata")
        }
    }
    
    @Test("Addition of zero and non-zero preserves metadata")
    func testAdditionWithZeroPreservesMetadata() {
        let zero = SourceTerms.zero(nCells: 50)
        
        let nonZero = SourceTerms(
            ionHeating: EvaluatedArray(evaluating: MLXArray.full([50], values: MLXArray(1.0))),
            electronHeating: .zeros([50]),
            particleSource: .zeros([50]),
            currentSource: .zeros([50]),
            metadata: SourceMetadataCollection(entries: [
                SourceMetadata(
                    modelName: "test",
                    category: .auxiliary,
                    ionPower: 1e6,
                    electronPower: 0
                )
            ])
        )
        
        let combined = zero + nonZero
        
        #expect(combined.metadata != nil)
        #expect(combined.metadata?.entries.count == 1, "Should preserve non-zero metadata")
    }
    
    @Test("Addition when both have nil metadata")
    func testAdditionBothNilMetadata() {
        // This shouldn't happen in practice, but test defensive behavior
        let source1 = SourceTerms(
            ionHeating: .zeros([50]),
            electronHeating: .zeros([50]),
            particleSource: .zeros([50]),
            currentSource: .zeros([50]),
            metadata: nil  // Explicit nil for test
        )
        
        let source2 = SourceTerms(
            ionHeating: .zeros([50]),
            electronHeating: .zeros([50]),
            particleSource: .zeros([50]),
            currentSource: .zeros([50]),
            metadata: nil
        )
        
        let combined = source1 + source2
        
        // Result should have nil metadata
        #expect(combined.metadata == nil, "Both nil should result in nil")
    }
    
    // MARK: - Metadata Preservation Tests
    
    @Test("Metadata is preserved through multiple additions")
    func testMetadataPreservedThroughMultipleAdditions() {
        let sources: [SourceTerms] = (0..<5).map { i in
            SourceTerms(
                ionHeating: .zeros([50]),
                electronHeating: .zeros([50]),
                particleSource: .zeros([50]),
                currentSource: .zeros([50]),
                metadata: SourceMetadataCollection(entries: [
                    SourceMetadata(
                        modelName: "model_\(i)",
                        category: .other,
                        ionPower: Float(i),
                        electronPower: Float(i)
                    )
                ])
            )
        }
        
        let total = sources.reduce(SourceTerms.zero(nCells: 50), +)
        
        #expect(total.metadata != nil)
        #expect(total.metadata?.entries.count == 5, "Should have all 5 metadata entries")
    }
    
    // MARK: - SourceMetadataCollection Tests
    
    @Test("Empty metadata collection")
    func testEmptyMetadataCollection() {
        let empty = SourceMetadataCollection.empty
        
        #expect(empty.entries.isEmpty)
    }
    
    @Test("Metadata collection with entries")
    func testMetadataCollectionWithEntries() {
        let metadata = SourceMetadataCollection(entries: [
            SourceMetadata(modelName: "A", category: .ohmic, ionPower: 0, electronPower: 10),
            SourceMetadata(modelName: "B", category: .fusion, ionPower: 5, electronPower: 5)
        ])
        
        #expect(metadata.entries.count == 2)
    }
    
    @Test("Metadata collection filtering by category")
    func testMetadataFilteringByCategory() {
        let metadata = SourceMetadataCollection(entries: [
            SourceMetadata(modelName: "ohmic", category: .ohmic, ionPower: 0, electronPower: 10),
            SourceMetadata(modelName: "fusion", category: .fusion, ionPower: 5, electronPower: 5),
            SourceMetadata(modelName: "ecrh", category: .auxiliary, ionPower: 0, electronPower: 20)
        ])
        
        let ohmicOnly = metadata.entries.filter { $0.category == .ohmic }
        #expect(ohmicOnly.count == 1)
        
        let auxOnly = metadata.entries.filter { $0.category == .auxiliary }
        #expect(auxOnly.count == 1)
    }
}
