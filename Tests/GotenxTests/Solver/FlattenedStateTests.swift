import Testing
import MLX
@testable import GotenxCore

@Suite("FlattenedState Tests")
struct FlattenedStateTests {

    @Test("FlattenedState construction from CoreProfiles")
    func testConstructionFromProfiles() throws {
        let profiles = CoreProfiles(
            ionTemperature: .full([5], value: Float(1.0)),
            electronTemperature: .full([5], value: Float(2.0)),
            electronDensity: .full([5], value: Float(3.0)),
            poloidalFlux: .full([5], value: Float(4.0))
        )

        let flattened = try FlattenedState(profiles: profiles)

        #expect(flattened.layout.cellCount == 5)
        #expect(flattened.layout.totalSize == 20)
        #expect(flattened.values.shape == [20])
    }

    @Test("FlattenedState layout validation")
    func testLayoutValidation() throws {
        let layout = try FlattenedState.StateLayout(cellCount: 10)

        #expect(layout.cellCount == 10)
        #expect(layout.totalSize == 40)
        #expect(layout.tiRange == 0..<10)
        #expect(layout.teRange == 10..<20)
        #expect(layout.neRange == 20..<30)
        #expect(layout.psiRange == 30..<40)

        // Should not throw
        try layout.validate()
    }

    @Test("FlattenedState round-trip conversion")
    func testRoundTripConversion() throws {
        let originalProfiles = CoreProfiles(
            ionTemperature: .full([5], value: Float(1.0)),
            electronTemperature: .full([5], value: Float(2.0)),
            electronDensity: .full([5], value: Float(3.0)),
            poloidalFlux: .full([5], value: Float(4.0))
        )

        let flattened = try FlattenedState(profiles: originalProfiles)
        let reconstructed = flattened.toCoreProfiles()

        #expect(originalProfiles == reconstructed)
    }

    @Test("FlattenedState shape mismatch error")
    func testShapeMismatchError() throws {
        // Create profiles with inconsistent shapes (should fail)
        let inconsistentProfiles = CoreProfiles(
            ionTemperature: .full([5], value: Float(1.0)),
            electronTemperature: .full([10], value: Float(2.0)),  // Different shape!
            electronDensity: .full([5], value: Float(3.0)),
            poloidalFlux: .full([5], value: Float(4.0))
        )

        #expect(throws: FlattenedState.FlattenedStateError.self) {
            let _ = try FlattenedState(profiles: inconsistentProfiles)
        }
    }

    @Test("FlattenedState invalid cell count")
    func testInvalidCellCount() throws {
        #expect(throws: FlattenedState.FlattenedStateError.self) {
            let _ = try FlattenedState.StateLayout(cellCount: 0)
        }

        #expect(throws: FlattenedState.FlattenedStateError.self) {
            let _ = try FlattenedState.StateLayout(cellCount: -1)
        }
    }
}
