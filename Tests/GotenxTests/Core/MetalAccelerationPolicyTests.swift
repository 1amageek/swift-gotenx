import Testing
@testable import GotenxCore

@Suite("Metal Acceleration Policy Tests")
struct MetalAccelerationPolicyTests {
    @Test("Default Metal device supports Metal 4")
    func defaultMetalDeviceSupportsMetal4() throws {
        let info = try MetalAccelerationPolicy.requireMetal4Device()

        #expect(!info.name.isEmpty)
        #expect(info.supportsMetal4)
        #expect(info.supportsMetal4CoreAPI)
        #expect(info.hasUnifiedMemory)
        #expect(info.recommendedMaxWorkingSetSize > 0)
    }
}
