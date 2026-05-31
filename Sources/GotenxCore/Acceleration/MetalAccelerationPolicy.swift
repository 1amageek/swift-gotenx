import Metal

#if !GOTENX_METAL4_REQUIRED
#error("GotenxCore must be built with GOTENX_METAL4_REQUIRED")
#endif

// MARK: - Metal Acceleration Policy

/// Runtime description of the Metal device used by the simulation.
public struct MetalDeviceInfo: Sendable, Equatable {
    public let name: String
    public let supportsMetal4: Bool
    public let supportsMetal4CoreAPI: Bool
    public let hasUnifiedMemory: Bool
    public let recommendedMaxWorkingSetSize: UInt64
}

/// Enforces the acceleration baseline for this package.
public enum MetalAccelerationPolicy {
    /// Require an Apple GPU that exposes the Metal 4 GPU family.
    public static func requireMetal4Device() throws -> MetalDeviceInfo {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ConfigurationError.invalidValue(
                key: "metal.device",
                value: "none",
                reason: "A Metal device is required for MLX acceleration"
            )
        }

        guard device.supportsFamily(.metal4) else {
            throw ConfigurationError.invalidValue(
                key: "metal.gpuFamily",
                value: device.name,
                reason: "swift-gotenx requires macOS 26.4+ and a Metal 4 capable device"
            )
        }

        guard let commandQueue = device.makeMTL4CommandQueue() else {
            throw ConfigurationError.invalidValue(
                key: "metal.commandQueue",
                value: device.name,
                reason: "A Metal 4 command queue is required"
            )
        }

        guard let commandAllocator = device.makeCommandAllocator() else {
            throw ConfigurationError.invalidValue(
                key: "metal.commandAllocator",
                value: device.name,
                reason: "A Metal 4 command allocator is required"
            )
        }

        guard let commandBuffer = device.makeCommandBuffer() else {
            throw ConfigurationError.invalidValue(
                key: "metal.commandBuffer",
                value: device.name,
                reason: "A Metal 4 command buffer is required"
            )
        }

        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.endEncoding()
        }
        commandBuffer.endCommandBuffer()
        _ = commandQueue

        return MetalDeviceInfo(
            name: device.name,
            supportsMetal4: true,
            supportsMetal4CoreAPI: true,
            hasUnifiedMemory: device.hasUnifiedMemory,
            recommendedMaxWorkingSetSize: device.recommendedMaxWorkingSetSize
        )
    }
}
