// CompilationCache.swift
// Memory-based compilation cache for MLX compiled functions

import Foundation

/// MLX compilation cache (memory-only)
///
/// **Important**: Unlike JAX's persistent disk cache, MLX does not support
/// saving compiled functions to disk. This cache is process-local and
/// memory-only.
///
/// **Use case**: Prevent recompilation when the same static configuration
/// is used multiple times within a single process (e.g., interactive menu
/// reruns with only dynamic parameter changes).
///
/// **Limitations**:
/// - Cache is lost when process exits
/// - Cannot share cache between processes
/// - No persistent storage to disk
///
/// **Comparison to JAX**:
/// - JAX: Persistent cache to `~/.cache/jax/`
/// - MLX: Memory-only cache (this implementation)
public actor CompilationCache {
    /// Cached compiled functions
    ///
    /// Key: CacheKey (based on static configuration)
    /// Value: Type-erased compiled function
    private var cache: [CacheKey: Any] = [:]

    /// Maximum number of cached entries
    private let maximumEntries: Int

    /// Initialization
    ///
    /// - Parameter maximumEntries: Maximum cache size (default: 10)
    public init(maximumEntries: Int = 10) {
        self.maximumEntries = maximumEntries
    }

    /// Return a cached compiled function, or compile and cache it.
    ///
    /// If a compiled function for the given key exists in cache, return it.
    /// Otherwise, compile the function, cache it, and return it.
    ///
    /// - Parameters:
    ///   - key: Cache key (based on static configuration)
    ///   - compile: Closure that compiles the function
    /// - Returns: Compiled function (cached or newly compiled)
    public func compiledFunction<In, Out>(
        for key: CacheKey,
        compile: () -> (In) -> Out
    ) -> (In) -> Out {
        // Check cache
        if let cached = cache[key] as? (In) -> Out {
            return cached
        }

        // Compile (not in cache)
        let compiled = compile()

        // Store in cache (with LRU eviction if full)
        if cache.count >= maximumEntries {
            // Remove first entry (simple eviction strategy)
            if let firstKey = cache.keys.first {
                cache.removeValue(forKey: firstKey)
            }
        }
        cache[key] = compiled

        return compiled
    }

    /// Clear entire cache
    ///
    /// Useful for freeing memory or forcing recompilation.
    public func clear() {
        cache.removeAll()
    }

    /// Remove specific entry from cache
    ///
    /// - Parameter key: Cache key to remove
    public func remove(key: CacheKey) {
        cache.removeValue(forKey: key)
    }

    /// Get cache statistics
    ///
    /// - Returns: Number of cached entries
    public func size() -> Int {
        return cache.count
    }
}

// MARK: - Cache Key

/// Cache key based on static configuration
///
/// Two configurations are considered identical for caching if:
/// - Mesh resolution is the same (cellCount)
/// - Solver type is the same
/// - Evolution flags are the same (which PDEs to solve)
///
/// Dynamic parameters (boundaries, transport coefficients, etc.) do NOT
/// affect the cache key because they don't change the computation graph
/// structure.
public struct CacheKey: Hashable {
    /// Hash of mesh configuration
    let meshHash: Int

    /// Solver type ("linear", "newton", "optimizer")
    let solverType: String

    /// Bitmask of evolution flags
    /// Bit 0: ion temperature
    /// Bit 1: electron temperature
    /// Bit 2: electron density
    /// Bit 3: poloidal flux
    let evolutionFlags: Int

    /// Theta parameter (for time integration scheme)
    let theta: Float

    /// Create cache key from static configuration
    ///
    /// - Parameter staticConfig: Static runtime configuration
    public init(staticConfig: StaticConfig) {
        // Hash mesh parameters that affect graph structure
        var hasher = Hasher()
        hasher.combine(staticConfig.mesh.cellCount)
        hasher.combine(staticConfig.mesh.geometryType.rawValue)
        self.meshHash = hasher.finalize()

        // Solver type affects computation graph
        self.solverType = staticConfig.solver.type

        // Evolution flags: which PDEs to evolve
        var flags = 0
        if staticConfig.evolution.ionHeat { flags |= (1 << 0) }
        if staticConfig.evolution.electronHeat { flags |= (1 << 1) }
        if staticConfig.evolution.electronDensity { flags |= (1 << 2) }
        if staticConfig.evolution.poloidalFlux { flags |= (1 << 3) }
        self.evolutionFlags = flags

        // Theta parameter affects discretization
        self.theta = staticConfig.scheme.theta
    }

    /// Hash function
    public func hash(into hasher: inout Hasher) {
        hasher.combine(meshHash)
        hasher.combine(solverType)
        hasher.combine(evolutionFlags)
        hasher.combine(theta)
    }

    /// Equality comparison
    public static func == (lhs: CacheKey, rhs: CacheKey) -> Bool {
        return lhs.meshHash == rhs.meshHash &&
               lhs.solverType == rhs.solverType &&
               lhs.evolutionFlags == rhs.evolutionFlags &&
               lhs.theta == rhs.theta
    }
}

// MARK: - Global Cache Instance

/// Global compilation cache
///
/// Shared across all SimulationOrchestrator instances to maximize reuse.
private let globalCompilationCache = CompilationCache()

extension CompilationCache {
    /// Access global compilation cache
    ///
    /// This is the recommended way to access the cache to ensure
    /// cache sharing across multiple simulation instances.
    public static var global: CompilationCache {
        return globalCompilationCache
    }
}

// MARK: - Environment Variable Control

extension CompilationCache {
    /// Check if compilation cache is enabled via environment variable
    ///
    /// Environment variable: `GOTENX_COMPILATION_CACHE_ENABLED`
    /// - "true" or "1": enabled (default)
    /// - "false" or "0": disabled
    ///
    /// - Returns: true if cache should be used
    public static func isEnabled() -> Bool {
        guard let envValue = ProcessInfo.processInfo.environment["GOTENX_COMPILATION_CACHE_ENABLED"] else {
            return true  // Enabled by default
        }

        let lowercased = envValue.lowercased()
        return lowercased == "true" || lowercased == "1"
    }

    /// Get cache size limit from environment variable
    ///
    /// Environment variable: `GOTENX_COMPILATION_CACHE_SIZE`
    /// Default: 10 entries
    ///
    /// - Returns: Maximum number of cache entries
    public static func sizeLimit() -> Int {
        guard let envValue = ProcessInfo.processInfo.environment["GOTENX_COMPILATION_CACHE_SIZE"],
              let size = Int(envValue), size > 0 else {
            return 10  // Default size
        }
        return size
    }
}
