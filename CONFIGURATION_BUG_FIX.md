# Configuration Bug Fix: adaptiveConfig Not Being Passed

## Date
2025-10-27

## Status
✅ **FIXED** - Configuration propagation issue resolved

---

## Summary

**Root Cause**: `SimulationRunner.swift` was not passing the `adaptiveConfig` parameter to `SimulationOrchestrator`, causing it to use the default configuration instead of the user-specified configuration.

**Impact**:
- Configured `minimumTimeStep: 1e-5` was being ignored
- Default `effectiveMinDt = 1e-4` was used instead
- This prevented dt retry loop from working when `nextDt < minimumTimestep`

**Fix**: Added `adaptiveConfig` parameter to SimulationOrchestrator initialization in SimulationRunner.swift

---

## Investigation

### 1. Configuration Flow (Before Fix)

```
SimulationPresets.swift (Gotenx app)
  ↓
  AdaptiveTimestepConfig(minimumTimeStep: 1e-5, ...)
  ↓
  SimulationConfiguration.time.adaptive
  ↓
  [ENCODED TO JSON]
  ↓
  Simulation.configurationData (SwiftData)
  ↓
  [DECODED FROM JSON]
  ↓
  SimulationRunner(config: config)
  ↓
  ❌ SimulationRunner.initialize() ← PARAMETER NOT PASSED!
  ↓
  SimulationOrchestrator(..., adaptiveConfig: .default)  ← Uses default!
  ↓
  TimeStepCalculator(minTimestep: .default.effectiveMinDt)
  ↓
  minimumTimestep = 1e-4  ← WRONG VALUE!
```

### 2. Default Configuration Values

From `TimeConfiguration.swift:66-72`:

```swift
public static let `default` = AdaptiveTimestepConfig(
    minimumTimeStep: nil,              // Use fraction instead
    minimumTimeStepFraction: 0.001,    // maximumTimeStep / 1000
    maximumTimeStep: 1e-1,             // 0.1s
    safetyFactor: 0.9,
    maximumTimeStepGrowth: 1.2
)
```

**Computed value**:
```swift
effectiveMinDt = maximumTimeStep * minimumTimeStepFraction
               = 1e-1 * 0.001
               = 1e-4  ← This is what we were seeing!
```

### 3. Test Results Showing Bug

From simulation logs:

```
[DEBUG-RETRY] Solver did not converge, evaluating retry:
[DEBUG-RETRY]   Current dt: 0.00018000001
[DEBUG-RETRY]   Next dt (halved): 9.000001e-05
[DEBUG-RETRY]   Minimum timestep: 0.000100000005  ← 1e-4 (default)
[DEBUG-RETRY]   nextDt < minimum? true
[DEBUG-RETRY] ❌ Cannot retry: nextDt (9.000001e-05) < minimumTimestep (0.000100000005)
```

**Expected**: `Minimum timestep: 1e-5` (from configuration)
**Actual**: `Minimum timestep: 1e-4` (from `.default`)

### 4. User Configuration (Ignored Before Fix)

From `SimulationPresets.swift:83-89`:

```swift
let adaptiveConfig = AdaptiveTimestepConfig(
    minimumTimeStep: 1e-5,             // ← Configured but ignored!
    minimumTimeStepFraction: nil,
    maximumTimeStep: 1e-3,
    safetyFactor: 0.9,
    maximumTimeStepGrowth: 1.2
)
```

With `minimumTimeStep: 1e-5` explicitly set:
```swift
effectiveMinDt = minimumTimeStep  // Explicit value takes precedence
               = 1e-5   // ← Should be this!
```

---

## The Fix

### File: `SimulationRunner.swift`

**Location**: Lines 75-86

**Before**:
```swift
// Initialize orchestrator with provided models
self.orchestrator = await SimulationOrchestrator(
    staticParams: staticParams,
    initialProfiles: serializableProfiles,
    transport: transportModel,
    sources: sourceModels,
    mhdModels: mhdModelsToUse,
    samplingConfig: .realTimePlotting
)
// ❌ adaptiveConfig parameter missing - uses .default
```

**After**:
```swift
// Get adaptive config from time configuration (or use default if not specified)
let adaptiveConfig = config.time.adaptive ?? .default

// Initialize orchestrator with provided models
self.orchestrator = await SimulationOrchestrator(
    staticParams: staticParams,
    initialProfiles: serializableProfiles,
    transport: transportModel,
    sources: sourceModels,
    mhdModels: mhdModelsToUse,
    samplingConfig: .realTimePlotting,
    adaptiveConfig: adaptiveConfig  // ✅ CRITICAL: Pass adaptive config
)
```

---

## Configuration Flow (After Fix)

```
SimulationPresets.swift (Gotenx app)
  ↓
  AdaptiveTimestepConfig(minimumTimeStep: 1e-5, ...)
  ↓
  SimulationConfiguration.time.adaptive
  ↓
  [ENCODED TO JSON]
  ↓
  Simulation.configurationData (SwiftData)
  ↓
  [DECODED FROM JSON]
  ↓
  SimulationRunner(config: config)
  ↓
  ✅ let adaptiveConfig = config.time.adaptive ?? .default
  ↓
  SimulationOrchestrator(..., adaptiveConfig: adaptiveConfig)
  ↓
  TimeStepCalculator(minTimestep: adaptiveConfig.effectiveMinDt)
  ↓
  minimumTimestep = 1e-5  ← ✅ CORRECT VALUE!
```

---

## Expected Behavior After Fix

### 1. Configuration Propagation

```
[DEBUG-PRESET] AdaptiveTimestepConfig created:
[DEBUG-PRESET]   minimumTimeStep: Optional(1e-05)
[DEBUG-PRESET]   minimumTimeStepFraction: nil
[DEBUG-PRESET]   maximumTimeStep: 0.001
[DEBUG-PRESET]   effectiveMinDt: 1e-05
    ↓
[DEBUG-INIT] AdaptiveTimestepConfig received:
[DEBUG-INIT]   minimumTimeStep: Optional(1e-05)
[DEBUG-INIT]   minimumTimeStepFraction: nil
[DEBUG-INIT]   maximumTimeStep: 0.001
[DEBUG-INIT]   effectiveMinDt: 1e-05
    ↓
[DEBUG-TSCALC] TimeStepCalculator init:
[DEBUG-TSCALC]   minTimestep: 1e-05  ← ✅ Correct!
[DEBUG-TSCALC]   maxTimestep: 0.001
[DEBUG-TSCALC]   stabilityFactor: 0.9
```

### 2. dt Retry Loop

When Newton solver fails with `dt = 1.8e-4`:

```
[DEBUG-RETRY] Solver did not converge, evaluating retry:
[DEBUG-RETRY]   Current dt: 0.00018000001
[DEBUG-RETRY]   Next dt (halved): 9.000001e-05  ← 9e-5
[DEBUG-RETRY]   Minimum timestep: 1.000000e-05  ← ✅ 1e-5 (configured)
[DEBUG-RETRY]   nextDt < minimum? false  ← ✅ Retry allowed!
[DEBUG-RETRY] ✅ Retrying with smaller dt
```

**Retry will succeed because**: `9e-5 > 1e-5` (above minimum)

---

## Testing Checklist

After rebuilding Gotenx app with updated swift-gotenx:

- [ ] **Step 0**: Should converge (already working)
- [ ] **Step 1**: Should converge (already working)
- [ ] **Step 2**: Should trigger retry (was failing before)
  - [x] dt growth cap: `6.4e-4 → 1.8e-4` (capped at 1.2×)
  - [x] Linear error check: `1.21e-02 > 1e-3` detected
  - [ ] **NEW**: dt retry: `1.8e-4 → 9e-5` (should now work!)
  - [ ] Convergence: Should eventually converge with smaller dt

---

## Related Files

1. **SimulationRunner.swift** (Fixed)
   - Lines 75-86: Added adaptiveConfig parameter

2. **SimulationOrchestrator.swift** (Already correct)
   - Lines 92-98: Debug logging for received config
   - Lines 103-107: Creates TimeStepCalculator with effectiveMinDt

3. **TimeConfiguration.swift** (Reference)
   - Lines 33-87: AdaptiveTimestepConfig definition
   - Lines 56-64: effectiveMinDt computed property

4. **TimeStepCalculator.swift** (Already correct)
   - Lines 38-42: Debug logging for init parameters
   - Lines 22, 52-53: minimumTimestep property

5. **SimulationPresets.swift** (Gotenx app - Already correct)
   - Lines 83-89: User configuration with minimumTimeStep: 1e-5

---

## Lessons Learned

### 1. Why Did This Happen?

- `adaptiveConfig` is an optional parameter with a default value
- No compiler error when parameter is omitted
- Configuration was "working" for other purposes (CFL safety factor)
- Only affected minTimestep behavior

### 2. Why Was It Hard to Find?

- Configuration was correctly set in SimulationPresets.swift
- Configuration was correctly encoded/decoded via JSON
- Debug logs showed "configuration being set" in SimulationPresets
- Problem was in a *different file* (SimulationRunner.swift)
- Clean rebuilds didn't help because code logic was wrong

### 3. How to Prevent Similar Issues?

**Option 1: Remove default parameter (breaking change)**
```swift
// Force caller to provide adaptiveConfig explicitly
public init(
    ...,
    adaptiveConfig: AdaptiveTimestepConfig  // No default
) async
```

**Option 2: Add validation in init**
```swift
public init(..., adaptiveConfig: AdaptiveTimestepConfig = .default) async {
    if adaptiveConfig == .default {
        print("[WARNING] Using default AdaptiveTimestepConfig - was this intentional?")
    }
}
```

**Option 3: Document requirement clearly**
```swift
/// - Parameter adaptiveConfig: Adaptive timestep configuration
///   **IMPORTANT**: Pass this explicitly from `config.time.adaptive`
///   to respect user configuration. Using default is rarely correct.
```

**Recommendation**: Use Option 3 (documentation) for backward compatibility, plus add debug logging.

---

## Build Verification

```bash
$ cd ~/Desktop/swift-gotenx
$ swift build
Building for debugging...
[5/8] Compiling GotenxCore SimulationRunner.swift
[6/8] Emitting module GotenxCore
Build complete! (3.52s)
```

✅ Build succeeded with no errors or warnings.

---

## Next Steps

1. **Rebuild Gotenx app** with updated swift-gotenx
2. **Run test simulation** (Step 0 → Step 2)
3. **Verify logs**:
   - `[DEBUG-INIT] effectiveMinDt: 1e-05` (not 1e-4)
   - `[DEBUG-TSCALC] minTimestep: 1e-05` (not 1e-4)
   - `[DEBUG-RETRY] nextDt < minimum? false` (retry succeeds)
4. **Observe behavior**:
   - dt retry should work: 1.8e-4 → 9e-5 → 4.5e-5 → ...
   - Eventually converges or reaches actual minimum (1e-5)
   - Step 2+ should complete successfully

---

## Status Update

**Before Fix**: ❌ Configuration ignored, used default minimumTimeStep=1e-4, retry failed
**After Fix**: ✅ Configuration respected, uses minimumTimeStep=1e-5, retry should work
**Build Status**: ✅ Compiled successfully
**Ready for Testing**: Yes - please rebuild Gotenx app and retest

---

## References

- **DT_GROWTH_CAP_IMPLEMENTATION.md**: Implementation of dt growth cap and early termination
- **DT_GROWTH_CAP_TEST_RESULTS.md**: Test results showing configuration issue
- **SIMULATION_RUNNABLE_INTEGRATION.md**: SimulationRunner protocol documentation
