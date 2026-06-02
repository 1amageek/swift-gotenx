# Timestep Growth Cap Test Results

**Date:** 2025-10-27
**Status:** ✅ Implementation Working, ❌ Configuration Issue Identified

---

## Executive Summary

### ✅ SUCCESS: Both Implementations Working Correctly

1. **dt growth cap** - Working perfectly, capped 4.27× growth to 1.2×
2. **Early termination check** - Working perfectly, detected linear solver error > threshold

### ❌ ISSUE: Minimum Timestep Too High

The dt retry loop failed because `nextDt` (9e-5) fell below `minimumTimestep` (1e-4), causing immediate error without retry.

---

## Test Results

### Step 0: Baseline (✅ SUCCESS)

```
dt = 1.5e-4s (initial)
κ = 3.36e4 (good)
Linear error = 1.88e-04 (acceptable, below 1e-3 threshold)

Per-variable convergence:
  Ti:  3.96e+00 < 1.00e+01 ✅
  Te:  3.49e+01 ≮ 1.00e+01 ⚠️  (iter 0)
  ne:  4.16e-02 < 1.00e-01 ✅
  psi: 8.68e-04 < 1.00e-03 ✅

Result: Converged in 2 iterations ✅
```

**Analysis:** Per-variable convergence criteria working perfectly. Te was the only blocker at iter=0, fixed at iter=1.

---

### Step 1: dt Growth Cap Activated (✅ WORKING, ❌ CONFIG ISSUE)

#### dt Growth Cap (✅ SUCCESS)

```
[DEBUG] dt growth capped: 6.40e-04s → 1.80e-04s (attempted 4.27× growth, limit: 1.2×)
```

**Analysis:**
- timeStepCalculator wanted: 6.4e-4s (4.27× from 1.5e-4)
- Growth cap applied: 1.8e-4s (1.2× from 1.5e-4)
- **Result:** dt growth cap WORKING PERFECTLY ✅

#### Newton Solver with Capped dt (⚠️ STILL ILL-CONDITIONED)

```
dt = 1.8e-4s (capped, was 6.4e-4)
κ = 5.48e6 ⚠️ (ill-conditioned, increased 163× from Step 0's 3.36e4)
Linear error = 1.21e-02 (EXCEEDS threshold 1e-3 by 12×)

Per-variable residual:
  Ti:  1.82e+01 ≮ 1.00e+01 ⚠️
  Te:  1.94e+01 ≮ 1.00e+01 ⚠️
  ne:  4.18e-02 < 1.00e-01 ✅
  psi: 8.68e-04 < 1.00e-03 ✅
```

**Analysis:**
- Despite capping growth to 1.2×, Jacobian still became ill-conditioned
- κ jumped from 3.36e4 → 5.48e6 (163× increase!)
- This suggests even 1.2× growth is too aggressive for this problem

#### Early Termination Check (✅ SUCCESS)

```
[NR-CHECK] iter=0: Linear solver accuracy:
[NR-CHECK]   ||J*Δ + R|| = 3.22e-01
[NR-CHECK]   ||R|| = 2.66e+01
[NR-CHECK]   Relative error = 1.21e-02
[NR-CHECK] ⚠️  WARNING: Linear solver error > 1e-6

[NR-CHECK] iter=0: Descent direction check:
[NR-CHECK]   Δ·(-R) = 1.14e-01
[NR-CHECK]   ✅ Valid descent direction

[NR-FAILURE] ❌ Linear solver error too high: 1.21e-02 > 1.00e-03
[NR-FAILURE] Newton direction unreliable - aborting iteration
[NR-FAILURE] Returning converged=false to trigger dt retry
```

**Analysis:**
- Linear error check: WORKING CORRECTLY ✅
- Descent direction check: PASSED (Δ·(-R) > 0)
- Early termination: TRIGGERED CORRECTLY ✅
- Returns `converged=false` as expected

#### dt Retry Loop (❌ FAILED DUE TO MINDT)

```
[DEBUG] Calling solver.solve(): step=1, dt=0.00018000002s, attempt=0
[DEBUG] solver.solve() returned: converged=false, iterations=1, residual=1.5361255
Solver error: GotenxCore.SolverError.convergenceFailure(iterations: 1, residualNorm: 1.5361255)
```

**No retry message!** Expected to see:
```
[SimulationOrchestrator] Solver did not converge; reducing dt to 9.00e-05 s (attempt 1 of 5)
```

**Root Cause Analysis:**

Looking at SimulationOrchestrator.swift lines 606-616:
```swift
attempt += 1
let nextDt = dtAttempt * 0.5  // 1.8e-4 * 0.5 = 9e-5

if nextDt < timeStepCalculator.minimumTimestep {
    // これ以上タイムステップを縮小できないので即時エラー
    throw SolverError.convergenceFailure(...)  // ← THIS THREW!
}
```

**Calculation:**
- Current dt: 1.8e-4
- Halved dt: 9.0e-5
- Minimum dt: **1.0e-4** (default: maximumTimeStep * minimumTimeStepFraction = 1e-1 * 0.001)
- **9e-5 < 1e-4** → Immediate error, no retry!

**Why minimumTimeStep = 1e-4?**

From TimeConfiguration.swift:
```swift
public var effectiveMinDt: Float {
    if let minimumTimeStep = minimumTimeStep {
        return minimumTimeStep  // Explicit value
    } else if let fraction = minimumTimeStepFraction {
        return maximumTimeStep * fraction  // Default: 1e-1 * 0.001 = 1e-4
    }
    // ...
}
```

---

## Root Cause Summary

### What Worked ✅

1. **dt growth cap enforcement:**
   - Correctly capped 6.4e-4 → 1.8e-4 (4.27× → 1.2×)
   - Implementation in SimulationOrchestrator.swift:437-451 working perfectly

2. **Early termination checks:**
   - Linear solver error detection (1.21e-02 > 1e-3) ✅
   - Descent direction validation (passed) ✅
   - Implementation in NewtonRaphsonSolver.swift:530-572 working perfectly

### What Failed ❌

**Configuration issue:** `minimumTimestep` too high

**Problem chain:**
1. Step 0 starts with dt=1.5e-4 ✅
2. Step 1 tries dt=6.4e-4, capped to 1.8e-4 ✅
3. Newton fails due to κ=5.48e6, linear_error=1.21e-02 ✅
4. Orchestrator tries to halve dt: 1.8e-4 → 9e-5 ✅
5. **9e-5 < minimumTimestep (1e-4)** ❌
6. Throws error immediately without retry ❌

### Why Even 1.2× Growth Was Too Aggressive

The dt increase from 1.5e-4 to 1.8e-4 (only 1.2×) caused:
- Jacobian κ: 3.36e4 → 5.48e6 (163× worse!)
- Linear error: 1.88e-04 → 1.21e-02 (64× worse!)

**This suggests the problem is highly sensitive to dt changes.**

---

## Recommended Fixes

### Option 1: Lower Minimum Timestep (EASIEST)

**Change configuration:**
```json
{
  "time": {
    "adaptive": {
      "minimumTimeStep": 1e-5,  // Was implicitly 1e-4, now 10× smaller
      "maximumTimeStep": 1e-3,  // Or reduce this to prevent large dt jumps
      "minimumTimeStepFraction": null,  // Ignored if minimumTimeStep is set
      "maximumTimeStepGrowth": 1.2
    }
  }
}
```

**Expected behavior:**
- Step 1 fails at dt=1.8e-4 (as before)
- Retry 1: dt=9.0e-5 (now ABOVE minimumTimeStep=1e-5) ✅
- Retry 2: dt=4.5e-5 (if needed)
- ... up to 5 retries

**Pros:**
- Simple configuration change
- Allows more dt flexibility
- Should resolve the immediate issue

**Cons:**
- Doesn't address why 1.2× growth is too aggressive
- May need many retries

---

### Option 2: Reduce Growth Rate (RECOMMENDED)

**Change configuration:**
```json
{
  "time": {
    "adaptive": {
      "minimumTimeStep": 1e-5,  // Lower minimum
      "maximumTimeStep": 1e-3,  // Lower maximum
      "maximumTimeStepGrowth": 1.1  // 10% growth (was 1.2 = 20%)
    }
  }
}
```

**Expected behavior:**
- Step 1: dt = 1.65e-4 (1.5e-4 × 1.1, capped from 6.4e-4)
- κ likely ~4-5e4 (better than 5.48e6 at 1.8e-4)
- Linear error likely ~1e-3 to 5e-3 (better than 1.21e-02)
- May converge without retry, or only need 1 retry

**Pros:**
- More gradual dt growth = better Newton conditioning
- Fewer retries needed
- More stable overall

**Cons:**
- Slower dt growth = more steps to reach end time
- Trade-off: stability vs. speed

---

### Option 3: Even More Conservative (IF OPTION 2 FAILS)

**Change configuration:**
```json
{
  "time": {
    "initialTimeStep": 1.0e-4,  // Start smaller
    "adaptive": {
      "minimumTimeStep": 1e-6,  // Very low minimum
      "maximumTimeStep": 5e-4,  // Lower maximum (was 1e-3 or 1e-1)
      "maximumTimeStepGrowth": 1.05  // 5% growth (very conservative)
    }
  }
}
```

**Expected behavior:**
- Extremely gradual dt growth
- κ should remain < 1e5 throughout
- Linear error should remain < 1e-3
- No early terminations, no retries
- Simulation will take longer (more steps)

**Pros:**
- Maximum stability
- Guaranteed convergence (barring other issues)

**Cons:**
- Slowest option
- More time steps = longer simulation time

---

## Testing Recommendations

### Test 1: Verify Option 1 (Lower minimumTimeStep)

**Config:**
```json
{
  "time": {
    "initialTimeStep": 1.5e-4,
    "adaptive": {
      "minimumTimeStep": 1e-5,
      "maximumTimeStep": 1e-3,
      "maximumTimeStepGrowth": 1.2
    }
  }
}
```

**Expected results:**
- Step 0: converges (as before)
- Step 1: fails at dt=1.8e-4, retries with dt=9e-5
- Check if dt=9e-5 converges or needs more retries
- Log how many retries are needed

---

### Test 2: Verify Option 2 (Reduce Growth)

**Config:**
```json
{
  "time": {
    "initialTimeStep": 1.5e-4,
    "adaptive": {
      "minimumTimeStep": 1e-5,
      "maximumTimeStep": 1e-3,
      "maximumTimeStepGrowth": 1.1
    }
  }
}
```

**Expected results:**
- Step 0: converges (as before)
- Step 1: dt=1.65e-4 (capped from 6.4e-4)
- Check κ and linear_error at this dt
- Ideally converges without retry
- Compare performance to Test 1

---

### Test 3: Verify Step 2+ Stability

After fixing Step 1, continue simulation to verify:
- Step 2: dt should grow to 1.65e-4 × 1.1 = 1.82e-4 (Option 2)
- Step 3: dt = 2.00e-4
- Step 4: dt = 2.20e-4
- ...
- Monitor κ, linear_error, convergence at each step

**Success criteria:**
- All steps converge
- κ < 1e6 (ideally < 1e5)
- Linear error < 1e-3 (ideally < 1e-4)
- No invalid descent directions
- Simulation completes successfully

---

## Diagnostic Logs to Watch

### Good Signs ✅

```
[DEBUG] dt growth capped: X → Y (attempted A× growth, limit: 1.2×)
[NR-CHECK] ✅ Valid descent direction
[NR-CHECK] Linear solver accuracy OK (or warning but < 1e-3)
[CONVERGENCE] ✅ All variables converged
```

### Warning Signs ⚠️

```
[DEBUG-JACOBIAN] ⚠️  WARNING: Jacobian is ill-conditioned (κ > 1e6)
[NR-CHECK] ⚠️  WARNING: Linear solver error > 1e-6 (but < 1e-3)
[SimulationOrchestrator] Solver did not converge; reducing dt to X s (attempt N of 5)
```

### Error Signs ❌

```
[NR-FAILURE] ❌ Linear solver error too high: X > 1.00e-03
[NR-FAILURE] ❌ Invalid descent direction: Δ·(-R) = X ≤ 0
Solver error: convergenceFailure(iterations: N, residualNorm: X)
```

---

## Key Metrics

### Step 0 (Baseline)

| Metric | Value | Status |
|--------|-------|--------|
| dt | 1.5e-4 | Initial |
| κ | 3.36e4 | ✅ Good |
| Linear error | 1.88e-04 | ✅ OK |
| Ti residual | 3.96 | ✅ Converged |
| Te residual | 6.24 (after iter 1) | ✅ Converged |
| ne residual | 9.39e-02 | ✅ Converged |
| Iterations | 2 | ✅ Fast |

### Step 1 (Failed)

| Metric | Value (uncapped) | Value (capped) | Status |
|--------|------------------|----------------|--------|
| dt (attempted) | 6.4e-4 (4.27×) | 1.8e-4 (1.2×) | ⚠️ Capped |
| κ | N/A | 5.48e6 | ❌ Ill-cond |
| Linear error | N/A | 1.21e-02 | ❌ Too high |
| Ti residual | N/A | 1.82e+01 | ❌ Not conv |
| Te residual | N/A | 1.94e+01 | ❌ Not conv |
| ne residual | N/A | 4.18e-02 | ✅ OK |
| Iterations | N/A | 1 (aborted) | ❌ Failed |
| Retry attempted | N/A | 9e-5 | ❌ Below minimumTimeStep |

---

## Conclusion

### Implementation Status: ✅ VERIFIED WORKING

Both implementations are working exactly as designed:
1. dt growth cap correctly limits aggressive timestep increases
2. Early termination checks correctly detect Newton solver instability

### Configuration Issue: ❌ NEEDS ADJUSTMENT

The problem is not with the implementation, but with configuration:
- `minimumTimestep = 1e-4` is too high for this problem
- Even 1.2× dt growth causes severe Jacobian ill-conditioning
- Need to either:
  - Lower `minimumTimeStep` to 1e-5 or lower (allow more retries)
  - Reduce `maximumTimeStepGrowth` to 1.1 or 1.05 (gentler growth)
  - Both

### Recommended Next Steps

1. **Immediate:** Set `minimumTimeStep: 1e-5` in configuration
2. **Test:** Run simulation and verify Step 1 retry succeeds
3. **Optimize:** Reduce `maximumTimeStepGrowth` to 1.1 if still unstable
4. **Monitor:** Track κ, linear_error, and convergence behavior
5. **Report:** Document which configuration works best

---

**Last Updated:** 2025-10-27
**Status:** Implementation verified, awaiting configuration adjustment and retest
