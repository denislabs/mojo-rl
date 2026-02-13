# Diagnosis: Physics Instability (Acceleration-Level Solver)

## Current Status
Migrated from velocity-level to acceleration-level constraint solving.
Free-fall test is stable (rootz settles at 0.575, max_pen ~7mm).
**Random-action stress test is UNSTABLE** — max pen 1.77m, robot flies to 2.19m.
MuJoCo with same scenario: max pen 26mm, completely stable.

## Bugs Found and Fixed

### 1. Newton/CG RHS Corruption (FIXED)
RHS was computed interleaved with warm-start application, corrupting rhs for later rows.
**Fix**: Split into two loops — compute ALL rhs first, then apply all warm-starts.

### 2. Newton Jacobi Initial Guess (FIXED)
Initial guess for reduced system solve used `K` instead of `AR[c,c] = K/imp` (Hessian diagonal
with regularizer). Overshoots by factor 1/imp. For imp=0.2: 5x too large.
**Fix**: Use `A[c * num_normals + c]` (includes regularizer). CPU + GPU.

### 3. Degenerate Friction Tangent Directions (FIXED)
HalfCheetah is a 2D planar robot. One tangent direction (out-of-plane) has K ≈ 1e-10.
Friction PGS divided by K=1e-10, producing huge deltas that dominated the Coulomb cone,
starving the real tangent direction of friction budget.
**Fix**: Skip friction updates for tangent rows where K < 1e-6. Applied to all 3 CPU solvers
and GPU friction solver. Zero warm-start for degenerate directions.

### 4. Self-Collision Display Bug (FIXED)
`model.body_name[Int(ct.body_b)]` with body_b = -1 (ground) wraps to last body.
**Fix**: `model.get_body_name()` returns "world" for -1.

## Newton QP Convergence: VERIFIED GOOD
projected_grad_norm ≈ 1e-30. Newton solver converges to machine precision.
Increasing from 15 to 100 iterations only improved max pen from 1.77m to 1.31m — NOT a convergence issue.

## Random-Action Stress Test: FAILED

Ran `test_random_action_stability.mojo` — 1000 steps with random torques in [-1, 1].

### Our Engine Results (dt=0.002, frame_skip=5)
- **Max penetration: 1.77m** (first anomaly at step 231: 89mm)
- **Max velocity: 10.0** (clamped by MAX_QVEL)
- **rootz range: [-1.53, 2.19]** (robot both penetrates ground and flies)
- **Max contacts: 6-11** during instability
- **STATUS: UNSTABLE**

### MuJoCo Reference (dt=0.01, frame_skip=5)
- **Max penetration: 26mm**
- **Max velocity: ~20 m/s** (no clamp)
- **rootz range: [-0.19, 0.0]** (always near ground)
- **Max contacts: 2**
- **STATUS: STABLE** — even starting at z=0.7 (23.6mm max pen)

### 75x Penetration Gap: Root Cause Investigation

Tested multiple hypotheses. Summary of what was tried:

| Change | Max Pen | Effect |
|--------|---------|--------|
| Baseline (imp=0.2, B no imp, dt=0.002, MAXV=10) | 1.77m | — |
| Newton iterations 15→100 | 1.31m | Marginal improvement |
| imp floor 0.2→0.0001 + B*imp (MuJoCo values) + dt=0.002 | 3.9m | WORSE (contacts too soft at small dt) |
| imp=0.2 + B*imp (fix damping) + dt=0.002 | 6.0m | MUCH WORSE (weaker damping) |
| imp=0.2 + B no imp + MAX_QVEL 10→100 | 0.43m | 4x better pen, but robot flies to 12.9m (bouncing) |
| imp=0.0001 + B*imp + dt=0.01 + MAXV=100 | 965m | CATASTROPHIC |

### Key Findings

1. **Sign conventions and Jacobians are correct** — contact normal points up (0,0,1), J·qvel>0 = separating, positive force pushes body away from ground.

2. **Integration flow is correct** — qacc → qvel += qacc*dt → clamp ±10 → qpos += qvel*dt.

3. **Our aref formula does NOT match MuJoCo's actual values:**
   - MuJoCo at pen=1mm, v=-5, d=0.016: aref = 572.99
   - Our formula: K_spring*d*pen + B*d*v = 3906*0.016*0.001 + 125*0.016*5 = 11.49
   - **50x discrepancy at shallow penetration!**
   - At pen=10mm, v=-5, d=0.8: MuJoCo aref=600, our formula gives 531 → closer but still off
   - MuJoCo's effective K_spring = 3571 (not 3906 = 1/(dmax²*tc²*dr²))
   - MuJoCo's effective B_damp = 143 (not 125 = 2/(dmax*tc))
   - These DON'T match any simple formula involving solref/solimp

4. **MuJoCo's R (regularizer) also doesn't match expected formula:**
   - At d=0.8, K=1: expected R = (1/d-1)*K = 0.25, MuJoCo actual R = 0.143
   - At d=0.016, K=1: expected R = 61.5, MuJoCo actual R = 4.37
   - The discrepancy is HUGE at shallow penetration (14x weaker regularizer than expected)
   - This means MuJoCo's contacts are much STIFFER at shallow pen than the simple formula suggests

5. **The impedance floor matters enormously:**
   - imp=0.2 (our value): firm contact from first touch, prevents initial penetration but causes bouncing
   - imp=0.0001 (MuJoCo's mjMINIMP): very soft at surface, needs MuJoCo's dt=0.01 to work
   - At dt=0.002 with imp=0.0001: contacts too soft to arrest impacts before deep penetration

6. **MAX_QVEL=10 clamp hurts recovery but isn't the root cause:**
   - Raising to 100 reduces max pen (0.43m vs 1.77m) but robot flies to 12.9m
   - The bouncing from stiff contacts (imp=0.2) injects energy

7. **Our contact detection generates 2x more contacts than MuJoCo:**
   - We generate 2 contacts per capsule (both endpoints), MuJoCo generates 1 (closest point)
   - HalfCheetah: we see 6-16 contacts, MuJoCo sees 1-2
   - Redundant contacts may cause conditioning issues in the solver

8. **dt mismatch:**
   - MuJoCo: dt=0.01 → qacc correction per step = qacc*0.01
   - Ours: dt=0.002 → qacc correction per step = qacc*0.002 (5x less per substep)
   - Over frame_skip=5, total correction similar, but intermediate penetration is worse

## Priority 1 Resolution: MuJoCo's Constraint Formula — VERIFIED CORRECT

**Our formulas match MuJoCo's source code exactly.** Verified by:
1. Reading `engine_core_constraint.c` (`mj_makeImpedance`, `mj_referenceConstraint`)
2. Running comprehensive Python tests comparing our reimplementation vs MuJoCo internals

### Confirmed Formulas
```
K = 1 / (dmax² × tc² × dr²) = 3906.25  (for solref=[0.02,1.0], dmax=0.8)
B = 2 / (dmax × tc) = 125.0
aref = -B*vel - K*imp*(pos - margin)     (imp scales K term only, NOT B)
R = (1-imp)/imp × diagApprox             (where diagApprox = J*M_inv*J^T)
D = 1/R
```

### Earlier Discrepancy Explained
The R=0.143 vs expected 0.25 discrepancy was because MuJoCo OVERWRITES `efc_diagApprox`
after computing R: `diagApprox[i] = R[i] * imp / (1-imp)`. The reported diagApprox was
the ADJUSTED value, not the one used to compute R. Mathematically this adjustment is
identity (restores original value), but the intermediate computation uses the original.

### MuJoCo's 5-Parameter Impedance Function
```c
x = pen / width (clamped to [0,1])
if x <= mid:  y = x^power / mid^(power-1)
if x > mid:   y = 1 - (1-x)^power / (1-mid)^(power-1)
imp = dmin + y * (dmax - dmin)
```
Our 3-parameter cubic smoothstep (`3x²-2x³`) is equivalent when mid=0.5, power=2 (the defaults).
Small differences at shallow penetration are masked by the 0.2 impedance floor.

### Solver Formulation Also Verified
Our PGS KKT condition `a + bias + R*lambda = 0` with `bias = -aref` is equivalent to
MuJoCo's `(K+R)*f = efc_b` where `efc_b = aref + J*qacc_unconstrained`.

**Conclusion: The impedance/constraint formulas are NOT the root cause of instability.**

## dt Experiment (dt=0.01)

Tested dt=0.01 (matching MuJoCo) with current settings (imp floor=0.2, MAX_QVEL=10):
- **Result: 6.11m max pen** — WORSE than dt=0.002 (1.77m)
- Larger dt means larger position updates per step, which dominates over the benefit
  of larger per-step velocity correction
- **dt=0.002 is correct for our engine** — smaller dt gives better stability

## Updated Root Cause Analysis

Since formulas are verified correct, the remaining suspects are:

### Priority 1 (NEW): Contact Detection Quality
We generate **2 contacts per capsule** (both endpoints), MuJoCo generates **1** (closest point).
- Our 6-16 contacts vs MuJoCo's 1-2 causes solver conditioning issues
- Multiple contacts on the same body split the constraint force between them
- Redundant contacts may conflict, causing the iterative solver to oscillate
- **Fix**: Implement closest-point capsule-plane contact (single contact per geom)

### Priority 2 (NEW): Impedance Floor Causing Energy Injection
The imp=0.2 floor makes contacts stiff even at surface touch:
- Contact spring force at pen=0.1mm: K*0.2*0.0001 = 0.078 m/s² (negligible)
- But the regularizer R = 4*K is only 4x (relatively stiff solver)
- Stiff contacts on impact → large constraint forces → bouncing → energy injection
- MuJoCo with imp≈0 at surface: R→∞, contacts are very soft, no bouncing
- MuJoCo handles this with dt=0.01 where one step corrects the full velocity
- **We can't just remove the floor** (tested: 3.9m pen at dt=0.002 without floor)
- May need to tune floor relative to dt, or use position-based correction post-solver

### Priority 3: MAX_QVEL=10 Limits Recovery
- Raising to 100 reduces pen from 1.77m to 0.43m (4x better)
- But causes 12.9m flying (bouncing from stiff contacts)
- Combined fix with softer impedance might work

### Priority 4: Consider Velocity-Level Fallback
The acceleration-level approach is theoretically correct but practically harder to stabilize
at dt=0.002. The velocity-level approach was more forgiving because it directly constrains
the velocity (what matters for position updates).

## Test Scripts Created
- `tests/test_zero_action_stability.mojo` — zero-torque stability test (PASSES)
- `tests/test_random_action_stability.mojo` — random-torque stress test (FAILS)
- `tests/mujoco_random_action_test.py` — MuJoCo reference (random torques)
- `tests/mujoco_high_start_test.py` — MuJoCo at z=0.7 start
- `tests/mujoco_solver_debug.py` — MuJoCo solver internals (efc_D, efc_R, efc_force, efc_aref)
- `tests/mujoco_impedance_compare.py` — impedance function comparison
- `tests/mujoco_impedance_deep_debug.py` — comprehensive formula verification

## Current Code State
All experimental changes have been reverted. Code is back to the working baseline:
- `constraint_builder.mojo`: imp floor=0.2, bias = -K_spring*imp*pen + B_damp*v_n
- `constraint_builder_gpu.mojo`: same
- `implicit_fast_integrator.mojo`: MAX_QVEL = 10.0
- `half_cheetah_def.mojo`: DT = 0.002, FRAME_SKIP = 5
- `half_cheetah.mojo`: timestep default = HalfCheetahParams.DT (was hardcoded 0.002, now reads from params)
- `newton_solver.mojo`: NEWTON_ITERATIONS = 15
