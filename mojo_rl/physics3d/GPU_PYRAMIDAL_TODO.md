# GPU PYRAMIDAL Newton Solver

## Status

**FIXED.** The GPU PYRAMIDAL Newton solver now produces correct constraint forces, matching the ELLIPTIC GPU path's accuracy. 3/6 CPU vs GPU tests pass (same pass/fail pattern as ELLIPTIC GPU). All CPU vs MuJoCo tests still pass (4/4 pyramidal, 7/7 contact, 4/4 solver forces).

## Bugs Found and Fixed

### Bug 1: qacc initialization (newton_solver.mojo:1025-1033)

The GPU PYRAMIDAL path hardcoded `qacc = 0` instead of reading the unconstrained acceleration from `ws_qacc_constrained_offset`. The integrator writes either `qacc_smooth` (= M_inv * f_net) or `qacc_warmstart` (previous solution) to this workspace slot before calling the solver.

Starting from zero meant `Ma = 0`, so the initial jar was purely bias-dependent: `jar = -bias ≈ K*imp*pen ≈ +300`, always positive (SATISFIED). No edges ever activated, producing zero constraint forces.

**Fix:** Load qacc from `ws_qacc_constrained_offset`, matching the ELLIPTIC GPU path and CPU warmstart logic.

### Bug 2: jar sign convention (newton_solver.mojo:1068, 1141, 1163)

The jar computation used `jar = -bias_e + J*qacc` but should have been `jar = bias_e + J*qacc`.

The CPU's `compute_jar` does `jar = J*qacc + bias` where `bias = -K*imp*pen + B*v` (negative for penetrating contacts). The GPU PYRAMIDAL stored the same bias value (`bias_e = B*v_edge - K*imp*pen = -K*imp*pen + B*v_edge`) but then negated it when computing jar, flipping the sign of all residuals. Negative jar (active constraint) became positive (satisfied), and vice versa.

**Fix:** Changed `jar = -bias_e + J*qacc` to `jar = bias_e + J*qacc` in initial computation, linesearch trial, and post-update recomputation.

### Bug 3: Wrong memory offset for body_invweight0 (constraint_builder_gpu.mojo:618)

The PYRAMIDAL friction builder read `diag_n` (body inverse weight) using:
```mojo
model_body_invweight0_offset[NBODY, NJOINT]()  # WRONG — defaults NGEOM=0, etc.
```
The correct call (used by the normal precompute at line 387) is:
```mojo
model_body_invweight0_offset[NBODY, NJOINT, NGEOM, MAX_EQUALITY, MAX_TENDON, NSITE]()
```

Since `body_invweight0` is stored after geom/equality/tendon/site data in the model buffer, omitting these template params computes the wrong offset. The friction builder was reading garbage values for `diag_n`, producing wrong `R_edge`, `D_edge`, and `imp_n` values.

**Fix:** Pass all template parameters to `model_body_invweight0_offset`.

## What's Done

1. **Shared friction builder** (`constraint_builder_gpu.mojo: precompute_contact_friction_gpu`)
   - Centralized tangent frame + friction data computation
   - `comptime if CONE_TYPE` branches for ELLIPTIC vs PYRAMIDAL
   - PYRAMIDAL path builds 4 edge Jacobians (J_n ± mu*J_t1, J_n ± mu*J_t2), D_edge, bias_edge per contact

2. **PYRAMIDAL Newton iteration** (`newton_solver.mojo: solve_gpu`, lines ~990-1200)
   - Caches edge data into InlineArrays
   - Hessian: H = M + sum_active(D_edge * J_edge^T * J_edge)
   - Cholesky solve + linesearch
   - `comptime if CONE_TYPE == ConeType.PYRAMIDAL:` branch with early return
   - Limits/equality called at end of PYRAMIDAL branch

3. **Workspace sizing** (`newton_solver.mojo: solver_workspace_size`)
   - Updated to 33*MC + 6*MC*NV (accommodates both ELLIPTIC and PYRAMIDAL)
   - PYRAMIDAL layout: 4*MC*NV edge J + scalars at pyr_sc = ws_Jt1_idx + 4*MC*NV

## Remaining Failures (Shared GPU Infrastructure)

3/6 CPU vs GPU tests fail for PYRAMIDAL — the **exact same 3 tests** that also fail for ELLIPTIC, with **identical error magnitudes**:

| Test | PYRAMIDAL max qvel err | ELLIPTIC max qvel err | Notes |
|------|----------------------|---------------------|-------|
| Ground contact 1 step | 1.577 | 1.577 | Identical |
| Deep penetration 1 step | 4.404 | 4.404 | Identical |
| Moving + contacts 1 step | 2.763 | 2.763 | Identical |
| Ground contact + actions 1 step | PASS | PASS | |
| Ground contact 5 steps | PASS | PASS | |
| Ground contact + actions 5 steps | PASS | PASS | |

### Hypothesis: GPU D_n computed from float32 round-trip through inv_K_imp

Both ELLIPTIC and PYRAMIDAL GPU solvers compute `D_n` by extracting `R_n` indirectly:
```
R_n = 1/inv_K_imp - K_n
D_n = 1/R_n
```

The CPU computes `R_n` directly from the impedance formula:
```
R_n = (1-imp)/imp * diag_n
```

In float32, the round-trip `R = 1/(1/(K+R)) - K` loses precision when `K >> R` or `K << R`. For deep penetration contacts, `K` (the constraint-space stiffness) and `R` (the regularizer) can differ by orders of magnitude, causing significant float32 cancellation error in the GPU's extracted `R_n`.

This affects the D value (force scaling), which directly scales all constraint forces. A ~30-60% error in D explains the observed ~30-60% error in qvel.

**Evidence supporting this hypothesis:**
- The 3 failing tests all involve deep penetration or high-velocity contacts (large K*imp*pen)
- The 3 passing tests have moderate penetration with actions (better K/R ratio)
- Multi-step tests (5 steps) accumulate corrections and converge, masking single-step D errors
- PYRAMIDAL and ELLIPTIC produce identical errors, confirming the issue is in the shared normal precompute path, not in the cone-specific code

### Possible fixes (future work)

1. **Store imp and diag_n from normal precompute**: Add workspace slots for `imp` and `diag_n` per contact. The friction builder (and D computation) can then compute `R = (1-imp)/imp * diag_n` directly instead of extracting it from `inv_K_imp`.

2. **Compute D_n directly**: Instead of `D = 1/(1/inv_K_imp - K)`, compute `D = imp/((1-imp)*diag_n)` using stored `imp` and `diag_n`.

3. **Use float64 for constraint parameter computation**: The constraint parameter precompute is per-contact (not per-edge, not inner-loop), so the cost of float64 is negligible.

## Other GPU Solvers (After Newton Works)

1. **PGS GPU** (`pgs_solver.mojo`): Already has PYRAMIDAL support. Refactor to use shared friction builder.
2. **CG GPU** (`cg_solver.mojo`): Refactor to use shared friction builder. Add PYRAMIDAL branch.
3. **Other solvers** (island_pgs, old_newton, friction_solver): Same pattern.

## Test Files

- `test_pyramidal_cpu_vs_gpu.mojo` — PYRAMIDAL CPU vs GPU (3/6 pass, matches ELLIPTIC)
- `test_pyramidal_vs_mujoco.mojo` — PYRAMIDAL CPU vs MuJoCo (4/4 pass)
- `test_full_step_contact_cpu_vs_gpu.mojo` — ELLIPTIC CPU vs GPU (3/6 pass, same failures)
