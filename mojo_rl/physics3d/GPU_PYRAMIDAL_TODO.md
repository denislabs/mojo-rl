# GPU PYRAMIDAL Newton Solver

## Status

**ALL PASSING.** 6/6 CPU vs GPU tests pass for both PYRAMIDAL and ELLIPTIC. All CPU vs MuJoCo tests still pass (4/4 pyramidal, 7/7 contact, 4/4 solver forces).

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

### Bug 4: GPU _compute_invweight0_gpu produces wrong translational body_invweight0 (model_def.mojo, model_def_from_xml.mojo)

The GPU `_compute_invweight0_gpu` kernel computes `body_invweight0[2*i]` (translational component) incorrectly — values differ by ~9x from the CPU's `compute_body_invweight0`. The rotational component `body_invweight0[2*i+1]` was correct.

This caused `diag_n` in the constraint builder to be ~9x too large, which made `R_n` ~9x too large and `D` (force scaling) ~9x too small. All constraint forces were ~9x weaker than they should be, producing dramatically wrong qvel/qpos.

Both `ModelDef.init_model_gpu` and `ModelDefFromXML.init_model_gpu` called `_compute_invweight0_gpu` after copying CPU-computed values, overwriting correct data with wrong GPU-computed values.

**Fix:** Removed the `_compute_invweight0_gpu` calls. For `ModelDefFromXML`, the CPU `setup_model_and_data` already computes correct `body_invweight0` which is serialized via `copy_invweight0_to_buffer`. For `ModelDef`, added CPU-side invweight0 computation using a temporary Model/Data.

### Additional: Store imp/diag_n in common normal workspace (constraint_builder_gpu.mojo)

Extended the common normal workspace block from `13*MC + 2*MC*NV` to `15*MC + 2*MC*NV` by adding `imp_n` (slot 13) and `diag_n` (slot 14). The friction builder now computes `R_n = (1-imp)/imp * diag_n` directly instead of the lossy float32 round-trip `R = 1/inv_K_imp - K`. While this was not the root cause of the large errors (Bug 4 was), it improves float32 precision for constraint parameters.

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

## Previous Failures (Now Fixed)

All 3/6 CPU vs GPU test failures were caused by Bug 4 (incorrect GPU `body_invweight0`), not by the float32 precision hypothesis. After fixing Bug 4, all 6/6 tests pass for both PYRAMIDAL and ELLIPTIC with errors at float32 rounding level (~1e-6).

## Other GPU Solvers (After Newton Works)

1. **PGS GPU** (`pgs_solver.mojo`): Already has PYRAMIDAL support. Refactor to use shared friction builder.
2. **CG GPU** (`cg_solver.mojo`): Refactor to use shared friction builder. Add PYRAMIDAL branch.
3. **Other solvers** (island_pgs, old_newton, friction_solver): Same pattern.

## Test Files

- `test_pyramidal_cpu_vs_gpu.mojo` — PYRAMIDAL CPU vs GPU (3/6 pass, matches ELLIPTIC)
- `test_pyramidal_vs_mujoco.mojo` — PYRAMIDAL CPU vs MuJoCo (4/4 pass)
- `test_full_step_contact_cpu_vs_gpu.mojo` — ELLIPTIC CPU vs GPU (3/6 pass, same failures)

---

## 2026-07-31 — blocked kernel aligned with the per-env pyramidal path

`_newton_blocked_fields_kernel` (the NVIDIA + PYRAMIDAL production path, one
env per block, cooperative in shared memory) had drifted from
`_newton_solve_env`. It built `ME = NE*MC + MAX_LIM` — contact edges and joint
limits only — and classified every edge as ONE-SIDED, including
`if force_sh[e] > 0` for the Hessian.

Two row types were therefore missing on NVIDIA:

* **dry-friction rows** (`mjCNSTR_FRICTION_DOF`) — added to the per-env path by
  `04a7c508` and never mirrored here.
* **tendon-limit rows** (`mjCNSTR_LIMIT_TENDON`) — added with the ball_in_cup
  port.

Aligned: `ME` now includes `MAX_FRIC + MAX_TLIM`; shared `kind_e_sh`, `R_e_sh`,
`floss_e_sh`, `state_e_sh`; all six one-sided sites (initial force, Hessian,
p0 derivatives, line-search reference cost, trial cost, accepted cost) go
through `scalar_row_state` / `scalar_row_force` / `scalar_row_cost`. The
host-side workaround that routed limited-tendon models AWAY from this kernel
was deleted — the path is correct now rather than avoided.

### How to test it

```bash
# The dedicated blocked-path tendon gate. Runs on Apple too (float32), but the
# NVIDIA run is the one that matters — it is the production configuration.
pixi run -e nvidia mojo run -I . tests/physics3d/test_newton_blocked_tendon_fields.mojo

# Sibling gate: the pre-existing blocked kernel golden (walker2d, no tendons).
# Proves the row-state refactor did not disturb one-sided rows.
pixi run -e nvidia mojo run -I . tests/physics3d/test_newton_blocked_fields.mojo

# The per-env pyramidal path, for the cross-check the two tests compare against.
pixi run -e nvidia mojo run -I . tests/physics3d/test_newton_solve_fields.mojo

# End-to-end model gate (per-env path; CPU, so NVIDIA adds nothing here).
pixi run mojo run -I . tests/dm_control/test_ball_in_cup_vs_dm_control.mojo
```

### Status of verification

| what | verified where |
|---|---|
| cooperative GPU carrying TENDON rows | ✅ Apple/Metal, `test_newton_blocked_tendon_fields`, rel err **0.0** |
| blocked vs per-env agreement on tendon rows | ✅ same test, rel err **0.0** |
| one-sided rows unchanged by the refactor | ✅ `test_newton_blocked_fields` golden unchanged |
| tendon-row physics vs MuJoCo | ✅ 8.9e-16 (via `euler.mojo` routed at `solve_newton_blocked`) |

### STILL OPEN

Item 1 needs an NVIDIA box. Item 2 does NOT — it was mis-scoped as
NVIDIA-only when first written. `solve_newton_blocked` runs on Metal at
float32 (that is exactly how `test_newton_blocked_tendon_fields` gates the
tendon rows), so the friction rows can be gated on Apple too, given a model
that expresses them.

1. **[NVIDIA] Shared-memory headroom on a large model.** `ME` drives
   `Je_sh = ME * V_SIZE`, the dominant threadgroup term. Growth is ~10-13%
   (dm_control humanoid `ME` 170 -> 201). Overflow is a LAUNCH FAILURE, which
   is loud, not a wrong answer — but it would take humanoid off this kernel.
   Run any humanoid GPU gate under `-e nvidia` and watch for a launch error.
2. **[Apple is fine] Dry-friction rows are UNEXERCISED by any model.** Only `finger` sets
   `frictionloss` and finger is ELLIPTIC, so it never reaches this kernel. The
   friction rows here are written but ungated — the same "the gate cannot
   express the defect" situation that has produced several silent bugs in this
   engine. Closing this needs a purpose-built PYRAMIDAL model with
   `frictionloss` (a two-body slider with friction and a ground contact is
   enough); until then treat those rows as unverified.
