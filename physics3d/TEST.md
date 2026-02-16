# Physics3D Test Plan

## Test Methodology

We validate our physics engine at three levels:
1. **MuJoCo vs CPU** — Compare our CPU engine output against MuJoCo (Python interop) for identical inputs
2. **CPU vs GPU** — Compare our GPU engine output against our CPU engine (ensures GPU kernels match)
3. **Standalone** — Analytical validation (e.g. pendulum period) and diagnostic/stress tests

All MuJoCo comparison tests use the HalfCheetah model. Tests run with:
```bash
cd mojo-rl && pixi run mojo run physics3d/tests/<test_file>.mojo        # CPU
cd mojo-rl && pixi run -e apple mojo run physics3d/tests/<test_file>.mojo  # GPU (Metal)
```

**Key principle:** Test each pipeline stage independently before testing combined stages.
Debugging a full-step failure is nearly impossible because errors compound through:
FK → M → bias → contacts → Jacobians → constraint params → solver → integration.
Isolate each stage first.

---

## Test Matrix: MuJoCo vs CPU vs GPU

### Stage 1: Dynamics (no contacts)

| Component              | MuJoCo vs CPU | CPU vs GPU | Notes                                    |
|------------------------|:-------------:|:----------:|------------------------------------------|
| Forward Kinematics     | DONE          | DONE       | xpos, xquat, xipos — 5 configs, err~1e-16 |
| Mass Matrix (CRBA)     | DONE          | DONE       | Full NV x NV matrix — 4 configs, err~2e-6 |
| Bias Forces (RNE)      | DONE          | DONE       | Coriolis + gravity — 5 configs           |
| Unconstrained Accel    | DONE          | DONE       | qacc_smooth — 5 configs, err~1e-4       |
| Full Step (no contact) | DONE          | TODO       | qpos/qvel after N steps, free flight     |

### Stage 2: Contact pipeline (per-stage, before solver)

| Component              | MuJoCo vs CPU | CPU vs GPU | Notes                                    |
|------------------------|:-------------:|:----------:|------------------------------------------|
| Contact Detection      | DONE          | TODO       | 6 configs, pos/normal/dist match ~1e-5   |
| Constraint Jacobians   | DONE          | TODO       | 4 configs, normal/friction/limit ~1e-7   |
| Constraint Parameters  | DONE          | GPU STALE  | 4 configs, imp+aref+D+R all match. GPU uses old `imp/K` formulas |

### Stage 3: Solver output

| Component              | MuJoCo vs CPU | CPU vs GPU | Notes                                    |
|------------------------|:-------------:|:----------:|------------------------------------------|
| Solver Forces          | DONE          | TODO       | qfrc_constraint vs MuJoCo (4 configs)    |
| Full Step (contact)    | PASS          | TODO       | 2 configs pass with relaxed tolerances   |

---

## Existing Tests

### MuJoCo Comparison Tests (CPU)

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_fk_vs_mujoco.mojo` | FK: xpos, xquat, xipos per body | PASS | 5 (default, zero, nonzero, extreme, large rootx) | pos: 1e-6, quat: 1e-5 |
| `test_mass_matrix_vs_mujoco.mojo` | Full mass matrix (CRBA + armature) | PASS | 4 (default, zero, nonzero, extreme) | abs: 1e-4, rel: 1e-3 |
| `test_bias_forces_vs_mujoco.mojo` | Bias forces RNE (qfrc_bias) | PASS | 5 (zero vel, nonzero joints, nonzero vel, extreme) | abs: 1e-4, rel: 1e-3 |
| `test_full_step_vs_mujoco.mojo` | Full step without contacts | PASS | 3 (free fall, actions, multi-step) | qpos: 1e-6, qvel: 1e-4 |
| `test_qacc0_vs_mujoco.mojo` | Unconstrained accel qacc_smooth | PASS | 5 (gravity, actions, vel, combo, contact pose) | abs: 1e-4, rel: 1e-3 |
| `test_contacts_vs_mujoco.mojo` | Contact detection (pos, normal, dist) | PASS | 6 (high, default, low, very low, bent, tilted) | pos: 1e-3, dist: 1e-3, normal dot>0.99 |
| `test_jacobian_vs_mujoco.mojo` | Constraint Jacobians (J rows) | PASS | 4 (low static, low moving, very low, bent) | abs: 1e-4, rel: 1e-3 |
| `test_constraint_params_vs_mujoco.mojo` | Constraint params (imp, aref, D, R) | PASS | 4 (low static, low moving, very low, bent) | imp: 1e-3, aref: 1e-2, D/R: 1e-2 |
| `test_solver_forces_vs_mujoco.mojo` | Solver forces (qacc, qfrc_constraint, efc_force) | PASS (4/4) | 4 (low static, low moving, very low, bent) | qacc/qfrc: 5e-2. Cost matches to 12 digits. See analysis below. |
| `test_full_step_contact_vs_mujoco.mojo` | Full step with contacts | PASS | 2 (ground contact, with actions) | relaxed tolerances |

### CPU vs GPU Comparison Tests

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_fk_cpu_vs_gpu.mojo` | FK: xpos, xquat, xipos per body (float32) | PASS | 5 (default, zero, nonzero, extreme, large rootx) | pos: 1e-4, quat: 1e-4 |
| `test_mass_matrix_cpu_vs_gpu.mojo` | Full mass matrix CRBA (float32) | PASS | 4 (default, zero, nonzero, extreme) | abs: 1e-3, rel: 1e-2 |
| `test_bias_forces_cpu_vs_gpu.mojo` | Bias forces RNE (float32) | PASS | 5 (zero vel, nonzero joints, nonzero vel, extreme) | abs: 1e-2, rel: 1e-2 |

### Analytical / Standalone Tests

| Test File | What | CPU/GPU | Status |
|-----------|------|---------|--------|
| `test_forward_kinematics.mojo` | FK on generic 1-2 body models (identity, 90-deg, double pendulum) | CPU | PASS |
| `test_pendulum.mojo` | Pendulum period (~2.016s) + energy conservation (<5% drift) | CPU | PASS |
| `test_pendulum_gpu.mojo` | Same pendulum on GPU (relaxed: period 5%, energy 10%) | GPU | PASS |
| `test_implicit_integrator.mojo` | LU factorization, ImplicitIntegrator compilation, zero-vel qDeriv | CPU | PASS |
| `test_robot.mojo` | HalfCheetahRobot instantiation (compilation check) | CPU | PASS |

### Diagnostic / Stress Tests (not pass/fail, produce logs)

| Test File | What | Model |
|-----------|------|-------|
| `test_cheetah_torque_diag.mojo` | 5 constant-torque scenarios (free fall, small, max, all-down, all-up) | HalfCheetah (Newton) |
| `test_cheetah_policy_diag.mojo` | PPO policy actions over 3 episodes (requires checkpoint) | HalfCheetah (Newton) |
| `test_cheetah_diagnostics.mojo` | Free fall per-step logging (body Z, contacts, penetration) | HalfCheetah (PGS) |
| `test_solver_debug.mojo` | Single-step solver trace (FK, M, constraints, solver iterations, writeback) | HalfCheetah (Newton) |

---

## TODO: MuJoCo vs CPU — Constraint Pipeline (Stage 2)

These tests break the constraint solver comparison into isolated stages.
Each test uses `mj_step1()` (which runs FK + collision + constraint setup but NOT the solver)
to get MuJoCo's intermediate values for comparison.

### Test 1: Unconstrained Acceleration (`test_qacc0_vs_mujoco.mojo`) — DONE

**What:** Compare `qacc_smooth = M_arm^{-1} * f_net` (acceleration before any constraints).
This validates that M, bias forces, actuator forces, and passive forces combine correctly.

**MuJoCo reference:** `mj_data.qacc_smooth` (NV vector, available after `mj_forward`)
- MuJoCo 3.x convention: `qfrc_smooth = -qfrc_bias + qfrc_passive + qfrc_actuator`
- Our convention: `f_net = qfrc - bias + passive` (equivalent)

**Configs:** 5 (gravity only, with actions, nonzero vel, full combo, ground contact pose)

**Result:** ALL PASS. Max errors: abs~1e-4, rel~1e-5.
Validates: mass matrix, bias forces, actuator mapping, passive forces (damping, stiffness), LDL solve.

### Test 2: Contact Detection (`test_contacts_vs_mujoco.mojo`) — DONE

**What:** Compare detected contacts against MuJoCo for identical configurations.

**MuJoCo reference:** `mj_data.contact` (after `mj_step1`)
- `contact.pos` — contact position (3D)
- `contact.frame` — contact frame (normal + tangent directions, 9 floats)
- `contact.dist` — signed penetration distance
- `contact.geom` — geom pair indices
- `mj_data.ncon` — number of contacts

**Configs:**
1. Robot just above ground (rootz=-0.3) — feet should be in contact
2. Robot low (rootz=-0.45) — multiple body parts in contact
3. Robot high (rootz=0.5) — no contacts expected
4. Bent legs (joint angles set) — different contact geometry

**What to compare per contact:**
- Number of contacts (must match)
- Contact position (abs tol: 1e-3, our geom shapes may differ slightly)
- Contact normal direction (dot product > 0.99)
- Penetration distance sign and magnitude (abs tol: 1e-3)
- Geom/body pair matching (same bodies in contact)

**Why this matters:** Wrong contact positions → wrong Jacobians → wrong forces.
This is the #1 suspect for full-step divergence.

### Test 3: Constraint Jacobians (`test_jacobian_vs_mujoco.mojo`) — DONE

**What:** Compare constraint Jacobian rows against MuJoCo's `efc_J`.

**MuJoCo reference:** `mj_data.efc_J` (nefc x NV matrix, after `mj_step1`)
- Each row is one constraint (normal, friction_t1, friction_t2, limit, etc.)
- `mj_data.efc_type` — constraint type per row

**Configs:** Same contact configs as Test 2 (only configs with contacts are interesting).

**What to compare:**
- Normal Jacobian rows: compare J_n[r,:] (NV values) per contact
- Friction Jacobian rows: compare J_f[r,:] per friction row
- Limit Jacobian rows (if any active)
- Row ordering may differ — match by constraint type + contact index

**Tolerance:** abs: 1e-4 (Jacobians depend on FK which we already validated)

**Why this matters:** J is the bridge between joint space and constraint space.
Wrong J means forces are applied in wrong directions.

### Test 4: Constraint Parameters (`test_constraint_params_vs_mujoco.mojo`) — DONE

**What:** Compare constraint impedance, reference acceleration, D, and R against MuJoCo.

**MuJoCo reference (after `mj_step1`):**
- `efc_KBIP` — [K, B, imp, pos] per constraint (4 values)
- `efc_aref` — reference acceleration (what the constraint tries to achieve)
- `efc_D` — Delassus diagonal approximation
- `efc_R` — regularizer

**D/R now match MuJoCo (using invweight0 formulas):**
- Contact normals: `diagApprox = body_invweight0[2*body_a] + body_invweight0[2*body_b]`
- Joint limits: `diagApprox = dof_invweight0[dof_adr]` (= M_inv[dof,dof])
- `R = (1-imp)/imp * diagApprox`, `D = 1/R`
- `inv_K_imp = 1/(K + R)` where K = exact Delassus diagonal (J @ M_inv @ J^T)

**Configs:** 4 (low static, low moving, very low, bent legs with limit)

**What to compare:**
- imp (impedance): abs tol 1e-3 → PASS
- aref (reference acceleration): abs tol 1e-2 → PASS
- D (Delassus diagonal): rel tol 1e-2 → PASS
- R (regularizer): rel tol 1e-2 → PASS
- Degenerate friction tangent directions (K < 1e-6) are skipped

**Bugs found and fixed:**
- solimp defaults were [0.9, 0.95, 0.001] (MuJoCo solver defaults), should be [0.0, 0.8, 0.01] (MuJoCo geom defaults)
- K_spring formula: was 1/(tc*dr), should be 1/(tc*dmax) — dmax from solimp, not dr from solref
- B_damp formula: was 2*dr/tc, should be 2*dr/(tc*dmax)
- Limit solimp width: was 0.01 (geom default), should be 0.03 (joint default)
- Test must set solref/solimp from HalfCheetahParams (like the actual environment does)
- D/R for limits: MuJoCo uses `dof_invweight0` (per-DOF), not `body_invweight0` (per-body)
- Friction aref: was `B * imp * v_t`, should be `B * v_t` (no impedance for friction bias)

**Result:** ALL 4 PASS. imp, aref, D, R all match MuJoCo within tolerances.

### Test 5: Solver Forces (`test_solver_forces_vs_mujoco.mojo`) — DONE

**What:** Compare constraint forces after the solver runs.

**MuJoCo reference (after `mj_step`, i.e., full step):**
- `efc_force` — per-constraint force (nefc vector)
- `qfrc_constraint` — net constraint force in joint space (NV vector, = J^T * efc_force)

**Configs:** 4 (low static, low moving, very low, bent legs)

**What to compare (5 levels):**
1. `qfrc_constraint` (NV): J^T * lambda vs mj_data.qfrc_constraint — abs: 5e-2, rel: 2e-1
2. `qacc` (NV): constrained acceleration — abs: 5e-2, rel: 2e-1
3. Total normal force: sum of normal lambdas — abs: 5e-2, rel: 1e-1
4. Per-contact normal force (informational) — abs: 1e-1, rel: 3e-1
5. Per-row efc_force (informational, printed for analysis)

**Pipeline replicated:** Full euler_integrator pipeline including passive forces (damping,
stiffness, frictionloss), M_hat with armature+dt*damping, qfrc_smooth, then
PrimalNewtonSolver.solve() with CONE_TYPE=ELLIPTIC.

**Tolerance:** Relaxed — abs: 5e-2, rel: 2e-1 (solver convergence paths differ)

**Result:** 0/4 PASS (all fail). Two bugs fixed, remaining error is solver architecture.

**Bugs fixed:**
1. **Tangent frame** (FIXED): T1 now uses capsule axis as hint via MuJoCo's `mju_makeFrame`
   (Gram-Schmidt orthogonalization). T1 K values now 0.27-0.77 (was 1e-10 degenerate).
   Files: `contact_detection.mojo`, `constraint_builder.mojo`, `friction_solver.mojo`, `pgs_solver.mojo`
2. **Friction aref** (FIXED): Removed spurious `imp` factor from friction bias.
   Was `B_damp * imp * v_t`, now `B_damp * v_t` (MuJoCo doesn't apply impedance to friction bias).
   File: `constraint_builder.mojo`

**Current state (after fixes):**
- All constraint parameters match MuJoCo (4/4 param tests PASS)
- Normal forces consistently ~7% higher than MuJoCo
- Friction T1 forces directionally correct, magnitude off ~5-30% (tracks normal error)
- T2 forces correctly zero (HalfCheetah is 2D, Y direction is degenerate)

| Config | qfrc max_abs | qfrc max_rel | qacc max_abs | qacc max_rel | total_N err% |
|--------|-------------|-------------|-------------|-------------|-------------|
| Low static | 301.4 | 82% | 232.3 | 37% | 7.6% |
| Low moving | 297.6 | 48% | 373.0 | 318% | 6.5% |
| Very low | 454.4 | 42% | 399.4 | 368% | 6.5% |
| Bent legs | 46.1 | 717% | 115.2 | 32% | 3.2% |

**Remaining root cause: Solver architecture mismatch.**
MuJoCo uses a true Newton method on the full coupled normal+friction QCQP problem
(constraint type CE = condim>1 elliptic, each contact is one 3D constraint).
Our solver decouples: Newton for normals → coupled PGS for friction.
This means:
- Our normal forces don't account for friction's effect (7% excess)
- Friction errors compound on top of normal force errors
- The friction-to-normal ratio is similar (~14-33%), confirming correct coupling within PGS,
  but the base normal force is too high

**Next steps to reduce error:**
1. Implement MuJoCo-style coupled Newton solver (normal+friction as single 3D QCQP per contact)
2. Or increase coupled PGS iterations (diminishing returns past ~50)

---

## TODO: GPU — Sync with CPU D/R Changes

The CPU constraint builder now uses MuJoCo's `invweight0` formulas for D/R computation.
The GPU constraint builder still uses the old `inv_K_imp = imp/K` formula.
The following changes are needed to bring GPU in sync:

### What changed on CPU

1. **`constraint_data.mojo`**: Added `diagApprox` field to `ConstraintRow`
2. **`types.mojo`**: Added `dof_invweight0[NV]` to Model (diagonal of M^{-1})
3. **`mass_matrix.mojo`**: `compute_body_invweight0` now also computes `dof_invweight0`
4. **`constraint_builder.mojo`**: `_compute_aref` returns `(bias, inv_K_imp, imp)` where
   `inv_K_imp = 1/(K + R)` with `R = (1-imp)/imp * diagApprox`
5. **`primal_newton_solver.mojo`**: Simplified — uses `primal_D(inv_K_imp, K)` instead of
   manual body_invweight0 lookup

### GPU files that need changes

| File | What to change |
|------|---------------|
| `gpu/buffer_utils.mojo` | Add `dof_invweight0[NV]` to GPU model buffer. Add model buffer index constants. |
| `gpu/constants.mojo` | Add `MODEL_META_IDX_DOF_INVWEIGHT0` or similar offset for dof_invweight0 in model buffer |
| `constraints/constraint_builder_gpu.mojo` | **Contact normals** (line ~309): Change `inv_K_imp = imp/k` → `inv_K_imp = 1/(k + (1-imp)/imp * diagApprox)` where `diagApprox = body_invweight0[2*body_a] + body_invweight0[2*body_b]`. Read body_invweight0 from model buffer. **Joint limits** (line ~581): Change `lim_inv_K = imp/K` → `lim_inv_K = 1/(K + (1-imp)/imp * dof_invweight0[dof])`. Read dof_invweight0 from model buffer. |
| `solver/friction_solver.mojo` | **R_n recovery** (lines ~500, ~754): Currently `R_n = 1/inv_K_imp - K`. This formula still works with the new `inv_K_imp` values — no change needed. |
| `solver/pgs_solver.mojo` | Normal PGS update: `R = 1/inv_K_imp - K` still works — no change needed. |
| `solver/newton_solver.mojo` | Same as PGS — R recovery from inv_K_imp unchanged. |
| `solver/cg_solver.mojo` | Same as PGS — R recovery from inv_K_imp unchanged. |
| `envs/half_cheetah/half_cheetah_gc.mojo` | Add `dof_invweight0` values to model buffer in `init_step_workspace_gpu`. Copy from `model.dof_invweight0[d]` for d in 0..NV. |
| `envs/hopper/hopper_gc.mojo` | Same as half_cheetah — add `dof_invweight0` to model buffer. |

### Key insight: solvers don't need changes

The GPU solvers (PGS, Newton, CG, friction) all recover R from `inv_K_imp` via:
```
R = 1/inv_K_imp - K
```
This formula works regardless of how `inv_K_imp` was computed. So only the **constraint builder GPU**
(which writes `inv_K_imp`) needs to change. The solvers that consume it are unaffected.

### Model buffer layout change

Currently `body_invweight0` is stored in the model buffer. Need to also store `dof_invweight0[NV]`.
Options:
1. Append after existing model buffer (simplest, just extend `MODEL_SIZE`)
2. Add to metadata section (if NV is small and fixed per model)

Recommended: Append NV floats after the current model buffer end. Add `model_dof_invweight0_offset[NBODY, NJOINT]()` function to constants.mojo.

---

## TODO: MuJoCo vs CPU — Full Step (Stage 3)

### Full Step with Contacts — PASS (relaxed tolerances)

`test_full_step_contact_vs_mujoco.mojo` now passes with 2 configs.
Can be tightened as solver accuracy improves.

**Approach for full-step debugging (if needed):**
1. Single step first (num_steps=1)
2. Compare qacc (acceleration output of solver)
3. Compare qvel_new = qvel + qacc * dt
4. Compare qpos_new (position integration)
5. Only then try multi-step (10, 100 steps)

---

## TODO: CPU vs GPU

For each component that passes MuJoCo vs CPU, add a CPU vs GPU comparison test.
**Only after the CPU is validated against MuJoCo.**

### Already Done
- Forward Kinematics (CPU vs GPU) — DONE
- Mass Matrix (CPU vs GPU) — DONE
- Bias Forces (CPU vs GPU) — DONE

### After GPU D/R sync
- Contact Detection (CPU vs GPU)
- Constraint Jacobians (CPU vs GPU)
- Constraint Parameters (CPU vs GPU) — will validate GPU D/R changes
- Solver Forces (CPU vs GPU)
- Full Step (CPU vs GPU)

---

## Implementation Priority

```
 DONE                          TODO (in order)
┌──────────────────┐    ┌─────────────────────────────────┐
│ 1. FK            │    │ 4. Unconstrained Accel (qacc0)  │ DONE
│ 2. Mass Matrix   │    │ 5. Contact Detection            │ DONE
│ 3. Bias Forces   │    │ 6. Constraint Jacobians         │ DONE
│                  │    │ 7. Constraint Parameters (D,R)  │ DONE (CPU). GPU sync needed.
│                  │    │ 8. Solver Forces                │ DONE (test written)
│                  │    │ 9. Full Step (contact)          │ PASS (relaxed tol)
└──────────────────┘    └─────────────────────────────────┘
```

Each test MUST pass before moving to the next. If test N fails, fix it before
writing test N+1. This prevents debugging compound errors.

---

## MuJoCo Python API Reference

Key fields available after `mj_step1()` (FK + collision + constraint setup, NO solver):
```python
mj_data.qpos          # Positions (NQ)
mj_data.qvel          # Velocities (NV)
mj_data.qacc0         # Unconstrained acceleration (NV) — M^{-1} * qfrc_bias + M^{-1} * qfrc_applied
mj_data.qfrc_bias     # Bias forces: Coriolis + gravity (NV)
mj_data.qfrc_applied  # Applied forces: ctrl → actuator → joint (NV)
mj_data.ncon          # Number of contacts
mj_data.contact       # Contact array (pos, frame, dist, geom, ...)
mj_data.nefc          # Number of constraint rows
mj_data.efc_J         # Constraint Jacobian (nefc x NV)
mj_data.efc_D         # Delassus diagonal approximation (nefc)
mj_data.efc_R         # Regularizer (nefc)
mj_data.efc_aref      # Reference acceleration (nefc)
mj_data.efc_b         # Bias = J*qacc0 - aref (nefc)
mj_data.efc_type      # Constraint type per row (nefc)
mj_data.efc_KBIP      # [K, B, imp, pos] per row (nefc x 4)
```

Key fields available after `mj_step()` (full step including solver):
```python
mj_data.qacc              # Constrained acceleration (NV)
mj_data.qfrc_constraint   # Net constraint force in joint space (NV)
mj_data.efc_force         # Per-constraint force (nefc)
```

---

## Test Naming Convention

| Pattern | Meaning |
|---------|---------|
| `test_<component>_vs_mujoco.mojo` | MuJoCo reference comparison (CPU) |
| `test_<component>_cpu_vs_gpu.mojo` | CPU vs GPU comparison |
| `test_<component>.mojo` | Standalone / analytical test (CPU) |
| `test_<component>_gpu.mojo` | Standalone GPU test |
| `test_<component>_diag.mojo` | Diagnostic (logging, not pass/fail) |
