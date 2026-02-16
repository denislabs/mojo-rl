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
| Constraint Parameters  | DONE          | TODO       | 4 configs, imp+aref match ~1e-3. D/R skipped (different formulas) |

### Stage 3: Solver output

| Component              | MuJoCo vs CPU | CPU vs GPU | Notes                                    |
|------------------------|:-------------:|:----------:|------------------------------------------|
| Solver Forces          | TODO          | TODO       | qfrc_constraint vs MuJoCo                |
| Full Step (contact)    | TODO          | TODO       | qpos/qvel after N steps with contacts    |

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
| `test_constraint_params_vs_mujoco.mojo` | Constraint params (imp, aref) | PASS | 4 (low static, low moving, very low, bent) | imp: 1e-3, aref: 1e-2 |
| `test_full_step_contact_vs_mujoco.mojo` | Full step with contacts | FAILING | 2 (ground contact, with actions) | relaxed tolerances |

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

**What:** Compare constraint impedance and reference acceleration against MuJoCo.

**MuJoCo reference (after `mj_step1`):**
- `efc_KBIP` — [K, B, imp, pos] per constraint (4 values)
- `efc_aref` — reference acceleration (what the constraint tries to achieve)
- `efc_D` — Delassus diagonal approximation (NOT compared — see below)
- `efc_R` — regularizer (NOT compared — see below)

**Why D/R are skipped:**
- Our engine: D = J @ M_inv @ J^T (exact Delassus diagonal)
- MuJoCo: D = imp/((1-imp) * invweight0) (body-level approximation)
- These are fundamentally different approaches, both valid for their respective solvers.

**Configs:** 4 (low static, low moving, very low, bent legs with limit)

**What to compare:**
- imp (impedance): abs tol 1e-3 → PASS (all contacts imp=0.8, limit imp=0.4)
- aref (reference acceleration): abs tol 1e-2, rel tol 1e-3 → PASS
- Degenerate friction tangent directions (K < 1e-6) are skipped

**Bugs found and fixed:**
- solimp defaults were [0.9, 0.95, 0.001] (MuJoCo solver defaults), should be [0.0, 0.8, 0.01] (MuJoCo geom defaults)
- K_spring formula: was 1/(tc²*dr²), should be 1/(tc²*dmax²) — dmax from solimp, not dr from solref
- B_damp formula: was 2*dr/tc, should be 2*dr/(tc*dmax)
- Limit solimp width: was 0.01 (geom default), should be 0.03 (joint default)
- Test must set solref/solimp from HalfCheetahParams (like the actual environment does)

**Result:** ALL 4 PASS. imp and aref match MuJoCo within tolerances.

### Test 5: Solver Forces (`test_solver_forces_vs_mujoco.mojo`) — TODO

**What:** Compare constraint forces after the solver runs.

**MuJoCo reference (after `mj_step`, i.e., full step):**
- `efc_force` — per-constraint force (nefc vector)
- `qfrc_constraint` — net constraint force in joint space (NV vector, = J^T * efc_force)

**Configs:** Same contact configs.

**What to compare:**
- `qfrc_constraint` (NV vector): this is the most robust comparison since it's
  solver-independent (different solvers may distribute forces differently across
  contacts but the NET joint-space force should match)
- Per-contact normal force magnitude (less strict, solver path-dependent)
- Total normal force (sum of all contact normal forces)

**Tolerance:** Relaxed — abs: 5e-2, rel: 2e-1 (solver convergence paths differ)

**Why this matters:** If Tests 1-4 all pass but this fails, the bug is in the solver itself.
If Tests 1-4 have errors, fix those first before looking at solver output.

---

## TODO: MuJoCo vs CPU — Full Step (Stage 3)

### Full Step with Contacts — BLOCKED on Stage 2

The existing `test_full_step_contact_vs_mujoco.mojo` is currently FAILING.
**Do not try to fix full-step tests until all Stage 2 tests pass.**

Once Stage 2 tests pass, re-run the full-step test. If it still fails, the error
is in integration (velocity/position update), not in the constraint pipeline.

**Approach for full-step debugging:**
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

### After Stage 2 passes
- Contact Detection (CPU vs GPU)
- Constraint Jacobians (CPU vs GPU)
- Constraint Parameters (CPU vs GPU)
- Solver Forces (CPU vs GPU)
- Full Step (CPU vs GPU)

---

## Implementation Priority

```
 DONE                          TODO (in order)
┌──────────────────┐    ┌─────────────────────────────────┐
│ 1. FK            │    │ 4. Unconstrained Accel (qacc0)  │ ✓ DONE
│ 2. Mass Matrix   │    │ 5. Contact Detection            │ ✓ DONE
│ 3. Bias Forces   │    │ 6. Constraint Jacobians         │ ✓ DONE
│                  │    │ 7. Constraint Parameters (D,R)  │ ← depends on Jacobians
│                  │    │ 8. Solver Forces                │ ← depends on all above
│                  │    │ 9. Full Step (contact)          │ ← integration test
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
