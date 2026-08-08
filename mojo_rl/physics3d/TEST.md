# Physics3D Test Plan

## Test Methodology

We validate our physics engine at three levels:
1. **MuJoCo vs CPU** — Compare our CPU engine output against MuJoCo (Python interop) for identical inputs
2. **CPU vs GPU** — Compare our GPU engine output against our CPU engine (ensures GPU kernels match)
3. **Standalone** — Analytical validation (e.g. pendulum period) and diagnostic/stress tests

MuJoCo comparison tests use both HalfCheetah (pyramidal cone) and Hopper (elliptic cone). Tests run with:
```bash
cd mojo-rl && pixi run mojo run physics3d/tests/<test_file>.mojo        # CPU
cd mojo-rl && pixi run -e apple mojo run physics3d/tests/<test_file>.mojo  # GPU (Metal)
```

**Models covered:**
- **HalfCheetah** — pyramidal cone (`ConeType.PYRAMIDAL`), `NQ=10, NV=10, NBODY=9, NJOINT=10, NGEOM=9`
- **Hopper** — elliptic cone (`ConeType.ELLIPTIC`), `NQ=6, NV=6, NBODY=5, NJOINT=6, NGEOM=5`
- **Ant** — free joint (quaternion DOFs), 4-leg tree, 3D locomotion, `NQ=15, NV=14, NBODY=14, NJOINT=9, NGEOM=15`
- **Swimmer** — no contacts (contype=0), 3-body planar chain, RK4 integrator, `NQ=5, NV=5, NBODY=4, NJOINT=5`

**Key principle:** Test each pipeline stage independently before testing combined stages.
Debugging a full-step failure is nearly impossible because errors compound through:
FK → M → bias → contacts → Jacobians → constraint params → solver → integration.
Isolate each stage first.

---

## Test Matrix: MuJoCo vs CPU vs GPU

### Integrator/Solver coverage

| Integrator     | Solver             | Cone      | MuJoCo vs CPU | CPU vs GPU | Used by          |
|----------------|--------------------|-----------|:-------------:|:----------:|------------------|
| Euler          | Newton             | Elliptic  | DONE          | DONE       | (tests only)     |
| Euler          | CG                 | Elliptic  | DONE          | —          | (tests only)     |
| Euler          | PGS (dual)         | Elliptic  | DONE          | —          | (tests only)     |
| Euler          | Newton             | Pyramidal | DONE          | DONE       | (tests only)     |
| ImplicitFast   | Newton             | Elliptic  | DONE          | DONE       | HalfCheetah      |
| ImplicitFast   | PGS                | Elliptic  | N/A           | DONE       | Hopper (Default) |
| Euler          | Newton             | Elliptic  | PENDING       | PENDING    | Hopper tests     |
| Implicit (full)| PGS                | Elliptic  | DONE          | DONE       | (tests only)     |
| RK4            | Newton             | Elliptic  | DONE          | DONE       | (tests only)     |

MuJoCo comparison uses `opt.integrator=0` (Euler) for Euler tests, `opt.integrator=1` (RK4) for RK4 tests,
`opt.integrator=3` (ImplicitFast) for ImplicitFast tests, and `opt.integrator=2` (Implicit) for full Implicit tests.

**CRITICAL: MuJoCo 3.3.6 integrator enum values:**
- `0` = mjINT_EULER
- `1` = mjINT_RK4 (NOT Implicit!)
- `2` = mjINT_IMPLICIT
- `3` = mjINT_IMPLICITFAST

ImplicitFast+PGS has no MuJoCo comparison because MuJoCo only allows Newton solver with implicitfast.

### Stage 1: Dynamics (no contacts) — Euler + Newton + Elliptic

| Component              | MuJoCo vs CPU | CPU vs GPU | Notes                                    |
|------------------------|:-------------:|:----------:|------------------------------------------|
| Forward Kinematics     | DONE          | DONE       | xpos, xquat, xipos — 5 configs, err~1e-16 |
| Mass Matrix (CRBA)     | DONE          | DONE       | Full NV x NV matrix — 4 configs, err~2e-6 |
| Bias Forces (RNE)      | DONE          | DONE       | Coriolis + gravity — 5 configs           |
| Unconstrained Accel    | DONE          | DONE       | qacc_smooth — 5 configs, err~1e-4       |
| Full Step (no contact) | DONE          | DONE       | qpos/qvel after N steps, free flight. 6 configs, err~2.6e-4 |

### Stage 2: Contact pipeline (per-stage, before solver) — shared across integrators

| Component              | MuJoCo vs CPU | CPU vs GPU | Notes                                    |
|------------------------|:-------------:|:----------:|------------------------------------------|
| Contact Detection      | DONE          | DONE       | 6 configs, pos/normal/dist match ~1e-5   |
| Constraint Jacobians   | DONE          | DONE       | 4 configs, J_n rows match ~6e-8          |
| Constraint Parameters  | DONE          | DONE       | 4 configs, K/bias/inv_K_imp match. GPU now uses diagApprox from body_invweight0 |

### Stage 3: Solver output — Euler + Newton + Elliptic

| Component              | MuJoCo vs CPU | CPU vs GPU | Notes                                    |
|------------------------|:-------------:|:----------:|------------------------------------------|
| Solver Forces          | DONE          | DONE       | Validated via full step (1-step qacc match) |
| Full Step (contact)    | DONE          | DONE       | 6/6 configs pass, all errors ~1e-6 (float32 rounding) |

### Stage 4: ImplicitFast integrator

| Component                        | Integrator   | Solver       | MuJoCo vs CPU | CPU vs GPU | Notes |
|----------------------------------|--------------|--------------|:-------------:|:----------:|-------|
| Full Step no contact             | ImplicitFast | Newton       | DONE          | DONE       | MuJoCo: 5 configs, err~5.6e-6. GPU: 3 configs, exact. |
| Full Step with contacts          | ImplicitFast | Newton       | DONE          | DONE       | MuJoCo: 4 configs, err~7.2e-6. GPU: 4 configs (deep pen skipped). |
| Full Step no contact             | ImplicitFast | PGS          | N/A           | DONE       | 3 configs, exact match (err=0) |
| Full Step with contacts          | ImplicitFast | PGS          | N/A           | DONE       | 5 configs, max qvel err 0.33 (deep pen) |

**ImplicitFast MuJoCo comparison notes:**
- No contacts: match within ~5.6e-6 (machine precision)
- With contacts: match within ~7.2e-6 (machine precision)
- Comparison now uses MuJoCo ImplicitFast (`opt.integrator=3`) directly
- **Bug found and fixed**: Constraint solver was using `M_inv` from `M + arm + dt*D`, but MuJoCo's
  constraint solver always uses `M + arm` only. Fix: use `M + arm` for constraint solver, then
  post-constraint re-solve with `M_hat = M + arm + dt*D` (matching MuJoCo's `mj_implicitSkip`).
  This reduced contact errors from ~15% to ~7e-6.
- **Earlier bug fixed**: `constraints.M_hat` was not filled by ImplicitFastIntegrator. The primal
  Newton solver used zero M_hat, causing the Hessian H = 0 + J^T*D*J (missing mass matrix term).

### Stage 5: Implicit (full) integrator

| Component                        | Integrator       | Solver       | MuJoCo vs CPU | CPU vs GPU | Notes |
|----------------------------------|------------------|--------------|:-------------:|:----------:|-------|
| Full Step no contact             | Implicit (full)  | PGS          | DONE          | DONE       | MuJoCo: 6 configs, err~1e-15. GPU: 8 configs (float32). |
| qDeriv finite diff               | Implicit (full)  | —            | DONE          | —          | RNE velocity derivative matches finite diff. |
| qDeriv vs MuJoCo                 | Implicit (full)  | —            | DONE          | —          | qDeriv matches MuJoCo's d->qDeriv at nonzero velocity. |

**Implicit (full) integrator notes:**
- Uses `M_hat = M + arm + dt*(D - qDeriv)` where `qDeriv = d(qfrc_bias)/d(qvel)` (RNE velocity derivative)
- Compared against MuJoCo `opt.integrator=2` (mjINT_IMPLICIT)
- **Critical enum bug found**: Previous code used `opt.integrator=1` thinking it was Implicit,
  but `1` = mjINT_RK4. Fixed to `opt.integrator=2`. This was the root cause of "Implicit differs
  from Euler at zero velocity" mystery.
- ALL MuJoCo integrators use symplectic Euler for position: `qpos += dt * qvel_new` (NOT midpoint)
- At zero velocity, qDeriv = 0 (Coriolis is quadratic in v), so Implicit == ImplicitFast == Euler
- **GPU subtree-COM convention**: `compute_rne_vel_derivative_gpu` rewritten to use subtree-COM
  convention matching the CPU version. Eliminates frame transfers in spatial propagation.
  Validated by CPU vs GPU test (8/8 pass, nonzero vel err ~2.8e-4 in float32).
- **cdof convention bug fixed (2026-04-03)**: `compute_rne_vel_derivative` was called with
  subtree_com-based cdof (from `compute_cdof(model, data, cdof, stcom_tmp)`), but it expects
  xipos-based cdof (from `compute_cdof(model, data, cdof)`). The internal body-origin conversion
  produces different results with subtree_com vs xipos reference points, causing qDeriv[2,2]
  to be ~5.5 instead of ~2.2 (HalfCheetah rooty DOF). Fix: compute a separate cdof without
  subtree_com for the RNE derivative. MuJoCo comparison now matches at ~1e-15 (was ~0.8 qvel error).
  GPU version still uses workspace cdof (subtree_com-based) — TODO: fix for GPU Implicit.

---

## Existing Tests

### MuJoCo Comparison Tests (CPU)

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_fk_vs_mujoco.mojo` | FK: xpos, xquat, xipos per body | PASS | 5 (default, zero, nonzero, extreme, large rootx) | pos: 1e-6, quat: 1e-5 |
| `test_mass_matrix_vs_mujoco.mojo` | Full mass matrix (CRBA + armature) | PASS | 4 (default, zero, nonzero, extreme) | abs: 1e-4, rel: 1e-3 |
| `test_bias_forces_vs_mujoco.mojo` | Bias forces RNE (qfrc_bias) | PASS | 5 (zero vel, nonzero joints, nonzero vel, extreme) | abs: 1e-4, rel: 1e-3 |
| `test_full_step_vs_mujoco.mojo` | Full step without contacts | PASS | 6 (free fall, standing, actions, moving, 10-step free fall, 10-step actions) | qpos: 1e-3, qvel: 1e-2 (actual err ~5e-6) |
| `test_qacc0_vs_mujoco.mojo` | Unconstrained accel qacc_smooth | PASS | 5 (gravity, actions, vel, combo, contact pose) | abs: 1e-4, rel: 1e-3 |
| `test_contacts_vs_mujoco.mojo` | Contact detection (pos, normal, dist) | PASS | 6 (high, default, low, very low, bent, tilted) | pos: 1e-3, dist: 1e-3, normal dot>0.99 |
| `test_jacobian_vs_mujoco.mojo` | Constraint Jacobians (J rows) | PASS | 4 (low static, low moving, very low, bent) | abs: 1e-4, rel: 1e-3 |
| `test_constraint_params_vs_mujoco.mojo` | Constraint params (imp, aref, D, R) | PASS | 4 (low static, low moving, very low, bent) | imp: 1e-3, aref: 1e-2, D/R: 1e-2 |
| `test_solver_forces_vs_mujoco.mojo` | Newton solver forces (qacc, qfrc_constraint, efc_force) | PASS (4/4) | 4 (low static, low moving, very low, bent) | qacc/qfrc: 5e-2. Cost matches to 12 digits. See analysis below. |
| `test_cg_vs_mujoco.mojo` | CG solver forces vs MuJoCo CG | PASS (4/4) | 4 (low static, low moving, very low, bent) | qacc/qfrc: 5e-2. Cost matches to ~12 digits. |
| `test_pgs_vs_mujoco.mojo` | PGS (dual) solver forces vs MuJoCo PGS | PASS (4/4) | 4 (low static, low moving, very low, bent) | qacc/qfrc: 1e-1. Forces match within ~3.6e-3. |
| `test_full_step_contact_vs_mujoco.mojo` | Full step with contacts | PASS | 2 (ground contact, with actions) | qpos: 2e-2, qvel: 2e-1 (actual err ~1e-5) |
| `test_implicit_fast_step_vs_mujoco.mojo` | ImplicitFast full step no contacts (ref: MuJoCo ImplicitFast) | PASS | 5 (free fall, actions, moving, 10-step) | qpos: 1e-6, qvel: 1e-4 (actual err ~5.6e-6) |
| `test_implicit_fast_step_contact_vs_mujoco.mojo` | ImplicitFast full step with contacts (ref: MuJoCo ImplicitFast) | PASS | 4 (default, actions, mild, deep) | qpos: 1e-6, qvel: 1e-4 (actual err ~7.2e-6) |
| `test_implicit_step_vs_mujoco.mojo` | Implicit (full) step no contacts (ref: MuJoCo Implicit) | PASS | 6 (free fall, zero vel+actions, moving+actions, fast spinning, 10-step variants) | qpos: 1e-4, qvel: 1e-2 (actual err ~5.9e-6) |
| `test_qderiv_finite_diff.mojo` | RNE velocity derivative vs finite differences | PASS | HalfCheetah nonzero vel | max err ~1e-6 |
| `test_qderiv_vs_mujoco.mojo` | qDeriv vs MuJoCo reference | PASS | HalfCheetah nonzero vel | matches MuJoCo |
| `test_pyramidal_vs_mujoco.mojo` | Pyramidal cone Newton solver forces (qacc, qfrc_constraint) | PASS (4/4) | 4 (low static, low moving, very low, bent) | qacc/qfrc: 5e-2 (actual ~1e-3). D/R match exactly. |
| `test_rk4_step_vs_mujoco.mojo` | RK4 full step no contact + contact (ref: MuJoCo RK4) | PASS (6/6) | 6 (free fall, actions, moving, fast spin, 10-step, ground contact) | qpos: 1e-3, qvel: 1e-2 (actual err ~1e-6 no-contact, ~3.5e-6 contact) |

#### Ant Tests (Free joint / 3D locomotion)

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_ant_fk_vs_mujoco.mojo` | FK: xpos, xquat, xipos per body (free joint + hinges) | PASS (5/5) | 5 (default, zero, nonzero joints, extreme, raised) | pos: 1e-6, quat: 1e-5 (actual err ~1e-16) |
| `test_ant_full_step_vs_mujoco.mojo` | Full step no contact + contact (RK4 + Newton) | PASS (5/6) | 5 pass: free fall, free fall+actions, default+no action, moving, 10-step. 1 FAIL: default+large actions+contacts | qpos: 1e-3, qvel: 1e-2 |

**Ant FK notes:**
- Ant has a **free joint** (JNT_FREE): 7 qpos DOFs [tx,ty,tz,qw,qx,qy,qz], 6 qvel DOFs [vx,vy,vz,wx,wy,wz]
- First test validating FK for a full 3D floating-body model
- Tree topology: torso → 4 legs (each: hip hinge → ankle hinge), no parent body for torso

**Ant full step notes:**
- Uses RK4 integrator + Newton solver (elliptic cone), no contacts for most tests
- 5/6 tests pass. One remaining failure: `test_default_pose_with_actions`
  - **Root cause**: 4 contacts (legs touching ground) + large motor forces (0.8 × gear=150 = 120 N·m) → contact solver precision differences cause 3-37% errors in free-joint velocity DOFs
  - This is a contact solver accuracy issue under extreme forces, not a physics modeling bug
- **Bug found and fixed (geom density)**: Ant XML uses `density="5.0"` in `<default><geom>`.
  Our `DefaultsData` was missing a `geom_density` field → always used hardcoded `MJ_DEFAULT_DENSITY=1000.0`.
  Result: all Ant body masses 200× too large → M[hip,hip] ≈ 6.2 (ours) vs ≈ 1.026 (MuJoCo) → motor-driven qacc ≈ 5× too small.
  Fix: added `geom_density` to `DefaultsData` and `density` to `GeomData` in `flat_model.mojo`,
  parsed `density` from XML defaults and per-geom in `full_parser.mojo`, computed `gd.mass = density * volume`
  inline during parsing for all geom types (sphere/capsule/box/cylinder).
  Files: `physics3d/parser/flat_model.mojo`, `physics3d/parser/full_parser.mojo`

#### Swimmer Tests (No contacts / pure dynamics)

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_swimmer_fk_vs_mujoco.mojo` | FK: xpos, xquat, xipos per body (slide+hinge chain) | PASS (6/6) | 6 (default, zero, bent joints, moving, extreme, large displacement) | pos: 1e-6, quat: 1e-5 (actual err ~1e-16) |
| `test_swimmer_full_step_vs_mujoco.mojo` | Full step no contact (RK4 + Newton, fluid disabled) | PASS (5/5) | 5 (zero state, bent joints, motor actions, moving+actions, 10-step) | qpos: 1e-3, qvel: 1e-2 |

**Swimmer FK notes:**
- Swimmer is entirely contact-free (contype=0 on all geoms) — pure rigid-body dynamics test
- Body chain: world_body → torso (slide_x, slide_y, free_body_rot hinges) → mid_body → back_body

**Swimmer full step notes:**
- First contact-free full step test (all existing full step tests have contacts or are free fall)
- MuJoCo fluid dynamics disabled (`viscosity=0, density=0`) since our engine has no fluid drag/buoyancy
- All 5 configs pass at RK4 accuracy; errors ~1e-6 to 1e-4
- Validates pure rigid-body dynamics: FK, mass matrix, bias forces, RK4 stages, actuator gear mapping

#### InvertedDoublePendulum Tests (Slide + hinge chain)

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_inverted_double_pendulum_fk_vs_mujoco.mojo` | FK: xpos, xquat, xipos per body (slide + 2 hinges) | PASS (5/5) | 5 (default, displaced cart, first hinge, both hinges, large tilt) | pos: 1e-6, quat: 1e-5 (actual err ~1e-16) |

**IDP FK notes:**
- Simplest model tested: 3 DOFs, 3 bodies (cart, pole, pole2), contype=0 (no contacts)
- Validates slide joint translation and chained hinge rotations in a lightweight model
- Errors at machine precision (~1e-16) for all 5 configurations

#### Walker2D Tests (Biped / two-leg)

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_walker2d_fk_vs_mujoco.mojo` | FK: xpos, xquat, xipos per body (two legs, off-center joints) | PASS (5/5) | 5 (default standing, large rootx, bent right leg, symmetric gait, extreme joints) | pos: 1e-6, quat: 1e-5 (actual err ~1e-16) |

**Walker2D FK notes:**
- Biped topology: torso + 2 × (thigh → leg → foot), complements Hopper's single-leg coverage
- `leg_joint` has `pos="0 0 0.25"` and `foot_joint` has `pos="-0.2 0 0.1"` — same off-center jnt_pos structure that exposed the cdof anchor bug in Hopper; passes cleanly
- `rootz` has `ref="1.25"` → qpos0[rootz]=1.25; default standing pose uses qpos[rootz]=1.25
- Errors at machine precision (~1e-16) for all 5 configurations

#### Humanoid Tests (Free joint + tendon constraints)

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_humanoid_fk_vs_mujoco.mojo` | FK: xpos, xquat, xipos per body (free joint + 17 hinges + body quats) | PASS (5/5) | 5 (default standing, bent knees, arms extended, rotated torso, full body pose) | pos: 5e-6, quat: 1e-5 (actual arm err ~1e-16, leg chain err ~1-3e-6) |

**Humanoid FK notes:**
- Most complex model: 24 NQ, 23 NV, 14 bodies, free joint + 17 hinges, 2 tendons
- First test with body-level `quat=` attributes (`lwaist` and `pelvis` have `quat="1.000 0 -0.002 0"`)
- **Bug found and fixed (`_parse_quat`)**: `_parse_quat` in `xml_parser.mojo` was reading quaternions as `"x y z w"` but MuJoCo XML stores all quaternion attributes as `"w x y z"`. This caused `lwaist`/`pelvis` body quats to be parsed as a 90° rotation instead of a ~0.1° tilt → all downstream bodies (pelvis, thighs, shins, feet) had 0.33m position errors. Fix: reversed the parse order so `parts[0]` is read as `qw`, returning `(qx, qy, qz, qw)`.
  File: `physics3d/parser/xml_parser.mojo`
  Impact: any XML model with explicit `quat=` on body, geom, or joint tags had wrong orientations. HalfCheetah/Hopper/Ant/Swimmer/Walker2D all use `fromto`/`axisangle`/`euler` instead, so were unaffected.
- Tolerance is 5e-6 (vs usual 1e-6) because `lwaist`'s tiny body quat rotation introduces cos/sin rounding that accumulates ~1-3 µm down the pelvis→thigh→shin→foot chain. Arms (no body quat in parent chain) still hit ~1e-16.

#### Hopper Tests (Elliptic cone)

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_hopper_fk_vs_mujoco.mojo` | FK: xpos, xquat, xipos per body | PASS (5/5) | 5 (default, zero, nonzero, extreme, raised) | pos: 1e-6, quat: 1e-5 (actual err ~1e-16) |
| `test_hopper_full_step_vs_mujoco.mojo` | Full step without contacts (Euler+Newton) | PENDING | 6 (free fall, standing, actions, moving, 10-step) | qpos: 1e-3, qvel: 1e-2 |
| `test_hopper_full_step_contact_vs_mujoco.mojo` | Full step with contacts (Euler+Newton, elliptic) | PENDING | 4 (ground contact, actions, deep pen, moving) | qpos: 2e-2, qvel: 2e-1 |
| `test_hopper_solver_forces_vs_mujoco.mojo` | Newton solver forces (qacc, qfrc_constraint) | PENDING | 4 (low static, low moving, very low, bent) | qacc/qfrc: 5e-2 |

**Hopper FK notes:**
- Hopper uses `ELLIPTIC` cone (condim=3) — complements HalfCheetah (pyramidal) coverage
- `qpos0` fix required: Hopper rootz joint has `ref="1.25"` in XML, so FK displacement = `qpos - qpos0`
- FK fixed by subtracting `model.qpos0[qpos_adr]` in slide/hinge joints (CPU + GPU)
- Hinge joint axes fixed: actuated joints use `axis="0 -1 0"` (axis_y=-1.0)
- All 5 FK configs pass with error ~1e-16 (machine precision)

**Hopper full step status (PENDING — friction forces diverge from MuJoCo):**
- Contact counts match (FK is correct)
- qvel[1] (vertical/rootz) matches perfectly — gravity is correct
- Errors in horizontal/rotational DOFs → friction forces differ from MuJoCo
- Root cause under investigation (condim=3, friction=2.0 verified correct)

### CPU vs GPU Comparison Tests

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_fk_cpu_vs_gpu.mojo` | FK: xpos, xquat, xipos per body (float32) | PASS | 5 (default, zero, nonzero, extreme, large rootx) | pos: 1e-4, quat: 1e-4 |
| `test_mass_matrix_cpu_vs_gpu.mojo` | Full mass matrix CRBA (float32) | PASS | 4 (default, zero, nonzero, extreme) | abs: 1e-3, rel: 1e-2 |
| `test_bias_forces_cpu_vs_gpu.mojo` | Bias forces RNE (float32) | PASS | 5 (zero vel, nonzero joints, nonzero vel, extreme) | abs: 1e-2, rel: 1e-2 |
| `test_full_step_cpu_vs_gpu.mojo` | Full step no contacts (float32) | PASS | 6 (free fall, actions, moving, 10-step free fall, 10-step moving, extreme vel) | qpos: 1e-3, qvel: 1e-2 (actual err ~2.6e-4) |
| `test_constraint_params_cpu_vs_gpu.mojo` | Constraint params K, bias, inv_K_imp (float32) | PASS | 4 (low static, low moving, very low, bent) | abs: 1e-2 (actual err ~1e-6) |
| `test_contacts_cpu_vs_gpu.mojo` | Contact detection pos, normal, dist (float32) | PASS | 6 (high, default, low, very low, bent, tilted) | pos: 1e-3, dist: 1e-3, normal_dot>0.999 |
| `test_jacobian_cpu_vs_gpu.mojo` | Normal Jacobian J_n rows (float32) | PASS | 4 (low static, low moving, very low, bent) | abs: 1e-3 (actual err ~6e-8) |
| `test_full_step_contact_cpu_vs_gpu.mojo` | Full step with contacts (float32) | PASS (6/6) | 6 (static, actions, deep pen, moving, 5-step) | qpos: 5e-2, qvel: 1.0 (actual err ~1e-6, near float32 rounding) |
| `test_implicit_fast_newton_cpu_vs_gpu.mojo` | ImplicitFast+Newton full step (float32) | 7/7 PASS | 7 (3 no-contact + 4 contact) | All pass. 5-step contact err ~1e-5. |
| `test_implicit_fast_pgs_cpu_vs_gpu.mojo` | ImplicitFast+PGS full step (float32) | 8/8 PASS | 8 (3 no-contact + 5 contact) | All pass. 5-step contact err ~2.3e-2. |
| `test_implicit_cpu_vs_gpu.mojo` | Implicit(full)+PGS full step (float32) | PASS | 8 (zero vel, nonzero vel, actions, contact) | qpos: 5e-2, qvel: 1.0 (nonzero vel err ~2.8e-4, contact ~0.01) |
| `test_pyramidal_cpu_vs_gpu.mojo` | Pyramidal cone Euler+Newton CPU vs GPU (float32) | PASS (6/6) | 6 (static, actions, deep pen, moving, 5-step) | qpos: 5e-2, qvel: 1.0 (actual err ~1e-6, near float32 rounding) |
| `test_rk4_cpu_vs_gpu.mojo` | RK4+Newton full step CPU vs GPU (float32) | PASS | 6 (free fall, actions, moving, fast spin, 10-step, ground contact) | qpos: 3e-2, qvel: 5e-1 (actual err ~0, exact match) |

#### Hopper CPU vs GPU Tests

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_hopper_full_step_contact_cpu_vs_gpu.mojo` | Hopper full step with contacts CPU vs GPU (float32, elliptic) | PENDING | 5 (ground contact, actions, deep pen, moving, 5-step) | qpos: 3e-2, qvel: 5e-1 |

#### Swimmer CPU vs GPU Tests

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_swimmer_fk_cpu_vs_gpu.mojo` | FK: xpos, xquat, xipos per body (float32, 4 bodies) | PASS | 5 (zeros, bent joints, displaced, moving+rotation, extreme joints) | pos: 1e-4, quat: 1e-4 |
| `test_swimmer_full_step_cpu_vs_gpu.mojo` | Full step RK4+Newton, no contacts (float32) | PASS | 5 (zero state 1-step, bent joints 1-step, motor actions 1-step, moving+actions 1-step, motor actions 10-steps) | qpos: 3e-2, qvel: 5e-1 |

#### Inverted Double Pendulum CPU vs GPU Tests

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_inverted_double_pendulum_fk_cpu_vs_gpu.mojo` | FK: xpos, xquat, xipos per body (float32, 4 bodies incl. tip site) | PASS | 5 (zeros, displaced cart, first hinge, both hinges, large tilt) | pos: 1e-4, quat: 1e-4 |

#### Walker2D CPU vs GPU Tests

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_walker2d_fk_cpu_vs_gpu.mojo` | FK: xpos, xquat, xipos per body (float32, 8 bodies) | PASS | 5 (default standing rootz=1.25, large rootx, bent right leg, symmetric gait, extreme joints) | pos: 1e-4, quat: 1e-4 |

#### Ant CPU vs GPU Tests

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_ant_fk_cpu_vs_gpu.mojo` | FK: xpos, xquat, xipos per body (float32, 14 bodies, 3D free joint) | PASS | 5 (default init_qpos, raised torso, nonzero joints, rotated torso 30°, extreme joint angles) | pos: 1e-4, quat: 1e-4 |
| `test_ant_full_step_cpu_vs_gpu.mojo` | Full step RK4+Newton, no contacts / free fall (float32) | PASS | 5 (free fall 1-step, free fall+actions 1-step, default joints raised 1-step, moving+actions 1-step, free fall 10-steps) | qpos: 3e-2, qvel: 5e-1 |

#### Humanoid CPU vs GPU Tests

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_humanoid_fk_cpu_vs_gpu.mojo` | FK: xpos, xquat, xipos per body (float32, 14 bodies, free joint + 17 hinge) | PASS | 5 (default standing, bent knees, arms extended, rotated torso 45°, full body pose) | pos: 1e-3, quat: 1e-3 (relaxed for nested quat chain) |

### Analytical / Standalone Tests

| Test File | What | CPU/GPU | Status |
|-----------|------|---------|--------|
| `test_forward_kinematics.mojo` | FK on generic 1-2 body models (identity, 90-deg, double pendulum) | CPU | PASS |
| `test_pendulum.mojo` | Pendulum period (~2.016s) + energy conservation (<5% drift) | CPU | PASS |
| `test_pendulum_gpu.mojo` | Same pendulum on GPU (relaxed: period 5%, energy 10%) | GPU | PASS |
| `test_implicit_integrator.mojo` | LU factorization, ImplicitIntegrator compilation, zero-vel qDeriv | CPU | PASS |

### Diagnostic / Stress Tests (not pass/fail, produce logs)

| Test File | What | Model |
|-----------|------|-------|
| `test_cheetah_torque_diag.mojo` | 5 constant-torque scenarios (free fall, small, max, all-down, all-up) | HalfCheetah (Newton) |
| `test_cheetah_policy_diag.mojo` | PPO policy actions over 3 episodes (requires checkpoint) | HalfCheetah (Newton) |
| `test_cheetah_diagnostics.mojo` | Free fall per-step logging (body Z, contacts, penetration) | HalfCheetah (PGS) |
_(`test_solver_debug.mojo` was listed here as a single-step solver trace. The
file was EMPTY — 0 bytes in every commit that contains it, back to the June
folder restructure — so the row documented something that never existed, while
the file itself reported "module does not define a `main` function" in every
sweep. Deleted 2026-08-08. A permanent red teaches people to ignore reds.)_

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
- Test must set solref/solimp from HalfCheetahModel (like the actual environment does)
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
stiffness, frictionloss), M with armature only (no dt*damping — solver uses M+arm, damping
is implicit in velocity integration), qfrc_smooth, then NewtonSolver.solve() with
CONE_TYPE=ELLIPTIC.

**Tolerance:** abs: 5e-2, rel: 2e-1

**Result:** 4/4 PASS. Cost matches MuJoCo to 12 significant digits. qacc errors ~1e-5.

**Solver:** NewtonSolver — MuJoCo-style primal Newton in qacc space with 3-zone
cone logic (TOP/BOTTOM/CONE), cone-aware Hessian, and Newton root-finding linesearch.
Handles all constraints (normals + friction cone + limits) in a unified optimization.
No separate PGS friction phase needed.

**Bugs fixed (cumulative):**
1. **Tangent frame** (FIXED): T1 now uses capsule axis as hint via MuJoCo's `mju_makeFrame`
   (Gram-Schmidt orthogonalization). T1 K values now 0.27-0.77 (was 1e-10 degenerate).
   Files: `contact_detection.mojo`, `constraint_builder.mojo`, `friction_solver.mojo`, `pgs_solver.mojo`
2. **Friction aref** (FIXED): Removed spurious `imp` factor from friction bias.
   Was `B_damp * imp * v_t`, now `B_damp * v_t` (MuJoCo doesn't apply impedance to friction bias).
   File: `constraint_builder.mojo`
3. **Zone condition & Dm formula** (FIXED): BOTTOM zone condition and Dm had spurious `group_size`
   factor. Was `group_size*mu*jar_n + T <= 0` and `Dm = D_n/(1+group_size*mu^2)`.
   Now `mu*jar_n + T <= 0` and `Dm = D_n/(1+mu^2)`.
   Root cause: MuJoCo's `con->mu = friction[0]` (per-direction mu for impratio=1),
   NOT `friction[0]*sqrt(group_size)` as previously assumed. The U-space mapping
   `U[0] = jar_n * con->mu` uses per-direction mu, so zone boundaries don't have a group_size factor.
   Files: `primal_common.mojo` (12 locations), `newton_solver.mojo` (1 location)

**Current results:**

| Config | qfrc max_abs | qfrc max_rel | qacc max_abs | qacc max_rel | cost match |
|--------|-------------|-------------|-------------|-------------|------------|
| Low static | 1.1e-3 | 6.3e-6 | 8.6e-4 | 6.8e-6 | 12 digits |
| Low moving | 2.1e-3 | 4.3e-6 | 1.9e-3 | 3.5e-5 | 12 digits |
| Very low | 3.1e-3 | 9.2e-6 | 2.5e-3 | 4.0e-5 | 12 digits |
| Bent legs | 2.0e-4 | 3.0e-5 | 9.4e-4 | 4.0e-6 | 12 digits |

### Stage 6: Pyramidal Cone — Euler + Newton

| Component              | MuJoCo vs CPU | CPU vs GPU | Notes                                    |
|------------------------|:-------------:|:----------:|------------------------------------------|
| Solver Forces          | DONE          | DONE       | 4 configs, qacc ~1e-3, D/R exact match   |

**Test:** `test_pyramidal_vs_mujoco.mojo` — Compares our Newton solver with `ConeType.PYRAMIDAL`
against MuJoCo's Newton solver with `opt.cone=0` (pyramidal).

**Pyramidal cone vs Elliptic:**
- Elliptic: separate normal + friction rows, QCQP projection for cone constraint
- Pyramidal: edge rows `J_edge± = J_n ± mu*J_t` with `lambda >= 0` (simple inequality)
- MuJoCo CG/Newton treats pyramidal edges as plain inequality constraints (no 3-zone cone logic)

**Pyramidal-specific D/R computation (differs from elliptic):**
- `diagApprox = tran + mu² * tran` (vs just `tran` for elliptic normal)
- `R_initial = (1-imp)/imp * diagApprox`
- `R_edge = 2 * mu² * R_initial` (MuJoCo engine_core_constraint.c:1484-1493)

**Bugs found and fixed:**
1. **Edge velocity source** (FIXED): Pyramidal edge bias used `qvel[i]` (function parameter)
   instead of `data.qvel[i]` (actual joint velocity). This was the ONLY place in the entire
   constraint builder that used the parameter instead of `data.qvel`. For static poses with
   gravity, this injected `B_damp * J_edge * qacc0 ≈ 125 * 9.81 = 1226` into all biases.
   File: `constraint_builder.mojo` line 586
2. **Edge R/D formula** (FIXED): Was using elliptic formula `R = (1-imp)/imp * diag_n`.
   Now uses MuJoCo's pyramidal-specific `R = 2*mu²*(1-imp)/imp*(tran + mu²*tran)`.
   File: `constraint_builder.mojo` lines 538-553

**Current results:**

| Config | qfrc max_abs | qfrc max_rel | qacc max_abs | qacc max_rel |
|--------|-------------|-------------|-------------|-------------|
| Low static | 4.9e-4 | 8.0e-6 | 1.5e-3 | 5.6e-6 |
| Low moving | 2.0e-3 | 4.4e-6 | 2.6e-3 | 1.8e-5 |
| Very low | 3.0e-3 | 1.2e-5 | 3.8e-3 | 2.5e-5 |
| Bent legs | 1.9e-4 | 5.9e-6 | 9.6e-4 | 4.6e-6 |

**CPU vs GPU (`test_pyramidal_cpu_vs_gpu.mojo`):**
CPU uses primal Newton solver (exact cone logic), GPU uses primal Newton (same algorithm).
Both use `ConeType.PYRAMIDAL`. 6/6 configs pass, errors at float32 rounding level (~1e-6).

| Config | qpos max_abs | qvel max_abs | Notes |
|--------|-------------|-------------|-------|
| Ground contact (1 step) | ~1e-8 | ~1e-6 | Near float32 precision |
| Ground + actions (1 step) | ~1e-8 | ~1e-6 | |
| Deep penetration (1 step) | ~1e-8 | ~1e-6 | 10 contacts |
| Moving + contacts (1 step) | ~1e-8 | ~1e-6 | |
| Ground contact (5 steps) | ~1e-7 | ~1e-5 | |
| Ground + actions (5 steps) | ~1e-7 | ~1e-5 | |

3. **GPU per-edge bias** (FIXED): GPU friction solver used normal-only bias (`bias_n`) for all
   pyramidal edges. MuJoCo uses per-edge velocity: `v_edge = (J_n ± mu*J_t) * qvel`.
   Fix: `bias_pos = bias_n + mu * B_damp * v_t`, `bias_neg = bias_n - mu * B_damp * v_t`.
   `B_damp * v_t` was already computed in workspace (`bf[d*MC+c]`).
   File: `friction_solver.mojo` lines 664, 724

### Stage 8: GPU body_invweight0 fix + direct R_n computation

**Root cause of all CPU vs GPU contact test failures (3/6 → 6/6):**

The GPU `_compute_invweight0_gpu` kernel produced incorrect **translational** `body_invweight0[2*i]`
values (~9x off from CPU). Rotational values `body_invweight0[2*i+1]` were correct. This caused
`diag_n` in the constraint builder to be ~9x too large → `R_n` ~9x too large → `D` (force scaling)
~9x too small → all constraint forces ~9x weaker than correct.

Both `ModelDef.init_model_gpu` and `ModelDefFromXML.init_model_gpu` called `_compute_invweight0_gpu`
after already having correct CPU-computed values, overwriting them.

**Fixes:**
1. **`model_def_from_xml.mojo`**: Removed `_compute_invweight0_gpu` call — CPU `setup_model_and_data`
   already computes correct `body_invweight0` which is serialized via `copy_invweight0_to_buffer`.
2. **`model_def.mojo`**: Replaced `_compute_invweight0_gpu` with CPU-side computation using
   a temporary Model/Data + `finalize()`.
3. **`constraint_builder_gpu.mojo`**: Extended common normal workspace from `13*MC+2*MC*NV` to
   `15*MC+2*MC*NV`, adding `imp_n` (slot 13) and `diag_n` (slot 14). Friction builder now computes
   `R_n = (1-imp)/imp * diag_n` directly instead of lossy `R = 1/inv_K_imp - K`.
4. **All GPU solvers**: Updated J_n/MinvJn offsets from `13*MC` to `15*MC`, workspace sizes +2*MC.
   Replaced lossy R_n recovery with direct computation from stored imp/diag_n.
   Files: `newton_solver.mojo`, `cg_solver.mojo`, `old_newton_solver.mojo`, `pgs_solver.mojo`,
   `island_pgs_solver.mojo`, `friction_solver.mojo`

**Test results after fix (2026-04-03):**

| Test | Before | After | Notes |
|------|--------|-------|-------|
| test_full_step_contact_cpu_vs_gpu (ELLIPTIC) | 3/6 | **6/6** | All single-step errors ~1e-6 |
| test_pyramidal_cpu_vs_gpu (PYRAMIDAL) | 3/6 | **6/6** | All errors ~1e-6 |
| test_implicit_fast_newton_cpu_vs_gpu | 3/7 | **7/7** | All pass. Warm-start bug fixed (was using prev step qacc as solver initial guess on GPU only). |
| test_implicit_fast_pgs_cpu_vs_gpu | 4/8 | **8/8** | All pass. Same warm-start fix. |
| test_pyramidal_vs_mujoco | 4/4 | 4/4 | Unchanged |
| test_full_step_contact_vs_mujoco | 7/7 | 7/7 | Unchanged |
| test_solver_forces_vs_mujoco | 4/4 | 4/4 | Unchanged |
| test_ant_full_step_cpu_vs_gpu | PASS | PASS | Unchanged |

### Stage 7: RK4 integrator

| Component                        | Integrator | Solver       | MuJoCo vs CPU | CPU vs GPU | Notes |
|----------------------------------|------------|--------------|:-------------:|:----------:|-------|
| Full Step no contact + contact   | RK4        | Newton       | DONE          | DONE       | MuJoCo: 6 configs, err~1e-6. GPU: 6 configs (float32). |

**RK4 MuJoCo comparison notes:**
- MuJoCo RK4 (`mj_RungeKutta`) runs FULL forward dynamics + constraint solver at EACH of the 4 RK4 stages
- Our RK4Integrator restructured to match: `_forward_dynamics` + `_solve_constraints` at each stage
- `_solve_constraints` extracted to separate function (its own stack frame) to avoid stack overflow
  from 4× ConstraintData allocations (~30KB each)
- Limited to 6 tests per run due to RK4 step() large stack frame (4× forward dynamics + InlineArrays)
- GPU version: 9 kernel launches per step (4 × stage_kernel + 4 × solver + 1 combine_kernel)
- GPU workspace: RK4 extras (q0, v0, A[0-3], C1, C2) appended after solver workspace

**Test results:**

| Config | qpos max_abs | qvel max_abs | Contacts |
|--------|-------------|-------------|----------|
| Free fall (1 step) | 7.4e-19 | 1.5e-16 | 0 |
| Actions (1 step) | 1.0e-8 | 1.8e-6 | 0 |
| Moving+actions (1 step) | 2.8e-8 | 5.2e-6 | 0 |
| Fast spinning (1 step) | 5.5e-8 | 9.9e-6 | 0 |
| Moving+actions (10 steps) | 5.3e-7 | 3.8e-6 | 0 |
| Ground contact+actions (1 step) | 2.2e-8 | 3.5e-6 | 6 |

**Bugs found and fixed:**
1. **Constraint-once approach** (FIXED): Original RK4 ran constraints once at the end.
   MuJoCo runs full dynamics+constraints at each of 4 stages. Contact test showed large errors
   (qpos ~0.05, qvel ~6.4). Fixed by restructuring to per-stage constraint solving.
2. **Missing M_hat/qfrc_smooth** (FIXED): RK4 integrator didn't fill `constraints.M_hat` and
   `constraints.qfrc_smooth` before calling Newton solver, causing crash on contact tests.
3. **Stack overflow** (WORKAROUND): 7th compare_step call crashes due to accumulated stack from
   RK4's large frame (4× forward dynamics). Extracted `_solve_constraints` to separate function
   and limited to 6 tests per run.

---

## DONE: GPU — Sync with CPU D/R Changes

The GPU constraint builder now uses MuJoCo's `invweight0` formulas for D/R computation,
matching the CPU implementation. Validated by `test_constraint_params_cpu_vs_gpu.mojo` (4/4 pass).

### Changes made:

### What changed on CPU

1. **`constraint_data.mojo`**: Added `diagApprox` field to `ConstraintRow`
2. **`types.mojo`**: Added `dof_invweight0[NV]` to Model (diagonal of M^{-1})
3. **`mass_matrix.mojo`**: `compute_body_invweight0` now also computes `dof_invweight0`
4. **`constraint_builder.mojo`**: `_compute_aref` returns `(bias, inv_K_imp, imp)` where
   `inv_K_imp = 1/(K + R)` with `R = (1-imp)/imp * diagApprox`
5. **`newton_solver.mojo`**: Simplified — uses `primal_D(inv_K_imp, K)` instead of
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

### Full Step with Contacts — PASS (err ~1e-5)

`test_full_step_contact_vs_mujoco.mojo` passes with 2 configs (qpos err ~1e-7, qvel err ~1e-5).

**Bugs fixed:**
1. **M vs M_hat separation** (FIXED): EulerIntegrator was adding `arm + dt*damp` to M diagonal
   (ImplicitFast behavior). MuJoCo's Euler uses `M + arm` only for the solver. Fixed to only add armature.
2. **Implicit velocity damping** (FIXED): MuJoCo 3.3.6's "Euler" integrator applies implicit
   damping in the velocity update: `(M + dt*D) * v_new = M * (v_old + dt * qacc)`.
   Our integrator was doing pure `qvel += dt*qacc`. Fixed by adding a second LDL factorization
   of `M_hat = M + arm + dt*D` and solving for `v_new`. This matches MuJoCo to machine precision.
3. **MuJoCo integrator setting** (FIXED): Test wasn't setting `opt.integrator = 0` (Euler).
   MuJoCo 3.x defaults may differ from older versions.
4. **body_invweight0 pose** (FIXED): Test computed invweight0 at test pose instead of reference
   pose (MuJoCo computes it once in mj_setConst at the default pose).

---

## TODO: CPU vs GPU

For each component that passes MuJoCo vs CPU, add a CPU vs GPU comparison test.
**Only after the CPU is validated against MuJoCo.**

### Already Done
- Forward Kinematics (CPU vs GPU) — DONE
- Mass Matrix (CPU vs GPU) — DONE
- Bias Forces (CPU vs GPU) — DONE

### After GPU D/R sync (DONE)
- Constraint Parameters (CPU vs GPU) — DONE (validated GPU D/R changes)

### Remaining Stage 2+ CPU vs GPU tests — ALL DONE
- Contact Detection (CPU vs GPU) — DONE (6/6, err ~0)
- Constraint Jacobians (CPU vs GPU) — DONE (4/4, err ~6e-8)
- Solver Forces (CPU vs GPU) — DONE (validated via full step single-step configs)
- Full Step with contacts (CPU vs GPU) — DONE (6/6, static ~1e-5, deep ~4e-3)

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
│                  │    │ 9. Full Step (contact)          │ DONE (err ~1e-5)
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


New Coverage Analysis                                                                                                                                                                 
                                                                                                                                                                                        
  ┌──────────────────────────┬───────────────────────────────────────────────────────────────────────┐                                                                                  
  │       Environment        │                  What's unique vs HalfCheetah/Hopper                  │
  ├──────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Swimmer                  │ No contacts (contype=0), pure dynamics, 3-body chain                  │
  ├──────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Inverted Double Pendulum │ Simplest env (3 DOF), slide+2hinges, no contacts, tip site            │
  ├──────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Walker2D                 │ Biped with 2 legs (vs Hopper's 1), RK4 integrator                     │
  ├──────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Ant                      │ First 3D free joint (quaternion DOFs), 4-leg tree, real 3D locomotion │
  ├──────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Humanoid                 │ Tendon constraints (fixed joint couplings), 24 DOF, most complex      │
  └──────────────────────────┴───────────────────────────────────────────────────────────────────────┘

  Proposed Tests (prioritized by unique coverage)

  Tier 1 — High value, different physics

  1. test_ant_fk_vs_mujoco.mojo — DONE (5/5 PASS). First test with a real 3D free joint (quaternion body). Validates FK for 3D multi-leg topology.
  2. test_ant_full_step_vs_mujoco.mojo — DONE (5/6 PASS). Free joint + 4-leg contacts. Most different from existing tests. 1 FAIL: large actions + contacts (contact solver accuracy).
  3. test_swimmer_fk_vs_mujoco.mojo — DONE (6/6 PASS). Pure dynamics, no contacts. Different body chain topology.
  4. test_swimmer_full_step_vs_mujoco.mojo — DONE (5/5 PASS). Contact-free full step (unique — all existing full step tests have contacts or are in free fall).

  Tier 2 — Good coverage additions

  5. test_inverted_double_pendulum_fk_vs_mujoco.mojo — DONE (5/5 PASS). Simplest model, slide+hinge chain. Machine precision.
  6. test_walker2d_fk_vs_mujoco.mojo — DONE (5/5 PASS). Biped topology, complements Hopper coverage. Machine precision.
  7. test_humanoid_fk_vs_mujoco.mojo — DONE (5/5 PASS). Body-level quat attrs, dense tree, free joint. Fixed _parse_quat bug (w,x,y,z order).

  CPU vs GPU Equivalents (CREATED)

  8.  test_swimmer_fk_cpu_vs_gpu.mojo — DONE. 5 configs (zeros, bent, displaced, moving, extreme). pos/quat tol=1e-4.
  9.  test_swimmer_full_step_cpu_vs_gpu.mojo — DONE. 5 configs (zero 1-step, bent 1-step, actions 1-step, moving+actions 1-step, actions 10-steps). No contacts.
  10. test_inverted_double_pendulum_fk_cpu_vs_gpu.mojo — DONE. 5 configs (zeros, cart, first hinge, both hinges, large tilt). NSITE=1 included.
  11. test_walker2d_fk_cpu_vs_gpu.mojo — DONE. 5 configs (default standing rootz=1.25, large rootx, bent right leg, symmetric gait, extreme joints). pos/quat tol=1e-4.
  12. test_ant_fk_cpu_vs_gpu.mojo — DONE. 5 configs (default init_qpos, raised, nonzero joints, rotated 30°, extreme). Free joint quaternion init. pos/quat tol=1e-4.
  13. test_ant_full_step_cpu_vs_gpu.mojo — DONE. 5 configs (free fall, free fall+actions, default joints raised, moving+actions, free fall 10-steps). All z=2.0 (no contacts).
  14. test_humanoid_fk_cpu_vs_gpu.mojo — DONE. 5 configs (default standing, bent knees, arms extended, rotated 45°, full body pose). pos/quat tol=1e-3 (lwaist quat accumulation).

---

## Full Test Run Results (2026-04-03, Apple Silicon)

### MuJoCo vs CPU Tests

| Test File | Result | Notes |
|-----------|--------|-------|
| test_fk_vs_mujoco | 5/5 PASS | |
| test_mass_matrix_vs_mujoco | 4/4 PASS | |
| test_contacts_vs_mujoco | 6/6 PASS | |
| test_pgs_vs_mujoco | 4/4 PASS | |
| test_constraint_params_vs_mujoco | 4/4 PASS | |
| test_jacobian_vs_mujoco | 4/4 PASS | |
| test_full_step_vs_mujoco | 6/6 PASS | |
| test_full_step_contact_vs_mujoco | 7/7 PASS | |
| test_solver_forces_vs_mujoco | 4/4 PASS | |
| test_cg_vs_mujoco | 4/4 PASS | |
| test_implicit_fast_step_vs_mujoco | 5/5 PASS | |
| test_implicit_fast_step_contact_vs_mujoco | 4/4 PASS | |
| test_implicit_step_vs_mujoco | 3/6 (3 fail) | Pre-existing: moving/spinning configs fail |
| test_rk4_step_vs_mujoco | 6/6 PASS | |
| test_pyramidal_vs_mujoco | 4/4 PASS | |
| test_qderiv_vs_mujoco | 3/3 PASS | |
| test_inertiafromgeom_vs_mujoco | 2/2 PASS | |
| test_ant_full_step_vs_mujoco | 6/6 PASS | |
| test_hopper_full_step_vs_mujoco | 6/6 PASS | |
| test_swimmer_full_step_vs_mujoco | 4/4 PASS | |
| test_walker2d_full_step_vs_mujoco | 6/6 PASS | |
| test_inverted_pendulum_full_step_vs_mujoco | 5/5 PASS | |
| test_humanoid_full_step_vs_mujoco | 6/6 PASS | |
| test_bias_forces_vs_mujoco | 0/5 (5 fail) | Pre-existing: CPU bias forces diverge |
| test_qacc0_vs_mujoco | 0/5 (5 fail) | Pre-existing: depends on bias forces |
| test_hopper_solver_forces_vs_mujoco | 0/4 (4 fail) | Pre-existing: Hopper friction forces diverge |

### CPU vs GPU Tests

| Test File | Result | Notes |
|-----------|--------|-------|
| test_full_step_contact_cpu_vs_gpu | **6/6 PASS** | Fixed by invweight0 bug fix (was 3/6) |
| test_pyramidal_cpu_vs_gpu | **6/6 PASS** | Fixed by invweight0 bug fix (was 3/6) |
| test_implicit_fast_newton_cpu_vs_gpu | 5/7 (2 fail) | 1-step: all pass. 5-step: accumulated float32 divergence |
| test_implicit_fast_pgs_cpu_vs_gpu | 6/8 (2 fail) | 1-step: all pass. 5-step: accumulated float32 divergence |
| test_ant_full_step_cpu_vs_gpu | ALL PASS | |