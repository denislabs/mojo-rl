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

---

## Test Matrix: MuJoCo vs CPU vs GPU

| Component              | MuJoCo vs CPU | CPU vs GPU | Notes                                    |
|------------------------|:-------------:|:----------:|------------------------------------------|
| Forward Kinematics     | DONE          | DONE       | xpos, xquat, xipos — 5 configs, err~1e-16 |
| Mass Matrix (CRBA)     | DONE          | DONE       | Full NV x NV matrix — 4 configs, err~2e-6 |
| Bias Forces (RNE)      | DONE          | DONE       | Coriolis + gravity — 5 configs           |
| Constraint Solver      | TODO          | TODO       | Contact forces, penetration              |
| Full Step (integrate)  | TODO          | TODO       | qpos/qvel after N steps                  |

---

## Existing Tests

### MuJoCo Comparison Tests (CPU)

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_fk_vs_mujoco.mojo` | FK: xpos, xquat, xipos per body | PASS | 5 (default, zero, nonzero, extreme, large rootx) | pos: 1e-6, quat: 1e-5 |
| `test_mass_matrix_vs_mujoco.mojo` | Full mass matrix (CRBA + armature) | PASS | 4 (default, zero, nonzero, extreme) | abs: 1e-4, rel: 1e-3 |
| `test_bias_forces_vs_mujoco.mojo` | Bias forces RNE (qfrc_bias) | NEW | 5 (zero vel, nonzero joints, nonzero vel, extreme) | abs: 1e-4, rel: 1e-3 |

### CPU vs GPU Comparison Tests

| Test File | What | Status | Configs | Tolerance |
|-----------|------|--------|---------|-----------|
| `test_fk_cpu_vs_gpu.mojo` | FK: xpos, xquat, xipos per body (float32) | NEW | 5 (default, zero, nonzero, extreme, large rootx) | pos: 1e-4, quat: 1e-4 |
| `test_mass_matrix_cpu_vs_gpu.mojo` | Full mass matrix CRBA (float32) | NEW | 4 (default, zero, nonzero, extreme) | abs: 1e-3, rel: 1e-2 |
| `test_bias_forces_cpu_vs_gpu.mojo` | Bias forces RNE (float32) | NEW | 5 (zero vel, nonzero joints, nonzero vel, extreme) | abs: 1e-2, rel: 1e-2 |

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

## TODO: MuJoCo vs CPU

### Bias Forces — DONE
- `test_bias_forces_vs_mujoco.mojo` — 5 configs (zero vel, nonzero joints, nonzero vel, extreme)
- Tests both gravity-only (qvel=0) and full Coriolis (qvel!=0)
- Compares against `mj_data.qfrc_bias` (NV vector)

### Constraint Solver / Contact Forces
- Compare contact normal/friction forces against MuJoCo's `mj_data.qfrc_constraint`
- Requires matching: contact detection, constraint building, solver output
- Hardest to compare due to solver sensitivity (different convergence paths)
- Suggested approach: compare `qfrc_constraint` (net constraint force in joint space) rather than per-contact forces

### Full Step
- Run N identical steps in both engines, compare final qpos/qvel
- Start from same qpos + same actions, compare after 1, 10, 100 steps
- This is the ultimate integration test

---

## TODO: CPU vs GPU

For each component that passes MuJoCo vs CPU, add a CPU vs GPU comparison test:

### Forward Kinematics (CPU vs GPU) — DONE
- `test_fk_cpu_vs_gpu.mojo` — same 5 configs as MuJoCo test
- Launches standalone GPU FK kernel, compares xpos/xquat/xipos per body
- Tolerance: 1e-4 (float32 precision)

### Mass Matrix (CPU vs GPU)
- Compare full M matrix from CPU CRBA vs GPU CRBA
- Same configs as MuJoCo test

### Bias Forces (CPU vs GPU) — DONE
- `test_bias_forces_cpu_vs_gpu.mojo` — same 5 configs
- GPU kernel runs FK + body_velocities + cdof + bias_forces_rne_gpu
- Tolerance: 1e-2 (float32, accumulated through full pipeline)

### Constraint Solver (CPU vs GPU)
- Compare qacc output from CPU solver vs GPU solver for identical inputs
- Test all 3 solvers: PGS, CG, Newton

### Full Step (CPU vs GPU)
- Run N identical steps on CPU and GPU, compare final state
- This catches any integration drift between CPU and GPU paths
- Test with zero actions (free fall) and with constant actions

---

## Test Naming Convention

| Pattern | Meaning |
|---------|---------|
| `test_<component>_vs_mujoco.mojo` | MuJoCo reference comparison (CPU) |
| `test_<component>_cpu_vs_gpu.mojo` | CPU vs GPU comparison |
| `test_<component>.mojo` | Standalone / analytical test (CPU) |
| `test_<component>_gpu.mojo` | Standalone GPU test |
| `test_<component>_diag.mojo` | Diagnostic (logging, not pass/fail) |
