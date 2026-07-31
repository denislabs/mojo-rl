# Environments Roadmap

## Current State

Native physics3d environments implemented:
- **HalfCheetah** — 2D planar cheetah, forward locomotion
- **Hopper** — 1D hopper, forward locomotion
- **Ant** — 4-legged quadruped, forward locomotion (typed model def, no XML)
- **InvertedPendulum** — cart-pole balancing (Phase 1 complete)
- **Swimmer** — 3-link planar swimmer, forward locomotion (Phase 1 complete)
- **Walker2d** — 2D bipedal walker, forward locomotion (Phase 1 complete)
- **InvertedDoublePendulum** — double-pole cart-pole, 9D custom obs (Phase 2 complete)
- **Humanoid** — 3D bipedal, simplified 45D obs, init qpos z=1.4 (Phase 2 complete)
- **HumanoidStandup** — 3D bipedal standup from lying, simplified 45D obs (Phase 2 complete)

Gymnasium Python wrappers exist for all remaining envs but no native implementations.

---

## Remaining Environments

### XML Dimensions Reference

| Env | NBODY | NQ | NV | NACT | NGEOM | NSITE | Tendons | Timestep |
|---|---|---|---|---|---|---|---|---|
| Swimmer | 4 | 5 | 5 | 2 | 4 | 0 | 0 | 0.01 |
| Walker2d | 9 | 9 | 9 | 6 | 9 | 0 | 0 | 0.002 |
| InvertedPendulum | 3 | 2 | 2 | 1 | 3 | 0 | 0 | 0.02 |
| InvDoublePendulum | 4 | 3 | 3 | 1 | 5 | 1 | 0 | 0.01 |
| Humanoid | 14 | 24 | 23 | 17 | 18 | 0 | 2 | 0.003 |
| HumanoidStandup | 14 | 24 | 23 | 17 | 18 | 0 | 2 | 0.003 |
| Reacher | 5 | 4 | 4 | 2 | 11 | 0 | 0 | 0.01 |
| Pusher | 12 | 11 | 11 | 7 | 26 | 0 | 0 | 0.01 |

Note: Humanoid NQ=24 (free joint 7 qpos + 17 hinges), NV=23 (free joint 6 qvel + 17 hinges). NGEOM=18 (floor + 3 torso + 1 lwaist + 1 pelvis/butt + 4 right leg + 4 left leg + 3 right arm + 3 left arm).

### XML Sources
All XMLs are in `Gymnasium-main/gymnasium/envs/mujoco/assets/`:
- `swimmer.xml`, `walker2d_v5.xml`, `inverted_pendulum.xml`, `inverted_double_pendulum.xml`
- `humanoid.xml`, `humanoidstandup.xml`
- `reacher.xml`, `pusher_v5.xml`

### Reward / Termination / Obs Summary

**Swimmer**
- Reward: `forward_weight * x_velocity - ctrl_cost_weight * ||action||²`
- x_velocity: delta qpos[0] / dt
- Termination: never
- Obs: qpos[2:] + qvel (skip x=0, y=1) → obs_qpos_skip=2, OBS_DIM=8
- Init qpos: all zero

**Walker2d**
- Reward: `healthy_reward + forward_weight * x_velocity - ctrl_cost_weight * ||action||²`
- healthy_reward=1.0 per step when alive
- Termination: z not in [0.8, 2.0] OR pitch angle not in [-1.0, 1.0]
- Obs: qpos[1:] + qvel (skip rootx) → obs_qpos_skip=1, OBS_DIM=17
- Init qpos: rootz=1.25 (non-zero! joint ref="1.25" in XML)

**InvertedPendulum**
- Reward: +1.0 per step if alive (0 if not)
- Termination: |pole_angle| > 0.2 OR any state not finite
- Obs: qpos + qvel (no skip) → obs_qpos_skip=0, OBS_DIM=4
- Init qpos: all zero, noise scale=0.01

**InvertedDoublePendulum**
- Reward: `alive_bonus(10) - 0.01*x_tip² - (y_tip - 2)² - (1e-3*v1² + 5e-3*v2²)`
  - x_tip, y_tip = world position of pole2 tip (from pole2 body xpos + site offset)
  - v1=qvel[1], v2=qvel[2]
- Termination: y_tip <= 1.0 (pole2 tip too low)
- Obs (9 elements, CUSTOM): `[qpos[0], sin(qpos[1]), sin(qpos[2]), cos(qpos[1]), cos(qpos[2]), clip(qvel[0..2], -10, 10), 0.0]`
  - Note: uses `sin`/`cos` transform on hinge angles, clips velocities; last element is qfrc_constraint[0] (not in state buffer → 0)
  - x_tip, z_tip computed analytically: `x_tip = cart_x + 0.6*sin(q1) + 0.6*sin(q1+q2)`, `z_tip = 0.6*cos(q1) + 0.6*cos(q1+q2)`
- Init qpos: uniform noise, scale=0.1

**Humanoid**
- Reward: `healthy_reward(5.0) + forward_weight * x_velocity - ctrl_cost_weight * ||action||² - contact_cost`
  - contact_cost = 5e-7 * clamp(cfrc_ext norm², max=10) — needs external contact forces
  - x_velocity from CoM delta (mass-weighted average of body xpos), not raw qpos[0]
- Termination: z-height not in [1.0, 2.0]
- Obs BASIC (45 elements): qpos[2:] + qvel (skip x=0, y=1) → obs_qpos_skip=2, OBS_DIM=45
- Obs FULL (348 elements): + cinert(130) + cvel(78) + qfrc_actuator(17) + cfrc_ext(78) — defer
- Init qpos: `[0, 0, 1.4, 1, 0, 0, 0, 0, 0, ...]` (z=1.4, quat_w=1.0 for free joint)
- Tendons: 2 fixed tendons (left_hipknee, right_hipknee)

**HumanoidStandup**
- Reward: `uph_cost - ctrl_cost_weight * ||action||² - impact_cost + 1`
  - uph_cost = 1 * CoM_z_height (reward for being upright / standing up)
  - impact_cost = 5e-8 * clamp(cfrc_ext norm², range=(-1, 10))
- Termination: never (always truncated at 1000 steps)
- Obs: same structure as Humanoid (basic or full)
- Init qpos: torso starts at z=0.105 (lying down), different from humanoid
- Same XML structure as Humanoid but different initial orientation

**Reacher**
- Reward: `-dist_weight * ||fingertip - target||₂ - ctrl_weight * ||action||²`
  - fingertip position = body COM of `fingertip` body (from xpos)
  - target position = body COM of `target` body (from xpos, driven by qpos[2:4])
- Termination: never
- Obs (10 elements, FULLY CUSTOM):
  `[cos(qpos[0]), sin(qpos[0]), qpos[2], qpos[3], cos(qpos[1]), sin(qpos[1]), qvel[0], qvel[1], fingertip_x-target_x, fingertip_y-target_y]`
  - Not a simple qpos/qvel slice
- Init qpos: arm joints ±0.1 uniform; target ±0.2 sphere (random per episode)
- Body indices: fingertip=body 3, target=body 4

**Pusher**
- Reward: `-dist_weight * ||object - goal||₂ - near_weight * ||fingertip - object||₂ - ctrl_weight * ||action||²`
  - tips_arm body COM = fingertip position
  - object and goal positions from their body COMs
- Termination: never
- Obs (23 elements, CUSTOM): `qpos[:7] + qvel[:7] + tips_arm_pos(3) + object_pos(3) + goal_pos(3)`
  - Includes body COM positions of three named bodies
- Init qpos: arm=0; object uniform [-0.3,0]×[-0.2,0.2] (min 0.17 from goal); goal=[0.45,-0.05,-0.323]
- Body indices: tips_arm=body 9, object=body 10, goal=body 11

---

## Framework Gaps to Fix First

### Gap 1: Non-zero init_qpos for reset — **RESOLVED**

**Walker2d**: The MJCF joint ref for `rootz` is 1.25, so the body origin in world space is already at z=1.25 when qpos=0. Health bounds adjusted to equivalent qpos-relative values: `MIN_Z = -0.45`, `MAX_Z = 0.75`. No hook needed.

**Framework hook**: `init_qpos_gpu` added to `Phyics3dEnvConfig` trait (default: no-op). Called by `_reset_env_gpu` after noise is applied. Signature:
```mojo
@always_inline @staticmethod
def init_qpos_gpu[DTYPE: DType, BATCH_SIZE: Int, STATE_SIZE: Int](
    states: LayoutTensor[...], env: Int, qpos_off: Int,
):
    pass  # default no-op; override for Humanoid, HumanoidStandup
```
Humanoid/HumanoidStandup will override to set:
- Humanoid: `states[env, qpos_off + 2] = 1.4`, `states[env, qpos_off + 3] = 1.0`
- HumanoidStandup: `states[env, qpos_off + 2] = 0.105`

### Gap 2: Body xpos access in reward/config — **RESOLVED**

**Was**: `compute_reward_and_done_gpu` received only `qpos_off` and `meta_offset`.

**Now**: The trait signature includes `xpos_off`, `xipos_off`, `cfrc_ext_off`, and `cvel_off`:
```mojo
def compute_reward_and_done_gpu[...](
    states, model, actions, env,
    qpos_off, xpos_off, xipos_off, cfrc_ext_off, cvel_off,
    meta_offset, curriculum_offset,
    step_count, frame_skip, timestep,
) -> Tuple[Scalar[DTYPE], Bool]
```
All existing configs (HalfCheetah, Hopper, Ant, InvertedPendulum, Swimmer, Walker2d) already use this updated signature.

### Gap 3: Custom observation extraction — **RESOLVED**

**Was**: obs extraction always `qpos[obs_qpos_skip:] + qvel[:]` hardcoded in `ModelDefFromXML`.

**Now**: `custom_extract_obs_gpu` hook added to `Phyics3dEnvConfig` (default: `return False`). `_extract_obs_rewards_dones_gpu` calls it first; falls back to `MODEL_DEF.extract_obs_gpu` only when it returns False. `QVEL_OFF` comptime provided alongside `QPOS_OFF`/`XPOS_OFF`.

```mojo
@always_inline @staticmethod
def custom_extract_obs_gpu[DTYPE, BATCH_SIZE, STATE_SIZE, OBS_DIM](
    states, obs, env, qpos_off, qvel_off, xpos_off,
) -> Bool:
    return False  # default: use model's qpos[skip:]+qvel extraction
```

InvDoublePendulum, Reacher, and Pusher will override to return `True` and write custom obs directly into `obs[env, :]`.

---

## Implementation Plan

### Phase 1 — No framework changes needed — **COMPLETE**

#### 1.1 InvertedPendulum — **DONE**
- `envs/inverted_pendulum/inverted_pendulum_xml.mojo` — NQ=2, NV=2, NBODY=3, NJOINT=2, NGEOM=3, NACT=1
- `envs/inverted_pendulum/inverted_pendulum_config.mojo`:
  - Reward: `+1.0 per step if alive`
  - Termination: `|qpos[0]| >= 1.0 OR |qpos[1]| >= 0.2` (cart OOB or pole tipped)
  - Integrator: `EulerIntegrator[NewtonSolver]`
- `envs/inverted_pendulum/inverted_pendulum.mojo`, `__init__.mojo`

#### 1.2 Swimmer — **DONE**
- `envs/swimmer/swimmer_xml.mojo` — NQ=5, NV=5, NBODY=4, NJOINT=5, NGEOM=4, NACT=2, obs_qpos_skip=2
- `envs/swimmer/swimmer_config.mojo`:
  - `pre_step_cpu/gpu`: save qpos[0] as prev_x
  - Reward: `1.0 * x_velocity - 0.0001 * sum(action²)`
  - Termination: never
  - Integrator: `RK4Integrator[NewtonSolver]`
- `envs/swimmer/swimmer.mojo`, `__init__.mojo`

#### 1.3 Walker2d — **DONE**
- `envs/walker2d/walker2d_xml.mojo` — NQ=9, NV=9, NBODY=9, NJOINT=9, NGEOM=9, NACT=6, obs_qpos_skip=1
- `envs/walker2d/walker2d_config.mojo`:
  - `pre_step_cpu/gpu`: save qpos[0] as prev_x
  - Reward: `1.0 + 1.0 * x_velocity - 0.001 * sum(action²)` when healthy
  - Termination: `qpos[1] not in (-0.45, 0.75) OR |qpos[2]| >= 1.0`
    - Equivalent to Gymnasium's world-z ∈ [0.8, 2.0] given rootz joint ref=1.25 (see Gap 1)
  - Integrator: `RK4Integrator[NewtonSolver]`
- `envs/walker2d/walker2d.mojo`, `__init__.mojo`

---

### Phase 2 — Fix framework gaps, then implement remaining envs

#### 2.0 Fix Gap 1: Non-zero init_qpos — **DONE**
- Walker2d: resolved via bounds adjustment (no hook needed)
- Hook `init_qpos_gpu` added to `Phyics3dEnvConfig` trait (default: no-op `pass`)
- `Phyics3dEnv._reset_env_gpu` calls `CONFIG.init_qpos_gpu` after noise is applied
- All existing configs (HalfCheetah, Hopper, Ant, InvertedPendulum, Swimmer, Walker2d) implement the no-op default
- Humanoid/HumanoidStandup will override: qpos[2]=1.4/qpos[3]=1.0 or qpos[2]=0.105

#### 2.1 Fix Gap 2: xpos_off in compute_reward_and_done_gpu — **DONE**
- `xpos_off`, `xipos_off`, `cfrc_ext_off`, `cvel_off` added to trait signature
- All existing configs updated (HalfCheetah, Hopper, Ant, InvertedPendulum, Swimmer, Walker2d)

#### 2.2 Fix Gap 3: Custom obs extraction — **DONE**
- `custom_extract_obs_gpu` hook added to `Phyics3dEnvConfig` trait (default: `return False`)
- `Phyics3dEnv._extract_obs_rewards_dones_gpu` calls hook first; falls back to `MODEL_DEF.extract_obs_gpu` when it returns False
- `QVEL_OFF` comptime added alongside existing `QPOS_OFF`/`XPOS_OFF` for use by custom obs hooks
- All existing configs implement the default `return False`
- Envs with custom obs (InvDoublePendulum, Reacher, Pusher) will override to return True and write obs directly

#### 2.3 InvertedDoublePendulum (needs Gap 2 + Gap 3)
Files to create:
- `envs/inverted_double_pendulum/inverted_double_pendulum_xml.mojo` — NQ=3, NV=3, NBODY=4, NJOINT=3, NGEOM=5, NACT=1, NSITE=1
- `envs/inverted_double_pendulum/inverted_double_pendulum_config.mojo`:
  - Custom obs (override Gap 3): `[qpos[0], sin(qpos[1]), sin(qpos[2]), cos(qpos[1]), cos(qpos[2]), clip(qvel, -10, 10)×3, pole2_tip_x, qfrc_constraint[0]]`
  - Reward needs pole2 body xpos (xpos_off + 3*body_idx) + local site offset for tip y
  - Termination: `y_tip <= 1.0`
- `envs/inverted_double_pendulum/inverted_double_pendulum.mojo`, `__init__.mojo`

#### 2.4 Humanoid (needs Gap 1 + Gap 2)
Files to create:
- `envs/humanoid/humanoid_xml.mojo` — NQ=17, NV=16, NBODY=14, NJOINT=17, NGEOM=16, NACT=17, max_tendon=2, obs_qpos_skip=2
- `envs/humanoid/humanoid_config.mojo`:
  - `init_qpos_offset_gpu`: set qpos[2]=1.4, qpos[3]=1.0 (z + quat_w for free joint)
  - `pre_step_cpu/gpu`: save CoM x (mass-weighted average or just qpos[0]) as prev_x
  - Reward: `5.0 + 1.25 * x_velocity - 0.1 * sum(action²) - contact_cost`
    - contact_cost needs cfrc_ext — implement as 0.0 initially, add later
  - Termination: qpos[2] not in [1.0, 2.0] (z-height from free joint)
  - Integrator: `RK4Integrator[NewtonSolver]`
- `envs/humanoid/curriculum.mojo` — progressive healthy_z_range
- `envs/humanoid/humanoid.mojo`, `__init__.mojo`

#### 2.5 HumanoidStandup (needs Gap 1 + Gap 2)
Files to create:
- `envs/humanoid_standup/humanoid_standup_xml.mojo` — same dims as Humanoid, different XML
- `envs/humanoid_standup/humanoid_standup_config.mojo`:
  - `init_qpos_offset_gpu`: set torso z to initial lying-down position (z≈0.105)
  - Reward: `1.0 * CoM_z - 0.00001 * sum(action²) - impact_cost + 1.0`
    - CoM_z from mass-weighted xpos of all bodies (or approximated as qpos[2])
  - Termination: never
- `envs/humanoid_standup/humanoid_standup.mojo`, `__init__.mojo`

#### 2.6 Walker2d (fix init_qpos after Gap 1)
Update `walker2d_config.mojo` to set rootz=1.25 in `init_qpos_offset_gpu`.

---

### Phase 3 — Manipulation environments (needs all 3 gaps)

#### 3.1 Reacher
Files to create:
- `envs/reacher/reacher_xml.mojo` — NQ=4, NV=4, NBODY=5, NJOINT=4, NGEOM=11, NACT=2
  - Note: obs_qpos_skip irrelevant (fully custom obs)
- `envs/reacher/reacher_config.mojo`:
  - Custom obs (10 elements): `[cos(qpos[0]), sin(qpos[0]), qpos[2], qpos[3], cos(qpos[1]), sin(qpos[1]), qvel[0], qvel[1], fingertip_x-target_x, fingertip_y-target_y]`
    - fingertip pos = xpos[body_idx=3] (fingertip body)
    - target pos = xpos[body_idx=4] (target body)
  - Reward: `-||fingertip - target||₂ - 0.1 * sum(action²)`
  - Termination: never
  - Reset: arm joints uniform ±0.1; target qpos[2:4] uniform on disc radius 0.2 (re-sampled if dist<0.01 from center)
- `envs/reacher/reacher.mojo`, `__init__.mojo`

**Note on Reacher reset**: The target position changes every episode via random qpos[2:4]. The standard `reset_env_gpu` just adds noise around zero, which works here since the target joints have zero default and ±0.2 noise gives a random target location. However, the constraint that the target must be reachable (dist from center < 0.2) may need a custom reset hook.

#### 3.2 Pusher
Files to create:
- `envs/pusher/pusher_xml.mojo` — NQ=11, NV=11, NBODY=12, NJOINT=11, NGEOM=26, NACT=7
- `envs/pusher/pusher_config.mojo`:
  - Custom obs (23 elements): `qpos[:7] + qvel[:7] + xpos[tips_arm_body]×3 + xpos[object_body]×3 + xpos[goal_body]×3`
    - Body indices: tips_arm=9, object=10, goal=11
  - Reward: `-||object - goal||₂ - 0.1 * ||fingertip - object||₂ - 0.1 * sum(action²)`
  - Termination: never
  - Reset: arm=0; object random in [-0.3,0]×[-0.2,0.2] rejecting if dist<0.17 from goal; goal fixed at [0.45,-0.05,-0.323]
    - Custom reset hook needed for rejection sampling on GPU (or use fixed grid of valid positions)
- `envs/pusher/pusher.mojo`, `__init__.mojo`

---

## File Checklist

### Phase 1 (no framework changes) — COMPLETE
- [x] `envs/inverted_pendulum/__init__.mojo`
- [x] `envs/inverted_pendulum/inverted_pendulum_xml.mojo`
- [x] `envs/inverted_pendulum/inverted_pendulum_config.mojo`
- [x] `envs/inverted_pendulum/inverted_pendulum.mojo`
- [x] `envs/swimmer/__init__.mojo`
- [x] `envs/swimmer/swimmer_xml.mojo`
- [x] `envs/swimmer/swimmer_config.mojo`
- [x] `envs/swimmer/swimmer.mojo`
- [x] `envs/walker2d/__init__.mojo`
- [x] `envs/walker2d/walker2d_xml.mojo`
- [x] `envs/walker2d/walker2d_config.mojo`
- [x] `envs/walker2d/walker2d.mojo`

### Phase 2 (framework extensions first)
- [x] `envs/phyics3d_env_config.mojo` — add `init_qpos_gpu` hook (Gap 1)
- [x] `envs/phyics3d_env_config.mojo` — add `xpos_off`/`xipos_off`/`cfrc_ext_off`/`cvel_off` to `compute_reward_and_done_gpu` (Gap 2)
- [x] `envs/phyics3d_env_config.mojo` — add `custom_extract_obs_gpu` hook (Gap 3)
- [x] `envs/phyics3d_env.mojo` — updated for Gap 1 (`init_qpos_gpu` call in reset), Gap 2 (offsets), Gap 3 (`custom_extract_obs_gpu` with `QVEL_OFF`)
- [x] Update all existing configs (HalfCheetah, Hopper, Ant, InvertedPendulum, Swimmer, Walker2d) for all three gap hooks
- [x] `envs/walker2d/walker2d_config.mojo` — rootz=1.25 handled via bounds adjustment (Gap 1 workaround)
- [x] `physics3d/parser/model_def_from_xml.mojo` — add `obs_dim_override: Int = -1` parameter; `OBS_DIM = override if override > 0 else nq-skip+nv`
- [x] `envs/inverted_double_pendulum/__init__.mojo`
- [x] `envs/inverted_double_pendulum/inverted_double_pendulum_xml.mojo` — obs_dim_override=9
- [x] `envs/inverted_double_pendulum/inverted_double_pendulum_config.mojo` — custom 9D obs (sin/cos), analytical tip reward, terminate on z_tip<=1.0
- [x] `envs/inverted_double_pendulum/inverted_double_pendulum.mojo`
- [x] `envs/humanoid/__init__.mojo`
- [x] `envs/humanoid/humanoid_xml.mojo` — NQ=24, NV=23, NBODY=14, max_tendon=2, obs_qpos_skip=2
- [x] `envs/humanoid/humanoid_config.mojo` — init_qpos (z+=1.4, quat_w+=1.0), 45D obs, healthy_reward=5.0
- [x] `envs/humanoid/humanoid.mojo`
- [x] `envs/humanoid_standup/__init__.mojo`
- [x] `envs/humanoid_standup/humanoid_standup_xml.mojo` — same dims as Humanoid, torso pos="0 0 .105"
- [x] `envs/humanoid_standup/humanoid_standup_config.mojo` — init_qpos (z+=0.105, quat_w+=1.0), uph reward, no termination
- [x] `envs/humanoid_standup/humanoid_standup.mojo`
- Note: curriculum.mojo skipped for Humanoid (not needed for basic training)

### Phase 3 (manipulation envs)
- [ ] `envs/reacher/__init__.mojo`
- [ ] `envs/reacher/reacher_xml.mojo`
- [ ] `envs/reacher/reacher_config.mojo`
- [ ] `envs/reacher/reacher.mojo`
- [ ] `envs/pusher/__init__.mojo`
- [ ] `envs/pusher/pusher_xml.mojo`
- [ ] `envs/pusher/pusher_config.mojo`
- [ ] `envs/pusher/pusher.mojo`

---

## Open Questions

1. **Humanoid cinert/cvel/cfrc_ext obs**: **RESOLVED** — `cfrc_ext`, `cvel`, `cinert`, and
   `qfrc_actuator` are now persisted to the state buffer (appended after `site_xpos`).
   Humanoid/HumanoidStandup model defs can read from these offsets in `extract_obs_gpu`.
   Full 348-element obs: `qpos[2:] + qvel + cinert[1:NBODY] + cvel[1:NBODY] + qfrc_actuator + cfrc_ext[1:NBODY]`.

2. **Pusher rejection sampling on reset**: **RESOLVED (no action needed)** — The constraint
   `object-goal distance >= 0.17 m` is geometrically impossible to violate given the sampling
   region (object x∈[-0.3,0], y∈[-0.2,0.2]; goal fixed at [0.45,-0.05]). Minimum possible
   distance = sqrt(0.45²+0²) = 0.45 m >> 0.17 m threshold. Standard uniform reset works.

3. **Reacher target distance constraint on reset**: Similar to Pusher. The ±0.2 uniform sampling rarely produces unreachable positions, so a simple uniform reset without rejection is likely acceptable for training.

4. **HumanoidStandup cfrc_ext impact cost**: **RESOLVED** — `cfrc_ext` is now in the state
   buffer (computed by `compute_cfrc_ext_gpu` after each physics step). Read from `cfrc_ext_off`
   in `compute_reward_and_done_gpu` to compute `impact_cost = 0.5e-6 * sum(cfrc_ext²)`.

5. **Humanoid CoM x-velocity for forward reward**: **RESOLVED** — `cvel` is now in the state
   buffer (computed by `compute_cvel_gpu`). Exact CoM velocity:
   `x_velocity = Σ(mass_b * cvel[b*6+3]) / total_mass`. No finite-difference approximation needed.

6. **Walker2d timestep=0.002**: **RESOLVED** — The smaller timestep is passed correctly via `get_timestep()` in the XML model def. The integrator handles it automatically with the same frame_skip=4 as other envs.

---

## dm_control suite port

Not tracked here. The ledger is `docs/DM_CONTROL_PORT.md` (progress log, gap
list G1-G11, task tiering, staged plan). Tier A and Tier B are complete;
Tier C (quadruped, manipulator, stacker) is in progress.
