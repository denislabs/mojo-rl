"""Phyics3dEnvConfig trait — captures what varies between Phyics3d environments.

Phyics3dEnv[MODEL_DEF: ModelDefLike, CONFIG: Phyics3dEnvConfig] delegates everything to CONFIG:
  - Model setup, integrator choice, reward, termination, GPU model init
  - Obs extraction, reset, enforce limits (delegates to Joints internally)
  - Action application (delegates to Actuators internally)

The config has full access to physics state (qpos, qvel, etc.) for reward
and termination — no hardcoded assumptions about which joints matter.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
)


trait Phyics3dEnvConfig:
    # === Physics ===
    comptime FRAME_SKIP: Int
    comptime MAX_STEPS: Int
    comptime INTEGRATOR_WS_EXTRA: Int  # 0 for RK4/Euler, >0 for ImplicitFast
    # Which fields integrator the facades dispatch on ("rk4" | "euler"), with
    # Newton as the solver. Default "rk4" (9/12 envs); HalfCheetah/Pusher/
    # MetaWorld override to "euler".
    comptime INTEGRATOR: StaticString = "rk4"

    # Refresh forward kinematics AFTER the frame-skip loop, so that reward and
    # observation hooks see xpos/xquat/xipos/site_xpos consistent with the
    # INTEGRATED qpos.
    #
    # Off by default because that matches raw `mj_step`, and therefore
    # Gymnasium: MuJoCo computes FK in mj_step1 and integrates in mj_step2, so
    # after `mj_step(n)` the derived fields still describe the state BEFORE the
    # last substep. Every Gym-derived env here (Ant/Humanoid/Sawyer read xpos,
    # xipos, cfrc_ext) is calibrated against that convention — flipping this
    # globally would shift their rewards by one step.
    #
    # dm_control does the opposite: `Physics._step_with_up_to_date_position_
    # velocity` runs mj_step2 then mj_step1 specifically so "(most of) mjData
    # is in sync with qpos and qvel" when the task reads it. Suite ports must
    # therefore set this True, or every xmat/site-based observation and reward
    # is silently one control step stale.
    comptime SYNC_FK_AFTER_STEP: Bool = False

    # === CPU: Pre-step hook — save any per-env state before physics ===
    @staticmethod
    def pre_step_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        """Save per-env state before physics step.

        The prev_x parameter is a single scalar stored per-env (in the
        metadata region on GPU). Configs use it to store whatever they need
        (e.g., rootx position for velocity computation).
        """
        # Default: memoryless reward — nothing to stash.
        pass

    # ── Model record lists on the CPU hooks ──────────────────────────────
    # `compute_reward_and_done_cpu`, `custom_extract_obs_cpu` and
    # `custom_reset_cpu` receive the four packed model record tensors
    # (`m_bodies`, `m_joints`, `m_geoms`, `m_sites`) alongside the state.
    #
    # They are the host `List`s behind `Model.bodies/joints/geoms/sites`,
    # borrowed (never copied), indexed with the usual column constants from
    # `physics3d.gpu.constants`:
    #
    #     m_bodies[b * MODEL_BODY_SIZE + BODY_IDX_MASS]
    #     m_geoms [g * MODEL_GEOM_SIZE + GEOM_IDX_SIZE_0]
    #
    # Added 2026-07-29: the hooks previously saw `Data` only, so a reward or
    # observation could not read a single model constant. That blocked every
    # mass-weighted sensor (subtreelinvel) and the dm_control tasks that size
    # their reward from the model (reacher/acrobot/ball_in_cup/point_mass read
    # geom_size / site_size; the joint randomizer needs joint ranges).
    #
    # Passing the record lists rather than `Model` itself keeps the hook
    # signatures free of Model's six extra compile-time parameters.

    # === CPU: Unified reward + termination ===
    @staticmethod
    def compute_reward_and_done_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """Compute reward and early termination from full physics state.

        Args:
            d: Fields physics state with qpos, qvel, xpos, etc.
            prev_x: Value saved by pre_step_cpu (e.g., previous x position).
            actions: Clamped action values.
            step_count: Current step count (for truncation checking outside).
            frame_skip: Number of physics substeps per env step.

        Returns:
            (reward, terminated) — terminated is True for early termination
            (NOT truncation — truncation is handled by the generic env).
        """
        ...

    # === CPU: Custom reset hook (called after _reset_state) ===
    @staticmethod
    def custom_reset_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """Custom reset logic (e.g., set initial mocap position, pin goal
        joints). The facade runs the fields FK after this hook, so writes to
        qpos/mocap take effect before the first observation. Default: no-op."""
        pass

    # === CPU: Custom observation extraction (default: use MODEL_DEF.extract_obs) ===
    @staticmethod
    def custom_extract_obs_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """Extract observations from data. Return True if handled, False for default.

        Override for envs that need non-standard observations
        (e.g., hand position + object position instead of qpos/qvel).
        """
        return False

    # === CPU: Custom action application (default: use MODEL_DEF.apply_actions) ===
    @staticmethod
    def custom_apply_actions_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        actions: List[Float64],
    ) -> Bool:
        """Apply actions to data. Return True if handled, False for default.

        Override for envs that need non-standard action application
        (e.g., mocap position control instead of torque motors).
        Default returns False, which causes Phyics3dEnv.step() to call
        MODEL_DEF.apply_actions() as usual.
        """
        return False

    # === CPU: Float getters (can't use Float64 as comptime in traits) ===
    @staticmethod
    def get_timestep() -> Float64:
        ...

    @staticmethod
    def get_reset_noise() -> Float64:
        # Default: no symmetric qpos/qvel jitter; envs that want it
        # override, envs that randomize in custom_reset_cpu do not.
        return 0.0

    # =====================================================================
    # GPU hooks.
    #
    # `pre_step_gpu` and `compute_reward_and_done_gpu` carry INERT DEFAULTS
    # (no-op / zero reward) rather than being abstract, so CPU-only envs need
    # not restate ~90 lines of stub each. That matters for the dm_control
    # suite: its rewards need body quaternions, which the batched hook ABI
    # does not pass (gap G10 in docs/DM_CONTROL_PORT.md), so all ~36 task
    # configs are CPU-only by construction.
    #
    # ANY env wired to a GPU-batched driver MUST override
    # `compute_reward_and_done_gpu`. Inheriting the default there gives a
    # flat-zero reward curve — that is the symptom to look for. All 11
    # Gym-derived configs override it today.
    # =====================================================================

    # === GPU inline: Pre-step hook (per-field tensors; G5) ===
    @always_inline
    @staticmethod
    def pre_step_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        """Save per-env state before physics (GPU inline version).

        Write to meta[env, META_IDX_PREV_X] to persist a value for use in
        compute_reward_and_done_gpu.
        """
        # Default: nothing to stash between steps.
        pass

    # === GPU inline: Unified reward + termination (per-field tensors; G5) ===
    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        ACTION_DIM: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        cfrc_ext: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        curriculum: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """Compute reward and early termination from the per-field GPU state.

        The hook reads exactly the field tensors it needs (joint state, FK
        products, contact forces, CoM velocities, hook metadata, curriculum
        params) — the legacy `[BATCH, STATE_SIZE]` slab + offset ABI died at
        the G5 fields sunset.

        Returns:
            (reward, terminated).
        """
        # Default: NOT IMPLEMENTED — see the section note above.
        return (Scalar[DTYPE](0), False)

    # === GPU: Observation-based termination (for model-based rollouts) ===
    @always_inline
    @staticmethod
    def is_terminal_from_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ) -> Bool:
        """Check termination from observations only (no full state access).

        Used by model-based agents (MBPO) during synthetic rollouts where
        only predicted observations are available. Default: no termination.

        Override for envs with observation-based termination conditions.
        Observation layout matches the env's observation vector.
        """
        return False

    # === GPU inline: Non-zero qpos init after reset (per-field; G5) ===
    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        env: Int,
    ):
        """Apply non-zero initial qpos offsets after noise (default: no-op).

        Override for envs whose initial qpos is non-zero (e.g., Humanoid
        z=1.4 / quat_w=1.0, HumanoidStandup z=0.105). Called by
        _reset_env_gpu after noise has been applied around zero.
        """
        pass

    # === GPU inline: Custom observation extraction (per-field; G5) ===
    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        OBS_DIM: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ) -> Bool:
        """Custom observation extraction (default: False = use model default).

        Override for envs that need non-standard observations (sin/cos
        transforms, body COM positions, etc.).  Return True and write the
        full observation into obs[env, :] to bypass the default
        qpos[obs_qpos_skip:] + qvel[:] extraction.

        Args:
            qpos: Generalized positions, [BATCH_SIZE, NQ].
            qvel: Generalized velocities, [BATCH_SIZE, NV].
            xpos: Body world positions, [BATCH_SIZE, NBODY * 3].
            obs: Output observation buffer to write into.
            env: Environment index.

        Returns:
            True if custom extraction was performed (skip model default).
            False to fall back to model's default extraction.
        """
        return False
