"""Phyics3dEnvConfig trait — captures what varies between Phyics3d environments.

Phyics3dEnv[MODEL_DEF: ModelDefLike, CONFIG: Phyics3dEnvConfig] delegates everything to CONFIG:
  - Model setup, integrator choice, reward, termination, GPU model init
  - Obs extraction, reset, enforce limits (delegates to Joints internally)
  - Action application (delegates to Actuators internally)

The config has full access to physics state (qpos, qvel, etc.) for reward
and termination — no hardcoded assumptions about which joints matter.
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data, Model, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_TENDON_SIZE,
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

    # Convex-hull vertex capacity for COLLIDABLE mesh geoms.
    #
    # ⚠⚠ ZERO MEANS MESH GEOMS DO NOT COLLIDE — it is not a "size hint". Both
    # narrow phases guard their mesh branch with `comptime if NMESH_VERTS > 0`
    # and otherwise emit no contact for the pair. This was hardcoded to 0 in
    # `Phyics3dEnv` for every environment, so the entire mesh collider was
    # unreachable from the env layer while the fixtures that build `Model`
    # directly kept passing.
    #
    # Default 0 is CORRECT for almost everything and must stay: of the ported
    # models only sawyer has collidable meshes (2 geoms — `l6` and
    # `eGripperBase`). Dog has 162 mesh geoms and every one is visual
    # (`contype=0 conaffinity=0`); the rest of the suite has no meshes at all.
    # A nonzero value compiles the whole GJK/EPA mesh branch into that model's
    # collision kernel, which costs compile time and registers and risks the
    # Metal wide-InlineArray cliff — so pay it only where a mesh can collide.
    #
    # It cannot be derived from the XML at comptime (mesh assets need file I/O
    # and the comptime parser cannot read an STL), so it is hand-supplied like
    # MAX_CONTACTS. `fields_build` RAISES if it is too small, quoting the
    # required count — an undersized value can no longer truncate silently.
    comptime NMESH_VERTS: Int = 0

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

    # Run `mj_rnePostConstraint` inside every substep, filling `Data.cacc`
    # and `Data.cfrc_int` (dynamics/rne_post.mojo). Needed ONLY by the
    # acceleration-stage sensors — `accelerometer`, `force`, `torque` — and
    # off by default because it is pure overhead for every other model.
    #
    # The values land at MuJoCo's `mj_sensorAcc` point: the state BEFORE the
    # substep's integration, with that substep's constrained qacc. After the
    # frame-skip loop they therefore describe the second-to-last state, which
    # is exactly what dm_control observes — `mj_step1` refreshes position and
    # velocity sensors at the new state but leaves the acceleration stage
    # alone. Do NOT "fix" this to agree with SYNC_FK_AFTER_STEP.
    #
    # Euler only: the RK4 integrator would need the hook inside its base
    # stage, and no in-scope model wants both.
    comptime RNE_POST: Bool = False

    # Does this config implement the GPU hooks (`compute_reward_and_done_gpu`
    # and, where the model default is not enough, `custom_extract_obs_gpu`)?
    #
    # ⚠ THIS EXISTS TO MAKE A SILENT FAILURE LOUD. Those hooks carry INERT
    # DEFAULTS — zero reward, "use the model default obs" — rather than being
    # abstract, so that CPU-only configs need not restate ~90 lines of stub
    # each. The cost is that wiring a CPU-only config to `Phyics3dBatchedEnv`
    # COMPILES AND RUNS, and trains against a flat-zero reward curve. That cost
    # was paid for real by the dm_control suite, whose ~36 task configs were
    # CPU-only by construction for months (gap G10).
    #
    # Mojo cannot ask "did you override this method?", so the config declares
    # it and `Phyics3dBatchedEnv` asserts it at compile time. Flipping this to
    # True without actually implementing the hooks reinstates exactly the trap
    # it closes — so flip it in the same commit as the hooks, never ahead.
    comptime HAS_GPU_HOOKS: Bool = False

    # Does this config drive a MOCAP body (a per-episode target parked on a
    # `<body mocap="true">`, the G4 workaround)? reacher, finger, fish, swimmer,
    # manipulator, stacker and SawyerReach all do.
    #
    # ⚠ WHY A DECLARATION AND NOT A MODEL QUERY. `is_mocap` is parsed by the
    # RUNTIME parser into `Model.bodies[.., BODY_IDX_MOCAP]`; the COMPTIME
    # parser does not carry it (`physics3d has TWO MJCF parsers`). Making the
    # batched env comptime-gate on it therefore needs either a declaration or
    # new comptime-parser work, and the declaration is the smaller change.
    #
    # ⚠ FORGETTING IT IS NOT SILENT. `Phyics3dBatchedEnv.__init__` reads the
    # built model and RAISES if any body is mocap-flagged while this is False —
    # the failure mode it prevents is a target frozen at its XML pose, i.e. a
    # silently easier task, which no gate would flag as an error.
    comptime USES_MOCAP: Bool = False

    # Raise the free root in 1 cm steps at reset until nothing is touching,
    # after `custom_reset_cpu` has set the orientation
    # (`Phyics3dEnv._find_non_contacting_height`). dm_control's quadruped is
    # the only user: it draws a random orientation per episode, so a fixed
    # spawn height would sometimes start the robot inside the floor.
    comptime RESET_FIND_HEIGHT: Bool = False

    # Does this config drive the actuators itself on the GPU path, via
    # `custom_apply_actions_gpu` below?
    #
    # ⚠ THE GPU TWIN CANNOT BE A `return False` DEFAULT LIKE THE CPU ONE.
    # `Phyics3dEnv` calls `custom_apply_actions_cpu` and branches on its return
    # value at runtime; the batched path has to choose between two KERNEL
    # LAUNCHES, so the choice must be comptime. Hence a declaration, checked
    # the same way `HAS_GPU_HOOKS` is.
    #
    # Setting this True and not implementing the hook gives a permanently zero
    # `qfrc` — every actuator dead, which trains to a flat curve rather than
    # crashing. Flip it in the same commit as the hook.
    comptime HAS_CUSTOM_ACTUATION_GPU: Bool = False

    # === CPU: Pre-step hook — save any per-env state before physics ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
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
    # `compute_reward_and_done_cpu`, `custom_extract_obs_cpu`,
    # `custom_reset_cpu` and `custom_apply_actions_cpu` receive the packed
    # model record tensors alongside the state.
    #
    # They are the host `List`s behind `Model.bodies/joints/geoms/sites`
    # (plus `Model.tendons` on the two action-side hooks), borrowed (never
    # copied), indexed with the usual column constants from
    # `physics3d.gpu.constants`:
    #
    #     m_bodies [b * MODEL_BODY_SIZE   + BODY_IDX_MASS]
    #     m_geoms  [g * MODEL_GEOM_SIZE   + GEOM_IDX_SIZE_0]
    #     m_tendons[t * MODEL_TENDON_SIZE + TENDON_IDX_COEF_0]
    #
    # Added 2026-07-29: the hooks previously saw `Data` only, so a reward or
    # observation could not read a single model constant. That blocked every
    # mass-weighted sensor (subtreelinvel) and the dm_control tasks that size
    # their reward from the model (reacher/acrobot/ball_in_cup/point_mass read
    # geom_size / site_size; the joint randomizer needs joint ranges).
    #
    # Passing the record lists rather than `Model` itself keeps the hook
    # signatures free of Model's six extra compile-time parameters.
    #
    # `m_tendons` (2026-07-31) is on `custom_apply_actions_cpu` and on the new
    # `custom_reset_model_cpu` only, not on the reward/obs hooks, because the
    # only thing that reads it is transmission: a fixed tendon's per-joint
    # `coef` IS its actuator moment arm (MuJoCo's `wrap_prm`), so a config that
    # drives the DOFs itself needs the coefs, and a config that randomizes the
    # control-to-joint mixing per episode needs to WRITE them.

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
        d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
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
            m_bodies: Flat body records of the model.
            m_joints: Flat joint records of the model.
            m_geoms: Flat geom records of the model.
            m_sites: Flat site records of the model.
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
        mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """Custom reset logic (e.g., set initial mocap position, pin goal
        joints). The facade runs the fields FK after this hook, so writes to
        qpos/mocap take effect before the first observation. Default: no-op."""
        pass

    # === CPU: Reset hook with the FULL model (kinematics + collision) ===
    @staticmethod
    def custom_reset_full_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        NGEOM: Int,
        NEQ: Int,
        NTEN: Int,
        NSITE: Int,
        NEXCL: Int,
        NMESHV: Int,
        NPAIR: Int,
        MAX_CONTACTS: Int,
    ](
        mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
        mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    ) raises:
        """Reset logic that needs the whole `Model`. Default: no-op.

        ⚠ WHY THIS EXISTS ALONGSIDE `custom_reset_cpu`. That hook is handed the
        record LISTS (`bodies`/`joints`/`geoms`/`sites`), which is enough to
        read a joint range or write a mocap pose and NOT enough to run forward
        kinematics, build a Jacobian or detect contacts — all of which take
        `Model` itself. dm_control's manipulation reset is exactly that shape:
        `ToolCenterPointInitializer` solves site IK and re-runs the narrow
        phase to reject colliding poses. Before this hook it could not be
        reached from a config at all, so `reach_site_features` reset to qpos0
        — a 55-contact pose.

        ⚠ RUNS AFTER `custom_reset_model_cpu` AND `custom_reset_cpu`, so it
        sees whatever they wrote. That ordering is load-bearing for
        manipulation: `set_grasp` closes the fingers in the state hook, and the
        IK here restores ONLY the arm joints on a rejected sample precisely so
        that the grasp survives.

        ⚠ IT MAY LEAVE `d` MID-KINEMATICS. The facade re-runs `_fields_fk` and
        `_fields_vel` after this returns, so a hook that moved `qpos` need not
        refresh them itself — but a hook that CACHED an FK product locally
        across the call is reading a stale one.

        `raises` because the routines it exists to call do (mesh collision,
        `detect_contacts`). The default body cannot raise; overriders may.

        Cf. `RESET_FIND_HEIGHT`, which solved the same problem for one env by
        putting a comptime flag on the config and the code on `Phyics3dEnv`.
        That does not generalise — the next env needs different code, not the
        same code behind a second flag.
        """
        pass

    # === CPU: Per-episode MODEL randomization (called before custom_reset_cpu) ===
    @staticmethod
    def custom_reset_model_cpu[
        DTYPE: DType,
    ](
        mut m_bodies: List[Scalar[DTYPE]],
        mut m_joints: List[Scalar[DTYPE]],
        mut m_geoms: List[Scalar[DTYPE]],
        mut m_sites: List[Scalar[DTYPE]],
        mut m_tendons: List[Scalar[DTYPE]],
    ):
        """Randomize MODEL constants per episode. Default: no-op.

        Distinct from `custom_reset_cpu`, which randomizes STATE, and it runs
        first so the state hook reads whatever this wrote. dm_control does the
        two together in `initialize_episode` (point_mass `hard` randomizes the
        joints, then the tendon mixing matrix); the split here is only so that
        the ~16 configs which touch state alone need not restate a model
        signature they never use.

        ⚠ Three things this hook does NOT do, all of which it would have to if
        the randomized quantity were anything but a transmission coefficient:

        1. It does not re-upload to the device. These are the HOST lists; a
           GPU-batched env would keep stepping the stale copy. Nothing but
           `Phyics3dEnv._reset_state` calls this hook today, so that is a
           real limit rather than a latent bug — but it is why the name says
           `_cpu`, and it is also why a batched driver cannot simply start
           calling it: `Model` is SHARED and unbatched, so per-episode model
           randomization there would apply one env's draw to all of them.
        2. It does not recompute anything DERIVED from the model. Masses feed
           the CRBA mass matrix; tendon coefs feed `tendon_invweight0`
           (`J M^-1 J^T` at qpos0, the limit row's diagApprox). Writing a coef
           here leaves that constant describing the OLD tendon. Harmless for
           point_mass, whose tendons carry no limit and no equality — check
           before randomizing a tendon that has either.
        3. It does not touch the COMPTIME actuator tables (`_acd.motor_trn_*`),
           which is where `MODEL_DEF.apply_actions` reads transmission from.
           A config that randomizes coefs must therefore also drive the DOFs
           itself via `custom_apply_actions_cpu` — the comptime path cannot
           see runtime writes, and would silently keep the XML coefficients.
        """
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
        d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """Extract observations from data. Return True if handled, False for default.

        Override for envs that need non-standard observations
        (e.g., hand position + object position instead of qpos/qvel).

        `act` is MuJoCo's `d->act` — the actuator ACTIVATION state, one scalar
        per activation variable, empty-but-length-1 on the models that have
        none (`MODEL_DEF.NA == 0`, which is all of them but quadruped). It is
        here because dm_control's quadruped puts `data.act` inside its
        `egocentric_state` block, i.e. in the MIDDLE of the observation, so
        the env cannot append it after the fact. See `Phyics3dEnv.act` for why
        the activation lives on the env rather than in `Data`.
        """
        return False

    # === CPU: Custom action application (default: use MODEL_DEF.apply_actions) ===
    @staticmethod
    def custom_apply_actions_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        m_tendons: List[Scalar[DTYPE]],
        m_actuators: List[Scalar[DTYPE]],
        m_act_tendons: List[Scalar[DTYPE]],
        actions: List[Float64],
    ) -> Bool:
        """Apply actions to data. Return True if handled, False for default.

        Override for envs that need non-standard action application
        (e.g., mocap position control instead of torque motors, or a
        transmission whose coefficients are randomized per episode and so
        cannot live in the shared actuator records).

        ⚠ `m_actuators` / `m_act_tendons` ARE `SpecFields`' PACKED RECORDS,
        added 2026-08-15 with phase 1a.4. Index them
        `i * MODEL_ACTUATOR_SIZE + ACT_IDX_*` and
        `t * MODEL_ACT_TENDON_SIZE + ACTTEN_IDX_*`. They are here for the same
        reason `m_joints` and `m_tendons` are: an override that reimplements
        actuation needs the gains and ranges, and the alternative was reading
        them off the comptime `_acd`, which a hook cannot see once the data is
        runtime.
        Default returns False, which causes Phyics3dEnv.step() to call
        MODEL_DEF.apply_actions() as usual.

        Returning True suppresses `MODEL_DEF.apply_actions` ENTIRELY, including
        the `qfrc` zeroing it opens with — an override that sums several
        transmissions onto one DOF must zero `d.qfrc` itself, or each control
        step adds to the previous one instead of replacing it.

        Note the call is ONCE PER CONTROL STEP, outside the frame-skip loop
        (see `Phyics3dEnv.step`), whereas `MODEL_DEF.apply_actions` runs every
        substep. Identical for a `<motor>`, whose force `gear*ctrl` is constant
        across the step; NOT identical for a state-dependent force law.
        """
        return False

    # === GPU inline: Custom action application (per-lane; G4 workaround) ===
    @always_inline
    @staticmethod
    def custom_apply_actions_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        NTENDON_F: Int,
        ACTION_DIM: Int,
        NA_F: Int,
        NACT_F: Int,
    ](
        qfrc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        tendons: LayoutTensor[
            DTYPE, Layout.row_major(NTENDON_F, MODEL_TENDON_SIZE), MutAnyOrigin
        ],
        acts: LayoutTensor[
            DTYPE, Layout.row_major(NACT_F * MODEL_ACTUATOR_SIZE), MutAnyOrigin
        ],
        act_tendons: LayoutTensor[
            DTYPE,
            Layout.row_major(NTENDON_F * MODEL_ACT_TENDON_SIZE),
            MutAnyOrigin,
        ],
        env: Int,
    ):
        """The batched twin of `custom_apply_actions_cpu`, one lane.

        Reached only when `HAS_CUSTOM_ACTUATION_GPU` is True, in which case it
        REPLACES `MODEL_DEF.apply_actions_kernel_gpu` entirely — including the
        `qfrc` zeroing that opens it. Zero `qfrc[env, :]` yourself.

        ⚠ UNLIKE THE CPU TWIN, THIS RUNS ONCE PER SUBSTEP. The batched env
        calls it at the top of every frame-skip iteration, exactly where it
        calls the model default, so a state-dependent force law is correct
        here where the CPU hook's once-per-control-step cadence would not be.
        `Phyics3dEnv` is the one that needs fixing if that ever matters.

        `meta` is here for the TASK_PARAM slots — a per-episode model
        parameter that cannot live in the shared `Model` (see
        `physics3d/gpu/constants.META_IDX_TASK_PARAM_0`). `tendons` and
        `joints` are the RUNTIME records, so a transmission read from them
        follows the model rather than the comptime tables.

        ⚠ `acts` / `act_tendons` ARE `SpecFields`' PACKED RECORDS, added
        2026-08-15 with phase 1a.4 — the same operands
        `apply_actions_kernel_gpu` reads, so an override and the default read
        one source. FLAT, addressed `i * MODEL_ACTUATOR_SIZE + ACT_IDX_*`.
        """
        pass

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
    # not restate ~90 lines of stub each.
    #
    # ⚠ `xquat` and `xvel` were added 2026-08-06 (G10 step 2, see
    # docs/DM_CONTROL_GPU_TRAINING_G10.md) — they are what the dm_control suite
    # needs and what its facades cite as the reason for being CPU-only.
    # `xquat` gives `xmat` on demand via `kinematics/xmat.xmat_elem_gpu` (no
    # NBODY*9 tensor); `xvel` is what `subtree_linvel` consumes. The Gym-derived
    # configs take both and ignore them.
    #
    # Operand budget: binding these brought the batched env's `extract_kernel`
    # to 15 of the measured Metal cliff of 28 (29 = JIT abort). The remaining
    # suite quantities (site_xpos, subtree_com, cacc, cfrc_int, contacts) fit
    # under it too — add plain operands, do not pack.
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
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
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
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        site_xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            DTYPE,
            Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE),
            MutAnyOrigin,
        ],
        sites: LayoutTensor[
            DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
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
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
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
        NJOINT: Int,
        NV: Int,
        NBODY: Int,
        NGEOM_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        mocap_pos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        mocap_quat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """Apply non-zero initial qpos offsets after noise (default: no-op).

        Override for envs whose initial qpos is non-zero (e.g., Humanoid
        z=1.4 / quat_w=1.0, HumanoidStandup z=0.105). Called by
        _reset_env_gpu after noise has been applied around zero.

        ⚠ MISNOMER, kept for continuity: it writes **qvel too**. `joints`,
        `seed` and `qvel` were all added 2026-08-06 (G10 step 4). Without them
        the hook could express a fixed qpos offset and nothing else — but the
        suite resets are real distributions over the FULL state: walker uses
        `randomizers.randomize_limited_and_rotational_joints` (needs per-joint
        RANGES + a draw) and cartpole draws Gaussian qpos AND qvel. The CPU
        counterpart `custom_reset_cpu` gets all of `Data`; this is the GPU
        hook's equivalent reach. The Gym configs ignore the three new args.

        `seed` is the same value `MODEL_DEF.reset_env_gpu` receives; derive an
        independent stream from it rather than reusing its exact Philox key, or
        the joint angles correlate with the reset noise. The shared helper
        `dm_control/gpu_reset.randomize_limited_and_rotational_joints_gpu`
        does this; prefer it over open-coding a draw.
        """
        pass

    # === GPU inline: initial ACTUATOR ACTIVATION (2026-08-12) ===
    @always_inline
    @staticmethod
    def init_act_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NA_F: Int,
    ](
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """Draw the episode's initial `d->act` (default: leave it at zero).

        ⚠ THE CALLER HAS ALREADY ZEROED `act[env, :]` — that is `mj_resetData`
        and it is NOT this hook's job, so a config that does not override this
        still gets correct MuJoCo reset semantics. Override only when the
        reference's `initialize_episode` draws a NON-ZERO activation.

        ⚠⚠ WHY THIS IS A SEPARATE HOOK RATHER THAN AN ARGUMENT TO
        `init_qpos_gpu`: that signature is implemented by ~20 configs, and
        widening it would touch every one of them to express something only
        dog needs. Cost of the split: two hooks fire per reset instead of one.

        Only relevant to actuators with a `dyntype` — a plain `<motor>` never
        reads `act`. dm_control's dog is the in-scope model that draws one
        (`act[i] = uniform(*ctrlrange[i])` for all 38 `dyntype="filter"`
        actuators, whose force IS `gainprm[0] * act`, so a zero activation
        makes the task materially easier).

        `seed` is the reset seed; derive an independent Philox stream from it
        rather than reusing a key another draw already consumed.
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
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
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
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        site_xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            DTYPE,
            Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE),
            MutAnyOrigin,
        ],
        sites: LayoutTensor[
            DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
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
            xquat: Body world orientations [x,y,z,w], [BATCH_SIZE, NBODY * 4].
                Use `kinematics/xmat.xmat_elem_gpu` for MuJoCo `xmat` entries.
            xvel: Body world linear velocities, [BATCH_SIZE, NBODY * 3].
            bodies: Model body records, [NBODY, MODEL_BODY_SIZE].
            site_xpos: Site world positions, [BATCH_SIZE, SITE_DIM].
            contacts: Contact records, [BATCH_SIZE, MC_F * CONTACT_SIZE].
            sites: Model site records, [NSITE_F, MODEL_SITE_SIZE].
            geoms: Model geom records, [NGEOM_F, MODEL_GEOM_SIZE].
            meta: Per-env metadata slots, [BATCH_SIZE, METADATA_SIZE].
            obs: Output observation buffer to write into.
            xipos: Body COM world positions, [BATCH_SIZE, NBODY * 3].
            xangvel: Body world angular velocities, [BATCH_SIZE, NBODY * 3].
            cvel: Body COM-frame spatial velocities, [BATCH_SIZE, NBODY * 6].
            cacc: Body COM-frame spatial accelerations, [BATCH_SIZE, NBODY * 6].
            cfrc_int: Body COM-frame interaction forces, [BATCH_SIZE, NBODY * 6].
            subtree_com: Subtree COM positions, [BATCH_SIZE, NBODY * 3].
            site_xpos_acc: `site_xpos` as it stood when `cacc`/`cfrc_int` were
                written. Acceleration-stage sensors must read this, not the
                post-integration `site_xpos` (see defect 19 in `fields/data`).
            xquat_acc: `xquat` at the same instant as `site_xpos_acc`.
            act: Actuator activations, [BATCH_SIZE, NA_F].
            env: Environment index.

        Returns:
            True if custom extraction was performed (skip model default).
            False to fall back to model's default extraction.
        """
        return False
