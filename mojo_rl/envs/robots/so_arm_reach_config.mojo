"""One reach-task config, shared by SO-ARM100 and SO-ARM101.

`docs/SO_ARM101_PORT_ASSESSMENT.md` §1 concludes "one env, two model files".
This file is what makes that structural rather than aspirational: both arms
have the same topology (nbody 8 / njnt 6 / nq 6 / nu 6) and — measured — the
same body indices for the end-effector (7) and the appended target (8), so a
single `Phyics3dEnvConfig` parameterised by four numbers serves both.

    observation = [qpos(6), qvel(6), ee_xyz(3), target_xyz(3), ee_to_target(3)]
    action      = 6 joint POSITION TARGETS, in RADIANS
    reward      = tolerance(||ee - target||, (0, TARGET_RADIUS), margin)
    episode     = MAX_STEPS control steps, no early termination

⚠⚠ THE ACTION SPACE IS RADIANS, NOT [-1, 1]. Both models drive `<position>`
servos whose `ctrlrange` is the joint's own range, so `action_low_at(i)` /
`action_high_at(i)` differ per actuator (`Pitch` is [-3.32, 0.174]). A policy
emitting tanh-bounded actions MUST be rescaled by the caller. This is the exact
shape of the defect `c1ebf61e` fixed — an env that advertised one scalar pair
for six differently-ranged actuators — so it is stated here rather than left to
be discovered. `apply_actions` clamps to the per-actuator range, so a policy
that ignores this is silently squashed onto the first radian of travel.

⚠ THE TARGET IS SAMPLED IN THE ARM'S OWN AZIMUTH CONE. `AZ_CENTER` differs
between the two models because their base frames do — SO-100's arm extends
along -y at qpos 0, SO-101's along +x — and both `Rotation`/`shoulder_pan`
ranges are ±1.92 rad, not a full circle. Sampling a full circle would make a
third of every episode physically unreachable and read as "the policy plateaus".

GPU HOOKS LANDED 2026-08-25 — `HAS_GPU_HOOKS = True`, so the batched path
compiles and trains. They were written against
`dm_control/reacher/reacher_config.mojo`, the only other config in the tree
with GPU hooks AND a mocap target, and the note this paragraph replaces was
right that writing them blind is how `ungated_generic_is_uncompiled_code`
happens — so they are gated CPU-vs-GPU rather than merely compiled.

⚠ THE OBSERVATION ORDER IS THE CONTRACT. The batched trainer writes a
checkpoint the single-env eval loads, so `custom_extract_obs_cpu` and
`custom_extract_obs_gpu` must emit the same 21 values in the same order. A
permutation is a policy that works on one device and is nonsense on the other,
with no error anywhere — which is why the two are compared directly rather
than each being checked against a description.

⚠ RESET DOES NOT USE `<keyframe>`. Our parser ignores the section
(`docs/TODDLERBOT_PORT_PLAN.md` §4.6), so SO-100's `home` pose is baked into
`SO_ARM100_HOME` in `so_arm100_config.mojo` and written here by index. When
`<keyframe>` lands, this should read the parsed key and the baked copy should
be DELETED rather than kept alongside it.
"""

from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.math import pi, sqrt, sin, cos

from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
    METADATA_SIZE,
)

from ..phyics3d_env_config import Phyics3dEnvConfig
from ..dm_control.rewards import (
    tolerance,
    SIGMOID_GAUSSIAN,
    DEFAULT_VALUE_AT_MARGIN,
)
from ..dm_control.gpu_reset import reset_seed
from ..dm_control.dtype_math import sin_dt, cos_dt


struct SoArmReachConfig[
    # ⚠ NOT `NMESHV`. `Phyics3dEnvConfig.custom_reset_full_cpu` is a default
    # trait method carrying its own `NMESHV` parameter, and a struct parameter
    # of the same name shadows it — which the compiler rejects outright
    # ("name conflict between parameter 'NMESHV' in the default trait method
    # and a parameter in the struct"), taking both SO-ARM envs down with it.
    # Any parameter named after one of that trait's is a build break waiting
    # for the next hook to be added.
    NMESHVERT: Int,
    EE_BODY: Int,
    TARGET_BODY: Int,
    TIMESTEP: Float64,
    # Centre of the reachable azimuth cone, radians, measured from +x in the
    # world XY plane. See the module docstring.
    AZ_CENTER: Float64,
    # Half-width of that cone. Both arms' base-rotation range is 1.92 rad.
    AZ_HALF: Float64 = 1.92,
    # Target shell, metres from the base origin.
    R_MIN: Float64 = 0.15,
    R_MAX: Float64 = 0.30,
    # Elevation band, radians above the XY plane through the base.
    EL_MIN: Float64 = 0.17,
    EL_MAX: Float64 = 1.22,
    # Height of the base origin, added to every sampled target so nothing is
    # ever generated below the floor.
    BASE_Z: Float64 = 0.05,
    # `tolerance`'s upper bound: inside this the reward is exactly 1.
    TARGET_RADIUS: Float64 = 0.02,
    # ...and the width of the shaped falloff outside it. Unlike dm_control's
    # `reacher` (margin 0, a hard indicator) this one is SHAPED, because a
    # 6-DOF arm reaching a 2 cm ball has far too little chance of stumbling
    # onto a sparse reward.
    REWARD_MARGIN: Float64 = 0.25,
    # Uniform noise on each joint at reset, radians, clipped to joint range.
    RESET_NOISE: Float64 = 0.05,
    MAX_STEPS_P: Int = 500,
    # The reset pose, joint by joint. ⚠ SIX SCALARS RATHER THAN AN ARRAY
    # because a comptime array parameter is not expressible here; the arm
    # modules pass them by name so the call site still reads as a pose.
    #
    # ⚠⚠ THIS EXISTS BECAUSE `<keyframe>` IS UNPARSED. SO-100 ships a `home`
    # key of [0, -1.57, 1.57, 1.57, -1.57, 0] which is what the reference
    # resets to; `qpos0` is all zeros, i.e. the arm fully extended — a
    # DIFFERENT robot in a different posture, and nothing raises. SO-101 has
    # no keyframe at all (measured, nkey 0), so it legitimately passes zeros.
    HOME_0: Float64 = 0.0,
    HOME_1: Float64 = 0.0,
    HOME_2: Float64 = 0.0,
    HOME_3: Float64 = 0.0,
    HOME_4: Float64 = 0.0,
    HOME_5: Float64 = 0.0,
](Phyics3dEnvConfig):
    # === Physics ===
    # Both models declare `timestep="0.002"`. FRAME_SKIP 10 gives a 50 Hz
    # control rate against 500 Hz physics — the rate the real STS3215 bus
    # runs at, and the same 1:10 ratio ToddlerBot's walk env uses.
    comptime FRAME_SKIP: Int = 10
    comptime MAX_STEPS: Int = Self.MAX_STEPS_P
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # Neither model names an integrator, so MuJoCo's Euler default applies.
    # ⚠ `opt.integrator` is one of the fields the layer-1 gate compares, so
    # this is pinned by measurement rather than chosen.
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # The per-episode target is a mocap body — `Model` is not batched,
    # `Data.mocap_pos` is. Same mechanism `reacher` uses.
    comptime USES_MOCAP: Bool = True
    comptime HAS_GPU_HOOKS: Bool = True

    # ⚠⚠ NOT A SIZE HINT. Zero silently disables every mesh contact — both
    # narrow phases guard their mesh branch on `NMESH_VERTS > 0`. The value is
    # the measured convex-hull vertex total, NOT the raw STL count.
    comptime NMESH_VERTS: Int = Self.NMESHVERT

    @staticmethod
    def _home(i: Int) -> Float64:
        """The reset pose, by qpos address.

        ⚠ A CHAIN, NOT AN `InlineArray`. `InlineArray[Float64, 6](a, b, ...)`
        does not construct under Mojo 1.0 (the variadic ctor wants
        `__list_literal__`), and the `fill=` + subscript-assign form runs into
        `feedback_mojo_inlinearray_subscript_is_a_copy`. Six branches read
        worse and are simply correct.
        """
        if i == 0:
            return Self.HOME_0
        if i == 1:
            return Self.HOME_1
        if i == 2:
            return Self.HOME_2
        if i == 3:
            return Self.HOME_3
        if i == 4:
            return Self.HOME_4
        return Self.HOME_5

    # === CPU: Observation ===
    @staticmethod
    def custom_extract_obs_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """qpos, qvel, the end-effector, the target, and the vector between.

        ⚠ `ee_to_target` is redundant with the two absolute positions and is
        included anyway: it is the only term whose scale matches the reward,
        and an arm policy that has to learn the subtraction wastes capacity.
        """
        for i in range(D.NQ):
            obs.append(d.qpos.data[i])
        for i in range(D.NV):
            obs.append(d.qvel.data[i])

        var ex = d.xpos.data[Self.EE_BODY * 3 + 0]
        var ey = d.xpos.data[Self.EE_BODY * 3 + 1]
        var ez = d.xpos.data[Self.EE_BODY * 3 + 2]
        var tx = d.xpos.data[Self.TARGET_BODY * 3 + 0]
        var ty = d.xpos.data[Self.TARGET_BODY * 3 + 1]
        var tz = d.xpos.data[Self.TARGET_BODY * 3 + 2]

        obs.append(ex)
        obs.append(ey)
        obs.append(ez)
        obs.append(tx)
        obs.append(ty)
        obs.append(tz)
        obs.append(tx - ex)
        obs.append(ty - ey)
        obs.append(tz - ez)
        return True

    # === CPU: Reset ===
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """The HOME pose plus small noise, then a fresh target.

        ⚠ The noise is CLIPPED TO EACH JOINT'S RANGE, read from the joint
        table rather than assumed. Every joint here is `limited`, and a reset
        that starts outside the limit hands the solver a violated constraint on
        step 0 — which looks like an unstable model, not a bad reset.
        """
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var base = j * MODEL_JOINT_SIZE
            var adr = Int(m_joints[base + JOINT_IDX_QPOS_ADR])
            if adr < 0 or adr >= D.NQ:
                continue
            var lo = Float64(m_joints[base + JOINT_IDX_RANGE_MIN])
            var hi = Float64(m_joints[base + JOINT_IDX_RANGE_MAX])
            # ⚠ The mocap target's body carries no joint, so `njoint` is 6 for
            # both arms and every `adr` lands inside `home`. The guard is here
            # anyway: a future task fragment with a free-jointed prop would
            # otherwise read past the end of a six-element pose.
            var q0 = Self._home(adr) if adr < 6 else Float64(d.qpos.data[adr])
            var q = q0 + (
                random_float64() * 2.0 - 1.0
            ) * Self.RESET_NOISE
            if lo > -1e9 and q < lo:
                q = lo
            if hi < 1e9 and q > hi:
                q = hi
            d.qpos.data[adr] = Scalar[DTYPE](q)
        for i in range(D.NV):
            d.qvel.data[i] = Scalar[DTYPE](0)

        # Target: uniform in the arm's azimuth cone, in an elevation band, in
        # a radial shell. Not uniform in VOLUME — deliberately, since uniform
        # volume concentrates targets at the outer radius where the arm is
        # least dexterous.
        var az = Self.AZ_CENTER + (random_float64() * 2.0 - 1.0) * Self.AZ_HALF
        var el = Self.EL_MIN + random_float64() * (Self.EL_MAX - Self.EL_MIN)
        var r = Self.R_MIN + random_float64() * (Self.R_MAX - Self.R_MIN)
        d.mocap_pos.data[Self.TARGET_BODY * 3 + 0] = Scalar[DTYPE](
            r * cos(el) * cos(az)
        )
        d.mocap_pos.data[Self.TARGET_BODY * 3 + 1] = Scalar[DTYPE](
            r * cos(el) * sin(az)
        )
        d.mocap_pos.data[Self.TARGET_BODY * 3 + 2] = Scalar[DTYPE](
            Self.BASE_Z + r * sin(el)
        )
        # Identity orientation, stored [x, y, z, w].
        d.mocap_quat.data[Self.TARGET_BODY * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[Self.TARGET_BODY * 4 + 1] = Scalar[DTYPE](0)
        d.mocap_quat.data[Self.TARGET_BODY * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[Self.TARGET_BODY * 4 + 3] = Scalar[DTYPE](1)

    # === CPU: Reward ===
    @staticmethod
    def compute_reward_and_done_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """Shaped `tolerance` on the end-effector-to-target distance.

        Never terminates early — like every dm_control task, and unlike the
        MuJoCo-gym arms. An arm has no "unhealthy" state to fall into.
        """
        var dx = Float64(
            d.xpos.data[Self.TARGET_BODY * 3 + 0]
            - d.xpos.data[Self.EE_BODY * 3 + 0]
        )
        var dy = Float64(
            d.xpos.data[Self.TARGET_BODY * 3 + 1]
            - d.xpos.data[Self.EE_BODY * 3 + 1]
        )
        var dz = Float64(
            d.xpos.data[Self.TARGET_BODY * 3 + 2]
            - d.xpos.data[Self.EE_BODY * 3 + 2]
        )
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        var r = tolerance(
            dist, 0.0, Self.TARGET_RADIUS, Self.REWARD_MARGIN
        )
        return (Scalar[DTYPE](r), False)


    # =====================================================================
    # GPU hooks — the batched twin of the three CPU hooks above.
    #
    # ⚠⚠ `Phyics3dBatchedEnv` carries a `comptime assert HAS_GPU_HOOKS`
    # precisely because a CPU-only config wired to it COMPILES AND RUNS,
    # training against a flat-zero reward. Flip the flag in the same
    # commit as the hooks, never ahead of them.
    #
    # Modelled on `dm_control/reacher/reacher_config.mojo`, the only other
    # config with GPU hooks AND a mocap target — the same task shape, at
    # 2 DoF and a 2D target instead of 6 and 3D.
    # =====================================================================

    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        OBS_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_F, MODEL_BODY_SIZE), MutAnyOrigin
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
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
    ) -> Bool:
        """qpos, qvel, the end-effector, the target, and the vector between.

        ⚠⚠ MUST STAY BYTE-FOR-BYTE IN THE SAME ORDER AS
        `custom_extract_obs_cpu`. The batched trainer writes a checkpoint the
        single-env eval loads, so a permuted observation here is a policy that
        works on the GPU and is nonsense on the CPU, with no error anywhere.
        `tests/robots/test_so_arm101_obs_cpu_vs_gpu.mojo` is what keeps the two
        honest.
        """
        for i in range(NQ_F):
            obs[env, i] = qpos[env, i]
        for i in range(NV_F):
            obs[env, NQ_F + i] = qvel[env, i]

        var ex = xpos[env, Self.EE_BODY * 3 + 0]
        var ey = xpos[env, Self.EE_BODY * 3 + 1]
        var ez = xpos[env, Self.EE_BODY * 3 + 2]
        var tx = xpos[env, Self.TARGET_BODY * 3 + 0]
        var ty = xpos[env, Self.TARGET_BODY * 3 + 1]
        var tz = xpos[env, Self.TARGET_BODY * 3 + 2]

        var b = NQ_F + NV_F
        obs[env, b + 0] = ex
        obs[env, b + 1] = ey
        obs[env, b + 2] = ez
        obs[env, b + 3] = tx
        obs[env, b + 4] = ty
        obs[env, b + 5] = tz
        obs[env, b + 6] = tx - ex
        obs[env, b + 7] = ty - ey
        obs[env, b + 8] = tz - ez
        return True

    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        ACTION_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_F, MODEL_BODY_SIZE), MutAnyOrigin
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
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
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
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """Shaped `tolerance` on end-effector-to-target distance.

        The GPU twin of `compute_reward_and_done_cpu`, same constants, same
        never-terminates rule. ⚠ `tolerance`'s SIGMOID and value-at-margin are
        spelled explicitly here because the GPU form takes them as parameters
        while the CPU call relies on their defaults — a silent mismatch would
        be a different reward curve on each device.
        """
        # ⚠ `rebind[Scalar[DTYPE]]`, not `Scalar[DTYPE](...)`. A LayoutTensor
        # subscript yields a layout-parameterised `element_type`; the Scalar
        # constructor has no conversion from it, `rebind` reinterprets it.
        # Same idiom as `quadruped_escape_config.mojo:497`.
        var dx = rebind[Scalar[DTYPE]](
            xpos[env, Self.TARGET_BODY * 3 + 0]
        ) - rebind[Scalar[DTYPE]](xpos[env, Self.EE_BODY * 3 + 0])
        var dy = rebind[Scalar[DTYPE]](
            xpos[env, Self.TARGET_BODY * 3 + 1]
        ) - rebind[Scalar[DTYPE]](xpos[env, Self.EE_BODY * 3 + 1])
        var dz = rebind[Scalar[DTYPE]](
            xpos[env, Self.TARGET_BODY * 3 + 2]
        ) - rebind[Scalar[DTYPE]](xpos[env, Self.EE_BODY * 3 + 2])
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        var r = tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE](
            dist,
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](Self.TARGET_RADIUS),
            Scalar[DTYPE](Self.REWARD_MARGIN),
        )
        return (r, False)

    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NJOINT_F: Int,
        NV_F: Int,
        NBODY_M: Int,
        NGEOM_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DTYPE, Layout.row_major(NJOINT_F, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        mocap_pos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_M * 3), MutAnyOrigin
        ],
        mocap_quat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_M * 4), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_M, MODEL_BODY_SIZE), MutAnyOrigin
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
        """HOME pose plus noise, then a fresh target — the GPU twin of
        `custom_reset_cpu`.

        ⚠ The noise is CLIPPED TO EACH JOINT'S RANGE, read from the joint
        table rather than assumed, for the same reason the CPU hook does it: a
        reset that starts outside a limit hands the solver a violated
        constraint on step 0, which reads as an unstable model rather than a
        bad reset.

        ⚠ ONE PHILOX DRAW PER VALUE, `[0]` only. Wasteful of lanes and
        deliberately so — the alternative is an index into a 4-lane block that
        has to stay in step with a future edit, and this runs once per episode.
        """
        # Joints: home + uniform noise, clipped to the model's own limits.
        # ⚠⚠ `joints[j, FIELD]`, NOT `joints[j * MODEL_JOINT_SIZE + FIELD]`.
        # The CPU hook reads a FLAT `List`; here `joints` is a 2-D
        # `LayoutTensor(NJOINT_F, MODEL_JOINT_SIZE)` and a single-integer
        # subscript selects a ROW, not an element. Same trap as the flat-index
        # TileTensor note; `gpu_reset.mojo:226` is the correct idiom.
        var rng = PhiloxRandom(seed=reset_seed(env, seed), offset=0)
        for j in range(NJOINT_F):
            var adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            if adr < 0 or adr >= NQ_F:
                continue
            var q0 = Scalar[DTYPE](Self._home(adr)) if adr < 6 else rebind[
                Scalar[DTYPE]
            ](qpos[env, adr])
            var u = rng.step_uniform()
            var q = q0 + (
                Scalar[DTYPE](u[0]) * Scalar[DTYPE](2.0) - Scalar[DTYPE](1.0)
            ) * Scalar[DTYPE](Self.RESET_NOISE)
            var lo = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MIN])
            var hi = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MAX])
            if lo > Scalar[DTYPE](-1e9) and q < lo:
                q = lo
            if hi < Scalar[DTYPE](1e9) and q > hi:
                q = hi
            qpos[env, adr] = q
        for i in range(NV_F):
            qvel[env, i] = Scalar[DTYPE](0)

        # Target: the same azimuth cone / elevation band / radial shell the CPU
        # hook draws from. NOT uniform in volume, deliberately — uniform volume
        # concentrates targets at the outer radius where the arm is least
        # dexterous.
        var ua = rng.step_uniform()
        var ue = rng.step_uniform()
        var ur = rng.step_uniform()
        var az = Scalar[DTYPE](Self.AZ_CENTER) + (
            Scalar[DTYPE](ua[0]) * Scalar[DTYPE](2.0) - Scalar[DTYPE](1.0)
        ) * Scalar[DTYPE](Self.AZ_HALF)
        var el = Scalar[DTYPE](Self.EL_MIN) + Scalar[DTYPE](ue[0]) * Scalar[
            DTYPE
        ](Self.EL_MAX - Self.EL_MIN)
        var r = Scalar[DTYPE](Self.R_MIN) + Scalar[DTYPE](ur[0]) * Scalar[
            DTYPE
        ](Self.R_MAX - Self.R_MIN)

        # ⚠ `sin_dt`/`cos_dt`, not `sin`/`cos`: the plain ones do not resolve
        # in a GPU kernel ("lacking evidence to prove correctness").
        var ce = cos_dt[DTYPE](el)
        mocap_pos[env, Self.TARGET_BODY * 3 + 0] = r * ce * cos_dt[DTYPE](az)
        mocap_pos[env, Self.TARGET_BODY * 3 + 1] = r * ce * sin_dt[DTYPE](az)
        mocap_pos[env, Self.TARGET_BODY * 3 + 2] = Scalar[DTYPE](
            Self.BASE_Z
        ) + r * sin_dt[DTYPE](el)
        # Identity orientation, [x, y, z, w].
        mocap_quat[env, Self.TARGET_BODY * 4 + 0] = Scalar[DTYPE](0)
        mocap_quat[env, Self.TARGET_BODY * 4 + 1] = Scalar[DTYPE](0)
        mocap_quat[env, Self.TARGET_BODY * 4 + 2] = Scalar[DTYPE](0)
        mocap_quat[env, Self.TARGET_BODY * 4 + 3] = Scalar[DTYPE](1)

    @staticmethod
    def get_timestep() -> Float64:
        return Self.TIMESTEP
