"""`dm_control` `manipulation/reach_site_features` task config.

Port of `manipulation/reach.py::Reach` with `use_site=True`, the first of the
13 Phase 7 `_features` tasks and the only one with no prop at all.

    observation = target_position(3) + arm joints_pos(12) + arm joints_torque(6)
                + arm joints_vel(6) + hand joints_pos(3) + hand joints_vel(3)
                + pinch_site_pos(3) + pinch_site_rmat(9)              (45)
    reward      = tolerance(|pinchsite - target_site|, (0, .05), margin .05)
    episode     = 250 control steps (10 s / .04 s), no early termination
    action      = 9 <velocity> actuators (6 arm, 3 finger)

⚠ THE OBSERVATION ORDER IS `observation_spec()`'s, WHICH IS NOT DECLARATION
ORDER. `target_position` is a TASK observable and composer emits the task's
before the entities'; within an entity the order is the `ObservableNames`
listing. Measured off the real env rather than read off the source, because
composer assembles it from three dicts and the flattening order is what a
policy sees:

    target_position                       (1, 3)
    jaco_arm/joints_pos                   (1, 6, 2)
    jaco_arm/joints_torque                (1, 6)
    jaco_arm/joints_vel                   (1, 6)
    jaco_arm/jaco_hand/joints_pos         (1, 3)
    jaco_arm/jaco_hand/joints_vel         (1, 3)
    jaco_arm/jaco_hand/pinch_site_pos     (1, 3)
    jaco_arm/jaco_hand/pinch_site_rmat    (1, 9)

TERM BY TERM, with the trap each one carries.

`target_position` is `MJCFFeature('pos', target_site)` — the site's MODEL
`pos`, not its `xpos`. The two agree here only because the target site hangs
off the worldbody; reading `site_xpos` would be a different quantity on any
task whose target rides a prop, which is every OTHER reach variant.

`joints_pos` for the ARM is `vstack([sin, cos]).T`, i.e. interleaved
[sin0, cos0, sin1, cos1, ...] — SINE FIRST. Jaco's joints 1, 4, 5 and 6 are
unlimited, so the reference bounds the observation this way rather than
emitting an angle that grows without limit. The HAND's `joints_pos` is the
raw angle (`base.JointsObservables`), because its fingers are limited to
[0.15, 1.35]. Two observables of the same name with different content.

⚠⚠ `joints_torque` IS AN ACCELERATION-STAGE SENSOR, and three separate things
have to line up for it:

  1. `<torque site=...>` needs `mj_rnePostConstraint`, hence `RNE_POST = True`
     and the Euler integrator (`Phyics3dEnv` asserts that pairing).
  2. It must read `site_xpos_acc`/`xquat_acc` — the FK snapshot from the
     instant `cfrc_int` was written — NOT the live post-integration products.
     That is defect 19; see `physics3d/fields/data.mojo`.
  3. The reference PROJECTS the 3-axis sensor onto the joint's rotation axis
     (`np.einsum('ij,ij->i', torques.reshape(-1, 3), joint_axes)`) and then
     the FTT corruptor applies `sign(x) * log1p(|x|)`.

⚠ THIS MODEL CANNOT DISCRIMINATE THE PROJECTION. All six Jaco arm joints have
`axis = (0, 0, 1)`, so the dot product degenerates to "take the z component"
and an implementation that hardcoded that would pass every gate here. The
general form is written anyway, because the next arm to arrive will not be
axis-aligned and the failure would then be silent. Same shape as
`manipulator`'s arm joint ORDER, which a symmetric pose also cannot see.

⚠ THE TORQUE SENSOR SITE IS THE JOINT'S PARENT BODY, and the site/body pairing
is NOT the identity: `joint_6_site` is site 9 on body 8 while sites 3..7 sit on
bodies 3..7, because `wristsite` is declared between them. Indices are listed
explicitly below rather than derived from a stride.

`pinch_site_rmat` is `site_xmat` — the 9 row-major entries of
`xquat[hand] * site_quat[pinchsite]`. `Data` stores no `site_xmat` by design
(one quaternion multiply per read beats a `[BATCH, NSITE*9]` tensor written in
four FK paths), so it is composed here.

REWARD is `tolerance(distance, bounds=(0, .05), margin=.05)` between the
pinch site and the target site — both `xpos`, unlike the observation's `pos`.
Default gaussian sigmoid, `value_at_margin` 0.1. A hand exactly one target
radius outside the target scores 0.1, not 0.

RESET. `Reach.initialize_episode` is three statements and all three are here:

    self._hand.set_grasp(physics, close_factors=random_state.uniform())
        -> `custom_reset_cpu`, ONE draw broadcast to three fingers
    self._tcp_initializer(physics, random_state)
        -> `custom_reset_full_cpu`, site IK under collision rejection
    physics.bind(self._target).pos = self._target_placer(random_state)
        -> `custom_reset_model_cpu`, the site's model `pos`

⚠ THE MIDDLE ONE NEEDS A HOOK THE TRAIT DID NOT HAVE. It runs forward
kinematics, builds a site Jacobian and re-runs the narrow phase, all of which
take `Model` itself — and `custom_reset_cpu` is handed the record LISTS.
`Phyics3dEnvConfig.custom_reset_full_cpu` exists for that.

⚠⚠ WITHOUT IT `reset()` LEFT THE ARM AT qpos0, WHICH IS A 55-CONTACT POSE.
Not "a different distribution" — an invalid one: the links are inside each
other and inside the floor. Gated by
`tests/dm_control/test_reach_site_reset_vs_dm_control.mojo`, which judges our
reset poses with DM_CONTROL'S OWN acceptance predicate rather than with ours.

⚠ THE PARITY GATE DOES NOT COVER THE RESET, and never did. It drives both
engines from injected qpos/qvel, which is this suite's standing discipline
(see `manipulator_config`'s closing note). Reproducing a specific dm_control
episode is not a goal — its draws come from a numpy `RandomState` — so what
is gated is that every pose we produce is one the reference would accept.

⚠ `set_grasp` DOES NOT ZERO THE HAND'S `ctrl`, and the reference's
`JacoHand.set_grasp` does. The fingers run `<velocity>` actuators, so a stale
non-zero control commands a finger velocity from the first step of the next
episode. `Phyics3dEnv` re-applies actions from the caller's action vector
every step and never carries `ctrl` across a reset, so there is nothing stale
to clear here — the reference needs it because `physics.ctrl` persists.
"""

from std.collections import InlineArray
from std.math import sin, cos, sqrt, abs
from std.random import random_float64

from mojo_rl.physics3d.fields import Data, Model, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_RANGE_UNLIMITED,
    MODEL_SITE_SIZE,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
)
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.rewards import tolerance
from mojo_rl.envs.dm_control.manipulation_obs import (
    append_robot_block,
    N_ARM,
    N_HAND,
    BODY_PINCH,
)

from .manipulation_reset import (
    set_grasp,
    sample_bbox_uniform,
    tool_center_point_initializer,
    BODY_ARM,
    BODY_HAND,
    BODY_FIXED,
)


# ── model indices, all read off MuJoCo's own tables ────────────────────────
# `mj_id2name` on the baked `reach_site_features` model; asserted against it in
# `tests/dm_control/test_reach_site_vs_dm_control.mojo` so a model change
# cannot leave these pointing at the wrong element.
comptime SITE_TARGET: Int = 0  # `target_site`, on the WORLD body
# `N_ARM`, `N_HAND`, `BODY_PINCH` and the torque site/body tables are shared
# with the other 12 tasks — see `manipulation_obs`.

# ⚠⚠ THE ROBOT'S SITE IDS ARE PER TASK, NOT INVARIANT. The 9 robot sites start
# after the task's own worldbody sites, and how many of those there are depends
# on where the task put its target site — 3 here (`target_site`,
# `tcp_spawn_area`, `target_spawn_area`), 2 for `reach_duplo`, whose target
# site goes on the brick. See `manipulation_obs`' table.
comptime ROBOT_SITE_BASE: Int = 3
comptime SITE_PINCH: Int = ROBOT_SITE_BASE + 8  # `jaco_hand/pinchsite`

comptime OBS_DIM: Int = 45

# `reach.py::_SITE_WORKSPACE.target_bbox`. ⚠ `tcp_bbox` is the SAME box for
# THIS task and a different one for `reach_duplo`, so a test here cannot tell
# the two apart — do not collapse them into one constant.
comptime TARGET_BBOX_LOWER_X: Float64 = -0.2
comptime TARGET_BBOX_LOWER_Y: Float64 = -0.2
comptime TARGET_BBOX_LOWER_Z: Float64 = 0.02
comptime TARGET_BBOX_UPPER_X: Float64 = 0.2
comptime TARGET_BBOX_UPPER_Y: Float64 = 0.2
comptime TARGET_BBOX_UPPER_Z: Float64 = 0.4

comptime TARGET_RADIUS: Float64 = 0.05  # `reach.py::_TARGET_RADIUS`

# `base.py::_get_joint_pos_sampling_bounds` gives an UNLIMITED HINGE this as
# its upper bound. Four of Jaco's six arm joints are unlimited.
comptime TWO_PI: Float64 = 6.283185307179586
# `workspaces.DOWN_QUATERNION` = (w, x, y, z) (0, .7071, .7071, 0) in MuJoCo
# order; both non-zero components share this value.
comptime DOWN_QUAT_XY: Float64 = 0.70710678118


struct ReachSiteFeaturesConfig(Phyics3dEnvConfig):
    # === Physics ===
    # `constants.CONTROL_TIMESTEP` 0.04 over `<option timestep="0.002">`.
    comptime FRAME_SKIP: Int = 20
    # `manipulation.load` wraps the task in `composer.Environment` with
    # `time_limit=10.0` -> 250 control steps.
    comptime MAX_STEPS: Int = 250
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # `<option>` names no integrator, so MuJoCo's Euler default.
    comptime INTEGRATOR: StaticString = "euler"
    # dm_control reads the task state AFTER mj_step2+mj_step1, so position and
    # velocity stage quantities must describe the integrated qpos.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # ⚠ REQUIRED BY `joints_torque`. Without it `cfrc_int` stays zero and all
    # six torque readings are a silent zero — the observation still has the
    # right SHAPE, which is what makes it worth a comment.
    comptime RNE_POST: Bool = True
    # ⚠ Jaco's 9 meshes are ALL collidable (`contype="3" conaffinity="2"`),
    # unlike sawyer's 2-of-14 and dog's 0-of-162, so this cannot be 0 —
    # 0 silently disables the mesh narrow phase entirely. `fields_build`
    # raises with the required count if this is too small.
    comptime NMESH_VERTS: Int = 60000
    # CPU only for now; the GPU obs hook would need the same acceleration-stage
    # snapshot plumbing and has no consumer yet.
    comptime HAS_GPU_HOOKS: Bool = False
    comptime USES_MOCAP: Bool = False

    @staticmethod
    def get_timestep() -> Float64:
        return 0.002

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
        """The eight `_features` observables, in `observation_spec()` order.

        `target_position` is a TASK observable, so composer emits it FIRST;
        the remaining 42 are the shared robot block.
        """
        try:
            # target_position: the site's MODEL pos, not its xpos. The two
            # agree here only because the target site hangs off the worldbody
            # — on every OTHER reach variant the target rides a prop.
            var tb = SITE_TARGET * MODEL_SITE_SIZE
            obs.append(m_sites[tb + SITE_IDX_POS_X])
            obs.append(m_sites[tb + SITE_IDX_POS_Y])
            obs.append(m_sites[tb + SITE_IDX_POS_Z])
            append_robot_block[DTYPE](
                d, m_bodies, m_joints, m_sites, ROBOT_SITE_BASE, obs
            )
        except:
            return False
        return True

    # === CPU: Reward ===
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
        """`Reach.get_reward` — a tolerance on the hand-to-target distance.

        ⚠ BOTH SITES BY `xpos` HERE, while `target_position` above reads the
        model `pos`. The reward would be unchanged if it read `pos` for the
        target on THIS task and wrong on every task whose target rides a prop.
        """
        var dx = Float64(d.site_xpos.data[SITE_PINCH * 3 + 0]) - Float64(
            d.site_xpos.data[SITE_TARGET * 3 + 0]
        )
        var dy = Float64(d.site_xpos.data[SITE_PINCH * 3 + 1]) - Float64(
            d.site_xpos.data[SITE_TARGET * 3 + 1]
        )
        var dz = Float64(d.site_xpos.data[SITE_PINCH * 3 + 2]) - Float64(
            d.site_xpos.data[SITE_TARGET * 3 + 2]
        )
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        # dm_control tasks never terminate early.
        return (
            tolerance[DTYPE=DTYPE](
                Scalar[DTYPE](dist),
                Scalar[DTYPE](0.0),
                Scalar[DTYPE](TARGET_RADIUS),
                Scalar[DTYPE](TARGET_RADIUS),
            ),
            False,
        )

    # === CPU: per-episode MODEL randomization — the target site's pos ======
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
        """`physics.bind(self._target).pos = self._target_placer(...)`.

        The target IS a model constant in dm_control too — a site's `pos` —
        so this is the reference's own storage rather than the mocap-body
        workaround the suite's movable targets use. It is legitimate here only
        because the site is inert: sites never collide, so nothing derived
        from the model changes when it moves. A target that had to be COLLIDED
        with would need the mocap route (see `reacher_config`).
        """
        var lower = InlineArray[Float64, 3](fill=0.0)
        lower[0] = TARGET_BBOX_LOWER_X
        lower[1] = TARGET_BBOX_LOWER_Y
        lower[2] = TARGET_BBOX_LOWER_Z
        var upper = InlineArray[Float64, 3](fill=0.0)
        upper[0] = TARGET_BBOX_UPPER_X
        upper[1] = TARGET_BBOX_UPPER_Y
        upper[2] = TARGET_BBOX_UPPER_Z
        var draws = InlineArray[Float64, 3](fill=0.0)
        for k in range(3):
            draws[k] = random_float64()
        try:
            var p = sample_bbox_uniform[DTYPE](lower, upper, draws)
            var tb = SITE_TARGET * MODEL_SITE_SIZE
            m_sites[tb + SITE_IDX_POS_X] = p[0]
            m_sites[tb + SITE_IDX_POS_Y] = p[1]
            m_sites[tb + SITE_IDX_POS_Z] = p[2]
        except:
            pass

    # === CPU: per-episode STATE — the grasp ================================
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`self._hand.set_grasp(physics, close_factors=uniform())`.

        ⚠ ONE draw, broadcast to all three fingers — `reach.py` passes a
        SCALAR, which `JacoHand.set_grasp` fans out. Drawing three would give
        an asymmetric grasp the reference never produces.

        ⚠ THE TCP INITIALIZER IS NOT RUN HERE; see the module docstring. The
        arm therefore stays at qpos0.
        """
        var qadr = InlineArray[Int, N_HAND](fill=0)
        var rmin = InlineArray[Float64, N_HAND](fill=0.0)
        var rmax = InlineArray[Float64, N_HAND](fill=0.0)
        var factors = InlineArray[Float64, N_HAND](fill=0.0)
        var close = random_float64()
        for i in range(N_HAND):
            var jb = (N_ARM + i) * MODEL_JOINT_SIZE
            qadr[i] = N_ARM + i
            rmin[i] = Float64(m_joints[jb + JOINT_IDX_RANGE_MIN])
            rmax[i] = Float64(m_joints[jb + JOINT_IDX_RANGE_MAX])
            factors[i] = close
        try:
            set_grasp[DTYPE, N_HAND](
                d.qpos.data, qadr, rmin, rmax, factors
            )
        except:
            pass

    # === CPU: the TCP initializer — needs the whole Model =================
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
        """`self._tcp_initializer(physics, random_state)` — the middle
        statement of `Reach.initialize_episode`.

        Damped-least-squares site IK onto a TCP pose drawn from `tcp_bbox`,
        under rejection sampling against the arm/hand/ground collision
        predicate. All of it already exists and is gated
        (`dynamics/ik_site.mojo`, `manipulation_reset.mojo`); this is the
        wiring, which needed a hook that gets `Model` rather than the record
        lists.

        ⚠ WHY THIS IS NOT OPTIONAL. Without it the arm resets to qpos0, and
        MuJoCo reports **55 contacts** there — the links are inside each other
        and inside the floor. Every episode would begin in a pose the task
        never produces, and nothing raises.

        ⚠⚠ THE BOUNDS ARE NOT `jnt_range`, AND "UNLIMITED" IS NOT SPELLED THE
        WAY MuJoCo SPELLS IT. `base.py::_get_joint_pos_sampling_bounds` gives
        an unlimited HINGE `[0, 2*pi]` rather than its (absent) range, and four
        of Jaco's six arm joints are unlimited. MuJoCo marks those with
        `jnt_range = [0, 0]` and `jnt_limited = 0`; OUR record has no `limited`
        column and encodes them as `[-1e10, +1e10]` instead
        (`JOINT_RANGE_UNLIMITED`).

        The first version of this hook tested `range_max <= range_min` — the
        MuJoCo spelling — so the test never fired, the four unlimited joints
        got `[-1e10, 1e10]`, and every IK retry pose was drawn from that. The
        TCP initializer then exhausted on **7 of 24 resets with 10/10 IK
        failures**, while dm_control's own IK reaches 30/30 of the same targets
        in a mean of 2.4 attempts. Nothing about the bound itself looked wrong;
        only the failure rate did.

        Read from the model rather than transcribed, so a rebake cannot leave a
        stale number here.

        ⚠ BODY CLASSES ARE AN INPUT AND CANNOT BE DERIVED. The predicate asks
        which ENTITY owns a body; a baked MJCF is flat and `flat_model.mojo`
        keeps no body names. The array below is dm_control's own labelling
        (`manipulation_ref.body_classes_reference`), asserted against it in
        `tests/dm_control/test_reach_site_reset_vs_dm_control.mojo`. Bodies 1
        and 9 are the entity ATTACHMENT FRAMES: they own no geoms, so their
        `BODY_FIXED` label is never read — asserted, not assumed.

        ⚠ ON EXHAUSTION THIS RAISES, and that is deliberate. The reference
        raises `EpisodeInitializationError` too, and the alternative here is
        worse than an exception: a silent fallthrough leaves the arm at the
        55-contact qpos0, i.e. exactly the state this hook exists to prevent,
        with a plausible-looking observation on top of it.
        """
        comptime MAX_ATT: Int = 10  # `max_ik_attempts`
        comptime MAX_SAMP: Int = 10  # `max_rejection_samples`

        # `_get_joint_pos_sampling_bounds`, read off the model.
        var dof_idx = InlineArray[Int, N_ARM](fill=0)
        var qpos_adr = InlineArray[Int, N_ARM](fill=0)
        var lower = InlineArray[Float64, N_ARM](fill=0.0)
        var upper = InlineArray[Float64, N_ARM](fill=0.0)
        for a in range(N_ARM):
            var jb = a * MODEL_JOINT_SIZE
            dof_idx[a] = a
            qpos_adr[a] = a
            var lo = Float64(mf.joints.data[jb + JOINT_IDX_RANGE_MIN])
            var hi = Float64(mf.joints.data[jb + JOINT_IDX_RANGE_MAX])
            if hi >= JOINT_RANGE_UNLIMITED or lo <= -JOINT_RANGE_UNLIMITED:
                lo = 0.0
                hi = TWO_PI
            lower[a] = lo
            upper[a] = hi

        # `distributions.Uniform(*tcp_bbox)`, one draw per rejection sample.
        var targets = List[Scalar[DTYPE]]()
        var lo_b = InlineArray[Float64, 3](fill=0.0)
        lo_b[0] = TARGET_BBOX_LOWER_X
        lo_b[1] = TARGET_BBOX_LOWER_Y
        lo_b[2] = TARGET_BBOX_LOWER_Z
        var hi_b = InlineArray[Float64, 3](fill=0.0)
        hi_b[0] = TARGET_BBOX_UPPER_X
        hi_b[1] = TARGET_BBOX_UPPER_Y
        hi_b[2] = TARGET_BBOX_UPPER_Z
        for _ in range(MAX_SAMP):
            var draws = InlineArray[Float64, 3](fill=0.0)
            for k in range(3):
                draws[k] = random_float64()
            var p = sample_bbox_uniform[DTYPE](lo_b, hi_b, draws)
            for k in range(3):
                targets.append(p[k])

        # `randomize_arm_joints` between IK attempts — uniform over the SAME
        # bounds, which is why they are computed before this.
        var retry = List[Scalar[DTYPE]]()
        for _ in range(MAX_SAMP * (MAX_ATT - 1)):
            for a in range(N_ARM):
                retry.append(
                    Scalar[DTYPE](
                        lower[a] + (upper[a] - lower[a]) * random_float64()
                    )
                )

        # `workspaces.DOWN_QUATERNION`. ⚠ MuJoCo spells it (w, x, y, z) =
        # (0, .7071, .7071, 0); our quaternions are (x, y, z, w), so the two
        # leading components are the ones that carry it.
        var down = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        down[0] = Scalar[DTYPE](DOWN_QUAT_XY)
        down[1] = Scalar[DTYPE](DOWN_QUAT_XY)

        var body_class = InlineArray[Int, NBODY](fill=BODY_FIXED)
        for b in range(NBODY):
            if b >= 2 and b <= 8:
                body_class[b] = BODY_ARM
            elif b >= 10:
                body_class[b] = BODY_HAND

        var res = tool_center_point_initializer[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS, N_ARM,
        ](
            d, mf, SITE_PINCH, targets, down, dof_idx, qpos_adr,
            lower, upper, retry, body_class, False, MAX_ATT, MAX_SAMP,
        )
        if not res.success:
            raise Error(
                "reach_site_features: the TCP initializer exhausted "
                + String(res.samples)
                + " samples ("
                + String(res.ik_failures)
                + " IK failures, "
                + String(res.collision_rejections)
                + " collision rejections). dm_control raises"
                " EpisodeInitializationError here; falling through would reset"
                " the arm into qpos0, which carries 55 contacts."
            )
