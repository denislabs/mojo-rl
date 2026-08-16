"""`dm_control` `manipulation/reach_duplo_features` task config.

Port of `manipulation/reach.py::Reach` with `prop=Duplo` — the same task class
as `reach_site_features`, taking its OTHER branch:

    observation = robot(42) + prop(13)                                   (55)
    reward      = tolerance(|tcp - prop|, (0, 0.05), margin=0.05)
    episode     = 250 control steps (10 s / .04 s), no early termination
    action      = 9 <velocity> actuators (6 arm, 3 finger)

⚠⚠ `use_site` IS NOT A FLAG ON ONE CONFIG, IT IS TWO TASKS. `Reach.__init__`
branches on `self._prop` and the branches disagree about what the target IS,
which observables exist, and how the episode initialises:

                       reach_site                  reach_duplo
    target             a site on the WORLD         the prop's BODY
    obs                45 (target_position + 42)   55 (42 + free prop)
    reset's 3rd stmt   write the site's model pos  PropPlacer + settle
    workspace          +-0.2, z 0.02..0.4          +-0.1, z 0.001 (prop)
                                                   +-0.1, z 0.2..0.4 (tcp)

⚠ THE TARGET IS THE PROP'S BODY, NOT ITS `target_site`. `Reach` DOES bolt a
`target_site` onto the prop — `_make_target_site(parent_entity=prop,
visible=False)` — and then throws the return value away: `self._target` is
what `add_free_entity` returns, the ATTACHMENT FRAME. On this model the site
happens to sit at the frame origin, so reading it would agree by accident. It
is not the same element and the `place_*` tasks are where that stops being
free.

⚠ THE TWO BOUNDING BOXES ARE DIFFERENT HERE, and `reach_site_features` could
not tell them apart — that task uses the same box for both. `target_bbox` is
where the brick is dropped (z = `_PROP_Z_OFFSET` = 1 mm, flat) and `tcp_bbox`
is where the gripper is solved to (z 0.2 .. 0.4). Swapping them puts the
gripper on the floor and the brick in the air.

RESET — and ⚠ THE ORDER IS `Reach`'s, WHICH IS THE OPPOSITE OF `Lift`'s:

    self._hand.set_grasp(...)          <- same in both
    self._tcp_initializer(...)         <- ARM FIRST
    self._prop_placer(...)             <- then the prop

`Lift` places the prop first so the arm is solved against a scene where it
already sits. `Reach` does not, and the consequence is that the prop placer's
rejection loop runs against an arm that is ALREADY THERE — so a brick drawn
under the gripper is genuinely rejected and redrawn, which never happens in
`lift_large_box` (`ignore_collisions=True` there, a single draw).

⚠ `PropPlacer` HERE HAS `ignore_collisions=False` — the default, and `reach.py`
does not override it. The rejection predicate is `PropPlacer`'s own
(`prop_has_penetrating_contact`), NOT the TCP initializer's: it has no notion
of which entity the other geom belongs to, so the brick resting on the table
would be a rejection. That is why the draw is 1 mm up and the settle comes
after.
"""

from std.collections import InlineArray
from std.math import abs, sqrt
from std.random import random_float64

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_RANGE_UNLIMITED,
)
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.rewards import tolerance
from mojo_rl.envs.dm_control.manipulation_obs import (
    append_robot_block,
    append_free_prop_block_site,
    N_ARM,
    N_HAND,
)
from mojo_rl.envs.dm_control.manipulation_prop import (
    place_free_prop,
    settle_free_prop,
    uniform_z_rotation,
)
from mojo_rl.envs.dm_control.manipulation_reset import (
    set_grasp,
    sample_bbox_uniform,
    tool_center_point_initializer,
    BODY_ARM,
    BODY_HAND,
    BODY_FIXED,
    BODY_FREE,
)

from .manipulation_reach_duplo_def import ReachDuploModel


# ── model indices, read off MuJoCo's own tables and asserted in the gate ───
comptime PROP_BODY: Int = 17  # `duplo2x4/`, the attachment frame
comptime PROP_FRAME_SITE: Int = 11  # `duplo2x4/bounding_box`
comptime PROP_QPOS_ADR: Int = 9  # its free joint
comptime PROP_DOF_ADR: Int = 9

# ⚠⚠ TWO, NOT THREE — AND THIS IS THE TRAP THIS TASK EXISTS TO TEACH. The 9
# robot sites start after the task's own worldbody sites. `Reach` WITH a prop
# puts its `target_site` on the brick rather than the arena, so this model has
# only `tcp_spawn_area` and `target_spawn_area` there and the whole robot block
# shifts down by one. Inheriting the other two tasks' 3 read the brick's
# `bounding_box` as the pinch site: the observation was off by 1.2, the torque
# sensors by 2.2 of readings up to 2.3, and the TCP initializer drove a site on
# a FREE body — 10 IK failures out of 10.
comptime ROBOT_SITE_BASE: Int = 2
comptime SITE_PINCH: Int = ROBOT_SITE_BASE + 8  # `jaco_hand/pinchsite` = 10

comptime OBS_DIM: Int = 55

# `reach.py::_DUPLO_WORKSPACE`. ⚠ `target_bbox` is FLAT in z at
# `_PROP_Z_OFFSET` — the brick is dropped from 1 mm up so the placer's
# rejection test sees it in the AIR, and the settle then drops it.
comptime PROP_Z_OFFSET: Float64 = 0.001
comptime TARGET_BBOX_LOWER_X: Float64 = -0.1
comptime TARGET_BBOX_LOWER_Y: Float64 = -0.1
comptime TARGET_BBOX_LOWER_Z: Float64 = PROP_Z_OFFSET
comptime TARGET_BBOX_UPPER_X: Float64 = 0.1
comptime TARGET_BBOX_UPPER_Y: Float64 = 0.1
comptime TARGET_BBOX_UPPER_Z: Float64 = PROP_Z_OFFSET
comptime TCP_BBOX_LOWER_X: Float64 = -0.1
comptime TCP_BBOX_LOWER_Y: Float64 = -0.1
comptime TCP_BBOX_LOWER_Z: Float64 = 0.2
comptime TCP_BBOX_UPPER_X: Float64 = 0.1
comptime TCP_BBOX_UPPER_Y: Float64 = 0.1
comptime TCP_BBOX_UPPER_Z: Float64 = 0.4

comptime TARGET_RADIUS: Float64 = 0.05  # `reach.py::_TARGET_RADIUS`
comptime TWO_PI: Float64 = 6.283185307179586
comptime DOWN_QUAT_XY: Float64 = 0.70710678118

# `PropPlacer.max_attempts_per_prop`.
comptime MAX_PROP_ATTEMPTS: Int = 20


struct ReachDuploConfig(Phyics3dEnvConfig):
    # === Physics === (identical to the other two; same arena, same arm)
    comptime FRAME_SKIP: Int = 20
    comptime MAX_STEPS: Int = 250
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime RNE_POST: Bool = True  # required by `joints_torque`
    comptime NMESH_VERTS: Int = 60000
    comptime HAS_GPU_HOOKS: Bool = False
    comptime USES_MOCAP: Bool = False

    @staticmethod
    def get_timestep() -> Float64:
        return 0.002

    # === CPU: Observation ===
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
        """Robot (42) then the prop (13).

        ⚠ NO `target_position` HERE, unlike `reach_site_features`. That
        observable exists only on the site branch — with a prop the brick IS
        the target, and composer's `task_observables` holds nothing enabled.
        So the robot block LEADS, and the 55-vector is laid out exactly like
        `lift_large_box`'s.

        ⚠ THE PROP'S FRAME IS A SITE, NOT A GEOM. The Duplo's
        `framepos`/`framequat`/`framelinvel`/`frameangvel` sensors name
        `bounding_box`, 11.9 mm above the body origin; `props.Primitive` (the
        large box) names its geom. Same 13 floats, different element.
        """
        try:
            append_robot_block[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
                d, m_bodies, m_joints, m_sites, ROBOT_SITE_BASE, obs
            )
            append_free_prop_block_site[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_sites, PROP_FRAME_SITE, obs)
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
        """`Reach.get_reward` — distance from the pinch site to the prop.

        ⚠ THE TARGET IS `d.xpos[PROP_BODY]`, the free body's origin —
        `physics.bind(self._target).xpos`, where `_target` is the attachment
        frame. NOT the `bounding_box` site the observation reads (11.9 mm up)
        and NOT the prop's `target_site`. Three plausible elements, one right
        answer; the observation and the reward deliberately read different
        ones.

        ⚠ THE DEFAULT GAUSSIAN SIGMOID, unlike `Lift`'s linear one — `Reach`
        passes neither `sigmoid` nor `value_at_margin`.
        """
        var dx = Float64(
            d.site_xpos.data[SITE_PINCH * 3 + 0]
            - d.xpos.data[PROP_BODY * 3 + 0]
        )
        var dy = Float64(
            d.site_xpos.data[SITE_PINCH * 3 + 1]
            - d.xpos.data[PROP_BODY * 3 + 1]
        )
        var dz = Float64(
            d.site_xpos.data[SITE_PINCH * 3 + 2]
            - d.xpos.data[PROP_BODY * 3 + 2]
        )
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        return (
            tolerance[DTYPE=DTYPE](
                Scalar[DTYPE](dist),
                Scalar[DTYPE](0.0),
                Scalar[DTYPE](TARGET_RADIUS),
                Scalar[DTYPE](TARGET_RADIUS),
            ),
            False,
        )

    # === CPU: per-episode STATE — the grasp ===============================
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
        """`set_grasp` — ONE draw broadcast to all three fingers."""
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
            set_grasp[DTYPE, N_HAND](d.qpos.data, qadr, rmin, rmax, factors)
        except:
            pass

    # === CPU: the TCP initializer, then the prop placer and its settle ====
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
        """`Reach.initialize_episode`'s second and third statements.

        ⚠ ARM BEFORE PROP. See the module docstring: `Reach`'s order is
        `set_grasp` (already done), `tcp_initializer`, `prop_placer` — the
        reverse of `Lift`'s.
        """
        comptime MAX_ATT: Int = 10
        comptime MAX_SAMP: Int = 10

        # ── 1. the TCP initializer ──────────────────────────────────────
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
            # ⚠ Our record spells "unlimited" as +-JOINT_RANGE_UNLIMITED, not
            # MuJoCo's [0, 0]. Testing `hi <= lo` never fires and draws retry
            # poses from +-1e10 — see that constant.
            if hi >= JOINT_RANGE_UNLIMITED or lo <= -JOINT_RANGE_UNLIMITED:
                lo = 0.0
                hi = TWO_PI
            lower[a] = lo
            upper[a] = hi

        var targets = List[Scalar[DTYPE]]()
        var lo_t = InlineArray[Float64, 3](fill=0.0)
        lo_t[0] = TCP_BBOX_LOWER_X
        lo_t[1] = TCP_BBOX_LOWER_Y
        lo_t[2] = TCP_BBOX_LOWER_Z
        var hi_t = InlineArray[Float64, 3](fill=0.0)
        hi_t[0] = TCP_BBOX_UPPER_X
        hi_t[1] = TCP_BBOX_UPPER_Y
        hi_t[2] = TCP_BBOX_UPPER_Z
        for _ in range(MAX_SAMP):
            var td = InlineArray[Float64, 3](fill=0.0)
            for k in range(3):
                td[k] = random_float64()
            var p = sample_bbox_uniform[DTYPE](lo_t, hi_t, td)
            for k in range(3):
                targets.append(p[k])

        var retry = List[Scalar[DTYPE]]()
        for _ in range(MAX_SAMP * (MAX_ATT - 1)):
            for a in range(N_ARM):
                retry.append(
                    Scalar[DTYPE](
                        lower[a] + (upper[a] - lower[a]) * random_float64()
                    )
                )

        var down = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        down[0] = Scalar[DTYPE](DOWN_QUAT_XY)
        down[1] = Scalar[DTYPE](DOWN_QUAT_XY)

        # ⚠ THE PROP IS `BODY_FREE`. dm_control's TCP predicate ignores
        # robot-versus-free-body contacts — a brick can be pushed aside, so
        # resting against one is not a bad initial pose. Note the prop is still
        # at qpos0 (the origin) when this runs, because `Reach` places the arm
        # FIRST; labelling it FIXED would reject arm poses over the origin.
        var body_class = InlineArray[Int, NBODY](fill=BODY_FIXED)
        for b in range(NBODY):
            if b >= 2 and b <= 8:
                body_class[b] = BODY_ARM
            elif b >= 10 and b <= 16:
                body_class[b] = BODY_HAND
            elif b == PROP_BODY:
                body_class[b] = BODY_FREE

        var res = tool_center_point_initializer[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS, N_ARM,
        ](
            d, mf, SITE_PINCH, targets, down, dof_idx, qpos_adr,
            lower, upper, retry, body_class, False, MAX_ATT, MAX_SAMP,
        )
        if not res.success:
            raise Error(
                "reach_duplo: the TCP initializer exhausted "
                + String(res.samples)
                + " samples ("
                + String(res.ik_failures)
                + " IK failures, "
                + String(res.collision_rejections)
                + " collision rejections)"
            )

        # ── 2. place the prop, REJECTING poses that penetrate ───────────
        # ⚠ 20 DRAWS ARE PREPARED, not one. `ignore_collisions` is False here,
        # so the loop is real: the arm is already placed and a brick drawn
        # under the gripper is rejected.
        var lo_p = InlineArray[Float64, 3](fill=0.0)
        lo_p[0] = TARGET_BBOX_LOWER_X
        lo_p[1] = TARGET_BBOX_LOWER_Y
        lo_p[2] = TARGET_BBOX_LOWER_Z
        var hi_p = InlineArray[Float64, 3](fill=0.0)
        hi_p[0] = TARGET_BBOX_UPPER_X
        hi_p[1] = TARGET_BBOX_UPPER_Y
        hi_p[2] = TARGET_BBOX_UPPER_Z
        var poses = List[Scalar[DTYPE]]()
        for _ in range(MAX_PROP_ATTEMPTS):
            var draws = InlineArray[Float64, 3](fill=0.0)
            for k in range(3):
                draws[k] = random_float64()
            var ppos = sample_bbox_uniform[DTYPE](lo_p, hi_p, draws)
            var pquat = uniform_z_rotation[DTYPE](random_float64())
            for k in range(3):
                poses.append(ppos[k])
            for k in range(4):
                poses.append(pquat[k])

        var pres = place_free_prop[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
        ](
            d, mf, PROP_BODY, PROP_QPOS_ADR, PROP_DOF_ADR, poses,
            List[Int](), False, MAX_PROP_ATTEMPTS,
        )
        if not pres.success:
            raise Error(
                "reach_duplo: the prop placer found no non-colliding pose in "
                + String(pres.attempts)
                + " attempts"
            )

        # ── 3. settle it, with the robot held static ────────────────────
        _ = settle_free_prop[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
            ReachDuploModel.CONE_TYPE,
            ReachDuploModel.MAX_CONDIM,
            ReachDuploModel.NOSLIP_ITER,
            N_ARM + N_HAND,
        ](d, mf, PROP_DOF_ADR, Self.get_timestep())
