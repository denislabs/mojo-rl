"""`dm_control` `manipulation/lift_large_box_features` task config.

Port of `manipulation/lift.py::Lift` with the box prop — the second Phase 7
task, and the first with anything to manipulate.

    observation = robot(42) + prop(13)                                   (55)
    reward      = tolerance(lowest_vertex_z, (target_height, inf),
                            margin=.3, value_at_margin=0, sigmoid='linear')
    episode     = 250 control steps (10 s / .04 s), no early termination
    action      = 9 <velocity> actuators (6 arm, 3 finger)

⚠⚠ THE REWARD DEPENDS ON PER-EPISODE STATE, WHICH NO OTHER PORTED TASK HAS.
`target_height` is `_DISTANCE_TO_LIFT + <the prop's lowest vertex after the
reset settles>` — it is computed in `initialize_episode` and read by
`get_reward` for the rest of the episode. It cannot be a constant: where the
prop settles depends on the draw. It lives in `Data.meta`'s
`META_IDX_TASK_PARAM_0`, which is per-env and survives the step (see
`gpu/constants`); `prev_x` is the wrong home because `pre_step_cpu` rewrites
it every step.

⚠ THE HEIGHT IS THE LOWEST OF EIGHT VERTEX SITES, not the body's z.
`_VertexSitesMixin` adds a site at each corner of the box, and
`_get_height_of_lowest_vertex` minimises over their `xpos[2]`. For a box tilted
by the settle that is a materially different number from the centre height,
and it is the one that decides whether the prop has been lifted.

⚠ `sigmoid='linear'` WITH `value_at_margin=0`, not the gaussian default. The
reward is 0 at `target_height - 0.3` and rises LINEARLY to 1 at the target —
so a policy gets gradient over the whole 30 cm, which a gaussian would not
give. Getting the sigmoid wrong is a plausible-looking curve, not a crash.

RESET — and ⚠ THE ORDER IS NOT `Reach`'s:

    self._hand.set_grasp(...)          <- same
    self._prop_placer(...)             <- PROP FIRST
    self._tcp_initializer(...)         <- then the arm

`Reach` places its target LAST and the order does not matter there because a
site is inert. Here it does: the TCP initializer rejects poses that collide
with an external body without a freejoint, and the prop HAS a freejoint, so it
is never a rejection reason — but the arm must be solved against a scene where
the prop already sits, because settling it afterwards would move it into an
arm that had been placed around empty space.

⚠ `PropPlacer(ignore_collisions=True)` here, unlike `place_*`. With one prop
and collisions ignored the rejection loop is a single draw, so what remains is
the placement and the settle.
"""

from std.collections import InlineArray
from std.math import abs, sqrt, inf
from std.random import random_float64

from mojo_rl.physics3d.fields import Data, Model, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_RANGE_UNLIMITED,
    MODEL_SITE_SIZE,
    SITE_IDX_POS_Z,
    META_IDX_TASK_PARAM_0,
)
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.dm_control.rewards import tolerance, SIGMOID_LINEAR
from mojo_rl.envs.dm_control.manipulation_obs import (
    append_robot_block,
    append_free_prop_block,
    N_ARM,
    N_HAND,
)
from mojo_rl.envs.dm_control.manipulation_prop import (
    set_free_prop_pose,
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

from .manipulation_lift_box_def import LiftLargeBoxModel


# ── model indices, read off MuJoCo's own tables and asserted in the gate ───
comptime SITE_TARGET_HEIGHT: Int = 0  # `target_height`, a worldbody bbox site
comptime PROP_BODY: Int = 17  # `unnamed_model/`
comptime PROP_GEOM: Int = 21  # `unnamed_model/geom`, the box
comptime PROP_QPOS_ADR: Int = 9  # its free joint
comptime PROP_DOF_ADR: Int = 9
comptime VERTEX_SITE_0: Int = 12  # `vertex_0` .. `vertex_7` are 12..19
comptime N_VERTICES: Int = 8

# ⚠⚠ THE ROBOT'S SITE IDS ARE PER TASK, NOT INVARIANT. The 9 robot sites start
# after the task's own worldbody sites, and how many of those there are depends
# on where the task put its target site — 3 here (`target_height`,
# `tcp_spawn_area`, `prop_spawn_area`), 2 for `reach_duplo`, whose target
# site goes on the brick. See `manipulation_obs`' table.
comptime ROBOT_SITE_BASE: Int = 3
comptime SITE_PINCH: Int = ROBOT_SITE_BASE + 8  # `jaco_hand/pinchsite`

comptime OBS_DIM: Int = 55

# `lift.py::_BOX_WORKSPACE`. ⚠ The prop box is FLAT in z — lower and upper both
# `_BOX_SIZE` — so the box is placed exactly at its own half-height, i.e.
# already resting. `tcp_bbox` is a different box and they are easy to confuse.
comptime PROP_BBOX_LOWER_X: Float64 = -0.1
comptime PROP_BBOX_LOWER_Y: Float64 = -0.1
comptime PROP_BBOX_LOWER_Z: Float64 = 0.09  # `_BOX_SIZE`
comptime PROP_BBOX_UPPER_X: Float64 = 0.1
comptime PROP_BBOX_UPPER_Y: Float64 = 0.1
comptime PROP_BBOX_UPPER_Z: Float64 = 0.09
comptime TCP_BBOX_LOWER_X: Float64 = -0.1
comptime TCP_BBOX_LOWER_Y: Float64 = -0.1
comptime TCP_BBOX_LOWER_Z: Float64 = 0.2
comptime TCP_BBOX_UPPER_X: Float64 = 0.1
comptime TCP_BBOX_UPPER_Y: Float64 = 0.1
comptime TCP_BBOX_UPPER_Z: Float64 = 0.4

comptime DISTANCE_TO_LIFT: Float64 = 0.3  # `lift.py::_DISTANCE_TO_LIFT`
comptime TWO_PI: Float64 = 6.283185307179586
comptime DOWN_QUAT_XY: Float64 = 0.70710678118


@always_inline
def lowest_vertex_z[DTYPE: DType, D: DimsLike](d: Data[DTYPE, D, 1]) -> Float64:
    """`Lift._get_height_of_lowest_vertex` — min over the 8 corner sites."""
    var lo = Float64(d.site_xpos.data[VERTEX_SITE_0 * 3 + 2])
    for v in range(1, N_VERTICES):
        var z = Float64(d.site_xpos.data[(VERTEX_SITE_0 + v) * 3 + 2])
        if z < lo:
            lo = z
    return lo


struct LiftLargeBoxConfig(Phyics3dEnvConfig):
    # === Physics === (identical to reach_site_features; same arena, same arm)
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
    def custom_extract_obs_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """Robot (42) then the prop (13). No task observable — `Lift` declares
        none, so the robot block leads."""
        try:
            append_robot_block[DTYPE](
                d, m_bodies, m_joints, m_sites, ROBOT_SITE_BASE, obs
            )
            append_free_prop_block[DTYPE](
                d, m_geoms, PROP_GEOM, obs
            )
        except:
            return False
        return True

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
        """`Lift.get_reward` — a LINEAR ramp over the last 30 cm of lift.

        ⚠ `target_height` comes from `META_IDX_TASK_PARAM_0`, written at
        reset. Reading a constant here would make the reward depend on where
        the floor is rather than on where the prop STARTED.
        """
        var target = Float64(d.meta.data[META_IDX_TASK_PARAM_0])
        var h = lowest_vertex_z[DTYPE](d)
        # bounds = (target_height, inf): no upper bound, so lifting further
        # never costs anything.
        return (
            tolerance[SIGMOID_LINEAR, 0.0, DTYPE](
                Scalar[DTYPE](h),
                Scalar[DTYPE](target),
                Scalar[DTYPE](inf[DTYPE]()),
                Scalar[DTYPE](DISTANCE_TO_LIFT),
            ),
            False,
        )

    # === CPU: per-episode STATE — the grasp ===============================
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`set_grasp` — ONE draw broadcast to all three fingers, exactly as
        `reach.py` and `lift.py` both do."""
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

    # === CPU: the prop placer, the settle, and the TCP initializer ========
    @staticmethod
    def custom_reset_full_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
    ) raises:
        """`Lift.initialize_episode`'s second and third statements, plus the
        target height it derives from them.

        ⚠ PROP BEFORE ARM. See the module docstring: the reference's order is
        `set_grasp` (already done), `prop_placer`, `tcp_initializer`.
        """
        comptime MAX_ATT: Int = 10
        comptime MAX_SAMP: Int = 10

        # ── 1. place the prop ───────────────────────────────────────────
        var lo_p = InlineArray[Float64, 3](fill=0.0)
        lo_p[0] = PROP_BBOX_LOWER_X
        lo_p[1] = PROP_BBOX_LOWER_Y
        lo_p[2] = PROP_BBOX_LOWER_Z
        var hi_p = InlineArray[Float64, 3](fill=0.0)
        hi_p[0] = PROP_BBOX_UPPER_X
        hi_p[1] = PROP_BBOX_UPPER_Y
        hi_p[2] = PROP_BBOX_UPPER_Z
        var draws = InlineArray[Float64, 3](fill=0.0)
        for k in range(3):
            draws[k] = random_float64()
        var ppos = sample_bbox_uniform[DTYPE](lo_p, hi_p, draws)
        var pquat = uniform_z_rotation[DTYPE](random_float64())
        set_free_prop_pose[DTYPE](
            d, PROP_QPOS_ADR, PROP_DOF_ADR, ppos, pquat
        )

        # ── 2. settle it, with the robot held static ────────────────────
        _ = settle_free_prop[DTYPE, LiftLargeBoxModel.CONE_TYPE, LiftLargeBoxModel.MAX_CONDIM, LiftLargeBoxModel.NOSLIP_ITER, N_ARM + N_HAND](d, mf, PROP_DOF_ADR, Self.get_timestep())

        # ── 3. the TCP initializer ──────────────────────────────────────
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

        # ⚠ THE PROP IS `BODY_FREE`, NOT `BODY_FIXED`. dm_control's predicate
        # ignores robot-versus-free-body contacts entirely — a prop can be
        # pushed aside, so resting against one is not a bad initial pose.
        # Labelling it FIXED would reject every arm pose near the box, which
        # is most of the reachable workspace for a LIFTING task.
        var body_class = InlineArray[Int, D.NBODY](fill=BODY_FIXED)
        for b in range(D.NBODY):
            if b >= 2 and b <= 8:
                body_class[b] = BODY_ARM
            elif b >= 10 and b <= 16:
                body_class[b] = BODY_HAND
            elif b == PROP_BODY:
                body_class[b] = BODY_FREE

        var res = tool_center_point_initializer[DTYPE, N_ARM](
            d, mf, SITE_PINCH, targets, down, dof_idx, qpos_adr,
            lower, upper, retry, body_class, False, MAX_ATT, MAX_SAMP,
        )
        if not res.success:
            raise Error(
                "lift_large_box: the TCP initializer exhausted "
                + String(res.samples)
                + " samples ("
                + String(res.ik_failures)
                + " IK failures, "
                + String(res.collision_rejections)
                + " collision rejections)"
            )

        # ── 4. the target height, derived from where the prop SETTLED ───
        # ⚠ AFTER the arm is placed, because `tool_center_point_initializer`
        # re-runs forward kinematics and the vertex sites move with it. Reading
        # it before would record the height under a different FK state — the
        # same instant-mismatch that defect 19 was.
        var h0 = lowest_vertex_z[DTYPE](d)
        d.meta.data[META_IDX_TASK_PARAM_0] = Scalar[DTYPE](
            DISTANCE_TO_LIFT + h0
        )
        # The reference also moves the `target_height` visual site; it is a
        # bbox marker with no dynamics, so it is written for a renderer's sake
        # and read by nothing.
        mf.sites.data[
            SITE_TARGET_HEIGHT * MODEL_SITE_SIZE + SITE_IDX_POS_Z
        ] = Scalar[DTYPE](DISTANCE_TO_LIFT + h0)
