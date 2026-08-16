"""`dm_control` `manipulation/stack_2_bricks_features` task config.

Port of `manipulation/bricks.py::Stack` with two bricks, a FIXED base and no
order randomisation — the smallest member of the seven-task brick family.

    observation = robot(42) + brick0(13) + brick1(13)                    (68)
    reward      = a stud-to-hole distance, shaped twice and averaged
    episode     = 250 control steps (10 s / .04 s), no early termination
    action      = 9 <velocity> actuators (6 arm, 3 finger)

⚠⚠ THE BASE BRICK HAS NO FREEJOINT, AND THAT IS DECIDED AT RESET.
`initialize_episode_mjcf` calls `_add_or_remove_freejoints`, which REMOVES the
freejoint from the brick at `desired_order[0]` when `moveable_base` is False.
`randomize_order` is False here, so `desired_order` is `[0, 1]` every episode
and brick 0 is always the fixed one — which is the only reason this task has a
static model at all. The three `*_random_order_*` tasks redraw that index every
episode and their model PERMUTES; see `manipulation_stack2_def`.

⚠ SO BRICK 0 IS PLACED BY WRITING `body_pos`/`body_quat`, exactly like
`Place`'s pedestal, and brick 1 by writing `qpos`. One `PropPlacer` call in the
reference, two different mechanisms underneath — `place_fixed_prop` and
`place_free_prop`.

⚠⚠ AND THE PLACER'S CONTACT-DISABLING PASS FINALLY MATTERS. `PropPlacer.
__call__` zeroes `contype`/`conaffinity` on every prop it is about to place and
restores them ONE AT A TIME as it places each. So brick 0 is drawn while brick
1 is invisible to collision — and it has to be, because brick 1 is still
wherever the last episode left it. The four single-prop tasks made this a
no-op; here it is the difference between a reject loop that works and one that
rejects the first draw forever.

⚠ THE REWARD IS A STUD-TO-HOLE DISTANCE, NOT A BODY DISTANCE.
`_min_stud_to_hole_distance` measures two CORNER sites — `studs[0,0]` and
`studs[1,3]` on the bottom brick against `holes[0,0]` and `holes[1,3]` on the
top — and takes the smaller of the two pairings, because a Duplo is
rotationally symmetric and a brick rotated 180 degrees is equally stacked.
Comparing only one pairing halves the reward on half the poses.

⚠ AND IT IS SHAPED TWICE AT VERY DIFFERENT SCALES: a `close` term with bounds
(0, 1 cm) and a 10 cm margin, a `clicked` term with bounds (0, 1 mm) and a
1 mm margin, averaged with weights 0.1 and 1.0. The `clicked` term is
essentially zero until the bricks are within a millimetre, so a gate that only
probes the coarse regime cannot tell the average from `close` alone.

⚠ THE HINT BRICKS ARE NOT PORTED, and that is correct rather than a shortcut.
`_build_stack` arranges the translucent goal-hint copies; they are contactless,
carry no observables and no sensors anything reads, so they affect the
renderer and nothing else. They still occupy bodies 18 and 20 and 82 of the
185 geoms, which is why the real bricks are 17 and 19 rather than 17 and 18.
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
    place_fixed_prop,
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

from .manipulation_stack2_def import Stack2BricksModel


# ── model indices, read off MuJoCo's own tables and asserted in the gate ───
#
# ⚠ TWO worldbody sites only (`tcp_spawn_area`, `prop_spawn_area`) — `Stack`
# has no target site on the arena, so the robot block starts at 2 like
# `reach_duplo` and NOT at 3 like `lift_*` and `place_*`.
comptime ROBOT_SITE_BASE: Int = 2
comptime SITE_PINCH: Int = ROBOT_SITE_BASE + 8  # `jaco_hand/pinchsite` = 10

# ⚠ 17 and 19, NOT 17 and 18: bodies 18 and 20 are the translucent hint twins.
comptime BRICK0_BODY: Int = 17  # `duplo2x4/`, the FIXED base
comptime BRICK1_BODY: Int = 19  # `duplo2x4_2/`, the free one
comptime BRICK0_FRAME_SITE: Int = 11  # `duplo2x4/bounding_box`
comptime BRICK1_FRAME_SITE: Int = 45  # `duplo2x4_2/bounding_box`
# `stud_00 .. stud_13` then `hole_00 .. hole_13`, 8 each, contiguous.
comptime BRICK0_STUD_0: Int = 12
comptime BRICK0_HOLE_0: Int = 20
comptime BRICK1_STUD_0: Int = 46
comptime BRICK1_HOLE_0: Int = 54
# `studs[[0, -1], [0, -1]]` selects `stud_00` and `stud_13` — offsets 0 and 7
# in the contiguous block, NOT 0 and 1.
comptime CORNER_A: Int = 0
comptime CORNER_B: Int = 7

comptime BRICK1_QPOS_ADR: Int = 9
comptime BRICK1_DOF_ADR: Int = 9

comptime OBS_DIM: Int = 68

# `bricks.py::_WORKSPACE`. ⚠ `tcp_bbox` starts at 0.15 here — not
# `reach_duplo`'s 0.2, not `place_*`'s 0.17.
comptime PROP_Z_OFFSET: Float64 = 1.0e-6
comptime PROP_BBOX_LOWER_X: Float64 = -0.1
comptime PROP_BBOX_LOWER_Y: Float64 = -0.1
comptime PROP_BBOX_LOWER_Z: Float64 = PROP_Z_OFFSET
comptime PROP_BBOX_UPPER_X: Float64 = 0.1
comptime PROP_BBOX_UPPER_Y: Float64 = 0.1
comptime PROP_BBOX_UPPER_Z: Float64 = PROP_Z_OFFSET
comptime TCP_BBOX_LOWER_X: Float64 = -0.1
comptime TCP_BBOX_LOWER_Y: Float64 = -0.1
comptime TCP_BBOX_LOWER_Z: Float64 = 0.15
comptime TCP_BBOX_UPPER_X: Float64 = 0.1
comptime TCP_BBOX_UPPER_Y: Float64 = 0.1
comptime TCP_BBOX_UPPER_Z: Float64 = 0.4

comptime CLOSE_THRESHOLD: Float64 = 0.01  # `bricks.py::_CLOSE_THRESHOLD`
comptime CLICK_THRESHOLD: Float64 = 0.001  # `bricks.py::_CLICK_THRESHOLD`
comptime CLOSE_COEF: Float64 = 0.1  # `_get_pairwise_stacking_rewards`
comptime TWO_PI: Float64 = 6.283185307179586
comptime DOWN_QUAT_XY: Float64 = 0.70710678118

comptime MAX_PROP_ATTEMPTS: Int = 20  # `max_attempts_per_prop`, the default


@always_inline
def _site_dist[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAXC: Int,
    NSITE: Int,
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAXC, nsite=NSITE], 1],
    a: Int,
    b: Int,
) -> Float64:
    var dx = Float64(d.site_xpos.data[a * 3 + 0] - d.site_xpos.data[b * 3 + 0])
    var dy = Float64(d.site_xpos.data[a * 3 + 1] - d.site_xpos.data[b * 3 + 1])
    var dz = Float64(d.site_xpos.data[a * 3 + 2] - d.site_xpos.data[b * 3 + 2])
    return sqrt(dx * dx + dy * dy + dz * dz)


def min_stud_to_hole_distance[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAXC: Int,
    NSITE: Int,
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAXC, nsite=NSITE], 1],
    bottom_stud_0: Int,
    top_hole_0: Int,
) -> Float64:
    """`bricks.py::_min_stud_to_hole_distance`.

    Two corner studs on the bottom brick against the two corner holes on the
    top one, summed — and then the SMALLER of the two pairings.

    ⚠⚠ THE SECOND PAIRING IS NOT OPTIONAL. A Duplo is rotationally symmetric,
    so a top brick rotated 180 degrees about z is equally well stacked; the
    reference reverses `stud_pos` and takes the min. Keeping only the first
    pairing reports roughly double the distance on half of all poses, which
    reads as a policy that cannot finish rather than as a reward bug.

    ⚠ IT IS A SUM OF TWO DISTANCES, NOT A DISTANCE. `_distance` is
    `sum(sqrt((diff*diff).sum(1)))` over the two rows — so the thresholds it is
    compared against (1 cm and 1 mm) apply to the TOTAL.
    """
    var sa = bottom_stud_0 + CORNER_A
    var sb = bottom_stud_0 + CORNER_B
    var ha = top_hole_0 + CORNER_A
    var hb = top_hole_0 + CORNER_B
    var d1 = (
        _site_dist[DTYPE, NQ, NV, NBODY, MAXC, NSITE](d, sa, ha)
        + _site_dist[DTYPE, NQ, NV, NBODY, MAXC, NSITE](d, sb, hb)
    )
    var d2 = (
        _site_dist[DTYPE, NQ, NV, NBODY, MAXC, NSITE](d, sb, ha)
        + _site_dist[DTYPE, NQ, NV, NBODY, MAXC, NSITE](d, sa, hb)
    )
    return d1 if d1 < d2 else d2


@always_inline
def pairwise_stacking_reward(dist: Float64) -> Float64:
    """`bricks.py::_get_pairwise_stacking_rewards` for ONE pair.

    ⚠ THE TWO MARGINS DIFFER BY A FACTOR OF 100. `close` decays over 10 cm
    from a 1 cm bound; `clicked` decays over 1 mm from a 1 mm bound. So
    `clicked` is numerically zero everywhere except the last couple of
    millimetres, and a probe that never gets that close cannot distinguish the
    weighted average from `close * 0.1 / 1.1`.

    ⚠ `np.average(..., weights=[0.1, 1.0])` NORMALISES — the divisor is 1.1,
    not 2 and not 1. Getting that wrong scales the whole reward by 1.8x.
    """
    var close = Float64(
        tolerance[DTYPE=DType.float64](
            dist, 0.0, CLOSE_THRESHOLD, CLOSE_THRESHOLD * 10.0
        )
    )
    var clicked = Float64(
        tolerance[DTYPE=DType.float64](
            dist, 0.0, CLICK_THRESHOLD, CLICK_THRESHOLD
        )
    )
    return (CLOSE_COEF * close + 1.0 * clicked) / (CLOSE_COEF + 1.0)


struct Stack2BricksConfig(Phyics3dEnvConfig):
    # === Physics === (identical to every other task in this family)
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
        """Robot (42), then the two bricks (13 each), in ATTACHMENT order.

        ⚠ THE HINT BRICKS CONTRIBUTE NOTHING. They are attached between the
        real ones and have no observables at all, so the observation is
        brick 0 then brick 1 — bodies 17 and 19, skipping 18.

        ⚠ NO `desired_order` HERE. That task observable exists only on the
        three `*_random_order_*` tasks, and it sorts FIRST when it does.
        """
        try:
            append_robot_block[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
                d, m_bodies, m_joints, m_sites, ROBOT_SITE_BASE, obs
            )
            append_free_prop_block_site[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_sites, BRICK0_FRAME_SITE, obs)
            append_free_prop_block_site[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_sites, BRICK1_FRAME_SITE, obs)
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
        """`Stack.get_reward` — the mean over pairs, and with two bricks there
        is exactly ONE pair: `(desired_order[0], desired_order[1])` = (0, 1),
        i.e. brick 1 stacked ON brick 0.

        ⚠ THE ORDER OF THE PAIR IS THE DIRECTION OF THE STACK. `pairs` is
        `zip(order[:-1], order[1:])`, so the first index is the BOTTOM brick,
        whose STUDS are measured, and the second is the top, whose HOLES are.
        Swapping them measures brick 0's holes against brick 1's studs — a
        different, equally plausible number.
        """
        var dist = min_stud_to_hole_distance[
            DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
        ](d, BRICK0_STUD_0, BRICK1_HOLE_0)
        return (Scalar[DTYPE](pairwise_stacking_reward(dist)), False)

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

    # === CPU: both bricks, then the settle, then the arm ==================
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
        """`Stack.initialize_episode` — bricks, grasp (already done), arm.

        ⚠ THE BRICKS GO FIRST, like `Lift` and unlike `Reach`. And the hint
        stack that the reference builds afterwards is renderer-only; see the
        module header.
        """
        comptime MAX_ATT: Int = 10
        comptime MAX_SAMP: Int = 10

        var lo_p = InlineArray[Float64, 3](fill=0.0)
        lo_p[0] = PROP_BBOX_LOWER_X
        lo_p[1] = PROP_BBOX_LOWER_Y
        lo_p[2] = PROP_BBOX_LOWER_Z
        var hi_p = InlineArray[Float64, 3](fill=0.0)
        hi_p[0] = PROP_BBOX_UPPER_X
        hi_p[1] = PROP_BBOX_UPPER_Y
        hi_p[2] = PROP_BBOX_UPPER_Z

        # ── 1. the FIXED base brick — a MODEL edit, with brick 1's contacts
        # switched off because it has not been placed yet.
        var poses0 = List[Scalar[DTYPE]]()
        for _ in range(MAX_PROP_ATTEMPTS):
            var dr = InlineArray[Float64, 3](fill=0.0)
            for k in range(3):
                dr[k] = random_float64()
            var pp = sample_bbox_uniform[DTYPE](lo_p, hi_p, dr)
            var pq = uniform_z_rotation[DTYPE](random_float64())
            for k in range(3):
                poses0.append(pp[k])
            for k in range(4):
                poses0.append(pq[k])
        var ignore1 = List[Int]()
        ignore1.append(BRICK1_BODY)
        var r0 = place_fixed_prop[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
        ](d, mf, BRICK0_BODY, 1, ignore1, poses0, MAX_PROP_ATTEMPTS)
        if not r0.success:
            raise Error(
                "stack_2_bricks: the base brick found no clear pose in "
                + String(r0.attempts)
                + " attempts"
            )

        # ── 2. the FREE brick, with every prop's contacts live again.
        var poses1 = List[Scalar[DTYPE]]()
        for _ in range(MAX_PROP_ATTEMPTS):
            var dr = InlineArray[Float64, 3](fill=0.0)
            for k in range(3):
                dr[k] = random_float64()
            var pp = sample_bbox_uniform[DTYPE](lo_p, hi_p, dr)
            var pq = uniform_z_rotation[DTYPE](random_float64())
            for k in range(3):
                poses1.append(pp[k])
            for k in range(4):
                poses1.append(pq[k])
        var r1 = place_free_prop[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
        ](
            d, mf, BRICK1_BODY, BRICK1_QPOS_ADR, BRICK1_DOF_ADR, poses1,
            List[Int](), False, MAX_PROP_ATTEMPTS,
        )
        if not r1.success:
            raise Error(
                "stack_2_bricks: the free brick found no non-colliding pose in "
                + String(r1.attempts)
                + " attempts"
            )

        # ── 3. settle. ⚠ Only brick 1 can move: brick 0 has no joint at all,
        # so the isolator's "hold everything but the props" reduces to holding
        # the robot's nine.
        _ = settle_free_prop[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
            Stack2BricksModel.CONE_TYPE,
            Stack2BricksModel.MAX_CONDIM,
            Stack2BricksModel.NOSLIP_ITER,
            N_ARM + N_HAND,
        ](d, mf, BRICK1_DOF_ADR, Self.get_timestep())

        # ── 4. the TCP initializer ──────────────────────────────────────
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

        # ⚠⚠ BRICK 0 IS `BODY_FIXED` AND BRICK 1 IS `BODY_FREE` — the same
        # brick model, two classes, because dm_control's predicate asks whether
        # the body's TOP-LEVEL body carries a freejoint and brick 0's does not
        # (this episode and every episode, since `randomize_order` is False).
        # An arm pose resting on the base brick is REJECTED; one resting on the
        # free brick is not.
        #
        # ⚠ THE HINT BRICKS (18, 20) STAY `BODY_FIXED`, which is harmless only
        # because they are contactless and so can never appear in a contact.
        var body_class = InlineArray[Int, NBODY](fill=BODY_FIXED)
        for b in range(NBODY):
            if b >= 2 and b <= 8:
                body_class[b] = BODY_ARM
            elif b >= 10 and b <= 16:
                body_class[b] = BODY_HAND
            elif b == BRICK1_BODY:
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
                "stack_2_bricks: the TCP initializer exhausted "
                + String(res.samples)
                + " samples ("
                + String(res.ik_failures)
                + " IK failures, "
                + String(res.collision_rejections)
                + " collision rejections)"
            )
