"""`dm_control` `stacker` task config — port of `suite/stacker.py::Stack`.

ONE task class over two models. `Stack` is parameterised by `n_boxes` and so is
this config; the two registered tasks are

    stack_2   DMStackerConfig[2]     obs 49
    stack_4   DMStackerConfig[4]     obs 63

    episode = 1000 control steps (10 s / .01 s), no early termination

OBSERVATION, term by term (`Stack.get_observation`, fully_observable=True):

  arm_pos     `bounded_joint_pos(_ARM_JOINTS)` = `np.vstack([sin, cos]).T`, so
              the flattened order is [sin0, cos0, sin1, cos1, ...] — NOT all the
              sines then all the cosines.                                   (16)
  arm_vel     `qvel[_ARM_JOINTS]`.                                           (8)
  touch       `np.log1p(sensordata)`.                                        (5)
  hand_pos    `body_2d_pose('hand')` = [xpos.x, xpos.z, xquat.qw, xquat.qy].
              A PLANAR pose: two of four quaternion components, because the
              model only moves in the x-z plane.                             (4)
  box_pos     the same 2-D pose for each box, in box order.                (4n)
  box_vel     `qvel[box_joint_names]`.                                     (3n)
  target_pos  `body_2d_pose('target', orientation=False)` — POSITION ONLY, so
              two floats, not four. The target never rotates.               (2)

TWO PERMUTATIONS, both silent under a symmetric drive:

  * `_ARM_JOINTS` lists finger/fingertip BEFORE thumb/thumbtip while the model
    declares the thumb chain first (`arm_joint_obs_order`, shared with
    `manipulator`).
  * `box_joint_names` is built with `for dim in 'xyz'` while the model declares
    each box's joints x, z, y — so every box's velocity triple is TRANSPOSED in
    its last two entries (`box_vel_qadr`). ⚠ `manipulator` builds the same list
    with `'xzy'` and needs no permutation at all, so this is a difference
    between two otherwise parallel domains and not a shared convention.

REWARD (`Stack.get_reward`):

    box_size     = geom_size['target', 0]                              = .022
    box_is_close = tolerance(min_i dist(box_i, target), margin=2*box_size)
    hand_is_far  = tolerance(dist(grasp, target), bounds=(.1, inf), margin=.01)
    reward       = box_is_close * hand_is_far

both over full 3-D SITE distances. `box_is_close` takes the MINIMUM over the
boxes, so the task is "get ANY box to the target", not a specific one, and
`hand_is_far` is what stops a policy from scoring while still holding it —
the reward is zero until the hand has backed at least 10 cm away.

⚠ `box_is_close` has bounds (0, 0), so it is at 1 only when the distance is
exactly 0 and decays from the very first millimetre — unlike `manipulator`'s
`_is_close`, which has a .01 flat top. The two domains' "close" are not the
same function.

RESET (`Stack.initialize_episode`) randomises the arm within its joint limits,
symmetrises the hand, places the target at one of `n_boxes` discrete heights,
scatters the boxes, and REDRAWS THE WHOLE THING while `physics.data.ncon > 0`.

WHAT THIS PORT DOES DIFFERENTLY, and why:
  * The target is a MOCAP body (see `stacker_xml`), so its per-episode pose is
    written to `d.mocap_pos` / `d.mocap_quat` instead of to the model.
  * The reference's rejection loop needs full collision detection from inside a
    reset hook, which is not available there. Boxes are placed one at a time
    against a CLOSED-FORM clearance test instead — `arm_clearance` for the arm,
    a half-diagonal against the floor, and a centre-distance test against the
    boxes already placed. Every one of those is CONSERVATIVE, so the accepted
    region is a strict SUBSET of the reference's: everything this accepts, the
    reference would accept too. The distinction matters because an approximate
    region drifts silently and a subset cannot.
    What it is NOT is the same DISTRIBUTION: the reference redraws the arm and
    every box together on any collision, and this holds the arm fixed and
    redraws one box at a time. Both yield collision-free states; the marginals
    differ. Episode-for-episode reproduction of a reference rollout is not a
    goal, and the parity tests set qpos in both engines directly.
  * MEASURED, not assumed: 1.36 (stack_2) / 1.53 (stack_4) draws per box on
    average and zero exhaustions of the 200-try budget in 20 000 resets, so
    exhausting it is a ~1e-91 event. The first version of `manipulator`'s
    equivalent test was sound but accepted 0.13% of draws, which made 77% of its
    resets fall through the budget with an arm-penetrating placement — hence the
    measurement rather than an argument that the bound is conservative.
  * Only the walls the sampling box can actually reach are tested. `wall1`,
    `wall2` and `background` are planes, but over x in [.1, .3] and z in [0, .7]
    the nearest of them clears a box's half-diagonal by more than 3 cm at every
    point, so testing them would only cost time. That is a claim about THIS
    sampling box: widening it re-opens the question.
"""

from std.random import random_float64
from std.math import sin, cos, sqrt, log, pi, inf
from ..dtype_math import log1p_dt

from mojo_rl.physics3d.fields import Data, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)
from mojo_rl.physics3d.sensors.touch import touch_sphere_site

from ..rewards import tolerance
from ..planar_arm import (
    arm_clearance,
    arm_joint_obs_order,
    touch_site_order,
    HAND_BODY_IDX,
    NARM_JOINTS,
    SITE_GRASP,
)
from .stacker_xml import (
    BOX_SIZE,
    box_body_idx,
    box_site_idx,
    box_vel_qadr,
    target_body_idx,
    target_site_idx,
    stacker_obs_dim,
    BOX_QADR_0,
)

from ...phyics3d_env_config import Phyics3dEnvConfig


# `stacker.py` module constants.
comptime CLOSE: Float64 = 0.01  # _CLOSE
comptime CONTROL_TIMESTEP: Float64 = 0.01  # _CONTROL_TIMESTEP
comptime TIME_LIMIT: Float64 = 10.0  # _TIME_LIMIT

comptime NARM: Int = NARM_JOINTS

# `initialize_episode`: target_x ~ U(-.37, .37); target_z is DISCRETE, at
# `box_size * (2*randint(n_boxes) + 1)` — i.e. the centre height of a stack of
# 1, 3, 5 or 7 half-boxes, so a stack of 1, 2, 3 or 4 cubes. A stack_2 target is
# never higher than two cubes.
comptime TARGET_X_LO: Float64 = -0.37
comptime TARGET_X_HI: Float64 = 0.37
# `model.body_pos['target', 'y']` is untouched, so it keeps the XML's .001.
comptime TARGET_Y: Float64 = 0.001

# Box sampling box: x ~ U(.1, .3), z ~ U(0, .7), angle ~ U(0, 2*pi).
comptime BOX_X_LO: Float64 = 0.1
comptime BOX_X_HI: Float64 = 0.3
comptime BOX_Z_LO: Float64 = 0.0
comptime BOX_Z_HI: Float64 = 0.7

# In-plane bounding radius of a .022 cube. The box's only rotational DOF is the
# `_y` hinge, about the axis the arm lies in, so its x-z cross-section is a
# .044 square turned by an arbitrary angle and its half-DIAGONAL is the only
# sound bound. (Its y extent is fixed at +/- .022 about y=0, and every arm geom
# and the floor sit at y=0, so the planar test never needs a y term.)
comptime BOX_BOUND_RAD: Float64 = BOX_SIZE * 1.4142135623730951

# See the module docstring: measured at 1.36 / 1.53 draws per box.
comptime MAX_PLACEMENT_TRIES: Int = 200


struct DMStackerConfig[N_BOXES: Int](Phyics3dEnvConfig):
    # === Physics ===
    # _CONTROL_TIMESTEP .01 / timestep .001.
    comptime FRAME_SKIP: Int = 10
    # _TIME_LIMIT 10 s / .01 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # The observation reads `site_xpos` (touch zones) and body quaternions, so
    # FK must be refreshed after the frame-skip loop.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # No `<option integrator>`, so MuJoCo's Euler default applies.
    comptime INTEGRATOR: StaticString = "euler"

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
        """`Stack.get_observation` with `fully_observable=True`."""
        # arm_pos — (sin, cos) INTERLEAVED per joint, per `np.vstack(...).T`.
        for k in range(NARM):
            var q = Float64(d.qpos.data[arm_joint_obs_order(k)])
            obs.append(Scalar[DTYPE](sin(q)))
            obs.append(Scalar[DTYPE](cos(q)))

        # arm_vel
        for k in range(NARM):
            obs.append(d.qvel.data[arm_joint_obs_order(k)])

        # touch — log1p of the summed normal force per zone. `log1p` compresses
        # a signal whose raw range spans several decades; the reference feeds
        # exactly this to the policy, so the compression is part of the task.
        for k in range(5):
            var f: Float64
            try:
                f = touch_sphere_site[DTYPE](d, m_sites, touch_site_order(k), 1.0)
            except:
                # `touch_sphere_site` raises only on an unsupported zone TYPE,
                # which is a model-construction error and cannot become true
                # mid-episode. Surfacing it from an obs hook is not possible, so
                # it degrades to 0 here and the parity tests gate the real values
                # against MuJoCo's `sensordata`.
                f = 0.0
            obs.append(log1p_dt[DTYPE](Scalar[DTYPE](f)))

        # hand_pos, then every box's 2-D pose.
        Self._append_2d_pose[DTYPE](
            d, HAND_BODY_IDX, obs
        )
        for i in range(Self.N_BOXES):
            Self._append_2d_pose[DTYPE](
                d, box_body_idx(i), obs
            )

        # box_vel — all of box i's three joints before box i+1's, and x, y, z
        # within each (NOT the model's x, z, y).
        for i in range(Self.N_BOXES):
            for k in range(3):
                obs.append(d.qvel.data[box_vel_qadr(i, k)])

        # target_pos — position only.
        var tb = target_body_idx(Self.N_BOXES)
        obs.append(d.xpos.data[tb * 3 + 0])
        obs.append(d.xpos.data[tb * 3 + 2])
        return True

    @staticmethod
    def _append_2d_pose[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        body: Int,
        mut obs: List[Scalar[DTYPE]],
    ):
        """`Physics.body_2d_pose` — [xpos.x, xpos.z, xquat.qw, xquat.qy].

        ⚠ `d.xquat` is stored (x, y, z, w); the reference reads ['qw', 'qy'],
        i.e. MuJoCo's (w, x, y, z) slots 0 and 2. So this appends w then y.
        """
        obs.append(d.xpos.data[body * 3 + 0])
        obs.append(d.xpos.data[body * 3 + 2])
        obs.append(d.xquat.data[body * 4 + 3])  # w
        obs.append(d.xquat.data[body * 4 + 1])  # y

    # === CPU: Reset ===
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`Stack.initialize_episode`, with the rejection loop replaced as the
        module docstring describes."""
        # Arm joints, uniform within their limits (unlimited -> [-pi, pi]).
        # `arm_root` is `limited="false"`, the rest carry ranges.
        for k in range(NARM):
            var j = arm_joint_obs_order(k)
            var lo = -pi
            var hi = pi
            var rlo = Float64(
                m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN]
            )
            var rhi = Float64(
                m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX]
            )
            if rlo >= -1e9 and rhi <= 1e9:
                lo = rlo
                hi = rhi
            d.qpos.data[j] = Scalar[DTYPE](lo + random_float64() * (hi - lo))

        # `data.qpos['finger'] = data.qpos['thumb']` — symmetrise the hand.
        # Model joint 4 is thumb and 6 is finger. ⚠ The reference symmetrises
        # the KNUCKLES only: `thumbtip` (5) and `fingertip` (7) keep their own
        # independent draws, so the hand starts near-symmetric rather than
        # symmetric.
        d.qpos.data[6] = d.qpos.data[4]

        # Target: a mocap pose, not a model write. Its height is DISCRETE —
        # `box_size * (2*randint(n_boxes) + 1)` — and it is never rotated, so
        # the quaternion stays identity (which `reset_data` does NOT provide:
        # `d.mocap_quat` is all zeros until written).
        var h = Int(random_float64() * Float64(Self.N_BOXES))
        if h >= Self.N_BOXES:  # random_float64() == 1.0 exactly
            h = Self.N_BOXES - 1
        var tz = BOX_SIZE * Float64(2 * h + 1)
        var tx = TARGET_X_LO + random_float64() * (TARGET_X_HI - TARGET_X_LO)

        var tb = target_body_idx(Self.N_BOXES)
        d.mocap_pos.data[tb * 3 + 0] = Scalar[DTYPE](tx)
        d.mocap_pos.data[tb * 3 + 1] = Scalar[DTYPE](TARGET_Y)
        d.mocap_pos.data[tb * 3 + 2] = Scalar[DTYPE](tz)
        d.mocap_quat.data[tb * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[tb * 4 + 1] = Scalar[DTYPE](0)
        d.mocap_quat.data[tb * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[tb * 4 + 3] = Scalar[DTYPE](1)

        # Boxes, one at a time, each clear of the floor, the arm, and the boxes
        # already placed.
        var q0 = Float64(d.qpos.data[0])
        var q1 = Float64(d.qpos.data[1])
        var q2 = Float64(d.qpos.data[2])
        var q3 = Float64(d.qpos.data[3])
        for i in range(Self.N_BOXES):
            var ox = 0.0
            var oz = 0.0
            for _try in range(MAX_PLACEMENT_TRIES):
                ox = BOX_X_LO + random_float64() * (BOX_X_HI - BOX_X_LO)
                oz = BOX_Z_LO + random_float64() * (BOX_Z_HI - BOX_Z_LO)
                # Floor: the plane is at z = 0, so the centre has to clear it by
                # the half-diagonal for any hinge angle.
                if oz <= BOX_BOUND_RAD:
                    continue
                if arm_clearance(q0, q1, q2, q3, ox, oz) <= BOX_BOUND_RAD:
                    continue
                # Two cubes of half-diagonal r cannot overlap once their centres
                # are more than 2r apart, whatever their angles.
                var hit = False
                for j in range(i):
                    var jx = Float64(d.qpos.data[BOX_QADR_0 + 3 * j + 0])
                    var jz = Float64(d.qpos.data[BOX_QADR_0 + 3 * j + 1])
                    var dx = ox - jx
                    var dz = oz - jz
                    if sqrt(dx * dx + dz * dz) <= 2.0 * BOX_BOUND_RAD:
                        hit = True
                        break
                if hit:
                    continue
                break

            # The slide joints' `ref` equals their body `pos`, so the joint value
            # IS the world coordinate and these three writes are the pose.
            d.qpos.data[BOX_QADR_0 + 3 * i + 0] = Scalar[DTYPE](ox)
            d.qpos.data[BOX_QADR_0 + 3 * i + 1] = Scalar[DTYPE](oz)
            d.qpos.data[BOX_QADR_0 + 3 * i + 2] = Scalar[DTYPE](
                random_float64() * 2.0 * pi
            )

    # === CPU: Reward ===
    @staticmethod
    def _site_distance[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        s1: Int,
        s2: Int,
    ) -> Float64:
        """`Physics.site_distance` — the full 3-D norm, not the planar one."""
        var dx = Float64(d.site_xpos.data[s1 * 3 + 0]) - Float64(
            d.site_xpos.data[s2 * 3 + 0]
        )
        var dy = Float64(d.site_xpos.data[s1 * 3 + 1]) - Float64(
            d.site_xpos.data[s2 * 3 + 1]
        )
        var dz = Float64(d.site_xpos.data[s1 * 3 + 2]) - Float64(
            d.site_xpos.data[s2 * 3 + 2]
        )
        return sqrt(dx * dx + dy * dy + dz * dz)

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
        """`Stack.get_reward` — `box_is_close * hand_is_far`."""
        var s_tgt = target_site_idx(Self.N_BOXES)

        var min_d = inf[DType.float64]()
        for i in range(Self.N_BOXES):
            var dist = Self._site_distance[DTYPE](d, box_site_idx(i), s_tgt)
            if dist < min_d:
                min_d = dist
        # bounds (0, 0): at 1 only at zero distance, decaying immediately.
        var box_is_close = tolerance(min_d, 0.0, 0.0, 2.0 * BOX_SIZE)

        var hand_d = Self._site_distance[DTYPE](d, SITE_GRASP, s_tgt)
        var hand_is_far = tolerance(hand_d, 0.1, inf[DType.float64](), CLOSE)

        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](box_is_close * hand_is_far), False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        # Both variants are built from the same `<option timestep="0.001">`.
        return 0.001


# The two registered tasks.
comptime DMStacker2Config = DMStackerConfig[2]
comptime DMStacker4Config = DMStackerConfig[4]
