"""`dm_control` `manipulator` task config — port of `suite/manipulator.py::Bring`.

ONE task class over four models. `Bring` is parameterised by `(use_peg,
insert)` and so is this config; the four registered tasks are

    bring_ball   DMManipulatorConfig[False, False]
    bring_peg    DMManipulatorConfig[True,  False]
    insert_ball  DMManipulatorConfig[False, True ]
    insert_peg   DMManipulatorConfig[True,  True ]

    observation = arm_pos(16) + arm_vel(8) + touch(5)
                + hand_pos(4) + object_pos(4) + object_vel(3)
                + target_pos(4)                                        (44)
    episode     = 1000 control steps (10 s / .01 s), no early termination

WHAT EACH FLAG CHANGES
  use_peg  the MODEL (peg + target_peg instead of ball + target_ball, which
           renumbers everything after the arm) and the REWARD (`_peg_reward`'s
           four terms instead of `_ball_reward`'s one).
  insert   the MODEL again (a `cup`/`slot` receptacle between the prop and the
           target) and the RESET (the receptacle is posed with the target, and
           the target angle narrows from U(-pi, pi) to U(-pi/3, pi/3)).
           It does NOT change the reward: `get_reward` never mentions the
           receptacle, so inserting is rewarded only through bringing.

OBSERVATION, term by term (`Bring.get_observation`, fully_observable=True):

  arm_pos     `bounded_joint_pos(_ARM_JOINTS)` = `np.vstack([sin, cos]).T`,
              so the flattened order is [sin0, cos0, sin1, cos1, ...] — NOT
              all the sines then all the cosines.
  arm_vel     `qvel[_ARM_JOINTS]`.
  touch       `np.log1p(sensordata[_TOUCH_SENSORS])`.
  hand_pos    `body_2d_pose('hand')` = [xpos.x, xpos.z, xquat.qw, xquat.qy].
              A PLANAR pose: two of four quaternion components, because the
              model only moves in the x-z plane.
  object_pos  same, for the prop.
  object_vel  `qvel` of the prop's three joints.
  target_pos  same 2-D pose, for the target.

⚠ `_ARM_JOINTS` lists finger/fingertip BEFORE thumb/thumbtip, while the MODEL
declares the thumb chain first. `arm_joint_obs_order` carries that permutation.
A symmetric pose hides the difference completely, which is exactly why the
parity tests drive an ASYMMETRIC one.

REWARD. `_is_close(d) = tolerance(d, bounds=(0, _CLOSE), margin=_CLOSE*2)` with
`_CLOSE = .01`, over full 3-D distances between SITES.

    _ball_reward = _is_close(dist(ball, target_ball))
    _peg_reward  = max(bringing, grasping / 3), where
                     grasping = mean(_is_close(dist(peg_grasp, grasp)),
                                     _is_close(dist(peg_pinch, pinch)))
                     bringing = mean(_is_close(dist(peg, target_peg)),
                                     _is_close(dist(target_peg_tip, peg_tip)))

The peg form is a shaped reward: holding the peg correctly is worth at most
1/3, and actually delivering it dominates as soon as either bring term lifts
off the floor. The `/3` is what keeps a policy from parking in the grasp.

RESET. `Bring.initialize_episode` randomises the arm within its joint limits,
SYMMETRISES the hand (`qpos['finger'] = qpos['thumb']`), places the target (and
the receptacle, when inserting), and then picks the object's start from three
cases with probabilities (.1 in-hand, .1 in-target, .8 uniform) — REJECTING the
whole draw while `physics.data.ncon > 0`.

WHAT THIS PORT DOES DIFFERENTLY, and why:
  * The target and the receptacle are MOCAP bodies (see `manipulator_xml`), so
    their per-episode poses are written to `d.mocap_pos` / `d.mocap_quat`
    instead of to the model.
  * The reference's rejection loop needs full collision detection from inside a
    reset hook, which is not available here — the same constraint ball_in_cup
    hit. Instead the object is placed with a CLOSED-FORM clearance test against
    the arm (`planar_arm.arm_clearance`), which is exact for the three arm links
    and conservative for the hand. The acceptance region is therefore a SUBSET
    of the reference's rather than an approximation of it: everything this
    accepts, the reference would also accept. The distinction matters, because
    an approximate region drifts silently and a subset cannot.
  * The `in_hand` case (which deliberately starts the object touching the hand,
    and needs `site_xmat['grasp']` mid-reset) is SKIPPED. `in_target` is
    skipped for the INSERT tasks only, where it places the object inside the
    receptacle: for insert_ball the ball's .022 radius overlaps the cup wall by
    ~4 mm at the cup origin, so the reference rejects that draw anyway.
  * Episode-for-episode reproduction of a reference rollout is not a goal; the
    parity tests set qpos in both engines directly.

⚠ THE FIRST VERSION OF THIS RESET was near-degenerate and its replacement is
the reason `arm_clearance` exists. It rejected any object within ARM_REACH
(.62 m, the arm's FULL extension) of the shoulder — a sound bound, but one that
accepts only 0.13% of draws from the reference's sampling box, so 77% of resets
exhausted all 200 tries and fell through with an arbitrary, usually
arm-penetrating, placement. Measured, not estimated. The clearance test accepts
88% (ball) and 70% (peg) and gives the same guarantee, because it tests the arm
where the arm actually IS rather than where it could reach.
"""

from std.random import random_float64
from std.math import sin, cos, sqrt, log, pi
from ..dtype_math import log1p_dt

from mojo_rl.physics3d.fields import Data, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)
from mojo_rl.physics3d.sensors.touch import touch_sphere_site
from mojo_rl.envs.dm_control.rewards import tolerance

from .manipulator_xml import (
    arm_joint_obs_order,
    touch_site_order,
    target_body_idx,
    receptacle_body_idx,
    site_object,
    site_object_pinch,
    site_object_grasp,
    site_object_tip,
    site_target,
    site_target_tip,
    HAND_BODY_IDX,
    OBJECT_BODY_IDX,
    OBJECT_QADR_X,
    OBJECT_QADR_Z,
    OBJECT_QADR_Y,
    SITE_GRASP,
    SITE_PINCH,
    NARM_JOINTS,
    MANIPULATOR_OBS_DIM,
)

from ..planar_arm import arm_clearance

from ...phyics3d_env_config import Phyics3dEnvConfig


# `manipulator.py` module constants.
comptime CLOSE: Float64 = 0.01  # _CLOSE
comptime CONTROL_TIMESTEP: Float64 = 0.01  # _CONTROL_TIMESTEP
comptime TIME_LIMIT: Float64 = 10.0  # _TIME_LIMIT

# Target box, `initialize_episode`: target_x ~ U(-.4, .4), target_z ~ U(.1, .4)
comptime TARGET_X_LO: Float64 = -0.4
comptime TARGET_X_HI: Float64 = 0.4
comptime TARGET_Z_LO: Float64 = 0.1
comptime TARGET_Z_HI: Float64 = 0.4

# Object box, the `uniform` case: object_x ~ U(-.5, .5), object_z ~ U(0, .7)
comptime OBJECT_X_LO: Float64 = -0.5
comptime OBJECT_X_HI: Float64 = 0.5
comptime OBJECT_Z_LO: Float64 = 0.0
comptime OBJECT_Z_HI: Float64 = 0.7
# `data.qvel[object + '_x'] = uniform(-5, 5)` in the uniform case.
comptime OBJECT_VX_LO: Float64 = -5.0
comptime OBJECT_VX_HI: Float64 = 5.0

comptime NARM: Int = NARM_JOINTS

# Bounding radius of the prop about its own origin, which is what the clearance
# is compared against. The ball is its geom radius. The peg's origin is at the
# blade's TOP and it hangs .113 below with a .005 capsule, so it needs a much
# larger ball — and its `peg_y` hinge means the orientation is random, so a
# sphere is the only sound bound.
comptime BALL_BOUND_RAD: Float64 = 0.022
comptime PEG_BOUND_RAD: Float64 = 0.12

# Keep-out disc about the receptacle origin, for the INSERT tasks: the
# receptacle is posed at the target and the object is not, so a uniform draw
# could otherwise land inside it. Sized from the receptacle's own extent plus
# the prop's bound. `cup` spans ~.07 from its origin, `slot` ~.16.
comptime CUP_CLEAR_RAD: Float64 = 0.07 + BALL_BOUND_RAD
comptime SLOT_CLEAR_RAD: Float64 = 0.16 + PEG_BOUND_RAD

# `initialize_episode`'s draw budget. The measured acceptance rates are 88%
# (ball) and 70% (peg), so exhausting this is a ~1e-104 event; it exists to
# bound the loop, not because it is expected to fire.
comptime MAX_PLACEMENT_TRIES: Int = 200


struct DMManipulatorConfig[USE_PEG: Bool, INSERT: Bool](Phyics3dEnvConfig):
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

    # Prop bounding radius and receptacle keep-out, selected by the flag rather
    # than branched at every use.
    comptime OBJ_RAD: Float64 = PEG_BOUND_RAD if Self.USE_PEG else BALL_BOUND_RAD
    comptime RECEPTACLE_RAD: Float64 = SLOT_CLEAR_RAD if Self.USE_PEG else CUP_CLEAR_RAD

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
        """`Bring.get_observation` with `fully_observable=True`."""
        # arm_pos — (sin, cos) INTERLEAVED per joint, per `np.vstack(...).T`.
        for k in range(NARM):
            var j = arm_joint_obs_order(k)
            var q = Float64(d.qpos.data[j])
            obs.append(Scalar[DTYPE](sin(q)))
            obs.append(Scalar[DTYPE](cos(q)))

        # arm_vel
        for k in range(NARM):
            obs.append(d.qvel.data[arm_joint_obs_order(k)])

        # touch — log1p of the summed normal force per zone. `log1p` compresses
        # a signal whose raw range spans several decades; the reference feeds
        # exactly this to the policy, so the compression is part of the task,
        # not a convenience.
        for k in range(5):
            var f: Float64
            try:
                f = touch_sphere_site[DTYPE](d, m_sites, touch_site_order(k), 1.0)
            except:
                # `touch_sphere_site` raises only on an unsupported zone TYPE,
                # which is a model-construction error and cannot become true
                # mid-episode. Surfacing it from an obs hook is not possible,
                # so it degrades to 0 here and the parity tests gate the real
                # values against MuJoCo's `sensordata`.
                f = 0.0
            obs.append(log1p_dt[DTYPE](Scalar[DTYPE](f)))

        # hand_pos / object_pos / target_pos — planar poses [x, z, qw, qy].
        Self._append_2d_pose[DTYPE](
            d, HAND_BODY_IDX, obs
        )
        Self._append_2d_pose[DTYPE](
            d, OBJECT_BODY_IDX, obs
        )

        # object_vel — the prop's three joints, BEFORE target_pos.
        obs.append(d.qvel.data[OBJECT_QADR_X])
        obs.append(d.qvel.data[OBJECT_QADR_Z])
        obs.append(d.qvel.data[OBJECT_QADR_Y])

        Self._append_2d_pose[DTYPE](
            d, target_body_idx(Self.USE_PEG, Self.INSERT), obs
        )
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
        """`Bring.initialize_episode`, minus the cases named in the docstring.
        """
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
        # Model joint 4 is thumb and 6 is finger.
        #
        # ⚠ THE KNUCKLES ONLY. `initialize_episode` has exactly this one line;
        # `thumbtip` (5) and `fingertip` (7) keep their own independent draws,
        # so the hand starts NEAR-symmetric rather than symmetric. This used to
        # copy 5 -> 7 as well, which is not in the reference; caught while
        # porting stacker, whose `initialize_episode` has the same single line.
        # No parity test moved, because the parity tests set qpos directly and
        # never run a reset — an infidelity here is invisible to every gate in
        # the suite and only shows up as a slightly wrong initial-state
        # distribution during training.
        d.qpos.data[6] = d.qpos.data[4]

        # Target: a mocap pose, not a model write. `target_angle` is a rotation
        # about y, so the quaternion is (cos(a/2), 0, sin(a/2), 0) in MuJoCo's
        # (w, x, y, z); `d.mocap_quat` is (x, y, z, w).
        #
        # The INSERT tasks narrow the angle to U(-pi/3, pi/3), because the
        # receptacle takes the same angle and an upside-down slot is not a task
        # any policy could solve.
        var tx = TARGET_X_LO + random_float64() * (TARGET_X_HI - TARGET_X_LO)
        var tz = TARGET_Z_LO + random_float64() * (TARGET_Z_HI - TARGET_Z_LO)
        var a_half = pi / 3.0 if Self.INSERT else pi
        var ta = -a_half + random_float64() * (2.0 * a_half)

        var tb = target_body_idx(Self.USE_PEG, Self.INSERT)
        Self._set_mocap_2d[DTYPE, D](
            d, tb, tx, 0.001, tz, ta
        )

        comptime if Self.INSERT:
            # `model.body_pos[receptacle, ['x','z']] = target_x, target_z`, with
            # `y` left at the XML value (0 for both `cup` and `slot`) — the
            # target's own y is .001 so the ghost renders in front, and the
            # receptacle has no such offset.
            Self._set_mocap_2d[DTYPE, D](
                d, receptacle_body_idx(Self.USE_PEG), tx, 0.0, tz, ta
            )

        # Object placement. `in_hand` is skipped in every task, `in_target` in
        # the INSERT ones (see the module docstring); what remains is
        # `in_target` at its reference probability .1, else uniform subject to
        # the arm-clearance test.
        var ox = 0.0
        var oz = 0.0
        var oa: Float64
        var vx = 0.0
        var in_target = False
        comptime if not Self.INSERT:
            in_target = random_float64() < 0.1

        if in_target:
            ox = tx
            oz = tz
            oa = ta
        else:
            var q0 = Float64(d.qpos.data[0])
            var q1 = Float64(d.qpos.data[1])
            var q2 = Float64(d.qpos.data[2])
            var q3 = Float64(d.qpos.data[3])
            for _try in range(MAX_PLACEMENT_TRIES):
                ox = OBJECT_X_LO + random_float64() * (
                    OBJECT_X_HI - OBJECT_X_LO
                )
                oz = OBJECT_Z_LO + random_float64() * (
                    OBJECT_Z_HI - OBJECT_Z_LO
                )
                if arm_clearance(q0, q1, q2, q3, ox, oz) <= Self.OBJ_RAD:
                    continue
                comptime if Self.INSERT:
                    var rx = ox - tx
                    var rz = oz - tz
                    if sqrt(rx * rx + rz * rz) <= Self.RECEPTACLE_RAD:
                        continue
                break
            oa = random_float64() * 2.0 * pi
            vx = OBJECT_VX_LO + random_float64() * (OBJECT_VX_HI - OBJECT_VX_LO)

        d.qpos.data[OBJECT_QADR_X] = Scalar[DTYPE](ox)
        d.qpos.data[OBJECT_QADR_Z] = Scalar[DTYPE](oz)
        d.qpos.data[OBJECT_QADR_Y] = Scalar[DTYPE](oa)
        d.qvel.data[OBJECT_QADR_X] = Scalar[DTYPE](vx)

    @staticmethod
    def _set_mocap_2d[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        body: Int,
        x: Float64,
        y: Float64,
        z: Float64,
        angle: Float64,
    ):
        """One planar mocap pose: position (x, y, z) and a rotation of `angle`
        about the y axis.

        `d.mocap_quat` is (x, y, z, w); the reference writes MuJoCo's
        `['qw','qy']` = (cos(a/2), sin(a/2)) and leaves qx/qz alone, which are
        zero for every body this is called on.
        """
        d.mocap_pos.data[body * 3 + 0] = Scalar[DTYPE](x)
        d.mocap_pos.data[body * 3 + 1] = Scalar[DTYPE](y)
        d.mocap_pos.data[body * 3 + 2] = Scalar[DTYPE](z)
        d.mocap_quat.data[body * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[body * 4 + 1] = Scalar[DTYPE](sin(angle * 0.5))
        d.mocap_quat.data[body * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[body * 4 + 3] = Scalar[DTYPE](cos(angle * 0.5))

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
    def _is_close(dist: Float64) -> Float64:
        """`Bring._is_close` — `tolerance(d, (0, _CLOSE), _CLOSE*2)`."""
        return tolerance(dist, 0.0, CLOSE, CLOSE * 2.0)

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
        """`Bring.get_reward` — `_peg_reward` or `_ball_reward`."""
        var s_obj = site_object(Self.USE_PEG)
        var s_tgt = site_target(Self.USE_PEG, Self.INSERT)
        var bring = Self._is_close(
            Self._site_distance[DTYPE](
                d, s_obj, s_tgt
            )
        )

        var r = bring
        comptime if Self.USE_PEG:
            var grasp = Self._is_close(
                Self._site_distance[DTYPE](
                    d, site_object_grasp(Self.USE_PEG), SITE_GRASP
                )
            )
            var pinch = Self._is_close(
                Self._site_distance[DTYPE](
                    d, site_object_pinch(Self.USE_PEG), SITE_PINCH
                )
            )
            var grasping = (grasp + pinch) / 2.0
            var bring_tip = Self._is_close(
                Self._site_distance[DTYPE](
                    d,
                    site_target_tip(Self.USE_PEG, Self.INSERT),
                    site_object_tip(Self.USE_PEG),
                )
            )
            var bringing = (bring + bring_tip) / 2.0
            r = bringing if bringing > grasping / 3.0 else grasping / 3.0

        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](r), False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        # Every variant is built from the same `<option timestep="0.001">`.
        return 0.001


# The four registered tasks.
comptime DMManipulatorBringBallConfig = DMManipulatorConfig[False, False]
comptime DMManipulatorBringPegConfig = DMManipulatorConfig[True, False]
comptime DMManipulatorInsertBallConfig = DMManipulatorConfig[False, True]
comptime DMManipulatorInsertPegConfig = DMManipulatorConfig[True, True]
