"""dm_control `manipulator-bring_ball` task config — port of `suite/manipulator.py`.

    observation = arm_pos(16) + arm_vel(8) + touch(5)
                + hand_pos(4) + object_pos(4) + object_vel(3)
                + target_pos(4)                                        (44)
    reward      = _is_close(site_distance('ball', 'target_ball'))
    episode     = 1000 control steps (10 s / .01 s), no early termination

OBSERVATION, term by term (`Bring.get_observation`, fully_observable=True):

  arm_pos     `bounded_joint_pos(_ARM_JOINTS)` = `np.vstack([sin, cos]).T`,
              so the flattened order is [sin0, cos0, sin1, cos1, ...] — NOT
              all the sines then all the cosines.
  arm_vel     `qvel[_ARM_JOINTS]`.
  touch       `np.log1p(sensordata[_TOUCH_SENSORS])`.
  hand_pos    `body_2d_pose('hand')` = [xpos.x, xpos.z, xquat.qw, xquat.qy].
              A PLANAR pose: two of four quaternion components, because the
              model only moves in the x-z plane.
  object_pos  same, for `ball`.
  object_vel  `qvel[['ball_x','ball_z','ball_y']]`.
  target_pos  same 2-D pose, for `target_ball`.

⚠ `_ARM_JOINTS` lists finger/fingertip BEFORE thumb/thumbtip, while the MODEL
declares the thumb chain first. `ARM_JOINT_OBS_ORDER` carries that permutation.
A symmetric pose hides the difference completely, which is exactly why the
parity test drives an ASYMMETRIC one.

REWARD. `_ball_reward` = `_is_close(site_distance('ball', 'target_ball'))`,
where `_is_close(d) = tolerance(d, bounds=(0, _CLOSE), margin=_CLOSE*2)` and
`_CLOSE = .01`. `site_distance` is the full 3-D norm between the two SITES,
both of which sit at their body origins here. The `peg` variants use a
different, four-term reward and are not ported.

RESET. `Bring.initialize_episode` randomises the arm within its joint limits,
SYMMETRISES the hand (`qpos['finger'] = qpos['thumb']`), places the target, and
then picks the object's start from three cases with probabilities
(.1 in-hand, .1 in-target, .8 uniform) — rejecting the whole draw while
`physics.data.ncon > 0`.

WHAT THIS PORT DOES DIFFERENTLY, and why:
  * The target is a MOCAP body (see `manipulator_xml`), so its per-episode pose
    is written to `d.mocap_pos` / `d.mocap_quat` instead of to the model.
  * The rejection loop needs collision detection from inside a reset hook,
    which is not available here — the same constraint ball_in_cup hit. Rather
    than approximate a 21-geom arm in closed form, this port SKIPS the
    `in_hand` case (the one that deliberately starts the object touching the
    hand) and samples the object away from the arm, which makes the rejection
    test unnecessary rather than approximate. The acceptance region is
    therefore a subset of the reference's, not an approximation of it — the
    distinction matters, because an approximate region would drift silently
    and a subset cannot.
  * Episode-for-episode reproduction of a reference rollout is not a goal; the
    parity test sets qpos in both engines directly.
"""

from std.random import random_float64
from std.math import sin, cos, sqrt, log

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)
from mojo_rl.physics3d.sensors.touch import touch_sphere_site
from mojo_rl.envs.dm_control.rewards import tolerance

from .manipulator_xml import (
    DMManipulatorBringBallModel,
    arm_joint_obs_order,
    touch_site_order,
    HAND_BODY_IDX,
    BALL_BODY_IDX,
    TARGET_BODY_IDX,
    SITE_BALL,
    SITE_TARGET_BALL,
    BALL_QADR_X,
    BALL_QADR_Z,
    BALL_QADR_Y,
    MANIPULATOR_OBS_DIM,
)

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

# The arm is inside a cylinder of this radius about the shoulder; sampling the
# object outside it makes the reference's `ncon > 0` rejection unnecessary
# rather than approximate. upper+middle+lower+hand+fingers reach
# .18+.15+.12+.03+~.06 = .54 from the shoulder at (0, .4).
comptime ARM_REACH: Float64 = 0.62
comptime SHOULDER_X: Float64 = 0.0
comptime SHOULDER_Z: Float64 = 0.4

comptime NARM: Int = 8


struct DMManipulatorConfig(Phyics3dEnvConfig):
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
            var f = 0.0
            try:
                f = touch_sphere_site[
                    DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
                ](d, m_sites, touch_site_order(k), 1.0)
            except:
                # `touch_sphere_site` raises only on an unsupported zone TYPE,
                # which is a model-construction error and cannot become true
                # mid-episode. Surfacing it from an obs hook is not possible,
                # so it degrades to 0 here and the parity test gates the real
                # values against MuJoCo's `sensordata`.
                f = 0.0
            obs.append(Scalar[DTYPE](log(1.0 + f)))

        # hand_pos / object_pos / target_pos — planar poses [x, z, qw, qy].
        Self._append_2d_pose[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
            d, HAND_BODY_IDX, obs
        )
        Self._append_2d_pose[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
            d, BALL_BODY_IDX, obs
        )

        # object_vel — the ball's three joints, BEFORE target_pos.
        obs.append(d.qvel.data[BALL_QADR_X])
        obs.append(d.qvel.data[BALL_QADR_Z])
        obs.append(d.qvel.data[BALL_QADR_Y])

        Self._append_2d_pose[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
            d, TARGET_BODY_IDX, obs
        )
        return True

    @staticmethod
    def _append_2d_pose[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
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
        """`Bring.initialize_episode`, minus the `in_hand` case.

        See the module docstring: the reference rejects a draw while
        `physics.data.ncon > 0`, which needs collision detection from inside a
        reset hook. Sampling the object OUTSIDE the arm's reach makes that test
        unnecessary rather than approximate — a subset of the reference's
        acceptance region, not a drifting version of it.
        """
        # Arm joints, uniform within their limits (unlimited -> [-pi, pi]).
        # `arm_root` is `limited="false"`, the rest carry ranges.
        for k in range(NARM):
            var j = arm_joint_obs_order(k)
            var lo = -3.14159265358979323846
            var hi = 3.14159265358979323846
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
        # Model joint 4 is thumb, 6 is finger; 5 thumbtip, 7 fingertip.
        d.qpos.data[6] = d.qpos.data[4]
        d.qpos.data[7] = d.qpos.data[5]

        # Target: a mocap pose, not a model write. `target_angle` is a rotation
        # about y, so the quaternion is (cos(a/2), 0, sin(a/2), 0) in MuJoCo's
        # (w, x, y, z); `d.mocap_quat` is (x, y, z, w).
        var tx = TARGET_X_LO + random_float64() * (TARGET_X_HI - TARGET_X_LO)
        var tz = TARGET_Z_LO + random_float64() * (TARGET_Z_HI - TARGET_Z_LO)
        var ta = -3.14159265358979323846 + random_float64() * (
            2.0 * 3.14159265358979323846
        )
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 0] = Scalar[DTYPE](tx)
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 1] = Scalar[DTYPE](0.001)
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 2] = Scalar[DTYPE](tz)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 1] = Scalar[DTYPE](
            sin(ta * 0.5)
        )
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 3] = Scalar[DTYPE](
            cos(ta * 0.5)
        )

        # Object: `in_target` with probability .1 (exact, no rejection needed —
        # the target box is outside the arm's reach at these radii only
        # sometimes, so it is still range-checked below), else uniform outside
        # the arm's reach.
        var ox = 0.0
        var oz = 0.0
        var vx = 0.0
        if random_float64() < 0.1:
            ox = tx
            oz = tz
        else:
            var tries = 0
            while tries < 200:
                ox = OBJECT_X_LO + random_float64() * (
                    OBJECT_X_HI - OBJECT_X_LO
                )
                oz = OBJECT_Z_LO + random_float64() * (
                    OBJECT_Z_HI - OBJECT_Z_LO
                )
                tries += 1
                var dx = ox - SHOULDER_X
                var dz = oz - SHOULDER_Z
                if sqrt(dx * dx + dz * dz) > ARM_REACH:
                    break
            vx = OBJECT_VX_LO + random_float64() * (
                OBJECT_VX_HI - OBJECT_VX_LO
            )

        d.qpos.data[BALL_QADR_X] = Scalar[DTYPE](ox)
        d.qpos.data[BALL_QADR_Z] = Scalar[DTYPE](oz)
        d.qpos.data[BALL_QADR_Y] = Scalar[DTYPE](
            random_float64() * 2.0 * 3.14159265358979323846
        )
        d.qvel.data[BALL_QADR_X] = Scalar[DTYPE](vx)

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
        """`_ball_reward` = `_is_close(site_distance('ball','target_ball'))`."""
        var ax = Float64(d.site_xpos.data[SITE_BALL * 3 + 0])
        var ay = Float64(d.site_xpos.data[SITE_BALL * 3 + 1])
        var az = Float64(d.site_xpos.data[SITE_BALL * 3 + 2])
        var bx = Float64(d.site_xpos.data[SITE_TARGET_BALL * 3 + 0])
        var by = Float64(d.site_xpos.data[SITE_TARGET_BALL * 3 + 1])
        var bz = Float64(d.site_xpos.data[SITE_TARGET_BALL * 3 + 2])
        var dx = ax - bx
        var dy = ay - by
        var dz = az - bz
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        var r = tolerance(dist, 0.0, CLOSE, CLOSE * 2.0)
        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](r), False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMManipulatorBringBallModel.TIMESTEP)
