"""dm_control `ball_in_cup-catch` task config — port of `suite/ball_in_cup.py`.

    observation = [qpos(4), qvel(4)]                                     (8)
    reward      = in_target()  -- SPARSE, exactly 0 or 1
    reset       = ball_x ~ U(-.2, .2), ball_z ~ U(.2, .5), rejected while
                  the ball starts in contact
    episode     = 1000 control steps (20 s / .02 s), no early termination

`in_target` is a BOX test in the x-z plane, not a radial one:

    |site_xpos[target] - xpos[ball]|  <  site_size[target] - geom_size[ball]

componentwise over (x, z), i.e. |dx| < .025 and |dz| < .025. Note the two
operands come from different tables — the TARGET is a SITE, the BALL is a
BODY (`named.data.xpos['ball']`, not the geom) — and this port keeps that
split rather than using the ball geom, which sits at the body origin here but
need not in general.

CONTROL TIMESTEP. `_CONTROL_TIMESTEP = .02` against a `.002` physics step
gives FRAME_SKIP = 10, unlike point_mass (which passes no control timestep and
so runs 1:1).
"""

from std.random import random_float64
from std.math import sqrt, abs
from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
)

from .ball_in_cup_xml import (
    DMBallInCupModel,
    BALL_BODY_IDX,
    TARGET_SITE_IDX,
    BALL_GEOM_IDX,
    CUP_GEOM_FIRST,
    CUP_GEOM_LAST,
    TARGET_HALF_X,
    TARGET_HALF_Z,
    BALL_RADIUS,
)

from ...phyics3d_env_config import Phyics3dEnvConfig


# Ball spawn box, `BallInCup.initialize_episode`.
comptime SPAWN_X_LO: Float64 = -0.2
comptime SPAWN_X_HI: Float64 = 0.2
comptime SPAWN_Z_LO: Float64 = 0.2
comptime SPAWN_Z_HI: Float64 = 0.5

# qpos layout: [cup_x, cup_z, ball_x, ball_z]
comptime QADR_BALL_X: Int = 2
comptime QADR_BALL_Z: Int = 3


struct DMBallInCupConfig(Phyics3dEnvConfig):
    # === Physics ===
    # _CONTROL_TIMESTEP .02 / timestep .002.
    comptime FRAME_SKIP: Int = 10
    # _DEFAULT_TIME_LIMIT 20 s / .02 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
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
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """`BallInCup.get_observation`: position then velocity, both whole."""
        for i in range(NQ):
            obs.append(d.qpos.data[i])
        for i in range(NV):
            obs.append(d.qvel.data[i])
        return True

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
        """`BallInCup.initialize_episode` — resample the ball until it starts
        out of contact.

        The reference calls `physics.after_reset()` and tests `data.ncon > 0`.
        Running collision detection from inside a reset hook is not available
        here, so the rejection test is done in closed form: with the cup at
        its own qpos0 the only geoms the ball can reach are the five cup
        capsules, and a sphere-capsule contact exists exactly when the
        centre-to-segment distance is below the radius sum. The ground is
        unreachable (ball z >= .2 - .025) and the cup cannot touch it either.

        Consequence of the closed form: it is the SAME acceptance region, but
        the accepted sample is not the same draw as dm_control's for a given
        seed. Episode-for-episode reproduction of a reference rollout is not
        a goal of this port; the parity test sets qpos in both engines.
        """
        var tries = 0
        while tries < 100:
            var bx = SPAWN_X_LO + random_float64() * (SPAWN_X_HI - SPAWN_X_LO)
            var bz = SPAWN_Z_LO + random_float64() * (SPAWN_Z_HI - SPAWN_Z_LO)
            tries += 1

            # Cup body is at its qpos0 here (cup_x = cup_z = 0), so the
            # capsule endpoints are their local `fromto` plus the body origin,
            # which FK has already written into d.xpos.
            var cup_x = Float64(d.xpos.data[1 * 3 + 0])
            var cup_y = Float64(d.xpos.data[1 * 3 + 1])
            var cup_z = Float64(d.xpos.data[1 * 3 + 2])

            var hit = False
            for g in range(CUP_GEOM_FIRST, CUP_GEOM_LAST + 1):
                var o = g * MODEL_GEOM_SIZE
                var r = Float64(m_geoms[o + GEOM_IDX_RADIUS])
                var hl = Float64(m_geoms[o + GEOM_IDX_HALF_LENGTH])
                # Capsule axis in the body frame: local z rotated by the geom
                # quaternion, scaled by the half-length.
                var qx = Float64(m_geoms[o + GEOM_IDX_QUAT_X])
                var qy = Float64(m_geoms[o + GEOM_IDX_QUAT_Y])
                var qz = Float64(m_geoms[o + GEOM_IDX_QUAT_Z])
                var qw = Float64(m_geoms[o + GEOM_IDX_QUAT_W])
                var ax = 2.0 * (qx * qz + qw * qy)
                var ay = 2.0 * (qy * qz - qw * qx)
                var az = 1.0 - 2.0 * (qx * qx + qy * qy)

                var cx = cup_x + Float64(m_geoms[o + GEOM_IDX_POS_X])
                var cy = cup_y + Float64(m_geoms[o + GEOM_IDX_POS_Y])
                var cz = cup_z + Float64(m_geoms[o + GEOM_IDX_POS_Z])

                # Closest point on the segment [c - hl*a, c + hl*a] to (bx,0,bz).
                var px = bx - cx
                var py = 0.0 - cy
                var pz = bz - cz
                var t = px * ax + py * ay + pz * az
                if t > hl:
                    t = hl
                elif t < -hl:
                    t = -hl
                var dx = px - t * ax
                var dy = py - t * ay
                var dz = pz - t * az
                var dist = sqrt(dx * dx + dy * dy + dz * dz)
                if dist < r + BALL_RADIUS:
                    hit = True
                    break

            if not hit:
                d.qpos.data[QADR_BALL_X] = Scalar[DTYPE](bx)
                d.qpos.data[QADR_BALL_Z] = Scalar[DTYPE](bz)
                return

        # Fell through 100 rejections (the spawn box is mostly free, so this
        # is unreachable in practice). Park the ball at the box centre, which
        # is well clear of every capsule.
        d.qpos.data[QADR_BALL_X] = Scalar[DTYPE](0.0)
        d.qpos.data[QADR_BALL_Z] = Scalar[DTYPE](0.35)

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
        """`Physics.in_target()` — 1.0 inside the target box, else 0.0."""
        # `named.data.site_xpos['target', ['x', 'z']]`
        var tx = Float64(d.site_xpos.data[TARGET_SITE_IDX * 3 + 0])
        var tz = Float64(d.site_xpos.data[TARGET_SITE_IDX * 3 + 2])
        # `named.data.xpos['ball', ['x', 'z']]` — the BODY, not the geom.
        var bx = Float64(d.xpos.data[BALL_BODY_IDX * 3 + 0])
        var bz = Float64(d.xpos.data[BALL_BODY_IDX * 3 + 2])

        var in_target = (
            abs(tx - bx) < TARGET_HALF_X - BALL_RADIUS
            and abs(tz - bz) < TARGET_HALF_Z - BALL_RADIUS
        )

        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](1.0) if in_target else Scalar[DTYPE](0.0), False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMBallInCupModel.TIMESTEP)
