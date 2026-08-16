"""`dm_control` `dog fetch` — port of `suite/dog.py`'s `Fetch` (Phase 5).

    observation = Stand's 223 + ball_state(6) + target_position(3) = 232
    reward      = prod(Stand's 6 factors) * reach_ball * fetch_ball
    reset       = Stand's reset, then the ball is THROWN at the origin from
                  0.75 * the floor half-extent
    episode     = 1000 control steps, no early termination

`Fetch` subclasses `Stand`, so both the observation and the reward reuse dog's
existing pieces (`_dog_obs_cpu`, `_stand_factors`) rather than restating them —
the same relationship the reference has.

⚠ THE FRAME IS THE HEAD SITE'S, NOT A BODY'S, and the conversion is
WORLD -> SITE. The reference writes `v.dot(head_frame)` with
`head_frame = site_xmat['head'].reshape(3,3)`; under numpy's row-vector
convention that is `R^T v`. Transposing it looks equally plausible, is wrong,
and would leave the ball's direction mirrored in a way no scalar check
notices — quadruped fetch's port carries the same warning for the same reason.
`quat_rotate_inverse` on the SITE's world quaternion is exactly `R^T v`.

⚠ `ball_pos` IS A GEOM POSITION, NOT THE BALL BODY'S. The reference reads
`geom_xpos['ball']`; the geom sits at the body origin here so the two coincide
numerically, but writing `xpos[ball_body]` would be right by accident and would
break the day the geom gains an offset. Same for `target`, which is a geom on
the WORLD body and has no body position at all.

⚠ THE REWARD HAS A DISCONTINUITY, and a gate that never crosses it proves
nothing about it:

    if ball_to_target_distance < 2 * target_radius:  reach_ball = 1

Below 0.2 m the "bring your mouth to the ball" term is waived entirely — the
dog is meant to let go once the ball is delivered. A fixture that only samples
poses far from the target exercises the other branch every time.

Run the task with:
    from mojo_rl.envs.dm_control.dog import DMDogFetch
"""

from std.math import abs, sqrt, sin, cos, pi
from std.random import random_float64

from ....physics3d.fields import Data, Dims, DimsLike
from ....physics3d.kinematics.geom_xpos import geom_xpos
from ....physics3d.kinematics.site_frame import site_world_quat_list
from ....physics3d.kinematics.quat_math import quat_rotate_inverse
from ....physics3d.sensors.frame_vel import point_velocity_world
from ....envs.phyics3d_env import Phyics3dEnvConfig
from ..rewards import tolerance, SIGMOID_RECIPROCAL

from .dog_config import (
    _dog_obs_cpu,
    _stand_factors,
    _dog_reset_cpu,
    _randn_pair,
)
from .dog_xml import (
    DOG_FRAME_SKIP,
    DOG_MAX_STEPS,
    DOG_SKULL_BODY_IDX,
)
from .dog_fetch_xml import (
    dfp,
    DOG_FETCH_OBS_DIM,
    FETCH_BALL_BODY_IDX,
    FETCH_BALL_QPOS_0,
    FETCH_BALL_DOF_0,
    FETCH_GEOM_BALL,
    FETCH_GEOM_TARGET,
    FETCH_SITE_HEAD,
    FETCH_SITE_UPPER_BITE,
    FETCH_SITE_LOWER_BITE,
    FETCH_BITE_RADIUS,
    FETCH_TARGET_RADIUS,
    FETCH_BRING_MARGIN,
    FETCH_THROW_RADIUS,
    FETCH_THROW_HEIGHT_MAX,
    FETCH_THROW_SPEED_MAX,
    FETCH_BALL_SPAWN_Z,
)

# `bite_margin` is a literal 2 in the reference, not a model quantity.
comptime FETCH_BITE_MARGIN: Float64 = 2.0


def _head_site_quat[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_sites: List[Scalar[DTYPE]],
) raises -> Tuple[Float64, Float64, Float64, Float64]:
    """The `head` site's world quaternion — `xquat[skull] * site_quat`."""
    var q = site_world_quat_list[DTYPE](
        m_sites, d.xquat.data, DOG_SKULL_BODY_IDX, FETCH_SITE_HEAD
    )
    return (Float64(q[0]), Float64(q[1]), Float64(q[2]), Float64(q[3]))


def _world_to_head(
    q: Tuple[Float64, Float64, Float64, Float64],
    vx: Float64,
    vy: Float64,
    vz: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """`v.dot(head_frame)` — WORLD -> HEAD SITE. See the module warning."""
    var r = quat_rotate_inverse[DType.float64](
        q[0], q[1], q[2], q[3], vx, vy, vz
    )
    return (r[0], r[1], r[2])


def _ball_world_pos[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_geoms: List[Scalar[DTYPE]],
) -> Tuple[Float64, Float64, Float64]:
    """`geom_xpos['ball']`."""
    return geom_xpos[DTYPE](
        d, m_geoms, FETCH_GEOM_BALL
    )


def _target_world_pos[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_geoms: List[Scalar[DTYPE]],
) -> Tuple[Float64, Float64, Float64]:
    """`geom_xpos['target']` — a geom on the WORLD body, so this is its own
    local offset and does not move."""
    return geom_xpos[DTYPE](
        d, m_geoms, FETCH_GEOM_TARGET
    )


def _ball_to_mouth_distance[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    m_geoms: List[Scalar[DTYPE]],
) -> Float64:
    """`0.5 * (|ball - upper_bite| + |ball - lower_bite|)`.

    The MEAN of the two bite sites, not the distance to either — a ball
    between the jaws scores better than one touching only the upper lip.
    """
    var b = _ball_world_pos[DTYPE](
        d, m_geoms
    )
    var total = Float64(0)
    # Written out rather than iterating a temporary InlineArray, which does
    # not construct here. Two sites is not worth a container.
    for s in range(2):
        var site = FETCH_SITE_UPPER_BITE if s == 0 else FETCH_SITE_LOWER_BITE
        var dx = b[0] - Float64(d.site_xpos.data[site * 3 + 0])
        var dy = b[1] - Float64(d.site_xpos.data[site * 3 + 1])
        var dz = b[2] - Float64(d.site_xpos.data[site * 3 + 2])
        total += sqrt(dx * dx + dy * dy + dz * dz)
    return 0.5 * total


def _ball_to_target_distance[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    m_geoms: List[Scalar[DTYPE]],
) -> Float64:
    """`|geom_xpos['ball'] - geom_xpos['target']|`."""
    var b = _ball_world_pos[DTYPE](
        d, m_geoms
    )
    var t = _target_world_pos[DTYPE](
        d, m_geoms
    )
    var dx = b[0] - t[0]
    var dy = b[1] - t[1]
    var dz = b[2] - t[2]
    return sqrt(dx * dx + dy * dy + dz * dz)


def _fetch_factors[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    m_geoms: List[Scalar[DTYPE]],
) -> Tuple[Float64, Float64]:
    """`(reach_ball, fetch_ball)` — the two factors `Fetch` adds to `Stand`.

    Both use `sigmoid='reciprocal'`, which dog is the only domain to ask for.
    The affine rescalings are the reference's own and are NOT tolerances:
    `reach_ball` is squashed into [1/7, 1] and `fetch_ball` into [1/2, 1], so
    neither can zero the product on its own.
    """
    var mouth = _ball_to_mouth_distance[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
    ](d, m_geoms)
    var to_target = _ball_to_target_distance[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
    ](d, m_geoms)

    var reach = Float64(
        tolerance[SIGMOID_RECIPROCAL, DTYPE = DType.float64](
            mouth, 0.0, FETCH_BITE_RADIUS, FETCH_BITE_MARGIN
        )
    )
    reach = (6.0 * reach + 1.0) / 7.0

    var near = Float64(
        tolerance[SIGMOID_RECIPROCAL, DTYPE = DType.float64](
            to_target, 0.0, FETCH_TARGET_RADIUS, FETCH_BRING_MARGIN
        )
    )
    var fetch = (near + 1.0) / 2.0

    # ⚠ THE DISCONTINUITY — see the module docstring. Applied AFTER the
    # rescaling, exactly as the reference does: `reach_ball = 1`, not
    # `(6*1 + 1)/7`.
    if to_target < 2.0 * FETCH_TARGET_RADIUS:
        reach = 1.0

    return (reach, fetch)


struct DMDogFetchConfig(Phyics3dEnvConfig):
    """`Fetch` — catch the thrown ball and bring it to the target."""

    comptime FRAME_SKIP: Int = DOG_FRAME_SKIP
    comptime MAX_STEPS: Int = DOG_MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    # dm_control runs mj_step1 after the last substep.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # accelerometer + the four force sensors need `mj_rnePostConstraint`.
    comptime RNE_POST: Bool = True
    # As Stand: the dog spawns at qpos0's height and the solver settles it.
    comptime RESET_FIND_HEIGHT: Bool = False

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(dfp.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.0

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`Fetch.initialize_episode` — Stand's reset, then throw the ball.

        The ball is placed on a circle of radius `0.75 * floor_half` at height
        0.05 and given a velocity aimed at the origin:

            vertical   v = sqrt(2 g h),  h ~ U(0, 3)   (energy, not a speed)
            horizontal speed ~ U(0, 5) along -(sin a, cos a) plus N(0, 0.05)

        ⚠ THE VERTICAL COMPONENT IS DERIVED FROM A HEIGHT, not drawn directly:
        `mv^2/2 = mgh`. Drawing a speed uniformly instead would change the
        distribution of apex heights and so the whole task difficulty.
        """
        # ⚠ Takes only `d` — the yaw draw and the three root velocities need
        # no model tables. Stand's own `custom_reset_cpu` passes the same.
        _dog_reset_cpu[DTYPE, D](d)

        var azimuth = 2.0 * pi * random_float64()
        var sa = sin(azimuth)
        var ca = cos(azimuth)
        d.qpos.data[FETCH_BALL_QPOS_0 + 0] = Scalar[DTYPE](
            FETCH_THROW_RADIUS * sa
        )
        d.qpos.data[FETCH_BALL_QPOS_0 + 1] = Scalar[DTYPE](
            FETCH_THROW_RADIUS * ca
        )
        d.qpos.data[FETCH_BALL_QPOS_0 + 2] = Scalar[DTYPE](FETCH_BALL_SPAWN_Z)
        # Free-joint orientation is [w, x, y, z] — identity.
        d.qpos.data[FETCH_BALL_QPOS_0 + 3] = Scalar[DTYPE](1)
        d.qpos.data[FETCH_BALL_QPOS_0 + 4] = Scalar[DTYPE](0)
        d.qpos.data[FETCH_BALL_QPOS_0 + 5] = Scalar[DTYPE](0)
        d.qpos.data[FETCH_BALL_QPOS_0 + 6] = Scalar[DTYPE](0)

        var height = FETCH_THROW_HEIGHT_MAX * random_float64()
        # `-opt.gravity[2]`, i.e. +9.81 for the standard downward gravity.
        var g = 9.81
        var v_up = sqrt(2.0 * g * height)
        var speed = FETCH_THROW_SPEED_MAX * random_float64()
        # Pointing at the centre, with a little noise on the direction.
        # One Box-Muller pair gives both direction noises; `_randn_pair` is
        # dog's own draw, reused so fetch cannot drift from Stand's RNG.
        var noise = _randn_pair()
        var dir_x = -sa + 0.05 * noise[0]
        var dir_y = -ca + 0.05 * noise[1]

        d.qvel.data[FETCH_BALL_DOF_0 + 0] = Scalar[DTYPE](speed * dir_x)
        d.qvel.data[FETCH_BALL_DOF_0 + 1] = Scalar[DTYPE](speed * dir_y)
        d.qvel.data[FETCH_BALL_DOF_0 + 2] = Scalar[DTYPE](v_up)
        d.qvel.data[FETCH_BALL_DOF_0 + 3] = Scalar[DTYPE](0)
        d.qvel.data[FETCH_BALL_DOF_0 + 4] = Scalar[DTYPE](0)
        d.qvel.data[FETCH_BALL_DOF_0 + 5] = Scalar[DTYPE](0)

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
        """Stand's 223, then `ball_state` (6) and `target_position` (3)."""
        try:
            # ⚠ Five arguments, not seven: dog's base observation reads no
            # joint or geom tables.
            if not _dog_obs_cpu[DTYPE](d, m_bodies, m_sites, act, obs):
                return False

            var hq = _head_site_quat[DTYPE](d, m_sites)
            var hx = Float64(d.site_xpos.data[FETCH_SITE_HEAD * 3 + 0])
            var hy = Float64(d.site_xpos.data[FETCH_SITE_HEAD * 3 + 1])
            var hz = Float64(d.site_xpos.data[FETCH_SITE_HEAD * 3 + 2])
            var b = _ball_world_pos[DTYPE](d, m_geoms)

            # --- ball_state: position THEN velocity, both head-frame --------
            var rel = _world_to_head(hq, b[0] - hx, b[1] - hy, b[2] - hz)

            # `object_velocity` for a geom and for a site — the same rigid
            # transport, off different attach points. See
            # `point_velocity_world`.
            var v_ball = point_velocity_world[DTYPE](
                d.xvel.data, d.xangvel.data, d.xipos.data,
                FETCH_BALL_BODY_IDX,
                Scalar[DTYPE](b[0]), Scalar[DTYPE](b[1]), Scalar[DTYPE](b[2]),
            )
            var v_head = point_velocity_world[DTYPE](
                d.xvel.data, d.xangvel.data, d.xipos.data,
                DOG_SKULL_BODY_IDX,
                Scalar[DTYPE](hx), Scalar[DTYPE](hy), Scalar[DTYPE](hz),
            )
            var rel_v = _world_to_head(
                hq,
                Float64(v_ball[0]) - Float64(v_head[0]),
                Float64(v_ball[1]) - Float64(v_head[1]),
                Float64(v_ball[2]) - Float64(v_head[2]),
            )
            # ⚠ UNPACKED, NOT LOOPED. A Tuple subscript needs a COMPTIME
            # index, so `rel[k]` with a loop variable is "cannot use a dynamic
            # value in type parameter". quadruped's config carries the same
            # note; I wrote the loop anyway and the compiler caught it.
            obs.append(Scalar[DTYPE](rel[0]))
            obs.append(Scalar[DTYPE](rel[1]))
            obs.append(Scalar[DTYPE](rel[2]))
            obs.append(Scalar[DTYPE](rel_v[0]))
            obs.append(Scalar[DTYPE](rel_v[1]))
            obs.append(Scalar[DTYPE](rel_v[2]))

            # --- target_position --------------------------------------------
            var t = _target_world_pos[DTYPE](d, m_geoms)
            var rel_t = _world_to_head(hq, t[0] - hx, t[1] - hy, t[2] - hz)
            obs.append(Scalar[DTYPE](rel_t[0]))
            obs.append(Scalar[DTYPE](rel_t[1]))
            obs.append(Scalar[DTYPE](rel_t[2]))
        except:
            return False
        return True

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
        """`prod(Stand's factors) * reach_ball * fetch_ball`; never
        terminates early."""
        var r: Float64
        try:
            r = _stand_factors[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
                d, m_bodies, m_sites
            )
            var f = _fetch_factors[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_geoms)
            r = r * f[0] * f[1]
        except:
            r = 0.0
        return (Scalar[DTYPE](r), False)
