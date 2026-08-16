"""`dm_control` `quadruped fetch` — port of `suite/quadruped.py`'s `Fetch`.

    observation = _common_observations(78) + ball_state(9) + target_position(3)
    reward      = _upright_reward * reach_reward * (0.5 + 0.5*fetch_reward)
    reset       = random azimuth + horizontal position, raised until nothing
                  touches; then the ball is dropped from z=2 with a random
                  horizontal kick
    episode     = 1000 control steps (20 s / 0.02 s), no early termination

WHY THIS TASK NEEDED ENGINE WORK FIRST. It is the only model in the tree with

  * `condim="6"` — until 004fe439 the pyramidal edge builder that Newton
    consumes hardcoded four edges per contact and never read `condim` or
    `friction[2..4]`, so the ball's torsional and rolling rows were written
    into a workspace nothing read. It would have spun and rolled unopposed,
    and `condim=3` agreeing to 2e-15 everywhere else is exactly why that went
    unnoticed. Gate: tests/physics3d/test_rolling_friction_vs_mujoco.mojo.
  * FOUR TILTED PLANE WALLS (`class="wall"` + `zaxis`) — a plane whose normal
    is not +z used to be treated as though it were, so the ball would have
    rolled straight through the arena boundary.

⚠ THE THREE FRAME CONVERSIONS ALL USE `xmat['torso']` UNTRANSPOSED, and the
reference writes them as `v.dot(torso_frame)`. With numpy's row-vector
convention that is `R^T v`, i.e. WORLD -> BODY. Transposing it looks equally
plausible, is wrong, and would leave every one of the twelve new observation
dims finite and plausible-looking — so it is written out longhand below rather
than delegated to a "rotate" helper whose direction has to be remembered.

⚠ `ball_state` STACKS THREE ROWS AND RAVELS, so the layout is
[rel_pos_body(3), rel_vel_body(3), rot_vel_body(3)] — position, then LINEAR
velocity, then ANGULAR velocity, each already rotated into the torso frame.
The linear part is a difference of two free joints' qvel (ball minus root) in
WORLD axes, not a relative velocity computed in the body frame.

⚠ BOTH DISTANCES IN THE REWARD ARE HORIZONTAL — `norm(...[:2])`. The ball
spends the first part of every episode falling from z=2, and a 3-D distance
would make the reward depend on how far it still has to drop.
"""

from std.math import sqrt, inf, cos, sin, log, pi
from std.random import random_float64
from std.collections import InlineArray

from mojo_rl.physics3d.fields import Data, Dims, DimsLike
from mojo_rl.physics3d.kinematics.xmat import xmat_elem, XMAT_ZZ

from ..rewards import tolerance, SIGMOID_LINEAR
from .quadruped_config import (
    _upright_reward,
    _common_obs_cpu,
    QUADRUPED_FRAME_SKIP,
    QUADRUPED_MAX_STEPS,
)
from .quadruped_xml import (
    qfp,
    QUADRUPED_FETCH_OBS_DIM,
    TORSO_BODY_IDX,
    TOE_BODY_0,
    TOE_BODY_STRIDE,
    FETCH_TORSO_SITE_IDX,
    FETCH_TOE_SITE_0,
    FETCH_TARGET_SITE_IDX,
    FETCH_WORKSPACE_SITE_IDX,
    FETCH_BALL_BODY_IDX,
    FETCH_BALL_QPOS_0,
    FETCH_BALL_DOF_0,
    FETCH_FLOOR_HALF,
    FETCH_WORKSPACE_RADIUS,
    FETCH_TARGET_RADIUS,
    FETCH_BALL_RADIUS,
)
from ...phyics3d_env_config import Phyics3dEnvConfig


# `arena_radius = geom_size['floor', 0] * sqrt(2)` — the corner-to-centre
# distance, so a ball in the far corner sits exactly at the margin.
comptime FETCH_ARENA_RADIUS: Float64 = FETCH_FLOOR_HALF * 1.4142135623730951
# `spawn_radius = 0.9 * geom_size['floor', 0]`, shared by the quadruped's own
# placement and the ball's.
comptime FETCH_SPAWN_RADIUS: Float64 = 0.9 * FETCH_FLOOR_HALF


@always_inline
def _randn_pair() -> InlineArray[Float64, 2]:
    """Two independent standard normals (Box-Muller).

    `Fetch` kicks the ball with `5*self.random.randn(2)`. A uniform draw would
    bound the speed at 5 and never produce the occasional hard shot across the
    arena, so the transform is load-bearing rather than decoration — the same
    reason `Move`'s orientation draw uses normals.
    """
    var u1 = random_float64()
    if u1 < 1e-300:
        u1 = 1e-300
    var u2 = random_float64()
    var r = sqrt(-2.0 * log(u1))
    var out = InlineArray[Float64, 2](fill=0.0)
    out[0] = r * cos(2.0 * pi * u2)
    out[1] = r * sin(2.0 * pi * u2)
    return out^


@always_inline
def _world_to_torso[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    vx: Float64,
    vy: Float64,
    vz: Float64,
) -> InlineArray[Float64, 3]:
    """`v.dot(xmat['torso'].reshape(3,3))` — numpy row-vector convention.

    That contracts v with the matrix's ROWS' first index, i.e. it computes
    `R^T v` (world -> body), NOT `R v`. Columns of R are the torso's axes in
    world coordinates, so component k is the projection of v onto torso axis k:

        out[k] = sum_i v[i] * R[i][k]
    """
    var out = InlineArray[Float64, 3](fill=0.0)
    for k in range(3):
        out[k] = (
            vx * xmat_elem(d, TORSO_BODY_IDX, 0 * 3 + k)
            + vy * xmat_elem(d, TORSO_BODY_IDX, 1 * 3 + k)
            + vz * xmat_elem(d, TORSO_BODY_IDX, 2 * 3 + k)
        )
    return out^


struct DMQuadrupedFetchConfig(Phyics3dEnvConfig):
    """`Fetch` — bring the ball to the target at the origin."""

    comptime FRAME_SKIP: Int = QUADRUPED_FRAME_SKIP
    comptime MAX_STEPS: Int = QUADRUPED_MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # accelerometer + force/torque still need `mj_rnePostConstraint`.
    comptime RNE_POST: Bool = True
    comptime RESET_FIND_HEIGHT: Bool = True

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(qfp.TIMESTEP)

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
        """`Fetch.initialize_episode`.

        ⚠ THIS IS NOT `Move`'s RESET. `Move` draws a uniform quaternion from
        four normals; `Fetch` draws a single AZIMUTH and builds a yaw-only
        quaternion `(cos(a/2), 0, 0, sin(a/2))`, so the quadruped always starts
        upright and only its heading is random. Reusing `Move`'s draw would
        start it on its back half the time and the upright term would swamp
        everything else.

        It also picks a random horizontal position, which `Move` leaves at the
        origin — `_find_non_contacting_height` takes x and y for that reason.
        """
        var azimuth = 2.0 * pi * random_float64()
        var half = 0.5 * azimuth
        var x_pos = (2.0 * random_float64() - 1.0) * FETCH_SPAWN_RADIUS
        var y_pos = (2.0 * random_float64() - 1.0) * FETCH_SPAWN_RADIUS

        # Free-joint qpos is [x, y, z, qw, qx, qy, qz] — w FIRST. The height is
        # left at 0 for `_find_non_contacting_height` to raise, which the env
        # does after this hook because it needs FK and broadphase.
        d.qpos.data[0] = Scalar[DTYPE](x_pos)
        d.qpos.data[1] = Scalar[DTYPE](y_pos)
        d.qpos.data[2] = Scalar[DTYPE](0)
        d.qpos.data[3] = Scalar[DTYPE](cos(half))
        d.qpos.data[4] = Scalar[DTYPE](0)
        d.qpos.data[5] = Scalar[DTYPE](0)
        d.qpos.data[6] = Scalar[DTYPE](sin(half))

        # --- ball: dropped from z=2 with a random horizontal kick ------------
        # ⚠ `5*randn(2)` IS GAUSSIAN, not uniform — a uniform kick would never
        # produce the occasional hard shot across the arena that makes the
        # task interesting, and would cap the ball's speed at 5 instead of
        # leaving it unbounded.
        d.qpos.data[FETCH_BALL_QPOS_0 + 0] = Scalar[DTYPE](
            (2.0 * random_float64() - 1.0) * FETCH_SPAWN_RADIUS
        )
        d.qpos.data[FETCH_BALL_QPOS_0 + 1] = Scalar[DTYPE](
            (2.0 * random_float64() - 1.0) * FETCH_SPAWN_RADIUS
        )
        d.qpos.data[FETCH_BALL_QPOS_0 + 2] = Scalar[DTYPE](2.0)
        # A fresh `Data` is all zeros, so the ball's quaternion would be the
        # DEGENERATE (0,0,0,0) rather than identity unless it is written.
        d.qpos.data[FETCH_BALL_QPOS_0 + 3] = Scalar[DTYPE](1.0)
        d.qpos.data[FETCH_BALL_QPOS_0 + 4] = Scalar[DTYPE](0)
        d.qpos.data[FETCH_BALL_QPOS_0 + 5] = Scalar[DTYPE](0)
        d.qpos.data[FETCH_BALL_QPOS_0 + 6] = Scalar[DTYPE](0)

        var g = _randn_pair()
        d.qvel.data[FETCH_BALL_DOF_0 + 0] = Scalar[DTYPE](5.0 * g[0])
        d.qvel.data[FETCH_BALL_DOF_0 + 1] = Scalar[DTYPE](5.0 * g[1])
        for k in range(2, 6):
            d.qvel.data[FETCH_BALL_DOF_0 + k] = Scalar[DTYPE](0)

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
        """`_common_observations` then `ball_state` then `target_position`."""
        if not _common_obs_cpu[DTYPE, FETCH_TORSO_SITE_IDX, FETCH_TOE_SITE_0](d, m_bodies, m_joints, m_geoms, m_sites, act, obs):
            return False

        # --- ball_state: three rows, each rotated into the torso frame ---
        var tx = Float64(d.xpos.data[TORSO_BODY_IDX * 3 + 0])
        var ty = Float64(d.xpos.data[TORSO_BODY_IDX * 3 + 1])
        var tz = Float64(d.xpos.data[TORSO_BODY_IDX * 3 + 2])
        var bx = Float64(d.xpos.data[FETCH_BALL_BODY_IDX * 3 + 0])
        var by = Float64(d.xpos.data[FETCH_BALL_BODY_IDX * 3 + 1])
        var bz = Float64(d.xpos.data[FETCH_BALL_BODY_IDX * 3 + 2])

        var rel_pos = _world_to_torso(d, bx - tx, by - ty, bz - tz)
        # `qvel['ball_root'][:3] - qvel['root'][:3]`. The root free joint
        # is joint 0, so its dofs are 0..5.
        var rel_vel = _world_to_torso(
            d,
            Float64(d.qvel.data[FETCH_BALL_DOF_0 + 0])
            - Float64(d.qvel.data[0]),
            Float64(d.qvel.data[FETCH_BALL_DOF_0 + 1])
            - Float64(d.qvel.data[1]),
            Float64(d.qvel.data[FETCH_BALL_DOF_0 + 2])
            - Float64(d.qvel.data[2]),
        )
        # `qvel['ball_root'][3:]` — absolute, NOT relative to the root.
        var rot_vel = _world_to_torso(
            d,
            Float64(d.qvel.data[FETCH_BALL_DOF_0 + 3]),
            Float64(d.qvel.data[FETCH_BALL_DOF_0 + 4]),
            Float64(d.qvel.data[FETCH_BALL_DOF_0 + 5]),
        )
        for k in range(3):
            obs.append(Scalar[DTYPE](rel_pos[k]))
        for k in range(3):
            obs.append(Scalar[DTYPE](rel_vel[k]))
        for k in range(3):
            obs.append(Scalar[DTYPE](rot_vel[k]))

        # --- target_position: site_xpos['target'] - xpos['torso'] --------
        var gx = Float64(d.site_xpos.data[FETCH_TARGET_SITE_IDX * 3 + 0])
        var gy = Float64(d.site_xpos.data[FETCH_TARGET_SITE_IDX * 3 + 1])
        var gz = Float64(d.site_xpos.data[FETCH_TARGET_SITE_IDX * 3 + 2])
        var tgt = _world_to_torso(d, gx - tx, gy - ty, gz - tz)
        for k in range(3):
            obs.append(Scalar[DTYPE](tgt[k]))
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
        """`Fetch.get_reward` — never terminates early."""
        var zz = xmat_elem(d, TORSO_BODY_IDX, XMAT_ZZ)
        var upright = _upright_reward(zz)

        var bx = Float64(d.xpos.data[FETCH_BALL_BODY_IDX * 3 + 0])
        var by = Float64(d.xpos.data[FETCH_BALL_BODY_IDX * 3 + 1])
        var wx = Float64(d.site_xpos.data[FETCH_WORKSPACE_SITE_IDX * 3 + 0])
        var wy = Float64(d.site_xpos.data[FETCH_WORKSPACE_SITE_IDX * 3 + 1])
        var gx = Float64(d.site_xpos.data[FETCH_TARGET_SITE_IDX * 3 + 0])
        var gy = Float64(d.site_xpos.data[FETCH_TARGET_SITE_IDX * 3 + 1])

        # Both horizontal — `norm(...[:2])`.
        var self_to_ball = sqrt(
            (wx - bx) * (wx - bx) + (wy - by) * (wy - by)
        )
        var ball_to_target = sqrt(
            (gx - bx) * (gx - bx) + (gy - by) * (gy - by)
        )

        # ⚠ THE UPPER BOUND IS workspace_radius + ball_radius, not either one
        # alone: the ball "reaches" the workspace when their SURFACES touch.
        var reach = tolerance[SIGMOID_LINEAR, 0.0](
            self_to_ball,
            0.0,
            FETCH_WORKSPACE_RADIUS + FETCH_BALL_RADIUS,
            FETCH_ARENA_RADIUS,
        )
        var fetch = tolerance[SIGMOID_LINEAR, 0.0](
            ball_to_target, 0.0, FETCH_TARGET_RADIUS, FETCH_ARENA_RADIUS
        )
        # `reach * (0.5 + 0.5*fetch)`: reaching the ball is worth half the
        # available reward on its own, so the agent has a gradient to follow
        # before it has ever moved the ball toward the target.
        return (
            Scalar[DTYPE](upright * reach * (0.5 + 0.5 * fetch)),
            False,
        )
