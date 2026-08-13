"""`composer.initializers.PropPlacer` — placing a free prop and settling it.

The third reset primitive of the manipulation family, after the site IK
(`physics3d/dynamics/ik_site.mojo`) and the collision-rejection loop
(`manipulation_reset.mojo`). Eleven of the 13 `_features` tasks use it; only
`reach_site_features` does not.

WHAT THE REFERENCE DOES, statement by statement
(`prop_initializer.py::PropPlacer.__call__`):

  1. disable contacts on every prop it is about to place, `forward()`;
  2. for each prop: draw a position and quaternion, `set_pose`, `forward()`,
     and if `ignore_collisions` is False reject any pose where that prop has a
     PENETRATING contact — up to `max_attempts_per_prop = 20`;
  3. if `settle_physics`, step the simulation with every NON-prop joint held
     static until the prop's `|qvel| < 1e-3` AND `|qacc| < 1e-2`, or 2 s of
     simulated time elapse; then restore `data.time`.

⚠⚠ THE ISOLATOR IS THE PART THAT LOOKS OPTIONAL AND IS NOT.
`JointStaticIsolator` caches every non-prop joint's `qpos`/`qvel` when it is
CONSTRUCTED — once, before the loop — and restores them when its context
exits, which is once PER STEP. So settling moves the prop and nothing else: the
arm is rewound after every step. Settling without it lets the arm fall under
gravity for up to 2 s of simulated time before the episode starts, which is
not a small difference — it is a different initial pose every episode.

⚠ `settle_physics` IS NOT ALWAYS A CORRECTION. For `lift_large_box` the prop
bbox places the box at z = its own half-height, i.e. already at rest, and
dm_control's own settle moves it by 5.2e-06. For a Duplo dropped from z = 0 it
does real work. The loop is written once and both cases go through it.

⚠ CONTACTS ARE NOT DISABLED HERE, and for the single-prop tasks that is
exact: step 1 exists to free contact-buffer space while OTHER props are
unplaced, and with one prop there are no others. A multi-prop task
(`stack_*`, `reassemble_*`) needs it, and will need this function extended
rather than reused as-is.

⚠ RANDOM DRAWS ARE INJECTED, like everywhere else in this family — see
`manipulation_reset`'s header. dm_control draws from a numpy `RandomState`
whose bit stream cannot be reproduced in Mojo, so a gate drives both sides
from the same numbers instead of comparing distributions.
"""

from std.collections import InlineArray
from std.math import abs, sqrt, sin, cos, pi

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics


# `prop_initializer._SETTLE_QVEL_TOL` / `_SETTLE_QACC_TOL`.
comptime SETTLE_QVEL_TOL: Float64 = 1e-3
comptime SETTLE_QACC_TOL: Float64 = 1e-2
# `PropPlacer.__init__`'s `max_settle_physics_time`, in SECONDS of simulated
# time — not steps. At the manipulation timestep of 0.002 that is 1000 steps.
comptime SETTLE_MAX_TIME: Float64 = 2.0


@always_inline
def uniform_z_rotation[
    DTYPE: DType
](draw: Float64) -> InlineArray[Scalar[DTYPE], 4]:
    """`workspaces.uniform_z_rotation` — a yaw drawn from U(-pi, pi).

    Returns (x, y, z, w), OUR quaternion order. The reference builds
    `[cos(a/2), 0, 0, sin(a/2)]` in MuJoCo's (w, x, y, z).
    """
    var angle = -pi + 2.0 * pi * draw
    var out = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    out[2] = Scalar[DTYPE](sin(0.5 * angle))
    out[3] = Scalar[DTYPE](cos(0.5 * angle))
    return out^


def set_free_prop_pose[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAXC: Int,
    NSITE: Int,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAXC, NSITE, 1],
    qpos_adr: Int,
    dof_adr: Int,
    pos: InlineArray[Scalar[DTYPE], 3],
    quat: InlineArray[Scalar[DTYPE], 4],
):
    """`prop.set_pose` for a prop on a free joint — 7 qpos, and zero its qvel.

    ⚠ THE VELOCITY ZEROING IS PART OF THE PLACEMENT, not tidiness. `set_pose`
    teleports the body; leaving the previous episode's `qvel` on the free
    joint launches the prop at the first step, and settling would then be
    settling a throw.

    ⚠ QPOS ORDER IS MuJoCo's: 3 position then a (w, x, y, z) quaternion. Our
    quaternions are (x, y, z, w) everywhere else, so this converts.
    """
    d.qpos.data[qpos_adr + 0] = pos[0]
    d.qpos.data[qpos_adr + 1] = pos[1]
    d.qpos.data[qpos_adr + 2] = pos[2]
    d.qpos.data[qpos_adr + 3] = quat[3]  # w
    d.qpos.data[qpos_adr + 4] = quat[0]  # x
    d.qpos.data[qpos_adr + 5] = quat[1]  # y
    d.qpos.data[qpos_adr + 6] = quat[2]  # z
    for k in range(6):
        d.qvel.data[dof_adr + k] = Scalar[DTYPE](0)


@fieldwise_init
struct SettleResult(Copyable, Movable):
    """Outcome of `settle_free_prop`.

    The reference LOGS a warning and carries on when settling fails
    (`raise_exception_on_settle_failure` defaults to False), so this reports
    rather than raises. `steps` is what says whether the tolerance was met
    early or the 2-second budget ran out.
    """

    var settled: Bool
    var steps: Int
    var max_qvel: Float64
    var max_qacc: Float64
