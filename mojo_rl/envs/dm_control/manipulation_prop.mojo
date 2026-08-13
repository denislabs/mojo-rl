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
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST,
    META_IDX_NUM_CONTACTS,
)


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


def prop_has_penetrating_contact[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAXC: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAXC, NSITE, 1],
    prop_body: Int,
) -> Bool:
    """`PropPlacer._has_collisions_with_prop` — the placer's reject predicate.

    True when ANY contact with `dist <= 0` touches the prop. That is a
    different question from `has_relevant_collisions` (the TCP initializer's),
    and the difference is not a detail: this one has no notion of which entity
    the other geom belongs to, so a prop RESTING ON THE TABLE is a rejection.
    That is why `reach.py` places the brick 1 mm up (`_PROP_Z_OFFSET`) and
    settles it afterwards — the draw is tested in the air, and only then
    allowed to fall.

    ⚠ THE REFERENCE ASKS BY GEOM, THIS ASKS BY BODY. dm_control collects the
    prop's geom ids and tests `contact.geom1 in prop_geom_ids`. Every prop in
    this family is ONE body carrying all its geoms — a Duplo's 41 geoms all
    hang off the attachment frame — so the two predicates select the same
    contacts. A prop with an articulated sub-body would need the geom form.

    ⚠ `dist > 0` CONTACTS ARE SKIPPED. Our narrow phase emits records out to
    `margin`, and the duplo's studs declare `margin=1e-4`, so a brick can carry
    contact records while touching nothing. Rejecting on `ncon` would reject
    poses the reference accepts.

    ⚠ THIS READS THE O(N^2) CONTACT CONVENTION — raw signed distance in
    `CONTACT_IDX_DIST`. The SAP path stores `dist - margin` in the same slot;
    see `has_relevant_collisions`, which carries the same warning.
    """
    var ncon = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    for c in range(ncon):
        var o = c * CONTACT_SIZE
        if Float64(d.contacts.data[o + CONTACT_IDX_DIST]) > 0.0:
            continue
        var ba = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_A]))
        var bb = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_B]))
        if ba == prop_body or bb == prop_body:
            return True
    return False


@fieldwise_init
struct PropPlaceResult(Copyable, Movable):
    """Outcome of `place_free_prop`.

    `attempts` is what says whether the draw was accepted immediately or the
    loop worked for it; the reference RAISES `EpisodeInitializationError` on
    exhaustion, and returning instead keeps that inspectable.
    """

    var success: Bool
    var attempts: Int


def place_free_prop[
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
    MAXC: Int,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAXC, NSITE, 1],
    mut mf: Model[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL, NMESHV, NPAIR
    ],
    prop_body: Int,
    qpos_adr: Int,
    dof_adr: Int,
    poses: List[Scalar[DTYPE]],
    ignore_collisions: Bool = False,
    max_attempts: Int = 20,
) raises -> PropPlaceResult:
    """`PropPlacer.__call__`'s `place_props`, for ONE free-jointed prop.

    `poses` is `7 * n` injected values — three position then a (x, y, z, w)
    quaternion per attempt, in OUR order; `set_free_prop_pose` converts. See
    this module's header for why the draws are injected rather than generated.

    ⚠ `ignore_collisions` IS PER TASK AND THE TWO CASES DIVERGE COMPLETELY.
    `lift.py` passes True, so the first draw is always taken and this is a
    single `set_pose`. `reach.py` leaves it False, so the loop is real — and
    with the arm ALREADY PLACED (see `Reach.initialize_episode`'s order) a
    brick drawn under the gripper is genuinely rejected.

    ⚠ CONTACT DISABLING IS NOT REPRODUCED, and for a single-prop task that is
    exact. The reference zeroes `contype`/`conaffinity` on every prop it is
    about to place before the loop, purely to free contact-buffer space while
    the OTHERS are unplaced; with one prop there are no others. A multi-prop
    task needs it, and needs this function extended rather than reused.

    ⚠ ON EXHAUSTION THE LAST DRAW IS LEFT IN PLACE. The reference raises, so it
    never has to say; a caller here must check `success` rather than assume the
    pose is usable.
    """
    var n = len(poses) // 7
    if n > max_attempts:
        n = max_attempts
    for a in range(n):
        var pos = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        var quat = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        for k in range(3):
            pos[k] = poses[a * 7 + k]
        for k in range(4):
            quat[k] = poses[a * 7 + 3 + k]
        set_free_prop_pose[DTYPE, NQ, NV, NBODY, MAXC, NSITE](
            d, qpos_adr, dof_adr, pos, quat
        )
        if ignore_collisions:
            return PropPlaceResult(True, a + 1)
        # The reference's `physics.forward()` — "so that we can detect if the
        # new pose results in collisions".
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)
        if not prop_has_penetrating_contact[
            DTYPE, NQ, NV, NBODY, MAXC, NSITE
        ](d, prop_body):
            return PropPlaceResult(True, a + 1)
    return PropPlaceResult(False, n)


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


def settle_free_prop[
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
    MAXC: Int,
    # ⚠ FROM THE TASK'S MODEL DEF, NOT DEFAULTED. `MAX_CONDIM` and
    # `NOSLIP_ITER` both have a default that silently disables the feature (3
    # and 0), so a settle run with the defaults is a DIFFERENT PHYSICS from the
    # one the episode then steps — the prop would settle against contacts the
    # env does not have.
    # ⚠ `Int`, not `ConeType`: `ModelDefFromXML.CONE_TYPE` and
    # `EulerIntegrator`'s parameter are both `Int`, and `ConeType` here would
    # only force the caller to convert twice.
    CONE: Int,
    MAX_CONDIM: Int,
    NOSLIP_ITER: Int,
    NHOLD: Int,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAXC, NSITE, 1],
    mut mf: Model[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL, NMESHV, NPAIR
    ],
    dof_adr: Int,
    timestep: Float64,
) raises -> SettleResult:
    """`PropPlacer`'s `place_and_settle` inner loop, for ONE free prop.

    Steps until the prop's six dofs are below `SETTLE_QVEL_TOL` /
    `SETTLE_QACC_TOL` or `SETTLE_MAX_TIME` of simulated time runs out, holding
    the first `NHOLD` dofs static.

    ⚠⚠ `NHOLD` IS `JointStaticIsolator`, AND IT IS NOT OPTIONAL. The reference
    caches every NON-isolated joint's `qpos`/`qvel` ONCE, at construction, and
    restores them when its context exits — which is once PER STEP. So settling
    moves the prop and nothing else. Without it the arm falls under gravity for
    up to 2 s of simulated time before the episode starts, which is not a small
    difference: it is a different initial pose every episode.

    ⚠ `NHOLD` COUNTS LEADING DOFS, and that is only the isolator's complement
    because the robot is attached FIRST in all 13 models — 9 hinges at qpos
    0..8 and qvel 0..8, so one bound covers both arrays. A model with a prop
    ahead of the robot would need an index list instead.

    ⚠ THE SETTLE IS NOT ALWAYS A CORRECTION. `lift_large_box` places its box at
    exactly its own half-height, i.e. already at rest, and dm_control's own
    settle moves it by 5.2e-06. A Duplo dropped from `_PROP_Z_OFFSET` = 1 mm
    does real work. One loop, both cases.
    """
    var hold_qpos = InlineArray[Float64, NHOLD](fill=0.0)
    var hold_qvel = InlineArray[Float64, NHOLD](fill=0.0)
    for i in range(NHOLD):
        hold_qpos[i] = Float64(d.qpos.data[i])
        hold_qvel[i] = Float64(d.qvel.data[i])

    var integ = EulerIntegrator[
        DTYPE, NQ, NV, NBODY, NJOINT, MAXC, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV,
        CONE,
        1,
        # `RNE_POST` is off here alone — the settle takes no observation.
        RNE_POST=False,
        MAX_CONDIM=MAX_CONDIM,
        NOSLIP_ITER=NOSLIP_ITER,
        NPAIR=NPAIR,
    ]()
    var max_steps = Int(SETTLE_MAX_TIME / timestep)
    var mv = 0.0
    var ma = 0.0
    var steps = 0
    for s in range(max_steps):
        integ.step["cpu"](d, mf)
        for i in range(NHOLD):
            d.qpos.data[i] = Scalar[DTYPE](hold_qpos[i])
            d.qvel.data[i] = Scalar[DTYPE](hold_qvel[i])
        mv = 0.0
        ma = 0.0
        for k in range(6):
            var v = abs(Float64(d.qvel.data[dof_adr + k]))
            var a = abs(Float64(d.qacc.data[dof_adr + k]))
            if v > mv:
                mv = v
            if a > ma:
                ma = a
        steps = s + 1
        if mv < SETTLE_QVEL_TOL and ma < SETTLE_QACC_TOL:
            forward_kinematics["cpu"](d, mf)
            return SettleResult(True, steps, mv, ma)
    forward_kinematics["cpu"](d, mf)
    return SettleResult(False, steps, mv, ma)
