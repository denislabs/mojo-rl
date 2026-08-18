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

from mojo_rl.physics3d.fields import Data, Model, Dims, DimsLike
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
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


comptime SETTLE_SOLVER: StaticString = "newton"
"""The solver `PropPlacer`'s settle must run — `Phyics3dEnv.SOLVER`'s value,
NOT `EulerIntegrator`'s default.

⚠⚠ NAMED RATHER THAN DEFAULTED, AND THAT IS THE WHOLE POINT. `EulerIntegrator`
defaults `SOLVER` to "pgs"; `Phyics3dEnv` passes "newton". `settle_free_props`
listed CONE, MAX_CONDIM and NOSLIP_ITER explicitly — with a comment saying a
defaulted parameter is a DIFFERENT PHYSICS from the one the episode steps — and
omitted SOLVER, so every manipulation prop settled under PGS and was then
stepped under Newton for the whole episode.

⚠ THIS IS A CONSTANT, NOT A PARAMETER THREADED FROM THE ENV, because the reset
hooks are `Phyics3dEnvConfig` STATICS and cannot see `Phyics3dEnv.SOLVER`. It
is correct for every env in this tree (all take the "newton" default) and it is
a single place to change if one ever does not. A config that overrides the env
solver must pass its own value rather than this."""


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


def set_free_prop_pose[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1],
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


def prop_has_penetrating_contact[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    prop_body: Int,
    ignore_bodies: List[Int],
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
        if ba != prop_body and bb != prop_body:
            continue
        # `ignore_bodies` — props not yet placed, whose contacts the reference
        # has switched off. See `place_free_prop`.
        var skip = False
        for k in range(len(ignore_bodies)):
            if ba == ignore_bodies[k] or bb == ignore_bodies[k]:
                skip = True
        if not skip:
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


def place_free_prop[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1],
    mut mf: Model[DTYPE, D],
    prop_body: Int,
    qpos_adr: Int,
    dof_adr: Int,
    poses: List[Scalar[DTYPE]],
    ignore_bodies: List[Int],
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

    ⚠⚠ `ignore_bodies` IS THE REFERENCE'S CONTACT-DISABLING PASS.
    `PropPlacer.__call__` zeroes `contype`/`conaffinity` on EVERY prop it is
    about to place, then restores them one prop at a time as it places each.
    So while brick 1 is being drawn, brick 2 is invisible to collision — and it
    has to be, because brick 2 is still sitting wherever the last episode left
    it. With ONE prop the list is empty and this is a no-op, which is why the
    four single-prop tasks never exercised it.

    ⚠ WE IGNORE CONTACTS RATHER THAN DISABLE THEM, and the difference is the
    contact BUFFER. The reference disables to free buffer space as well as to
    skip the test; ignoring costs buffer entries that a 400-`nconmax` model
    might miss. Ours is 128 and the measured worst case is far below it, but a
    6-brick task would want the real thing.

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
        set_free_prop_pose[DTYPE](
            d, qpos_adr, dof_adr, pos, quat
        )
        if ignore_collisions:
            return PropPlaceResult(True, a + 1)
        # The reference's `physics.forward()` — "so that we can detect if the
        # new pose results in collisions".
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)
        if not prop_has_penetrating_contact[DTYPE](d, prop_body, ignore_bodies):
            return PropPlaceResult(True, a + 1)
    return PropPlaceResult(False, n)


def place_fixed_prop[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1],
    mut mf: Model[DTYPE, D],
    frame_body: Int,
    n_bodies: Int,
    ignore_bodies: List[Int],
    poses: List[Scalar[DTYPE]],
    max_attempts: Int = 20,
) raises -> PropPlaceResult:
    """`PropPlacer.__call__` for a prop with NO FREE JOINT — `Place`'s pedestal.

    ⚠⚠ THIS WRITES A MODEL CONSTANT, NOT STATE. `composer.Entity.set_pose`
    branches on `mjcf.get_frame_freejoint`: with one it writes `qpos`, and
    WITHOUT one it writes the attachment frame's `body_pos`. `Place` attaches
    its pedestal with `arena.attach`, not `add_free_entity`, so the pedestal
    moves by editing `mf.bodies`. That is why `place_*` has 20 bodies and still
    only 10 joints — counting bodies and expecting a matching joint is the
    natural mistake here.

    `poses` is `7 * n` injected values — three position then a (x, y, z, w)
    quaternion per attempt, in OUR order, exactly like `place_free_prop`.

    ⚠ THE QUATERNION IS NOT ALWAYS IDENTITY, and assuming it was is a mistake
    this function made first. `Place`'s pedestal placer leaves `quaternion` at
    `rotations.IDENTITY_QUATERNION`, so there it is — but `Stack`'s brick
    placer passes `workspaces.uniform_z_rotation`, and with a FIXED base brick
    that yaw lands on the attachment frame's `body_quat`. Same code path, one
    task varies it and the other does not.

    `frame_body` is the attachment frame and `n_bodies` how many consecutive
    bodies the entity spans (the pedestal is 2: the pillar and its cradle), so
    the rejection test can ask about the whole entity.

    ⚠ `ignore_bodies` REPRODUCES `ignore_contacts_with_entities` and the
    contact-disabling pass. `Place` passes `[self._prop]` — the brick has not
    been placed yet and is sitting wherever the last episode left it, so its
    contacts must not veto the pedestal. Empty list for none.

    ⚠⚠ THE REJECTION LOOP IS PRESENT BUT NOT EXERCISED BY `place_*`, and saying
    so is better than implying coverage. Measured on the reference: over 5
    resets the pedestal placer's predicate was called 5 times and rejected 0,
    and at qpos0 no penetrating contact touches the pedestal at all. It is here
    because the reference has it and because the multi-prop tasks will lean on
    it, not because this task proves it works.

    ⚠ A STATIC BODY CANNOT COLLIDE WITH ANOTHER STATIC BODY. The pedestal's
    capsule reaches well below z = 0 and MuJoCo reports no ground contact,
    because both are welded to the world. Our narrow phase agrees — 8/8
    contacts and 4/4 pedestal-touching at two in-range poses — but a port that
    got the weld filter wrong would see a permanent ground contact here and
    nowhere else in this family.
    """
    var n = len(poses) // 7
    if n > max_attempts:
        n = max_attempts
    var fb = frame_body * MODEL_BODY_SIZE
    for a in range(n):
        mf.bodies.data[fb + BODY_IDX_POS_X] = poses[a * 7 + 0]
        mf.bodies.data[fb + BODY_IDX_POS_Y] = poses[a * 7 + 1]
        mf.bodies.data[fb + BODY_IDX_POS_Z] = poses[a * 7 + 2]
        mf.bodies.data[fb + BODY_IDX_QUAT_X] = poses[a * 7 + 3]
        mf.bodies.data[fb + BODY_IDX_QUAT_Y] = poses[a * 7 + 4]
        mf.bodies.data[fb + BODY_IDX_QUAT_Z] = poses[a * 7 + 5]
        mf.bodies.data[fb + BODY_IDX_QUAT_W] = poses[a * 7 + 6]
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)
        var bad = False
        var ncon = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
        for c in range(ncon):
            var o = c * CONTACT_SIZE
            if Float64(d.contacts.data[o + CONTACT_IDX_DIST]) > 0.0:
                continue
            var ba = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_A]))
            var bb = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_B]))
            var skip = False
            for k in range(len(ignore_bodies)):
                if ba == ignore_bodies[k] or bb == ignore_bodies[k]:
                    skip = True
            if skip:
                continue
            var a_in = ba >= frame_body and ba < frame_body + n_bodies
            var b_in = bb >= frame_body and bb < frame_body + n_bodies
            if a_in or b_in:
                bad = True
                break
        if not bad:
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


def settle_free_props[

    DTYPE: DType,
    # ⚠⚠ FROM THE TASK'S MODEL DEF / THE ENV, NEVER DEFAULTED. `MAX_CONDIM`,
    # `NOSLIP_ITER` and `SOLVER` all have an `EulerIntegrator` default that
    # silently changes the physics (3, 0 and **"pgs"**), so a settle run with
    # the defaults is a DIFFERENT PHYSICS from the one the episode then steps —
    # the prop would settle against contacts the env does not have.
    #
    # ⚠⚠ `SOLVER` IS IN THIS LIST BECAUSE IT IS THE ONE THAT GOT MISSED, and
    # the note above was already written when it did. `EulerIntegrator.SOLVER`
    # defaults to "pgs" while `Phyics3dEnv.SOLVER` is "newton", so every prop
    # in the manipulation family settled under PGS and was then stepped under
    # Newton. That is not merely a parity gap: on `stack_3_random` PGS DIVERGED
    # on a single near-zero-penetration brick-on-brick row (dist -3.75e-08,
    # |n| = 1.0, condim 3) appearing at settle step 2 — all twelve free-brick
    # dofs saturated at ±100 m/s with qacc 1.077e11 in ONE step, ejecting two
    # bricks at ~160 m/s, and 2 of 24 bricks over 12 resets ended outside
    # `prop_bbox`. Under Newton: 0 of 24, every brick at |qvel| ≈ 7.9e-05.
    #
    # It therefore has NO DEFAULT. A caller forced to name it cannot inherit
    # the wrong one by omission, which is exactly how this survived.
    # ⚠ `Int`, not `ConeType`: `ModelDefFromXML.CONE_TYPE` and
    # `EulerIntegrator`'s parameter are both `Int`, and `ConeType` here would
    # only force the caller to convert twice.
    CONE: Int,
    MAX_CONDIM: Int,
    NOSLIP_ITER: Int,
    NHOLD: Int,
    SOLVER: StaticString,

    D: DimsLike,
](
    mut d: Data[DTYPE, D, 1],
    mut mf: Model[DTYPE, D],
    dof_adrs: List[Int],
    timestep: Float64,
    label: String = String(""),
) raises -> SettleResult:
    """`PropPlacer`'s `place_and_settle` inner loop, for N free props.

    ⚠ THE TOLERANCE IS OVER EVERY PROP JOINT AT ONCE. The reference binds
    `self._prop_joints` — ALL of them — and tests `np.max(np.abs(qvel))`, so a
    scene settles when the LAST prop has settled. Looping per prop and
    stopping at the first one to go quiet would return while another is still
    moving.

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
        DTYPE,
        D,
        CONE,
        1,
        SOLVER=SOLVER,
        # `RNE_POST` is off here alone — the settle takes no observation.
        RNE_POST=False,
        MAX_CONDIM=MAX_CONDIM,
        NOSLIP_ITER=NOSLIP_ITER,
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
        for p in range(len(dof_adrs)):
            for k in range(6):
                var v = abs(Float64(d.qvel.data[dof_adrs[p] + k]))
                var a = abs(Float64(d.qacc.data[dof_adrs[p] + k]))
                if v > mv:
                    mv = v
                if a > ma:
                    ma = a
        steps = s + 1
        if mv < SETTLE_QVEL_TOL and ma < SETTLE_QACC_TOL:
            forward_kinematics["cpu"](d, mf)
            return SettleResult(True, steps, mv, ma)
    forward_kinematics["cpu"](d, mf)
    # ⚠ THE WARNING LIVES HERE, NOT AT THE CALL SITES. All seven of them
    # discarded this result with `_ =`, so a scene that never settled was
    # indistinguishable from one that settled on step 1 — and that is exactly
    # the state the PGS/Newton mix-up (see `SOLVER` above) left behind for
    # months. Reporting from inside the function is the only version a new
    # call site cannot forget.
    #
    # ⚠ IT WARNS RATHER THAN RAISES, and that is the REFERENCE's choice, not a
    # softened one: `PropPlacer.__init__` defaults
    # `raise_exception_on_settle_failure=False` and every manipulation task
    # takes the default, so `place_and_settle` logs `_SETTLING_PHYSICS_FAILED`
    # and returns False (`prop_initializer.py:262-280`). Raising here would
    # abort episodes dm_control completes.
    #
    # ⚠⚠ IT FIRES ON A REAL DIVERGENCE, NOT ON NOISE, AND THAT IS WHY IT IS
    # UNCONDITIONAL. Measured over 12 resets each of lift_brick, reach_duplo and
    # stack_2 at float32: OURS failed 3/36, dm_control's own settle failed
    # **0/36** on the same three tasks with the same tolerances (1e-3 / 1e-2 /
    # 2 s, asserted identical). So this is not a tolerance we inherited badly —
    # our settle leaves a residual |qacc| of 0.015..0.046 where the reference
    # gets under 0.01. See the task filed for that; the warning exists so the
    # next person does not have to rediscover it from a frozen-looking scene.
    #
    # ⚠ AND THE REFERENCE'S RETRY IS ALREADY MATCHED. `place_and_settle` loops
    # `max_settle_physics_attempts` times, RE-PLACING the props each round —
    # but that parameter defaults to 1 and all five manipulation `PropPlacer`
    # call sites take the default (reach, place, lift, bricks; verified in
    # `dm_control/manipulation/*.py`), so one settle with no re-place IS the
    # reference behaviour here. Same for `min_settle_physics_time`, which
    # defaults to 0 and would otherwise forbid returning on the first step.
    # ⚠ NAME THE TOLERANCE THAT MISSED, because the two fail for different
    # reasons and the qacc one does NOT mean "the props are still moving".
    # Measured across 36 resets (lift_brick / reach_duplo / stack_2, float32),
    # every failure had |qvel| 1.5e-05 .. 5.5e-05 — twenty to seventy times
    # BELOW its own tolerance — and missed on |qacc| alone, 0.015 .. 0.046
    # against 0.01. The prop is at rest and carrying a residual contact
    # acceleration; a message saying it is moving would send the reader after
    # the wrong quantity.
    var why = String("")
    if mv >= SETTLE_QVEL_TOL and ma >= SETTLE_QACC_TOL:
        why = "|qvel| AND |qacc|"
    elif mv >= SETTLE_QVEL_TOL:
        why = "|qvel|"
    else:
        why = "|qacc|"
    print(
        "settle_free_props: FAILED TO SETTLE"
        + ((" [" + label + "]") if label.byte_length() > 0 else "")
        + " on "
        + why
        + " after "
        + String(steps)
        + " steps ("
        + String(SETTLE_MAX_TIME)
        + " s): |qvel| "
        + String(mv)
        + " (tol "
        + String(SETTLE_QVEL_TOL)
        + "), |qacc| "
        + String(ma)
        + " (tol "
        + String(SETTLE_QACC_TOL)
        + ")"
    )
    return SettleResult(False, steps, mv, ma)


def settle_free_prop[
    DTYPE: DType,
    CONE: Int,
    MAX_CONDIM: Int,
    NOSLIP_ITER: Int,
    NHOLD: Int,
    SOLVER: StaticString,
    D: DimsLike,
](
    mut d: Data[DTYPE, D, 1],
    mut mf: Model[DTYPE, D],
    dof_adr: Int,
    timestep: Float64,
    label: String = String(""),
) raises -> SettleResult:
    """`settle_free_props` for the single-prop case — the seven tasks with one
    free prop read better this way, and it keeps their call sites unchanged."""
    var adrs = List[Int]()
    adrs.append(dof_adr)
    return settle_free_props[
        DTYPE, CONE, MAX_CONDIM, NOSLIP_ITER, NHOLD, SOLVER
    ](d, mf, adrs, timestep, label)
