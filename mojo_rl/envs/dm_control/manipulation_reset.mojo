"""The dm_control manipulation reset path — the task-level half.

`manipulation/reach.py::Reach.initialize_episode` is three statements:

    self._hand.set_grasp(physics, close_factors=random_state.uniform())
    self._tcp_initializer(physics, random_state)
    physics.bind(self._target).pos = self._target_placer(random_state)

The middle one is inverse kinematics and lives in
`physics3d/dynamics/ik_site.mojo`, because it is generic engine maths. The
other two are TASK POLICY — which joints count as fingers, which bounding box
a target is sampled from — so they live here rather than pushing dm_control's
task structure down into the physics engine.

⚠ WITH `use_site=True` (the `reach_site_features` task) THE PROP BRANCH IS
DEAD. `initialize_episode` picks `_prop_placer` or `_target_placer` on
`self._prop`, and `_reach(..., use_site=True)` passes `prop=None`. So the
first manipulation task needs NO `PropPlacer` and no settling — a fact worth
stating, because scoping "the reset path" as one lump overstates it
considerably.

⚠ RANDOM DRAWS ARE INJECTED, NOT GENERATED. Every sampler here takes its
values as arguments. dm_control draws them from a numpy `RandomState`, whose
bit stream cannot be reproduced in Mojo; passing them in lets a gate drive
both implementations from the SAME numbers instead of comparing
distributions. Same reasoning as `set_site_to_xpos`'s `retry_poses`.

`tool_center_point_initializer` below is the second statement's OUTER loop —
`composer/initializers/tcp_initializer.py::ToolCenterPointInitializer.__call__`
— which wraps the IK in rejection sampling against a collision predicate. The
predicate is task policy of the purest kind (it asks which ENTITY owns a
geom), so it lives here too.
"""

from std.collections import InlineArray

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.dynamics.ik_site import set_site_to_xpos
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST,
    META_IDX_NUM_CONTACTS,
)


# Body classes for `has_relevant_collisions`. dm_control asks which MJCF MODEL
# ROOT owns a geom (`geom.root is arm_model`), plus whether its top-level body
# carries a freejoint. Both are composer/entity facts.
#
# ⚠⚠ THIS ARRAY IS AN INPUT, AND CANNOT BE DERIVED FROM OUR MODEL. A baked
# MJCF is FLAT: the entity boundary survives only in the name prefixes
# (`jaco_arm/`, `jaco_arm/jaco_hand/`), and `parser/flat_model.mojo` does not
# keep body names at all — `BodyData` has no `name` field. So the caller must
# supply the classification, exactly as it supplies the random draws above.
# `tests/dm_control/manipulation_ref.py::body_classes_reference` derives it
# with dm_control's OWN predicate, so a gate drives both sides from one
# labelling and tests the RULE rather than a re-derivation of it.
comptime BODY_ARM: Int = 0
comptime BODY_HAND: Int = 1
comptime BODY_FREE: Int = 2  # external, under a top-level body WITH a freejoint
comptime BODY_FIXED: Int = 3  # external without one — INCLUDING the world


def has_relevant_collisions[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAXC: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAXC, NSITE, 1],
    body_class: InlineArray[Int, NBODY],
) -> Bool:
    """`tcp_initializer.py::ToolCenterPointInitializer._has_relevant_collisions`.

    True when any PENETRATING contact is one of:

      * arm-arm or arm-hand (either order), or
      * robot (arm or hand) versus an external body WITHOUT a freejoint.

    ⚠ HAND-HAND IS DELIBERATELY NOT RELEVANT. The reference enumerates
    `arm/arm`, `arm/hand`, `hand/arm` and stops; fingers touching each other
    is what `set_grasp` just asked for, so rejecting it would reject every
    closed grasp. Writing this as "any two robot geoms" is the obvious
    simplification and it is WRONG.

    ⚠ ROBOT-VERSUS-FREE-BODY IS ALSO NOT RELEVANT. A prop with a freejoint can
    be pushed out of the way, so resting against one is not a bad initial
    pose. Only the immovable ones count.

    ⚠ THE GROUND IS RELEVANT, AND IT IS THE DOMINANT CASE. The arena plane
    sits on the world body, which has no freejoint, so it classifies as
    `BODY_FIXED` — arm-versus-ground rejects a pose exactly as arm-versus-arm
    does. Measured on `reach_site_features`, plane-mesh is the most common
    contact type by a wide margin (1868 of 2571 over 400 poses), so a
    predicate tested only on self-collision would be tested on the rare case.

    ⚠ `dist > 0` CONTACTS ARE SKIPPED, and they are not hypothetical: our
    narrow phase emits contacts out to `margin`, so a pose can carry contact
    records while touching nothing. Reading `ncon` instead of the distances
    would reject poses the reference accepts.

    ⚠ THIS READS THE O(N^2) CONTACT CONVENTION. `detect_contacts` stores the
    raw signed distance in `CONTACT_IDX_DIST`; the SAP path stores
    `dist - margin` in the same slot (see `broadphase_sap.mojo`'s docstring),
    so feeding this SAP records would apply the threshold to a different
    quantity and silently change which poses are rejected.
    """
    var ncon = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    for c in range(ncon):
        var o = c * CONTACT_SIZE
        if Float64(d.contacts.data[o + CONTACT_IDX_DIST]) > 0.0:
            continue
        var ba = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_A]))
        var bb = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_B]))
        # ⚠ The plane branches write BODY_B = 0 (world) on the O(N^2) path and
        # -1 on SAP. 0 is a real body index and -1 is not, so normalise here
        # rather than trusting the caller to have used the right path.
        if ba < 0:
            ba = 0
        if bb < 0:
            bb = 0
        if ba >= NBODY or bb >= NBODY:
            continue
        var ca = body_class[ba]
        var cb = body_class[bb]
        var a_robot = ca == BODY_ARM or ca == BODY_HAND
        var b_robot = cb == BODY_ARM or cb == BODY_HAND
        if ca == BODY_ARM and cb == BODY_ARM:
            return True
        if ca == BODY_ARM and cb == BODY_HAND:
            return True
        if ca == BODY_HAND and cb == BODY_ARM:
            return True
        if a_robot and cb == BODY_FIXED:
            return True
        if ca == BODY_FIXED and b_robot:
            return True
    return False


def set_grasp[
    DTYPE: DType, NHAND: Int
](
    mut qpos: List[Scalar[DTYPE]],
    qpos_adr: InlineArray[Int, NHAND],
    range_min: InlineArray[Float64, NHAND],
    range_max: InlineArray[Float64, NHAND],
    close_factors: InlineArray[Float64, NHAND],
):
    """`kinova/jaco_hand.py::JacoHand.set_grasp`.

    Each finger goes to `min + (max - min) * factor`, so 0 is fully open and
    1 fully closed.

    ⚠ THE REFERENCE TAKES A SCALAR OR A SEQUENCE. `reach.py` passes a scalar
    (`random_state.uniform()`), which it broadcasts to every finger:
    `close_factors = (close_factors,) * len(self.joints)`. This always takes
    the per-finger form; a caller reproducing `reach` fills all NHAND entries
    with the same draw. Keeping the broadcast at the call site means a task
    that DOES vary its fingers needs no second entry point.

    ⚠ NOT DONE HERE, and deliberately: the reference also calls
    `physics.after_reset()` and zeroes the hand actuators' `ctrl`. Those touch
    actuator and derived state that this function is not given, so they belong
    to the env's reset. The `ctrl = 0` in particular is not cosmetic — the
    hand runs `<velocity>` actuators, so a stale non-zero ctrl commands a
    finger velocity from the first step.
    """
    for i in range(NHAND):
        var f = close_factors[i]
        qpos[qpos_adr[i]] = Scalar[DTYPE](
            range_min[i] + (range_max[i] - range_min[i]) * f
        )


def sample_bbox_uniform[
    DTYPE: DType
](
    lower: InlineArray[Float64, 3],
    upper: InlineArray[Float64, 3],
    draws: InlineArray[Float64, 3],
) -> InlineArray[Scalar[DTYPE], 3]:
    """`distributions.Uniform(*bbox)` — one point in an axis-aligned box.

    `draws` are three independent uniforms on [0, 1); the reference obtains
    the same thing as `random_state.uniform(lower, upper)`, which is
    `lower + (upper - lower) * u` componentwise.

    Used twice by `reach` with DIFFERENT boxes, and they are easy to confuse:
    `tcp_bbox` positions the gripper (through the IK initializer) and
    `target_bbox` positions the goal site. For `reach_site_features` both are
    `(-0.2, -0.2, 0.02) .. (0.2, 0.2, 0.4)`, so a test on that task alone
    CANNOT tell them apart — `reach_duplo` is where they differ.
    """
    var out = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    for k in range(3):
        out[k] = Scalar[DTYPE](
            lower[k] + (upper[k] - lower[k]) * draws[k]
        )
    return out^


@fieldwise_init
struct TCPInitResult(Copyable, Movable):
    """Outcome of `tool_center_point_initializer`.

    The reference RAISES `EpisodeInitializationError` when it runs out of
    samples. Returning instead keeps the failure inspectable: `samples` says
    how hard it had to work, and the two counters separate the two ways a
    sample dies — which matters because they have opposite fixes. A run that
    is all `ik_failures` wants a wider `max_ik_attempts` or a reachable
    workspace; one that is all `collision_rejections` wants a different
    bounding box. A bare `success` flag cannot tell them apart.
    """

    var success: Bool
    var samples: Int
    var ik_failures: Int
    var collision_rejections: Int


def tool_center_point_initializer[
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
    # See `ik_site.set_site_to_xpos` — a literal `0` here restricts CALLERS,
    # not models.
    NPAIR: Int,
    MAXC: Int,
    NDOF: Int,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAXC, NSITE, 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    site: Int,
    target_positions: List[Scalar[DTYPE]],
    target_quat: InlineArray[Scalar[DTYPE], 4],
    dof_idx: InlineArray[Int, NDOF],
    qpos_adr: InlineArray[Int, NDOF],
    lower: InlineArray[Float64, NDOF],
    upper: InlineArray[Float64, NDOF],
    retry_poses: List[Scalar[DTYPE]],
    body_class: InlineArray[Int, NBODY],
    ignore_collisions: Bool = False,
    max_ik_attempts: Int = 10,
    max_rejection_samples: Int = 10,
) raises -> TCPInitResult:
    """`ToolCenterPointInitializer.__call__` — IK under rejection sampling.

    Draw a TCP target, solve IK for it, recompute contacts, and accept the
    pose only if `has_relevant_collisions` says no. On either failure restore
    the arm and draw again, up to `max_rejection_samples` times.

    ⚠ TARGET POSES ARE INJECTED. `target_positions` is `3 * n` values, one xyz
    per sample, and `retry_poses` is the flat IK-retry sequence consumed at
    `sample * (max_ik_attempts - 1) * NDOF`. dm_control draws both off one
    `RandomState` in that order — see this module's header.

    ⚠ ONLY THE ARM JOINTS ARE SAVED AND RESTORED. The reference restores
    `physics.bind(self._arm.joints).qpos`, which is `qpos_adr` and nothing
    else. That is not an economy: `set_grasp` has ALREADY run by this point in
    `initialize_episode`, so restoring the whole of `qpos` would silently
    undo the grasp this initializer is supposed to preserve.

    ⚠ ON ACCEPTANCE NOTHING IS RESTORED, AND THAT IS THE POINT. `d` is left
    holding the accepted `qpos` with FK and contacts already current for it.
    On exhaustion the arm is back at its entry pose, so a caller that ignores
    the return value gets the pose it started with rather than the last
    rejected one.

    ⚠ FK + NARROW PHASE ARE RE-RUN BEFORE THE PREDICATE. The reference's
    `physics.forward()` carries the comment "Recalculate contacts", and it is
    load-bearing: `set_site_to_xpos` finishes by CANONICALISING the joints
    (wrapping unlimited hinges into range), which moves `qpos` after the last
    solve. Testing contacts without re-running FK would test the pre-wrap
    pose.
    """
    var initial = InlineArray[Scalar[DTYPE], NDOF](fill=Scalar[DTYPE](0))
    for a in range(NDOF):
        initial[a] = d.qpos.data[qpos_adr[a]]

    var ik_failures = 0
    var collision_rejections = 0
    var per_sample = (max_ik_attempts - 1) * NDOF
    if per_sample < 0:
        per_sample = 0

    for s in range(max_rejection_samples):
        if (s + 1) * 3 > len(target_positions):
            # Out of injected targets. Stop rather than reuse the last one,
            # which would burn the remaining budget on a pose already known
            # to fail and report it as a genuine exhaustion.
            break
        var target_pos = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        for k in range(3):
            target_pos[k] = target_positions[s * 3 + k]

        var res = set_site_to_xpos[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAXC, NDOF,
        ](
            d, mf, site, target_pos, target_quat, dof_idx, qpos_adr,
            lower, upper, retry_poses, max_ik_attempts, s * per_sample,
        )

        if res.success:
            forward_kinematics["cpu"](d, mf)
            detect_contacts["cpu"](d, mf)
            var bad = False
            if not ignore_collisions:
                bad = has_relevant_collisions[
                    DTYPE, NQ, NV, NBODY, MAXC, NSITE
                ](d, body_class)
            if not bad:
                return TCPInitResult(
                    True, s + 1, ik_failures, collision_rejections
                )
            collision_rejections += 1
        else:
            ik_failures += 1

        for a in range(NDOF):
            d.qpos.data[qpos_adr[a]] = initial[a]

    return TCPInitResult(
        False, max_rejection_samples, ik_failures, collision_rejections
    )
