"""`bricks.py::Stack` with `randomize_order=True` — the parts both such tasks
share.

`stack_3_bricks_random_order_features` and
`stack_2_of_3_bricks_random_order_features` are the SAME model — the baked XML
is byte-identical, 267 geoms, three bricks with brick 2 welded down — and
differ only in `target_height`: 3 against 2. That changes the length of
`desired_order` (and so the observation, 84 against 83) and the number of
stacked pairs the reward averages, and nothing else. So the logic lives here
once, parameterised by `n_order`, and each config is model wiring.

⚠ `sigma` DEPENDS ONLY ON `order[0]`, which is why one implementation serves
both. `stack_2_of_3` draws a 2-SUBSET rather than a permutation, so one
reference brick is in no pair at all — but it still has to be placed and still
has to appear in the observation, and the mapping for it falls out of the same
"remaining indices in increasing order" rule.

    observation = desired_order(n) + robot(42) + 3 x brick(13)
    reward      = mean over the n - 1 stacked pairs of a stud-to-hole shaping
    episode     = 250 control steps (10 s / .04 s), no early termination
    action      = 9 <velocity> actuators (6 arm, 3 finger)

⚠⚠⚠ THE REFERENCE CHANGES ITS MODEL EVERY EPISODE AND WE DO NOT. THIS FILE IS
THE WORKAROUND, SO READ THIS BEFORE CHANGING ANYTHING IN IT.

`initialize_episode_mjcf` draws `desired_order` and then REMOVES the freejoint
from the brick at `desired_order[0]`, adding one to every other brick. So which
BODY is fixed permutes per episode — measured over 30 resets, all three occur
(14 / 7 / 9). A comptime `ModelDefFromXML` bakes one XML and cannot express
that.

THE WORKAROUND IS A RELABELING, AND IT IS EXACT RATHER THAN AN APPROXIMATION.
The three bricks are DYNAMICALLY IDENTICAL: every geom's type, condim,
contype/conaffinity, size, pos, quat, friction, solref, solimp and margin, the
body's mass, inertia, ipos and iquat, and every site's type, pos and size are
bit-identical across all three. The ONLY difference is `rgba`, which is not in
the `_features` observation. So the physical scene "brick r fixed, others free"
is the SAME scene as "our brick 2 fixed, others free" up to the names on the
bricks.

    sigma: reference brick index -> OUR physical brick index
    sigma(order[0]) = FIXED_BRICK          (always, by construction)
    the remaining reference indices, in increasing order, map to the remaining
    physical bricks, in increasing order

and then

    the observation emits block(sigma(0)), block(sigma(1)), block(sigma(2))
    the reward pairs are (sigma(order[i]), sigma(order[i+1]))
    `desired_order` is emitted as drawn

`sigma` is recomputed from `desired_order` on every read, so only the order has
to persist — it lives in `Data.meta`'s `META_IDX_TASK_PARAM_0..2`.

⚠⚠ THE OBVIOUS ALTERNATIVE IS WRONG, AND IT WAS MEASURED. Keeping every brick
free and "freezing" the base — holding its qpos each step, giving it enormous
mass, or welding it with an equality — does NOT reproduce a removed joint. A
brick whose freejoint has been removed is welded to the WORLD, so
`body_weldid` is 0 and MuJoCo's weld filter drops every pair between it and the
ground: measured, a FIXED brick lying flat on the table generates ZERO contacts
while a FREE brick in the identical pose generates FOUR. Any freeze leaves
`body_weldid` non-zero and injects four phantom contact rows into every solve,
for the whole episode.

⚠ AND IT DOES NOT EXTEND TO THE `_vision` VARIANTS. Colour is observed there,
and colour is the one property relabeling permutes.

RESET, in `Stack.initialize_episode` order:

    self._brick_placer(...)       <- all three, in REFERENCE index order
    self._hand.set_grasp(...)     <- ours runs first, in the state hook
    self._tcp_initializer(...)
    _build_stack(hint bricks)     <- renderer-only, not ported

⚠ THE PLACER WALKS `self._bricks`, i.e. REFERENCE index order, not physical
order. With the contact-disabling pass that matters: at step r every brick not
yet placed is invisible to collision, and "not yet placed" is defined by the
reference's order, not ours.
"""

from std.collections import InlineArray
from std.math import abs, sqrt
from std.random import random_float64

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_RANGE_UNLIMITED,
    META_IDX_TASK_PARAM_0,
)
from mojo_rl.envs.dm_control.manipulation_obs import (
    append_robot_block,
    append_free_prop_block_site,
    N_ARM,
    N_HAND,
)
from mojo_rl.envs.dm_control.manipulation_prop import (
    place_free_prop,
    place_fixed_prop,
    settle_free_props,
    uniform_z_rotation,
)
from mojo_rl.envs.dm_control.manipulation_reset import (
    set_grasp,
    sample_bbox_uniform,
    tool_center_point_initializer,
    BODY_ARM,
    BODY_HAND,
    BODY_FIXED,
    BODY_FREE,
)
from mojo_rl.envs.dm_control.manipulation_stack2_config import (
    min_stud_to_hole_distance,
    pairwise_stacking_reward,
    CLOSE_THRESHOLD,
    CLICK_THRESHOLD,
    CLOSE_COEF,
)

comptime N_BRICKS: Int = 3

# ── model indices, read off MuJoCo's own tables and asserted in the gate ───
#
# ⚠ TWO worldbody sites (`tcp_spawn_area`, `prop_spawn_area`) — `Stack` puts no
# target site on the arena, so the robot block starts at 2.
comptime ROBOT_SITE_BASE: Int = 2
comptime SITE_PINCH: Int = ROBOT_SITE_BASE + 8  # `jaco_hand/pinchsite` = 10

# ⚠ 17 / 19 / 21, NOT 17 / 18 / 19 — the hint twins interleave.
comptime BRICK_BODY_0: Int = 17  # `duplo2x4/`      FREE in this bake
comptime BRICK_BODY_1: Int = 19  # `duplo2x4_2/`    FREE in this bake
comptime BRICK_BODY_2: Int = 21  # `duplo2x4_4/`    FIXED in this bake
comptime FIXED_BRICK: Int = 2
"""Which PHYSICAL brick this bake left without a freejoint.

⚠ NOT a task constant — it is whatever `initialize_episode_mjcf` happened to
draw when the XML was baked, and the gate asserts it against a freshly
constructed, once-reset reference env (which is exactly what the generator
saw). Everything else in this file is written in terms of it."""

comptime BRICK_FRAME_SITE_0: Int = 11
comptime BRICK_FRAME_SITE_1: Int = 45
comptime BRICK_FRAME_SITE_2: Int = 79
comptime BRICK_STUD_0: Int = 12
comptime BRICK_STUD_1: Int = 46
comptime BRICK_STUD_2: Int = 80
comptime BRICK_HOLE_0: Int = 20
comptime BRICK_HOLE_1: Int = 54
comptime BRICK_HOLE_2: Int = 88

# ⚠ The two FREE bricks' addresses. 7 qpos / 6 dof each, so qpos 9 and 16 pair
# with dof 9 and 15 — the offsets diverge and mixing them writes into the
# neighbouring brick's orientation.
comptime BRICK_QPOS_ADR_0: Int = 9
comptime BRICK_DOF_ADR_0: Int = 9
comptime BRICK_QPOS_ADR_1: Int = 16
comptime BRICK_DOF_ADR_1: Int = 15

# `bricks.py::_WORKSPACE`, as `stack_2_bricks`.
comptime PROP_Z_OFFSET: Float64 = 1.0e-6
comptime PROP_BBOX_LOWER_X: Float64 = -0.1
comptime PROP_BBOX_LOWER_Y: Float64 = -0.1
comptime PROP_BBOX_LOWER_Z: Float64 = PROP_Z_OFFSET
comptime PROP_BBOX_UPPER_X: Float64 = 0.1
comptime PROP_BBOX_UPPER_Y: Float64 = 0.1
comptime PROP_BBOX_UPPER_Z: Float64 = PROP_Z_OFFSET
comptime TCP_BBOX_LOWER_X: Float64 = -0.1
comptime TCP_BBOX_LOWER_Y: Float64 = -0.1
comptime TCP_BBOX_LOWER_Z: Float64 = 0.15
comptime TCP_BBOX_UPPER_X: Float64 = 0.1
comptime TCP_BBOX_UPPER_Y: Float64 = 0.1
comptime TCP_BBOX_UPPER_Z: Float64 = 0.4

comptime TWO_PI: Float64 = 6.283185307179586
comptime DOWN_QUAT_XY: Float64 = 0.70710678118
comptime MAX_PROP_ATTEMPTS: Int = 20


# ── the relabeling ─────────────────────────────────────────────────────────


@always_inline
def brick_body_of(phys: Int) -> Int:
    """Body id of PHYSICAL brick `phys`. The hint twins interleave, so the
    stride is 2 and not 1."""
    return BRICK_BODY_0 + 2 * phys


@always_inline
def brick_frame_site_of(phys: Int) -> Int:
    """`bounding_box` site of PHYSICAL brick `phys`.

    ⚠ NOT an arithmetic stride from the body id. A Duplo contributes 34 sites
    (bounding_box + 8 studs + 8 holes, plus its hint twin's 17), so the frame
    sites are 11, 45, 79.
    """
    return BRICK_FRAME_SITE_0 + 34 * phys


@always_inline
def brick_stud_0_of(phys: Int) -> Int:
    return BRICK_STUD_0 + 34 * phys


@always_inline
def brick_hole_0_of(phys: Int) -> Int:
    return BRICK_HOLE_0 + 34 * phys


@always_inline
def brick_qpos_adr_of(free_slot: Int) -> Int:
    """qpos address of the `free_slot`-th FREE brick, counting from 0.

    ⚠ INDEXED BY FREE SLOT, NOT BY PHYSICAL BRICK. Physical brick
    `FIXED_BRICK` has no address at all, so the free bricks occupy slots 0 and
    1 in physical order skipping it.
    """
    return BRICK_QPOS_ADR_0 + 7 * free_slot


@always_inline
def brick_dof_adr_of(free_slot: Int) -> Int:
    return BRICK_DOF_ADR_0 + 6 * free_slot


@always_inline
def free_slot_of(phys: Int) -> Int:
    """Which free slot physical brick `phys` occupies, or -1 if it is fixed."""
    if phys == FIXED_BRICK:
        return -1
    return phys if phys < FIXED_BRICK else phys - 1


def sigma_of(order: InlineArray[Int, N_BRICKS]) -> InlineArray[Int, N_BRICKS]:
    """`sigma`: reference brick index -> our physical brick index.

    `sigma(order[0]) = FIXED_BRICK`, because the reference fixes `order[0]` and
    this model's fixed brick is the only one that can play that part. The
    remaining reference indices go to the remaining physical bricks, both in
    increasing order — any bijection would do, since the free bricks are
    interchangeable, but it has to be the SAME bijection everywhere, so it is
    written once here.

    ⚠ THIS IS A PERMUTATION OF BODIES, NOT OF qpos SLICES. The qpos layout
    depends on which brick the bake fixed; `free_slot_of` is what turns a
    physical brick into an address.
    """
    var sigma = InlineArray[Int, N_BRICKS](fill=-1)
    sigma[order[0]] = FIXED_BRICK
    var nxt = 0
    for r in range(N_BRICKS):
        if r == order[0]:
            continue
        while nxt == FIXED_BRICK:
            nxt += 1
        sigma[r] = nxt
        nxt += 1
    return sigma^


@always_inline
def read_order[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAXC: Int,
    NSITE: Int,
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAXC, nsite=NSITE], 1]
) -> InlineArray[Int, N_BRICKS]:
    """`desired_order`, as written at reset into `META_IDX_TASK_PARAM_0..2`.

    ⚠ `Data.meta` is where per-episode task state lives (see `gpu/constants`);
    `prev_x` is rewritten every step and is the wrong home.
    """
    var order = InlineArray[Int, N_BRICKS](fill=0)
    for i in range(N_BRICKS):
        order[i] = Int(Float64(d.meta.data[META_IDX_TASK_PARAM_0 + i]))
    return order^




# ── the four task hooks, parameterised by `n_order` ────────────────────────


def append_stack_random_obs[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int,
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    m_bodies: List[Scalar[DTYPE]],
    m_joints: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    n_order: Int,
    mut obs: List[Scalar[DTYPE]],
) raises:
    """`desired_order` (n), robot (42), then the bricks in REFERENCE order.

    ⚠⚠ `desired_order` LEADS. It is a TASK observable and composer emits those
    before any entity's, so it is the first `n` numbers — not the last, and not
    adjacent to the brick blocks it refers to.

    ⚠⚠ THE BRICK BLOCKS ARE EMITTED THROUGH `sigma`. Reference brick `r` is our
    physical brick `sigma(r)`, so `block(sigma(0))` comes first even though that
    may be the brick our model keeps at qpos 16. Emitting them in physical order
    compiles, runs, and hands the policy three shuffled 13-vectors that
    disagree with `desired_order`.

    ⚠ ALL THREE BRICKS ARE ALWAYS EMITTED, even when `n_order` is 2. The
    observation is per ENTITY, not per stacked brick; `stack_2_of_3` leaves one
    brick out of the ORDER, not out of the scene.
    """
    var order = read_order[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](d)
    for i in range(n_order):
        obs.append(Scalar[DTYPE](order[i]))
    append_robot_block[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
        d, m_bodies, m_joints, m_sites, ROBOT_SITE_BASE, obs
    )
    var sigma = sigma_of(order)
    for r in range(N_BRICKS):
        append_free_prop_block_site[
            DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
        ](d, m_sites, brick_frame_site_of(sigma[r]), obs)


def stack_random_reward[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int,
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    n_order: Int,
) -> Float64:
    """`Stack.get_reward` — the MEAN over `n_order - 1` pairs.

    ⚠ `pairs = zip(order[:-1], order[1:])` and the mean is over PAIRS, so with
    three bricks it is two terms averaged and with `target_height=2` exactly
    one — not a sum and not the worst. Each pair is the lower brick's STUDS
    against the upper brick's HOLES, through `sigma`.
    """
    var order = read_order[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](d)
    var sigma = sigma_of(order)
    var total = 0.0
    for i in range(n_order - 1):
        var bottom = sigma[order[i]]
        var top = sigma[order[i + 1]]
        var dist = min_stud_to_hole_distance[
            DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
        ](d, brick_stud_0_of(bottom), brick_hole_0_of(top))
        total += pairwise_stacking_reward(dist)
    return total / Float64(n_order - 1)


def stack_random_set_grasp_and_order[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    m_joints: List[Scalar[DTYPE]],
    n_order: Int,
) raises:
    """`set_grasp`, and the ORDER DRAW.

    ⚠ The order is drawn HERE rather than in the full hook because it must be
    in `Data.meta` before anything reads an observation, and because a reset
    that failed the IK would otherwise leave a stale order behind.

    ⚠ `random_state.choice(3, size=n, replace=False)` is a SUBSET when
    `n < 3`, not a permutation. A partial Fisher-Yates gives the same thing:
    shuffle the whole array and take the first `n`. The remaining entries stay
    meaningful — `sigma` needs every reference index mapped, not just the
    ordered ones — so the full array is written to `meta`.
    """
    var qadr = InlineArray[Int, N_HAND](fill=0)
    var rmin = InlineArray[Float64, N_HAND](fill=0.0)
    var rmax = InlineArray[Float64, N_HAND](fill=0.0)
    var factors = InlineArray[Float64, N_HAND](fill=0.0)
    var close = random_float64()
    for i in range(N_HAND):
        var jb = (N_ARM + i) * MODEL_JOINT_SIZE
        qadr[i] = N_ARM + i
        rmin[i] = Float64(m_joints[jb + JOINT_IDX_RANGE_MIN])
        rmax[i] = Float64(m_joints[jb + JOINT_IDX_RANGE_MAX])
        factors[i] = close
    set_grasp[DTYPE, N_HAND](d.qpos.data, qadr, rmin, rmax, factors)

    var order = InlineArray[Int, N_BRICKS](fill=0)
    for i in range(N_BRICKS):
        order[i] = i
    for i in range(N_BRICKS - 1, 0, -1):
        var j = Int(random_float64() * Float64(i + 1))
        if j > i:
            j = i
        var t = order[i]
        order[i] = order[j]
        order[j] = t
    for i in range(N_BRICKS):
        d.meta.data[META_IDX_TASK_PARAM_0 + i] = Scalar[DTYPE](order[i])


def stack_random_reset_full[
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
    MAX_CONTACTS: Int,
    # ⚠ From the task's model def, never defaulted — see `settle_free_props`.
    CONE: Int,
    MAX_CONDIM: Int,
    NOSLIP_ITER: Int,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    timestep: Float64,
) raises:
    """`Stack.initialize_episode` — bricks, grasp (already done), arm.

    ⚠ ALL THREE BRICKS ARE PLACED whatever `target_height` is: `PropPlacer`
    walks `self._bricks`, not `desired_order`.

    ⚠ AND IT WALKS THEM IN REFERENCE INDEX ORDER, which the contact-disabling
    pass makes visible — at step `r` every brick NOT YET PLACED is invisible to
    collision, and "not yet placed" is defined by that order, not by ours.
    """
    comptime MAX_ATT: Int = 10
    comptime MAX_SAMP: Int = 10

    var order = read_order[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](d)
    var sigma = sigma_of(order)

    var lo_p = InlineArray[Float64, 3](fill=0.0)
    lo_p[0] = PROP_BBOX_LOWER_X
    lo_p[1] = PROP_BBOX_LOWER_Y
    lo_p[2] = PROP_BBOX_LOWER_Z
    var hi_p = InlineArray[Float64, 3](fill=0.0)
    hi_p[0] = PROP_BBOX_UPPER_X
    hi_p[1] = PROP_BBOX_UPPER_Y
    hi_p[2] = PROP_BBOX_UPPER_Z

    for r in range(N_BRICKS):
        var phys = sigma[r]
        var ignore = List[Int]()
        for r2 in range(r + 1, N_BRICKS):
            ignore.append(brick_body_of(sigma[r2]))

        var poses = List[Scalar[DTYPE]]()
        for _ in range(MAX_PROP_ATTEMPTS):
            var dr = InlineArray[Float64, 3](fill=0.0)
            for k in range(3):
                dr[k] = random_float64()
            var pp = sample_bbox_uniform[DTYPE](lo_p, hi_p, dr)
            var pq = uniform_z_rotation[DTYPE](random_float64())
            for k in range(3):
                poses.append(pp[k])
            for k in range(4):
                poses.append(pq[k])

        if phys == FIXED_BRICK:
            var rf = place_fixed_prop[
                DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE,
                NEXCL, NMESHV, NPAIR, MAX_CONTACTS,
            ](
                d, mf, brick_body_of(phys), 1, ignore, poses,
                MAX_PROP_ATTEMPTS,
            )
            if not rf.success:
                raise Error(
                    "stack_random: the base brick found no clear pose in "
                    + String(rf.attempts)
                    + " attempts"
                )
        else:
            var slot = free_slot_of(phys)
            var rr = place_free_prop[
                DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE,
                NEXCL, NMESHV, NPAIR, MAX_CONTACTS,
            ](
                d, mf, brick_body_of(phys), brick_qpos_adr_of(slot),
                brick_dof_adr_of(slot), poses, ignore, False,
                MAX_PROP_ATTEMPTS,
            )
            if not rr.success:
                raise Error(
                    "stack_random: brick "
                    + String(r)
                    + " found no non-colliding pose in "
                    + String(rr.attempts)
                    + " attempts"
                )

    # ⚠ EVERY free brick at once: the reference's tolerance is a max over all
    # prop joints, so the scene settles when the LAST brick does.
    var dofs = List[Int]()
    for p in range(N_BRICKS - 1):
        dofs.append(brick_dof_adr_of(p))
    _ = settle_free_props[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS, CONE, MAX_CONDIM, NOSLIP_ITER,
        N_ARM + N_HAND,
    ](d, mf, dofs, timestep)

    var dof_idx = InlineArray[Int, N_ARM](fill=0)
    var qpos_adr = InlineArray[Int, N_ARM](fill=0)
    var lower = InlineArray[Float64, N_ARM](fill=0.0)
    var upper = InlineArray[Float64, N_ARM](fill=0.0)
    for a in range(N_ARM):
        var jb = a * MODEL_JOINT_SIZE
        dof_idx[a] = a
        qpos_adr[a] = a
        var lo = Float64(mf.joints.data[jb + JOINT_IDX_RANGE_MIN])
        var hi = Float64(mf.joints.data[jb + JOINT_IDX_RANGE_MAX])
        if hi >= JOINT_RANGE_UNLIMITED or lo <= -JOINT_RANGE_UNLIMITED:
            lo = 0.0
            hi = TWO_PI
        lower[a] = lo
        upper[a] = hi

    var targets = List[Scalar[DTYPE]]()
    var lo_t = InlineArray[Float64, 3](fill=0.0)
    lo_t[0] = TCP_BBOX_LOWER_X
    lo_t[1] = TCP_BBOX_LOWER_Y
    lo_t[2] = TCP_BBOX_LOWER_Z
    var hi_t = InlineArray[Float64, 3](fill=0.0)
    hi_t[0] = TCP_BBOX_UPPER_X
    hi_t[1] = TCP_BBOX_UPPER_Y
    hi_t[2] = TCP_BBOX_UPPER_Z
    for _ in range(MAX_SAMP):
        var td = InlineArray[Float64, 3](fill=0.0)
        for k in range(3):
            td[k] = random_float64()
        var p = sample_bbox_uniform[DTYPE](lo_t, hi_t, td)
        for k in range(3):
            targets.append(p[k])

    var retry = List[Scalar[DTYPE]]()
    for _ in range(MAX_SAMP * (MAX_ATT - 1)):
        for a in range(N_ARM):
            retry.append(
                Scalar[DTYPE](
                    lower[a] + (upper[a] - lower[a]) * random_float64()
                )
            )

    var down = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    down[0] = Scalar[DTYPE](DOWN_QUAT_XY)
    down[1] = Scalar[DTYPE](DOWN_QUAT_XY)

    # ⚠ THE BODY CLASSES DO NOT DEPEND ON `sigma`. dm_control asks whether a
    # body's top-level body carries a freejoint, and in OUR model that is a
    # fixed fact: `FIXED_BRICK` never has one. The relabeling moves which
    # LOGICAL brick sits there, not which physical one is welded.
    var body_class = InlineArray[Int, NBODY](fill=BODY_FIXED)
    for b in range(NBODY):
        if b >= 2 and b <= 8:
            body_class[b] = BODY_ARM
        elif b >= 10 and b <= 16:
            body_class[b] = BODY_HAND
    for phys in range(N_BRICKS):
        if phys != FIXED_BRICK:
            body_class[brick_body_of(phys)] = BODY_FREE

    var res = tool_center_point_initializer[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS, N_ARM,
    ](
        d, mf, SITE_PINCH, targets, down, dof_idx, qpos_adr,
        lower, upper, retry, body_class, False, MAX_ATT, MAX_SAMP,
    )
    if not res.success:
        raise Error(
            "stack_random: the TCP initializer exhausted "
            + String(res.samples)
            + " samples ("
            + String(res.ik_failures)
            + " IK failures, "
            + String(res.collision_rejections)
            + " collision rejections)"
        )
