"""Subtree velocity sensors — MuJoCo `subtreelinvel`.

MuJoCo fills `data.subtree_linvel[body]` in `mj_subtreeVel` and exposes it via
a `<subtreelinvel body="..."/>` sensor. It is the linear velocity of the
subtree's centre of mass:

    subtree_linvel[b] = (sum_i m_i * v_i) / (sum_i m_i),  i in subtree(b)

where `v_i` is the world-frame linear velocity of body i's CoM. Our
`Data.xvel` is exactly that (`_vel_body` propagates
`v = v_parent + w_parent x (xipos_i - xipos_parent)`), so no extra kinematics
are needed — only the mass-weighted walk.

Used by the dm_control suite for the forward-speed term in cheetah, walker,
hopper and humanoid (`sensordata['torso_subtreelinvel']`).

TWO PRECONDITIONS, both easy to get wrong:

  * `xvel` must be current with the integrated `qvel`. The integrator writes
    it mid-step, so an env reading this from a reward hook must set
    `Phyics3dEnvConfig.SYNC_FK_AFTER_STEP` (which also runs `_fields_vel`).
  * body records are the packed `Model.bodies` host list, indexed with the
    `BODY_IDX_*` column constants.

⚠ COMPILE-TIME FOOTGUN — do NOT "simplify" `walk_to_root` back to a `while`.
The parent walk is a BOUNDED `for ... break` on purpose. Written as an
unbounded, data-dependent `while`, calling `subtree_linvel` from inside a loop
took the Mojo compiler from ~2 s to >150 s (never finished). The NESTING is
the trigger: the same `while` called once, not from a loop, compiles fine —
as does `continue`, multiple `List` params, genericity over `DType`, and an
inner `for`. Minimal reproducer, no physics3d involved:

    def g(xs: List[Float64], body: Int, root: Int) -> Bool:
        var b = body
        while b >= 0:                        # <- unbounded, data-dependent
            if b == root: return True
            b = Int(xs[b]) - 1
        return False

    def f(xs: List[Float64], n: Int, root: Int) -> Float64:
        var acc = Float64(0)
        for i in range(n):
            if not g(xs, i, root): continue
            acc += xs[i]
        return acc

    def main():
        var xs = List[Float64]()
        for i in range(21): xs.append(1.0)
        var a = Float64(0)
        for b in range(7): a += f(xs, 7, b)  # <- drop this loop => 2 s
        print(a)

Bounding is not a hack here: a parent chain cannot revisit a body, so `nbody`
is an exact bound.
"""

from layout import Layout, LayoutTensor

from ..gpu.constants import MODEL_BODY_SIZE, BODY_IDX_MASS, BODY_IDX_PARENT


def walk_to_root[
    DTYPE: DType
](
    m_bodies: List[Scalar[DTYPE]], body: Int, root: Int, nbody: Int
) -> Bool:
    """True when `body` is `root` or a descendant of it.

    Bounded by `nbody`. Read the module docstring before making this a
    `while`.
    """
    var b = body
    for _ in range(nbody):
        if b < 0:
            break
        if b == root:
            return True
        b = Int(m_bodies[b * MODEL_BODY_SIZE + BODY_IDX_PARENT])
    return False


def subtree_linvel[
    DTYPE: DType
](
    xvel: List[Scalar[DTYPE]],
    m_bodies: List[Scalar[DTYPE]],
    nbody: Int,
    root: Int,
    mut vx: Float64,
    mut vy: Float64,
    mut vz: Float64,
):
    """`data.subtree_linvel[root]` — CoM velocity of the subtree at `root`.

    `xvel` is `Data.xvel.data` (NBODY*3, world-frame CoM velocity per body);
    `m_bodies` is `Model.bodies.data`. Writes (0,0,0) for a massless subtree,
    matching MuJoCo's guard.
    """
    var total_mass = Float64(0)
    var px = Float64(0)
    var py = Float64(0)
    var pz = Float64(0)

    for b in range(nbody):
        if not walk_to_root(m_bodies, b, root, nbody):
            continue
        var mass = Float64(m_bodies[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
        if mass == 0.0:
            continue
        total_mass += mass
        px += mass * Float64(xvel[b * 3 + 0])
        py += mass * Float64(xvel[b * 3 + 1])
        pz += mass * Float64(xvel[b * 3 + 2])

    if total_mass <= 0.0:
        vx = 0.0
        vy = 0.0
        vz = 0.0
        return
    vx = px / total_mass
    vy = py / total_mass
    vz = pz / total_mass


# =============================================================================
# GPU-batched counterparts
# =============================================================================
#
# The functions above take host `List`s and compute in Float64, so a kernel can
# call neither (Metal has no `double`). These are the same arithmetic over the
# batched field/model tensors, in `DTYPE`.
#
# ⚠ The `for ... break` bound in `walk_to_root_gpu` is load-bearing for the
# SAME reason as on the host — read the module docstring. On the GPU there is a
# second reason: an unbounded data-dependent `while` inside a kernel is a
# divergence hazard, and every lane walks a different chain length.


@always_inline
def walk_to_root_gpu[
    DTYPE: DType, NBODY: Int
](
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    body: Int,
    root: Int,
) -> Bool:
    """True when `body` is `root` or a descendant of it. Batched `walk_to_root`.

    `bodies` is the SHARED (unbatched) `Model.bodies` tensor — the kinematic
    tree is model state, identical across lanes.
    """
    var b = body
    for _ in range(NBODY):
        if b < 0:
            break
        if b == root:
            return True
        b = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
    return False


@always_inline
def subtree_linvel_gpu[
    DTYPE: DType, BATCH_SIZE: Int, NBODY: Int
](
    xvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    env: Int,
    root: Int,
    mut vx: Scalar[DTYPE],
    mut vy: Scalar[DTYPE],
    mut vz: Scalar[DTYPE],
):
    """`data.subtree_linvel[root]` for one lane. Batched `subtree_linvel`.

    Writes (0,0,0) for a massless subtree, matching MuJoCo's guard and the host
    version. This is what feeds every `torso_subtreelinvel` reward in the
    suite's locomotion tasks (cheetah, walker, hopper, humanoid, humanoid_cmu).
    """
    comptime ZERO = Scalar[DTYPE](0)
    var total_mass = ZERO
    var px = ZERO
    var py = ZERO
    var pz = ZERO

    for b in range(NBODY):
        if not walk_to_root_gpu[DTYPE, NBODY](bodies, b, root):
            continue
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        if mass == ZERO:
            continue
        total_mass += mass
        px += mass * rebind[Scalar[DTYPE]](xvel[env, b * 3 + 0])
        py += mass * rebind[Scalar[DTYPE]](xvel[env, b * 3 + 1])
        pz += mass * rebind[Scalar[DTYPE]](xvel[env, b * 3 + 2])

    if total_mass <= ZERO:
        vx = ZERO
        vy = ZERO
        vz = ZERO
        return
    vx = px / total_mass
    vy = py / total_mass
    vz = pz / total_mass
