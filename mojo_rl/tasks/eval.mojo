"""Evaluating a bound goal — P2c.

## ⚠⚠ THE PREDICATES ARE GENERIC OVER DTYPE, AND METAL IS WHY

They used to take `Float64`. **A GPU kernel cannot call that**: Metal has no
double, and an `f64` multiply-add inside a kernel is an LLVM-IR verification
failure, not a slow path — the same wall `so101_park_config`'s repark loop hit.
Parameterising on `T` lets the host call them at float64 and the reward kernel
at `DT` (float32) from ONE definition, which is the only way "the device leg
calls the same predicates" is true rather than aspirational.

⚠ THE BANDS BELOW ARE COMPTIME `Float64` AND ARE CONVERTED AT THE CALL SITE.
`Scalar[T](IN_HALF_HEIGHT)` folds at compile time, so no f64 reaches a kernel.

⚠ PARITY BETWEEN THE LEGS IS STILL EXACT, and not by luck: both legs read the
SAME float32 state out of `Data`, and a comparison is a comparison. What would
break it is one leg computing in f64 from f32 inputs while the other stays in
f32 — so the CPU parity leg must widen nothing the device does not.

## ⚠⚠ THE FACTORING IS THE POINT: SCALAR PREDICATES, HOST DRIVER

Every predicate below is a `def` over **plain scalars** — `pred_in(px, py, pz,
sx, sy, sz, ...)`. Nothing takes a `List`, a `Data`, or a tensor. That is what
lets P3's reward kernel call the SAME function on per-lane values it already
holds, instead of a second implementation that drifts from this one
(`_a_rule_written_inline_twice_drifts` is the shape, and this tree's most
recurring defect).

`eval_goal` — the loop over the term array — is host-side and `List`-based,
because that is what a CPU caller has. **P3 writes its own loop over the device
tape and calls the identical scalar predicates.** The loop is four lines; the
semantics are all down here.

⚠ THE FORWARD SWEEP IS LEGAL BECAUSE THE TERMS ARE POST-ORDER — every child's
index is lower than its parent's, asserted in `test_goal_language`. A parent
reading a later term would read an unwritten slot, and only on device.

## ⚠ WHERE THE GEOMETRY IS NOT YET SPECIFIED, AND IT IS VISIBLE

A region carries a site and an XY rectangle and NO HEIGHT (`spec.mojo`). So
`In` and `On` have to supply a z band themselves, and the constants below are
where that guess lives. They are deliberately in one place with a note rather
than folded into the arithmetic: when a task needs a real containment volume,
the fix is a `height=` on the region, not a tuned constant here.
"""

from std.math import sqrt

from mojo_rl.physics3d.kinematics.quat_math import gpu_quat_mul, gpu_quat_rotate

from .spec import FamilySpec
from .predicates import (
    BoundGoal,
    OP_IN, OP_ON, OP_NEAR, OP_ABOVE, OP_UPRIGHT, OP_OPEN, OP_AT_REGION,
    OP_TOUCHING, OP_GRASPED, OP_AND, OP_OR, OP_NOT, OP_JOINT, OP_ON_BODY,
    CMP_LT, CMP_LE, CMP_GT, CMP_GE,
    op_name, slot_body_id,
)


# ⚠⚠ THE FALLBACK Z BAND, AND IT IS NO LONGER THE ONLY ONE. A region now
# CARRIES a `half_height` (`spec.RegionSpec`), and this is what one gets when
# the `.family` does not say — `DEFAULT_REGION_HALF_HEIGHT`, restated there
# because `spec` cannot import this module without a cycle.
#
# ⚠ IT STAYS 0.12 ON PURPOSE. Changing the default would silently retune every
# region in every family that never asked for a band; the point of the field is
# that a task which NEEDS a real volume says so. What 0.12 costs when nobody
# says so is on the record: against a 0.20 x 0.20 rect it is a 0.0096 m^3 box
# in the middle of the arm's workspace, and an UNTRAINED greedy actor met
# `AtRegion(robot_gripperframe, table_top)` on 64 of 64 episodes.
comptime IN_HALF_HEIGHT: Float64 = 0.12

# `On` is one-sided: resting ON a surface means ABOVE it, within a prop's
# height. ⚠ A small negative slack absorbs the solver's penetration allowance —
# an object at rest sits microscopically INSIDE the surface it stands on, so a
# hard `dz >= 0` reads a settled object as not-on-the-table.
comptime ON_MIN_DZ: Float64 = -0.005
comptime ON_MAX_DZ: Float64 = 0.08


# ── the scalar predicates — a kernel calls these unchanged ─────────────────


@always_inline
def pred_in_rect[T: DType = DType.float64](
    px: Scalar[T], py: Scalar[T], pz: Scalar[T],
    sx: Scalar[T], sy: Scalar[T], sz: Scalar[T],
    x_min: Scalar[T], y_min: Scalar[T], x_max: Scalar[T], y_max: Scalar[T],
    dz_min: Scalar[T], dz_max: Scalar[T],
) -> Bool:
    """Is point `p` inside the rectangle around site `s`, within a z band?

    The shared body of `In` and `On`: they differ ONLY in the band, which is
    why there is one function and two callers rather than two near-copies.
    """
    var dx = px - sx
    var dy = py - sy
    var dz = pz - sz
    return (
        dx >= x_min and dx <= x_max
        and dy >= y_min and dy <= y_max
        and dz >= dz_min and dz <= dz_max
    )


@always_inline
def pred_near[T: DType = DType.float64](
    ax: Scalar[T], ay: Scalar[T], az: Scalar[T],
    bx: Scalar[T], by: Scalar[T], bz: Scalar[T],
    d: Scalar[T],
) -> Bool:
    """⚠ COMPARED SQUARED. A `sqrt` per term per lane per step buys nothing,
    and the two forms agree exactly for a non-negative `d`."""
    var dx = ax - bx
    var dy = ay - by
    var dz = az - bz
    return dx * dx + dy * dy + dz * dz <= d * d


@always_inline
def pred_above[T: DType = DType.float64](
    az: Scalar[T], bz: Scalar[T], margin: Scalar[T]
) -> Bool:
    """Higher than `b` by at least `margin` metres.

    ⚠ NO XY TEST, matching LIBERO's own `Above` — "the brick is above the
    table" is about height, and adding a footprint test would silently change
    what every existing goal means.

    ⚠⚠ THE MARGIN IS NOT A REFINEMENT, IT IS WHAT MAKES THE GOAL MEAN
    ANYTHING. Body ORIGINS are what these are: a prop resting on a table's top
    face already sits above the table body's origin, so `margin = 0` makes
    "lift the brick" true before the arm moves.
    """
    return az - bz >= margin


@always_inline
def pred_upright[T: DType = DType.float64](
    qw: Scalar[T], qx: Scalar[T], qy: Scalar[T], qz: Scalar[T],
    tol: Scalar[T],
) -> Bool:
    """Is the body's local +z still pointing along world +z, within `tol`?

    ⚠ `R[2][2] = 1 - 2(x^2 + y^2)` — the third diagonal of the rotation
    matrix, which IS the cosine between the two axes. Building the whole
    matrix to read one element is the obvious version and costs eight more
    multiplies per lane per step.

    `tol` is in cosine units: 0 demands exact, 1 accepts a right angle.
    """
    var c = Scalar[T](1) - Scalar[T](2) * (qx * qx + qy * qy)
    _ = qw
    _ = qz
    return c >= Scalar[T](1) - tol


# ── LIBERO's SiteObject constants, quoted (L3) ─────────────────────────────
# `libero/envs/objects/site_object.py`:
#   in_box:  lb = pos - |R @ size|; lb[2] -= 0.01; ub = pos + |R @ size|;
#            all(other > lb) and all(other < ub)
#   under:   delta = R @ (other - pos);
#            size[2] - 0.005 < delta[2] < size[2] + 0.10
#            and all(|delta[:2]| < size[:2])
# ⚠ `R @ v`, NOT `R^T @ v`. LIBERO multiplies by `site_xmat` where a frame
# change would use its transpose. For the corpus' fixed yaws of 0 and pi
# the two coincide on every axis that is tested; for the wine rack's tilted
# site (`quat 0.855 -0.519 0 0`) they do not, and the benchmark's success is
# defined by the line as written. Transcribed, not corrected.
comptime BOX_IN_Z_SLACK: Float64 = 0.01
comptime BOX_UNDER_Z_SLACK: Float64 = 0.005
comptime BOX_UNDER_HEIGHT: Float64 = 0.10
# `ObjectState.check_ontop`: `norm(this[:2] - other[:2]) < 0.03`.
comptime ONTOP_XY: Float64 = 0.03


@always_inline
def pred_joint[T: DType = DType.float64](
    q: Scalar[T], cmp: Int, thr: Scalar[T]
) -> Bool:
    """`q <cmp> thr` — the four comparisons of `Joint`, spelled ONCE."""
    if cmp == CMP_LT:
        return q < thr
    if cmp == CMP_LE:
        return q <= thr
    if cmp == CMP_GT:
        return q > thr
    if cmp == CMP_GE:
        return q >= thr
    return False


@always_inline
def pred_ontop[T: DType = DType.float64](
    ax: Scalar[T], ay: Scalar[T], az: Scalar[T],
    bx: Scalar[T], by: Scalar[T], bz: Scalar[T],
    touching: Bool,
) -> Bool:
    """`On(a, b)` for two OBJECTS — LIBERO's `b.check_ontop(a)`, verbatim:

        this = b, other = a
        this_z <= other_z  and  contact(this, other)  and  |xy(this)-xy(a)| < 0.03

    ⚠⚠ THE ARGUMENT INVERSION IS RESOLVED HERE. `predicates.On.__call__` is
    `arg2.check_ontop(arg1)`, so inside LIBERO "this" is the TARGET. A
    reader who transcribes `check_ontop(self, other)` with self = the
    subject gets the z order backwards and `On(bowl, plate)` true when the
    bowl is UNDER the plate. `a` here is the subject named first in the goal.

    ⚠ COMPARED SQUARED, as `pred_near`: `norm < 0.03` and `dx²+dy² < 0.03²`
    agree exactly for a non-negative bound.
    """
    var dx = ax - bx
    var dy = ay - by
    var r = Scalar[T](ONTOP_XY)
    return bz <= az and touching and (dx * dx + dy * dy) < r * r


@always_inline
def pred_box_in[T: DType = DType.float64](
    px: Scalar[T], py: Scalar[T], pz: Scalar[T],
    sx: Scalar[T], sy: Scalar[T], sz: Scalar[T],
    qx: Scalar[T], qy: Scalar[T], qz: Scalar[T], qw: Scalar[T],
    cx: Scalar[T], cy: Scalar[T],
    hx: Scalar[T], hy: Scalar[T], hz: Scalar[T],
) -> Bool:
    """LIBERO's `SiteObject.in_box` — a WORLD-axis-aligned box of half-size
    `|R @ (hx, hy, hz)|` around the site, 1 cm of extra room below.

    `(cx, cy)` is the rectangle's centre in the site's frame — zero for a
    fixture's own site, the zone's centroid for a table region anchored on
    the workspace site (LIBERO gives each zone its own site AT the
    centroid; ours share the anchor and carry the offset). The box sits at
    `s + R @ (cx, cy, 0)`, which IS the zone site's position.

    ⚠ `|R @ size|` IS WHAT THEY DO — their own comment calls it "a little
    bit hacky". For the cabinet's drawers (`quat 0.707 0 0.707 0`, 90 deg
    about y) it swaps the x and z half-sizes, and that swapped box is the
    benchmark's definition of "in the drawer".
    """
    var c = gpu_quat_rotate[T](qx, qy, qz, qw, cx, cy, Scalar[T](0))
    var ts = gpu_quat_rotate[T](qx, qy, qz, qw, hx, hy, hz)
    var tx = ts[0] if ts[0] >= Scalar[T](0) else -ts[0]
    var ty = ts[1] if ts[1] >= Scalar[T](0) else -ts[1]
    var tz = ts[2] if ts[2] >= Scalar[T](0) else -ts[2]
    var ox = sx + c[0]
    var oy = sy + c[1]
    var oz = sz + c[2]
    var slack = Scalar[T](BOX_IN_Z_SLACK)
    return (
        px > ox - tx and px < ox + tx
        and py > oy - ty and py < oy + ty
        and pz > oz - tz - slack and pz < oz + tz
    )


@always_inline
def pred_box_under[T: DType = DType.float64](
    px: Scalar[T], py: Scalar[T], pz: Scalar[T],
    sx: Scalar[T], sy: Scalar[T], sz: Scalar[T],
    qx: Scalar[T], qy: Scalar[T], qz: Scalar[T], qw: Scalar[T],
    cx: Scalar[T], cy: Scalar[T],
    hx: Scalar[T], hy: Scalar[T], hz: Scalar[T],
    touching: Bool,
) -> Bool:
    """LIBERO's `SiteObject.under` — `On(obj, site)`: the object's origin,
    taken through `R @ (p - site)`, is within the site's xy half-sizes and
    between `hz - 0.005` and `hz + 0.10` above it; plus the contact the
    region names (`SiteObjectState.check_ontop` calls
    `env.check_contact(parent_object, other_object)` when the site has a
    parent). `touching` is True when the region names no contact slot.
    """
    var c = gpu_quat_rotate[T](qx, qy, qz, qw, cx, cy, Scalar[T](0))
    var d = gpu_quat_rotate[T](
        qx, qy, qz, qw, px - (sx + c[0]), py - (sy + c[1]), pz - (sz + c[2])
    )
    var ax = d[0] if d[0] >= Scalar[T](0) else -d[0]
    var ay = d[1] if d[1] >= Scalar[T](0) else -d[1]
    return (
        touching
        and d[2] > hz - Scalar[T](BOX_UNDER_Z_SLACK)
        and d[2] < hz + Scalar[T](BOX_UNDER_HEIGHT)
        and ax < hx and ay < hy
    )


@always_inline
def body_in_slot(x: Int, root: Int, parent: List[Int]) -> Bool:
    """Is body `x` the slot root `root` or one of its descendants?

    ⚠ BODY IDS ARE IN TREE ORDER, so every descendant's id exceeds its
    ancestor's and the walk `x = parent[x]` while `x > root` is bounded by
    the tree depth and needs no visited set. `parent[0]` (the world) is
    never read because `root >= 1` for any slot. The device leg walks
    `bodies[x, BODY_IDX_PARENT]` under the same rule.
    """
    var b = x
    while b > root:
        b = parent[b]
    return b == root


def slots_touching(
    root_a: Int, root_b: Int,
    ncon: Int, con_a: List[Int], con_b: List[Int], parent: List[Int],
) -> Bool:
    """Any contact whose two bodies fall one in each slot — robosuite's
    `check_contact(model_a, model_b)` over the models' contact geoms,
    where a model's geoms are the geoms of its bodies."""
    for k in range(ncon):
        var ba = con_a[k]
        var bb = con_b[k]
        if body_in_slot(ba, root_a, parent) and body_in_slot(bb, root_b, parent):
            return True
        if body_in_slot(bb, root_a, parent) and body_in_slot(ba, root_b, parent):
            return True
    return False


def region_sites(f: FamilySpec, site_names: List[String]) raises -> List[Int]:
    """Each region's site id, in family region order.

    ⚠ ONE SPELLING, resolved once. `eval_goal` and the sampler both need
    "where is region r right now", and a caller that resolved it inline in two
    places would eventually resolve it two different ways. Raises if a region
    names a site the scene does not have — which is exactly the defect P2a
    found sitting in the first family file.
    """
    from .predicates import site_id

    var out = List[Int]()
    for i in range(len(f.regions)):
        out.append(site_id(f.regions[i].site, site_names))
    return out^


def region_rects(f: FamilySpec) -> List[List[Float64]]:
    """`[xmin, ymin, xmax, ymax]` per region, in family order.

    ⚠ THE NO-RECTANGLE CASE IS RESOLVED HERE, ONCE. A region with no rect is
    the site's own extent, which `eval_goal` renders as a token radius; the
    device tape has no `has_rect` flag and must not grow one, so the two
    readers agree by both taking the rectangle from this function.
    """
    var out = List[List[Float64]]()
    for i in range(len(f.regions)):
        ref r = f.regions[i]
        var q = List[Float64]()
        if r.has_rect:
            q.append(r.x_min)
            q.append(r.y_min)
            q.append(r.x_max)
            q.append(r.y_max)
        else:
            q.append(-0.02)
            q.append(-0.02)
            q.append(0.02)
            q.append(0.02)
        out.append(q^)
    return out^


def region_half_heights(f: FamilySpec) -> List[Float64]:
    """The z half-band per region, in family order — the device's copy.

    ⚠ A SEPARATE FUNCTION AND NOT A FIFTH ELEMENT OF `region_rects`. Every
    caller of that unpacks `rects[i][0..3]` positionally, and a list that is
    sometimes four long and sometimes five is the kind of change that reads
    correctly at every call site and is wrong at one of them.

    ⚠ THE NO-RECTANGLE CASE STILL HAS A BAND. A rect-less region degenerates
    to the site's own extent in XY (`region_rects` renders that as +-0.02) and
    its z band is untouched by that — `has_rect` and `has_height` are
    independent, and defaulting one from the other would silently give a
    site-extent region a 2 cm z band it never asked for.
    """
    var out = List[Float64]()
    for i in range(len(f.regions)):
        out.append(f.regions[i].half_height)
    return out^


def region_box_flags(f: FamilySpec) -> List[Int]:
    """1 for a `:box:` region, else 0 — word 6 of the device region table."""
    var out = List[Int]()
    for i in range(len(f.regions)):
        out.append(1 if f.regions[i].is_box else 0)
    return out^


def region_contact_bodies(
    f: FamilySpec, body_names: List[String]
) raises -> List[Int]:
    """The root body of each region's contact slot, or -1 — word 7 of the
    device region table. Resolved ONCE; `parse_family` already refused a
    contact slot that is not a slot."""
    var out = List[Int]()
    for i in range(len(f.regions)):
        ref r = f.regions[i]
        if r.is_box and r.contact.byte_length() > 0:
            out.append(slot_body_id(r.contact, body_names))
        else:
            out.append(-1)
    return out^


# ── the host state — what L3's predicates read beyond xpos/xquat ──────────


struct HostState(Copyable, Movable):
    """Everything `eval_goal` reads, as flat host lists.

    ⚠ ONE STRUCT INSTEAD OF A TWELVE-ARGUMENT SIGNATURE, and every list is
    OPTIONAL BY LENGTH: a caller with no contacts leaves `con_a` empty and
    `ncon` 0, one with no joints leaves `qpos` empty. The evaluator RAISES
    when a term needs a list that is empty — it never reads a default.

    Layouts match `Data`: `xquat` (x, y, z, w) W LAST; `site_quat` the
    model's LOCAL site quaternion per site, also (x, y, z, w) as
    `SITE_IDX_QUAT_*`; `site_body` the site's body id (world = 0);
    `body_parent[b]` the parent id (index 0 is the world, unread);
    `con_a[k]` / `con_b[k]` the two body ids of contact k, `k < ncon`.
    """

    var xpos: List[Float64]
    var xquat: List[Float64]
    var site_xpos: List[Float64]
    var qpos: List[Float64]
    var site_body: List[Int]
    var site_quat: List[Float64]
    var body_parent: List[Int]
    var ncon: Int
    var con_a: List[Int]
    var con_b: List[Int]

    def __init__(
        out self,
        var xpos: List[Float64],
        var xquat: List[Float64],
        var site_xpos: List[Float64],
    ):
        self.xpos = xpos^
        self.xquat = xquat^
        self.site_xpos = site_xpos^
        self.qpos = List[Float64]()
        self.site_body = List[Int]()
        self.site_quat = List[Float64]()
        self.body_parent = List[Int]()
        self.ncon = 0
        self.con_a = List[Int]()
        self.con_b = List[Int]()


# ── the host driver ───────────────────────────────────────────────────────


def eval_goal(
    g: BoundGoal,
    f: FamilySpec,
    xpos: List[Float64],
    xquat: List[Float64],
    site_xpos: List[Float64],
    region_site: List[Int],
) raises -> Bool:
    """The pre-L3 signature: poses only. Builds a `HostState` with no
    joints, no site frames and no contacts and evaluates; a term or a box
    region that needs one of those RAISES from the wide overload."""
    var st = HostState(xpos.copy(), xquat.copy(), site_xpos.copy())
    var no_contact = List[Int]()
    return eval_goal(g, f, st, region_site, no_contact)


def eval_goal(
    g: BoundGoal,
    f: FamilySpec,
    st: HostState,
    region_site: List[Int],
    region_contact: List[Int],
) raises -> Bool:
    """True when the goal holds. `xpos`/`xquat` are indexed by BODY ID
    (world at 0, matching `FlatModelDef.body_names`); `site_xpos` by site id;
    `region_site[r]` is the site id region `r` attaches to and
    `region_contact[r]` the root body of its contact slot or -1
    (`region_contact_bodies`; may be empty when no region names one).

    ⚠ TIER B RAISES HERE, and so does any term whose input `st` does not
    carry (see `HostState`). A silent False is a goal that never fires: the
    task trains against a flat-zero reward and every curve looks healthy.
    """
    var n = len(g.terms)
    if n == 0:
        raise Error("tasks: cannot evaluate an empty goal")
    var val = List[Bool]()
    for _ in range(n):
        val.append(False)

    for i in range(n):
        ref t = g.terms[i]
        # ⚠ NO INITIALISER: every branch below assigns, and the final `else`
        # raises. A `= False` default would make an op this switch forgot
        # evaluate as "goal not met" instead of failing loudly.
        var r: Bool

        if t.op == OP_AND:
            r = val[t.a] and val[t.b]
        elif t.op == OP_OR:
            r = val[t.a] or val[t.b]
        elif t.op == OP_NOT:
            r = not val[t.a]
        elif t.op == OP_NEAR:
            r = pred_near(
                st.xpos[t.a * 3], st.xpos[t.a * 3 + 1], st.xpos[t.a * 3 + 2],
                st.xpos[t.b * 3], st.xpos[t.b * 3 + 1], st.xpos[t.b * 3 + 2],
                t.param,
            )
        elif t.op == OP_ABOVE:
            r = pred_above(
                st.xpos[t.a * 3 + 2], st.xpos[t.b * 3 + 2], t.param
            )
        elif t.op == OP_UPRIGHT:
            # ⚠⚠ `Data.xquat` IS (x, y, z, w) — W IS LAST. Verified against
            # five independent consumers (`sensors/touch.mojo:146`,
            # `dynamics/tendon.mojo:122`, `pose_transmission.mojo:479`,
            # `fluid_forces.mojo:146`, and the studio's draw), every one of
            # which reads `[b*4 + 3]` as w.
            #
            # ⚠ THIS WAS WRONG FOR A COMMIT, and the gate could not see it:
            # `test_task_eval` CONSTRUCTED the quaternion array under the same
            # (w,x,y,z) assumption the evaluator made, so the two agreed and
            # both were wrong — `_a_gate_that_shares_its_reference_
            # implementation_is_blind`. It surfaced from reading the studio's
            # render code, not from a test. `test_task_reset_steps` now
            # evaluates Upright against a REAL `Data.xquat` so it cannot
            # drift back.
            r = pred_upright(
                st.xquat[t.a * 4 + 3],
                st.xquat[t.a * 4 + 0],
                st.xquat[t.a * 4 + 1],
                st.xquat[t.a * 4 + 2],
                t.param,
            )
        elif t.op == OP_JOINT:
            if len(st.qpos) == 0:
                raise Error(
                    "tasks: Joint needs `qpos`, which this HostState does"
                    " not carry"
                )
            r = pred_joint(st.qpos[t.a], t.b, t.param)
        elif t.op == OP_TOUCHING or t.op == OP_ON_BODY:
            if len(st.body_parent) == 0:
                raise Error(
                    "tasks: " + op_name(t.op) + " needs the contact list and"
                    " `body_parent`, which this HostState does not carry"
                )
            var touching = slots_touching(
                t.a, t.b, st.ncon, st.con_a, st.con_b, st.body_parent
            )
            if t.op == OP_TOUCHING:
                r = touching
            else:
                r = pred_ontop(
                    st.xpos[t.a * 3], st.xpos[t.a * 3 + 1], st.xpos[t.a * 3 + 2],
                    st.xpos[t.b * 3], st.xpos[t.b * 3 + 1], st.xpos[t.b * 3 + 2],
                    touching,
                )
        elif t.op == OP_IN or t.op == OP_ON or t.op == OP_AT_REGION:
            ref reg = f.regions[t.b]
            var s = region_site[t.b]
            # ⚠ AT_REGION's SUBJECT IS A SITE, `In`/`On`'s IS A BODY. One
            # branch, two coordinate sources — mixing them up reads a site id
            # out of the body array and lands on a real, wrong position.
            var px: Float64
            var py: Float64
            var pz: Float64
            if t.op == OP_AT_REGION:
                px = st.site_xpos[t.a * 3]
                py = st.site_xpos[t.a * 3 + 1]
                pz = st.site_xpos[t.a * 3 + 2]
            else:
                px = st.xpos[t.a * 3]
                py = st.xpos[t.a * 3 + 1]
                pz = st.xpos[t.a * 3 + 2]
            if reg.is_box and t.op != OP_AT_REGION:
                # ── LIBERO's SiteObject (L3) ──────────────────────────────
                if len(st.site_quat) == 0 or len(st.site_body) == 0:
                    raise Error(
                        "tasks: box region '" + reg.name + "' needs the"
                        " site table (site_body, site_quat), which this"
                        " HostState does not carry"
                    )
                # the site's WORLD orientation: xquat[site body] * local
                var sb = st.site_body[s]
                var wq = gpu_quat_mul[DType.float64](
                    st.xquat[sb * 4 + 0], st.xquat[sb * 4 + 1],
                    st.xquat[sb * 4 + 2], st.xquat[sb * 4 + 3],
                    st.site_quat[s * 4 + 0], st.site_quat[s * 4 + 1],
                    st.site_quat[s * 4 + 2], st.site_quat[s * 4 + 3],
                )
                var cx = 0.5 * (reg.x_min + reg.x_max)
                var cy = 0.5 * (reg.y_min + reg.y_max)
                if t.op == OP_IN:
                    r = pred_box_in(
                        px, py, pz,
                        st.site_xpos[s * 3], st.site_xpos[s * 3 + 1],
                        st.site_xpos[s * 3 + 2],
                        wq[0], wq[1], wq[2], wq[3],
                        cx, cy, reg.half_x(), reg.half_y(), reg.half_height,
                    )
                else:
                    var touching = True
                    if reg.contact.byte_length() > 0:
                        if len(st.body_parent) == 0 or len(region_contact) == 0:
                            raise Error(
                                "tasks: box region '" + reg.name + "' names"
                                " a contact slot and needs the contact list"
                                " plus `region_contact` (see"
                                " `region_contact_bodies`), which this call"
                                " does not carry"
                            )
                        touching = slots_touching(
                            t.a, region_contact[t.b], st.ncon, st.con_a,
                            st.con_b, st.body_parent,
                        )
                    r = pred_box_under(
                        px, py, pz,
                        st.site_xpos[s * 3], st.site_xpos[s * 3 + 1],
                        st.site_xpos[s * 3 + 2],
                        wq[0], wq[1], wq[2], wq[3],
                        cx, cy, reg.half_x(), reg.half_y(), reg.half_height,
                        touching,
                    )
            else:
                # ⚠ THE REGION'S OWN BAND, NOT THE CONSTANT. `reg.half_height`
                # is `DEFAULT_REGION_HALF_HEIGHT` unless the `.family` said
                # otherwise, so this is identical for every region that does
                # not state one — and is the whole fix for the ones that do.
                var dz_min = -reg.half_height
                var dz_max = reg.half_height
                if t.op == OP_ON:
                    dz_min = ON_MIN_DZ
                    dz_max = ON_MAX_DZ
                # ⚠ A REGION WITH NO RECTANGLE IS THE SITE'S OWN EXTENT, which
                # the spec defines and this file has to honour: with no rect
                # there is no area, so containment degenerates to "within the
                # z band and within a token XY radius". Left explicit rather
                # than silently accepting everything, which is what a
                # zero-sized rect would do.
                var x0 = reg.x_min
                var y0 = reg.y_min
                var x1 = reg.x_max
                var y1 = reg.y_max
                if not reg.has_rect:
                    x0 = -0.02
                    y0 = -0.02
                    x1 = 0.02
                    y1 = 0.02
                r = pred_in_rect(
                    px, py, pz,
                    st.site_xpos[s * 3], st.site_xpos[s * 3 + 1],
                    st.site_xpos[s * 3 + 2],
                    x0, y0, x1, y1, dz_min, dz_max,
                )
        elif t.op == OP_GRASPED or t.op == OP_OPEN:
            raise Error(
                "tasks: " + op_name(t.op) + " cannot be evaluated by"
                " `eval_goal` — it has no defined semantics (Grasped) or is"
                " not bound (Open; use Joint). Refused rather than returning"
                " False: a goal that silently never fires trains against a"
                " flat-zero reward and every curve looks healthy."
            )
        else:
            raise Error("tasks: unhandled predicate op " + String(t.op))
        val[i] = r

    return val[g.root()]
