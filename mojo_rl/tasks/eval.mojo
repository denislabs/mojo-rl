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

from math import sqrt

from .spec import FamilySpec
from .predicates import (
    BoundGoal,
    OP_IN, OP_ON, OP_NEAR, OP_ABOVE, OP_UPRIGHT, OP_OPEN, OP_AT_REGION,
    OP_TOUCHING, OP_GRASPED, OP_AND, OP_OR, OP_NOT,
    op_name,
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


# ── the host driver ───────────────────────────────────────────────────────


def eval_goal(
    g: BoundGoal,
    f: FamilySpec,
    xpos: List[Float64],
    xquat: List[Float64],
    site_xpos: List[Float64],
    region_site: List[Int],
) raises -> Bool:
    """True when the goal holds. `xpos`/`xquat` are indexed by BODY ID
    (world at 0, matching `FlatModelDef.body_names`); `site_xpos` by site id;
    `region_site[r]` is the site id region `r` attaches to.

    ⚠ TIER B RAISES HERE. `Touching`/`Grasped` need the contact array, which
    this signature deliberately does not take — a caller that has contacts can
    evaluate them, and one that does not must not get a silent `False`. A
    silent False is a goal that never fires: the task trains against a
    flat-zero reward and every curve looks healthy.
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
                xpos[t.a * 3], xpos[t.a * 3 + 1], xpos[t.a * 3 + 2],
                xpos[t.b * 3], xpos[t.b * 3 + 1], xpos[t.b * 3 + 2],
                t.param,
            )
        elif t.op == OP_ABOVE:
            r = pred_above(
                xpos[t.a * 3 + 2], xpos[t.b * 3 + 2], t.param
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
                xquat[t.a * 4 + 3],
                xquat[t.a * 4 + 0],
                xquat[t.a * 4 + 1],
                xquat[t.a * 4 + 2],
                t.param,
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
                px = site_xpos[t.a * 3]
                py = site_xpos[t.a * 3 + 1]
                pz = site_xpos[t.a * 3 + 2]
            else:
                px = xpos[t.a * 3]
                py = xpos[t.a * 3 + 1]
                pz = xpos[t.a * 3 + 2]
            # ⚠ THE REGION'S OWN BAND, NOT THE CONSTANT. `reg.half_height`
            # is `DEFAULT_REGION_HALF_HEIGHT` unless the `.family` said
            # otherwise, so this is identical for every region that does not
            # state one — and is the whole fix for the ones that do.
            var dz_min = -reg.half_height
            var dz_max = reg.half_height
            if t.op == OP_ON:
                dz_min = ON_MIN_DZ
                dz_max = ON_MAX_DZ
            # ⚠ A REGION WITH NO RECTANGLE IS THE SITE'S OWN EXTENT, which the
            # spec defines and this file has to honour: with no rect there is
            # no area, so containment degenerates to "within the z band and
            # within a token XY radius". Left explicit rather than silently
            # accepting everything, which is what a zero-sized rect would do.
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
                site_xpos[s * 3], site_xpos[s * 3 + 1], site_xpos[s * 3 + 2],
                x0, y0, x1, y1, dz_min, dz_max,
            )
        elif t.op == OP_TOUCHING or t.op == OP_GRASPED or t.op == OP_OPEN:
            raise Error(
                "tasks: " + op_name(t.op) + " cannot be evaluated by"
                " `eval_goal` — it needs state this signature does not carry"
                " (contacts, or a joint range). Refused rather than returning"
                " False: a goal that silently never fires trains against a"
                " flat-zero reward and every curve looks healthy."
            )
        else:
            raise Error("tasks: unhandled predicate op " + String(t.op))
        val[i] = r

    return val[g.root()]
