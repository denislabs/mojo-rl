"""Evaluating a bound goal — P2c.

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


# ⚠ THE Z BAND FOR CONTAINMENT — see the module header. A region has no height
# in the spec, so `In` accepts anything within this much of the site's plane.
# 12 cm is "a prop sitting in a bin", chosen to be larger than any prop this
# tree ships and smaller than the arm's reach.
comptime IN_HALF_HEIGHT: Float64 = 0.12

# `On` is one-sided: resting ON a surface means ABOVE it, within a prop's
# height. ⚠ A small negative slack absorbs the solver's penetration allowance —
# an object at rest sits microscopically INSIDE the surface it stands on, so a
# hard `dz >= 0` reads a settled object as not-on-the-table.
comptime ON_MIN_DZ: Float64 = -0.005
comptime ON_MAX_DZ: Float64 = 0.08


# ── the scalar predicates — a kernel calls these unchanged ─────────────────


@always_inline
def pred_in_rect(
    px: Float64, py: Float64, pz: Float64,
    sx: Float64, sy: Float64, sz: Float64,
    x_min: Float64, y_min: Float64, x_max: Float64, y_max: Float64,
    dz_min: Float64, dz_max: Float64,
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
def pred_near(
    ax: Float64, ay: Float64, az: Float64,
    bx: Float64, by: Float64, bz: Float64,
    d: Float64,
) -> Bool:
    """⚠ COMPARED SQUARED. A `sqrt` per term per lane per step buys nothing,
    and the two forms agree exactly for a non-negative `d`."""
    var dx = ax - bx
    var dy = ay - by
    var dz = az - bz
    return dx * dx + dy * dy + dz * dz <= d * d


@always_inline
def pred_above(az: Float64, bz: Float64) -> Bool:
    """Strictly higher. ⚠ NO XY TEST, matching LIBERO's own `Above` — "the
    brick is above the table" is about height, and adding a footprint test
    here would silently change what every existing goal means."""
    return az > bz


@always_inline
def pred_upright(qw: Float64, qx: Float64, qy: Float64, qz: Float64,
                 tol: Float64) -> Bool:
    """Is the body's local +z still pointing along world +z, within `tol`?

    ⚠ `R[2][2] = 1 - 2(x^2 + y^2)` — the third diagonal of the rotation
    matrix, which IS the cosine between the two axes. Building the whole
    matrix to read one element is the obvious version and costs eight more
    multiplies per lane per step.

    `tol` is in cosine units: 0 demands exact, 1 accepts a right angle.
    """
    var c = 1.0 - 2.0 * (qx * qx + qy * qy)
    _ = qw
    _ = qz
    return c >= 1.0 - tol


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
            r = pred_above(xpos[t.a * 3 + 2], xpos[t.b * 3 + 2])
        elif t.op == OP_UPRIGHT:
            r = pred_upright(
                xquat[t.a * 4], xquat[t.a * 4 + 1],
                xquat[t.a * 4 + 2], xquat[t.a * 4 + 3],
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
            var dz_min = -IN_HALF_HEIGHT
            var dz_max = IN_HALF_HEIGHT
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
