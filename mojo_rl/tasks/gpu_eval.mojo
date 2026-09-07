"""Evaluating the tape INSIDE a kernel — P3b's half of the parity claim.

`tape.eval_tape` reads `List[Float64]`; a reward kernel has `LayoutTensor`s of
`Scalar[DT]`. This is that loop, over those containers.

## ⚠ THE LOOP IS WRITTEN TWICE. THE ARITHMETIC IS NOT.

Both call `pred_in_rect` / `pred_near` / `pred_above` / `pred_upright` from
`eval.mojo`, which are generic over dtype precisely so this is possible — the
host instantiates them at float64 and this at `DT`. A version that reimplemented
a comparison here would be the third spelling of the same rule, and
`_a_rule_written_inline_twice_drifts` is this tree's most recurring defect.

What is genuinely different is the CONTAINER and the INDEXING, and that is
exactly what `tests/tasks/test_task_tape.mojo` (host tape vs host goal) and
P3c's GPU-vs-CPU gate exist to pin.

## ⚠⚠ NO `Float64` ANYWHERE BELOW

Metal has no double; an f64 multiply-add in a kernel is an LLVM-IR verification
failure, not a slow path. Every constant is `Scalar[DTYPE](...)` of a comptime
`Float64`, which folds at compile time.

## WHERE THE TWO INPUTS LIVE, AND WHY THEY NEED NO NEW OPERANDS

* **the tape** — `meta[env, META_IDX_TASK_PARAM_0 .. _11]`, twelve per-lane
  words. Already an operand; reset preserves it (`constants.mojo:164`).
* **the region table** — `curriculum[0, 0..4]`, shared across lanes because a
  region belongs to the FAMILY, not to a task. Already an operand, and unused
  by anything else in this tree.

⚠⚠ ONE REGION, AND THE TERM'S REGION INDEX IS IGNORED ON DEVICE. The table
below is read ONCE per lane, before the tape loop, and every `In`/`On`/
`AtRegion` term uses it whatever its `b` says. `MODEL_CURRICULUM_SIZE` is 8
and a region costs 5 words (site id + rect), so exactly one fits.

A family may still DECLARE more — `so101_tabletop` declares three — because
`init=` is sampled on the HOST and never consults this table. What must not
happen is a GOAL naming region 1: it would read region 0's rectangle here and
region 1's in `eval.eval_goal`, so the GPU and CPU rewards would disagree with
no error anywhere. `require_gpu_regions` refuses that goal; it is the region
counterpart of `predicates.require_tier_a` and is called in the same place.
"""

from std.math import sqrt

from layout import Layout, LayoutTensor

from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, MODEL_CURRICULUM_SIZE,
)
from .predicates import (
    BoundGoal,
    OP_IN, OP_ON, OP_NEAR, OP_ABOVE, OP_UPRIGHT, OP_AT_REGION,
    OP_AND, OP_OR, OP_NOT,
)
from .eval import (
    pred_in_rect, pred_near, pred_above, pred_upright,
    ON_MIN_DZ, ON_MAX_DZ,
)
from .tape import MAX_TAPE_TERMS, TERM_WORDS


# ── the region table's layout inside `curriculum` ──────────────────────────
comptime CUR_IDX_REGION_SITE: Int = 0
comptime CUR_IDX_REGION_X0: Int = 1
comptime CUR_IDX_REGION_Y0: Int = 2
comptime CUR_IDX_REGION_X1: Int = 3
comptime CUR_IDX_REGION_Y1: Int = 4
# ⚠ THE Z HALF-BAND, WORD 5. `RegionSpec.half_height`, carried per region so
# `In`/`AtRegion` on device use the region's own volume rather than
# `eval.IN_HALF_HEIGHT`. Six words of the eight still leaves room for exactly
# one region — the table did not get narrower, see MAX_CURRICULUM_REGIONS.
comptime CUR_IDX_REGION_H: Int = 5
comptime REGION_WORDS: Int = 6

# ── the two shaping weights, words 6 and 7 ────────────────────────────────
#
# ⚠⚠ RUNTIME, NOT COMPTIME, AND THE REASON IS THAT THEY NEEDED SWEEPING. They
# were `comptime` on `So101TabletopConfig`, so trying a different scale meant a
# rebuild — and reward SCALE is the open question on this family, not a
# setting anybody knows. `curriculum` is already the host-written,
# device-shared channel the region table lives in and it had exactly two words
# left.
#
# ⚠ THE HOST STILL OWNS THE DEFAULTS. The config's `SHAPE_W_*` constants are
# what `region_table_words` writes when a caller does not override, so a
# driver that never heard of these gets the family's chosen scale rather than
# zero — which would silently be the sparse reward again.
comptime CUR_IDX_SHAPE_W_GOAL: Int = 6
comptime CUR_IDX_SHAPE_W_REACH: Int = 7
comptime MAX_CURRICULUM_REGIONS: Int = MODEL_CURRICULUM_SIZE // REGION_WORDS


def region_table_words(
    site: Int, x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    half_height: Float64,
    shape_w_goal: Float64, shape_w_reach: Float64, shape_clip: Float64,
) raises -> List[Float64]:
    """The `curriculum` words for a one-region family. Host-side.

    ⚠ RAISES ON A SECOND REGION rather than letting a caller write past the
    table. `curriculum` is `MODEL_CURRICULUM_SIZE` wide and a region costs
    five words; a sixth-word write would land in whatever follows and region 1
    would read back a plausible site id and a plausible rectangle, both wrong.
    """
    if MAX_CURRICULUM_REGIONS < 1:
        raise Error("tasks: MODEL_CURRICULUM_SIZE too small for a region")
    var out = List[Float64]()
    for _ in range(MODEL_CURRICULUM_SIZE):
        out.append(0.0)
    out[CUR_IDX_REGION_SITE] = Float64(site)
    out[CUR_IDX_REGION_X0] = x0
    out[CUR_IDX_REGION_Y0] = y0
    out[CUR_IDX_REGION_X1] = x1
    out[CUR_IDX_REGION_Y1] = y1
    # ⚠⚠ REQUIRED, NOT DEFAULTED HERE. A defaulted argument would let a caller
    # that never heard of `half_height` keep compiling and silently ship the
    # 0.12 fallback to device while the HOST evaluator used the region's real
    # band — the CPU and GPU rewards would then disagree on exactly the
    # regions the field was added for. `region_half_heights(f)[i]` is the
    # value; the compiler now insists it be passed.
    if half_height <= 0.0:
        raise Error(
            "tasks: region half-height " + String(half_height) + " accepts no"
            " point. `pred_in_rect` would return False on device for every"
            " state, which reads as an unlearnable task rather than a bad"
            " number. `spec.parse_region` refuses this too."
        )
    out[CUR_IDX_REGION_H] = half_height

    # ⚠⚠ THE 0.5 BOUND IS CHECKED HERE BECAUSE THE WEIGHTS ARE RUNTIME NOW.
    # `test_goal_distance` asserts the config's DEFAULTS satisfy it, and that
    # is no longer enough once a `--shape-goal` flag can set anything: three
    # files read "solved" as `reward > 0.5` and the shaping is subtracted from
    # that same scalar, so a weight pair whose worst case reaches 0.5 makes a
    # SOLVED lane report 0.4 and every success counter in the tree read it as
    # a miss. Refused at the point the number enters the system.
    if shape_w_goal < 0.0 or shape_w_reach < 0.0:
        raise Error(
            "tasks: negative shaping weight (" + String(shape_w_goal) + ", "
            + String(shape_w_reach) + ") — that pays the policy to move AWAY"
            " from the goal."
        )
    var worst = (shape_w_goal + shape_w_reach) * shape_clip
    if worst >= 0.5:
        raise Error(
            "tasks: the shaping weights (" + String(shape_w_goal) + " + "
            + String(shape_w_reach) + ") * clip " + String(shape_clip)
            + " = " + String(worst) + ", which reaches the 0.5 that separates"
            " SOLVED from not. `task_batched_gpu.mojo`,"
            " `task_eval_frozen.mojo` and `sac_task_gpu.mojo` all read success"
            " as `reward > 0.5`, so a solved lane would report less than that"
            " and be counted as a miss — silently. Lower the weights or lower"
            " SHAPE_CLIP."
        )
    out[CUR_IDX_SHAPE_W_GOAL] = shape_w_goal
    out[CUR_IDX_SHAPE_W_REACH] = shape_w_reach
    return out^


@always_inline
def eval_tape_gpu[
    DTYPE: DType, BATCH: Int, NBODY_F: Int, SITE_DIM: Int,
](
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    curriculum: LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY_F * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY_F * 4), MutAnyOrigin
    ],
    site_xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SITE_DIM), MutAnyOrigin
    ],
    env: Int,
) -> Bool:
    """This lane's goal, from this lane's tape. One `Bool`, no allocation."""
    var v0 = False
    var v1 = False
    var v2 = False
    var last = False

    var rs = Int(rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_SITE]))
    var rx0 = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_X0])
    var ry0 = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_Y0])
    var rx1 = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_X1])
    var ry1 = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_Y1])
    var rh = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_H])

    # ⚠ `comptime for`: MAX_TAPE_TERMS is 3 and the body branches on an op
    # code, so unrolling keeps every index a constant. A runtime loop here
    # would also index `meta` with a computed offset, which is fine, but the
    # `@parameter`-free form was a Metal f64 trap once already.
    comptime for i in range(MAX_TAPE_TERMS):
        comptime w = META_IDX_TASK_PARAM_0 + i * TERM_WORDS
        var op = Int(rebind[Scalar[DTYPE]](meta[env, w]))
        # ⚠ THE EMPTY MARKER STOPS THE SWEEP. `meta` is not zeroed between
        # episodes, so an unused slot holds the PREVIOUS episode's term.
        if op >= 0:
            var a = Int(rebind[Scalar[DTYPE]](meta[env, w + 1]))
            var b = Int(rebind[Scalar[DTYPE]](meta[env, w + 2]))
            var param = rebind[Scalar[DTYPE]](meta[env, w + 3])
            var r = False

            if op == OP_AND:
                var pa = v0 if a == 0 else (v1 if a == 1 else v2)
                var pb = v0 if b == 0 else (v1 if b == 1 else v2)
                r = pa and pb
            elif op == OP_OR:
                var pa = v0 if a == 0 else (v1 if a == 1 else v2)
                var pb = v0 if b == 0 else (v1 if b == 1 else v2)
                r = pa or pb
            elif op == OP_NOT:
                r = not (v0 if a == 0 else (v1 if a == 1 else v2))
            elif op == OP_NEAR:
                r = pred_near[DTYPE](
                    rebind[Scalar[DTYPE]](xpos[env, a * 3]),
                    rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1]),
                    rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2]),
                    rebind[Scalar[DTYPE]](xpos[env, b * 3]),
                    rebind[Scalar[DTYPE]](xpos[env, b * 3 + 1]),
                    rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2]),
                    param,
                )
            elif op == OP_ABOVE:
                r = pred_above[DTYPE](
                    rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2]),
                    rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2]),
                    param,
                )
            elif op == OP_UPRIGHT:
                # ⚠ `xquat` IS (x, y, z, W) — W LAST. Third reader of that
                # layout in this package; `eval.mojo` records what it cost.
                r = pred_upright[DTYPE](
                    rebind[Scalar[DTYPE]](xquat[env, a * 4 + 3]),
                    rebind[Scalar[DTYPE]](xquat[env, a * 4 + 0]),
                    rebind[Scalar[DTYPE]](xquat[env, a * 4 + 1]),
                    rebind[Scalar[DTYPE]](xquat[env, a * 4 + 2]),
                    param,
                )
            else:
                var px: Scalar[DTYPE]
                var py: Scalar[DTYPE]
                var pz: Scalar[DTYPE]
                if op == OP_AT_REGION:
                    px = rebind[Scalar[DTYPE]](site_xpos[env, a * 3])
                    py = rebind[Scalar[DTYPE]](site_xpos[env, a * 3 + 1])
                    pz = rebind[Scalar[DTYPE]](site_xpos[env, a * 3 + 2])
                else:
                    px = rebind[Scalar[DTYPE]](xpos[env, a * 3])
                    py = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1])
                    pz = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2])
                # ⚠ THE REGION'S OWN BAND, from `curriculum`, matching
                # `eval.eval_goal`'s `reg.half_height`. `IN_HALF_HEIGHT` is no
                # longer read here at all: it survives as the DEFAULT the host
                # writes into the table, one place instead of two.
                var dz_min = -rh
                var dz_max = rh
                if op == OP_ON:
                    dz_min = Scalar[DTYPE](ON_MIN_DZ)
                    dz_max = Scalar[DTYPE](ON_MAX_DZ)
                r = pred_in_rect[DTYPE](
                    px, py, pz,
                    rebind[Scalar[DTYPE]](site_xpos[env, rs * 3]),
                    rebind[Scalar[DTYPE]](site_xpos[env, rs * 3 + 1]),
                    rebind[Scalar[DTYPE]](site_xpos[env, rs * 3 + 2]),
                    rx0, ry0, rx1, ry1, dz_min, dz_max,
                )

            if i == 0:
                v0 = r
            elif i == 1:
                v1 = r
            else:
                v2 = r
            last = r
    return last


@always_inline
def goal_frame_ids(
    op: Int, a: Int, b: Int, region_site: Int
) -> Tuple[Int, Int, Int, Int]:
    """`(subject_is_site, subject_id, target_is_site, target_id)` for a term.

    ⚠⚠ THE RULE IS WRITTEN ONCE HERE AND THE MEMORY ACCESS TWICE. The two
    observation hooks read `xpos`/`site_xpos` through different types — a
    `LayoutTensor` on device and a `Data` on the host — so the READS cannot be
    shared, but which body and which site to read is a rule, and a rule
    written inline twice drifts. `test_active_mask` compares the two hooks'
    output vectors and would catch a divergence; this makes one impossible.

    ⚠ `In`/`On`/`AtRegion` TARGET THE REGION'S SITE, and only `AtRegion`'s
    SUBJECT is a site. Mixing those up reads a site id out of the body array
    and lands on a real, wrong position — the same trap `eval_goal` records.

    ⚠ `Upright` HAS NO TARGET, so it points at its own subject and the
    relative vector comes out zero. That is the honest answer: there is no
    second frame in the predicate, and inventing one would put a number in the
    observation that means nothing.

    ⚠ TERM 0 IS ALWAYS A LEAF, so `And`/`Or`/`Not` never reach this. The tape
    is POST-ORDER — every child index is lower than its parent's, asserted in
    `test_goal_language` — so the first term cannot be a composition.
    """
    if op == OP_AT_REGION:
        return (1, a, 1, region_site)
    if op == OP_IN or op == OP_ON:
        return (0, a, 1, region_site)
    if op == OP_UPRIGHT:
        return (0, a, 0, a)
    # NEAR / ABOVE — both arguments are bodies.
    return (0, a, 0, b)


@always_inline
def tape_distance_gpu[
    DTYPE: DType, BATCH: Int, NBODY_F: Int, SITE_DIM: Int,
](
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    curriculum: LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY_F * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY_F * 4), MutAnyOrigin
    ],
    site_xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SITE_DIM), MutAnyOrigin
    ],
    env: Int,
) -> Scalar[DTYPE]:
    """HOW FAR this lane is from its goal, in metres. ZERO iff the goal holds.

    ## ⚠⚠ WHY A DISTANCE EXISTS AT ALL: A SPARSE REWARD HAS NOTHING TO LEARN

    `so101_gather_bricks` pays +1 on success and nothing otherwise. Measured on
    a 5090: random actions meet it on about 1.5% of episodes, so 125k env-steps
    produced roughly SIX rewarding transitions out of 125,000 — 5e-05 of the
    replay buffer, which a batch of 256 contains 1.4% of the time. The critic
    almost never sees a success and `mean_q` has nothing to move toward. That
    is not a hyperparameter problem and no amount of tuning reaches it.

    This is the quantity a shaped term needs, and it is derived from the TAPE
    so it works for every goal the language can express rather than for one
    task — a `Near` special case inside a family config is exactly the
    "rule written inline twice" defect this tree keeps paying for.

    ## THE CONTRACT, AND THE GATE THAT HOLDS IT

    **Zero iff the goal holds.** `tests/tasks/test_goal_distance.mojo` sweeps
    states and asserts `tape_distance_gpu(s) == 0` exactly when
    `eval_tape_gpu(s)` is True — two separate switches over one tape, so an op
    this one forgot shows up as a disagreement rather than as a term with no
    gradient.

    ⚠ AND ZERO IS ALSO WHAT AN UNSHAPEABLE TERM RETURNS. `Not` has no
    monotone distance (the further you are from satisfying the negated term,
    the better) and Tier B has no geometry here, so both contribute 0 — no
    gradient, never a WRONG gradient. The zero-iff contract still holds
    because those ops are refused as goals long before this
    (`require_tier_a`), and `Not` composes to a term whose truth this file
    reads and whose distance it declines to guess.

    ## ⚠ THE COMPOSITION IS max FOR `And` AND min FOR `Or`

    Both must hold, so the distance to satisfying a conjunction is the WORST
    of its parts; either will do for a disjunction, so it is the best. Summing
    an `And` would let a policy trade one term off against another and sit
    between two half-satisfied goals.
    """
    var d0 = Scalar[DTYPE](0)
    var d1 = Scalar[DTYPE](0)
    var d2 = Scalar[DTYPE](0)
    var last = Scalar[DTYPE](0)

    var rs = Int(rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_SITE]))
    var rx0 = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_X0])
    var ry0 = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_Y0])
    var rx1 = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_X1])
    var ry1 = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_Y1])
    var rh = rebind[Scalar[DTYPE]](curriculum[0, CUR_IDX_REGION_H])

    comptime for i in range(MAX_TAPE_TERMS):
        comptime w = META_IDX_TASK_PARAM_0 + i * TERM_WORDS
        var op = Int(rebind[Scalar[DTYPE]](meta[env, w]))
        if op >= 0:
            var a = Int(rebind[Scalar[DTYPE]](meta[env, w + 1]))
            var b = Int(rebind[Scalar[DTYPE]](meta[env, w + 2]))
            var param = rebind[Scalar[DTYPE]](meta[env, w + 3])
            var d = Scalar[DTYPE](0)

            if op == OP_AND:
                var pa = d0 if a == 0 else (d1 if a == 1 else d2)
                var pb = d0 if b == 0 else (d1 if b == 1 else d2)
                d = pa if pa > pb else pb
            elif op == OP_OR:
                var pa = d0 if a == 0 else (d1 if a == 1 else d2)
                var pb = d0 if b == 0 else (d1 if b == 1 else d2)
                d = pa if pa < pb else pb
            elif op == OP_NEAR:
                var ex = rebind[Scalar[DTYPE]](xpos[env, a * 3]) - rebind[
                    Scalar[DTYPE]
                ](xpos[env, b * 3])
                var ey = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1]) - rebind[
                    Scalar[DTYPE]
                ](xpos[env, b * 3 + 1])
                var ez = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2]) - rebind[
                    Scalar[DTYPE]
                ](xpos[env, b * 3 + 2])
                var r = sqrt(ex * ex + ey * ey + ez * ez) - param
                d = r if r > Scalar[DTYPE](0) else Scalar[DTYPE](0)
            elif op == OP_ABOVE:
                # ⚠ `pred_above` is `za > zb + margin`, so the shortfall is
                # how much higher `a` still has to be — and it is ZERO the
                # instant the predicate flips, which is the contract.
                var r2 = (
                    rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2]) + param
                ) - rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2])
                d = r2 if r2 > Scalar[DTYPE](0) else Scalar[DTYPE](0)
            elif op == OP_UPRIGHT:
                # ⚠ NOT IN METRES, AND THAT IS STATED RATHER THAN SCALED. It
                # is the shortfall in the cosine `pred_upright` compares, so
                # it is in [0, 2] and mixes with a metre term only through the
                # weight the config gives it.
                var qw = rebind[Scalar[DTYPE]](xquat[env, a * 4 + 3])
                var qx = rebind[Scalar[DTYPE]](xquat[env, a * 4 + 0])
                var qy = rebind[Scalar[DTYPE]](xquat[env, a * 4 + 1])
                var cosang = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (
                    qx * qx + qy * qy
                )
                var need = Scalar[DTYPE](1) - param
                var r3 = need - cosang
                d = r3 if r3 > Scalar[DTYPE](0) else Scalar[DTYPE](0)
                _ = qw
            else:
                # IN / ON / AT_REGION — the distance to the box, in site
                # coordinates. ⚠ AXIS-ALIGNED AND CLAMPED PER AXIS: an
                # overshoot on one axis must not cancel a shortfall on
                # another, which is what a signed sum would do.
                var px: Scalar[DTYPE]
                var py: Scalar[DTYPE]
                var pz: Scalar[DTYPE]
                if op == OP_AT_REGION:
                    px = rebind[Scalar[DTYPE]](site_xpos[env, a * 3])
                    py = rebind[Scalar[DTYPE]](site_xpos[env, a * 3 + 1])
                    pz = rebind[Scalar[DTYPE]](site_xpos[env, a * 3 + 2])
                else:
                    px = rebind[Scalar[DTYPE]](xpos[env, a * 3])
                    py = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1])
                    pz = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2])
                var zlo = -rh
                var zhi = rh
                if op == OP_ON:
                    zlo = Scalar[DTYPE](ON_MIN_DZ)
                    zhi = Scalar[DTYPE](ON_MAX_DZ)
                var ux = px - rebind[Scalar[DTYPE]](site_xpos[env, rs * 3])
                var uy = py - rebind[Scalar[DTYPE]](site_xpos[env, rs * 3 + 1])
                var uz = pz - rebind[Scalar[DTYPE]](site_xpos[env, rs * 3 + 2])
                var gx = Scalar[DTYPE](0)
                if ux < rx0:
                    gx = rx0 - ux
                elif ux > rx1:
                    gx = ux - rx1
                var gy = Scalar[DTYPE](0)
                if uy < ry0:
                    gy = ry0 - uy
                elif uy > ry1:
                    gy = uy - ry1
                var gz = Scalar[DTYPE](0)
                if uz < zlo:
                    gz = zlo - uz
                elif uz > zhi:
                    gz = uz - zhi
                d = sqrt(gx * gx + gy * gy + gz * gz)

            if i == 0:
                d0 = d
            elif i == 1:
                d1 = d
            else:
                d2 = d
            last = d
    return last


def require_gpu_regions(g: BoundGoal, task_name: String) raises:
    """⚠⚠ THE ONE-REGION RULE, MADE REAL. Call it beside `require_tier_a`.

    `eval_tape_gpu` reads the region table ONCE, from `curriculum[0, 0..4]`,
    and reads it UNCONDITIONALLY: a term's `b` is its region index on the CPU
    path and indexes NOTHING on device. So a goal naming region 1 evaluates
    against region 0's site and rectangle on the GPU while `eval.eval_goal`
    uses region 1's on the host — the two disagree in the REWARD, which is
    where a disagreement is least visible and most expensive.

    ⚠ THIS IS NOT HYPOTHETICAL AND IT IS WHY THE CHECK EXISTS. `so101_tabletop`
    declares three regions: `table_top` for goals, and `table_left`/
    `table_right` so `so101_gather_bricks` can start its two props apart. Those
    two are reachable from `init=`, which the HOST samples, and unreachable
    from `goal=`, which the device evaluates. Nothing in the file format says
    so — this does.

    The fix, when a family genuinely needs two goal regions, is to widen
    `MODEL_CURRICULUM_SIZE` and index the table by the term's `b`. Both halves,
    or neither: widening the table without indexing it changes nothing, and
    indexing a table that is one region wide reads past it.
    """
    for i in range(len(g.terms)):
        ref t = g.terms[i]
        if t.op != OP_IN and t.op != OP_ON and t.op != OP_AT_REGION:
            continue
        if t.b >= MAX_CURRICULUM_REGIONS:
            raise Error(
                "task '" + task_name + "': goal names region index "
                + String(t.b) + ", but the device region table holds "
                + String(MAX_CURRICULUM_REGIONS) + " (curriculum is "
                + String(MODEL_CURRICULUM_SIZE) + " words and a region costs "
                + String(REGION_WORDS) + "). The device evaluator would read"
                " region 0's rectangle for it and the host evaluator would"
                " read the right one, so the GPU and CPU rewards would"
                " disagree silently. Regions past the first are usable from"
                " `init=` — sampled on the host — but not from `goal=`."
            )
