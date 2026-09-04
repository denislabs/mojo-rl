"""The goal as TWELVE FLOATS — what a reward kernel reads. P3a.

    var tape = encode_goal(bound_goal)      # host, once per episode
    var holds = eval_tape(tape, 0, ...)     # device, every step

## ⚠⚠ IT FITS IN `meta` — NO NEW KERNEL OPERANDS

`TASK_LAYER_IMPLEMENTATION.md` Gap E scoped P3 as "two new per-lane operands
against the measured Metal cliff of 28". That is not needed. `Data.meta` is
`[BATCH, METADATA_SIZE]` — already **per-lane**, already an operand of
`compute_reward_and_done_gpu` — and `gpu/constants.mojo:168` names
`META_IDX_TASK_PARAM_0..11` as "the home of per-episode task state that is not
a model field", with precedent (the brick tasks' `desired_order`).

Twelve slots at four words a term is **three terms**, which is what the three
shipped tasks need: `reach` 1, `lift` 1, `gather` 3 (`And` + two `On`).

⚠ AND RESET PRESERVES THEM. `_reset_env_lane` writes `META_IDX_STEP_COUNT` and
leaves the rest, so a tape written once per episode survives into every step of
it. That is stated at `constants.mojo:164` and is the reason this works without
a second upload per step.

⚠ THREE TERMS IS A HARD CEILING AND `encode_goal` REFUSES A FOURTH rather than
truncating. A truncated goal still evaluates — to something simpler than what
the author wrote — and would train a policy on a task nobody specified.

## ⚠ THE PREDICATES ARE THE ONES `eval.mojo` ALREADY GATES

`eval_tape` calls `pred_in_rect` / `pred_near` / `pred_above` / `pred_upright`
unchanged. It is a different LOOP over a different CONTAINER, not different
arithmetic — which is the whole reason those were written as scalar `def`s.
`test_task_tape.mojo` asserts the two agree on the same states, so the device
leg cannot drift from the host one.

## ⚠ EVERYTHING HERE IS SCALARS AND FLAT ARRAYS

No `FamilySpec`, no `List[BoundTerm]`, no string. The region table arrives as
five parallel flat arrays because that is what a kernel can be handed. A
version that took the spec would have to be rewritten for the device, and then
there would be two.
"""

from .predicates import (
    BoundGoal,
    OP_IN, OP_ON, OP_NEAR, OP_ABOVE, OP_UPRIGHT, OP_AT_REGION,
    OP_AND, OP_OR, OP_NOT, op_name, op_is_tier_a,
)
from .eval import (
    pred_in_rect, pred_near, pred_above, pred_upright,
    ON_MIN_DZ, ON_MAX_DZ,
)


comptime TERM_WORDS: Int = 4
comptime MAX_TAPE_TERMS: Int = 3
comptime TAPE_WORDS: Int = TERM_WORDS * MAX_TAPE_TERMS   # 12 == the meta slots

# ⚠ THE EMPTY MARKER. `meta` is NOT zeroed between episodes beyond
# `META_IDX_STEP_COUNT`, so a lane that ran a 3-term goal and then a 1-term one
# would still hold the old terms 1 and 2. `encode_goal` writes OP_NONE into
# every unused slot for that reason, and `eval_tape` stops at the first one.
comptime OP_NONE: Float64 = -1.0


def encode_goal(g: BoundGoal) raises -> List[Float64]:
    """A bound goal as `TAPE_WORDS` floats. Host-side, once per episode."""
    if len(g.terms) > MAX_TAPE_TERMS:
        raise Error(
            "tasks: goal has " + String(len(g.terms)) + " terms; the device"
            " tape holds " + String(MAX_TAPE_TERMS) + " (twelve `meta` words"
            " at four a term). Refused rather than truncated — a truncated"
            " goal evaluates to something simpler than the author wrote."
        )
    for i in range(len(g.terms)):
        if not op_is_tier_a(g.terms[i].op):
            raise Error(
                "tasks: " + op_name(g.terms[i].op) + " is TIER B and cannot go"
                " on the device tape — it reads contacts, which the reward"
                " kernel does not carry. See TASK_LAYER_PLAN.md §5.1."
            )
    var out = List[Float64]()
    for i in range(MAX_TAPE_TERMS):
        if i < len(g.terms):
            out.append(Float64(g.terms[i].op))
            out.append(Float64(g.terms[i].a))
            out.append(Float64(g.terms[i].b))
            out.append(g.terms[i].param)
        else:
            out.append(OP_NONE)
            out.append(0.0)
            out.append(0.0)
            out.append(0.0)
    return out^


@always_inline
def eval_tape(
    tape: List[Float64], base: Int,
    xpos: List[Float64], xquat: List[Float64], site_xpos: List[Float64],
    reg_site: List[Int],
    reg_xmin: List[Float64], reg_ymin: List[Float64],
    reg_xmax: List[Float64], reg_ymax: List[Float64],
    reg_h: List[Float64],
) -> Bool:
    """Evaluate the tape at `tape[base : base + TAPE_WORDS]`.

    ⚠ A FORWARD SWEEP WITH NO STACK, which is legal only because the terms are
    POST-ORDER — every child index is lower than its parent's, asserted in
    `test_goal_language`. A parent reading a later term would read a slot this
    loop has not written, and on device that is uninitialised memory.

    ⚠ `reg_h` IS PER REGION AND REQUIRED. It used to be `eval.IN_HALF_HEIGHT`,
    one constant for every region in every family — see `spec.RegionSpec` for
    what that cost `so101_reach_brick`.

    ⚠ `base` EXISTS SO A KERNEL CAN PASS `env * METADATA_SIZE +
    META_IDX_TASK_PARAM_0` and read its own lane's tape out of the shared
    `meta` tensor. The host passes 0.
    """
    var v0 = False
    var v1 = False
    var v2 = False
    var last = False

    for i in range(MAX_TAPE_TERMS):
        var w = base + i * TERM_WORDS
        var op = Int(tape[w])
        if op < 0:
            break
        var a = Int(tape[w + 1])
        var b = Int(tape[w + 2])
        var param = tape[w + 3]
        # ⚠ NO INITIALISER: every branch assigns, and the last is a catch-all.
        # A `= False` default would make an op this switch forgot evaluate as
        # "goal not met" instead of being impossible to miss.
        var r: Bool

        if op == OP_AND:
            r = (v0 if a == 0 else (v1 if a == 1 else v2)) and (
                v0 if b == 0 else (v1 if b == 1 else v2)
            )
        elif op == OP_OR:
            r = (v0 if a == 0 else (v1 if a == 1 else v2)) or (
                v0 if b == 0 else (v1 if b == 1 else v2)
            )
        elif op == OP_NOT:
            r = not (v0 if a == 0 else (v1 if a == 1 else v2))
        elif op == OP_NEAR:
            r = pred_near(
                xpos[a * 3], xpos[a * 3 + 1], xpos[a * 3 + 2],
                xpos[b * 3], xpos[b * 3 + 1], xpos[b * 3 + 2], param,
            )
        elif op == OP_ABOVE:
            r = pred_above(xpos[a * 3 + 2], xpos[b * 3 + 2], param)
        elif op == OP_UPRIGHT:
            # ⚠ `xquat` IS (x, y, z, W) — W LAST. The same trap `eval.mojo`
            # records; two readers of one layout, and this is the second.
            r = pred_upright(
                xquat[a * 4 + 3], xquat[a * 4 + 0],
                xquat[a * 4 + 1], xquat[a * 4 + 2], param,
            )
        else:
            # IN / ON / AT_REGION
            var s = reg_site[b]
            var px: Float64
            var py: Float64
            var pz: Float64
            if op == OP_AT_REGION:
                px = site_xpos[a * 3]
                py = site_xpos[a * 3 + 1]
                pz = site_xpos[a * 3 + 2]
            else:
                px = xpos[a * 3]
                py = xpos[a * 3 + 1]
                pz = xpos[a * 3 + 2]
            # ⚠ THE REGION'S OWN BAND — `eval.region_half_heights(f)[b]`,
            # matching `eval.eval_goal` and `gpu_eval.eval_tape_gpu`. Three
            # readers of one quantity now, which is why it is a per-region
            # ARRAY here and not the constant it used to be.
            var dz_min = -reg_h[b]
            var dz_max = reg_h[b]
            if op == OP_ON:
                dz_min = ON_MIN_DZ
                dz_max = ON_MAX_DZ
            r = pred_in_rect(
                px, py, pz,
                site_xpos[s * 3], site_xpos[s * 3 + 1], site_xpos[s * 3 + 2],
                reg_xmin[b], reg_ymin[b], reg_xmax[b], reg_ymax[b],
                dz_min, dz_max,
            )

        if i == 0:
            v0 = r
        elif i == 1:
            v1 = r
        else:
            v2 = r
        last = r
    return last
