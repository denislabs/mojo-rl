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

⚠ ONE REGION. `MODEL_CURRICULUM_SIZE` is 8 and a region costs 5 words
(site id + rect), so the family may declare one until that table needs more
room. `region_table_words` asserts it rather than silently reading garbage for
region 1 — which would resolve to a real site and a real rectangle, both wrong.
"""

from layout import Layout, LayoutTensor

from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, MODEL_CURRICULUM_SIZE,
)
from .predicates import (
    OP_IN, OP_ON, OP_NEAR, OP_ABOVE, OP_UPRIGHT, OP_AT_REGION,
    OP_AND, OP_OR, OP_NOT,
)
from .eval import (
    pred_in_rect, pred_near, pred_above, pred_upright,
    IN_HALF_HEIGHT, ON_MIN_DZ, ON_MAX_DZ,
)
from .tape import MAX_TAPE_TERMS, TERM_WORDS


# ── the region table's layout inside `curriculum` ──────────────────────────
comptime CUR_IDX_REGION_SITE: Int = 0
comptime CUR_IDX_REGION_X0: Int = 1
comptime CUR_IDX_REGION_Y0: Int = 2
comptime CUR_IDX_REGION_X1: Int = 3
comptime CUR_IDX_REGION_Y1: Int = 4
comptime REGION_WORDS: Int = 5
comptime MAX_CURRICULUM_REGIONS: Int = MODEL_CURRICULUM_SIZE // REGION_WORDS


def region_table_words(
    site: Int, x0: Float64, y0: Float64, x1: Float64, y1: Float64
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
                var dz_min = Scalar[DTYPE](-IN_HALF_HEIGHT)
                var dz_max = Scalar[DTYPE](IN_HALF_HEIGHT)
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
