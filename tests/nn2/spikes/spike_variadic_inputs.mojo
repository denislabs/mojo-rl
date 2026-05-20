"""Spike DR.2 — variadic `*INPUTS: TileTensor` on a Module-style method?

Three increasingly-realistic attempts. Each is a separate function so
compile failures of one don't block the others.

Attempt A: variadic with **same TileTensor type** per input. This is
    what `Sequential[*MODULES]` already does at the type level but not
    at the value level — we want to confirm value-variadic TileTensor
    works at all.

Attempt B: fixed-arity-3 with **per-input distinct layouts**. The
    practical fallback — most multi-input Modules (Sub, ElemMin, Min,
    SACLoss) need 2–4 inputs, not N. If arity-3 with distinct layouts
    works, we have multi-input Modules without true variadic.

Attempt C: same-type variadic (`*inputs: TileTensor[DT, L, O]`) — the
    homogeneous case. Many ops (Sum, Concat) accept N inputs of the
    same shape; this might compile when the heterogeneous form doesn't.
"""

from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT


# ──────────────────────────────────────────────────────────────────────
# Attempt B — arity-3, distinct layouts. The practical fallback.
# ──────────────────────────────────────────────────────────────────────


def sum3_distinct_layouts[
    BATCH: Int, FEAT: Int,
    L0: TensorLayout, L1: TensorLayout, L2: TensorLayout, LOUT: TensorLayout,
    O0: MutOrigin, O1: MutOrigin, O2: MutOrigin, OOUT: MutOrigin,
](
    in0: TileTensor[DT, L0, O0],
    in1: TileTensor[DT, L1, O1],
    in2: TileTensor[DT, L2, O2],
    mut output: TileTensor[DT, LOUT, OOUT],
):
    """Three inputs of distinct TileTensor types (different layouts allowed).
    All share batch+feat shape at the comptime contract level."""
    comptime assert in0.flat_rank == 2
    comptime assert in1.flat_rank == 2
    comptime assert in2.flat_rank == 2
    comptime assert output.flat_rank == 2
    for b in range(BATCH):
        for d in range(FEAT):
            output[b, d] = in0[b, d] + in1[b, d] + in2[b, d]


# ──────────────────────────────────────────────────────────────────────
# Attempt C — value-level variadic, same TileTensor type per input.
# ──────────────────────────────────────────────────────────────────────
# Verdict: `var *inputs: TileTensor[DT, L, O]` REJECTED. Mojo treats
# each TileTensor's `MutOrigin` as a per-source distinct type — variadic
# value args require *exactly* the same type per element, but `origin_of(a)`
# != `origin_of(b)`. Origins do NOT unify across variadic packs.
# Indexing also fails: `inputs[k][b, d]` can't prove the rank of each
# variadic element's TileTensor.
# Decision: variadic value TileTensor not viable in current Mojo nightly.


# ──────────────────────────────────────────────────────────────────────
# Smoke driver.
# ──────────────────────────────────────────────────────────────────────


def smoke_attempt_b() raises:
    print("--- spike attempt B: arity-3 distinct layouts ---")
    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](1.0))
    var b = List[Scalar[DT]](length=4, fill=Scalar[DT](2.0))
    var c = List[Scalar[DT]](length=4, fill=Scalar[DT](3.0))
    var out = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))
    var ta = TileTensor(a.unsafe_ptr(), row_major[2, 2]())
    var tb = TileTensor(b.unsafe_ptr(), row_major[2, 2]())
    var tc = TileTensor(c.unsafe_ptr(), row_major[2, 2]())
    var to = TileTensor(out.unsafe_ptr(), row_major[2, 2]())
    sum3_distinct_layouts[2, 2](ta, tb, tc, to)
    print("  out=[", out[0], out[1], out[2], out[3], "]  expected 6.0 ×4")
    var ok = out[0] == 6.0 and out[1] == 6.0 and out[2] == 6.0 and out[3] == 6.0
    if ok:
        print("  attempt B: PASSED")
    else:
        print("  attempt B: FAILED")


def main() raises:
    print("=" * 70)
    print("DR.2 — variadic multi-input feasibility spike")
    print("=" * 70)
    smoke_attempt_b()
    print("  attempt C (`*inputs: TileTensor[DT, L, O]`): REJECTED at compile")
    print("    reason: each TileTensor's MutOrigin is per-source distinct;")
    print("    variadic value args require homogeneous types; origins don't unify.")
    print("=" * 70)
