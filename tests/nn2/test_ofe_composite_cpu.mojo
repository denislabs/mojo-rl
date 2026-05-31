"""O.1 — OFE composite (CPU) smoke.

Gates four things at small dims (OBS=3, ACT=1, per_unit=2, num_layers=6):

  (1) SkipConcat[Sequential[Linear, LayerNorm, SiLU]] — verify
      `OUT_DIM == IN + per_unit` and that the first IN columns of the
      forward output equal the input verbatim (the skip path is the
      definition of the layer).
  (2) OFEStateBranch6 — `OUT_DIM == OBS + 6*per_unit`. Forward runs,
      output is finite, backward runs in both `mode="all"` and
      `mode="input_only"` and produces a finite grad_input.
  (3) OFEActionBranch6 — same checks with SA_IN = OBS+6*per_unit + ACT.
  (4) OFEPredictorHead — final Linear of correct shape.

Bit-identity vs the legacy `mojo_rl/nn/composites_ofenet.mojo` is *not*
a goal — legacy and nn2 use different Linear / LayerNorm initialisers.
What we gate here is the *contract*: layer shape, skip-path
correctness, and gradient propagation on both modes."""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.combinators.skip_concat import SkipConcat
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.layer_norm import LayerNorm
from mojo_rl.nn2.primitives.silu import SiLU
from mojo_rl.nn2.initializer import Xavier

from mojo_rl.deep_agents2.redq_ofe.ofe_nets import (
    OFEDenseBlock,
    OFEStateBranch6,
    OFEActionBranch6,
    OFEPredictorHead,
    state_branch_out_dim,
    action_branch_out_dim,
)


comptime BATCH = 4
comptime OBS = 3
comptime ACT = 1
comptime PER_UNIT = 2
comptime N_BLOCKS = 6

# Dense block at the *first* block's width: IN = OBS, OUT = OBS + PER_UNIT.
comptime BLOCK_IN = OBS
comptime BLOCK_OUT = OBS + PER_UNIT

comptime PHI_S_DIM = OBS + N_BLOCKS * PER_UNIT          # 3 + 12 = 15
comptime SA_IN = PHI_S_DIM + ACT                        # 16
comptime PHI_SA_DIM = SA_IN + N_BLOCKS * PER_UNIT       # 28


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _is_finite(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Bool:
    for i in range(n):
        # NaN trip: NaN != NaN. Inf trip: |x| > 1e30 is a strong proxy
        # for the small-dim scales here (sane outputs sit < 100).
        if p[i] != p[i]:
            return False
        if _abs(p[i]) > Scalar[DT](1e30):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────
# (1) SkipConcat[Sequential[Linear, LayerNorm, SiLU]] — primitive
# ─────────────────────────────────────────────────────────────────────────


def test_skip_concat_identity_on_skip() raises:
    """The first IN columns of forward output MUST equal input — that
    IS the definition of the layer."""
    print("--- (1) SkipConcat skip-path identity ---")
    seed(42)

    var block = OFEDenseBlock[BLOCK_IN, PER_UNIT].make[
        target="cpu", INIT=Xavier,
    ]()

    var N_X = BATCH * BLOCK_IN
    var N_Y = BATCH * BLOCK_OUT
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    for i in range(N_X):
        x[i] = Scalar[DT](-0.5 + 0.13 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, BLOCK_IN]())
    var y_t = TileTensor(y, row_major[BATCH, BLOCK_OUT]())
    block.forward["cpu", BATCH](x_t, output=y_t)

    # Verify y[:, 0:IN] == x[:, 0:IN] (the skip path).
    var max_skip_diff: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(BLOCK_IN):
            var diff = _abs(y[b * BLOCK_OUT + d] - x[b * BLOCK_IN + d])
            if diff > max_skip_diff:
                max_skip_diff = diff
    print("  max |y[:, 0:IN] - x|:", max_skip_diff)
    assert_true(
        max_skip_diff == Scalar[DT](0),
        "SkipConcat must copy input bit-identically into first IN columns",
    )

    # Verify inner-path output is finite.
    assert_true(
        _is_finite(y + BATCH * BLOCK_IN, BATCH * PER_UNIT),
        "inner path output (SiLU(LayerNorm(Linear(x)))) must be finite",
    )

    # ── Backward (mode="all") with random grad_output ──────────────────
    block.zero_grad["cpu"]()
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    for i in range(N_Y):
        go[i] = Scalar[DT](0.1 - 0.07 * Float64(i))
    for i in range(N_X):
        gi[i] = Scalar[DT](0.0)
    var go_t = TileTensor(go, row_major[BATCH, BLOCK_OUT]())
    var gi_t = TileTensor(gi, row_major[BATCH, BLOCK_IN]())
    block.vjp["cpu", BATCH](go_t, gi_t)
    assert_true(
        _is_finite(gi, N_X), "grad_input must be finite after mode='all'",
    )

    # ── Backward (mode="input_only") — verify it runs cleanly ──────────
    block.zero_grad["cpu"]()
    for i in range(N_X):
        gi[i] = Scalar[DT](0.0)
    block.vjp["cpu", BATCH, mode="input_only"](go_t, gi_t)
    assert_true(
        _is_finite(gi, N_X),
        "grad_input must be finite under mode='input_only'",
    )

    x.free()
    y.free()
    go.free()
    gi.free()

    print("PASS — SkipConcat skip-path identity + both vjp modes.")


# ─────────────────────────────────────────────────────────────────────────
# (2) OFEStateBranch6 — composition of 6 DenseBlocks
# ─────────────────────────────────────────────────────────────────────────


def test_ofe_state_branch6() raises:
    print("--- (2) OFEStateBranch6 forward+vjp ---")
    seed(42)

    var branch = OFEStateBranch6[OBS, PER_UNIT].make[
        target="cpu", INIT=Xavier,
    ]()

    # Comptime shape sanity.
    comptime expected_out = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)
    comptime assert expected_out == PHI_S_DIM, "PHI_S_DIM helper out of sync"

    var N_X = BATCH * OBS
    var N_Y = BATCH * PHI_S_DIM
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    for i in range(N_X):
        x[i] = Scalar[DT](0.2 - 0.05 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, OBS]())
    var y_t = TileTensor(y, row_major[BATCH, PHI_S_DIM]())
    branch.forward["cpu", BATCH](x_t, output=y_t)
    assert_true(_is_finite(y, N_Y), "state branch output must be finite")

    # First OBS columns of branch(x) — after the first block these are
    # `x` itself, but after 6 stacked SkipConcats the leading OBS slot
    # is still `x` (skip path propagates through every block).
    var max_skip_diff: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(OBS):
            var diff = _abs(y[b * PHI_S_DIM + d] - x[b * OBS + d])
            if diff > max_skip_diff:
                max_skip_diff = diff
    print("  max |y[:, 0:OBS] - x|:", max_skip_diff)
    assert_true(
        max_skip_diff == Scalar[DT](0),
        "state branch must preserve original obs in leading OBS columns",
    )

    # ── Backward in both modes ─────────────────────────────────────────
    branch.zero_grad["cpu"]()
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    for i in range(N_Y):
        go[i] = Scalar[DT](0.01 + 0.003 * Float64(i))
    for i in range(N_X):
        gi[i] = Scalar[DT](0.0)
    var go_t = TileTensor(go, row_major[BATCH, PHI_S_DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, OBS]())
    branch.vjp["cpu", BATCH](go_t, gi_t)
    assert_true(_is_finite(gi, N_X), "grad_input must be finite (mode='all')")

    branch.zero_grad["cpu"]()
    for i in range(N_X):
        gi[i] = Scalar[DT](0.0)
    branch.vjp["cpu", BATCH, mode="input_only"](go_t, gi_t)
    assert_true(
        _is_finite(gi, N_X),
        "grad_input must be finite (mode='input_only')",
    )

    x.free()
    y.free()
    go.free()
    gi.free()

    print("PASS — OFEStateBranch6 forward+vjp both modes.")


# ─────────────────────────────────────────────────────────────────────────
# (3) OFEActionBranch6 — takes pre-concatenated (φ(s), a)
# ─────────────────────────────────────────────────────────────────────────


def test_ofe_action_branch6() raises:
    print("--- (3) OFEActionBranch6 forward+vjp ---")
    seed(42)

    comptime expected_out = action_branch_out_dim(
        OBS, ACT, N_BLOCKS, PER_UNIT,
    )
    comptime assert expected_out == PHI_SA_DIM, (
        "PHI_SA_DIM helper out of sync"
    )

    var branch = OFEActionBranch6[SA_IN, PER_UNIT].make[
        target="cpu", INIT=Xavier,
    ]()

    var N_X = BATCH * SA_IN
    var N_Y = BATCH * PHI_SA_DIM
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    for i in range(N_X):
        x[i] = Scalar[DT](0.3 - 0.02 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, SA_IN]())
    var y_t = TileTensor(y, row_major[BATCH, PHI_SA_DIM]())
    branch.forward["cpu", BATCH](x_t, output=y_t)
    assert_true(_is_finite(y, N_Y), "action branch output must be finite")

    # ── Backward ───────────────────────────────────────────────────────
    branch.zero_grad["cpu"]()
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    for i in range(N_Y):
        go[i] = Scalar[DT](0.005 + 0.001 * Float64(i))
    for i in range(N_X):
        gi[i] = Scalar[DT](0.0)
    var go_t = TileTensor(go, row_major[BATCH, PHI_SA_DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, SA_IN]())
    branch.vjp["cpu", BATCH](go_t, gi_t)
    assert_true(_is_finite(gi, N_X), "action branch grad_input finite")

    x.free()
    y.free()
    go.free()
    gi.free()

    print("PASS — OFEActionBranch6 forward+vjp.")


# ─────────────────────────────────────────────────────────────────────────
# (4) OFEPredictorHead — final Linear
# ─────────────────────────────────────────────────────────────────────────


def test_ofe_predictor_head() raises:
    print("--- (4) OFEPredictorHead = Linear[PHI_SA_DIM, OBS] ---")
    seed(42)

    var head = OFEPredictorHead[PHI_SA_DIM, OBS].make[
        target="cpu", INIT=Xavier,
    ]()

    var N_X = BATCH * PHI_SA_DIM
    var N_Y = BATCH * OBS
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    for i in range(N_X):
        x[i] = Scalar[DT](0.1 + 0.001 * Float64(i))
    var x_t = TileTensor(x, row_major[BATCH, PHI_SA_DIM]())
    var y_t = TileTensor(y, row_major[BATCH, OBS]())
    head.forward["cpu", BATCH](x_t, output=y_t)
    assert_true(_is_finite(y, N_Y), "predictor head output finite")

    x.free()
    y.free()

    print("PASS — OFEPredictorHead = Linear sanity.")


def main() raises:
    test_skip_concat_identity_on_skip()
    test_ofe_state_branch6()
    test_ofe_action_branch6()
    test_ofe_predictor_head()
    print("=" * 70)
    print("ALL PASS — O.1 OFE composite (CPU)")
    print("=" * 70)
