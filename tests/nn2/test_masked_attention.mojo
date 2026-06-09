"""MaskedAttention CPU — PHASE 0 SPIKE parity gate (docs/DREAMER4_PORT_PLAN.md).

Three checks, the decisive go/no-go signal for the Dreamer 4 port:

  1. FINITE-DIFF GRADCHECK with a real modality mask — proves the masked
     backward is correct (the §4.3 claim: masked weights = 0 ⇒ no backward
     special-casing needed).
  2. ALL-ALLOW ≡ non-causal ScaledDotProductAttention (forward + vjp,
     bit-close) — the default mask reduces to plain attention.
  3. CAUSAL mask ≡ causal ScaledDotProductAttention (forward + vjp,
     bit-close) — MaskedAttention SUBSUMES the causal op.
"""

from std.memory import alloc
from std.math import abs, sqrt
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.attention import ScaledDotProductAttention
from mojo_rl.nn2.primitives.masked_attention import (
    MaskedAttention,
    causal_mask,
    build_modality_mask,
)


comptime EPS: Float64 = 2e-3
comptime TOL: Float64 = 1.5e-2
comptime PARITY_TOL: Float64 = 1e-5


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def _loss(
    y: UnsafePointer[Scalar[DT], MutAnyOrigin],
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += Float64(y[i]) * Float64(go[i])
    return s


def test_gradcheck_modality_mask() raises:
    """FD gradcheck with a 2-modality encoder mask (latents + one modality)."""
    print("test_gradcheck_modality_mask ...")
    comptime DIM = 4
    comptime N_HEADS = 2
    comptime SEQ = 5
    comptime BATCH = 2
    comptime IN_N = BATCH * SEQ * DIM * 3
    comptime OUT_N = BATCH * SEQ * DIM

    var op = MaskedAttention[DIM, N_HEADS, SEQ].make[target="cpu", INIT=Zero]()
    # 2 latent tokens (id 0), then 3 image tokens (id 1): encoder mode.
    var mods = [0, 0, 1, 1, 1]
    op.set_mask(build_modality_mask["encoder"](mods, n_latents=2))

    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    for i in range(IN_N):
        x[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        go[i] = _spread(i, 4.1)

    var x_t = TileTensor(x, row_major[BATCH, SEQ * DIM * 3]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ * DIM]())
    op.forward["cpu", BATCH](x_t, output=y_t)
    var go_t = TileTensor(go, row_major[BATCH, SEQ * DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, SEQ * DIM * 3]())
    op.vjp["cpu", BATCH](go_t, gi_t)

    var max_err: Float64 = 0.0
    for k in range(IN_N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _loss(y, go, OUT_N)
        x[k] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _loss(y, go, OUT_N)
        x[k] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(gi[k]) - fd)
        if d > max_err:
            max_err = d
    print("   max|analytic - FD| =", max_err)
    assert_true(max_err < TOL, "modality-mask grad_input vs FD")
    print("  ok")


def _parity[
    DIM: Int, N_HEADS: Int, SEQ: Int, CAUSAL: Bool
](name: String, use_causal_mask: Bool) raises:
    """Run MaskedAttention vs ScaledDotProductAttention on identical input;
    compare forward outputs and grad_inputs."""
    print(name, "...")
    comptime BATCH = 2
    comptime IN_N = BATCH * SEQ * DIM * 3
    comptime OUT_N = BATCH * SEQ * DIM

    var sdpa = ScaledDotProductAttention[DIM, N_HEADS, SEQ, CAUSAL].make[
        target="cpu", INIT=Zero
    ]()
    var op = MaskedAttention[DIM, N_HEADS, SEQ].make[target="cpu", INIT=Zero]()
    if use_causal_mask:
        op.set_mask(causal_mask(SEQ))  # else: keep default all-allow

    var x = _alloc(IN_N)
    var go = _alloc(OUT_N)
    for i in range(IN_N):
        x[i] = _spread(i, 2.7)
    for i in range(OUT_N):
        go[i] = _spread(i, 0.4)

    var yr = _alloc(OUT_N)
    var ym = _alloc(OUT_N)
    var gir = _alloc(IN_N)
    var gim = _alloc(IN_N)

    var x_t = TileTensor(x, row_major[BATCH, SEQ * DIM * 3]())
    var go_t = TileTensor(go, row_major[BATCH, SEQ * DIM]())

    var yr_t = TileTensor(yr, row_major[BATCH, SEQ * DIM]())
    var ym_t = TileTensor(ym, row_major[BATCH, SEQ * DIM]())
    sdpa.forward["cpu", BATCH](x_t, output=yr_t)
    op.forward["cpu", BATCH](x_t, output=ym_t)

    var gir_t = TileTensor(gir, row_major[BATCH, SEQ * DIM * 3]())
    var gim_t = TileTensor(gim, row_major[BATCH, SEQ * DIM * 3]())
    sdpa.vjp["cpu", BATCH](go_t, gir_t)
    op.vjp["cpu", BATCH](go_t, gim_t)

    var max_fwd: Float64 = 0.0
    for i in range(OUT_N):
        var d = abs(Float64(yr[i]) - Float64(ym[i]))
        if d > max_fwd:
            max_fwd = d
    var max_bwd: Float64 = 0.0
    for i in range(IN_N):
        var d = abs(Float64(gir[i]) - Float64(gim[i]))
        if d > max_bwd:
            max_bwd = d
    print("   max fwd diff =", max_fwd, "  max bwd diff =", max_bwd)
    assert_true(max_fwd < PARITY_TOL, name + ": forward parity")
    assert_true(max_bwd < PARITY_TOL, name + ": backward parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("MaskedAttention CPU — Dreamer 4 Phase 0 spike")
    print("=" * 70)
    test_gradcheck_modality_mask()
    _parity[4, 2, 3, False]("test_allallow_eq_noncausal_sdpa", False)
    _parity[6, 1, 4, False]("test_allallow_eq_noncausal_singlehead", False)
    _parity[4, 2, 3, True]("test_causalmask_eq_causal_sdpa", True)
    _parity[6, 2, 5, True]("test_causalmask_eq_causal_sdpa_seq5", True)
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
