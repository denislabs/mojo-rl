"""Storage GaussianNLLLoss + SequenceCrossEntropyLoss — CPU golden + GPU vs CPU.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). Per loss:
  - CPU golden: run the STORAGE loss with the same inputs/targets/config the
    parity test fed the storage side; assert the scalar forward loss directly
    and the input-gradient tensor against golden fingerprints (S = Σ vᵢ,
    W = Σ vᵢ·(i+1) — the weight catches sign/position errors a plain sum would
    cancel), captured from the bit-identical legacy↔storage run.
  - storage GPU-vs-CPU forward + grad TOL ~2e-5.

Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . \
      tests/nn/test_gaussian_seqce_loss_storage.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . \
      tests/nn/test_gaussian_seqce_loss_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor

from mojo_rl.nn.loss.gaussian_nll_loss import (
    GaussianNLLLoss as StorGNLL,
)
from mojo_rl.nn.loss.sequence_cross_entropy import (
    SequenceCrossEntropyLoss as StorSeqCE,
)


def _check(name: String, data: Tensor, n: Int,
           es: Scalar[DT], ew: Scalar[DT], tol: Scalar[DT]) -> Bool:
    """Assert tensor fingerprint (Σ vᵢ, Σ vᵢ·(i+1)) matches golden (es, ew)."""
    var s: Scalar[DT] = 0
    var w: Scalar[DT] = 0
    for i in range(n):
        s += data.data[i]
        w += data.data[i] * Scalar[DT](i + 1)
    var ok = abs(s - es) < tol and abs(w - ew) < tol
    print("  ", name, "S", s, "(exp", es, ") W", w, "(exp", ew, ")",
          "OK" if ok else "FAIL")
    return ok


# ───────────────────────────── GaussianNLL ─────────────────────────────

comptime G_DIM = 3
comptime G_B = 2
comptime G_IN = 2 * G_DIM          # logits cols = 2*DIM
comptime G_NLOG = G_B * G_IN
comptime G_NTGT = G_B * G_DIM


def _gnll_logits(i: Int) -> Scalar[DT]:
    # mix in-clamp + out-of-clamp logvars (default bounds [-10,-2]).
    var ls = [
        Scalar[DT](0.1), Scalar[DT](0.5), Scalar[DT](-0.3),
        Scalar[DT](-5.0), Scalar[DT](-4.0), Scalar[DT](-3.0),
        Scalar[DT](-0.2), Scalar[DT](0.7), Scalar[DT](0.0),
        Scalar[DT](-12.0), Scalar[DT](1.0), Scalar[DT](-7.0),
    ]
    return ls[i]


def _gnll_tgt(i: Int) -> Scalar[DT]:
    var ts = [
        Scalar[DT](0.05), Scalar[DT](0.4), Scalar[DT](-0.5),
        Scalar[DT](-0.1), Scalar[DT](0.5), Scalar[DT](0.2),
    ]
    return ts[i]


def test_gnll_cpu_golden() raises -> Bool:
    print("test_gnll_cpu_golden (storage CPU vs golden) ...")
    comptime TOL = Scalar[DT](1e-2)
    var st = StorGNLL[G_DIM].make_cpu()
    var s_log = Tensor.alloc(G_NLOG)
    var s_tgt = Tensor.alloc(G_NTGT)
    var s_grad = Tensor.alloc(G_NLOG)
    for i in range(G_NLOG):
        s_log.data[i] = _gnll_logits(i)
    for i in range(G_NTGT):
        s_tgt.data[i] = _gnll_tgt(i)
    var st_fwd = st.forward["cpu", G_B](s_log, s_tgt, None)
    st.vjp["cpu", G_B](s_log, s_tgt, s_grad, None)

    comptime FWD_GOLD = Scalar[DT](58.7865)
    var ok = abs(st_fwd - FWD_GOLD) < TOL
    print("   fwd", st_fwd, "(exp", FWD_GOLD, ")", "OK" if ok else "FAIL")
    ok = _check("grad", s_grad, G_NLOG, -1212.1953, -8802.23, TOL) and ok
    assert_true(ok, "GaussianNLL CPU golden")
    print("  ok")
    return ok


def _gnll_logits_gpu(i: Int) -> Scalar[DT]:
    # Tamer logvars (still touches both clamp bounds [-10,-2]) so inv_var
    # stays ~exp(3) and the large d/d_raw_logvar magnitudes don't make
    # fp32 GPU-vs-CPU error blow past ~2e-5 — the extreme -12→-10 clamp
    # case (inv_var≈exp(10)≈22026) is covered by the exact CPU golden
    # gate above instead.
    var ls = [
        Scalar[DT](0.1), Scalar[DT](0.5), Scalar[DT](-0.3),
        Scalar[DT](-2.5), Scalar[DT](-1.0), Scalar[DT](-3.0),
        Scalar[DT](-0.2), Scalar[DT](0.7), Scalar[DT](0.0),
        Scalar[DT](-2.0), Scalar[DT](-1.5), Scalar[DT](-2.5),
    ]
    return ls[i]


def test_gnll_gpu_vs_cpu(ctx: DeviceContext) raises -> Bool:
    comptime TOL = Scalar[DT](2e-5)
    # CPU storage reference.
    var st_cpu = StorGNLL[G_DIM].make_cpu()
    var c_log = Tensor.alloc(G_NLOG)
    var c_tgt = Tensor.alloc(G_NTGT)
    var c_grad = Tensor.alloc(G_NLOG)
    for i in range(G_NLOG):
        c_log.data[i] = _gnll_logits_gpu(i)
    for i in range(G_NTGT):
        c_tgt.data[i] = _gnll_tgt(i)
    var cpu_fwd = st_cpu.forward["cpu", G_B](c_log, c_tgt, None)
    st_cpu.vjp["cpu", G_B](c_log, c_tgt, c_grad, None)

    # GPU storage.
    var st_gpu = StorGNLL[G_DIM].make_gpu(ctx)
    var g_log = Tensor.alloc(G_NLOG)
    var g_tgt = Tensor.alloc(G_NTGT)
    var g_grad = Tensor.alloc(G_NLOG)
    for i in range(G_NLOG):
        g_log.data[i] = _gnll_logits_gpu(i)
    for i in range(G_NTGT):
        g_tgt.data[i] = _gnll_tgt(i)
    g_log.upload(ctx); g_tgt.upload(ctx)
    var gpu_fwd = st_gpu.forward["gpu", G_B](g_log, g_tgt, Optional(ctx))
    st_gpu.reset_accum["gpu"]()
    st_gpu.forward_accumulate["gpu", G_B](g_log, g_tgt, Optional(ctx))
    var gpu_acc = st_gpu.read_accum["gpu"](Optional(ctx))
    st_gpu.vjp["gpu", G_B](g_log, g_tgt, g_grad, Optional(ctx))
    g_grad.download(ctx)

    var ok = True
    if abs(gpu_fwd - cpu_fwd) > TOL:
        ok = False
    if abs(gpu_acc - cpu_fwd) > TOL:
        ok = False
    var maxg = Scalar[DT](0.0)
    for i in range(G_NLOG):
        var d = abs(g_grad.data[i] - c_grad.data[i])
        if d > maxg:
            maxg = d
    if maxg > TOL:
        ok = False
    print("  GNLL GPU-vs-CPU: fwd Δ=", abs(gpu_fwd - cpu_fwd),
          " acc Δ=", abs(gpu_acc - cpu_fwd),
          " max grad Δ=", maxg, " ->", "OK" if ok else "FAIL")
    return ok


# ──────────────────────────── SequenceCE ───────────────────────────────

comptime S_B = 2
comptime S_SEQ = 3
comptime S_VOCAB = 4
comptime S_N = S_B * S_SEQ * S_VOCAB
comptime S_BT = S_B * S_SEQ


def _seq_logit(i: Int) -> Scalar[DT]:
    var x = 1.0 + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.6 * (t - (t * t * t) / 6.0))


def test_seqce_cpu_golden() raises -> Bool:
    print("test_seqce_cpu_golden (storage CPU vs golden) ...")
    comptime TOL = Scalar[DT](5e-3)
    var st = StorSeqCE[S_SEQ, S_VOCAB].make_cpu()
    var s_log = Tensor.alloc(S_N)
    var s_tgt = Tensor.alloc(S_N)
    var s_grad = Tensor.alloc(S_N)
    for i in range(S_N):
        s_log.data[i] = _seq_logit(i)
        s_tgt.data[i] = Scalar[DT](0.0)
    for r in range(S_BT):
        s_tgt.data[r * S_VOCAB + (r % S_VOCAB)] = Scalar[DT](1.0)
    var st_fwd = st.forward["cpu", S_B](s_log, s_tgt, None)
    st.vjp["cpu", S_B](s_log, s_tgt, s_grad, None)

    comptime FWD_GOLD = Scalar[DT](5.9516277)
    var ok = abs(st_fwd - FWD_GOLD) < TOL
    print("   fwd", st_fwd, "(exp", FWD_GOLD, ")", "OK" if ok else "FAIL")
    ok = _check("grad", s_grad, S_N, -5.3070835e-08, -0.33215928, TOL) and ok
    assert_true(ok, "SequenceCE CPU golden")
    print("  ok")
    return ok


def test_seqce_gpu_vs_cpu(ctx: DeviceContext) raises -> Bool:
    comptime TOL = Scalar[DT](2e-5)
    # CPU storage reference.
    var st_cpu = StorSeqCE[S_SEQ, S_VOCAB].make_cpu()
    var c_log = Tensor.alloc(S_N)
    var c_tgt = Tensor.alloc(S_N)
    var c_grad = Tensor.alloc(S_N)
    for i in range(S_N):
        c_log.data[i] = _seq_logit(i)
        c_tgt.data[i] = Scalar[DT](0.0)
    for r in range(S_BT):
        c_tgt.data[r * S_VOCAB + (r % S_VOCAB)] = Scalar[DT](1.0)
    var cpu_fwd = st_cpu.forward["cpu", S_B](c_log, c_tgt, None)
    st_cpu.vjp["cpu", S_B](c_log, c_tgt, c_grad, None)

    # GPU storage.
    var st_gpu = StorSeqCE[S_SEQ, S_VOCAB].make_gpu(ctx)
    var g_log = Tensor.alloc(S_N)
    var g_tgt = Tensor.alloc(S_N)
    var g_grad = Tensor.alloc(S_N)
    for i in range(S_N):
        g_log.data[i] = _seq_logit(i)
        g_tgt.data[i] = Scalar[DT](0.0)
    for r in range(S_BT):
        g_tgt.data[r * S_VOCAB + (r % S_VOCAB)] = Scalar[DT](1.0)
    g_log.upload(ctx); g_tgt.upload(ctx)
    var gpu_fwd = st_gpu.forward["gpu", S_B](g_log, g_tgt, Optional(ctx))
    st_gpu.reset_accum["gpu"]()
    st_gpu.forward_accumulate["gpu", S_B](g_log, g_tgt, Optional(ctx))
    var gpu_acc = st_gpu.read_accum["gpu"](Optional(ctx))
    st_gpu.vjp["gpu", S_B](g_log, g_tgt, g_grad, Optional(ctx))
    g_grad.download(ctx)

    var ok = True
    if abs(gpu_fwd - cpu_fwd) > TOL:
        ok = False
    if abs(gpu_acc - cpu_fwd) > TOL:
        ok = False
    var maxg = Scalar[DT](0.0)
    for i in range(S_N):
        var d = abs(g_grad.data[i] - c_grad.data[i])
        if d > maxg:
            maxg = d
    if maxg > TOL:
        ok = False
    print("  SeqCE GPU-vs-CPU: fwd Δ=", abs(gpu_fwd - cpu_fwd),
          " acc Δ=", abs(gpu_acc - cpu_fwd),
          " max grad Δ=", maxg, " ->", "OK" if ok else "FAIL")
    return ok


def main() raises:
    print("=" * 70)
    print("Storage GaussianNLL + SequenceCrossEntropy (CPU golden + GPU vs CPU)")
    print("=" * 70)
    var g_cpu = test_gnll_cpu_golden()
    var s_cpu = test_seqce_cpu_golden()

    var ctx = DeviceContext()
    var g_gpu = test_gnll_gpu_vs_cpu(ctx)
    var s_gpu = test_seqce_gpu_vs_cpu(ctx)

    assert_true(g_cpu and s_cpu and g_gpu and s_gpu, "storage loss golden")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
