"""Attention bf16-FLOW composability (AMP "Step B") — fp32 NoAMP regression +
bf16-flow GPU compiles/runs, for BOTH `ScaledDotProductAttention` and
`MaskedAttention`.

Attention is an fp32-INTERNAL leaf: QKᵀ / softmax / attn·V and the per-sample
cache are ALL fp32; only the I/O ACTIVATIONS (input/output/grad_output/
grad_input) flow at `ADT`, cast at the kernel boundary. So:

  - fp32 (`ADT = DT`): the legacy NoAMP leaf, byte-for-byte. This test pins
    fp32 forward/backward fingerprints (Σ vᵢ, Σ vᵢ·(i+1)) + GPU-vs-CPU parity.
  - bf16 (`ADT = bfloat16`): GPU-only; constructs + runs fwd+vjp. Because the
    math stays fp32, the only bf16 ops are the I/O casts → the bf16 result is
    a faithful round-trip of the fp32 result (NOT garbage), so we additionally
    assert it is CLOSE to the fp32 GPU result at a bf16 tolerance.

Run:
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_attention_bf16_flow.mojo
"""

from std.math import abs
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.attention import ScaledDotProductAttention
from mojo_rl.nn.primitives.masked_attention import MaskedAttention


comptime BF16 = DType.bfloat16
comptime D = 4
comptime NH = 2
comptime S = 3
comptime B = 2
comptime IN_N = B * S * D * 3
comptime OUT_N = B * S * D


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def _check(name: String, data: Tensor, n: Int,
           es: Scalar[DT], ew: Scalar[DT], tol: Scalar[DT]) -> Bool:
    var s: Scalar[DT] = 0
    var w: Scalar[DT] = 0
    for i in range(n):
        s += data.data[i]
        w += data.data[i] * Scalar[DT](i + 1)
    var ok = abs(s - es) < tol and abs(w - ew) < tol
    print("    ", name, "S", s, "(exp", es, ") W", w, "(exp", ew, ")",
          "OK" if ok else "FAIL")
    return ok


def _rel_max(b: List[Scalar[BF16]], ref_: Tensor, n: Int) -> Scalar[DT]:
    """Max per-element RELATIVE error between a bf16 result and the fp32 GPU
    reference (with a small absolute floor). bf16 has ~2^-8 relative precision,
    so the I/O-cast round-trip of an fp32-internal result lands here — NOT at a
    tight absolute tol (the grads have large magnitude → big absolute Δ)."""
    var m: Scalar[DT] = 0
    for i in range(n):
        var r = ref_.data[i]
        var d = abs(b[i].cast[DT]() - r)
        var denom = abs(r)
        if denom < Scalar[DT](1.0):
            denom = Scalar[DT](1.0)
        m = max(m, d / denom)
    return m


# ── fp32 NoAMP regression (CPU golden fingerprints, then captured into the
#    asserts below from the first clean run) ────────────────────────────────


def test_sdpa_fp32_golden() raises:
    print("test_sdpa_fp32_golden (ScaledDotProductAttention CPU golden) ...")
    # Golden fingerprints (Σ vᵢ, Σ vᵢ·(i+1)) captured from the first clean run
    # of the fp32 NoAMP path — pinned to catch any fp32 regression.
    comptime TOL = Scalar[DT](5e-3)
    var att = ScaledDotProductAttention[D, NH, S].make["cpu", Deterministic]()
    var sx = Tensor.alloc(IN_N)
    var sgo = Tensor.alloc(OUT_N)
    var sout = Tensor.alloc(OUT_N)
    var sgi = Tensor.alloc(IN_N)
    for i in range(IN_N):
        sx.data[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        sgo.data[i] = _spread(i, 4.1)
    att.forward["cpu", B](TensorRefs[1](sx), sout, None)
    att.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)
    var ok = _check("out", sout, OUT_N, -71.48699, -716.0237, TOL)
    ok = _check("gi", sgi, IN_N, -206.89229, -13272.312, TOL) and ok
    assert_true(ok, "SDPA fp32 golden fingerprint")
    print("  ok")


def test_sdpa_gpu_vs_cpu_and_bf16() raises:
    print("test_sdpa_gpu_vs_cpu_and_bf16 ...")
    comptime TOLP = Scalar[DT](2e-4)     # GPU(fp32) vs CPU
    comptime TOLB = Scalar[DT](0.06)     # bf16 vs GPU(fp32) MAX REL err — round-trip
    var c = DeviceContext()

    # CPU reference.
    var cpu = ScaledDotProductAttention[D, NH, S].make["cpu", Deterministic]()
    var sx = Tensor.alloc(IN_N)
    var sgo = Tensor.alloc(OUT_N)
    for i in range(IN_N):
        sx.data[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        sgo.data[i] = _spread(i, 4.1)
    var c_out = Tensor.alloc(OUT_N)
    var c_gi = Tensor.alloc(IN_N)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    # GPU fp32.
    var gpu = ScaledDotProductAttention[D, NH, S].make["gpu", Deterministic](
        Optional(c)
    )
    var gx = Tensor.alloc(IN_N)
    var ggo = Tensor.alloc(OUT_N)
    for i in range(IN_N):
        gx.data[i] = sx.data[i]
    for i in range(OUT_N):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(OUT_N)
    var g_gi = Tensor.alloc(IN_N)
    g_out.upload(c)
    g_gi.upload(c)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    var mo: Scalar[DT] = 0
    for i in range(OUT_N):
        mo = max(mo, abs(g_out.data[i] - c_out.data[i]))
    var mgi: Scalar[DT] = 0
    for i in range(IN_N):
        mgi = max(mgi, abs(g_gi.data[i] - c_gi.data[i]))
    print("  fp32 GPU vs CPU: out", mo, " gi", mgi)
    assert_true(mo < TOLP and mgi < TOLP, "SDPA GPU vs CPU fp32")

    # GPU bf16-flow. Activations stored/flowed at bf16; math fp32 internally.
    var bf = ScaledDotProductAttention[D, NH, S, False, True, BF16].make[
        "gpu", Deterministic
    ](Optional(c))
    var bx = TensorImpl[BF16].alloc(IN_N)
    var bgo = TensorImpl[BF16].alloc(OUT_N)
    for i in range(IN_N):
        bx.data[i] = sx.data[i].cast[BF16]()
    for i in range(OUT_N):
        bgo.data[i] = sgo.data[i].cast[BF16]()
    bx.upload(c)
    bgo.upload(c)
    var b_out = TensorImpl[BF16].alloc(OUT_N)
    var b_gi = TensorImpl[BF16].alloc(IN_N)
    b_out.upload(c)
    b_gi.upload(c)
    bf.forward["gpu", B](TensorRefs[1, ADT=BF16](bx), b_out, Optional(c))
    bf.vjp["gpu", B](
        TensorRefs[1, ADT=BF16](bx), bgo, TensorRefs[1, ADT=BF16](b_gi),
        Optional(c),
    )
    b_out.download(c)
    b_gi.download(c)
    var mob = _rel_max(b_out.data, g_out, OUT_N)
    var mgib = _rel_max(b_gi.data, g_gi, IN_N)
    print("  bf16-flow vs GPU fp32 (max rel err): out", mob, " gi", mgib)
    assert_true(mob < TOLB and mgib < TOLB, "SDPA bf16-flow round-trip")
    print("  ok")


def test_masked_gpu_vs_cpu_and_bf16() raises:
    print("test_masked_gpu_vs_cpu_and_bf16 (MaskedAttention) ...")
    comptime TOLP = Scalar[DT](2e-4)
    comptime TOLB = Scalar[DT](0.06)  # bf16 max rel err
    var c = DeviceContext()

    # CPU (default all-allow mask = non-causal SDPA).
    var cpu = MaskedAttention[D, NH, S].make["cpu", Deterministic]()
    var sx = Tensor.alloc(IN_N)
    var sgo = Tensor.alloc(OUT_N)
    for i in range(IN_N):
        sx.data[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        sgo.data[i] = _spread(i, 4.1)
    var c_out = Tensor.alloc(OUT_N)
    var c_gi = Tensor.alloc(IN_N)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    # GPU fp32.
    var gpu = MaskedAttention[D, NH, S].make["gpu", Deterministic](Optional(c))
    var gx = Tensor.alloc(IN_N)
    var ggo = Tensor.alloc(OUT_N)
    for i in range(IN_N):
        gx.data[i] = sx.data[i]
    for i in range(OUT_N):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(OUT_N)
    var g_gi = Tensor.alloc(IN_N)
    g_out.upload(c)
    g_gi.upload(c)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    var mo: Scalar[DT] = 0
    for i in range(OUT_N):
        mo = max(mo, abs(g_out.data[i] - c_out.data[i]))
    var mgi: Scalar[DT] = 0
    for i in range(IN_N):
        mgi = max(mgi, abs(g_gi.data[i] - c_gi.data[i]))
    print("  fp32 GPU vs CPU: out", mo, " gi", mgi)
    assert_true(mo < TOLP and mgi < TOLP, "Masked GPU vs CPU fp32")

    # GPU bf16-flow.
    var bf = MaskedAttention[D, NH, S, True, BF16].make["gpu", Deterministic](
        Optional(c)
    )
    var bx = TensorImpl[BF16].alloc(IN_N)
    var bgo = TensorImpl[BF16].alloc(OUT_N)
    for i in range(IN_N):
        bx.data[i] = sx.data[i].cast[BF16]()
    for i in range(OUT_N):
        bgo.data[i] = sgo.data[i].cast[BF16]()
    bx.upload(c)
    bgo.upload(c)
    var b_out = TensorImpl[BF16].alloc(OUT_N)
    var b_gi = TensorImpl[BF16].alloc(IN_N)
    b_out.upload(c)
    b_gi.upload(c)
    bf.forward["gpu", B](TensorRefs[1, ADT=BF16](bx), b_out, Optional(c))
    bf.vjp["gpu", B](
        TensorRefs[1, ADT=BF16](bx), bgo, TensorRefs[1, ADT=BF16](b_gi),
        Optional(c),
    )
    b_out.download(c)
    b_gi.download(c)
    var mob = _rel_max(b_out.data, g_out, OUT_N)
    var mgib = _rel_max(b_gi.data, g_gi, IN_N)
    print("  bf16-flow vs GPU fp32 (max rel err): out", mob, " gi", mgib)
    assert_true(mob < TOLB and mgib < TOLB, "Masked bf16-flow round-trip")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Attention bf16-flow (fp32 NoAMP regression + bf16 compiles/runs)")
    print("=" * 70)
    test_sdpa_fp32_golden()
    test_sdpa_gpu_vs_cpu_and_bf16()
    test_masked_gpu_vs_cpu_and_bf16()
    print("ALL PASSED")
