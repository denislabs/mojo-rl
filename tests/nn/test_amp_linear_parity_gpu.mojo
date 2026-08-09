"""AMP Linear bf16 vs fp32 NUMERIC parity (Phase 1 + 2) — NVIDIA gate.

Apple Metal's `linalg.matmul` mis-computes bf16 GEMMs (see test_amp_weight_cache),
so this NUMERIC parity test is meaningful only on NVIDIA (cutlass bf16). It checks
the bf16 path (cached weight forward + bf16 backward grad_w/grad_x) against the
fp32 path on realistic, NON-cancelling random data, where bf16 should track fp32
to a few percent.

Run on NVIDIA:
  pixi run -e nvidia mojo run -I . tests/nn/test_amp_linear_parity_gpu.mojo
(On Apple it will "FAIL" purely due to the known Metal bf16 linalg bug — expected.)
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear


comptime BF16 = DType.bfloat16
comptime IN = 256
comptime OUT = 256
comptime B = 64
comptime W = IN * OUT
comptime RELTOL = Scalar[DT](0.05)   # bf16 vs fp32, non-cancelling data


def _rand(i: Int) -> Scalar[DT]:
    """Cheap LCG hash → ~uniform [-0.5, 0.5] (non-cancelling, NN-like)."""
    var h = (UInt32(i) * UInt32(1103515245) + UInt32(12345)) & UInt32(0x7FFFFFFF)
    return Scalar[DT](Float64(Int(h % UInt32(1000))) / 1000.0 - 0.5)


def _to_f32(b: TensorImpl[BF16], n: Int) raises -> Tensor:
    """Copy a (downloaded) bf16 activation tensor into an fp32 Tensor so the
    fp32 `_relerr` can compare it."""
    var out = Tensor.alloc(n)
    for i in range(n):
        out.data[i] = b.data[i].cast[DT]()
    return out^


def _relerr(a: Tensor, b: Tensor, n: Int) -> Scalar[DT]:
    """max|a-b| / max(|b|, eps) — a scale-aware relative error over the tensor."""
    var md: Scalar[DT] = 0
    var mr: Scalar[DT] = 0
    for i in range(n):
        var d = abs(a.data[i] - b.data[i])
        if d > md:
            md = d
        if abs(b.data[i]) > mr:
            mr = abs(b.data[i])
    if mr < Scalar[DT](1e-6):
        mr = Scalar[DT](1e-6)
    return md / mr


def main() raises:
    print("=" * 70)
    print("AMP Linear bf16 vs fp32 numeric parity (Phase 1+2) —", IN, "x", OUT, "B", B)
    print("=" * 70)
    var c = DeviceContext()
    var amp = Linear[IN, OUT, BF16].make["gpu", Deterministic](Optional(c))
    var fp = Linear[IN, OUT].make["gpu", Deterministic](Optional(c))

    for k in range(W):
        amp.weight.val.data[k] = _rand(k)
        fp.weight.val.data[k] = amp.weight.val.data[k]
    for k in range(OUT):
        amp.bias.val.data[k] = _rand(k + 999983)
        fp.bias.val.data[k] = amp.bias.val.data[k]
    amp.weight.val.upload(c); amp.bias.val.upload(c)
    fp.weight.val.upload(c); fp.bias.val.upload(c)
    amp.weight.val.version += 1   # force the bf16 weight cast

    # fp32 input + its bf16 mirror (bf16-flow activations flow at bf16).
    var x = Tensor.alloc(B * IN)
    var xb = TensorImpl[BF16].alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = _rand(i + 7)
        xb.data[i] = x.data[i].cast[BF16]()
    x.upload(c); xb.upload(c)

    # ---- forward ----
    var ya = TensorImpl[BF16].alloc(B * OUT)   # bf16-flow output
    var yf = Tensor.alloc(B * OUT)              # fp32 output
    amp.forward["gpu", B](TensorRefs[1, ADT=BF16](xb), ya, Optional(c))
    fp.forward["gpu", B](TensorRefs[1](x), yf, Optional(c))
    ya.download(c); yf.download(c)
    var e_fwd = _relerr(_to_f32(ya, B * OUT), yf, B * OUT)

    # ---- backward ----
    var go = Tensor.alloc(B * OUT)
    var gob = TensorImpl[BF16].alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = _rand(i + 31)
        gob.data[i] = go.data[i].cast[BF16]()
    go.upload(c); gob.upload(c)
    var gia = TensorImpl[BF16].alloc(B * IN)   # bf16-flow grad_x
    var gif = Tensor.alloc(B * IN)
    amp.zero_grad["gpu"](Optional(c))
    fp.zero_grad["gpu"](Optional(c))
    amp.vjp["gpu", B](
        TensorRefs[1, ADT=BF16](xb), gob, TensorRefs[1, ADT=BF16](gia),
        Optional(c),
    )
    fp.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gif), Optional(c))
    gia.download(c); gif.download(c)
    amp.weight.grd.download(c); fp.weight.grd.download(c)
    var e_gx = _relerr(_to_f32(gia, B * IN), gif, B * IN)
    var e_gw = _relerr(amp.weight.grd, fp.weight.grd, W)

    print("  forward   rel.err =", e_fwd, "OK" if e_fwd < RELTOL else "FAIL")
    print("  grad_x    rel.err =", e_gx, "OK" if e_gx < RELTOL else "FAIL")
    print("  grad_w    rel.err =", e_gw, "OK" if e_gw < RELTOL else "FAIL")
    var ok = e_fwd < RELTOL and e_gx < RELTOL and e_gw < RELTOL
    assert_true(ok, "AMP Linear bf16 vs fp32 parity")
    print("ALL PASSED")
