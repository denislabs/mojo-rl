"""Slice / Concat / Add bf16-flow smoke (EZv2-bf16.2a-i) — dtype-transparent, CPU.

These three carry an `ADT` activation-flow dtype but do NO math (pure copy /
slice-split / elementwise sum), so `ADT=bfloat16` just flows bf16 through. No
GEMM → numerically valid on Apple. Each is checked: ACT_DT==bf16, forward (+ vjp
for Slice) finite, and ~matches the fp32 op within bf16 input-quantization
tolerance. Mirrors test_batch_norm_1d_bf16_smoke; 2-ary inputs use TensorPack so
the refs share one origin.

Run: pixi run mojo run -I . tests/nn/test_slice_concat_add_bf16_smoke.mojo
"""

from std.math import isnan, abs
from std.testing import assert_true
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.primitives.slice import Slice
from mojo_rl.nn.primitives.concat import Concat
from mojo_rl.nn.primitives.add import Add

comptime BF16 = DType.bfloat16


def _rng(mut xs: UInt64) -> Scalar[DT]:
    xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
    return Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[DT](1.0)


def _maxd(bf: TensorImpl[BF16], f32: Tensor, n: Int) raises -> Scalar[DT]:
    var m = Scalar[DT](0.0)
    for i in range(n):
        assert_true(not isnan(bf.data[i].cast[DT]()), "bf16 finite")
        var d = abs(bf.data[i].cast[DT]() - f32.data[i])
        if d > m:
            m = d
    return m


def main() raises:
    comptime B = 6
    print("Slice/Concat/Add bf16-flow smoke (CPU, dtype-transparent)")

    # ── Slice[8, 2, 6] → OUT=4 (arity-1) ────────────────────────────────
    comptime assert Slice[8, 2, 6, ADT=BF16].ACT_DT == BF16, "Slice ACT_DT"
    var s32 = Slice[8, 2, 6].make["cpu", Kaiming]()
    var sbf = Slice[8, 2, 6, ADT=BF16].make["cpu", Kaiming]()
    var xf = Tensor.alloc(B * 8)
    var xb = TensorImpl[BF16].alloc(B * 8)
    var xs = UInt64(0x9E3779B97F4A7C15)
    for i in range(B * 8):
        var v = _rng(xs); xf.data[i] = v; xb.data[i] = v.cast[BF16]()
    var of = Tensor.alloc(B * 4)
    var ob = TensorImpl[BF16].alloc(B * 4)
    s32.forward["cpu", B](TensorRefs[1](xf), of, None)
    sbf.forward["cpu", B](TensorRefs[1, ADT=BF16](xb), ob, None)
    print("  Slice fwd max|bf16-fp32| =", _maxd(ob, of, B * 4))
    var gof = Tensor.alloc(B * 4)
    var gob = TensorImpl[BF16].alloc(B * 4)
    for i in range(B * 4):
        gof.data[i] = Scalar[DT](0.5); gob.data[i] = Scalar[BF16](0.5)
    var gif = Tensor.alloc(B * 8)
    var gib = TensorImpl[BF16].alloc(B * 8)
    s32.vjp["cpu", B](TensorRefs[1](xf), gof, TensorRefs[1](gif), None)
    sbf.vjp["cpu", B](
        TensorRefs[1, ADT=BF16](xb), gob, TensorRefs[1, ADT=BF16](gib), None
    )
    print("  Slice vjp max|bf16-fp32| =", _maxd(gib, gif, B * 8))
    assert_true(_maxd(gib, gif, B * 8) < Scalar[DT](0.1), "Slice vjp ~fp32")

    # ── Concat[4, 4] → OUT=8 (arity-2 → TensorPack) ─────────────────────
    comptime assert Concat[4, 4, ADT=BF16].ACT_DT == BF16, "Concat ACT_DT"
    var c32 = Concat[4, 4].make["cpu", Kaiming]()
    var cbf = Concat[4, 4, ADT=BF16].make["cpu", Kaiming]()
    var cif = TensorPack[2]()
    var cib = TensorPack[2, ADT=BF16]()
    cif[0].ensure(B * 4); cif[1].ensure(B * 4)
    cib[0].ensure(B * 4); cib[1].ensure(B * 4)
    var cs = UInt64(0x1234567)
    for i in range(B * 4):
        var v0 = _rng(cs); cif[0].data[i] = v0; cib[0].data[i] = v0.cast[BF16]()
        var v1 = _rng(cs); cif[1].data[i] = v1; cib[1].data[i] = v1.cast[BF16]()
    var cof = Tensor.alloc(B * 8); var cob = TensorImpl[BF16].alloc(B * 8)
    c32.forward["cpu", B](TensorRefs[2](cif[0], cif[1]), cof, None)
    cbf.forward["cpu", B](TensorRefs[2, ADT=BF16](cib[0], cib[1]), cob, None)
    print("  Concat fwd max|bf16-fp32| =", _maxd(cob, cof, B * 8))
    assert_true(_maxd(cob, cof, B * 8) < Scalar[DT](0.1), "Concat fwd ~fp32")

    # ── Add[8] (z = a + b) (arity-2 → TensorPack) ───────────────────────
    comptime assert Add[8, ADT=BF16].ACT_DT == BF16, "Add ACT_DT"
    var d32 = Add[8].make["cpu", Kaiming]()
    var dbf = Add[8, ADT=BF16].make["cpu", Kaiming]()
    var dif = TensorPack[2]()
    var dib = TensorPack[2, ADT=BF16]()
    dif[0].ensure(B * 8); dif[1].ensure(B * 8)
    dib[0].ensure(B * 8); dib[1].ensure(B * 8)
    var ds = UInt64(0xABCDEF)
    for i in range(B * 8):
        var v0 = _rng(ds); dif[0].data[i] = v0; dib[0].data[i] = v0.cast[BF16]()
        var v1 = _rng(ds); dif[1].data[i] = v1; dib[1].data[i] = v1.cast[BF16]()
    var dof = Tensor.alloc(B * 8); var dob = TensorImpl[BF16].alloc(B * 8)
    d32.forward["cpu", B](TensorRefs[2](dif[0], dif[1]), dof, None)
    dbf.forward["cpu", B](TensorRefs[2, ADT=BF16](dib[0], dib[1]), dob, None)
    print("  Add fwd max|bf16-fp32| =", _maxd(dob, dof, B * 8))
    assert_true(_maxd(dob, dof, B * 8) < Scalar[DT](0.1), "Add fwd ~fp32")

    _ = s32^; _ = sbf^; _ = c32^; _ = cbf^; _ = d32^; _ = dbf^
    print("PASS — Slice/Concat/Add bf16-flow (dtype-transparent) match fp32")
