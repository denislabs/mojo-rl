"""BatchNorm2D bf16-flow smoke (A4.1) — fp32-internal correctness, CPU.

BatchNorm2D[..., DType.bfloat16] accepts/emits bf16 activations but computes
stats/normalize in fp32 internally (AMP §3). BN has NO GEMM, so unlike bf16
matmul this is numerically valid on Apple — we can check the bf16-flow output
matches the fp32 BatchNorm2D within bf16 tolerance (both use gamma=1/beta=0 from
`make`). Also asserts forward + vjp are finite.

Run: pixi run mojo run -I . tests/nn/test_batch_norm_2d_bf16_smoke.mojo
"""

from std.math import isnan, isinf, abs
from std.testing import assert_true
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D

comptime BF16 = DType.bfloat16


def main() raises:
    comptime C = 4
    comptime H = 2
    comptime W = 2
    comptime B = 4
    comptime FLAT = C * H * W
    comptime N = B * FLAT

    print("BatchNorm2D bf16-flow smoke (CPU, fp32-internal)")

    var bn32 = BatchNorm2D[C, H, W].make["cpu", Kaiming]()
    var bnbf = BatchNorm2D[C, H, W, ADT=BF16].make["cpu", Kaiming]()
    bn32.set_training(True)
    bnbf.set_training(True)

    # input
    var xf = Tensor.alloc(N)
    var xb = TensorImpl[BF16].alloc(N)
    var xs = UInt64(0x9E3779B97F4A7C15)
    for i in range(N):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        var v = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[DT](1.0)
        xf.data[i] = v
        xb.data[i] = v.cast[BF16]()

    var of = Tensor.alloc(N)
    var ob = TensorImpl[BF16].alloc(N)
    bn32.forward["cpu", B](TensorRefs[1](xf), of, None)
    bnbf.forward["cpu", B](TensorRefs[1, ADT=BF16](xb), ob, None)

    var maxd = Scalar[DT](0.0)
    for i in range(N):
        var d = abs(ob.data[i].cast[DT]() - of.data[i])
        if d > maxd:
            maxd = d
        assert_true(not isnan(ob.data[i].cast[DT]()), "bf16 out finite")
    print("  fwd max|bf16 - fp32| =", maxd)
    assert_true(maxd < Scalar[DT](0.1), "bf16 fwd ~matches fp32 (BN no GEMM)")

    # vjp finite
    var gof = Tensor.alloc(N)
    var gob = TensorImpl[BF16].alloc(N)
    for i in range(N):
        gof.data[i] = Scalar[DT](0.5)
        gob.data[i] = Scalar[BF16](0.5)
    var gif = Tensor.alloc(N)
    var gib = TensorImpl[BF16].alloc(N)
    bn32.vjp["cpu", B](TensorRefs[1](xf), gof, TensorRefs[1](gif), None)
    bnbf.vjp["cpu", B](
        TensorRefs[1, ADT=BF16](xb), gob, TensorRefs[1, ADT=BF16](gib), None
    )
    var maxgd = Scalar[DT](0.0)
    for i in range(N):
        assert_true(not isnan(gib.data[i].cast[DT]()), "bf16 grad finite")
        var d = abs(gib.data[i].cast[DT]() - gif.data[i])
        if d > maxgd:
            maxgd = d
    print("  vjp max|bf16 - fp32| =", maxgd)
    assert_true(maxgd < Scalar[DT](0.1), "bf16 vjp ~matches fp32")

    _ = bn32^
    _ = bnbf^
    print("PASS — BatchNorm2D bf16-flow (fp32-internal) matches fp32")
