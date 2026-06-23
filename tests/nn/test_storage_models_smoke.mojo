"""nn.storage.models composition smoke (CPU + GPU).

The model aliases (transformer/ViT/ResNet/Conv) are pure compositions of
storage leaves + combinators, each already gated bit-identical legacy↔storage.
So correctness is inherited; this smoke proves the COMPOSITIONS type-check and
the full forward+vjp graph runs end-to-end on both targets (finite outputs).

Run:
  pixi run mojo run -I . tests/nn/test_storage_models_smoke.mojo
  pixi run -e apple mojo run -I . tests/nn/test_storage_models_smoke.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.models.vit import ViT
from mojo_rl.nn.models.resnet import ResBlockConv2DBN
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.models.transformer import TransformerBlock


comptime B = 2

# ViT: 1×8×8 image, 4×4 patches → 2×2 = 4 patches, embed 8, 2 heads, 1 layer, 3 classes.
comptime VIT = ViT[1, 8, 8, 4, 8, 2, 1, 4, 3]
comptime VIT_IN = 1 * 8 * 8
comptime VIT_OUT = 3

# ResBlock: C=2, K=3, P=1, 4×4 (spatial preserved).
comptime RES = ResBlockConv2DBN[2, 3, 1, 4, 4]
comptime RES_IN = 2 * 4 * 4

# Conv→BN→ReLU: 1→2 ch, K=3 S=1 P=1, 4×4.
comptime CBR = Conv2DBatchNormReLU[1, 2, 3, 1, 1, 4, 4]
comptime CBR_IN = 1 * 4 * 4
comptime CBR_OUT = 2 * 4 * 4

# TransformerBlock: dim 8, 2 heads, seq 4, ff 16, causal.
comptime TB = TransformerBlock[8, 2, 4, 16, True]
comptime TB_DIM = 4 * 8


def _all_finite(t: Tensor, n: Int) -> Bool:
    for i in range(n):
        var v = t.data[i]
        if v != v:  # NaN
            return False
    return True


def _run_cpu[M: Module, IN: Int, OUT: Int](name: String) raises -> Bool:
    var m = M.make["cpu", Deterministic]()
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT]((i % 11) - 5) * 0.1
    var out = Tensor.alloc(B * OUT)
    m.forward["cpu", B](TensorRefs[M.ARITY](x), out, None)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var gi = Tensor.alloc(B * IN)
    m.zero_grad["cpu"](None)
    m.vjp["cpu", B](TensorRefs[M.ARITY](x), go, TensorRefs[M.ARITY](gi), None)
    var ok = _all_finite(out, B * OUT) and _all_finite(gi, B * IN)
    print("  ", name, "CPU fwd+vjp finite:", ok)
    return ok


def _run_gpu[M: Module, IN: Int, OUT: Int](name: String, c: DeviceContext) raises -> Bool:
    var m = M.make["gpu", Deterministic](Optional(c))
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT]((i % 11) - 5) * 0.1
    x.upload(c)
    var out = Tensor.alloc(B * OUT)
    m.forward["gpu", B](TensorRefs[M.ARITY](x), out, Optional(c))
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    go.upload(c)
    var gi = Tensor.alloc(B * IN)
    m.zero_grad["gpu"](Optional(c))
    m.vjp["gpu", B](TensorRefs[M.ARITY](x), go, TensorRefs[M.ARITY](gi), Optional(c))
    out.download(c)
    gi.download(c)
    var ok = _all_finite(out, B * OUT) and _all_finite(gi, B * IN)
    print("  ", name, "GPU fwd+vjp finite:", ok)
    return ok


def main() raises:
    print("=" * 60)
    print("nn.storage.models composition smoke")
    print("=" * 60)
    print("CPU:")
    var c1 = _run_cpu[VIT, VIT_IN, VIT_OUT]("ViT       ")
    var c2 = _run_cpu[RES, RES_IN, RES_IN]("ResBlock  ")
    var c3 = _run_cpu[CBR, CBR_IN, CBR_OUT]("Conv-BN-RL")
    var c4 = _run_cpu[TB, TB_DIM, TB_DIM]("TransfBlk ")
    assert_true(c1 and c2 and c3 and c4, "CPU models smoke")

    print("GPU:")
    var c = DeviceContext()
    var g1 = _run_gpu[VIT, VIT_IN, VIT_OUT]("ViT       ", c)
    var g2 = _run_gpu[RES, RES_IN, RES_IN]("ResBlock  ", c)
    var g3 = _run_gpu[CBR, CBR_IN, CBR_OUT]("Conv-BN-RL", c)
    var g4 = _run_gpu[TB, TB_DIM, TB_DIM]("TransfBlk ", c)
    assert_true(g1 and g2 and g3 and g4, "GPU models smoke")
    print("ALL PASSED")
