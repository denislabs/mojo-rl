"""Conv2D OC==1 GPU forward regression gate (max_matmul N=1 miscompute).

`max_matmul[transpose_b=True]` on GPU SILENTLY returns wrong values when the
GEMM's N dimension is 1 (verified on Metal: maxdiff ~9 vs exact 0.0 at N>=2).
Conv2D's GPU forward computes `out_packed[BS, OC] = col @ Wᵀ`, so any conv
with OC=1 — e.g. a grayscale decoder output conv (DreamerV3 Pong C=1) —
produced a corrupt FORWARD while all vjp GEMMs (differently shaped) stayed
correct: training chased garbage reconstructions with "correct" gradients
(obs_loss frozen at the sigmoid(0) level; caught on the Phase B-2 run).
Fixed by `_fwd_oc1_matvec_kernel` (direct fused matvec + bias at OC==1).

Gates (CPU↔GPU parity, identical host-RNG init):
  1. Conv2D k5 s1 p2 OC=1 forward (the DreamerV3 pool-decoder output shape)
  2. OC=2 control (GEMM path, must also match)
  3. OC=1 vjp: input grads + weight/bias grads
  4. DreamerDecoderCNNPool end-to-end forward (the production composite)

Run (GPU env required):
    pixi run -e apple  mojo run -I . tests/nn/test_conv2d_oc1_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_conv2d_oc1_gpu.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, LAYOUT_NCHW
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import TruncNormalIn
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.nets_cnn import DreamerDecoderCNNPool

comptime H = 16
comptime W = 16
comptime IC = 8
comptime B = 4
comptime TOL = Scalar[DT](1e-4)


struct _GradDump(ParamVisitor):
    var grads: List[List[Scalar[DT]]]

    def __init__(out self):
        self.grads = List[List[Scalar[DT]]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            grad.download(ctx.value())
        var g = List[Scalar[DT]]()
        for i in range(min(grad.n, N)):
            g.append(grad.data[i])
        self.grads.append(g^)


def _maxdiff(a: Tensor, b: Tensor, n: Int) -> Scalar[DT]:
    var md = Scalar[DT](0.0)
    for i in range(n):
        var d = a.data[i] - b.data[i]
        d = d if d >= Scalar[DT](0.0) else -d
        if d > md:
            md = d
    return md


def _fill_pair(mut a: Tensor, mut b: Tensor, n: Int, ctx: DeviceContext) raises:
    for i in range(n):
        a.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        b.data[i] = a.data[i]
    b.upload(ctx)


def test_conv_fwd[OC: Int](ctx: DeviceContext) raises:
    print("test_conv_fwd OC =", OC, "...")
    comptime Cv = Conv2D[IC, OC, 5, 1, 2, H, W, DT, LAYOUT_NCHW]
    var octx = Optional[DeviceContext](ctx)
    seed(3)
    var cc = Cv.make["cpu", TruncNormalIn](None)
    seed(3)
    var cg = Cv.make["gpu", TruncNormalIn](octx)
    seed(55)
    var x = Tensor.alloc(B * IC * H * W)
    var xg = Tensor.alloc(B * IC * H * W)
    _fill_pair(x, xg, B * IC * H * W, ctx)
    var oc_ = Tensor()
    cc.forward["cpu", B](TensorRefs[1](x), oc_, None)
    var og = Tensor()
    cg.forward["gpu", B](TensorRefs[1](xg), og, octx)
    og.download(ctx)
    var md = _maxdiff(oc_, og, B * OC * H * W)
    assert_true(
        md < TOL, "conv fwd OC=" + String(OC) + " maxdiff " + String(md)
    )
    print("  ok (maxdiff", md, ")")


def test_conv_oc1_vjp(ctx: DeviceContext) raises:
    print("test_conv_oc1_vjp ...")
    comptime Cv = Conv2D[IC, 1, 5, 1, 2, H, W, DT, LAYOUT_NCHW]
    var octx = Optional[DeviceContext](ctx)
    seed(3)
    var cc = Cv.make["cpu", TruncNormalIn](None)
    seed(3)
    var cg = Cv.make["gpu", TruncNormalIn](octx)
    seed(56)
    var x = Tensor.alloc(B * IC * H * W)
    var xg = Tensor.alloc(B * IC * H * W)
    _fill_pair(x, xg, B * IC * H * W, ctx)
    var go = Tensor.alloc(B * H * W)
    var gog = Tensor.alloc(B * H * W)
    _fill_pair(go, gog, B * H * W, ctx)
    # forward first (vjp may reuse the forward's col buffer)
    var oc_ = Tensor()
    cc.forward["cpu", B](TensorRefs[1](x), oc_, None)
    var og = Tensor()
    cg.forward["gpu", B](TensorRefs[1](xg), og, octx)
    var gx_c = Tensor()
    cc.zero_grad["cpu"](None)
    cc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gx_c), None)
    var gx_g = Tensor()
    cg.zero_grad["gpu"](octx)
    cg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gx_g), octx)
    gx_g.download(ctx)
    var md = _maxdiff(gx_c, gx_g, B * IC * H * W)
    assert_true(md < TOL, "conv OC=1 input-grad maxdiff " + String(md))
    var dc = _GradDump()
    cc.for_each_param["cpu"](dc, None)
    var dg = _GradDump()
    cg.for_each_param["gpu"](dg, octx)
    for k in range(len(dc.grads)):
        var pmd = Scalar[DT](0.0)
        for i in range(len(dc.grads[k])):
            var d = dc.grads[k][i] - dg.grads[k][i]
            d = d if d >= Scalar[DT](0.0) else -d
            if d > pmd:
                pmd = d
        assert_true(pmd < TOL, "conv OC=1 param-grad maxdiff " + String(pmd))
    print("  ok")


def test_dec_pool_fwd(ctx: DeviceContext) raises:
    print("test_dec_pool_fwd (composite decoder forward parity) ...")
    comptime C = 1
    comptime IMG = 16
    comptime BASE = 4
    comptime DETER = 8
    comptime SC = 8
    comptime UNITS = 6
    comptime FEATIN = DETER + SC
    comptime Dec = DreamerDecoderCNNPool[
        FEATIN, DETER, C, IMG, IMG, BASE, UNITS, SwishOp
    ]
    var octx = Optional[DeviceContext](ctx)
    seed(7)
    var dc = Dec.make["cpu", TruncNormalIn](None)
    seed(7)
    var dg = Dec.make["gpu", TruncNormalIn](octx)
    seed(99)
    var feat = Tensor.alloc(B * FEATIN)
    var featg = Tensor.alloc(B * FEATIN)
    _fill_pair(feat, featg, B * FEATIN, ctx)
    var rc = Tensor()
    dc.forward["cpu", B](TensorRefs[1](feat), rc, None)
    var rg = Tensor()
    dg.forward["gpu", B](TensorRefs[1](featg), rg, octx)
    rg.download(ctx)
    var md = _maxdiff(rc, rg, B * C * IMG * IMG)
    assert_true(md < TOL, "dec pool fwd maxdiff " + String(md))
    print("  ok (maxdiff", md, ")")


def main() raises:
    print("Conv2D OC=1 GPU gates (max_matmul N=1 miscompute regression)")
    with DeviceContext() as ctx:
        test_conv_fwd[1](ctx)
        test_conv_fwd[2](ctx)
        test_conv_oc1_vjp(ctx)
        test_dec_pool_fwd(ctx)
    print("CONV2D OC1 GPU OK")
