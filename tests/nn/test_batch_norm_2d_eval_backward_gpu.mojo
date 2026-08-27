# +--------------------------------------------------------------------------+ #
# | BatchNorm2D eval-mode backward — CPU and GPU disagree
# +--------------------------------------------------------------------------+ #
"""⚠⚠ THIS TEST DOCUMENTS A DIVERGENCE. It is not currently a pass/fail gate.

    pixi run -e apple mojo run -I . tests/nn/test_batch_norm_2d_eval_backward_gpu.mojo

One `BatchNorm2D`, CPU against GPU, backward, in both modes. Measured:

    training=True   fwd 3.6e-07  grad_in 3.0e-08  dgamma 8.9e-08  dbeta 1.5e-08
    training=False  fwd 0.0      grad_in 0.0      dgamma 0.136    dbeta 0.100

In EVAL the forward and grad_input agree BIT-EXACTLY while dgamma and dbeta
differ by precisely the CPU's own magnitude — i.e. **the GPU produces exactly
zero**. The two paths each made a different choice and each documented it as
fine:

    cpu  batch_norm_2d.mojo:1286  accumulates dgamma/dbeta,
                                  "harmless for a frozen backbone"
    gpu  batch_norm_2d.mojo:1352  `_bn2d_eval_bwd_kernel` is passed only
                                  (grad_output, gamma, running_var, gin) —
                                  it never touches gamma.grd / beta.grd

The CPU is mathematically right: the loss does depend on gamma and beta, so the
GPU drops real gradient. It is invisible in ordinary training (BN runs in
training mode) and bites exactly the frozen-BN fine-tuning setup — which is what
ACT's own reference uses (`FrozenBatchNorm2d`), and what any pretrained-backbone
transfer run would use.

Deciding WHICH way to converge them is a call with blast radius: making the GPU
accumulate would start updating gamma/beta in existing GPU runs that currently
do not. Hence a report rather than a fix, and hence this file prints rather than
asserts. Found via `tests/nn/test_resnet18_gpu.mojo`, which pins the current
behaviour so a change cannot pass unnoticed.
"""
from max.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D

comptime B = 2
comptime C = 4
comptime H = 8
comptime W = 8
comptime N = B * C * H * W


struct Grab(ParamVisitor):
    var v: List[Scalar[DT]]
    var nm: List[String]
    def __init__(out self):
        self.v = List[Scalar[DT]]()
        self.nm = List[String]()
    def __init__(out self, *, deinit move: Self):
        self.v = move.v^
        self.nm = move.nm^
    def visit[target: StaticString, K: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target != "cpu":
            grad.download(ctx.value())
        self.nm.append(String(name))
        for i in range(K):
            self.v.append(grad.data[i])


def run[training: Bool](ctx: DeviceContext) raises:
    var mc = BatchNorm2D[C, H, W].make["cpu", Kaiming]()
    var mg = BatchNorm2D[C, H, W].make["gpu", Kaiming](ctx)
    var tv = Scalar[DT](1.0) if training else Scalar[DT](0.0)
    mc.set_attr["training"](tv)
    mg.set_attr["training"](tv)

    var pc = TensorPack[1](); var pg = TensorPack[1]()
    pc[0].ensure(N); pg[0].ensure(N)
    for i in range(N):
        var x = Scalar[DT](0.1 * Float64(i % 17) - 0.8)
        pc[0].data[i] = x; pg[0].data[i] = x
    pg[0].upload(ctx)

    var oc = Tensor(); var og = Tensor()
    mc.forward["cpu", B](TensorRefs[1, MutAnyOrigin](pc[0]), oc)
    mg.forward["gpu", B](TensorRefs[1, MutAnyOrigin](pg[0]), og, ctx)
    ctx.synchronize(); og.download(ctx)
    var fw = Float64(0.0)
    for i in range(N):
        fw = max(fw, abs(Float64(oc.data[i]) - Float64(og.data[i])))

    var gc = Tensor(); var gg = Tensor()
    gc.ensure(N); gg.ensure(N)
    for i in range(N):
        var g = Scalar[DT](0.01 * Float64((i * 7) % 13) - 0.06)
        gc.data[i] = g; gg.data[i] = g
    gg.upload(ctx)
    var ggc = TensorPack[1](); var ggg = TensorPack[1]()
    ggc[0].ensure(N); ggg[0].ensure_gpu(ctx, N)
    mc.zero_grad["cpu"](None); mg.zero_grad["gpu"](ctx)
    mc.vjp["cpu", B](TensorRefs[1, MutAnyOrigin](pc[0]), gc,
                     TensorRefs[1, MutAnyOrigin](ggc[0]))
    mg.vjp["gpu", B](TensorRefs[1, MutAnyOrigin](pg[0]), gg,
                     TensorRefs[1, MutAnyOrigin](ggg[0]), ctx)
    ctx.synchronize(); ggg[0].download(ctx)
    var gi = Float64(0.0)
    for i in range(N):
        gi = max(gi, abs(Float64(ggc[0].data[i]) - Float64(ggg[0].data[i])))

    var a = Grab(); mc.for_each_param["cpu"](a, None, String(""))
    var b = Grab(); mg.for_each_param["gpu"](b, ctx, String(""))
    print("  training=" + String(training)
          + "  fwd " + String(fw) + "  grad_in " + String(gi))
    for k in range(len(a.nm)):
        var lo = k * C
        var d = Float64(0.0)
        var cm = Float64(0.0)
        for i in range(lo, lo + C):
            d = max(d, abs(Float64(a.v[i]) - Float64(b.v[i])))
            cm = max(cm, abs(Float64(a.v[i])))
        print("    " + a.nm[k] + ": max|cpu-gpu| " + String(d)
              + "   max|cpu| " + String(cm))


def main() raises:
    var ctx = DeviceContext()
    print("BatchNorm2D CPU-vs-GPU backward  [" + String(ctx.name()) + "]")
    run[False](ctx)
    run[True](ctx)
