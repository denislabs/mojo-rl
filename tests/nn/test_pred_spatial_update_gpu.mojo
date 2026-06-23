"""GPU: after one real forward→vjp→Adam step on the spatial pred net, does the
POLICY output weight (1.0.4.weight) actually change ON DEVICE — like the VALUE
output weight (1.1.4.weight)? If policy stays 0 while value moves, the device
update of the policy head is broken (a transfer/optimizer bug invisible to
loss-only parity)."""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.optimizer.grad_clip import clip_grad_norm
from mojo_rl.deep_agents.muzero.nets_spatial import MZPredNetC4Spatial
from mojo_rl.deep_agents.muzero.loss_ops import soft_ce_slice_loss_and_grad


struct WReader(ParamVisitor):
    var pol: Scalar[DT]
    var val: Scalar[DT]

    def __init__(out self):
        self.pol = 0
        self.val = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if name == "1.0.4.weight" or name == "1.1.4.weight":
            comptime if target == "gpu":
                param.download(ctx.value())
            var s = Scalar[DT](0)
            for i in range(N):
                s += abs(param.data[i])
            if name == "1.0.4.weight":
                self.pol = s
            else:
                self.val = s


def main() raises:
    comptime CH = 64
    comptime ACT = 7
    comptime BINS = 51
    comptime HH = 6
    comptime WW = 7
    comptime NB = 3
    comptime LATENT = CH * HH * WW
    comptime PRED_OUT = ACT + BINS
    comptime B = 16
    comptime Pred = MZPredNetC4Spatial[CH, ACT, BINS, HH, WW, NB]

    var ctx = DeviceContext()
    var pred = Pred.make["gpu", INIT=Kaiming](Optional(ctx))
    var opt = Adam(lr=Scalar[DT](2e-3))

    # device latent
    var z = Tensor.alloc(B * LATENT)
    for i in range(B * LATENT):
        z.data[i] = Scalar[DT](0.05) * Scalar[DT](((i * 3) % 11) - 5)
    z.ensure_gpu(ctx, B * LATENT)
    z.upload(ctx)

    # peaked policy + value targets (host)
    var pol_tgt = List[Scalar[DT]](length=B * ACT, fill=0)
    var val_tgt = List[Scalar[DT]](length=B * BINS, fill=0)
    for b in range(B):
        pol_tgt[b * ACT + (b % ACT)] = Scalar[DT](1.0)
        val_tgt[b * BINS + (b % BINS)] = Scalar[DT](1.0)

    print("=== before step ===")
    var r0 = WReader()
    pred.for_each_param["gpu"](r0, Optional(ctx))
    print("policy-out sum|w| =", r0.pol, " value-out sum|w| =", r0.val,
          " (both 0 at zero-init)")

    var pout = Tensor()
    pout.ensure_gpu(ctx, B * PRED_OUT)
    for step in range(5):
        pred.zero_grad["gpu"](Optional(ctx))
        pred.forward["gpu", B](TensorRefs[Pred.ARITY](z), pout, Optional(ctx))
        pout.download(ctx)
        ctx.synchronize()
        # compute grads on host, upload
        var gpout = Tensor.alloc(B * PRED_OUT)
        var gscale = Scalar[DT](1.0) / Scalar[DT](B)
        _ = soft_ce_slice_loss_and_grad[B, PRED_OUT, 0, ACT](
            pout.data, pol_tgt, gscale, gpout.data
        )
        _ = soft_ce_slice_loss_and_grad[B, PRED_OUT, ACT, BINS](
            pout.data, val_tgt, gscale * Scalar[DT](0.25), gpout.data
        )
        gpout.ensure_gpu(ctx, B * PRED_OUT)
        gpout.upload(ctx)
        var gpin = Tensor()
        gpin.ensure_gpu(ctx, B * LATENT)
        pred.vjp["gpu", B](
            TensorRefs[Pred.ARITY](z), gpout,
            TensorRefs[Pred.ARITY](gpin), Optional(ctx),
        )
        _ = clip_grad_norm["gpu", Pred](pred, Scalar[DT](1.0), Optional(ctx))
        opt.begin_step()
        pred.for_each_param["gpu"](opt, Optional(ctx))
        ctx.synchronize()

    print("=== after 5 steps ===")
    var r1 = WReader()
    pred.for_each_param["gpu"](r1, Optional(ctx))
    print("policy-out sum|w| =", r1.pol, " value-out sum|w| =", r1.val)
    if r1.pol < Scalar[DT](1e-9):
        print(">>> POLICY OUTPUT WEIGHT DID NOT UPDATE ON DEVICE — BUG <<<")
    elif r1.val < Scalar[DT](1e-9):
        print(">>> value didn't update either — broader update bug <<<")
    else:
        print(">>> both policy + value updated on device — device path OK <<<")
