"""GPU twin of test_mz_dyn_reward_learns: does the dyn reward head learn ON GPU?

The live MuZero run is GPU. The GPU COL=1 conv vjp is correct on NVIDIA, and the
pred net updates on NVIDIA GPU — but the dyn net additionally exercises, on GPU,
the ComputeGraph vjp + Slice/BroadcastTokens/LayerNorm/Concat/Add/MinMaxNorm,
none of which the pred net touches. If any of those GPU vjps is wrong on NVIDIA,
the dyn net learns garbage → loss_reward stuck → dead policy. This trains the dyn
reward head on GPU with clean targets and checks loss_reward drops + the reward
output weight escapes zero-init.

  apple GPU: should learn (loss drops, weight grows).
  NVIDIA GPU: if it does NOT learn → an NVIDIA GPU dyn-graph vjp is broken = the bug.
"""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.muzero.nets_spatial import MZDynNetC4Spatial
from mojo_rl.deep_agents.muzero.loss_ops import soft_ce_slice_loss_and_grad
from mojo_rl.deep_agents.zero.twohot_targets import mz_two_hot_target_batch


struct RewW(ParamVisitor):
    var s: Scalar[DT]

    def __init__(out self):
        self.s = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if name.endswith("rew.4.weight"):
            comptime if target == "gpu":
                param.download(ctx.value())
            var a = Scalar[DT](0)
            for i in range(N):
                a += abs(param.data[i])
            self.s = a


def main() raises:
    comptime CH = 8
    comptime ACT = 7
    comptime BINS = 51
    comptime HH = 6
    comptime WW = 7
    comptime NB = 2
    comptime LATENT = CH * HH * WW
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime B = 16
    comptime Dyn = MZDynNetC4Spatial[CH, ACT, BINS, HH, WW, NB]
    var ctx = DeviceContext()
    var dyn = Dyn.make["gpu", INIT=Kaiming](Optional(ctx))
    var opt = Adam(lr=Scalar[DT](2e-3))

    var din = Tensor.alloc(B * DYN_IN)
    for b in range(B):
        var base = b * DYN_IN
        for i in range(LATENT):
            din.data[base + i] = Scalar[DT](0.05) * Scalar[DT](((b * 7 + i) % 11) - 5)
        for a in range(ACT):
            din.data[base + LATENT + a] = Scalar[DT](0.0)
        din.data[base + LATENT + (b % ACT)] = Scalar[DT](1.0)
    din.ensure_gpu(ctx, B * DYN_IN); din.upload(ctx)

    var r_scalars = List[Scalar[DT]](length=B, fill=0)
    for b in range(B):
        r_scalars[b] = Scalar[DT]((b % 3) - 1)
    var twr = Tensor.alloc(B * BINS)
    mz_two_hot_target_batch[B, BINS](r_scalars, 0, Scalar[DT](-1.0), Scalar[DT](1.0), twr.data, 0)

    var dout = Tensor(); dout.ensure_gpu(ctx, B * DYN_OUT)
    var gdout = Tensor.alloc(B * DYN_OUT); gdout.ensure_gpu(ctx, B * DYN_OUT)
    var gdin = Tensor(); gdin.ensure_gpu(ctx, B * DYN_IN)
    var gscale = Scalar[DT](1.0) / Scalar[DT](B)

    var w0 = RewW(); dyn.for_each_param["gpu"](w0, Optional(ctx))
    print("reward-out sum|w| at init =", w0.s, "(zero-init -> 0)")

    for step in range(401):
        dyn.zero_grad["gpu"](Optional(ctx))
        dyn.forward["gpu", B](TensorRefs[Dyn.ARITY](din), dout, Optional(ctx))
        dout.download(ctx); ctx.synchronize()
        for i in range(B * DYN_OUT):
            gdout.data[i] = Scalar[DT](0)
        var rl = soft_ce_slice_loss_and_grad[B, DYN_OUT, LATENT, BINS](
            dout.data, twr.data, gscale, gdout.data
        )
        gdout.upload(ctx)
        dyn.vjp["gpu", B](
            TensorRefs[Dyn.ARITY](din), gdout, TensorRefs[Dyn.ARITY](gdin), Optional(ctx)
        )
        opt.begin_step()
        dyn.for_each_param["gpu"](opt, Optional(ctx))
        ctx.synchronize()
        if step % 50 == 0:
            print("step", step, " loss_reward(per-sample) =", rl / Scalar[DT](B))

    var w1 = RewW(); dyn.for_each_param["gpu"](w1, Optional(ctx))
    print("reward-out sum|w| after =", w1.s)
    if w1.s < Scalar[DT](1e-9):
        print(">>> REWARD HEAD GOT NO GPU UPDATE — dyn graph GPU vjp BROKEN <<<")
    else:
        print(">>> reward head updated on GPU; check loss_reward actually dropped <<<")
