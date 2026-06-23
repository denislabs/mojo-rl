"""Does MZDynNetC4Spatial's REWARD head learn a clean target?

The before/after log comparison shows loss_reward learns pre-migration (1.3→0.4)
but is STUCK post-migration (~0.95) — and that gates the whole bootstrap (no
model reward signal → no promotions → dead policy). The reward head lives in the
dyn ComputeGraph (node `rew`, reading the next latent `zp`). This isolates it:
fixed [z|onehot(a)] inputs, a distinct peaked reward target per sample, seed ONLY
the reward slice of grad_output, run vjp+Adam, and watch (a) loss_reward drop and
(b) the reward output weight (graph.rew.4.weight) leave zero-init.

  loss_reward drops + rew weight grows  → reward path OK; degradation is upstream
                                          (dynamics latent / data)
  loss_reward flat + rew weight ~0       → dyn ComputeGraph reward backprop BROKEN
                                          (the migration bug)
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.muzero.nets_spatial import MZDynNetC4Spatial
from mojo_rl.deep_agents.muzero.loss_ops import soft_ce_slice_loss_and_grad
from mojo_rl.deep_agents.zero.twohot_targets import mz_two_hot_target_batch
from std.gpu.host import DeviceContext


struct RewW(ParamVisitor):
    var s: Scalar[DT]

    def __init__(out self):
        self.s = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        # reward output Linear (InitWith is name-transparent → '...rew.4.weight')
        if name.endswith("rew.4.weight"):
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
    var dyn = Dyn.make["cpu", INIT=Kaiming]()
    var opt = Adam(lr=Scalar[DT](2e-3))

    var v_min = Scalar[DT](-1.0)
    var v_max = Scalar[DT](1.0)

    # fixed inputs: distinct latent per sample + onehot(action = b % ACT)
    var din = Tensor.alloc(B * DYN_IN)
    for b in range(B):
        var base = b * DYN_IN
        for i in range(LATENT):
            din.data[base + i] = Scalar[DT](0.05) * Scalar[DT](((b * 7 + i) % 11) - 5)
        for a in range(ACT):
            din.data[base + LATENT + a] = Scalar[DT](0.0)
        din.data[base + LATENT + (b % ACT)] = Scalar[DT](1.0)

    # distinct reward target per sample r_b in {-1, 0, 1}; two-hot encode
    var r_scalars = List[Scalar[DT]](length=B, fill=0)
    for b in range(B):
        r_scalars[b] = Scalar[DT]((b % 3) - 1)
    var twr = Tensor.alloc(B * BINS)
    mz_two_hot_target_batch[B, BINS](r_scalars, 0, v_min, v_max, twr.data, 0)

    var dout = Tensor.alloc(B * DYN_OUT)
    var gdout = Tensor.alloc(B * DYN_OUT)
    var gdin = Tensor.alloc(B * DYN_IN)
    var gscale = Scalar[DT](1.0) / Scalar[DT](B)

    var w0 = RewW(); dyn.for_each_param["cpu"](w0, None)
    print("reward-out sum|w| at init =", w0.s, "(zero-init → 0)")

    for step in range(401):
        dyn.zero_grad["cpu"](None)
        dyn.forward["cpu", B](TensorRefs[Dyn.ARITY](din), dout, None)
        # zero the carry (latent) grad slice; seed ONLY the reward slice
        for i in range(B * DYN_OUT):
            gdout.data[i] = Scalar[DT](0)
        var rl = soft_ce_slice_loss_and_grad[B, DYN_OUT, LATENT, BINS](
            dout.data, twr.data, gscale, gdout.data
        )
        dyn.vjp["cpu", B](
            TensorRefs[Dyn.ARITY](din), gdout, TensorRefs[Dyn.ARITY](gdin), None
        )
        opt.begin_step()
        dyn.for_each_param["cpu"](opt, None)
        if step % 50 == 0:
            print("step", step, " loss_reward(per-sample) =", rl / Scalar[DT](B))

    var w1 = RewW(); dyn.for_each_param["cpu"](w1, None)
    print("reward-out sum|w| after =", w1.s)
    if w1.s < Scalar[DT](1e-9):
        print(">>> REWARD HEAD GOT NO UPDATE — dyn graph reward backprop BROKEN <<<")
    else:
        print(">>> reward head updated; check whether loss_reward actually dropped <<<")
