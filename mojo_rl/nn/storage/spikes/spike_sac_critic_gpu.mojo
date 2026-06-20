"""SAC critic step on GPU — validates GPU ComputeGraph execution.

Same critic as spike_sac_critic but on GPU: exercises device pool-seeding
(enqueue_copy), GPU node forward/vjp, the device grad-accumulate kernel, and
Adam-on-GPU driven through for_each_param. Fits q(s,a) → target_y; loss falls.

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spikes/spike_sac_critic_gpu.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.primitives.concat import Concat2
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.combinators.graph_decl import InputSlot, Node
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.loss.mse_loss import MSELoss
from mojo_rl.nn.storage.core.initializer import Deterministic


def main() raises:
    comptime B = 8
    comptime S = 4
    comptime A = 2
    comptime H = 32
    comptime SA = S + A
    var c = DeviceContext()

    var critic = ComputeGraph[
        InputSlot["s", S],
        InputSlot["a", A],
        Node["concat", Concat2[S, A], "s", "a"],
        Node["l1", Linear[SA, H], "concat"],
        Node["relu", ReLU[H], "l1"],
        Node["q", Linear[H, 1], "relu"],
    ].make["gpu", Deterministic](Optional(c))

    var s = Tensor.alloc(B * S)
    var a = Tensor.alloc(B * A)
    var target_y = Tensor.alloc(B * 1)
    for i in range(B * S):
        s.data[i] = Scalar[DT]((i % 7) - 3) * 0.25
    for i in range(B * A):
        a.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for b in range(B):
        target_y.data[b] = Scalar[DT]((b % 4) - 2) * 0.5
    s.upload(c)
    a.upload(c)
    target_y.upload(c)
    critic.set_input["s", B](s, Optional(c))
    critic.set_input["a", B](a, Optional(c))

    var mse = MSELoss[1].make_gpu(c)
    var opt = Adam(lr=0.01)
    var q = Tensor.alloc(B * 1)
    var grad_q = Tensor.alloc(B * 1)

    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(120):
        critic.zero_grad["gpu"](Optional(c))
        critic.forward[B, "gpu"](q, Optional(c))
        var loss = mse.forward["gpu", B](q, target_y, Optional(c))
        if step == 0:
            first = loss
        last = loss
        if step % 24 == 0:
            print("step", step, " critic_loss", loss)
        mse.vjp["gpu", B](q, target_y, grad_q, Optional(c))
        critic.vjp[B, "gpu"](grad_q, Optional(c))
        opt.begin_step()
        critic.for_each_param["gpu"](opt, Optional(c))

    print("final critic_loss", last)
    if last < first * 0.05:
        print("SAC CRITIC GPU OK — q fits target_y; loss", first, "->", last)
    else:
        print("SAC CRITIC GPU WEAK (", first, "->", last, ")")
