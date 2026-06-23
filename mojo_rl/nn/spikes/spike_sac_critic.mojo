"""SAC critic step — first agent-block assembly gate (CPU).

Critic = ComputeGraph[2, Concat2(state,action), Linear, ReLU, Linear(q)] +
MSELoss + Adam. Trains q(s,a) → target_y on a fixed batch; loss must fall.
Proves the full SAC critic training loop assembles from existing storage pieces:
multi-input graph (concat of state+action), MLP, the SAC-convention MSELoss, and
the Adam optimizer driven through for_each_param.

  slots: 0=state 1=action | 2=Concat2 3=Linear 4=ReLU 5=Linear(q)

Run: pixi run mojo run -I . mojo_rl/nn/storage/spikes/spike_sac_critic.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.concat import Concat2
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import InputSlot, Node
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.loss.mse_loss import MSELoss
from mojo_rl.nn.core.initializer import Deterministic


def main() raises:
    comptime B = 8
    comptime S = 4       # state dim
    comptime A = 2       # action dim
    comptime H = 32      # critic hidden
    comptime SA = S + A

    var critic = ComputeGraph[
        InputSlot["s", S],
        InputSlot["a", A],
        Node["concat", Concat2[S, A], "s", "a"],   # Concat2(state, action)
        Node["l1", Linear[SA, H], "concat"],       # Linear(concat)
        Node["relu", ReLU[H], "l1"],               # ReLU
        Node["q", Linear[H, 1], "relu"],           # Linear → q
    ].make["cpu", Deterministic]()

    var s = Tensor.alloc(B * S)
    var a = Tensor.alloc(B * A)
    var target_y = Tensor.alloc(B * 1)
    for i in range(B * S):
        s.data[i] = Scalar[DT]((i % 7) - 3) * 0.25
    for i in range(B * A):
        a.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for b in range(B):
        target_y.data[b] = Scalar[DT]((b % 4) - 2) * 0.5   # fixed regression target
    critic.set_input["s", B](s)
    critic.set_input["a", B](a)

    var mse = MSELoss[1].make_cpu()
    var opt = Adam(lr=0.01)
    var q = Tensor.alloc(B * 1)
    var grad_q = Tensor.alloc(B * 1)

    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(120):
        critic.zero_grad["cpu"](None)
        critic.forward[B](q)
        var loss = mse.forward["cpu", B](q, target_y, None)
        if step == 0:
            first = loss
        last = loss
        if step % 24 == 0:
            print("step", step, " critic_loss", loss)
        mse.vjp["cpu", B](q, target_y, grad_q, None)
        critic.vjp[B](grad_q)
        opt.begin_step()
        critic.for_each_param["cpu"](opt, None)

    print("final critic_loss", last)
    if last < first * 0.05:
        print("SAC CRITIC STEP OK — q fits target_y; loss", first, "->", last)
    else:
        print("SAC CRITIC STEP WEAK (", first, "->", last, ")")
