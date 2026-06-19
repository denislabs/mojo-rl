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
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.primitives.concat import Concat2
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.loss.mse_loss import MSELoss
from mojo_rl.nn.storage.core.initializer import Deterministic


def main() raises:
    comptime B = 8
    comptime S = 4       # state dim
    comptime A = 2       # action dim
    comptime H = 32      # critic hidden
    comptime SA = S + A

    var critic = ComputeGraph[
        2, Concat2[S, A], Linear[SA, H], ReLU[H], Linear[H, 1]
    ].make["cpu", Deterministic]()
    var edges = List[List[Int]]()
    edges.append([0, 1])   # Concat2(state, action)
    edges.append([2])      # Linear(concat)
    edges.append([3])      # ReLU
    edges.append([4])      # Linear → q

    var inp = TensorPack[2]()
    inp[0].ensure(B * S)
    inp[1].ensure(B * A)
    var target_y = Tensor.alloc(B * 1)
    for i in range(B * S):
        inp[0].data[i] = Scalar[DT]((i % 7) - 3) * 0.25
    for i in range(B * A):
        inp[1].data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for b in range(B):
        target_y.data[b] = Scalar[DT]((b % 4) - 2) * 0.5   # fixed regression target

    var mse = MSELoss[1].make_cpu()
    var opt = Adam(lr=0.01)
    var q = Tensor.alloc(B * 1)
    var grad_q = Tensor.alloc(B * 1)
    var gin = TensorPack[2]()

    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(120):
        critic.zero_grad["cpu"](None)
        critic.forward[B](edges, inp, q)
        var loss = mse.forward["cpu", B](q, target_y, None)
        if step == 0:
            first = loss
        last = loss
        if step % 24 == 0:
            print("step", step, " critic_loss", loss)
        mse.vjp["cpu", B](q, target_y, grad_q, None)
        critic.vjp[B](edges, grad_q, gin)
        opt.begin_step()
        critic.for_each_param["cpu"](opt, None)

    print("final critic_loss", last)
    if last < first * 0.05:
        print("SAC CRITIC STEP OK — q fits target_y; loss", first, "->", last)
    else:
        print("SAC CRITIC STEP WEAK (", first, "->", last, ")")
