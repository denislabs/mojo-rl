"""Overfit smoke (CPU): the AlphaZero loss graph trains the prediction net to
fit a fixed synthetic batch of (obs, [mcts_policy | z]) targets.

Validates the training-side machinery end-to-end on CPU on the STORAGE surface:
net (ExternalNode) + ``AZLossOp`` node compose into a storage ComputeGraph whose
forward computes the AZ loss and whose vjp routes gradient into the net's params
(net threaded as a `forward`/`vjp` external arg), driven by storage Adam
(`begin_step` + `for_each_param`). Asserts the mean loss collapses (overfit).

Run:
    pixi run mojo run -I . tests/deep_agents/test_az_loss_overfit_cpu.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.initializer import Kaiming, Zero
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import InputSlot, Node, ExternalNode

from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.loss_ops import AZLossOp


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime B = 8
    comptime W = ACT + 1
    comptime STEPS = 500
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Graph = ComputeGraph[
        InputSlot["obs", OBS],
        ExternalNode["pred", Net, "obs"],
        InputSlot["tgt", W],
        Node["loss", AZLossOp[ACT], "pred", "tgt"],
    ]

    var net = Net.make["cpu", Kaiming]()
    var opt = Adam(lr=0.01)
    var graph = Graph.make["cpu", Zero]()

    # ── Fixed synthetic batch ──
    var obs = Tensor.alloc(B * OBS)
    var tgt = Tensor.alloc(B * W)
    for b in range(B):
        # pseudo-board obs: a couple of cells set, varying per row
        for j in range(OBS):
            obs.data[b * OBS + j] = Scalar[DT](
                1.0 if (j % 7) == (b % 7) else 0.0
            )
        # one-hot target policy at action (b % ACT). One-hot ⇒ target entropy
        # 0, so the soft-CE floor is 0 and a successful overfit drives the
        # total loss to ~0.
        var peak = b % ACT
        for a in range(ACT):
            tgt.data[b * W + a] = Scalar[DT](1.0 if a == peak else 0.0)
        # value target z in (-1, 1)
        tgt.data[b * W + ACT] = Scalar[DT](Float64((b % 3) - 1) * 0.5)

    var loss = Tensor.alloc(B)
    var grad = Tensor.alloc(B)

    var first_loss: Scalar[DT] = 0.0
    var last_loss: Scalar[DT] = 0.0

    for step in range(STEPS):
        net.zero_grad["cpu"](None)
        graph.set_input["obs", B](obs, None)
        graph.set_input["tgt", B](tgt, None)
        graph.forward[B, "cpu"](loss, None, net)

        var mean: Scalar[DT] = 0.0
        for b in range(B):
            mean += loss.data[b]
        mean /= Scalar[DT](B)
        if step == 0:
            first_loss = mean
        last_loss = mean

        for b in range(B):
            grad.data[b] = Scalar[DT](1.0) / Scalar[DT](B)
        graph.vjp[B, "cpu"](grad, None, net)
        opt.begin_step()
        net.for_each_param["cpu"](opt, None)

    print("AZ overfit: first_loss=", first_loss, " last_loss=", last_loss)
    # One-hot targets ⇒ irreducible loss floor is 0; a working forward+autodiff+
    # optimizer drives the mean loss well below 0.2 (from ~2.8).
    assert_true(
        Float64(last_loss) < 0.2,
        "AZ loss did not overfit the fixed batch (last_loss="
        + String(last_loss) + ")",
    )

    _ = net^
    print("AZ loss overfit CPU: OK")
