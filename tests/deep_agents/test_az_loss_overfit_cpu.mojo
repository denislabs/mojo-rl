"""Overfit smoke (CPU): the AlphaZero loss graph trains the prediction net to
fit a fixed synthetic batch of (obs, [mcts_policy | z]) targets.

Validates the training-side machinery end-to-end on CPU: net (ExternalNode) +
``AZLossOp`` node compose into a ComputeGraph whose forward computes the AZ loss
and whose vjp routes gradient into the net's params, driven by Adam. Asserts the
mean loss collapses (overfit) — i.e. forward + autodiff + optimizer are wired
correctly. GPU parity lands with the trainer.

Run:
    pixi run mojo run -I . tests/deep_agents/test_az_loss_overfit_cpu.mojo
"""

from std.memory import alloc
from std.math import exp
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming, Zero
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_nodes import InputSlot, Node, ExternalNode

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
        1,
        InputSlot["obs", OBS],
        ExternalNode["pred", Net, "obs"],
        InputSlot["tgt", W],
        Node["loss", AZLossOp[ACT], "pred", "tgt"],
    ]

    var net = Net.make["cpu", INIT=Kaiming]()
    var opt = Adam.make["cpu", M=Net](net)
    opt.lr = 0.01
    var graph = Graph.make["cpu", INIT=Zero]()

    # ── Fixed synthetic batch ──
    var obs = alloc[Scalar[DT]](B * OBS)
    var tgt = alloc[Scalar[DT]](B * W)
    for b in range(B):
        # pseudo-board obs: a couple of cells set, varying per row
        for j in range(OBS):
            obs[b * OBS + j] = Scalar[DT](
                1.0 if (j % 7) == (b % 7) else 0.0
            )
        # one-hot target policy at action (b % ACT). One-hot ⇒ target entropy
        # 0, so the soft-CE floor is 0 and a successful overfit drives the
        # total loss to ~0 (soft-CE against a *soft* target would floor at its
        # entropy instead).
        var peak = b % ACT
        for a in range(ACT):
            tgt[b * W + a] = Scalar[DT](1.0 if a == peak else 0.0)
        # value target z in (-1, 1)
        tgt[b * W + ACT] = Scalar[DT](Float64((b % 3) - 1) * 0.5)

    var obs_t = TileTensor(obs, row_major[B, OBS]())
    var tgt_t = TileTensor(tgt, row_major[B, W]())

    var loss_buf = alloc[Scalar[DT]](B)
    var grad_buf = alloc[Scalar[DT]](B)
    var loss_t = TileTensor(loss_buf, row_major[B, 1]())
    var grad_t = TileTensor(grad_buf, row_major[B, 1]())

    var first_loss: Scalar[DT] = 0.0
    var last_loss: Scalar[DT] = 0.0

    for step in range(STEPS):
        opt.zero_grad["cpu", M=Net](net)
        graph.set_external["pred", Net](net)
        graph.set_input["obs", B](obs_t)
        graph.set_input["tgt", B](tgt_t)
        graph.forward["cpu", B](loss_t)

        var mean: Scalar[DT] = 0.0
        for b in range(B):
            mean += loss_buf[b]
        mean /= Scalar[DT](B)
        if step == 0:
            first_loss = mean
        last_loss = mean

        for b in range(B):
            grad_buf[b] = Scalar[DT](1.0) / Scalar[DT](B)
        graph.vjp["cpu", B](grad_t)
        opt.step["cpu", M=Net](net)

    print("AZ overfit: first_loss=", first_loss, " last_loss=", last_loss)
    # One-hot targets ⇒ irreducible loss floor is 0; a working forward+autodiff+
    # optimizer drives the mean loss well below 0.2 (from ~2.8).
    assert_true(
        Float64(last_loss) < 0.2,
        "AZ loss did not overfit the fixed batch (last_loss="
        + String(last_loss) + ")",
    )

    obs.free()
    tgt.free()
    loss_buf.free()
    grad_buf.free()
    _ = net^
    print("AZ loss overfit CPU: OK")
