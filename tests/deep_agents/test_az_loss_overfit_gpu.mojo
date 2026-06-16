"""Overfit smoke (GPU): the AlphaZero loss graph trains the prediction net on
device — validates the GPU training path (AZLossOp GPU fwd/bwd kernels + graph
GPU forward/vjp + Adam GPU) end-to-end. Mirror of the CPU overfit test.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_loss_overfit_gpu.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext
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

    var ctx = DeviceContext()
    var net = Net.make["gpu", INIT=Kaiming](ctx=ctx)
    var opt = Adam.make["gpu", M=Net](net, ctx)
    opt.lr = 0.01
    var graph = Graph.make["gpu", INIT=Zero](ctx=ctx)

    # ── Fixed synthetic batch (host → device) ──
    var obs_dev = ctx.enqueue_create_buffer[DT](B * OBS)
    var tgt_dev = ctx.enqueue_create_buffer[DT](B * W)
    var grad_dev = ctx.enqueue_create_buffer[DT](B)
    var loss_dev = ctx.enqueue_create_buffer[DT](B)
    var obs_h = ctx.enqueue_create_host_buffer[DT](B * OBS)
    var tgt_h = ctx.enqueue_create_host_buffer[DT](B * W)
    var grad_h = ctx.enqueue_create_host_buffer[DT](B)
    ctx.synchronize()

    for b in range(B):
        for j in range(OBS):
            obs_h.unsafe_ptr()[b * OBS + j] = Scalar[DT](
                1.0 if (j % 7) == (b % 7) else 0.0
            )
        var peak = b % ACT
        for a in range(ACT):
            tgt_h.unsafe_ptr()[b * W + a] = Scalar[DT](
                1.0 if a == peak else 0.0
            )
        tgt_h.unsafe_ptr()[b * W + ACT] = Scalar[DT](
            Float64((b % 3) - 1) * 0.5
        )
        grad_h.unsafe_ptr()[b] = Scalar[DT](1.0) / Scalar[DT](B)
    ctx.enqueue_copy(obs_dev, obs_h)
    ctx.enqueue_copy(tgt_dev, tgt_h)
    ctx.enqueue_copy(grad_dev, grad_h)
    ctx.synchronize()

    var obs_t = TileTensor(obs_dev, row_major[B, OBS]())
    var tgt_t = TileTensor(tgt_dev, row_major[B, W]())
    var loss_t = TileTensor(loss_dev, row_major[B, 1]())
    var grad_t = TileTensor(grad_dev, row_major[B, 1]())

    var first_loss: Float64 = 0.0
    var last_loss: Float64 = 0.0
    var loss_h = ctx.enqueue_create_host_buffer[DT](B)
    ctx.synchronize()

    for step in range(STEPS):
        opt.zero_grad["gpu", M=Net](net)
        graph.set_external["pred", Net](net)
        graph.set_input["obs", B](obs_t)
        graph.set_input["tgt", B](tgt_t)
        graph.forward["gpu", B](loss_t)

        if step == 0 or step == STEPS - 1:
            ctx.enqueue_copy(loss_h, loss_dev)
            ctx.synchronize()
            var mean: Float64 = 0.0
            for b in range(B):
                mean += Float64(loss_h.unsafe_ptr()[b])
            mean /= Float64(B)
            if step == 0:
                first_loss = mean
            else:
                last_loss = mean

        graph.vjp["gpu", B](grad_t)
        opt.step["gpu", M=Net](net)

    print("AZ overfit GPU: first_loss=", first_loss, " last_loss=", last_loss)
    assert_true(last_loss < 0.2, "AZ GPU loss did not overfit (last=" + String(last_loss) + ")")
    _ = net^
    print("AZ loss overfit GPU: OK")
