"""Overfit smoke (GPU): the AlphaZero loss graph trains the prediction net on
device — validates the GPU training path on the STORAGE surface (AZLossOp GPU
fwd/bwd kernels + storage ComputeGraph GPU forward/vjp with the net threaded as
an external arg + storage Adam GPU). Mirror of the CPU overfit test. This is the
ExternalNode-on-GPU path (the wildcard-matmul-fix design) — gate on Apple here;
NVIDIA re-gate pending.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_loss_overfit_gpu.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

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

    var ctx = DeviceContext()
    var octx = Optional[DeviceContext](ctx)
    var net = Net.make["gpu", Kaiming](octx)
    var opt = Adam(lr=0.01)
    var graph = Graph.make["gpu", Zero](octx)

    # ── Fixed synthetic batch (host fill → upload) ──
    var obs = Tensor.alloc(B * OBS)
    var tgt = Tensor.alloc(B * W)
    var grad = Tensor.alloc(B)
    for b in range(B):
        for j in range(OBS):
            obs.data[b * OBS + j] = Scalar[DT](1.0 if (j % 7) == (b % 7) else 0.0)
        var peak = b % ACT
        for a in range(ACT):
            tgt.data[b * W + a] = Scalar[DT](1.0 if a == peak else 0.0)
        tgt.data[b * W + ACT] = Scalar[DT](Float64((b % 3) - 1) * 0.5)
        grad.data[b] = Scalar[DT](1.0) / Scalar[DT](B)
    obs.upload(ctx)
    tgt.upload(ctx)
    grad.upload(ctx)

    var loss = Tensor.alloc_gpu(ctx, B)

    var first_loss: Float64 = 0.0
    var last_loss: Float64 = 0.0

    for step in range(STEPS):
        net.zero_grad["gpu"](octx)
        graph.set_input["obs", B](obs, octx)
        graph.set_input["tgt", B](tgt, octx)
        graph.forward[B, "gpu"](loss, octx, net)

        if step == 0 or step == STEPS - 1:
            loss.download(ctx)
            var mean: Float64 = 0.0
            for b in range(B):
                mean += Float64(loss.data[b])
            mean /= Float64(B)
            if step == 0:
                first_loss = mean
            else:
                last_loss = mean

        graph.vjp[B, "gpu"](grad, octx, net)
        opt.begin_step()
        net.for_each_param["gpu"](opt, octx)

    print("AZ overfit GPU: first_loss=", first_loss, " last_loss=", last_loss)
    assert_true(
        last_loss < 0.2,
        "AZ GPU loss did not overfit (last=" + String(last_loss) + ")",
    )
    _ = net^
    print("AZ loss overfit GPU: OK")
