"""Smoke: nn2 AlphaZero net wrapped in the AZPredGPU adapter runs a batched
GPU prediction through the planner's PredictionGPU trait surface.

Validates the core Phase-A integration seam: an nn2 ``Module`` (self-contained
GPU params) satisfies ``planners.tree_search.PredictionGPU`` and the
``dtype``↔``DT`` LayoutTensor/TileTensor bridge works end-to-end on device.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents2/test_az_pred_adapter_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.deep_agents2.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents2.zero.mcts_adapters import AZPredGPU


def main() raises:
    comptime OBS = 27   # TicTacToe canonical board
    comptime ACT = 9
    comptime H = 64
    comptime B = 4
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime PRED_OUT = ACT + 1

    var ctx = DeviceContext()
    var net = Net.make["gpu", INIT=Kaiming](ctx=ctx)
    var adapter = AZPredGPU[OBS, ACT, Net].make(net)

    # Input hidden = obs (B, OBS); fill with a constant on host, copy to device.
    var hidden_dev = ctx.enqueue_create_buffer[dtype](B * OBS)
    var pred_dev = ctx.enqueue_create_buffer[dtype](B * PRED_OUT)
    var h_host = ctx.enqueue_create_host_buffer[dtype](B * OBS)
    ctx.synchronize()
    for i in range(B * OBS):
        h_host.unsafe_ptr()[i] = Scalar[dtype](0.1)
    ctx.enqueue_copy(hidden_dev, h_host)
    ctx.synchronize()

    var hidden_lt = LayoutTensor[
        dtype, Layout.row_major(B, OBS), MutAnyOrigin
    ](hidden_dev.unsafe_ptr())
    var pred_lt = LayoutTensor[
        dtype, Layout.row_major(B, PRED_OUT), MutAnyOrigin
    ](pred_dev.unsafe_ptr())

    adapter.predict_gpu[B](ctx, hidden_lt, pred_lt)

    var p_host = ctx.enqueue_create_host_buffer[dtype](B * PRED_OUT)
    ctx.enqueue_copy(p_host, pred_dev)
    ctx.synchronize()

    var all_finite = True
    for i in range(B * PRED_OUT):
        var v = Float64(p_host.unsafe_ptr()[i])
        if v != v:  # NaN
            all_finite = False
    assert_true(all_finite, "AZ prediction output contained NaN")

    _ = net^  # keepalive: adapter holds a non-owning pointer into net
    print("AZ pred adapter smoke: OK (B=", B, " PRED_OUT=", PRED_OUT, ")")
