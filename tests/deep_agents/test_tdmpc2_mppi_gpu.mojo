"""TD-MPC2 GPU batched MPPI — nn world model ↔ MPPIGPUBatched (Apple Metal).

Builds a GPU world model + TDMPC2RolloutCallbackGPU + the batched planner,
runs one plan_gpu for N_ENVS, and checks the selected actions are finite and
in [-action_scale, action_scale]. Validates the GPU callback bridge (batched
policy/dynamics/reward/Q forwards + build-za / extract-mean / decode / avg-2
kernels + Philox Q-pair). On Apple this is launch-bound (slow) but correct;
the practical fast path is NVIDIA.

Run: `pixi run -e apple mojo run -I . tests/deep_agents/test_tdmpc2_mppi_gpu.mojo`
"""

from std.random import seed
from std.math import isfinite
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.planners.trajectory.mppi import MPPIGPUBatched
from mojo_rl.deep_agents.tdmpc2.nets import (
    TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy,
)
from mojo_rl.deep_agents.tdmpc2.callback import TDMPC2RolloutCallbackGPU

comptime ACT = 2
comptime LATENT = 16
comptime MLP = 16
comptime BINS = 11
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime NUM_Q = 2
# small batched planning config
comptime HORIZON = 3
comptime NUM_SAMPLES = 16
comptime NUM_PI_TRAJS = 4
comptime NUM_ELITES = 8
comptime NUM_ITERS = 2
comptime N_ENVS = 2
comptime BT = N_ENVS * (NUM_SAMPLES + NUM_PI_TRAJS)   # BATCH_TOTAL


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def test_mppi_gpu_plan() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime DynT = TDMPC2Dynamics[LATENT, ACT, MLP, SN]
    comptime RewT = TDMPC2Reward[LATENT, ACT, MLP, BINS]
    comptime QNetT = TDMPC2QNet[LATENT, ACT, MLP, BINS]
    comptime PolicyT = TDMPC2Policy[LATENT, ACT, MLP]
    comptime CB = TDMPC2RolloutCallbackGPU[
        ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, NUM_Q, BT
    ]
    comptime Planner = MPPIGPUBatched[
        LATENT, ACT, HORIZON, NUM_SAMPLES, NUM_PI_TRAJS, NUM_ELITES,
        NUM_ITERS, N_ENVS,
    ]

    var dyn = DynT.make["gpu", INIT=Kaiming](ctx=ctx)
    var rew = RewT.make["gpu", INIT=Kaiming](ctx=ctx)
    var pol = PolicyT.make["gpu", INIT=Kaiming](ctx=ctx)
    var qt = List[QNetT]()
    qt.append(QNetT.make["gpu", INIT=Kaiming](ctx=ctx))
    qt.append(QNetT.make["gpu", INIT=Kaiming](ctx=ctx))

    var action_scale = Scalar[DT](1.0)
    var cb = CB.make(dyn, rew, pol, qt, action_scale, ctx)
    var planner = Planner(ctx)

    # z0 [N_ENVS, LATENT] + out_act [N_ENVS*ACT] device buffers.
    var d_z0 = ctx.enqueue_create_buffer[DT](N_ENVS * LATENT)
    var h_z0 = ctx.enqueue_create_host_buffer[DT](N_ENVS * LATENT)
    ctx.synchronize()
    for i in range(N_ENVS * LATENT):
        h_z0.unsafe_ptr()[i] = Scalar[DT](0.1 * Float64(i % 7) - 0.3)
    ctx.enqueue_copy(d_z0, h_z0)
    ctx.synchronize()
    var d_out = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)

    var z0_t = LayoutTensor[DT, Layout.row_major(N_ENVS, LATENT), MutAnyOrigin](
        _p(d_z0)
    )
    var out_t = LayoutTensor[DT, Layout.row_major(N_ENVS * ACT), MutAnyOrigin](
        _p(d_out)
    )

    planner.plan_gpu[CB](
        ctx, cb, z0_t, out_t, gamma=0.99, temperature=0.5,
        action_scale=1.0, deterministic=True,
    )

    var h_out = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()
    for i in range(N_ENVS * ACT):
        var v = h_out.unsafe_ptr()[i]
        assert_true(isfinite(v), "action finite")
        assert_true(v >= -1.0001 and v <= 1.0001, "action in [-scale, scale]")
    print("  GPU MPPI actions:", h_out.unsafe_ptr()[0], h_out.unsafe_ptr()[1])

    _ = dyn^
    _ = rew^
    _ = pol^
    _ = qt^
    _ = cb^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
