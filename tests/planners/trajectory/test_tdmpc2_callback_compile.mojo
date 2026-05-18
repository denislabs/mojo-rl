"""Phase 2: TDMPC2RolloutCallback compile smoke test.

Instantiates ``TDMPC2RolloutCallback`` with tiny synthetic Sequential
networks (one Linear each) just to validate that the type wiring
compiles. Does NOT run plan_gpu — that requires real WorldModel
networks + proper Q-target buffer setup, which is task 38's
territory (agent migration).

The point: catch compile errors in the adapter's trait method bodies
before they break the production TDMPC2 build.
"""

from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.testing import assert_true
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.deep_agents.tdmpc2 import TDMPC2RolloutCallback
from mojo_rl.planners.trajectory import MPPIGPUBatched


comptime LATENT_DIM: Int = 4
comptime ACTION_DIM: Int = 2
comptime ZA_DIM: Int = LATENT_DIM + ACTION_DIM
comptime NUM_BINS: Int = 8
comptime NUM_Q: Int = 5
comptime POL_OUT: Int = 2 * ACTION_DIM
comptime MAX_BATCH: Int = 16

comptime StubDyn = Sequential[Linear[ZA_DIM, LATENT_DIM]]
comptime StubRew = Sequential[Linear[ZA_DIM, NUM_BINS]]
comptime StubPol = Sequential[Linear[LATENT_DIM, POL_OUT]]
comptime StubQ = Sequential[Linear[ZA_DIM, NUM_BINS]]

comptime AdamLR = Adam[LR=1e-3]


def test_tdmpc2_callback_constructs() raises:
    """Construct TDMPC2RolloutCallback against stub networks and
    raw (zero) parameter pointers.
    """
    var ctx = DeviceContext()

    # Synthetic param buffers — content doesn't matter for compile
    # check; the trait methods are not invoked here.
    var dyn_p = ctx.enqueue_create_buffer[dtype](StubDyn.PARAM_SIZE)
    var rew_p = ctx.enqueue_create_buffer[dtype](StubRew.PARAM_SIZE)
    var pol_p = ctx.enqueue_create_buffer[dtype](StubPol.PARAM_SIZE)
    var bins_buf = ctx.enqueue_create_buffer[dtype](NUM_BINS)

    # Each Q target needs its own param buffer for the production path;
    # for the compile smoke we share one allocation (the trait methods
    # aren't invoked).
    var q_p = ctx.enqueue_create_buffer[dtype](StubQ.PARAM_SIZE)
    var q_param_bufs = InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], NUM_Q
    ](fill=q_p.unsafe_ptr())

    var cb = TDMPC2RolloutCallback[
        StubDyn, AdamLR,
        StubRew, AdamLR,
        StubPol, AdamLR,
        StubQ, AdamLR,
        LATENT_DIM, ACTION_DIM, NUM_BINS, NUM_Q, MAX_BATCH,
    ](
        ctx,
        dyn_p.unsafe_ptr(),
        rew_p.unsafe_ptr(),
        pol_p.unsafe_ptr(),
        q_param_bufs,
        bins_buf.unsafe_ptr(),
    )

    # If we reach here the adapter constructed without crashing.
    assert_true(True, "TDMPC2RolloutCallback constructed")


def test_tdmpc2_callback_through_planner() raises:
    """End-to-end smoke: MPPIGPUBatched.plan_gpu through the
    adapter dispatches every trait method against real (stub)
    networks. Outputs are numerically garbage (uninit params), but
    the test fails if any kernel signature / shape / type mismatch
    crops up at instantiation time.
    """
    comptime HORIZON: Int = 2
    comptime NUM_SAMPLES: Int = 4
    comptime NUM_PI_TRAJS: Int = 2
    comptime NUM_ELITES: Int = 3
    comptime NUM_ITERATIONS: Int = 1
    comptime N_ENVS: Int = 1
    comptime TOTAL: Int = NUM_SAMPLES + NUM_PI_TRAJS
    comptime BATCH_TOTAL: Int = N_ENVS * TOTAL

    var ctx = DeviceContext()

    # Stub param buffers — content uninitialized but allocated.
    var dyn_p = ctx.enqueue_create_buffer[dtype](StubDyn.PARAM_SIZE)
    var rew_p = ctx.enqueue_create_buffer[dtype](StubRew.PARAM_SIZE)
    var pol_p = ctx.enqueue_create_buffer[dtype](StubPol.PARAM_SIZE)
    var q_p = ctx.enqueue_create_buffer[dtype](StubQ.PARAM_SIZE)
    var bins_buf = ctx.enqueue_create_buffer[dtype](NUM_BINS)

    var q_param_bufs = InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], NUM_Q
    ](fill=q_p.unsafe_ptr())

    var cb = TDMPC2RolloutCallback[
        StubDyn, AdamLR,
        StubRew, AdamLR,
        StubPol, AdamLR,
        StubQ, AdamLR,
        LATENT_DIM, ACTION_DIM, NUM_BINS, NUM_Q, BATCH_TOTAL,
    ](
        ctx,
        dyn_p.unsafe_ptr(),
        rew_p.unsafe_ptr(),
        pol_p.unsafe_ptr(),
        q_param_bufs,
        bins_buf.unsafe_ptr(),
    )

    var planner = MPPIGPUBatched[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ELITES,
        NUM_ITERATIONS,
        N_ENVS,
    ](ctx)

    var z0_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * LATENT_DIM)
    var out_act_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var z0_tensor = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, LATENT_DIM), MutAnyOrigin
    ](z0_buf.unsafe_ptr())
    var out_act_tensor = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACTION_DIM), MutAnyOrigin
    ](out_act_buf.unsafe_ptr())

    planner.plan_gpu(
        ctx,
        cb,
        z0_tensor,
        out_act_tensor,
        gamma=0.95,
        temperature=0.5,
        action_scale=1.0,
        deterministic=True,
        rng_base_seed=UInt32(7),
    )
    ctx.synchronize()
    assert_true(True, "MPPIGPUBatched.plan_gpu via TDMPC2 adapter completed")


def main() raises:
    print("=== Phase 2: TDMPC2RolloutCallback compile smoke ===")
    test_tdmpc2_callback_constructs()
    print("  PASS TDMPC2RolloutCallback constructs")
    test_tdmpc2_callback_through_planner()
    print(
        "  PASS MPPIGPUBatched.plan_gpu dispatches via adapter"
        " (compile + run, not parity)"
    )
    print("OK")
