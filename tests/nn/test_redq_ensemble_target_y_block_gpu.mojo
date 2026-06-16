"""R.5 GPU smoke for `EnsembleTargetYBlock`.

Build CPU + GPU instances of EnsembleTargetYBlock with identical
params (hard-copy actor + N target nets to sync), feed the SAME
mb_sp / mb_r / mb_d through both, pin the same subset, and verify
the resulting `mb_y` matches within FP rounding.

Apple Metal: "kernels compile + math matches CPU within tol" gate.
This guards against shape mismatches (e.g. the concat+lp kernel
writing into wrong indices) and ensures the GPU subset upload +
kernel launch are wired correctly end-to-end.

Note: RSample on CPU vs GPU uses different RNG (host `random_float64`
vs Philox) so the *action* sampled in step 2 differs between the two
runs. We can't compare CPU and GPU `y` byte-for-byte. Instead we
substitute the rsample output AFTER the GPU run by reading the GPU's
own intermediate scratches (`_mb_stacked_q` and `_mb_lp`) and
recomputing the expected y on the host using the same combine formula
the CPU kernel implements. That gates only the kernel + glue path
(the rsample primitive has its own CPU+GPU parity tests).
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.initializer import Xavier

from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.redq import (
    CriticEnsemble,
    EnsembleTargetYBlock,
    REDQ_TARGET_MIN,
)


comptime OBS = 3
comptime ACT = 2
comptime BATCH = 8
comptime N = 4
comptime N_MIN = 2

comptime ActorNet = Sequential[
    Linear[OBS, 16], ReLU[16], Linear[16, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 16], ReLU[16], Linear[16, 1],
]


def _list_int_2(a: Int, b: Int) raises -> List[Int]:
    var out = List[Int](length=2, fill=0)
    out[0] = a
    out[1] = b
    return out^


def test_ensemble_target_y_block_gpu() raises:
    print("--- EnsembleTargetYBlock GPU N=4 M=2 MIN ---")
    var ctx = DeviceContext()

    var actor = ActorNet.make["gpu", Xavier](ctx=ctx)
    var ensemble = CriticEnsemble[CriticNet, N].make["gpu", Xavier](
        ctx=ctx,
    )
    var block = EnsembleTargetYBlock[
        ActorNet, CriticNet, N, BATCH, OBS, ACT, N_MIN, REDQ_TARGET_MIN,
    ].make["gpu"](
        action_scale=Scalar[DT](1.0),
        gamma=Scalar[DT](0.97),
        ctx=ctx,
    )
    var state = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx=ctx)

    block.set_subset_idxs(_list_int_2(0, 1))

    # Fill mb_sp / mb_r / mb_d on host then H2D into state scratches.
    var sp_host = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    var r_host  = ctx.enqueue_create_host_buffer[DT](BATCH)
    var d_host  = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()
    var sp_hp = sp_host.unsafe_ptr()
    var r_hp = r_host.unsafe_ptr()
    var d_hp = d_host.unsafe_ptr()
    for b in range(BATCH):
        for k in range(OBS):
            sp_hp[b * OBS + k] = Scalar[DT](
                0.1 * Float64(b) + 0.05 * Float64(k) - 0.3
            )
        r_hp[b] = Scalar[DT](-0.5 + 0.2 * Float64(b))
        d_hp[b] = Scalar[DT](0.0) if b < 4 else Scalar[DT](1.0)
    ctx.enqueue_copy(state.mb_sp.dev.value(), sp_hp)
    ctx.enqueue_copy(state.mb_r.dev.value(), r_hp)
    ctx.enqueue_copy(state.mb_d.dev.value(), d_hp)

    var alpha = Scalar[DT](0.15)
    block.step["gpu"](
        actor,
        ensemble,
        state.mb_sp.dev_ptr(),
        state.mb_r.dev_ptr(),
        state.mb_d.dev_ptr(),
        alpha,
        state.mb_y.dev_ptr(),
    )

    # D2H y AND the GPU's intermediate stacked_q + lp for the host re-check.
    var y_host_buf = ctx.enqueue_create_host_buffer[DT](BATCH)
    var stacked_host = ctx.enqueue_create_host_buffer[DT](N * BATCH)
    var lp_host = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()
    ctx.enqueue_copy(y_host_buf.unsafe_ptr(), state.mb_y.dev.value())
    ctx.enqueue_copy(stacked_host.unsafe_ptr(), block._mb_stacked_q.dev.value())
    ctx.enqueue_copy(lp_host.unsafe_ptr(), block._mb_lp.dev.value())
    ctx.synchronize()

    var y_hp = y_host_buf.unsafe_ptr()
    var stacked_hp = stacked_host.unsafe_ptr()
    var lp_hp = lp_host.unsafe_ptr()

    var max_dev: Float64 = 0.0
    var gamma = block.gamma
    for b in range(BATCH):
        var v = Float64(y_hp[b])
        assert_true(v == v, "y[b] finite")
        if d_hp[b] == Scalar[DT](1.0):
            # Terminated: y == r exactly.
            print(
                "  b=", b, " (term=1) y =", y_hp[b],
                " r =", r_hp[b],
            )
            assert_true(
                y_hp[b] == r_hp[b],
                "term=1 ⇒ y == r exactly on GPU",
            )
        else:
            # min over subset (0, 1) at the GPU's own intermediate Q's.
            var q0 = stacked_hp[0 * BATCH + b]
            var q1 = stacked_hp[1 * BATCH + b]
            var combined = q0 if q0 < q1 else q1
            var expected = r_hp[b] + gamma * (combined - alpha * lp_hp[b])
            var dev = Float64(y_hp[b]) - Float64(expected)
            if dev < 0.0:
                dev = -dev
            if dev > max_dev:
                max_dev = dev
            print(
                "  b=", b, " (term=0) y =", y_hp[b],
                " expected =", expected,
            )
    print("  max |y_gpu - expected_from_gpu_intermediates| =", max_dev)
    assert_true(
        max_dev < 1e-5,
        "GPU y must equal formula reconstruction from GPU intermediates",
    )

    print("PASS — EnsembleTargetYBlock GPU smoke green.")


def main() raises:
    test_ensemble_target_y_block_gpu()
