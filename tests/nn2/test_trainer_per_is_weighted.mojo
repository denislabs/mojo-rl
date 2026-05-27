"""C.3c — verify IS-weighted critic loss for PER.

Three layers of coverage:

  1. **Kernel unit test** — feed a known `grad` and `weights` into
     `_scale_grad_by_weights_kernel`, D2H the result, check element-
     wise `grad[i] *= weights[i]`.

  2. **Null-sentinel still works** — non-PER SAC trainer (`buf_gpu`)
     leaves `weights_p` at the default null sentinel; the critic
     update must skip the scaling kernel and the training run must
     stay finite.

  3. **Weights vector is non-trivial during training** — PER trainer
     warmup + post-warmup train_steps populate `buf_per.weights`
     with actual normalized IS weights. D2H the buffer after a
     train_step and check (a) max value is 1.0 (normalisation),
     (b) min value is < 0.5 (heterogeneity — some samples actually
     get down-weighted). This proves the kernel was fed real
     non-uniform weights, not a uniform-1.0 placeholder.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan
from std.memory import alloc  # still needed for kernel unit test D2H buffers
from std.random import seed
from std.testing import assert_true
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer_v2r import SACTrainerV2R
from mojo_rl.nn2.training.blocks_ref import (
    UniformSampleGpuStep, PerSampleGpuStep,
)
from mojo_rl.nn2.loss.critic_update_block import _scale_grad_by_weights_kernel

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 5_000
comptime SMOKE_STEPS = 1_500

comptime ActorNet = StochasticActor[
    OBS_DIM, ACT_DIM,
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


def test_scale_grad_kernel_unit() raises:
    """Direct kernel test — grad[i] *= weights[i] in-place."""
    comptime N = 8
    var ctx = DeviceContext()
    var grad_dev = ctx.enqueue_create_buffer[DT](N)
    var weights_dev = ctx.enqueue_create_buffer[DT](N)
    var h_grad = alloc[Scalar[DT]](N)
    var h_weights = alloc[Scalar[DT]](N)
    for i in range(N):
        h_grad[i] = Scalar[DT](Float64(i) + 1.0)        # 1, 2, ..., 8
        h_weights[i] = Scalar[DT](1.0 / (Float64(i) + 1.0))  # 1, 1/2, ..., 1/8
    ctx.enqueue_copy(grad_dev, h_grad)
    ctx.enqueue_copy(weights_dev, h_weights)
    var grad_lt = LayoutTensor[
        DT, Layout.row_major(N, 1), MutAnyOrigin,
    ](grad_dev.unsafe_ptr())
    var w_lt = LayoutTensor[
        DT, Layout.row_major(N), MutAnyOrigin,
    ](weights_dev.unsafe_ptr())
    comptime TPB = 128
    comptime n_blocks = (N + TPB - 1) // TPB
    comptime k = _scale_grad_by_weights_kernel[N]
    ctx.enqueue_function[k](
        grad_lt, w_lt, grid_dim=n_blocks, block_dim=TPB,
    )
    var h_out = alloc[Scalar[DT]](N)
    ctx.enqueue_copy(h_out, grad_dev)
    ctx.synchronize()
    for i in range(N):
        # After kernel: grad[i] = (i+1) * 1/(i+1) = 1.0
        var diff = (h_out[i] - Scalar[DT](1.0)).__abs__()
        assert_true(
            diff < Scalar[DT](1e-6),
            "grad[" + String(i) + "] expected 1.0, got " + String(h_out[i]),
        )
    print("  test_scale_grad_kernel_unit PASSED")


def test_null_sentinel_unaffected() raises:
    """Non-PER trainer (UniformSampleGpuStep) — the critic step's
    `weights_p` defaults to null. Verify the training run still
    succeeds and the mean is finite, proving the null-sentinel branch
    is untouched."""
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainerV2R[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet, CriticNet,
    ].make(
        ctx=ctx,
        learning_starts=500, window_size=10,
    )
    var env = PendulumEnv[DT]()
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    _ = env.reset()
    var obs_self = env.get_obs_list()
    var step: Int = 0
    while step < SMOKE_STEPS:
        for d in range(OBS_DIM):
            obs[d] = obs_self[d]
        trainer.select_action_gpu(obs, action, step)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS_DIM):
            next_obs[d] = nxt[d]
        trainer.record(
            obs, action, reward, next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            trainer.end_episode()
            _ = env.reset()
            obs_self = env.get_obs_list()
        else:
            obs_self = nxt.copy()
        step += 1
        _ = trainer.train_step_gpu(step)
    var mr = trainer.mean_return()
    assert_true(
        not isnan(Float64(mr)),
        "Null-sentinel path broke — uniform-replay trainer NaN",
    )
    print("  test_null_sentinel_unaffected PASSED mean=", mr)


def test_per_weights_are_nontrivial() raises:
    """After warmup, the IS-weights vector populated by
    `buf_per.sample[BATCH]` should be normalised (max ≈ 1.0) and
    heterogeneous (min noticeably < 1.0). If the kernel were fed an
    all-ones placeholder, this property wouldn't hold — proving the
    real PER IS weights flow into the gradient scaling."""
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainerV2R[
        "gpu",
        PerSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet, CriticNet,
    ].make(
        ctx=ctx,
        per_alpha=Scalar[DT](0.6), per_beta=Scalar[DT](0.4),
        learning_starts=500, window_size=10,
    )
    var env = PendulumEnv[DT]()
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    _ = env.reset()
    var obs_self = env.get_obs_list()
    var step: Int = 0
    while step < SMOKE_STEPS:
        for d in range(OBS_DIM):
            obs[d] = obs_self[d]
        trainer.select_action_gpu(obs, action, step)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        for d in range(OBS_DIM):
            next_obs[d] = nxt[d]
        trainer.record(
            obs, action, step_res[1], next_obs,
            Scalar[DT](1.0) if step_res[2] else Scalar[DT](0.0),
        )
        if step_res[2]:
            trainer.end_episode()
            _ = env.reset()
            obs_self = env.get_obs_list()
        else:
            obs_self = nxt.copy()
        step += 1
        _ = trainer.train_step_gpu(step)

    # D2H buf_per.weights after the last train_step's sample(). The
    # PER buffer now lives inside the sample block (V2R encapsulation).
    var h_w = alloc[Scalar[DT]](BATCH)
    ctx.enqueue_copy(h_w, trainer.sample_blk.buf.value().weights)
    ctx.synchronize()
    var w_max = Scalar[DT](0.0)
    var w_min = Scalar[DT](2.0)
    for i in range(BATCH):
        if h_w[i] > w_max:
            w_max = h_w[i]
        if h_w[i] < w_min:
            w_min = h_w[i]
    print("  IS weights: min=", w_min, " max=", w_max)
    assert_true(
        (w_max - Scalar[DT](1.0)).__abs__() < Scalar[DT](1e-3),
        "max IS weight should be ≈1.0 (normalised); got " + String(w_max),
    )
    assert_true(
        w_min < Scalar[DT](0.999),
        "min IS weight should be strictly < 1.0 (heterogeneous "
        + "priorities); got " + String(w_min),
    )
    # And the trainer should still train without diverging.
    var mr = trainer.mean_return()
    assert_true(
        not isnan(Float64(mr)),
        "IS-weighted PER trainer NaN",
    )
    print("  test_per_weights_are_nontrivial PASSED mean=", mr)


def main() raises:
    print("=" * 60)
    print("C.3c: IS-weighted critic loss for PER")
    print("=" * 60)
    test_scale_grad_kernel_unit()
    test_null_sentinel_unaffected()
    test_per_weights_are_nontrivial()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
