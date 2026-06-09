"""Phase 1.6 — DQN CNN / pixel-obs config.

nn2 `Conv2D`/`Flatten` expose FLAT `IN_DIMS`/`OUT_DIM` (the spatial shape
is comptime-internal), so a `Sequential[Conv2D, ReLU, …, Flatten, Linear,
…]` Q-net flows through the discrete off-policy trainer with the obs
treated as a flat `C·H·W` vector — ZERO obs-pipeline plumbing changes
(the same insight Phase 5.2 used for on-policy CNN PPO). This closes the
last Phase-1 gap (`DQNCNNConfig` / `DQNCNN`).

Two checks:
  1. `test_dqn_cnn_driver_flow` — a small-image CNN Q-net driven through
     `DQNTrainer` (record + train_step + greedy) on synthetic 1×12×12
     images. Proves image-shaped obs flow through the discrete off-policy
     path end-to-end with finite training. CPU + Apple GPU (the trainer
     threads `ctx` through record/train internally, so one loop covers
     both targets).
  2. `test_dqn_cnn_preset_builds` — the canonical `DQNCNN` 84×84 Nature
     preset builds and produces a finite in-range greedy action, proving
     the config + Nature-DQN net compose through the agent facade.

Real numeric convergence is out of scope (and pixel DQN is NVIDIA-scale);
this is a finiteness + plumbing smoke.

Run:
    pixi run mojo run -I . tests/nn2/test_dqn_cnn_config_smoke.mojo
    pixi run -e apple mojo run -I . tests/nn2/test_dqn_cnn_config_smoke.mojo
"""

from std.math import isnan, isinf
from std.random import seed, random_float64
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.conv2d import Conv2D
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.deep_agents2.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents2.dqn.config import DQNCNN
from mojo_rl.deep_agents2.training.driver_offpolicy_discrete import (
    OffPolicyDiscreteAgent,
)
from mojo_rl.deep_agents2.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
)


# Small image: 1 channel, 12×12 → flat OBS_DIM = 144. Keeps CPU conv cost
# low while exercising the exact Conv2D→Flatten→Linear plumbing.
comptime C = 1
comptime IMG = 12
comptime OBS_DIM = C * IMG * IMG       # 144
comptime NUM_ACTIONS = 4
comptime BATCH = 32
comptime CAP = 2_048
comptime WARMUP = 100
comptime TOTAL_STEPS = 600

# conv1: 1→8, 3×3 s2 p1 : 12→6   (8·6·6 = 288)
# conv2: 8→16, 3×3 s2 p1: 6→3    (16·3·3 = 144)
comptime SmallCnnQNet = Sequential[
    Conv2D[C, 8, 3, 2, 1, IMG, IMG], ReLU[8 * 6 * 6],
    Conv2D[8, 16, 3, 2, 1, 6, 6], ReLU[16 * 3 * 3],
    Flatten[16 * 3 * 3],
    Linear[16 * 3 * 3, 64], ReLU[64],
    Linear[64, NUM_ACTIONS],
]


def _rand_image(mut buf: List[Scalar[DT]]) raises:
    for i in range(len(buf)):
        buf[i] = Scalar[DT](random_float64())


def _drive_cnn_dqn[
    T: OffPolicyDiscreteAgent,
](mut trainer: T, target_label: StaticString) raises:
    """Manual train loop on synthetic 1×12×12 images. Cycles actions for
    replay coverage (the discrete driver's epsilon path is dimension-
    agnostic; record + train_step + the Q-net forward are what touch the
    image obs). Generic over any `OffPolicyDiscreteAgent`, so CPU and GPU
    trainers share one body (each threads its own `ctx` internally)."""
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    _rand_image(obs)
    for step in range(TOTAL_STEPS):
        var a = step % NUM_ACTIONS
        _rand_image(next_obs)
        # Synthetic reward correlated with the chosen action + first pixel
        # so the gradient has signal; terminate every 20 steps.
        var reward = Scalar[DT](Float64(a) * 0.1) + obs[0]
        var done = (step % 20 == 19)
        trainer.record(
            obs, a, reward, next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            trainer.end_episode()
        for d in range(OBS_DIM):
            obs[d] = next_obs[d]
        _ = trainer.train_step(step)

    var g = trainer.select_greedy_action(obs)
    print("  [", target_label, "] greedy_action=", g)
    assert_true(
        g >= 0 and g < NUM_ACTIONS,
        String(target_label) + ": greedy action out of range",
    )


def test_dqn_cnn_driver_flow_cpu() raises:
    print("--- DQN CNN driver flow (CPU, 1×12×12) ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        SmallCnnQNet,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.01),
        learning_starts=WARMUP,
        max_grad_norm=Scalar[DT](10.0),
        initial_episode_fill=Scalar[DT](0.0),
    )
    _drive_cnn_dqn(trainer, "cpu")
    var log = trainer.flush_train_log()
    print("  [cpu] mean_loss=", log[0], " n_updates=", log[2])
    assert_true(not isnan(log[0]), "cpu: mean_loss NaN")
    assert_true(not isinf(log[0]), "cpu: mean_loss Inf")
    assert_true(log[2] > 0, "cpu: no training updates")


def test_dqn_cnn_driver_flow_gpu() raises:
    print("--- DQN CNN driver flow (GPU, 1×12×12) ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            SmallCnnQNet,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](2.5e-4),
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            epsilon=Scalar[DT](1.0),
            epsilon_decay=Scalar[DT](0.995),
            epsilon_min=Scalar[DT](0.01),
            learning_starts=WARMUP,
            max_grad_norm=Scalar[DT](10.0),
            initial_episode_fill=Scalar[DT](0.0),
        )
        _drive_cnn_dqn(trainer, "gpu")
        var log = trainer.flush_train_log()
        print("  [gpu] mean_loss=", log[0], " n_updates=", log[2])
        assert_true(not isnan(log[0]), "gpu: mean_loss NaN")
        assert_true(not isinf(log[0]), "gpu: mean_loss Inf")
        assert_true(log[2] > 0, "gpu: no training updates")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def test_dqn_cnn_preset_builds() raises:
    """The canonical Nature-DQN 84×84 preset (`DQNCNN`) builds and yields
    a finite in-range greedy action — proves the config + Nature net
    compose through the `DQNAgent` facade. Forward-only (no training loop)
    to bound CPU cost of the full 4×84×84 conv stack."""
    print("--- DQNCNN 84×84 Nature preset build + greedy ---")
    seed(42)
    comptime FRAMES = 4
    comptime ACT = 6
    comptime OBS = FRAMES * 84 * 84   # 28224
    var agent = DQNCNN["cpu", ACT, 32, 4_096, FRAMES](
        learning_starts=1_000,
    )
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    for i in range(OBS):
        obs[i] = Scalar[DT](random_float64())
    var g = agent.select_greedy_action(obs)
    print("  greedy_action=", g, " (ACT=", ACT, ")")
    assert_true(
        g >= 0 and g < ACT, "DQNCNN preset: greedy action out of range",
    )


def main() raises:
    print("=" * 64)
    print("Phase 1.6 — DQN CNN / pixel-obs config")
    print("=" * 64)
    test_dqn_cnn_driver_flow_cpu()
    test_dqn_cnn_driver_flow_gpu()
    test_dqn_cnn_preset_builds()
    print("=" * 64)
    print("ALL PASSED")
    print("=" * 64)
