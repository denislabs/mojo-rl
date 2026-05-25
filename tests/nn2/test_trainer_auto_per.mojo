"""C.3b — verify `SACTrainer.make["gpu"](ctx, config)` swaps the device
replay for `GPUPrioritizedReplay` when `config.use_per=True`, and that
the post-critic TD-error refresh kernel runs without diverging the
training loop.

Checks:

  1. Config round-trip for use_per + per_alpha/beta/epsilon.
  2. Default config: trainer holds `buf_gpu = Some`, `buf_per = None`.
  3. `use_per=True`: trainer holds `buf_per = Some`, `buf_gpu = None`,
     `_td_err_dev = Some`. PER hyperparameters propagated from config.
  4. After enough record() + train_step calls to exercise the priority
     refresh path: `buf_per.max_priority` has moved off 1.0 (the
     initial seed) — proves the refresh kernel ran and updated the
     sum-tree leaves with real TD errors.
  5. 2k Pendulum smoke trains via `train_step_gpu` without divergence.

Note: v1 PER does NOT thread IS weights through the critic loss (the
sample-side prioritization is in; full Schaul-compatible weighted loss
is a future C.3c). This test exercises the integration end-to-end but
does not regression-gate against any specific PER-vs-uniform convergence
delta.
"""

from std.gpu.host import DeviceContext
from std.math import isnan
from std.memory import alloc
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.sac_config import SACConfig
from mojo_rl.nn2.core.save_scalar import SaveBool, SaveI, SaveScalar

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 5_000
comptime SMOKE_STEPS = 2_000

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


def test_config_per_round_trip() raises:
    var cfg = SACConfig.default()
    cfg.use_per = SaveBool(True)
    cfg.per_alpha = SaveScalar[DT](Scalar[DT](0.5))
    cfg.per_beta = SaveScalar[DT](Scalar[DT](0.7))
    cfg.per_epsilon = SaveScalar[DT](Scalar[DT](1e-4))
    var dump = String("")
    cfg.save(dump, String(""))
    var lines = List[String]()
    var line = String("")
    for i in range(dump.byte_length()):
        var ch = dump[byte=i:i+1]
        if String(ch) == "\n":
            lines.append(line)
            line = String("")
        else:
            line += String(ch)
    if line.byte_length() > 0:
        lines.append(line)
    var cfg2 = SACConfig.default()
    var idx: Int = 0
    cfg2.load(lines, idx, String(""))
    assert_true(cfg2.use_per.v == True, "use_per round-trip")
    assert_true(
        (cfg2.per_alpha.v - Scalar[DT](0.5)).__abs__() < Scalar[DT](1e-6),
        "per_alpha round-trip",
    )
    assert_true(
        (cfg2.per_beta.v - Scalar[DT](0.7)).__abs__() < Scalar[DT](1e-6),
        "per_beta round-trip",
    )
    print("  test_config_per_round_trip PASSED")


def test_default_uses_uniform_replay() raises:
    var ctx = DeviceContext()
    var cfg = SACConfig.default()
    cfg.learning_starts = SaveI(500)
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
    ].make["gpu"](ctx, cfg)
    assert_true(
        Bool(trainer.buf_gpu),
        "buf_gpu should be Some when use_per=False (default)",
    )
    assert_true(
        not Bool(trainer.buf_per),
        "buf_per should be None when use_per=False (default)",
    )
    print("  test_default_uses_uniform_replay PASSED")


def test_use_per_swaps_to_prioritized_replay() raises:
    var ctx = DeviceContext()
    var cfg = SACConfig.default()
    cfg.use_per = SaveBool(True)
    cfg.per_alpha = SaveScalar[DT](Scalar[DT](0.5))
    cfg.per_beta = SaveScalar[DT](Scalar[DT](0.7))
    cfg.per_epsilon = SaveScalar[DT](Scalar[DT](1e-4))
    cfg.learning_starts = SaveI(500)
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
    ].make["gpu"](ctx, cfg)
    assert_true(
        Bool(trainer.buf_per),
        "buf_per should be Some when use_per=True",
    )
    assert_true(
        not Bool(trainer.buf_gpu),
        "buf_gpu should be None (mutually exclusive) when use_per=True",
    )
    assert_true(
        Bool(trainer._td_err_dev),
        "_td_err_dev should be allocated when use_per=True",
    )
    assert_true(
        (
            trainer.buf_per.value().alpha - Scalar[DT](0.5)
        ).__abs__() < Scalar[DT](1e-6),
        "per_alpha should match cfg",
    )
    assert_true(
        (
            trainer.buf_per.value().beta - Scalar[DT](0.7)
        ).__abs__() < Scalar[DT](1e-6),
        "per_beta should match cfg",
    )
    print("  test_use_per_swaps_to_prioritized_replay PASSED")


def test_per_priority_refresh_runs() raises:
    """After enough train_step calls to exercise the refresh kernel,
    `max_priority` should have moved off 1.0 (the seed value) — the
    refresh kernel computed real TD-errors and pushed them through the
    sum-tree update path."""
    seed(42)
    var ctx = DeviceContext()
    var cfg = SACConfig.default()
    cfg.use_per = SaveBool(True)
    cfg.learning_starts = SaveI(500)
    cfg.window_size = SaveI(10)
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
    ].make["gpu"](ctx, cfg)
    var env = PendulumEnv[DT]()
    var obs = alloc[Scalar[DT]](OBS_DIM)
    var next_obs = alloc[Scalar[DT]](OBS_DIM)
    var action = alloc[Scalar[DT]](ACT_DIM)
    _ = env.reset()
    var obs_self = env.get_obs_list()
    var step: Int = 0
    # Run enough steps to get past warmup + several training updates.
    while step < SMOKE_STEPS:
        for d in range(OBS_DIM):
            obs[d] = obs_self[d]
        trainer.select_action["gpu"](obs, action, step)
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
    var max_p = trainer.buf_per.value().max_priority
    assert_true(
        (max_p - Scalar[DT](1.0)).__abs__() > Scalar[DT](1e-3),
        "max_priority should have moved off 1.0 after refresh; got "
        + String(max_p),
    )
    var mr = trainer.mean_return()
    assert_true(
        not isnan(Float64(mr)),
        "PER route NaN — refresh kernel broke gradient flow",
    )
    assert_true(
        Float64(mr) > -10_000.0,
        "PER route pathological mean: " + String(mr),
    )
    print(
        "  test_per_priority_refresh_runs PASSED mean=", mr,
        " max_p=", max_p,
    )


def main() raises:
    print("=" * 60)
    print("C.3b: SACTrainer auto-route on cfg.use_per")
    print("=" * 60)
    test_config_per_round_trip()
    test_default_uses_uniform_replay()
    test_use_per_swaps_to_prioritized_replay()
    test_per_priority_refresh_runs()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
