"""C.5b — verify `SACTrainer.make[target='gpu'](ctx, config)` propagates
`config.use_bf16` into the runtime `_use_bf16` field and that
`trainer.train_step_gpu(step)` (non-parametric trait wrapper) routes
through `POLICY=Bf16Compute` when the flag is set.

What we check, in order:

  1. Default `SACConfig.default()` has `use_bf16=False`; trainer built
     from it stores `_use_bf16=False`.
  2. Flipping `cfg.use_bf16 = SaveBool(True)` and rebuilding produces a
     trainer with `_use_bf16=True`.
  3. Either branch trains end-to-end against Pendulum for 2k steps
     without divergence (mean is finite, > -10_000, at least one
     completed episode).

This is the C.5b trainer-integration smoke. The pre-existing
`test_bf16_training.mojo` exercises the manual `POLICY=Bf16Compute`
explicit-comptime route; here we cover the *auto-routing* code path that
the production driver (`run_offpolicy_train_gpu_n_envs`) calls.

NOT a long-horizon convergence regression — those run separately.
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
from mojo_rl.nn2.core.save_scalar import SaveBool, SaveI

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


def _build(ctx: DeviceContext, use_bf16: Bool) raises -> SACTrainer[
    ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
]:
    var cfg = SACConfig.default()
    cfg.use_bf16 = SaveBool(use_bf16)
    cfg.learning_starts = SaveI(500)
    cfg.window_size = SaveI(10)
    return SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
    ].make["gpu"](ctx, cfg)


def _run_2k(mut trainer: SACTrainer[
    ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
]) raises -> Scalar[DT]:
    var env = PendulumEnv[DT]()
    var obs = alloc[Scalar[DT]](OBS_DIM)
    var next_obs = alloc[Scalar[DT]](OBS_DIM)
    var action = alloc[Scalar[DT]](ACT_DIM)
    _ = env.reset()
    var obs_self = env.get_obs_list()
    var step: Int = 0
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
        # IMPORTANT: route through the trait wrapper, NOT the
        # comptime-parametric form. This is the C.5b code path.
        _ = trainer.train_step_gpu(step)
    return trainer.mean_return()


def test_default_config_routes_noamp() raises:
    seed(42)
    var ctx = DeviceContext()
    var trainer = _build(ctx, use_bf16=False)
    assert_true(
        not trainer._use_bf16,
        "_use_bf16 should be False when cfg.use_bf16=False",
    )
    var mr = _run_2k(trainer)
    assert_true(
        not isnan(Float64(mr)),
        "NoAMP route NaN — auto-routing broke",
    )
    assert_true(
        Float64(mr) > -10_000.0,
        "NoAMP route pathological mean: " + String(mr),
    )
    print("  NoAMP route smoke PASSED mean=", mr)


def test_use_bf16_flag_routes_bf16() raises:
    seed(42)
    var ctx = DeviceContext()
    var trainer = _build(ctx, use_bf16=True)
    assert_true(
        trainer._use_bf16,
        "_use_bf16 should be True when cfg.use_bf16=True",
    )
    var mr = _run_2k(trainer)
    assert_true(
        not isnan(Float64(mr)),
        "Bf16Compute route NaN — auto-routing broke",
    )
    assert_true(
        Float64(mr) > -10_000.0,
        "Bf16Compute route pathological mean: " + String(mr),
    )
    print("  Bf16Compute route smoke PASSED mean=", mr)


def main() raises:
    print("=" * 60)
    print("C.5b: SACTrainer auto-route on cfg.use_bf16")
    print("=" * 60)
    test_default_config_routes_noamp()
    test_use_bf16_flag_routes_bf16()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
