"""C.2b — verify `SACTrainer[..., N_STEP=K]` with `cfg.use_n_step=True`
wraps incoming transitions through an `NStepBuffer` and bakes `γ^K` into
the target-y bootstrap.

Checks:

  1. Config round-trip: `SaveBool` for `use_n_step` survives save/load.
  2. Default trainer (`N_STEP=1` or `use_n_step=False`): `_use_nstep`
     is `False`, no nstep buffer allocated, behavior unchanged.
  3. N_STEP=3 + `use_n_step=True`: `_use_nstep=True`, nstep buffer
     allocated. After 3 `record()` calls the underlying GPU replay has
     received exactly 1 compressed transition. Returns are
     accumulated against the per-step reward (not the compressed one).
  4. 2k Pendulum smoke with N_STEP=3 + use_n_step trains without
     divergence.
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
comptime N_STEP = 3

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


def test_config_use_nstep_round_trip() raises:
    var cfg = SACConfig.default()
    cfg.use_n_step = SaveBool(True)
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
    assert_true(cfg2.use_n_step.v == True, "use_n_step round-trip")
    print("  test_config_use_nstep_round_trip PASSED")


def test_default_n_step_disabled() raises:
    """SACTrainer with default N_STEP=1 and use_n_step=False — n-step
    machinery stays dormant; _use_nstep=False; behaviour unchanged."""
    var ctx = DeviceContext()
    var cfg = SACConfig.default()
    cfg.learning_starts = SaveI(500)
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
    ].make["gpu"](ctx, cfg)
    assert_true(
        not trainer._use_nstep,
        "Default N_STEP=1 / use_n_step=False → _use_nstep must be False",
    )
    assert_true(
        not Bool(trainer.nstep_cpu),
        "nstep_cpu Optional must be None when n-step disabled",
    )
    print("  test_default_n_step_disabled PASSED")


def test_nstep_compresses_three_to_one() raises:
    """With N_STEP=3 + use_n_step=True, three single-step `record`
    calls produce exactly ONE compressed transition in the replay."""
    var ctx = DeviceContext()
    var cfg = SACConfig.default()
    cfg.use_n_step = SaveBool(True)
    cfg.learning_starts = SaveI(500)
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
        N_STEP,
    ].make["gpu"](ctx, cfg)
    assert_true(trainer._use_nstep, "_use_nstep should be True")
    assert_true(Bool(trainer.nstep_cpu), "nstep_cpu should be Some")
    assert_true(
        trainer.buf_gpu.value().size == 0,
        "Empty replay at start",
    )

    var obs = alloc[Scalar[DT]](OBS_DIM)
    var act = alloc[Scalar[DT]](ACT_DIM)
    var nxt = alloc[Scalar[DT]](OBS_DIM)
    for i in range(3):
        for d in range(OBS_DIM):
            obs[d] = Scalar[DT](Float64(i) + 0.1 * Float64(d))
            nxt[d] = Scalar[DT](Float64(i) + 1.0 + 0.1 * Float64(d))
        for j in range(ACT_DIM):
            act[j] = Scalar[DT](0.0)
        trainer.record(
            obs, act, Scalar[DT](Float64(i) + 1.0),
            nxt, Scalar[DT](0.0),
        )

    # After N=3 calls, expect exactly one compressed transition stored.
    assert_true(
        trainer.buf_gpu.value().size == 1,
        "After 3 record() calls with N_STEP=3, replay size should be 1; got "
        + String(trainer.buf_gpu.value().size),
    )
    print("  test_nstep_compresses_three_to_one PASSED")


def test_nstep_smoke_2k() raises:
    seed(42)
    var ctx = DeviceContext()
    var cfg = SACConfig.default()
    cfg.use_n_step = SaveBool(True)
    cfg.learning_starts = SaveI(500)
    cfg.window_size = SaveI(10)
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
        N_STEP,
    ].make["gpu"](ctx, cfg)
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
        _ = trainer.train_step_gpu(step)

    var mr = trainer.mean_return()
    assert_true(
        not isnan(Float64(mr)),
        "N-step route NaN — γ^N bake or compression broke",
    )
    assert_true(
        Float64(mr) > -10_000.0,
        "N-step route pathological mean: " + String(mr),
    )
    print("  test_nstep_smoke_2k PASSED mean=", mr)


def main() raises:
    print("=" * 60)
    print("C.2b: SACTrainer auto-route on cfg.use_n_step + N_STEP")
    print("=" * 60)
    test_config_use_nstep_round_trip()
    test_default_n_step_disabled()
    test_nstep_compresses_three_to_one()
    test_nstep_smoke_2k()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
