"""C.4b — verify `SACTrainer.make["gpu"](ctx, config)` flips the device-
resident GPUReplay into ERE mode when `config.use_ere=True`, and that
the resulting trainer still trains end-to-end without divergence.

What we check:

  1. Config round-trip: SaveBool / SaveScalar / SaveI for the new
     ERE fields survive save/load.
  2. Default config (`use_ere=False`) → trainer's
     `buf_gpu.value().ere_enabled == False`.
  3. Config with `use_ere=True, ere_eta=0.5, ere_c_min=256, ere_k_max=4`
     → trainer's `buf_gpu.value().ere_enabled == True` and parameters
     match.
  4. 2k smoke against Pendulum exercises the ERE sampling path and
     does not diverge.
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


def test_config_ere_round_trip() raises:
    """SaveBool + SaveScalar + SaveI for ERE fields round-trip via
    SACConfig.save / SACConfig.load."""
    var cfg = SACConfig.default()
    cfg.use_ere = SaveBool(True)
    cfg.ere_eta = SaveScalar[DT](Scalar[DT](0.5))
    cfg.ere_c_min = SaveI(123)
    cfg.ere_k_max = SaveI(7)
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
    assert_true(cfg2.use_ere.v == True, "use_ere round-trip")
    assert_true(
        (cfg2.ere_eta.v - Scalar[DT](0.5)).__abs__() < Scalar[DT](1e-6),
        "ere_eta round-trip",
    )
    assert_true(cfg2.ere_c_min.v == 123, "ere_c_min round-trip")
    assert_true(cfg2.ere_k_max.v == 7, "ere_k_max round-trip")
    print("  test_config_ere_round_trip PASSED")


def _build(ctx: DeviceContext, use_ere: Bool) raises -> SACTrainer[
    ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
]:
    var cfg = SACConfig.default()
    cfg.use_ere = SaveBool(use_ere)
    cfg.ere_eta = SaveScalar[DT](Scalar[DT](0.5))
    cfg.ere_c_min = SaveI(256)
    cfg.ere_k_max = SaveI(4)
    cfg.learning_starts = SaveI(500)
    cfg.window_size = SaveI(10)
    return SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
    ].make["gpu"](ctx, cfg)


def test_default_does_not_enable_ere() raises:
    var ctx = DeviceContext()
    var trainer = _build(ctx, use_ere=False)
    assert_true(
        not trainer.buf_gpu.value().ere_enabled,
        "ere_enabled should be False when cfg.use_ere=False",
    )
    print("  test_default_does_not_enable_ere PASSED")


def test_use_ere_flag_enables_ere() raises:
    var ctx = DeviceContext()
    var trainer = _build(ctx, use_ere=True)
    assert_true(
        trainer.buf_gpu.value().ere_enabled,
        "ere_enabled should be True when cfg.use_ere=True",
    )
    assert_true(
        (
            trainer.buf_gpu.value().ere_eta - Scalar[DT](0.5)
        ).__abs__() < Scalar[DT](1e-6),
        "ere_eta should match cfg value 0.5",
    )
    assert_true(
        trainer.buf_gpu.value()._ere_c_min == 256,
        "ere_c_min should match cfg value 256",
    )
    assert_true(
        trainer.buf_gpu.value()._ere_k_max == 4,
        "ere_k_max should match cfg value 4",
    )
    print("  test_use_ere_flag_enables_ere PASSED")


def test_use_ere_smoke_2k() raises:
    """ERE-enabled trainer trains 2k steps without divergence."""
    seed(42)
    var ctx = DeviceContext()
    var trainer = _build(ctx, use_ere=True)
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
        "ERE route NaN — sampling broke",
    )
    assert_true(
        Float64(mr) > -10_000.0,
        "ERE route pathological mean: " + String(mr),
    )
    print("  test_use_ere_smoke_2k PASSED mean=", mr)


def main() raises:
    print("=" * 60)
    print("C.4b: SACTrainer auto-route on cfg.use_ere")
    print("=" * 60)
    test_config_ere_round_trip()
    test_default_does_not_enable_ere()
    test_use_ere_flag_enables_ere()
    test_use_ere_smoke_2k()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
