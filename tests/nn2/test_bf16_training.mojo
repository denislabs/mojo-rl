"""C.5 — SAC bf16 (Bf16Compute) GPU smoke.

Phase C.5 wires `POLICY: AMPPolicy = NoAMP` through `SACTrainer.train_step`
and every loss/critic/target-y helper. With `POLICY=Bf16Compute`, the
forward/vjp matmuls run in bf16 compute while params + Adam moments
stay fp32.

Smoke (NOT a long-horizon convergence regression — that's a follow-up):

  - Build SAC trainer with `make["gpu"](ctx, config)` where
    `config.use_bf16 = True` (Saveable hint).
  - Run 2k env-step transitions on Pendulum V1; verify
      * Tracker mean moved off `initial_episode_fill = -1250`.
      * At least one episode completed.
      * No NaN / divergence (mean is finite, > -10_000).
  - Repeat at `POLICY=NoAMP` and confirm both branches type-check
    and run side-by-side.

A 30k Pendulum bf16 → mean10 ≤ -300 convergence regression is the
plan's C.5 validation target; we defer it to a separate run since
the existing single-env GPU SAC convergence (NoAMP) takes ~17 min
on Apple Metal. This test exercises the dispatch surface end-to-end.
"""

from std.gpu.host import DeviceContext
from std.math import isnan
from std.memory import alloc
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP, Bf16Compute
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.sac_config import SACConfig
from mojo_rl.nn2.core.save_scalar import SaveBool

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


def _run_smoke[POLICY: AMPPolicy](label: String) raises -> Scalar[DT]:
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY
    ].make["gpu"](
        ctx,
        actor_lr=Scalar[DT](3e-4), critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4), gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005), action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2), target_entropy=Scalar[DT](-1.0),
        learning_starts=500,
        window_size=10, initial_episode_fill=Scalar[DT](-1250.0),
    )
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
        _ = trainer.train_step["gpu", POLICY=POLICY](step)

    var mr = trainer.mean_return()
    var n_eps = trainer.ep_count()
    assert_true(
        n_eps >= 1,
        label + ": expected >=1 completed episode, got " + String(n_eps),
    )
    assert_true(
        not isnan(Float64(mr)),
        label + ": tracker mean is NaN — bf16 path likely diverged",
    )
    assert_true(
        (mr - Scalar[DT](-1250.0)).__abs__() > Scalar[DT](1.0),
        label + ": tracker should have advanced past initial_fill; mean="
        + String(mr),
    )
    assert_true(
        Float64(mr) > -10_000.0,
        label + ": tracker mean looks pathological; got " + String(mr),
    )
    print(
        "  ", label, " smoke PASSED (eps=", n_eps,
        " mean_ret(10)=", mr, ")",
    )
    return mr


def test_noamp_smoke() raises:
    _ = _run_smoke[NoAMP](String("NoAMP"))


def test_bf16_smoke() raises:
    _ = _run_smoke[Bf16Compute](String("Bf16Compute"))


def test_config_use_bf16_round_trip() raises:
    """`SACConfig.use_bf16` is a Saveable Bool — round-trip it
    through `save` / `load` to verify the SaveBool wrapper works
    end-to-end through the Config's reflection walker."""
    var cfg = SACConfig.default()
    cfg.use_bf16 = SaveBool(True)
    var dump = String("")
    cfg.save(dump, String(""))
    # Parse back.
    var lines = List[String]()
    var line = String("")
    for i in range(len(dump)):
        var ch = dump[byte=i:i+1]
        if String(ch) == "\n":
            lines.append(line)
            line = String("")
        else:
            line += String(ch)
    if len(line) > 0:
        lines.append(line)
    var cfg2 = SACConfig.default()
    var idx: Int = 0
    cfg2.load(lines, idx, String(""))
    assert_true(
        cfg2.use_bf16.v == True,
        "use_bf16 should round-trip True, got "
        + String(cfg2.use_bf16.v),
    )
    print("  test_config_use_bf16_round_trip PASSED")


def main() raises:
    print("=" * 60)
    print("C.5 SAC bf16 smoke (Pendulum 2k steps)")
    print("=" * 60)
    test_config_use_bf16_round_trip()
    test_noamp_smoke()
    test_bf16_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
