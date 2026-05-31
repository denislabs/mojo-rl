"""R.5+ — REDQAgent + SmallREDQ preset + checkpoint round-trip on Pendulum.

End-to-end integration gate combining the three pieces shipped in
this slice:

  (1) The SmallREDQ preset builds a REDQAgent for Pendulum's shape
      with one line.
  (2) `agent.train_single(env, ...)` runs the off-policy driver
      unchanged (verified by the actual mean_return improvement).
  (3) `agent.save(path)` → `agent.load(path)` on a FRESH agent
      reproduces the original's greedy actions on the same probe obs.

Pendulum CPU 10k env steps → expected mean_ret > -200 (matches the
R.4 raw-trainer smoke).
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.redq import SmallREDQ
from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 256
comptime CAP = 50_000
comptime TOTAL_TIMESTEPS = 10_000
comptime WARMUP = 1_000


def test_redq_agent_pendulum_smoke() raises:
    print("=" * 70)
    print("R.5+ — SmallREDQ preset + checkpoint on Pendulum V1 (CPU)")
    print("=" * 70)
    seed(42)

    var agent = SmallREDQ["cpu", OBS, ACT, BATCH, CAP](
        # SmallREDQ defaults pre-tuned for Pendulum-shape continuous control.
        action_scale=Scalar[DT](2.0),       # Pendulum torque range
        target_entropy=Scalar[DT](-1.0),    # = -ACT
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
        window_size=10,
    )
    var env = PendulumEnv[DT]()

    var ep_returns = agent.train_single(
        env,
        TOTAL_TIMESTEPS,
        print_every=2_000,
        verbose=True,
    )

    var trained_mean = Float64(agent.mean_return())
    var ep = agent.ep_count()
    print("Final mean ep return (last 10):", trained_mean)
    print("Episodes completed:            ", ep)

    # (1) Driver ran end-to-end.
    assert_true(
        len(ep_returns) > 0,
        "driver must return at least one episode",
    )
    assert_true(ep > 0, "tracker saw at least one episode")
    # (2) Improvement over random baseline.
    assert_true(
        trained_mean > -1200.0,
        "SmallREDQ must learn within 10k env steps",
    )
    # Bonus (best-effort): the R.4 raw-trainer smoke crossed -200 at
    # step 6000 — verify the preset path matches that ballpark.
    if trained_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200) via the preset.")

    # (3) Save + load round-trip → greedy actions match.
    var path = String("/tmp/redq_agent_pendulum.bin")
    agent.save(path)
    print("Saved checkpoint to:", path)

    var fresh = SmallREDQ["cpu", OBS, ACT, BATCH, CAP](
        action_scale=Scalar[DT](2.0),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
        window_size=10,
    )
    fresh.load(path)
    print("Loaded checkpoint into fresh agent.")

    var probe = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    probe[0] = Scalar[DT](0.7)
    probe[1] = Scalar[DT](-0.3)
    probe[2] = Scalar[DT](1.5)
    var act_trained = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var act_loaded = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe, act_trained)
    fresh.select_greedy_action(probe, act_loaded)
    print(
        "Greedy [trained]:", act_trained[0],
        " [loaded]:", act_loaded[0],
    )
    var dev = Float64(act_trained[0]) - Float64(act_loaded[0])
    if dev < 0.0:
        dev = -dev
    print("Greedy action |Δ|:", dev)
    assert_true(
        dev < 1e-3,
        "loaded agent must reproduce greedy action within format tol",
    )

    print("PASS — REDQAgent + preset + checkpoint smoke green.")


def main() raises:
    test_redq_agent_pendulum_smoke()
