"""Phase-2 Step 5d — interleaved-training smoke test on CartPole.

Drives `select_action` + `train_step` in a single loop and watches the
mean episode return over training. The goal is to confirm the full
collect → train pipeline is wired correctly and that training is
**effective** — i.e. mean return in the final quarter of training is
meaningfully above the first quarter. We do *not* claim CartPole
convergence here.

Why a "trend" assertion rather than a hard "≥ 450" bound:

  • The plan target (≥ 450 in 50k env steps, paper Table 3 hyperparams)
    needs the full 1024-d projector + 256 batch + paper LR schedule, which
    are all sized for fast smoke-test iteration here. With the smoke
    config we should still see clear learning, but not full convergence.
  • 50k env steps × (select_action + 1 train_step / 4 env steps) takes
    ~12-20 min wall on Apple Silicon. Test runtimes need to stay short.

Default loop: 3000 env steps, train_step every 4 env steps once buffer is
ready. With this much data the agent should clearly learn to balance the
pole longer than random (~22 ± 10 ep length on CartPole-v1).

Configuration to crank up for a real convergence run is in the
`# CONVERGENCE NOTE` block below.
"""

from std.math import abs
from std.random import seed
from std.time import perf_counter_ns
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.envs.cartpole import CartPoleEnv
from mojo_rl.nn.constants import dtype


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


def _expect(
    cond: Bool,
    label: String,
    mut passed: Int,
    mut total: Int,
):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _mean(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = Float64(0.0)
    for i in range(len(xs)):
        s += xs[i]
    return s / Float64(len(xs))


def main():
    print("=== EZ-V2 Phase 2 / Step 5d — CartPole interleaved-training smoke ===")
    var passed = 0
    var total = 0

    # CONVERGENCE NOTE
    # ────────────────
    # For a real convergence run targeting paper-style ≥ 450 in 50k env
    # steps, bump LATENT/HIDDEN to 128, PROJ to 256, BS to 64, K_GUMBEL to
    # 8, SIMS to 32, NUM_ENV_STEPS to 50000, and TRAIN_INTERVAL down to 1
    # — and expect ~12-20 min wall on Apple Silicon. The test below uses a
    # smaller smoke config so CI is fast.
    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=64,
        HIDDEN=64,
        PROJ=128,
        PRED_BOTTLENECK=64,
        BINS=21,
        BS=16,
        K_UNROLL=3,
        N_TD=5,
        SIMS=8,
        NODES=32,
        K_GUMBEL=2,
        # Slower learning rate than default 1e-3 so the over-strong L_G
        # doesn't push the projector to trivial collapse on the first
        # ~hundred train_steps. Paper's actual setup uses 5e-4 per Table 3.
        LR=Float64(5e-4),
        # Smaller consistency weight for the smoke test — paper's λ_G=2.0
        # was visibly saturating on this size network in step 5c, suggesting
        # the projector is collapsing toward trivial alignment when the
        # buffer is small. Halve to 1.0 for the smoke run.
        LAMBDA_G=Float64(1.0),
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99,
        v_min=-15.0,
        v_max=15.0,
        temperature=1.0,
        temperature_decay_steps=2000,
    )
    var env = CartPoleEnv[DType.float32]()

    comptime NUM_ENV_STEPS = 3000
    comptime TRAIN_INTERVAL = 4   # one train_step per 4 env steps once ready
    comptime LOG_EVERY = 500

    var ep_returns = List[Float64]()
    var ep_return = Float64(0.0)
    var ep_len_steps = 0
    var obs = env.reset_obs_list()
    var num_train_calls = 0
    var any_nan_loss = False
    var max_loss_seen = Float64(0.0)
    var min_loss_seen = Float64(1e18)

    var t0 = perf_counter_ns()

    for env_step in range(NUM_ENV_STEPS):
        var result = agent.select_action(obs, training=True)
        var action = result[0]
        var policy = result[1]
        var root_value = result[2]
        var step_result = env.step_obs(action)
        var next_obs = step_result[0].copy()
        var reward = Float64(step_result[1])
        var done = step_result[2]
        agent.store_transition(
            obs, action, reward, policy, root_value, done
        )
        ep_return += reward
        ep_len_steps += 1

        if done:
            ep_returns.append(ep_return)
            ep_return = Float64(0.0)
            ep_len_steps = 0
            obs = env.reset_obs_list()
        else:
            obs = next_obs^

        # Decay temperature
        agent.decay_temperature()

        # Train periodically once we have a full window.
        if (
            agent.state.is_ready()
            and (env_step + 1) % TRAIN_INTERVAL == 0
        ):
            var t = agent.train_step()
            var L_total = t[0]
            num_train_calls += 1
            if not _is_finite(L_total):
                any_nan_loss = True
            if L_total > max_loss_seen:
                max_loss_seen = L_total
            if L_total < min_loss_seen:
                min_loss_seen = L_total

        if (env_step + 1) % LOG_EVERY == 0:
            var recent = List[Float64]()
            var window = 30
            var n_eps = len(ep_returns)
            for i in range(max(0, n_eps - window), n_eps):
                recent.append(ep_returns[i])
            print(
                "    env_step",
                env_step + 1,
                ": episodes=",
                len(ep_returns),
                " recent mean ep return=",
                _mean(recent),
                " train_calls=",
                num_train_calls,
                " temp=",
                agent.temperature,
            )

    var t1 = perf_counter_ns()
    var wall_ms = Float64(t1 - t0) / 1.0e6

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("--- Run summary ---")
    print("    env steps         =", NUM_ENV_STEPS)
    print("    episodes finished =", len(ep_returns))
    print("    train_step calls  =", num_train_calls)
    print("    wall time         =", wall_ms, "ms")

    var n_eps = len(ep_returns)
    if n_eps == 0:
        print("FAIL: no episodes finished — env didn't terminate?")
        return

    # Compute first vs last quartile mean episode return.
    var q = n_eps // 4
    if q < 1:
        q = 1
    var first_q = List[Float64]()
    var last_q = List[Float64]()
    for i in range(q):
        first_q.append(ep_returns[i])
    for i in range(n_eps - q, n_eps):
        last_q.append(ep_returns[i])
    var first_mean = _mean(first_q)
    var last_mean = _mean(last_q)

    print("    first quartile mean ep return =", first_mean)
    print("    last  quartile mean ep return =", last_mean)
    print("    improvement                   =", last_mean - first_mean)
    print("    L_total range (train)         =", min_loss_seen, "..", max_loss_seen)

    # ── Assertions ───────────────────────────────────────────────────────
    _expect(
        not any_nan_loss,
        "no NaN/Inf loss seen during training",
        passed, total,
    )
    _expect(
        num_train_calls > 0,
        "train_step actually fired (replay was ready in time)",
        passed, total,
    )
    _expect(
        n_eps >= 4,
        "at least 4 episodes finished (so quartile stat is meaningful)",
        passed, total,
    )
    # Training-is-effective signal: last-quartile mean ≥ first × 1.3.
    # CartPole baseline (random) is ~22; we need to see meaningful upward
    # movement, but we don't assert convergence (≥ 450).
    _expect(
        last_mean >= first_mean * 1.3 or last_mean >= first_mean + 10.0,
        "last-quartile mean ≥ first × 1.3 OR ≥ first + 10 (training works)",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
