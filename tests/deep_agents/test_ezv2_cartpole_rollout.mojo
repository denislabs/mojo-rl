"""Phase-2 Step 5a smoke test — `GenericEfficientZeroV2Agent` rollout.

Drives the (untrained) EZ-V2 agent on CartPole for ~500 env steps and
verifies the inference + episode-management plumbing end-to-end:

  1. The agent's `select_action()` returns a legal action and a valid
     improved policy distribution every step.
  2. Episode buffers fill correctly during a rollout, and `store_transition`
     with `done=True` flushes them into the replay buffer's
     SequenceReplayBuffer + parallel MCTS-target arrays.
  3. The replay buffer eventually holds at least one full
     (K + N + 1)-step window so `state.is_ready()` flips to True — the
     gate that step 5b's `train()` will use to decide when to start.
  4. No NaNs leak from the GumbelMCTS simulations (a freshly-initialized
     network occasionally produces extreme values; the search must
     handle them).

We do NOT expect convergence — there's no training yet. With Kaiming-init
networks and no learning, episode lengths should hover near random
(~22 ± 10 on CartPole-v1).
"""

from std.math import abs
from std.random import seed
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


def main():
    print("=== EZ-V2 Phase 2 / Step 5a — CartPole rollout smoke test ===")
    var passed = 0
    var total = 0

    # ── Tiny config for fast smoke test ──────────────────────────────────
    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=32,
        HIDDEN=32,
        PROJ=64,
        PRED_BOTTLENECK=32,
        BINS=21,
        BS=8,
        K_UNROLL=3,
        N_TD=5,
        SIMS=8,
        NODES=32,
        K_GUMBEL=2,
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99,
        v_min=-10.0,
        v_max=10.0,
        temperature=1.0,
        temperature_decay_steps=10000,
    )
    var env = CartPoleEnv[DType.float32]()

    var num_episodes = 25
    var max_steps_per_ep = 50
    var total_steps = 0
    var num_invalid_actions = 0
    var num_bad_policies = 0
    var num_bad_values = 0
    var ep_lengths = List[Int]()

    for ep in range(num_episodes):
        var obs = env.reset_obs_list()
        var ep_len = 0
        for _step in range(max_steps_per_ep):
            var result = agent.select_action(obs, training=True)
            var action = result[0]
            var policy = result[1]
            var root_value = result[2]

            # Action must be legal.
            if action < 0 or action >= 2:
                num_invalid_actions += 1
            # Policy must sum to ~1 with all entries in [0, 1].
            var sum_p = Float64(0.0)
            var min_p = Float64(1e18)
            var max_p = Float64(-1e18)
            for a in range(2):
                sum_p += policy[a]
                if policy[a] < min_p:
                    min_p = policy[a]
                if policy[a] > max_p:
                    max_p = policy[a]
            if not (
                sum_p > 0.999
                and sum_p < 1.001
                and min_p >= -1e-6
                and max_p <= 1.0 + 1e-6
            ):
                num_bad_policies += 1
            # Root value must be finite.
            if not _is_finite(root_value):
                num_bad_values += 1

            var step_result = env.step_obs(action)
            var next_obs = step_result[0].copy()
            var reward = Float64(step_result[1])
            var done = step_result[2]

            agent.store_transition(
                obs, action, reward, policy, root_value, done
            )

            ep_len += 1
            total_steps += 1
            obs = next_obs^
            if done:
                break
        ep_lengths.append(ep_len)

    # ── Aggregates ───────────────────────────────────────────────────────
    var sum_len = 0
    var min_len = ep_lengths[0]
    var max_len = ep_lengths[0]
    for i in range(len(ep_lengths)):
        sum_len += ep_lengths[i]
        if ep_lengths[i] < min_len:
            min_len = ep_lengths[i]
        if ep_lengths[i] > max_len:
            max_len = ep_lengths[i]
    var mean_len = Float64(sum_len) / Float64(num_episodes)

    print()
    print("--- Rollout summary ---")
    print("    episodes        =", num_episodes)
    print("    total env steps =", total_steps)
    print("    ep length       =", mean_len, " (min", min_len, "max", max_len, ")")
    print("    invalid actions =", num_invalid_actions)
    print("    bad policies    =", num_bad_policies)
    print("    bad values      =", num_bad_values)
    print("    replay buf len  =", agent.state.buffer.size)
    print("    state.is_ready  =", agent.state.is_ready())

    # ── Assertions ───────────────────────────────────────────────────────
    _expect(
        num_invalid_actions == 0,
        "every action returned by select_action is in [0, ACT)",
        passed, total,
    )
    _expect(
        num_bad_policies == 0,
        "every search returned a valid distribution (sum=1, in [0,1])",
        passed, total,
    )
    _expect(
        num_bad_values == 0,
        "root value is finite on every step",
        passed, total,
    )
    _expect(
        total_steps > 0,
        "the rollout actually advanced (non-zero env steps)",
        passed, total,
    )
    _expect(
        agent.state.buffer.size == total_steps,
        "replay buffer received exactly `total_steps` transitions",
        passed, total,
    )
    # K + N + 1 = 3 + 5 + 1 = 9 transitions. With ~25 episodes the buffer
    # should comfortably exceed that.
    _expect(
        agent.state.is_ready(),
        "replay buffer holds enough transitions for a (K+N+1) sample",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
