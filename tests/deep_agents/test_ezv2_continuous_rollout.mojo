"""End-to-end continuous-agent rollout smoke (Phase 3.5.2).

Drives `GenericEZV2ContinuousAgent` through:
    select_action  →  step (synthetic env)  →  store_transition  →
    flush_episode (auto on done)  →  train_step_gpu

over ~50 episodes of a 30-step synthetic environment, then verifies:

    (a) the agent's `_episode_*` buffers reset cleanly between episodes,
    (b) the replay state fills (>200 transitions),
    (c) all chosen actions stay inside (-MAX_ACTION, MAX_ACTION),
    (d) train_step_gpu runs without NaN over 100 calls,
    (e) composite L decreases ≥ 5%,
    (f) all four loss components remain finite.

Synthetic env: obs is a 4-dim random walk; reward = -‖a‖² penalizes
large actions; done after 30 steps. This gives a non-trivial value
signal so the value head has something to fit.
"""

from std.random import seed, random_float64
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    EZV2GPUStateBase,
    GenericEZV2ContinuousAgent,
)
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


def main() raises:
    print("=== EZ-V2 continuous agent rollout smoke (Phase 3.5) ===")
    var passed = 0
    var total = 0

    comptime OBS = 4
    comptime ACT_DIM = 2
    comptime LATENT = 32
    comptime HIDDEN = 32
    comptime BINS = 21
    comptime BS = 8
    comptime K = 3
    comptime SIMS = 8
    comptime NODES = 32
    comptime K_ROOT = 4
    comptime MAX_ACTION = 1.0

    comptime Config = EZV2ContinuousMLPConfig[
        OBS=OBS,
        ACT_DIM=ACT_DIM,
        LATENT=LATENT,
        HIDDEN=HIDDEN,
        PROJ=64,
        PRED_BOTTLENECK=32,
        BINS=BINS,
        BS=BS,
        K_UNROLL=K,
        N_TD=5,
        SIMS=SIMS,
        NODES=NODES,
        K_ROOT=K_ROOT,
        K_NON_ROOT=2,
        MAX_ACTION=MAX_ACTION,
    ]

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99, v_min=-10.0, v_max=10.0, temperature=1.0,
    )
    var ctx = DeviceContext()

    # ── Roll out synthetic episodes ─────────────────────────────────────
    print()
    print("--- Rolling out 50 synthetic episodes ---")
    var num_episodes = 50
    var max_steps_per_ep = 30
    var any_action_oor = False
    var any_action_nan = False
    var any_value_nan = False
    var max_action_seen = Float64(0.0)
    var total_actions_collected = 0

    for _ep in range(num_episodes):
        # Reset env: random initial obs.
        var obs = List[Scalar[dtype]](capacity=OBS)
        for _ in range(OBS):
            obs.append(Scalar[dtype](random_float64(-1.0, 1.0)))

        for _step in range(max_steps_per_ep):
            var result = agent.select_action(obs, training=True)
            var action = result[0].copy()
            var root_value = result[1]

            # Safety checks. Allow `|av| == MAX_ACTION` exactly because
            # fp32 `tanh(large)` saturates to 1.0 — the squashed-Gaussian
            # output's range is mathematically (-MAX, MAX) but at fp32
            # precision it includes the boundary.
            for d in range(ACT_DIM):
                var av = Float64(action[d])
                if av != av:
                    any_action_nan = True
                if av > MAX_ACTION + 1.0e-5 or av < -MAX_ACTION - 1.0e-5:
                    any_action_oor = True
                if av > max_action_seen:
                    max_action_seen = av
                if -av > max_action_seen:
                    max_action_seen = -av
            if not _is_finite(root_value):
                any_value_nan = True

            # Synthetic env step.
            var reward = Float64(0.0)
            for d in range(ACT_DIM):
                var av = Float64(action[d])
                reward -= av * av
            var done = _step == max_steps_per_ep - 1

            # Random next obs (random walk).
            var next_obs = List[Scalar[dtype]](capacity=OBS)
            for d in range(OBS):
                var v = Float64(obs[d]) + random_float64(-0.1, 0.1)
                if v > 1.0:
                    v = 1.0
                if v < -1.0:
                    v = -1.0
                next_obs.append(Scalar[dtype](v))

            agent.store_transition(obs, action, reward, root_value, done)
            total_actions_collected += 1

            obs = next_obs^
            if done:
                break

    print("    transitions collected:", total_actions_collected)
    print("    buffer.size           =", agent.state.buffer.size)
    print("    max |action| seen     =", max_action_seen)

    _expect(
        not any_action_nan,
        "no NaN actions across rollout",
        passed, total,
    )
    _expect(
        not any_action_oor,
        "all chosen actions in [-MAX, MAX] per dim (fp32 saturation OK)",
        passed, total,
    )
    _expect(
        not any_value_nan,
        "all root values finite",
        passed, total,
    )
    _expect(
        agent.state.buffer.size >= 200,
        "replay state populated (≥ 200 transitions)",
        passed, total,
    )
    _expect(
        agent.state.is_ready(),
        "replay buffer ready to train",
        passed, total,
    )
    # Episode buffers should have been reset on done.
    var any_residual = False
    for env_id in range(agent.n_envs):
        if len(agent._episode_obs[env_id]) > 0:
            any_residual = True
        if len(agent._episode_actions[env_id]) > 0:
            any_residual = True
        if len(agent._episode_action_targets[env_id]) > 0:
            any_residual = True
    _expect(
        not any_residual,
        "episode buffers reset cleanly after each done",
        passed, total,
    )

    # ── GPU state + train ─────────────────────────────────────────────────
    print()
    print("--- Building GPU state + training 100 steps ---")
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()

    var num_train_steps = 100
    var L_total_first = Float64(0.0)
    var L_total_last = Float64(0.0)
    var L_R_last = Float64(0.0)
    var L_P_last = Float64(0.0)
    var L_V_last = Float64(0.0)
    var L_G_last = Float64(0.0)
    var any_nan = False
    for step in range(num_train_steps):
        var t = agent.train_step_gpu(gpu, ctx)
        var L_total = t[0]
        var L_R = t[1]
        var L_P = t[2]
        var L_V = t[3]
        var L_G = t[4]
        if not _is_finite(L_total):
            any_nan = True
        if step == 0:
            L_total_first = L_total
        if step == num_train_steps - 1:
            L_total_last = L_total
            L_R_last = L_R
            L_P_last = L_P
            L_V_last = L_V
            L_G_last = L_G
        if step % 20 == 0:
            print(
                "    step", step,
                ": L_total=", L_total,
                " L_R=", L_R,
                " L_P=", L_P,
                " L_V=", L_V,
                " L_G=", L_G,
            )

    print()
    print("--- After training ---")
    print("    L_total: first =", L_total_first, ", last =", L_total_last)
    print("    L_R last =", L_R_last)
    print("    L_P last =", L_P_last)
    print("    L_V last =", L_V_last)
    print("    L_G last =", L_G_last)

    _expect(
        not any_nan,
        "no NaN/Inf encountered across 100 train steps",
        passed, total,
    )
    _expect(
        _is_finite(L_R_last)
        and _is_finite(L_P_last)
        and _is_finite(L_V_last)
        and _is_finite(L_G_last),
        "all four loss components finite at end",
        passed, total,
    )
    _expect(
        L_total_last < 0.95 * L_total_first,
        "composite L decreased ≥ 5%",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
