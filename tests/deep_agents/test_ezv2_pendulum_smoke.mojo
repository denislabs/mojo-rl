"""Pendulum smoke test for the continuous EZ-V2 agent (Phase 3.4.2).

Drives `GenericEZV2ContinuousAgent` on the real `PendulumEnv` for a tight
window — 5 warmup episodes (1000 env steps) + 60 train steps — and
verifies the wiring works end-to-end on a real continuous-control env.

Not a convergence test. Pendulum needs ~30k env steps and 30+ minutes of
GPU time; that lives in `examples/pendulum/ezv2_pendulum_training.mojo`.
Here we only check:

    (a) no NaN actions / values during rollout,
    (b) all chosen actions in [-MAX_ACTION, MAX_ACTION],
    (c) replay state reaches `is_ready()`,
    (d) train_step_gpu runs without NaN over 60 calls,
    (e) all four loss components finite.
"""

from std.random import seed
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    EZV2GPUStateBase,
    GenericEZV2ContinuousAgent,
)
from mojo_rl.envs.pendulum import PendulumEnv
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
    print("=== EZ-V2 continuous Pendulum smoke (Phase 3.4) ===")
    var passed = 0
    var total = 0

    # Tiny config — cheap enough to keep the test under a couple minutes.
    comptime Config = EZV2ContinuousMLPConfig[
        OBS=3,
        ACT_DIM=1,
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
        K_ROOT=4,
        K_NON_ROOT=2,
        MAX_ACTION=2.0,
        MIN_STD=0.1,
    ]

    comptime NUM_EPISODES = 5
    comptime MAX_STEPS_PER_EPISODE = 200

    seed(2026)
    var env = PendulumEnv[dtype]()
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99, v_min=-20.0, v_max=2.0, temperature=1.0,
    )
    var ctx = DeviceContext()
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()

    # ── Rollout ─────────────────────────────────────────────────────────
    print()
    print("--- Rolling out", NUM_EPISODES, "Pendulum episodes ---")
    var any_action_nan = False
    var any_action_oor = False
    var any_value_nan = False
    var ep_returns = List[Float64]()
    var total_steps = 0

    for _ep in range(NUM_EPISODES):
        var obs = env.reset_obs_list()
        var ep_reward = Float64(0.0)
        for _step in range(MAX_STEPS_PER_EPISODE):
            var sel = agent.select_action(obs, training=True)
            var action = sel[0].copy()
            var root_value = sel[1]

            for d in range(Config.action_dim):
                var av = Float64(action[d])
                if av != av:
                    any_action_nan = True
                if av > 2.0 + 1.0e-5 or av < -2.0 - 1.0e-5:
                    any_action_oor = True
            if not _is_finite(root_value):
                any_value_nan = True

            var step_result = env.step_continuous_vec(action)
            var next_obs = step_result[0].copy()
            var reward = Float64(step_result[1])
            var done = step_result[2]
            ep_reward += reward
            agent.store_transition(obs, action, reward, root_value, done)
            total_steps += 1
            obs = next_obs^
            if done:
                break
        ep_returns.append(ep_reward)

    print("    episodes done   :", len(ep_returns))
    print("    total env steps :", total_steps)
    var sum_r = Float64(0.0)
    for r in ep_returns:
        sum_r += r
    print(
        "    mean ep reward  :",
        sum_r / Float64(len(ep_returns)) if len(ep_returns) > 0 else 0.0,
    )

    _expect(not any_action_nan, "no NaN actions during rollout", passed, total)
    _expect(
        not any_action_oor,
        "actions in [-MAX, MAX]",
        passed, total,
    )
    _expect(
        not any_value_nan,
        "root values finite during rollout",
        passed, total,
    )
    _expect(
        agent.state.is_ready(),
        "replay state ready to train after warmup",
        passed, total,
    )

    # ── Train ───────────────────────────────────────────────────────────
    print()
    print("--- Running 60 train_step_gpu calls ---")
    var num_train = 60
    var any_nan_loss = False
    var L_R_last = Float64(0.0)
    var L_P_last = Float64(0.0)
    var L_V_last = Float64(0.0)
    var L_G_last = Float64(0.0)
    for step in range(num_train):
        var t = agent.train_step_gpu(gpu, ctx)
        if not _is_finite(t[0]):
            any_nan_loss = True
        if step == num_train - 1:
            L_R_last = t[1]
            L_P_last = t[2]
            L_V_last = t[3]
            L_G_last = t[4]
        if step % 15 == 0:
            print(
                "    step", step,
                " L_total=", t[0],
                " L_R=", t[1], " L_P=", t[2],
                " L_V=", t[3], " L_G=", t[4],
            )

    _expect(
        not any_nan_loss,
        "no NaN/Inf across 60 train_step_gpu calls",
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

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
