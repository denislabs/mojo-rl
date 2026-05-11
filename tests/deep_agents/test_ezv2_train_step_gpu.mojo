"""GPU train_step smoke test for EZ-V2.

Mirrors `test_ezv2_train_step.mojo` (CPU) but uses the new
`train_step_gpu` path:

  1. Roll out a small fixed dataset on CPU (search runs CPU-side; replay
     buffer + MCTS targets stay CPU-resident).
  2. Build `EZV2GPUStateBase`, upload params from the agent's CPU
     state.
  3. Loop `train_step_gpu()` for a few hundred steps.
  4. Verify:
     • no NaN/Inf in any loss component or composite,
     • composite loss drops meaningfully,
     • L_V and L_R (the two simplest pred-head CE paths) drop too,
     • after training, the GPU weights downloaded back to the CPU
       state still produce sane action selection (smoke check that
       the upload/download path didn't corrupt anything).

This is a *training-machinery* test, not a convergence test — we don't
expect CartPole to be solved here.
"""

from std.math import abs
from std.random import seed
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    EZV2GPUStateBase,
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


def main() raises:
    print("=== EZ-V2 GPU train_step smoke test ===")
    var passed = 0
    var total = 0

    # Match the CPU smoke test config so behaviour is comparable.
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
        gamma=0.99, v_min=-10.0, v_max=10.0, temperature=1.0,
    )
    var env = CartPoleEnv[DType.float32]()
    var ctx = DeviceContext()

    # ── Build GPU state + upload initial weights ────────────────────────
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)

    # ── Roll out replay buffer with a fixed dataset ──────────────────────
    print()
    print("--- Filling replay buffer ---")
    var num_episodes = 60
    var max_steps_per_ep = 60
    for _ep in range(num_episodes):
        var obs = env.reset_obs_list()
        for _step in range(max_steps_per_ep):
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
            obs = next_obs^
            if done:
                break
    print("    replay buf size =", agent.state.buffer.size)

    _expect(
        agent.state.is_ready(),
        "replay buffer holds enough data to train",
        passed, total,
    )

    # ── Initial CPU-side loss snapshot ───────────────────────────────────
    print()
    print("--- Initial loss components (CPU forward-only) ---")
    var initial = agent.compute_loss_components()
    var L_R0 = initial[0]
    var L_P0 = initial[1]
    var L_V0 = initial[2]
    var L_G0 = initial[3]
    var L0_total = (
        Config.lambda_reward * L_R0
        + Config.lambda_policy * L_P0
        + Config.lambda_value * L_V0
        + Config.lambda_consistency * L_G0
    )
    print("    L_R =", L_R0, "  L_P =", L_P0)
    print("    L_V =", L_V0, "  L_G =", L_G0)
    print("    L_total =", L0_total)

    # ── Run train_step_gpu and watch losses ──────────────────────────────
    print()
    print("--- Running 100 train_step_gpu calls ---")
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
                "    step",
                step,
                ": L_total=",
                L_total,
                " L_R=",
                L_R,
                " L_P=",
                L_P,
                " L_V=",
                L_V,
                " L_G=",
                L_G,
            )

    print()
    print("--- After training ---")
    print("    L_total: first =", L_total_first, ", last =", L_total_last)
    print("    L_R: ", L_R0, "→", L_R_last)
    print("    L_P: ", L_P0, "→", L_P_last)
    print("    L_V: ", L_V0, "→", L_V_last)
    print("    L_G: ", L_G0, "→", L_G_last)

    _expect(
        not any_nan,
        "no NaN/Inf encountered across train_step_gpu calls",
        passed, total,
    )
    _expect(
        _is_finite(L_total_last),
        "final L_total is finite",
        passed, total,
    )
    _expect(
        L_total_last < 0.9 * L_total_first,
        "composite L decreased ≥ 10% over 100 GPU steps",
        passed, total,
    )
    _expect(
        L_V_last < 0.9 * L_V0,
        "L_V dropped ≥ 10% (value head trained via pred backward on GPU)",
        passed, total,
    )
    _expect(
        L_R_last < 0.9 * L_R0,
        "L_R dropped ≥ 10% (reward head trained via dyn backward on GPU)",
        passed, total,
    )

    # ── Sync GPU weights back to CPU and confirm action selection works ──
    print()
    print("--- Sync GPU weights → CPU + smoke action selection ---")
    gpu.download_to(agent.state, ctx)

    var probe_obs = env.reset_obs_list()
    var probe_result = agent.select_action(probe_obs, training=False)
    var probe_action = probe_result[0]
    var probe_value = probe_result[2]
    print("    probe action =", probe_action, "  probe SVE =", probe_value)
    _expect(
        probe_action >= 0 and probe_action < Config.action_dim,
        "post-train CPU action selection returns a legal action",
        passed, total,
    )
    _expect(
        _is_finite(probe_value),
        "post-train CPU SVE value is finite",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
