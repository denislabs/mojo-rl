"""GPU-sampling variant smoke test for EZ-V2.

Mirrors `test_ezv2_train_step_gpu.mojo` but exercises the new
`train_step_gpu_with_replay` entry point — which replaces section 1's
host-side priority sampling with `ezv2_gpu_sample_and_gather`
(kernels 1-4 from `gpu_sampling.mojo`).

The CPU `state.priorities` array stays the source of truth between
`gpu_replay.upload_from_cpu` syncs and `_flush_episode` writes; the new
method downloads the GPU-picked `batch_start_idx` and runs the existing
host writeback for section 10. Default `VALUE_TARGET_MODE = SEARCH`
config is required (SARSA/MIXED still need a host target-net forward
that hasn't been ported yet — see deferred work item 5 in
`docs/EZV2_FULL_GPU_PLAN.md`).

Verifies:
  • no NaN/Inf across 100 train_step_gpu_with_replay calls,
  • composite loss drops ≥ 10%,
  • L_V and L_R drop too.

This is *not* a parity test against `train_step_gpu` — the two methods
sample different transitions (host random vs Philox), so loss
trajectories differ. We only test that the GPU-sampling path is
healthy on its own.
"""

from std.random import seed
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    EZV2GPUStateBase,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_replay import (
    EZV2GPUReplayBuffer,
)
from mojo_rl.envs.cartpole import CartPoleEnv


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
    print("=== EZ-V2 train_step_gpu_with_replay smoke test ===")
    var passed = 0
    var total = 0

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

    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)

    # ── Roll out replay buffer ──────────────────────────────────────────
    print()
    print("--- Filling CPU replay buffer ---")
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
    print("    CPU replay buf size =", agent.state.buffer.size)
    _expect(
        agent.state.is_ready(),
        "CPU replay buffer holds enough data to train",
        passed, total,
    )

    # ── Build + sync GPU replay ─────────────────────────────────────────
    print()
    print("--- Allocating GPU replay buffer + syncing from CPU ---")
    var gpu_replay = EZV2GPUReplayBuffer[
        50000, Config.obs_dim, Config.action_dim
    ](ctx)
    gpu_replay.upload_from_cpu(agent.state, ctx)
    gpu_replay.max_priority = agent.max_priority
    ctx.synchronize()
    print("    GPU replay size =", gpu_replay.size)
    print("    GPU replay ptr  =", gpu_replay.ptr)

    # ── Initial CPU loss snapshot ───────────────────────────────────────
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

    # ── Run train_step_gpu_with_replay ──────────────────────────────────
    print()
    print("--- Running 100 train_step_gpu_with_replay calls ---")
    var num_train_steps = 100
    var L_total_first = Float64(0.0)
    var L_total_last = Float64(0.0)
    var L_R_last = Float64(0.0)
    var L_P_last = Float64(0.0)
    var L_V_last = Float64(0.0)
    var L_G_last = Float64(0.0)
    var any_nan = False
    for step in range(num_train_steps):
        var t = agent.train_step_gpu_with_replay(
            gpu, gpu_replay, ctx, UInt32(7 + step)
        )
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
    print("--- After GPU-sampling training ---")
    print("    L_total: first =", L_total_first, ", last =", L_total_last)
    print("    L_R: ", L_R0, "→", L_R_last)
    print("    L_P: ", L_P0, "→", L_P_last)
    print("    L_V: ", L_V0, "→", L_V_last)
    print("    L_G: ", L_G0, "→", L_G_last)

    _expect(
        not any_nan,
        "no NaN/Inf encountered across 100 calls",
        passed, total,
    )
    _expect(
        _is_finite(L_total_last),
        "final L_total is finite",
        passed, total,
    )
    _expect(
        L_total_last < 0.9 * L_total_first,
        "composite L decreased ≥ 10% over 100 GPU-sampling steps",
        passed, total,
    )
    _expect(
        L_V_last < 0.9 * L_V0,
        "L_V dropped ≥ 10% (value head trains under GPU sampling)",
        passed, total,
    )
    _expect(
        L_R_last < 0.9 * L_R0,
        "L_R dropped ≥ 10% (reward head trains under GPU sampling)",
        passed, total,
    )

    # ── Sync GPU weights → CPU + smoke action selection ──────────────────
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
