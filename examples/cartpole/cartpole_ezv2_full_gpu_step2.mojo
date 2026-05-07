"""EZ-V2 CartPole — Step 2: GPU MCTS for action selection (N_ENVS=1).

Builds on Step 1. The CPU `agent.select_action(obs)` call is replaced by
driving `run_gumbel_search_gpu` directly, reading the existing GPU
networks (`gpu.representation/dynamics/prediction`). After search:
  • download `policies_out` to host (the improved policy);
  • download root `visit_count`/`total_value` to host and compute
    SVE = Σ total_value(root, a) / Σ visit_count(root, a);
  • sample an action with temperature on host using the same logic as
    the CPU `select_action`.

The agent's CPU MCTS engine (`agent.mcts`) is no longer touched in the
collection loop, but it stays alive to serve `agent.reanalyze`. CPU
weight mirror is refreshed at SYNC_INTERVAL train_steps, same as
Step 0/1, so reanalyze sees fresh weights.

Numerical caveat: GPU MCTS uses PhiloxRandom + float32, CPU MCTS used
the `random` module + float64. Per-call action choices will diverge —
but the *distributions* should agree (validated by
`test_ezv2_gumbel_search_gpu.mojo`). We test end-to-end training health,
not bit parity.

Plan:    docs/EZV2_FULL_GPU_PLAN.md
Gate:    rerun, compare against logs/ezv2_full_gpu_step1.log.
         Loss curves and ep-return curves should track within ±10%
         (wider band than Step 1 because the search RNG differs).

Run:
    pixi run mojo run -I . examples/cartpole/cartpole_ezv2_full_gpu_step2.mojo
"""

from std.math import abs, exp, log
from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from gpu import block_dim, block_idx, thread_idx
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    EZV2GPUStateBase,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_mcts import EZV2GPUMCTSState
from mojo_rl.deep_agents.efficient_zero_v2.strategies import compute_sve
from mojo_rl.envs.cartpole import CartPoleEnv
from mojo_rl.nn.constants import dtype


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


def _mean(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = Float64(0.0)
    for i in range(len(xs)):
        s += xs[i]
    return s / Float64(len(xs))


@always_inline
def _extract_obs_kernel[
    BATCH: Int,
    STATE_SIZE: Int,
    OBS_DIM: Int,
](
    states: LayoutTensor[
        dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
):
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    for d in range(OBS_DIM):
        obs[i, d] = states[i, d]


def main() raises:
    print("=== EZ-V2 CartPole demo — Step 2 (GPU MCTS, N_ENVS=1) ===")

    # Match Step 0/1 baseline config exactly.
    comptime NUM_ENV_STEPS = 50_000
    comptime TRAIN_INTERVAL = 4
    comptime LOG_EVERY = 2_000
    comptime EVAL_WINDOW = 100
    comptime CONVERGENCE_TARGET = 450.0

    comptime SYNC_INTERVAL = 50
    comptime TARGET_SYNC_INTERVAL = 200
    comptime REANALYZE_INTERVAL = 200
    comptime REANALYZE_SAMPLES = 32
    comptime REANALYZE_WARMUP = 1000

    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=128,
        HIDDEN=128,
        PROJ=256,
        PRED_BOTTLENECK=128,
        BINS=21,
        BS=64,
        K_UNROLL=3,
        N_TD=5,
        SIMS=16,
        NODES=64,
        K_GUMBEL=2,
        LR=Float64(5e-4),
        LAMBDA_G=Float64(1.0),
    ]

    comptime ACT = Config.action_dim
    comptime OBS = Config.obs_dim
    comptime LATENT = Config.latent_dim
    comptime BINS = Config.num_bins
    comptime SIMS = Config.num_simulations
    comptime NODES = Config.max_nodes
    comptime MAX_K = Config.num_root_candidates
    comptime TEMPERATURE = 1.0  # init agent temp; agent.temperature is the canonical value

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99,
        v_min=-15.0,
        v_max=15.0,
        temperature=TEMPERATURE,
        temperature_decay_steps=10_000_000,
    )
    var env = CartPoleEnv[DType.float32]()
    var ctx = DeviceContext()

    print()
    print("--- Run config ---")
    print("    NUM_ENV_STEPS         =", NUM_ENV_STEPS)
    print("    TRAIN_INTERVAL        =", TRAIN_INTERVAL)
    print("    SYNC_INTERVAL         =", SYNC_INTERVAL, "train_steps")
    print("    TARGET_SYNC_INTERVAL  =", TARGET_SYNC_INTERVAL, "train_steps")
    print("    REANALYZE_INTERVAL    =", REANALYZE_INTERVAL, "train_steps")
    print("    REANALYZE_SAMPLES     =", REANALYZE_SAMPLES)
    print("    REANALYZE_WARMUP      =", REANALYZE_WARMUP, "train_steps")
    print("    EVAL_WINDOW           =", EVAL_WINDOW, "episodes")
    print("    CONVERGENCE_TARGET    =", CONVERGENCE_TARGET)
    print(
        "    Config: LATENT=", LATENT,
        " PROJ=", Config.proj_dim,
        " BINS=", BINS,
    )
    print(
        "    BS=", Config.batch_size,
        " K_UNROLL=", Config.unroll_steps,
        " N_TD=", Config.td_steps,
        " SIMS=", SIMS,
        " K_GUMBEL=", MAX_K,
    )
    print(
        "    λ_R=", Config.lambda_reward,
        " λ_P=", Config.lambda_policy,
        " λ_V=", Config.lambda_value,
        " λ_G=", Config.lambda_consistency,
    )
    print()

    # ── Allocate GPU state (networks) + initial upload ───────────────────
    print("--- Allocating GPU state ---")
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()

    # ── Allocate GPU env buffers (Step 1 — N_ENVS=1) ─────────────────────
    comptime N_ENVS = 1
    comptime STATE_SIZE = CartPoleEnv[DType.float32].STATE_SIZE  # 5
    comptime OBS_DIM = CartPoleEnv[DType.float32].OBS_DIM  # 4

    var states_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var dones_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)

    var host_obs = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS_DIM)
    var host_action = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var host_reward = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var host_done = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

    # ── Allocate GPU MCTS state + workspace ──────────────────────────────
    print("    GPU env buffers allocated")
    var mcts_gpu = EZV2GPUMCTSState[
        N_ENVS, NODES, ACT, LATENT, BINS, MAX_K
    ](ctx)

    # Workspace big enough for any of the three networks at this batch.
    comptime WS_R = Config.RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = Config.DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS_AB = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS = MAX_WS_AB if MAX_WS_AB > WS_P else WS_P
    comptime WS_TOTAL = N_ENVS * MAX_WS if MAX_WS > 0 else 1
    var workspace = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

    # Host buffers for downloading search outputs each step.
    # policies_out: [N_ENVS × ACT]
    # visit_count, total_value: [N_ENVS × NODES × ACT] (we read root only)
    var host_policies = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    var host_visits = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * ACT
    )
    var host_total_value = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * ACT
    )

    # Initial reset on GPU. Then populate obs_buf from state via the
    # extract kernel (reset_kernel_gpu only writes state).
    CartPoleEnv[DType.float32].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, states_buf, rng_seed=UInt64(2026)
    )

    comptime extract_obs = _extract_obs_kernel[N_ENVS, STATE_SIZE, OBS_DIM]
    comptime tpb = 32
    comptime blocks = (N_ENVS + tpb - 1) // tpb

    @parameter
    @always_inline
    def _launch_extract_obs() raises:
        var st = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var ob = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        ctx.enqueue_function[extract_obs, extract_obs](
            st, ob, grid_dim=(blocks,), block_dim=(tpb,)
        )

    _launch_extract_obs()
    ctx.synchronize()
    print("    GPU MCTS state + workspace ready")
    print("    GPU env buffers ready, initial reset complete")
    print()

    # ── Training loop ────────────────────────────────────────────────────
    var ep_returns = List[Float64]()
    var ep_return = Float64(0.0)

    var num_train_calls = 0
    var num_gpu_syncs = 0
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    var best_ep_return = Float64(0.0)
    var step_seed: UInt64 = 1
    var mcts_seed: UInt32 = 0

    var t0 = perf_counter_ns()

    # We'll need a per-step CPU obs view ONLY for store_transition. The
    # MCTS reads obs_buf directly on device — no host roundtrip needed
    # for action selection itself.
    ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
    ctx.synchronize()
    var obs = List[Scalar[dtype]]()
    for d in range(OBS_DIM):
        obs.append(host_obs[d])

    for env_step in range(NUM_ENV_STEPS):
        # ── GPU MCTS — replaces agent.select_action(obs) ─────────────────
        # Drive Gumbel search via the wrapper on the GPU state. See the
        # method's docstring for the rationale (Mojo type-alias unification).
        gpu.mcts_search[N_ENVS, NODES, MAX_K, SIMS](
            ctx,
            mcts_gpu,
            obs_buf,
            workspace,
            v_min=agent.v_min,
            v_max=agent.v_max,
            gamma=agent.gamma,
            rng_seed=mcts_seed,
            apply_legal=False,
            k_actual=MAX_K,
        )
        mcts_seed += UInt32(1)

        # Download search outputs.
        ctx.enqueue_copy(host_policies.unsafe_ptr(), mcts_gpu.policies_out)
        ctx.enqueue_copy(host_visits.unsafe_ptr(), mcts_gpu.visit_count)
        ctx.enqueue_copy(
            host_total_value.unsafe_ptr(), mcts_gpu.total_value
        )
        ctx.synchronize()

        # Compute SVE on host from root-node stats (env=0, node=0, *).
        var sum_value = Float64(0.0)
        var sum_visits = 0
        for a in range(ACT):
            sum_value += Float64(host_total_value[a])
            sum_visits += Int(Float64(host_visits[a]))
        var root_value = compute_sve(sum_value, sum_visits)

        # Build the improved policy as InlineArray[Float64, ACT].
        var policy = InlineArray[Float64, ACT](uninitialized=True)
        for a in range(ACT):
            policy[a] = Float64(host_policies[a])

        # Action sampling — mirror the CPU `select_action` logic exactly.
        var action: Int
        if agent.temperature < 0.01:
            action = 0
            var best = policy[0]
            for a in range(1, ACT):
                if policy[a] > best:
                    best = policy[a]
                    action = a
        else:
            var temp_policy = InlineArray[Float64, ACT](uninitialized=True)
            var inv_t = 1.0 / agent.temperature
            var sum_p = Float64(0.0)
            for a in range(ACT):
                if policy[a] > 0.0:
                    temp_policy[a] = exp(inv_t * log(policy[a]))
                else:
                    temp_policy[a] = Float64(0.0)
                sum_p += temp_policy[a]
            if sum_p > 0.0:
                for a in range(ACT):
                    temp_policy[a] /= sum_p
            else:
                for a in range(ACT):
                    temp_policy[a] = 1.0 / Float64(ACT)
            var u = random_float64(0.0, 1.0)
            var cumsum = Float64(0.0)
            action = ACT - 1
            for a in range(ACT):
                cumsum += temp_policy[a]
                if u <= cumsum:
                    action = a
                    break

        # ── Upload action and step env on GPU ────────────────────────────
        host_action[0] = Scalar[dtype](Float64(action))
        ctx.enqueue_copy(actions_buf, host_action.unsafe_ptr())

        CartPoleEnv[DType.float32].step_kernel_gpu[
            N_ENVS, STATE_SIZE, OBS_DIM
        ](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=step_seed,
        )

        ctx.enqueue_copy(host_reward.unsafe_ptr(), rewards_buf)
        ctx.enqueue_copy(host_done.unsafe_ptr(), dones_buf)
        ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
        ctx.synchronize()

        var reward = Float64(host_reward[0])
        var done = host_done[0] > Scalar[dtype](0.5)

        agent.store_transition(
            obs, action, reward, policy, root_value, done
        )
        ep_return += reward

        if done:
            ep_returns.append(ep_return)
            if ep_return > best_ep_return:
                best_ep_return = ep_return
            ep_return = Float64(0.0)

            step_seed += 1
            CartPoleEnv[DType.float32].reset_kernel_gpu[N_ENVS, STATE_SIZE](
                ctx, states_buf, rng_seed=step_seed
            )
            _launch_extract_obs()
            ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
            ctx.synchronize()

        # Refresh CPU obs view from device (for next iter's store_transition).
        obs = List[Scalar[dtype]]()
        for d in range(OBS_DIM):
            obs.append(host_obs[d])

        # ── GPU train (unchanged from Step 0/1) ──────────────────────────
        if (
            agent.state.is_ready()
            and (env_step + 1) % TRAIN_INTERVAL == 0
        ):
            var t = agent.train_step_gpu(gpu, ctx)
            num_train_calls += 1
            last_L_R = t[1]
            last_L_P = t[2]
            last_L_V = t[3]
            last_L_G = t[4]
            if not _is_finite(t[0]):
                any_nan_loss = True

            if num_train_calls % SYNC_INTERVAL == 0:
                gpu.download_to(agent.state, ctx)
                ctx.synchronize()
                num_gpu_syncs += 1

            if num_train_calls % TARGET_SYNC_INTERVAL == 0:
                agent.update_target_networks(tau=1.0)
            if (
                num_train_calls >= REANALYZE_WARMUP
                and num_train_calls % REANALYZE_INTERVAL == 0
            ):
                _ = agent.reanalyze(num_samples=REANALYZE_SAMPLES)

        step_seed += 1

        if (env_step + 1) % LOG_EVERY == 0:
            var t_now = perf_counter_ns()
            var wall_s = Float64(t_now - t0) / 1.0e9
            var window = 30
            var n_eps = len(ep_returns)
            var recent = List[Float64]()
            var start = (
                n_eps - window if n_eps > window else 0
            )
            for i in range(start, n_eps):
                recent.append(ep_returns[i])
            print(
                "[step ", env_step + 1,
                " ep=", n_eps,
                " train=", num_train_calls,
                " syncs=", num_gpu_syncs,
                " wall=", wall_s, "s",
                "] recent_mean_ret=", _mean(recent),
                "  best=", best_ep_return,
                "  L=(R", last_L_R,
                ", P", last_L_P,
                ", V", last_L_V,
                ", G", last_L_G, ")",
            )

    var t_end = perf_counter_ns()
    var wall_s_total = Float64(t_end - t0) / 1.0e9

    print()
    print("--- Final GPU → CPU sync ---")
    gpu.download_to(agent.state, ctx)
    ctx.synchronize()
    num_gpu_syncs += 1

    print()
    print("=== Run summary ===")
    print("    wall time             =", wall_s_total, "s")
    print("    env steps             =", NUM_ENV_STEPS)
    print("    train_step_gpu calls  =", num_train_calls)
    print("    GPU→CPU syncs         =", num_gpu_syncs)
    print("    episodes finished     =", len(ep_returns))
    print("    best episode return   =", best_ep_return)
    print("    any NaN loss          =", any_nan_loss)
    print("    final loss components:")
    print("        L_R =", last_L_R)
    print("        L_P =", last_L_P)
    print("        L_V =", last_L_V)
    print("        L_G =", last_L_G)

    var n_eps = len(ep_returns)
    if n_eps < EVAL_WINDOW:
        print()
        print(
            "FAIL: only", n_eps,
            "episodes finished — need ≥", EVAL_WINDOW,
            "to evaluate.",
        )
        return

    var last_window = List[Float64]()
    for i in range(n_eps - EVAL_WINDOW, n_eps):
        last_window.append(ep_returns[i])
    var final_mean = _mean(last_window)
    var first_window = List[Float64]()
    var first_n = EVAL_WINDOW if EVAL_WINDOW < n_eps else n_eps
    for i in range(first_n):
        first_window.append(ep_returns[i])
    var initial_mean = _mean(first_window)

    print(
        "    first ",
        EVAL_WINDOW,
        " ep mean return =",
        initial_mean,
    )
    print(
        "    last ",
        EVAL_WINDOW,
        " ep mean return =",
        final_mean,
    )
    print("    convergence target    =", CONVERGENCE_TARGET)

    print()
    if any_nan_loss:
        print("FAIL: NaN/Inf loss during training")
    elif final_mean >= CONVERGENCE_TARGET:
        print(
            "PASS: CartPole converged ≥",
            CONVERGENCE_TARGET,
            "(got",
            final_mean,
            ")",
        )
    else:
        print(
            "INCONCLUSIVE: CartPole did not hit",
            CONVERGENCE_TARGET,
            "— got",
            final_mean,
            "(improvement",
            initial_mean,
            "→",
            final_mean,
            "= ",
            final_mean - initial_mean,
            ")",
        )
