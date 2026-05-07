"""EZ-V2 CartPole — Step 1 toward full-GPU training.

Same as `cartpole_ezv2_gpu.mojo` (Step 0 baseline) except the CPU
`env.step_obs(action)` call is replaced by `CartPoleEnv.step_kernel_gpu`
on a single env. The MCTS / replay paths stay on CPU exactly as in
Step 0, so any divergence is purely from running physics on GPU
(Float32) instead of CPU (Float32 too — `CartPoleEnv[DType.float32]`).
The reset RNG also moves from CPython `random_float64` to PhiloxRandom,
so per-episode initial states differ — but the *training trajectory
shape* should match Step 0 within stochastic noise.

Plan:    docs/EZV2_FULL_GPU_PLAN.md
Gate:    rerun, compare against logs/ezv2_full_gpu_baseline_step0.log.
         Same ep-return curve (±5% noise), same loss trajectory shape.
         Wall time may regress at N=1 — host↔device round-trip per
         step is the cost of plumbing validation. Speed comes back at
         N_ENVS≥4 in Step 3.

Run:
    pixi run mojo run -I . examples/cartpole/cartpole_ezv2_full_gpu_step1.mojo
"""

from std.math import abs
from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from gpu import block_dim, block_idx, thread_idx
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    EZV2DiscreteGPUState,
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


def _mean(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = Float64(0.0)
    for i in range(len(xs)):
        s += xs[i]
    return s / Float64(len(xs))


# Tiny launch helper: copy state[:OBS_DIM] → obs for the single env.
# CartPole's obs is the first 4 elements of state, so after reset (which
# only writes states_buf) we need this to populate obs_buf.
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
    print("=== EZ-V2 CartPole demo — Step 1 (GPU env step, N_ENVS=1) ===")

    # Match Step 0 baseline config exactly.
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

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99,
        v_min=-15.0,
        v_max=15.0,
        temperature=1.0,
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
        "    Config: LATENT=", Config.latent_dim,
        " PROJ=", Config.proj_dim,
        " BINS=", Config.num_bins,
    )
    print(
        "    BS=", Config.batch_size,
        " K_UNROLL=", Config.unroll_steps,
        " N_TD=", Config.td_steps,
        " SIMS=", Config.num_simulations,
        " K_GUMBEL=", Config.num_root_candidates,
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
    var gpu = EZV2DiscreteGPUState[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()
    print("    GPU state ready, initial upload complete")

    # ── Allocate GPU env buffers (Step 1 — N_ENVS=1) ─────────────────────
    # Matches the layout in `gpu_offpolicy_train.mojo` but at N=1.
    comptime N_ENVS = 1
    comptime STATE_SIZE = CartPoleEnv[DType.float32].STATE_SIZE  # 5
    comptime OBS_DIM = CartPoleEnv[DType.float32].OBS_DIM  # 4

    var states_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var dones_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)

    # Host-pinned buffers for per-step round-trip. At N_ENVS=1 we eat the
    # full DMA latency every step — Step 3 amortizes this with N_ENVS≥4.
    var host_obs = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS_DIM)
    var host_action = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var host_reward = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var host_done = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

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
    ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
    ctx.synchronize()
    print("    GPU env buffers ready, initial reset complete")
    print()

    # ── Training loop ────────────────────────────────────────────────────
    var ep_returns = List[Float64]()
    var ep_return = Float64(0.0)

    # `obs` is the canonical CPU view, fed to MCTS each step.
    var obs = List[Scalar[dtype]]()
    for d in range(OBS_DIM):
        obs.append(host_obs[d])

    var num_train_calls = 0
    var num_gpu_syncs = 0
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    var best_ep_return = Float64(0.0)
    var step_seed: UInt64 = 1

    var t0 = perf_counter_ns()

    for env_step in range(NUM_ENV_STEPS):
        # CPU MCTS, unchanged.
        var result = agent.select_action(obs, training=True)
        var action = result[0]
        var policy = result[1]
        var root_value = result[2]

        # Upload action (single int as float) to actions_buf.
        host_action[0] = Scalar[dtype](Float64(action))
        ctx.enqueue_copy(actions_buf, host_action.unsafe_ptr())

        # GPU env step. Writes rewards/dones/terminated and overwrites obs_buf
        # with the next-state obs.
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

        # Download outputs to host (single sync per step at N=1).
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

            # Reset GPU env. At N=1 a full reset is equivalent to selective
            # reset — the whole batch is "this env".
            step_seed += 1
            CartPoleEnv[DType.float32].reset_kernel_gpu[N_ENVS, STATE_SIZE](
                ctx, states_buf, rng_seed=step_seed
            )
            _launch_extract_obs()
            ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
            ctx.synchronize()

        # Refresh CPU obs view from device.
        obs = List[Scalar[dtype]]()
        for d in range(OBS_DIM):
            obs.append(host_obs[d])

        # Train on GPU (unchanged from Step 0).
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
            "to evaluate. Try increasing NUM_ENV_STEPS.",
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
