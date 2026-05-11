"""EZ-V2 CartPole — Step 4: GPU-resident replay buffer + verification.

Builds on Step 3. Allocates `EZV2GPUReplayBuffer` and bulk-mirrors the
CPU state's buffer + parallel arrays at every SYNC_INTERVAL train_step
boundary (same cadence as the network weight sync). At the end of the
run, downloads the GPU buffer back into host-pinned scratch and diffs
field-by-field against the CPU state.

`train_step_gpu` is unchanged from Step 3 — it still samples from the
CPU buffer + uploads sampled batches. Step 5 replaces that path with
GPU-side priority sampling that reads directly from the GPU buffer.

Plan:    docs/EZV2_FULL_GPU_PLAN.md
Gate:    rerun, training trajectory must match Step 3 within float32
         noise (the GPU buffer is inert during training — only Step 5
         starts reading it). Plus: end-of-run diff between CPU buffer
         and GPU-downloaded mirror must be 0 across all fields.

Run:
    pixi run mojo run -I . examples/cartpole/cartpole_ezv2_full_gpu_step4.mojo
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
from mojo_rl.deep_agents.efficient_zero_v2.gpu_replay import (
    EZV2GPUReplayBuffer,
)
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
    print("=== EZ-V2 CartPole demo — Step 4 (GPU replay buffer) ===")

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
    comptime N_ENVS = 4
    comptime CAP = 50000  # matches `EZV2DiscreteCPUState`'s default _CAP

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99,
        v_min=-15.0,
        v_max=15.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        n_envs=N_ENVS,
    )
    var env = CartPoleEnv[DType.float32]()
    var ctx = DeviceContext()

    print()
    print("--- Run config ---")
    print("    NUM_ENV_STEPS         =", NUM_ENV_STEPS)
    print("    N_ENVS                =", N_ENVS)
    print("    TRAIN_INTERVAL        =", TRAIN_INTERVAL, "(per env-batch)")
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
    print()

    # ── Allocate GPU state, env buffers, MCTS state, REPLAY BUFFER ───────
    print("--- Allocating GPU state ---")
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()

    comptime STATE_SIZE = CartPoleEnv[DType.float32].STATE_SIZE
    comptime OBS_DIM = CartPoleEnv[DType.float32].OBS_DIM

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

    var mcts_gpu = EZV2GPUMCTSState[
        N_ENVS, NODES, ACT, LATENT, BINS, MAX_K
    ](ctx)
    comptime WS_R = Config.RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = Config.DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS_AB = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS = MAX_WS_AB if MAX_WS_AB > WS_P else WS_P
    comptime WS_TOTAL = N_ENVS * MAX_WS if MAX_WS > 0 else 1
    var workspace = ctx.enqueue_create_buffer[dtype](WS_TOTAL)
    var host_policies = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    var host_visits = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * ACT
    )
    var host_total_value = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * ACT
    )

    # ── Step 4 — GPU replay buffer ──────────────────────────────────────
    var gpu_replay = EZV2GPUReplayBuffer[CAP, OBS, ACT](ctx)
    ctx.synchronize()
    print("    GPU replay buffer ready (CAP=", CAP, ")")

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
    print("    GPU env + MCTS state ready, initial reset complete")
    print()

    # ── Training loop ────────────────────────────────────────────────────
    var ep_returns = List[Float64]()
    var ep_return_per_env = List[Float64]()
    for _ in range(N_ENVS):
        ep_return_per_env.append(Float64(0.0))

    ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
    ctx.synchronize()
    var obs_per_env = List[List[Scalar[dtype]]]()
    for e in range(N_ENVS):
        var lst = List[Scalar[dtype]]()
        for d in range(OBS_DIM):
            lst.append(host_obs[e * OBS_DIM + d])
        obs_per_env.append(lst^)

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
    var total_env_steps = 0
    var num_buffer_uploads = 0

    var t0 = perf_counter_ns()

    while total_env_steps < NUM_ENV_STEPS:
        # GPU MCTS
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

        ctx.enqueue_copy(host_policies.unsafe_ptr(), mcts_gpu.policies_out)
        ctx.enqueue_copy(host_visits.unsafe_ptr(), mcts_gpu.visit_count)
        ctx.enqueue_copy(
            host_total_value.unsafe_ptr(), mcts_gpu.total_value
        )
        ctx.synchronize()

        var actions_per_env = List[Int]()
        var policies_per_env = List[InlineArray[Float64, ACT]]()
        var root_value_per_env = List[Float64]()
        for e in range(N_ENVS):
            var root_off = e * NODES * ACT
            var sum_value = Float64(0.0)
            var sum_visits = 0
            for a in range(ACT):
                sum_value += Float64(host_total_value[root_off + a])
                sum_visits += Int(Float64(host_visits[root_off + a]))
            root_value_per_env.append(compute_sve(sum_value, sum_visits))

            var policy = InlineArray[Float64, ACT](uninitialized=True)
            var pol_off = e * ACT
            for a in range(ACT):
                policy[a] = Float64(host_policies[pol_off + a])
            policies_per_env.append(policy)

            var action: Int
            if agent.temperature < 0.01:
                action = 0
                var best = policy[0]
                for a in range(1, ACT):
                    if policy[a] > best:
                        best = policy[a]
                        action = a
            else:
                var temp_policy = InlineArray[Float64, ACT](
                    uninitialized=True
                )
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
            actions_per_env.append(action)
            host_action[e] = Scalar[dtype](Float64(action))

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
        ctx.synchronize()

        var any_done = False
        for e in range(N_ENVS):
            var reward = Float64(host_reward[e])
            var done = host_done[e] > Scalar[dtype](0.5)

            agent.store_transition(
                obs_per_env[e],
                actions_per_env[e],
                reward,
                policies_per_env[e],
                root_value_per_env[e],
                done,
                env_id=e,
            )
            ep_return_per_env[e] += reward
            total_env_steps += 1

            if done:
                ep_returns.append(ep_return_per_env[e])
                if ep_return_per_env[e] > best_ep_return:
                    best_ep_return = ep_return_per_env[e]
                ep_return_per_env[e] = Float64(0.0)
                any_done = True

        if any_done:
            step_seed += 1
            CartPoleEnv[DType.float32].selective_reset_kernel_gpu[
                N_ENVS, STATE_SIZE
            ](ctx, states_buf, dones_buf, rng_seed=step_seed)
            _launch_extract_obs()

        ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
        ctx.synchronize()
        for e in range(N_ENVS):
            obs_per_env[e].clear()
            for d in range(OBS_DIM):
                obs_per_env[e].append(host_obs[e * OBS_DIM + d])

        # GPU train
        if (
            agent.state.is_ready()
            and (total_env_steps // N_ENVS) % TRAIN_INTERVAL == 0
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

                # Step 4 — bulk-mirror CPU buffer to GPU at the same
                # cadence as the network sync. Cheap (~3MB) and keeps
                # GPU buffer fresh for Step 5's GPU sampling.
                gpu_replay.upload_from_cpu(agent.state, ctx)
                gpu_replay.max_priority = agent.max_priority
                ctx.synchronize()
                num_buffer_uploads += 1

            if num_train_calls % TARGET_SYNC_INTERVAL == 0:
                agent.update_target_networks(tau=1.0)
            if (
                num_train_calls >= REANALYZE_WARMUP
                and num_train_calls % REANALYZE_INTERVAL == 0
            ):
                _ = agent.reanalyze(num_samples=REANALYZE_SAMPLES)

        step_seed += 1

        if total_env_steps % LOG_EVERY == 0 or (
            total_env_steps + N_ENVS > NUM_ENV_STEPS
            and total_env_steps != 0
        ):
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
                "[step ", total_env_steps,
                " ep=", n_eps,
                " train=", num_train_calls,
                " syncs=", num_gpu_syncs,
                " buf_up=", num_buffer_uploads,
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
    print("--- Final GPU → CPU sync (networks) ---")
    gpu.download_to(agent.state, ctx)
    ctx.synchronize()
    num_gpu_syncs += 1

    # ─── Final buffer upload before verification (catch any post-last-sync transitions) ─
    print("--- Final CPU → GPU buffer upload (then verify) ---")
    gpu_replay.upload_from_cpu(agent.state, ctx)
    gpu_replay.max_priority = agent.max_priority
    ctx.synchronize()
    num_buffer_uploads += 1

    # ─── Verification: GPU → host download, diff against CPU buffer ────
    var host_obs_dl = ctx.enqueue_create_host_buffer[dtype](CAP * OBS)
    var host_act_dl = ctx.enqueue_create_host_buffer[dtype](CAP * ACT)
    var host_rew_dl = ctx.enqueue_create_host_buffer[dtype](CAP)
    var host_done_dl = ctx.enqueue_create_host_buffer[dtype](CAP)
    var host_term_dl = ctx.enqueue_create_host_buffer[dtype](CAP)
    var host_pol_dl = ctx.enqueue_create_host_buffer[dtype](CAP * ACT)
    var host_val_dl = ctx.enqueue_create_host_buffer[dtype](CAP)
    var host_pri_dl = ctx.enqueue_create_host_buffer[dtype](CAP)
    var host_step_dl = ctx.enqueue_create_host_buffer[DType.uint32](CAP)

    ctx.enqueue_copy(host_obs_dl.unsafe_ptr(), gpu_replay.obs)
    ctx.enqueue_copy(host_act_dl.unsafe_ptr(), gpu_replay.actions)
    ctx.enqueue_copy(host_rew_dl.unsafe_ptr(), gpu_replay.rewards)
    ctx.enqueue_copy(host_done_dl.unsafe_ptr(), gpu_replay.dones)
    ctx.enqueue_copy(host_term_dl.unsafe_ptr(), gpu_replay.terminations)
    ctx.enqueue_copy(host_pol_dl.unsafe_ptr(), gpu_replay.mcts_policies)
    ctx.enqueue_copy(host_val_dl.unsafe_ptr(), gpu_replay.mcts_values)
    ctx.enqueue_copy(host_pri_dl.unsafe_ptr(), gpu_replay.priorities)
    ctx.enqueue_copy(host_step_dl.unsafe_ptr(), gpu_replay.step_at_write)
    ctx.synchronize()

    var max_diff_obs = Float64(0.0)
    var max_diff_act = Float64(0.0)
    var max_diff_rew = Float64(0.0)
    var max_diff_done = Float64(0.0)
    var max_diff_term = Float64(0.0)
    var max_diff_pol = Float64(0.0)
    var max_diff_val = Float64(0.0)
    var max_diff_pri = Float64(0.0)
    var step_match = True

    for i in range(CAP * OBS):
        var d = abs(
            Float64(host_obs_dl[i])
            - Float64(agent.state.buffer.obs[i])
        )
        if d > max_diff_obs:
            max_diff_obs = d
    for i in range(CAP * ACT):
        var d = abs(
            Float64(host_act_dl[i])
            - Float64(agent.state.buffer.actions[i])
        )
        if d > max_diff_act:
            max_diff_act = d
    for i in range(CAP):
        var d = abs(
            Float64(host_rew_dl[i])
            - Float64(agent.state.buffer.rewards[i])
        )
        if d > max_diff_rew:
            max_diff_rew = d
        d = abs(
            Float64(host_done_dl[i])
            - Float64(agent.state.buffer.dones[i])
        )
        if d > max_diff_done:
            max_diff_done = d
        d = abs(
            Float64(host_term_dl[i])
            - Float64(agent.state.buffer.terminations[i])
        )
        if d > max_diff_term:
            max_diff_term = d
        d = abs(
            Float64(host_val_dl[i])
            - Float64((agent.state.mcts_values + i)[])
        )
        if d > max_diff_val:
            max_diff_val = d
        d = abs(
            Float64(host_pri_dl[i])
            - Float64((agent.state.priorities + i)[])
        )
        if d > max_diff_pri:
            max_diff_pri = d
        if host_step_dl[i] != (agent.state.step_at_write + i)[]:
            step_match = False
    for i in range(CAP * ACT):
        var d = abs(
            Float64(host_pol_dl[i])
            - Float64((agent.state.mcts_policies + i)[])
        )
        if d > max_diff_pol:
            max_diff_pol = d

    print()
    print("=== GPU replay buffer verification ===")
    print("    CPU buffer.ptr        =", agent.state.buffer.ptr)
    print("    CPU buffer.size       =", agent.state.buffer.size)
    print("    GPU replay.ptr        =", gpu_replay.ptr)
    print("    GPU replay.size       =", gpu_replay.size)
    print("    max |obs|             =", max_diff_obs)
    print("    max |actions|         =", max_diff_act)
    print("    max |rewards|         =", max_diff_rew)
    print("    max |dones|           =", max_diff_done)
    print("    max |terminations|    =", max_diff_term)
    print("    max |mcts_policies|   =", max_diff_pol)
    print("    max |mcts_values|     =", max_diff_val)
    print("    max |priorities|      =", max_diff_pri)
    print("    step_at_write match   =", step_match)
    print("    buffer uploads        =", num_buffer_uploads)

    var verify_ok = (
        max_diff_obs == 0.0
        and max_diff_act == 0.0
        and max_diff_rew == 0.0
        and max_diff_done == 0.0
        and max_diff_term == 0.0
        and max_diff_pol == 0.0
        and max_diff_val == 0.0
        and max_diff_pri == 0.0
        and step_match
        and gpu_replay.ptr == agent.state.buffer.ptr
        and gpu_replay.size == agent.state.buffer.size
    )

    print()
    print("=== Run summary ===")
    print("    wall time             =", wall_s_total, "s")
    print("    env steps             =", total_env_steps)
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
    print("    GPU buffer mirror OK  =", verify_ok)

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
        "    first ", EVAL_WINDOW, " ep mean return =", initial_mean,
    )
    print(
        "    last ", EVAL_WINDOW, " ep mean return =", final_mean,
    )
    print("    convergence target    =", CONVERGENCE_TARGET)

    print()
    if any_nan_loss:
        print("FAIL: NaN/Inf loss during training")
    elif not verify_ok:
        print("FAIL: GPU replay buffer mirror verification did not match CPU buffer")
    elif final_mean >= CONVERGENCE_TARGET:
        print(
            "PASS: CartPole converged ≥", CONVERGENCE_TARGET,
            "(got", final_mean, ")",
        )
    else:
        print(
            "INCONCLUSIVE: did not hit", CONVERGENCE_TARGET,
            "— got", final_mean,
            "(buffer mirror OK so storage path is healthy)"
        )
