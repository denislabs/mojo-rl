"""EZ-V2 CartPole — Step 5: GPU priority sample + window gather verification.

Runs the same N_ENVS=4 training as Step 4 for a short warmup window
(enough to fill the buffer past K-step horizons), then exercises the
new GPU sampling kernels (`gpu_sampling.mojo`) end-to-end and verifies
their output bit-for-bit against a host-computed expectation.

Verification protocol:

  1. Train normally with `agent.train_step_gpu` until buffer is sized
     (≥ K+10 transitions). CPU buffer is the ground truth.
  2. Bulk-upload CPU buffer to GPU replay.
  3. Run kernels 1+2 (cum-prio scan, sample starts) → download
     `batch_start_idx`.
  4. Run kernels 3+4 (gather, cum-rewards) → fills device batch buffers.
  5. Independently on host, given the same `batch_start_idx`, fill
     expected batch arrays from the CPU replay buffer.
  6. Download device batch buffers, compare per-field max-abs diffs to
     host expectation. Should all be 0.
  7. Sanity-check that every picked start respects the "no done in
     first K positions" constraint.

Plan:    docs/EZV2_FULL_GPU_PLAN.md
Gate:    all per-field diffs == 0 (gather is bit-exact for a given
         batch_start_idx). Sample picks are all valid windows.

Run:
    pixi run mojo run -I . examples/cartpole/cartpole_ezv2_full_gpu_step5.mojo
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
from mojo_rl.deep_agents.efficient_zero_v2.gpu_sampling import (
    ezv2_gpu_sample_and_gather,
    ezv2_priority_writeback_kernel,
)
from mojo_rl.deep_agents.efficient_zero_v2.strategies import compute_sve
from mojo_rl.envs.cartpole import CartPoleEnv
from mojo_rl.nn.constants import dtype, TPB


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
    print("=== EZ-V2 CartPole demo — Step 5 (GPU sampling kernels) ===")

    # Short warmup — just enough to populate buffer with valid windows.
    comptime WARMUP_ENV_STEPS = 4_000
    comptime TRAIN_INTERVAL = 4
    comptime LOG_EVERY = 1_000

    comptime SYNC_INTERVAL = 50

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
    comptime BATCH = Config.batch_size
    comptime K = Config.unroll_steps
    comptime N_ENVS = 4
    comptime CAP = 50000

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
    print("    WARMUP_ENV_STEPS      =", WARMUP_ENV_STEPS)
    print("    N_ENVS                =", N_ENVS)
    print("    BATCH                 =", BATCH)
    print("    K_UNROLL              =", K)
    print("    OBS / ACT             =", OBS, "/", ACT)
    print("    CAP                   =", CAP)
    print()

    # ── GPU state + env buffers + MCTS state + replay buffer ─────────────
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

    var gpu_replay = EZV2GPUReplayBuffer[CAP, OBS, ACT](ctx)
    ctx.synchronize()
    print("    GPU state + replay buffer ready")

    CartPoleEnv[DType.float32].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, states_buf, rng_seed=UInt64(2026)
    )

    comptime extract_obs = _extract_obs_kernel[N_ENVS, STATE_SIZE, OBS_DIM]
    comptime tpb_ext = 32
    comptime blocks_ext = (N_ENVS + tpb_ext - 1) // tpb_ext

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
            st, ob, grid_dim=(blocks_ext,), block_dim=(tpb_ext,)
        )

    _launch_extract_obs()
    ctx.synchronize()
    print()

    # ── Warmup training loop (CPU sampling, GPU train) ───────────────────
    print("--- Warmup ", WARMUP_ENV_STEPS, " env-steps ---")
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
    var any_nan_loss = False
    var step_seed: UInt64 = 1
    var mcts_seed: UInt32 = 0
    var total_env_steps = 0

    var t0 = perf_counter_ns()

    while total_env_steps < WARMUP_ENV_STEPS:
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
            ctx, states_buf, actions_buf, rewards_buf, dones_buf,
            terminated_buf, obs_buf, rng_seed=step_seed,
        )

        ctx.enqueue_copy(host_reward.unsafe_ptr(), rewards_buf)
        ctx.enqueue_copy(host_done.unsafe_ptr(), dones_buf)
        ctx.synchronize()

        var any_done = False
        for e in range(N_ENVS):
            var reward = Float64(host_reward[e])
            var done = host_done[e] > Scalar[dtype](0.5)
            agent.store_transition(
                obs_per_env[e], actions_per_env[e], reward,
                policies_per_env[e], root_value_per_env[e], done,
                env_id=e,
            )
            ep_return_per_env[e] += reward
            total_env_steps += 1
            if done:
                ep_returns.append(ep_return_per_env[e])
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

        if (
            agent.state.is_ready()
            and (total_env_steps // N_ENVS) % TRAIN_INTERVAL == 0
        ):
            var t = agent.train_step_gpu(gpu, ctx)
            num_train_calls += 1
            if not _is_finite(t[0]):
                any_nan_loss = True
            if num_train_calls % SYNC_INTERVAL == 0:
                gpu.download_to(agent.state, ctx)
                ctx.synchronize()

        step_seed += 1

        if total_env_steps % LOG_EVERY == 0:
            var t_now = perf_counter_ns()
            print(
                "    [step ", total_env_steps,
                " ep=", len(ep_returns),
                " train=", num_train_calls,
                " wall=", Float64(t_now - t0) / 1.0e9, "s]",
            )

    print()
    print("--- Warmup complete ---")
    print("    buffer.size           =", agent.state.buffer.size)
    print("    buffer.ptr            =", agent.state.buffer.ptr)
    print("    train_step calls      =", num_train_calls)
    print("    any NaN loss          =", any_nan_loss)
    print()

    # ── Bulk-upload CPU buffer to GPU replay ─────────────────────────────
    print("--- Uploading CPU buffer to GPU replay ---")
    gpu_replay.upload_from_cpu(agent.state, ctx)
    gpu_replay.max_priority = agent.max_priority
    ctx.synchronize()
    print("    GPU replay.size       =", gpu_replay.size)
    print("    GPU replay.ptr        =", gpu_replay.ptr)
    print()

    # ── Run GPU sample + gather kernels ──────────────────────────────────
    print("--- Running GPU sample + gather kernels ---")

    # Scratch buffers for kernels 1+2.
    var cum_prio_buf = ctx.enqueue_create_buffer[dtype](CAP)
    var cand_starts_buf = ctx.enqueue_create_buffer[DType.int32](CAP)
    var n_valid_buf = ctx.enqueue_create_buffer[DType.int32](1)
    var total_prio_buf = ctx.enqueue_create_buffer[dtype](1)
    var batch_start_idx_buf = ctx.enqueue_create_buffer[DType.int32](
        BATCH
    )

    var oldest = (
        agent.state.buffer.ptr - agent.state.buffer.size + CAP
    ) % CAP
    var current_train_step = agent.train_step_count

    ezv2_gpu_sample_and_gather[CAP, BATCH, K, OBS, ACT](
        ctx,
        gpu_replay.priorities,
        gpu_replay.dones,
        gpu_replay.obs,
        gpu_replay.actions,
        gpu_replay.rewards,
        gpu_replay.mcts_policies,
        gpu_replay.mcts_values,
        gpu_replay.step_at_write,
        cum_prio_buf,
        cand_starts_buf,
        n_valid_buf,
        total_prio_buf,
        batch_start_idx_buf,
        gpu.batch_obs_buf,
        gpu.batch_actions_buf,
        gpu.batch_rewards_buf,
        gpu.batch_mcts_pol_buf,
        gpu.batch_mcts_val_buf,
        gpu.batch_age_buf,
        gpu.cum_rewards_buf,
        oldest=oldest,
        buf_size=agent.state.buffer.size,
        current_train_step=UInt32(current_train_step),
        rng_seed=UInt32(7),
    )
    ctx.synchronize()

    # ── Download GPU outputs ─────────────────────────────────────────────
    var host_batch_start = ctx.enqueue_create_host_buffer[DType.int32](
        BATCH
    )
    var host_n_valid = ctx.enqueue_create_host_buffer[DType.int32](1)
    var host_total_prio = ctx.enqueue_create_host_buffer[dtype](1)

    var host_dst_obs = ctx.enqueue_create_host_buffer[dtype](
        BATCH * (K + 1) * OBS
    )
    var host_dst_actions = ctx.enqueue_create_host_buffer[dtype](
        BATCH * K * ACT
    )
    var host_dst_rewards = ctx.enqueue_create_host_buffer[dtype](BATCH * K)
    var host_dst_pol = ctx.enqueue_create_host_buffer[dtype](
        BATCH * (K + 1) * ACT
    )
    var host_dst_val = ctx.enqueue_create_host_buffer[dtype](
        BATCH * (K + 1)
    )
    var host_dst_age = ctx.enqueue_create_host_buffer[DType.int32](
        BATCH * (K + 1)
    )
    var host_dst_cum = ctx.enqueue_create_host_buffer[dtype](BATCH * K)

    ctx.enqueue_copy(host_batch_start.unsafe_ptr(), batch_start_idx_buf)
    ctx.enqueue_copy(host_n_valid.unsafe_ptr(), n_valid_buf)
    ctx.enqueue_copy(host_total_prio.unsafe_ptr(), total_prio_buf)
    ctx.enqueue_copy(host_dst_obs.unsafe_ptr(), gpu.batch_obs_buf)
    ctx.enqueue_copy(host_dst_actions.unsafe_ptr(), gpu.batch_actions_buf)
    ctx.enqueue_copy(host_dst_rewards.unsafe_ptr(), gpu.batch_rewards_buf)
    ctx.enqueue_copy(host_dst_pol.unsafe_ptr(), gpu.batch_mcts_pol_buf)
    ctx.enqueue_copy(host_dst_val.unsafe_ptr(), gpu.batch_mcts_val_buf)
    ctx.enqueue_copy(host_dst_age.unsafe_ptr(), gpu.batch_age_buf)
    ctx.enqueue_copy(host_dst_cum.unsafe_ptr(), gpu.cum_rewards_buf)
    ctx.synchronize()

    print("    GPU n_valid           =", Int(host_n_valid[0]))
    print("    GPU total_prio        =", Float64(host_total_prio[0]))
    print()

    # ── Validate sample picks (no done in first K positions) ─────────────
    print("--- Validating sample picks ---")
    var n_valid_picks = 0
    for b in range(BATCH):
        var start = Int(host_batch_start[b])
        var ok = True
        for k in range(K):
            var idx = (start + k) % CAP
            if Float64(agent.state.buffer.dones[idx]) > 0.5:
                ok = False
                break
        if ok:
            n_valid_picks += 1
    print(
        "    valid picks            =", n_valid_picks, "/", BATCH,
    )
    print()

    # ── Compute host expectation given GPU's batch_start_idx ─────────────
    print("--- Computing host expectation, diffing against GPU ---")
    var max_diff_obs = Float64(0.0)
    var max_diff_act = Float64(0.0)
    var max_diff_rew = Float64(0.0)
    var max_diff_pol = Float64(0.0)
    var max_diff_val = Float64(0.0)
    var max_diff_age = Int(0)
    var max_diff_cum = Float64(0.0)

    for b in range(BATCH):
        var start = Int(host_batch_start[b])
        for k in range(K + 1):
            var idx = (start + k) % CAP
            for d in range(OBS):
                var exp_v = Float64(
                    agent.state.buffer.obs[idx * OBS + d]
                )
                var got_v = Float64(
                    host_dst_obs[(b * (K + 1) + k) * OBS + d]
                )
                var d_v = abs(exp_v - got_v)
                if d_v > max_diff_obs:
                    max_diff_obs = d_v
            for a in range(ACT):
                var exp_p = Float64(
                    (agent.state.mcts_policies + idx * ACT + a)[]
                )
                var got_p = Float64(
                    host_dst_pol[(b * (K + 1) + k) * ACT + a]
                )
                var d_p = abs(exp_p - got_p)
                if d_p > max_diff_pol:
                    max_diff_pol = d_p
            var exp_val = Float64(
                (agent.state.mcts_values + idx)[]
            )
            var got_val = Float64(host_dst_val[b * (K + 1) + k])
            var d_val = abs(exp_val - got_val)
            if d_val > max_diff_val:
                max_diff_val = d_val
            var sw = Int(
                (agent.state.step_at_write + idx)[]
            )
            var exp_age = current_train_step - sw
            if exp_age < 0:
                exp_age = 0
            var got_age = Int(host_dst_age[b * (K + 1) + k])
            var d_age = abs(exp_age - got_age)
            if d_age > max_diff_age:
                max_diff_age = d_age

        var cum = Float64(0.0)
        for k in range(K):
            var idx = (start + k) % CAP
            for a in range(ACT):
                var exp_a = Float64(
                    agent.state.buffer.actions[idx * ACT + a]
                )
                var got_a = Float64(
                    host_dst_actions[(b * K + k) * ACT + a]
                )
                var d_a = abs(exp_a - got_a)
                if d_a > max_diff_act:
                    max_diff_act = d_a
            var exp_r = Float64(agent.state.buffer.rewards[idx])
            var got_r = Float64(host_dst_rewards[b * K + k])
            var d_r = abs(exp_r - got_r)
            if d_r > max_diff_rew:
                max_diff_rew = d_r
            cum += exp_r
            var got_cum = Float64(host_dst_cum[b * K + k])
            var d_c = abs(cum - got_cum)
            if d_c > max_diff_cum:
                max_diff_cum = d_c

    print()
    print("=== Per-field max-abs diffs (GPU vs host expectation) ===")
    print("    obs                   =", max_diff_obs)
    print("    actions               =", max_diff_act)
    print("    rewards               =", max_diff_rew)
    print("    mcts_policies         =", max_diff_pol)
    print("    mcts_values           =", max_diff_val)
    print("    age (int)             =", max_diff_age)
    print("    cum_rewards           =", max_diff_cum)

    var gather_ok = (
        max_diff_obs == 0.0
        and max_diff_act == 0.0
        and max_diff_rew == 0.0
        and max_diff_pol == 0.0
        and max_diff_val == 0.0
        and max_diff_age == 0
        and max_diff_cum == 0.0
    )
    var sample_ok = n_valid_picks == BATCH

    # ── Priority writeback test ──────────────────────────────────────────
    # Stage a synthetic priorities_out array on host, upload to a device
    # buffer, run the writeback kernel, download priorities back, and
    # check that batch_start_idx slots were updated to the synthetic
    # values.
    print()
    print("--- Priority writeback test ---")
    var test_pri_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    var test_pri_dev = ctx.enqueue_create_buffer[dtype](BATCH)
    for b in range(BATCH):
        test_pri_host[b] = Scalar[dtype](100.0 + Float64(b))
    ctx.enqueue_copy(test_pri_dev, test_pri_host.unsafe_ptr())

    comptime writeback = ezv2_priority_writeback_kernel[BATCH, CAP]
    var bsi_t = LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ](batch_start_idx_buf.unsafe_ptr())
    var po_t = LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ](test_pri_dev.unsafe_ptr())
    var pri_t = LayoutTensor[
        dtype, Layout.row_major(CAP), MutAnyOrigin
    ](gpu_replay.priorities.unsafe_ptr())
    comptime wb_blocks = (BATCH + TPB - 1) // TPB
    ctx.enqueue_function[writeback, writeback](
        bsi_t, po_t, pri_t,
        grid_dim=(wb_blocks,), block_dim=(TPB,),
    )

    var post_pri_host = ctx.enqueue_create_host_buffer[dtype](CAP)
    ctx.enqueue_copy(post_pri_host.unsafe_ptr(), gpu_replay.priorities)
    ctx.synchronize()

    var writeback_ok = True
    for b in range(BATCH):
        var idx = Int(host_batch_start[b])
        var got = Float64(post_pri_host[idx])
        var expect = 100.0 + Float64(b)
        # Multiple b's may share the same idx — last write wins on GPU
        # (matches CPU loop). Find the latest b that maps to this idx.
        var latest_b = b
        for bb in range(b + 1, BATCH):
            if Int(host_batch_start[bb]) == idx:
                latest_b = bb
        expect = 100.0 + Float64(latest_b)
        if abs(got - expect) > 1e-6:
            writeback_ok = False
            print(
                "      mismatch at b=", b, "idx=", idx,
                "got=", got, "expect=", expect,
            )

    print("    writeback OK          =", writeback_ok)

    print()
    print("=== Step 5 verification summary ===")
    print("    sample picks valid    =", sample_ok)
    print("    gather bit-exact      =", gather_ok)
    print("    priority writeback OK =", writeback_ok)
    print()

    if sample_ok and gather_ok and writeback_ok:
        print("PASS: GPU sampling + gather + writeback kernels all verified")
    else:
        print("FAIL: one or more kernels did not match host expectation")
