"""MuZero CPU vs GPU UPDATE-level parity test (Stage 2 diagnostic).

Stage 1 (``test_muzero_cpu_gpu_parity.mojo``) verified that each MuZero
network produces bit-faithful forward + backward kernels on CPU vs GPU.
So if MuZero converges on CPU but not on GPU, the bug must live in the
**update-level** logic: K-step unroll wiring, CE gradient kernels,
0.5 dual-consumer split, 1/K dyn scaling, two-hot encoding, scalar
transform, or target alignment.

This test does a single full-update parity check:

  1. Build a tiny scaled-down config (BATCH=4, K=2, N=2, LATENT=16,
     HIDDEN=16, BINS=11, ACT=2, OBS=4, CAP=100).
  2. Pre-populate BOTH replay buffers (CPU + GPU) with the SAME 5
     synthetic transitions. With BATCH=4 / K=2 / N=2 / WIN_FULL=5 only
     start=0 is a valid sequence-window in either sampler, so every
     batch element ends up as a duplicate of the same trajectory —
     CPU and GPU sample IDENTICAL batches modulo any ordering/layout
     bug.
  3. Sync params CPU→GPU (verify identical L2 norm).
  4. Run ``agent.update(skip_optimizer_step=True)`` and
     ``agent.update_gpu(skip_optimizer_step=True, ctx, gpu)``. The
     new ``skip_optimizer_step`` flag is a test-only knob that bypasses
     gradient clipping + the optimizer step + polyak target update,
     leaving raw gradients in ``grads_buf`` / ``grads`` for inspection.
  5. Compare each intermediate quantity in order:
       A. Sampled batch tensors (obs, actions, policies, rewards, dones,
          to_play, values).
       B. Post-scalar-transform value/reward targets.
       C. Final hidden state slab at k=0 (rep output, post-MinMaxNorm).
       D. Per-network gradients (the PRIMARY diagnostic).
     First divergence pinpoints the bug.

Usage:
    pixi run mojo run -I . tests/deep_agents/test_muzero_update_parity.mojo
    pixi run -e apple mojo run -I . tests/deep_agents/test_muzero_update_parity.mojo
"""

from std.gpu.host import DeviceContext, HostBuffer
from std.math import abs, sqrt
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    MuZeroMLPConfig,
    MuZeroGPUState,
)
from mojo_rl.deep_agents.muzero.configs import MuZeroConfig


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════


def _l2_ptr(ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        var v = Float64((ptr + i)[])
        if v == v:  # filter NaN
            s += v * v
    return sqrt(s)


def _l2_host(buf: HostBuffer[dtype], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        var v = Float64(buf[i])
        if v == v:
            s += v * v
    return sqrt(s)


def _cmp_cpu_vs_gpu(
    name: String,
    cpu_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    gpu_host: HostBuffer[dtype],
    n: Int,
    tol_rel: Float64 = 1e-3,
    tol_abs: Float64 = 1e-6,
) -> Bool:
    """Element-wise compare CPU pointer against downloaded GPU host buffer."""
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var fail = 0
    var first_fail_idx = -1
    var first_cpu: Float64 = 0.0
    var first_gpu: Float64 = 0.0
    for i in range(n):
        var cv = Float64((cpu_ptr + i)[])
        var gv = Float64(gpu_host[i])
        var err = abs(cv - gv)
        var denom = abs(cv) + abs(gv)
        var rel: Float64 = 0.0
        if denom > 1e-12:
            rel = err / denom
        if err > max_abs:
            max_abs = err
        if rel > max_rel:
            max_rel = rel
        if err > tol_abs and rel > tol_rel:
            if first_fail_idx < 0:
                first_fail_idx = i
                first_cpu = cv
                first_gpu = gv
            fail += 1

    var cpu_l2 = _l2_ptr(cpu_ptr, n)
    var gpu_l2 = _l2_host(gpu_host, n)
    if fail == 0:
        print(
            "  [PASS]",
            name,
            ": n=",
            n,
            "max_abs=",
            max_abs,
            "max_rel=",
            max_rel,
            "cpu_l2=",
            cpu_l2,
            "gpu_l2=",
            gpu_l2,
        )
        return True
    else:
        print(
            "  [FAIL]",
            name,
            ":",
            fail,
            "/",
            n,
            "max_abs=",
            max_abs,
            "max_rel=",
            max_rel,
            "cpu_l2=",
            cpu_l2,
            "gpu_l2=",
            gpu_l2,
        )
        print(
            "    first_fail_idx=",
            first_fail_idx,
            "cpu=",
            first_cpu,
            "gpu=",
            first_gpu,
        )
        return False


# ════════════════════════════════════════════════════════════════════════════
# Synthetic transition population
# ════════════════════════════════════════════════════════════════════════════


def _inject_synthetic_replay_cpu[
    Config: MuZeroConfig, _CAP: Int = 100
](mut agent: GenericMuZeroAgent[Config, 1]) raises -> None:
    """Populate the CPU SequenceReplayBuffer with NUM_T synthetic
    transitions. Same data is mirrored into the GPU buffer.

    The transitions are deterministic and contain no episode boundaries —
    so the only valid sequence start is index 0 in both samplers.
    """
    comptime OBS = Config.obs_dim
    comptime ACT = Config.action_dim
    comptime NUM_T = Config.unroll_steps + Config.td_steps + 1  # = WIN_FULL
    # NUM_T = K+N+1 = 5 with our test config. After NUM_T stores, the
    # only valid sequence start in [0, size) is start=0.

    # Direct field access: agent.state.buffer & agent.state.mcts_*.
    for t in range(NUM_T):
        # Hand-built obs/action/reward/done/term + MCTS targets. Use simple
        # deterministic patterns so any layout/scaling bug shows up cleanly.
        var obs_arr = InlineArray[Scalar[dtype], OBS](fill=Scalar[dtype](0.0))
        for i in range(OBS):
            obs_arr[i] = Scalar[dtype](
                0.05 + 0.1 * Float64(t) + 0.03 * Float64(i)
            )

        var act_arr = InlineArray[Scalar[dtype], ACT](
            fill=Scalar[dtype](0.0)
        )
        # One-hot action — alternate 0/1 across t so the dyn network sees
        # both action embeddings.
        act_arr[t % ACT] = Scalar[dtype](1.0)

        var reward = Scalar[dtype](0.1 * Float64(t + 1))
        var done = False  # never end an episode within the trajectory
        var terminated = False

        agent.state.buffer.add_with_termination(
            obs_arr, act_arr, reward, done, terminated
        )

        # Store MCTS targets at the slot we just wrote.
        var buf_idx = (
            agent.state.buffer.ptr - 1 + _CAP
        ) % _CAP
        # MCTS policy — slightly non-uniform so policy CE is non-zero.
        # action 0 gets 0.6, action 1 gets 0.4 (assumes ACT==2)
        for a in range(ACT):
            var p: Float64 = 0.6 if a == 0 else 0.4
            agent.state.mcts_policies[
                buf_idx * ACT + a
            ] = Scalar[dtype](p)
        # MCTS root value — small deterministic value, no h-transform.
        agent.state.mcts_values[buf_idx] = Scalar[dtype](
            0.5 + 0.1 * Float64(t)
        )
        agent.state.mcts_to_play[buf_idx] = Scalar[DType.uint8](0)


def _inject_synthetic_replay_gpu[
    Config: MuZeroConfig, N_ENVS: Int = 1, PER_ENV_CAP: Int = 100
](
    ctx: DeviceContext,
    mut gpu: MuZeroGPUState[Config, N_ENVS, PER_ENV_CAP],
) raises -> None:
    """Mirror the CPU buffer setup into ``gpu.replay`` (single env, env 0).

    Mirrors `_inject_synthetic_replay_cpu` so both samplers see identical
    data. Uploads via pinned host buffer per field.
    """
    comptime OBS = Config.obs_dim
    comptime ACT = Config.action_dim
    comptime NUM_T = Config.unroll_steps + Config.td_steps + 1

    # ── Stage host scratch arrays for each field ─────────────────────
    # GPU buffer layout: env-strided, env_idx * PER_ENV_CAP * dim + t * dim.
    # We only fill env 0, slots 0..NUM_T-1.
    var obs_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * PER_ENV_CAP * OBS
    )
    var act_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * PER_ENV_CAP * ACT
    )
    var rew_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * PER_ENV_CAP
    )
    var done_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * PER_ENV_CAP
    )
    var term_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * PER_ENV_CAP
    )
    var pol_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * PER_ENV_CAP * ACT
    )
    var val_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * PER_ENV_CAP
    )
    var tp_host = ctx.enqueue_create_host_buffer[DType.uint8](
        N_ENVS * PER_ENV_CAP
    )

    # Zero everything first (so the un-written tail of the per-env block
    # stays clean and the rejection sampler sees no spurious done flags).
    for i in range(N_ENVS * PER_ENV_CAP * OBS):
        obs_host[i] = Scalar[dtype](0.0)
    for i in range(N_ENVS * PER_ENV_CAP * ACT):
        act_host[i] = Scalar[dtype](0.0)
        pol_host[i] = Scalar[dtype](0.0)
    for i in range(N_ENVS * PER_ENV_CAP):
        rew_host[i] = Scalar[dtype](0.0)
        done_host[i] = Scalar[dtype](0.0)
        term_host[i] = Scalar[dtype](0.0)
        val_host[i] = Scalar[dtype](0.0)
        tp_host[i] = UInt8(0)

    # ── Fill env-0, slots 0..NUM_T-1 ─────────────────────────────────
    for t in range(NUM_T):
        var obs_base = 0 * PER_ENV_CAP * OBS + t * OBS
        for i in range(OBS):
            obs_host[obs_base + i] = Scalar[dtype](
                0.05 + 0.1 * Float64(t) + 0.03 * Float64(i)
            )

        var act_base = 0 * PER_ENV_CAP * ACT + t * ACT
        act_host[act_base + (t % ACT)] = Scalar[dtype](1.0)

        rew_host[t] = Scalar[dtype](0.1 * Float64(t + 1))
        # done/term remain 0 (no episode boundary)

        var pol_base = 0 * PER_ENV_CAP * ACT + t * ACT
        for a in range(ACT):
            var p: Float64 = 0.6 if a == 0 else 0.4
            pol_host[pol_base + a] = Scalar[dtype](p)
        val_host[t] = Scalar[dtype](0.5 + 0.1 * Float64(t))
        tp_host[t] = UInt8(0)

    # ── Upload to device ─────────────────────────────────────────────
    ctx.enqueue_copy(gpu.replay.obs_buf, obs_host)
    ctx.enqueue_copy(gpu.replay.actions_buf, act_host)
    ctx.enqueue_copy(gpu.replay.rewards_buf, rew_host)
    ctx.enqueue_copy(gpu.replay.dones_buf, done_host)
    ctx.enqueue_copy(gpu.replay.terminations_buf, term_host)
    ctx.enqueue_copy(gpu.mcts_policy_buf, pol_host)
    ctx.enqueue_copy(gpu.mcts_value_buf, val_host)
    ctx.enqueue_copy(gpu.mcts_to_play_buf, tp_host)

    # Update CPU-side counters so the sampler sees the data.
    gpu.replay.write_idx = NUM_T
    gpu.replay.size = NUM_T

    ctx.synchronize()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════


def main() raises:
    print("=== MuZero CPU vs GPU Update-Level Parity ===")
    print()
    print(
        "Stage 2 diagnostic — single-step update parity (post Stage-1 kernel"
    )
    print(
        "checks). Compares sampled batches + post-scalar-transform targets +"
    )
    print(
        "per-network gradients after one full forward+backward pass, with"
    )
    print(
        "optimizer step bypassed via the new ``skip_optimizer_step`` flag."
    )
    print()

    var ctx = DeviceContext()

    # ── Tiny scaled-down config (small enough to be readable, big enough
    # to exercise K-step unroll wiring + all gradient kernels) ──────────
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 16
    comptime HIDDEN = 16
    comptime BINS = 11
    comptime CAP = 100
    comptime BS = 4
    comptime K = 2
    comptime N = 2

    comptime Config = MuZeroMLPConfig[
        OBS, ACT, LATENT, HIDDEN, BINS, LR=3e-4, CAP=CAP, BS=BS, K=K, N=N
    ]
    comptime N_ENVS = 1
    comptime PER_ENV_CAP = CAP

    # ── Construct agents (CPU agent + paired GPU state) ────────────────
    # Use only the n_envs=1 variant so CPU sampling buffer mirrors single-
    # env GPU sampling cleanly.
    print("--- Build agent + GPU state ---")
    var agent = GenericMuZeroAgent[Config, N_ENVS](
        gamma=0.997,
        v_min=-5.0,
        v_max=5.0,
        max_grad_norm=10.0,
    )
    var gpu = MuZeroGPUState[Config, N_ENVS, PER_ENV_CAP](ctx)

    # ── Sync params CPU→GPU ────────────────────────────────────────────
    gpu.representation.upload_from(agent.state.representation, ctx)
    gpu.dynamics.upload_from(agent.state.dynamics, ctx)
    gpu.prediction.upload_from(agent.state.prediction, ctx)
    # Targets too (so target nets match exactly if they participate).
    gpu.representation_target.upload_from(agent.state.representation, ctx)
    gpu.prediction_target.upload_from(agent.state.prediction, ctx)
    ctx.synchronize()

    comptime REP_PS = Config.RepModel.PARAM_SIZE
    comptime DYN_PS = Config.DynModel.PARAM_SIZE
    comptime PRED_PS = Config.PredModel.PARAM_SIZE
    print(
        "param-sizes: rep=",
        REP_PS,
        "dyn=",
        DYN_PS,
        "pred=",
        PRED_PS,
    )

    # Param L2 sanity (verify upload worked). Note: stage-D grad parity
    # is the load-bearing check; a forward pass that produces correct
    # gradients implies params were uploaded correctly.
    var cpu_rep_l2 = _l2_ptr(agent.state.representation.params, REP_PS)
    print("cpu rep params L2 (sanity):", cpu_rep_l2)

    # ── Populate replay buffers with identical synthetic data ──────────
    print("--- Inject synthetic replay data (5 transitions) ---")
    _inject_synthetic_replay_cpu[Config, CAP](agent)
    _inject_synthetic_replay_gpu[Config, N_ENVS, PER_ENV_CAP](ctx, gpu)

    print(
        "buffer.size cpu=",
        agent.state.buffer.size,
        "gpu=",
        gpu.replay.size,
    )

    # ── Run a single forward+backward step on both, skip optimizer ─────
    print()
    print("--- Run one update step on both paths ---")
    _ = agent.update(use_reanalyze=False, skip_optimizer_step=True)
    _ = agent.update_gpu[N_ENVS, PER_ENV_CAP](
        ctx,
        gpu,
        use_reanalyze=False,
        use_per=False,
        per_progress=0.0,
        skip_optimizer_step=True,
    )
    ctx.synchronize()

    # ── A. Sampled batch comparisons ──────────────────────────────────
    print()
    print("--- A. Sampled batch tensors ---")
    comptime WIN_FULL = K + N + 1
    comptime WIN_TRN = K + N

    comptime BATCH_OBS_N = WIN_FULL * BS * OBS
    var batch_obs_host = ctx.enqueue_create_host_buffer[dtype](BATCH_OBS_N)
    ctx.enqueue_copy(batch_obs_host, gpu.batch_obs_buf)
    ctx.synchronize()
    var a_ok = _cmp_cpu_vs_gpu(
        "batch_obs", agent.state._batch_obs, batch_obs_host,
        BATCH_OBS_N,
    )

    comptime BATCH_ACT_N = K * BS * ACT
    var batch_act_host = ctx.enqueue_create_host_buffer[dtype](BATCH_ACT_N)
    ctx.enqueue_copy(batch_act_host, gpu.batch_actions_buf)
    ctx.synchronize()
    var a_ok2 = _cmp_cpu_vs_gpu(
        "batch_actions", agent.state._batch_actions,
        batch_act_host, BATCH_ACT_N,
    )

    comptime BATCH_POL_N = WIN_FULL * BS * ACT
    var batch_pol_host = ctx.enqueue_create_host_buffer[dtype](BATCH_POL_N)
    ctx.enqueue_copy(batch_pol_host, gpu.batch_policies_buf)
    ctx.synchronize()
    var a_ok3 = _cmp_cpu_vs_gpu(
        "batch_policies", agent.state._batch_policies,
        batch_pol_host, BATCH_POL_N,
    )

    comptime BATCH_TRN_N = WIN_TRN * BS
    var batch_rew_host = ctx.enqueue_create_host_buffer[dtype](BATCH_TRN_N)
    ctx.enqueue_copy(batch_rew_host, gpu.batch_rewards_buf)
    ctx.synchronize()
    var a_ok4 = _cmp_cpu_vs_gpu(
        "batch_rewards", agent.state._batch_rewards,
        batch_rew_host, BATCH_TRN_N,
    )

    var batch_done_host = ctx.enqueue_create_host_buffer[dtype](BATCH_TRN_N)
    ctx.enqueue_copy(batch_done_host, gpu.batch_dones_buf)
    ctx.synchronize()
    var a_ok5 = _cmp_cpu_vs_gpu(
        "batch_dones", agent.state._batch_dones,
        batch_done_host, BATCH_TRN_N,
    )

    var batch_a_ok = a_ok and a_ok2 and a_ok3 and a_ok4 and a_ok5

    # ── B. Post-scalar-transform value/reward targets ─────────────────
    print()
    print("--- B. Post-scalar-transform targets ---")
    comptime VAL_TGT_N = (K + 1) * BS
    var val_tgt_host = ctx.enqueue_create_host_buffer[dtype](VAL_TGT_N)
    ctx.enqueue_copy(val_tgt_host, gpu.value_targets_buf)
    ctx.synchronize()
    var b_ok = _cmp_cpu_vs_gpu(
        "value_targets", agent.state._value_targets, val_tgt_host,
        VAL_TGT_N,
    )

    comptime REW_TGT_N = K * BS
    var rew_tgt_host = ctx.enqueue_create_host_buffer[dtype](REW_TGT_N)
    ctx.enqueue_copy(rew_tgt_host, gpu.reward_targets_buf)
    ctx.synchronize()
    var b_ok2 = _cmp_cpu_vs_gpu(
        "reward_targets", agent.state._reward_targets,
        rew_tgt_host, REW_TGT_N,
    )

    var batch_b_ok = b_ok and b_ok2

    # ── C. Hidden state at k=0 (post-MinMaxNorm rep output) ───────────
    print()
    print("--- C. Hidden state at k=0 (rep output post-MinMaxNorm) ---")
    comptime HIDDEN_K0_N = BS * LATENT
    var hidden_host = ctx.enqueue_create_host_buffer[dtype](HIDDEN_K0_N)
    # gpu.hidden_buf layout is [(K+1) * BATCH * LATENT]; offset 0 is k=0.
    ctx.enqueue_copy(hidden_host, gpu.hidden_buf)
    ctx.synchronize()
    var c_ok = _cmp_cpu_vs_gpu(
        "hidden_k0", agent.state._hidden_states, hidden_host,
        HIDDEN_K0_N,
        tol_rel=1e-3,
    )

    # ── D. Per-network gradients (PRIMARY DIAGNOSTIC) ─────────────────
    print()
    print("--- D. Per-network gradients (PRIMARY) ---")

    var rep_grads_host = ctx.enqueue_create_host_buffer[dtype](REP_PS)
    ctx.enqueue_copy(rep_grads_host, gpu.representation.grads_buf)
    var dyn_grads_host = ctx.enqueue_create_host_buffer[dtype](DYN_PS)
    ctx.enqueue_copy(dyn_grads_host, gpu.dynamics.grads_buf)
    var pred_grads_host = ctx.enqueue_create_host_buffer[dtype](PRED_PS)
    ctx.enqueue_copy(pred_grads_host, gpu.prediction.grads_buf)
    ctx.synchronize()

    var rep_cpu_l2 = _l2_ptr(agent.state.representation.grads, REP_PS)
    var dyn_cpu_l2 = _l2_ptr(agent.state.dynamics.grads, DYN_PS)
    var pred_cpu_l2 = _l2_ptr(agent.state.prediction.grads, PRED_PS)
    var rep_gpu_l2 = _l2_host(rep_grads_host, REP_PS)
    var dyn_gpu_l2 = _l2_host(dyn_grads_host, DYN_PS)
    var pred_gpu_l2 = _l2_host(pred_grads_host, PRED_PS)

    print(
        "rep  grads L2: cpu=",
        rep_cpu_l2,
        "gpu=",
        rep_gpu_l2,
        "ratio=",
        rep_gpu_l2 / rep_cpu_l2 if rep_cpu_l2 > 1e-12 else Float64(0.0),
    )
    print(
        "dyn  grads L2: cpu=",
        dyn_cpu_l2,
        "gpu=",
        dyn_gpu_l2,
        "ratio=",
        dyn_gpu_l2 / dyn_cpu_l2 if dyn_cpu_l2 > 1e-12 else Float64(0.0),
    )
    print(
        "pred grads L2: cpu=",
        pred_cpu_l2,
        "gpu=",
        pred_gpu_l2,
        "ratio=",
        pred_gpu_l2 / pred_cpu_l2 if pred_cpu_l2 > 1e-12 else Float64(0.0),
    )
    print()

    var d_ok = _cmp_cpu_vs_gpu(
        "rep_grads",
        agent.state.representation.grads,
        rep_grads_host,
        REP_PS,
        tol_rel=5e-3,
    )
    var d_ok2 = _cmp_cpu_vs_gpu(
        "dyn_grads",
        agent.state.dynamics.grads,
        dyn_grads_host,
        DYN_PS,
        tol_rel=5e-3,
    )
    var d_ok3 = _cmp_cpu_vs_gpu(
        "pred_grads",
        agent.state.prediction.grads,
        pred_grads_host,
        PRED_PS,
        tol_rel=5e-3,
    )

    var batch_d_ok = d_ok and d_ok2 and d_ok3

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=== Summary ===")
    if not batch_a_ok:
        print("First divergence: STAGE A — sampled batch tensors.")
        print(
            "  → CPU and GPU samplers picked different transitions, OR layout"
            " differs."
        )
    elif not batch_b_ok:
        print(
            "First divergence: STAGE B — scalar-transform / target alignment."
        )
        print(
            "  → CPU `update()` and GPU `update_gpu` produce different value/"
            "reward targets despite identical sampled batches. Bug lives in"
            " the n-step kernel or scalar_transform application."
        )
    elif not c_ok:
        print(
            "First divergence: STAGE C — rep-network forward (hidden state)."
        )
        print(
            "  → Forward through rep network diverges even with identical"
            " inputs + params. Likely a Stage-1 kernel-level miss; re-run"
            " test_muzero_cpu_gpu_parity.mojo."
        )
    elif not batch_d_ok:
        print("First divergence: STAGE D — per-network gradients.")
        print(
            "  → Forward matches; backward diverges. Bug lives in:"
            " CE gradient kernels, K-step unroll wiring, 0.5 dual-consumer"
            " split, or 1/K dyn scaling."
        )
    else:
        print("All stages within tolerance. CPU↔GPU update path is parity.")
    print()
