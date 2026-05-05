"""TD-MPC2 — K-step BPTT gradient correctness on GPU (Test 2 of 5, GPU port).

This is the most diagnostically valuable GPU test: it directly checks
whether GPU vjp kernels accumulate gradients correctly across multiple
backward calls (the bug class swept in
project_autodiff_multicall_accumulation.md).

Same setup as the CPU version but goes through forward_gpu_with_cache
+ backward_gpu. FD perturbations happen by downloading params, perturbing
on host, re-uploading, and recomputing the K-step forward loss.

Sub-tests:
  2a — FD vs analytic for K=3 on a sample of dynamics params, GPU.
  2b — Grad-norm at K=3 differs from K=1 (accumulation, not overwrite).
  2c — FD vs analytic for K=1 (sanity).
"""

from std.math import sqrt
from std.random import seed, random_float64
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Normal
from mojo_rl.deep_agents.tdmpc2.world_model import WorldModel


comptime OBS = 4
comptime ACT = 2
comptime LATENT = 16
comptime MLP = 32
comptime ENC = 16
comptime SIMPLEX = 4
comptime BATCH = 4
comptime ZA = LATENT + ACT
comptime KMAX = 3

comptime ENC_LR = 9e-5
comptime DYN_LR = 3e-4

comptime WM = WorldModel[
    OBS_DIM=OBS,
    ACTION_DIM=ACT,
    LATENT_DIM=LATENT,
    MLP_DIM=MLP,
    ENC_DIM=ENC,
    NUM_BINS=11,
    NUM_Q=2,
    SIMPLEX_DIM=SIMPLEX,
    ENC_LR=ENC_LR,
    WM_LR=DYN_LR,
]
comptime EncModel = WM.EncModel
comptime DynModel = WM.DynModel
comptime EncOpt = Adam[LR=ENC_LR]
comptime DynOpt = Adam[LR=DYN_LR]
comptime ENC_WS_SIZE = (
    BATCH * EncModel.WORKSPACE_SIZE_PER_SAMPLE
    if EncModel.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
)
comptime DYN_WS_SIZE = (
    BATCH * DynModel.WORKSPACE_SIZE_PER_SAMPLE
    if DynModel.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
)


def _expect(cond: Bool, label: String, mut passed: Int, mut total: Int):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _l2_norm_host(h: HostBuffer[dtype], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v * v
    return sqrt(s)


# Heap-allocated dataset (so it persists across function calls).
struct GPUDataset(Movable):
    var obs0_dev: DeviceBuffer[dtype]  # [B, OBS]
    var acts_dev: DeviceBuffer[dtype]  # [KMAX, B, ACT]
    var obsK_dev: DeviceBuffer[dtype]  # [KMAX, B, OBS]
    var acts_host: HostBuffer[dtype]  # mirror for ZA building

    def __init__(
        out self,
        obs0_dev: DeviceBuffer[dtype],
        acts_dev: DeviceBuffer[dtype],
        obsK_dev: DeviceBuffer[dtype],
        acts_host: HostBuffer[dtype],
    ):
        self.obs0_dev = obs0_dev
        self.acts_dev = acts_dev
        self.obsK_dev = obsK_dev
        self.acts_host = acts_host


def _build_dataset(ctx: DeviceContext) raises -> GPUDataset:
    var obs0_h = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
    var acts_h = ctx.enqueue_create_host_buffer[dtype](
        KMAX * BATCH * ACT
    )
    var obsK_h = ctx.enqueue_create_host_buffer[dtype](
        KMAX * BATCH * OBS
    )
    for i in range(BATCH * OBS):
        obs0_h[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
    for i in range(KMAX * BATCH * ACT):
        acts_h[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
    for i in range(KMAX * BATCH * OBS):
        obsK_h[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
    var obs0_d = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
    var acts_d = ctx.enqueue_create_buffer[dtype](KMAX * BATCH * ACT)
    var obsK_d = ctx.enqueue_create_buffer[dtype](KMAX * BATCH * OBS)
    ctx.enqueue_copy(obs0_d, obs0_h)
    ctx.enqueue_copy(acts_d, acts_h)
    ctx.enqueue_copy(obsK_d, obsK_h)
    return GPUDataset(obs0_d, acts_d, obsK_d, acts_h^)


# Forward-only K-step loss on GPU. Used for FD; encoder is FROZEN, no
# gradient accumulation. Returns total loss (averaged over K).
def _kstep_forward_loss(
    K: Int,
    ctx: DeviceContext,
    ds: GPUDataset,
    mut enc: GPUNetworkState[EncModel, EncOpt, dtype],
    mut dyn: GPUNetworkState[DynModel, DynOpt, dtype],
    enc_ws: DeviceBuffer[dtype],
    dyn_ws: DeviceBuffer[dtype],
) raises -> Float64:
    var obs0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](ds.obs0_dev.unsafe_ptr())
    var z_carry_d = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
    var z_carry_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z_carry_d.unsafe_ptr())
    Network[EncModel, EncOpt].forward_gpu[BATCH](
        ctx, obs0_t, z_carry_t,
        enc.params_view(), enc.model_state_view(), enc_ws,
    )

    var loss: Float64 = 0.0
    var z_carry_h = ctx.enqueue_create_host_buffer[dtype](
        BATCH * LATENT
    )
    var za_h = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
    var z_pred_h = ctx.enqueue_create_host_buffer[dtype](
        BATCH * LATENT
    )
    var z_target_h = ctx.enqueue_create_host_buffer[dtype](
        BATCH * LATENT
    )

    for t in range(K):
        # Encoder forward target.
        var obs_t1 = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](ds.obsK_dev.unsafe_ptr() + t * BATCH * OBS)
        var z_target_d = ctx.enqueue_create_buffer[dtype](
            BATCH * LATENT
        )
        var z_target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_target_d.unsafe_ptr())
        Network[EncModel, EncOpt].forward_gpu[BATCH](
            ctx, obs_t1, z_target_t,
            enc.params_view(), enc.model_state_view(), enc_ws,
        )

        # Build za on host: read z_carry, splice in action, upload.
        ctx.enqueue_copy(z_carry_h, z_carry_d)
        ctx.synchronize()
        for b in range(BATCH):
            for k in range(LATENT):
                za_h[b * ZA + k] = z_carry_h[b * LATENT + k]
            for k in range(ACT):
                za_h[b * ZA + LATENT + k] = (
                    ds.acts_host[t * BATCH * ACT + b * ACT + k]
                )
        var za_d = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        ctx.enqueue_copy(za_d, za_h)
        var za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](za_d.unsafe_ptr())

        # Dynamics forward (no cache — forward only).
        var z_pred_d = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
        var z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_pred_d.unsafe_ptr())
        Network[DynModel, DynOpt].forward_gpu[BATCH](
            ctx, za_t, z_pred_t,
            dyn.params_view(), dyn.model_state_view(), dyn_ws,
        )

        # MSE on host.
        ctx.enqueue_copy(z_pred_h, z_pred_d)
        ctx.enqueue_copy(z_target_h, z_target_d)
        ctx.synchronize()
        var lt: Float64 = 0.0
        for i in range(BATCH * LATENT):
            var d = (
                Float64(z_pred_h[i]) - Float64(z_target_h[i])
            )
            lt += d * d
        lt /= Float64(BATCH * LATENT)
        loss += lt

        # Carry: z_carry = z_pred. Copy device-to-device.
        ctx.enqueue_copy(z_carry_d, z_pred_d)

    return loss / Float64(K)


# K-step BPTT on GPU. ZEROS dyn.grads at start. After return,
# dyn.grads holds the analytic gradient w.r.t. dyn.params for the
# K-step loss = (1/K) * sum_t MSE_t. Returns final loss.
def _kstep_backward(
    K: Int,
    ctx: DeviceContext,
    ds: GPUDataset,
    mut enc: GPUNetworkState[EncModel, EncOpt, dtype],
    mut dyn: GPUNetworkState[DynModel, DynOpt, dtype],
    enc_ws: DeviceBuffer[dtype],
    dyn_ws: DeviceBuffer[dtype],
) raises -> Float64:
    dyn.zero_grads(ctx)

    # Persistent caches across all K steps (dyn cache per step,
    # z_chain[K+1] for carry, z_pred per step, z_target per step).
    var z_chain_d = ctx.enqueue_create_buffer[dtype](
        (K + 1) * BATCH * LATENT
    )
    var z_pred_chain_d = ctx.enqueue_create_buffer[dtype](
        K * BATCH * LATENT
    )
    var z_target_chain_d = ctx.enqueue_create_buffer[dtype](
        K * BATCH * LATENT
    )
    var dyn_caches_d = ctx.enqueue_create_buffer[dtype](
        K * BATCH * DynModel.CACHE_SIZE
    )

    # Encoder forward s_0 (no cache — encoder is FROZEN here, no enc
    # backward).
    var obs0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](ds.obs0_dev.unsafe_ptr())
    var z0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z_chain_d.unsafe_ptr())
    Network[EncModel, EncOpt].forward_gpu[BATCH](
        ctx, obs0_t, z0_t,
        enc.params_view(), enc.model_state_view(), enc_ws,
    )

    # Per-step host scratch for MSE / ZA building.
    var z_carry_h = ctx.enqueue_create_host_buffer[dtype](
        BATCH * LATENT
    )
    var za_h = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
    var z_pred_h = ctx.enqueue_create_host_buffer[dtype](
        BATCH * LATENT
    )
    var z_target_h = ctx.enqueue_create_host_buffer[dtype](
        BATCH * LATENT
    )

    var loss: Float64 = 0.0
    for t in range(K):
        # Target encoder forward.
        var obs_t1 = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](ds.obsK_dev.unsafe_ptr() + t * BATCH * OBS)
        var z_target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_target_chain_d.unsafe_ptr() + t * BATCH * LATENT)
        Network[EncModel, EncOpt].forward_gpu[BATCH](
            ctx, obs_t1, z_target_t,
            enc.params_view(), enc.model_state_view(), enc_ws,
        )

        # Build za on host (z_chain[t] + a_t). Download whole z_chain
        # then slice (simpler than DeviceBuffer slicing).
        var z_chain_h = ctx.enqueue_create_host_buffer[dtype](
            (K + 1) * BATCH * LATENT
        )
        ctx.enqueue_copy(z_chain_h, z_chain_d)
        ctx.synchronize()
        for b in range(BATCH):
            for k in range(LATENT):
                za_h[b * ZA + k] = z_chain_h[
                    t * BATCH * LATENT + b * LATENT + k
                ]
            for k in range(ACT):
                za_h[b * ZA + LATENT + k] = (
                    ds.acts_host[t * BATCH * ACT + b * ACT + k]
                )
        var za_d = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        ctx.enqueue_copy(za_d, za_h)
        var za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](za_d.unsafe_ptr())

        # Dynamics forward with cache, into per-step slices.
        var z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_pred_chain_d.unsafe_ptr() + t * BATCH * LATENT)
        var dyn_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ](
            dyn_caches_d.unsafe_ptr()
            + t * BATCH * DynModel.CACHE_SIZE
        )
        Network[DynModel, DynOpt].forward_gpu_with_cache[BATCH](
            ctx, za_t, z_pred_t,
            dyn.params_view(), dyn.model_state_view(),
            dyn_cache_t, dyn_ws,
        )

        # MSE on host: download full z_pred and z_target chains.
        var z_pred_chain_h = ctx.enqueue_create_host_buffer[dtype](
            K * BATCH * LATENT
        )
        var z_target_chain_h = ctx.enqueue_create_host_buffer[dtype](
            K * BATCH * LATENT
        )
        ctx.enqueue_copy(z_pred_chain_h, z_pred_chain_d)
        ctx.enqueue_copy(z_target_chain_h, z_target_chain_d)
        ctx.synchronize()

        var lt: Float64 = 0.0
        for i in range(BATCH * LATENT):
            var d = (
                Float64(z_pred_chain_h[t * BATCH * LATENT + i])
                - Float64(z_target_chain_h[t * BATCH * LATENT + i])
            )
            lt += d * d
        lt /= Float64(BATCH * LATENT)
        loss += lt

        # Carry: z_chain[t+1] = z_pred[t]. Copy slice to slice via host.
        for b in range(BATCH):
            for k in range(LATENT):
                z_chain_h[
                    (t + 1) * BATCH * LATENT + b * LATENT + k
                ] = z_pred_chain_h[t * BATCH * LATENT + b * LATENT + k]
        ctx.enqueue_copy(z_chain_d, z_chain_h)

    loss /= Float64(K)

    # Backward in reverse, ACCUMULATE into dyn.grads.
    var grad_z_carry_h = ctx.enqueue_create_host_buffer[dtype](
        BATCH * LATENT
    )
    for i in range(BATCH * LATENT):
        grad_z_carry_h[i] = Scalar[dtype](0.0)
    var grad_z_carry_d = ctx.enqueue_create_buffer[dtype](
        BATCH * LATENT
    )
    ctx.enqueue_copy(grad_z_carry_d, grad_z_carry_h)

    var loss_scale = 1.0 / Float64(K)
    var mse_scale = (
        loss_scale * 2.0 / Float64(BATCH * LATENT)
    )

    # Need full host views again for grad_z_pred building each iter.
    var z_pred_chain_h = ctx.enqueue_create_host_buffer[dtype](
        K * BATCH * LATENT
    )
    var z_target_chain_h = ctx.enqueue_create_host_buffer[dtype](
        K * BATCH * LATENT
    )
    ctx.enqueue_copy(z_pred_chain_h, z_pred_chain_d)
    ctx.enqueue_copy(z_target_chain_h, z_target_chain_d)
    ctx.synchronize()

    for t_rev in range(K):
        var t = K - 1 - t_rev

        # grad_z_pred = mse_scale * (z_pred - z_target) + grad_z_carry
        ctx.enqueue_copy(grad_z_carry_h, grad_z_carry_d)
        ctx.synchronize()

        var grad_z_pred_h = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        for b in range(BATCH):
            for k in range(LATENT):
                var diff = (
                    Float64(z_pred_chain_h[
                        t * BATCH * LATENT + b * LATENT + k
                    ])
                    - Float64(z_target_chain_h[
                        t * BATCH * LATENT + b * LATENT + k
                    ])
                )
                grad_z_pred_h[b * LATENT + k] = (
                    Scalar[dtype](mse_scale * diff)
                    + grad_z_carry_h[b * LATENT + k]
                )
        var grad_z_pred_d = ctx.enqueue_create_buffer[dtype](
            BATCH * LATENT
        )
        ctx.enqueue_copy(grad_z_pred_d, grad_z_pred_h)
        var grad_z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_pred_d.unsafe_ptr())

        # Dynamics backward — should ACCUMULATE into dyn.grads.
        var grad_za_d = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za_d.unsafe_ptr())
        var dyn_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ](
            dyn_caches_d.unsafe_ptr()
            + t * BATCH * DynModel.CACHE_SIZE
        )
        var dyn_grads_v = dyn.grads_view()
        Network[DynModel, DynOpt].backward_gpu[BATCH](
            ctx, grad_z_pred_t, grad_za_t,
            dyn.params_view(), dyn.model_state_view(),
            dyn_cache_t, dyn_grads_v, dyn_ws,
        )

        # Extract z portion → grad_z_carry.
        var grad_za_h = ctx.enqueue_create_host_buffer[dtype](
            BATCH * ZA
        )
        ctx.enqueue_copy(grad_za_h, grad_za_d)
        ctx.synchronize()
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_carry_h[b * LATENT + k] = grad_za_h[b * ZA + k]
        ctx.enqueue_copy(grad_z_carry_d, grad_z_carry_h)

    return loss


def main() raises:
    seed(0xBEEF42)
    print("=" * 70)
    print("TD-MPC2 Test 2 GPU — K-step BPTT gradient correctness")
    print("=" * 70)

    var passed = 0
    var total = 0

    with DeviceContext() as ctx:
        # Build CPU init + upload to GPU.
        var enc_cpu = NetworkState[EncModel, EncOpt]()
        enc_cpu.initialize[Normal[0.0, 0.02]]()
        var dyn_cpu = NetworkState[DynModel, DynOpt]()
        dyn_cpu.initialize[Normal[0.0, 0.02]]()
        var enc = GPUNetworkState[EncModel, EncOpt, dtype](ctx)
        enc.upload_from(enc_cpu, ctx)
        var dyn = GPUNetworkState[DynModel, DynOpt, dtype](ctx)
        dyn.upload_from(dyn_cpu, ctx)

        var enc_ws = ctx.enqueue_create_buffer[dtype](ENC_WS_SIZE)
        var dyn_ws = ctx.enqueue_create_buffer[dtype](DYN_WS_SIZE)

        var ds = _build_dataset(ctx)

        # ─── 2a — FD vs analytic for K=3 ────────────────────────────
        print()
        print("--- 2a. FD gradcheck on dynamics, K=3 (GPU) ---")
        var loss_k3 = _kstep_backward(3, ctx, ds, enc, dyn, enc_ws, dyn_ws)
        print("    forward+bwd loss (K=3):", loss_k3)

        # Snapshot analytic dyn grads.
        var dyn_grads_h = ctx.enqueue_create_host_buffer[dtype](
            DynModel.PARAM_SIZE
        )
        ctx.enqueue_copy(dyn_grads_h, dyn.grads_buf)
        ctx.synchronize()

        # Backup params for restoration.
        var dyn_params_h = ctx.enqueue_create_host_buffer[dtype](
            DynModel.PARAM_SIZE
        )
        ctx.enqueue_copy(dyn_params_h, dyn.params_buf)
        ctx.synchronize()

        comptime N_SAMPLES = 30
        var eps = 1e-3
        var max_rel: Float64 = 0.0
        var max_abs: Float64 = 0.0
        for s in range(N_SAMPLES):
            var idx = (s * 7919 + 1009) % DynModel.PARAM_SIZE
            var orig = Float64(dyn_params_h[idx])

            # +eps
            dyn_params_h[idx] = Scalar[dtype](orig + eps)
            ctx.enqueue_copy(dyn.params_buf, dyn_params_h)
            ctx.synchronize()
            var Lp = _kstep_forward_loss(
                3, ctx, ds, enc, dyn, enc_ws, dyn_ws
            )

            # -eps
            dyn_params_h[idx] = Scalar[dtype](orig - eps)
            ctx.enqueue_copy(dyn.params_buf, dyn_params_h)
            ctx.synchronize()
            var Lm = _kstep_forward_loss(
                3, ctx, ds, enc, dyn, enc_ws, dyn_ws
            )

            # Restore.
            dyn_params_h[idx] = Scalar[dtype](orig)
            ctx.enqueue_copy(dyn.params_buf, dyn_params_h)
            ctx.synchronize()

            var fd = (Lp - Lm) / (2.0 * eps)
            var ana = Float64(dyn_grads_h[idx])
            var d = _abs(fd - ana)
            var denom = _abs(fd) + _abs(ana) + 1e-9
            var rel = d / denom
            if d > max_abs:
                max_abs = d
            if rel > max_rel:
                max_rel = rel

        print(
            "    sampled", N_SAMPLES, "params  max |Δ| =", max_abs,
            "  max rel err =", max_rel,
        )
        _expect(
            max_rel < 0.05,
            "2a — analytic K=3 GPU dyn grad matches FD (5% rel)",
            passed,
            total,
        )

        # ─── 2b — Grad-norm scales / differs across K ───────────────
        print()
        print("--- 2b. GPU grad-norm differs between K=1 and K=3 ---")
        _ = _kstep_backward(1, ctx, ds, enc, dyn, enc_ws, dyn_ws)
        ctx.enqueue_copy(dyn_grads_h, dyn.grads_buf)
        ctx.synchronize()
        var gn_k1 = _l2_norm_host(dyn_grads_h, DynModel.PARAM_SIZE)

        _ = _kstep_backward(3, ctx, ds, enc, dyn, enc_ws, dyn_ws)
        ctx.enqueue_copy(dyn_grads_h, dyn.grads_buf)
        ctx.synchronize()
        var gn_k3 = _l2_norm_host(dyn_grads_h, DynModel.PARAM_SIZE)
        print("    |grad K=1| =", gn_k1, "  |grad K=3| =", gn_k3)
        _expect(
            _abs(gn_k3 - gn_k1) > 0.1 * gn_k1,
            "2b — |grad K=3| differs >10% from |grad K=1|",
            passed,
            total,
        )

        # ─── 2c — FD vs analytic for K=1 (sanity) ───────────────────
        print()
        print("--- 2c. FD gradcheck on dynamics, K=1 (GPU sanity) ---")
        var loss_k1 = _kstep_backward(
            1, ctx, ds, enc, dyn, enc_ws, dyn_ws
        )
        print("    forward+bwd loss (K=1):", loss_k1)
        ctx.enqueue_copy(dyn_grads_h, dyn.grads_buf)
        ctx.synchronize()

        var max_rel_k1: Float64 = 0.0
        for s in range(N_SAMPLES):
            var idx = (s * 7919 + 1009) % DynModel.PARAM_SIZE
            var orig = Float64(dyn_params_h[idx])

            dyn_params_h[idx] = Scalar[dtype](orig + eps)
            ctx.enqueue_copy(dyn.params_buf, dyn_params_h)
            ctx.synchronize()
            var Lp = _kstep_forward_loss(
                1, ctx, ds, enc, dyn, enc_ws, dyn_ws
            )
            dyn_params_h[idx] = Scalar[dtype](orig - eps)
            ctx.enqueue_copy(dyn.params_buf, dyn_params_h)
            ctx.synchronize()
            var Lm = _kstep_forward_loss(
                1, ctx, ds, enc, dyn, enc_ws, dyn_ws
            )
            dyn_params_h[idx] = Scalar[dtype](orig)
            ctx.enqueue_copy(dyn.params_buf, dyn_params_h)
            ctx.synchronize()

            var fd = (Lp - Lm) / (2.0 * eps)
            var ana = Float64(dyn_grads_h[idx])
            var denom = _abs(fd) + _abs(ana) + 1e-9
            var rel = _abs(fd - ana) / denom
            if rel > max_rel_k1:
                max_rel_k1 = rel
        print("    K=1 max rel err =", max_rel_k1)
        _expect(
            max_rel_k1 < 0.05,
            "2c — analytic K=1 GPU dyn grad matches FD (5% rel)",
            passed,
            total,
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
