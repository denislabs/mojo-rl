"""TD-MPC2 — encoder gradient reach, GPU (Test 3 of 5, GPU port).

Mirrors test_tdmpc2_encoder_grad_reach.mojo but runs through GPU
kernels (forward_gpu_with_cache + backward_gpu) so we exercise the
production paths.

Sub-tests:
  3a — consistency-only loss reaches encoder
  3b — reward-only loss reaches encoder
  3c — Q1-only loss reaches encoder
  3d — termination-only loss reaches encoder
  3e — sum(per-head ∇enc) ≈ combined ∇enc (linearity of GPU backprop)
"""

from std.math import sqrt
from std.random import seed, random_float64
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import alloc, memset

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
comptime BINS = 11
comptime BATCH = 4
comptime ZA = LATENT + ACT
comptime ENC_LR = 9e-5
comptime WM_LR = 3e-4

comptime WM = WorldModel[
    OBS_DIM=OBS,
    ACTION_DIM=ACT,
    LATENT_DIM=LATENT,
    MLP_DIM=MLP,
    ENC_DIM=ENC,
    NUM_BINS=BINS,
    NUM_Q=2,
    SIMPLEX_DIM=SIMPLEX,
    ENC_LR=ENC_LR,
    WM_LR=WM_LR,
]
comptime EncModel = WM.EncModel
comptime DynModel = WM.DynModel
comptime RewModel = WM.RewModel
comptime QModel = WM.QModel
comptime TermModel = WM.TermModel
comptime EncOpt = Adam[LR=ENC_LR]
comptime WMOpt = Adam[LR=WM_LR]


def _ws_size[M: AnyType](batch: Int) -> Int:
    return 1


def _expect(cond: Bool, label: String, mut passed: Int, mut total: Int):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _l2_norm_host(h: HostBuffer[dtype], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v * v
    return sqrt(s)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


# Returns: (grad_norm, snapshot host buffer of encoder grads)
struct GradReachResult(Movable):
    var grad_norm: Float64
    var snap: HostBuffer[dtype]

    def __init__(
        out self, grad_norm: Float64, snap: HostBuffer[dtype]
    ):
        self.grad_norm = grad_norm
        self.snap = snap


def _backward_through_gpu(
    ctx: DeviceContext,
    which: Int,
    obs0_dev: DeviceBuffer[dtype],
    obs1_dev: DeviceBuffer[dtype],
    act_host: HostBuffer[dtype],
    mut enc: GPUNetworkState[EncModel, EncOpt, dtype],
    mut dyn: GPUNetworkState[DynModel, WMOpt, dtype],
    mut rew: GPUNetworkState[RewModel, WMOpt, dtype],
    mut q1: GPUNetworkState[QModel, WMOpt, dtype],
    mut term: GPUNetworkState[TermModel, WMOpt, dtype],
    enc_ws: DeviceBuffer[dtype],
    dyn_ws: DeviceBuffer[dtype],
    rew_ws: DeviceBuffer[dtype],
    q_ws: DeviceBuffer[dtype],
    term_ws: DeviceBuffer[dtype],
) raises -> GradReachResult:
    # Encoder forward (s_0) with cache.
    var z0_dev = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
    var z0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z0_dev.unsafe_ptr())
    var enc_cache_dev = ctx.enqueue_create_buffer[dtype](
        BATCH * EncModel.CACHE_SIZE
    )
    var enc_cache_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, EncModel.CACHE_SIZE),
        MutAnyOrigin,
    ](enc_cache_dev.unsafe_ptr())
    var obs0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](obs0_dev.unsafe_ptr())
    Network[EncModel, EncOpt].forward_gpu_with_cache[BATCH](
        ctx, obs0_t, z0_t,
        enc.params_view(), enc.model_state_view(),
        enc_cache_t, enc_ws,
    )

    # Build za on host then upload.
    var z0_host = ctx.enqueue_create_host_buffer[dtype](BATCH * LATENT)
    ctx.enqueue_copy(z0_host, z0_dev)
    ctx.synchronize()
    var za_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
    for b in range(BATCH):
        for k in range(LATENT):
            za_host[b * ZA + k] = z0_host[b * LATENT + k]
        for k in range(ACT):
            za_host[b * ZA + LATENT + k] = act_host[b * ACT + k]
    var za_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
    ctx.enqueue_copy(za_dev, za_host)
    var za_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ](za_dev.unsafe_ptr())

    # Accumulator for grad_z (carry into encoder).
    var grad_z_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * LATENT
    )
    for i in range(BATCH * LATENT):
        grad_z_host[i] = Scalar[dtype](0.0)

    # ---- Consistency path ----
    if which == 0 or which == 4 or which == 5 or which == 6:
        var obs1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](obs1_dev.unsafe_ptr())
        var z_target_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * LATENT
        )
        var z_target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_target_dev.unsafe_ptr())
        Network[EncModel, EncOpt].forward_gpu[BATCH](
            ctx, obs1_t, z_target_t,
            enc.params_view(), enc.model_state_view(), enc_ws,
        )
        var z_pred_dev = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
        var z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_pred_dev.unsafe_ptr())
        var dyn_cache_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * DynModel.CACHE_SIZE
        )
        var dyn_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ](dyn_cache_dev.unsafe_ptr())
        Network[DynModel, WMOpt].forward_gpu_with_cache[BATCH](
            ctx, za_t, z_pred_t,
            dyn.params_view(), dyn.model_state_view(),
            dyn_cache_t, dyn_ws,
        )
        # MSE grad on host
        var z_pred_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        var z_target_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        ctx.enqueue_copy(z_pred_host, z_pred_dev)
        ctx.enqueue_copy(z_target_host, z_target_dev)
        ctx.synchronize()
        var grad_z_pred_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        var sc = 2.0 / Float64(BATCH * LATENT)
        for b in range(BATCH):
            for k in range(LATENT):
                var diff = (
                    Float64(z_pred_host[b * LATENT + k])
                    - Float64(z_target_host[b * LATENT + k])
                )
                grad_z_pred_host[b * LATENT + k] = Scalar[dtype](
                    sc * diff
                )
        var grad_z_pred_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * LATENT
        )
        ctx.enqueue_copy(grad_z_pred_dev, grad_z_pred_host)
        var grad_z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_pred_dev.unsafe_ptr())
        var grad_za_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za_dev.unsafe_ptr())
        dyn.zero_grads(ctx)
        var dyn_grads_v = dyn.grads_view()
        Network[DynModel, WMOpt].backward_gpu[BATCH](
            ctx, grad_z_pred_t, grad_za_t,
            dyn.params_view(), dyn.model_state_view(),
            dyn_cache_t, dyn_grads_v, dyn_ws,
        )
        # extract z portion → grad_z_host
        var grad_za_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * ZA
        )
        ctx.enqueue_copy(grad_za_host, grad_za_dev)
        ctx.synchronize()
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_host[b * LATENT + k] = (
                    grad_z_host[b * LATENT + k]
                    + grad_za_host[b * ZA + k]
                )

    # ---- Reward path ----
    if which == 1 or which == 4 or which == 5 or which == 6:
        var rew_logits_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * BINS
        )
        var rew_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
        ](rew_logits_dev.unsafe_ptr())
        var rew_cache_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * RewModel.CACHE_SIZE
        )
        var rew_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, RewModel.CACHE_SIZE),
            MutAnyOrigin,
        ](rew_cache_dev.unsafe_ptr())
        Network[RewModel, WMOpt].forward_gpu_with_cache[BATCH](
            ctx, za_t, rew_logits_t,
            rew.params_view(), rew.model_state_view(),
            rew_cache_t, rew_ws,
        )
        # Pseudo-loss: L = mean(logits) → grad_logits = 1 / (B*BINS)
        var grad_logits_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * BINS
        )
        for i in range(BATCH * BINS):
            grad_logits_host[i] = Scalar[dtype](
                1.0 / Float64(BATCH * BINS)
            )
        var grad_logits_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * BINS
        )
        ctx.enqueue_copy(grad_logits_dev, grad_logits_host)
        var grad_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
        ](grad_logits_dev.unsafe_ptr())
        var grad_za_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za_dev.unsafe_ptr())
        rew.zero_grads(ctx)
        var rew_grads_v = rew.grads_view()
        Network[RewModel, WMOpt].backward_gpu[BATCH](
            ctx, grad_logits_t, grad_za_t,
            rew.params_view(), rew.model_state_view(),
            rew_cache_t, rew_grads_v, rew_ws,
        )
        var grad_za_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * ZA
        )
        ctx.enqueue_copy(grad_za_host, grad_za_dev)
        ctx.synchronize()
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_host[b * LATENT + k] = (
                    grad_z_host[b * LATENT + k]
                    + grad_za_host[b * ZA + k]
                )

    # ---- Q1 path ----
    if which == 2 or which == 4:
        var q_logits_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * BINS
        )
        var q_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
        ](q_logits_dev.unsafe_ptr())
        var q_cache_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * QModel.CACHE_SIZE
        )
        var q_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, QModel.CACHE_SIZE),
            MutAnyOrigin,
        ](q_cache_dev.unsafe_ptr())
        Network[QModel, WMOpt].forward_gpu_with_cache[BATCH](
            ctx, za_t, q_logits_t,
            q1.params_view(), q1.model_state_view(),
            q_cache_t, q_ws,
        )
        var grad_logits_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * BINS
        )
        for i in range(BATCH * BINS):
            grad_logits_host[i] = Scalar[dtype](
                1.0 / Float64(BATCH * BINS)
            )
        var grad_logits_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * BINS
        )
        ctx.enqueue_copy(grad_logits_dev, grad_logits_host)
        var grad_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
        ](grad_logits_dev.unsafe_ptr())
        var grad_za_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za_dev.unsafe_ptr())
        q1.zero_grads(ctx)
        var q1_grads_v = q1.grads_view()
        Network[QModel, WMOpt].backward_gpu[BATCH](
            ctx, grad_logits_t, grad_za_t,
            q1.params_view(), q1.model_state_view(),
            q_cache_t, q1_grads_v, q_ws,
        )
        var grad_za_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * ZA
        )
        ctx.enqueue_copy(grad_za_host, grad_za_dev)
        ctx.synchronize()
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_host[b * LATENT + k] = (
                    grad_z_host[b * LATENT + k]
                    + grad_za_host[b * ZA + k]
                )

    # ---- Termination path (takes z only) ----
    if which == 3 or which == 4 or which == 6:
        var term_out_dev = ctx.enqueue_create_buffer[dtype](BATCH * 1)
        var term_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](term_out_dev.unsafe_ptr())
        var term_cache_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * TermModel.CACHE_SIZE
        )
        var term_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, TermModel.CACHE_SIZE),
            MutAnyOrigin,
        ](term_cache_dev.unsafe_ptr())
        Network[TermModel, WMOpt].forward_gpu_with_cache[BATCH](
            ctx, z0_t, term_out_t,
            term.params_view(), term.model_state_view(),
            term_cache_t, term_ws,
        )
        var grad_term_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
        for i in range(BATCH):
            grad_term_host[i] = Scalar[dtype](1.0 / Float64(BATCH))
        var grad_term_dev = ctx.enqueue_create_buffer[dtype](BATCH)
        ctx.enqueue_copy(grad_term_dev, grad_term_host)
        var grad_term_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](grad_term_dev.unsafe_ptr())
        var grad_z_local_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * LATENT
        )
        var grad_z_local_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_local_dev.unsafe_ptr())
        term.zero_grads(ctx)
        var term_grads_v = term.grads_view()
        Network[TermModel, WMOpt].backward_gpu[BATCH](
            ctx, grad_term_t, grad_z_local_t,
            term.params_view(), term.model_state_view(),
            term_cache_t, term_grads_v, term_ws,
        )
        var grad_z_local_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        ctx.enqueue_copy(grad_z_local_host, grad_z_local_dev)
        ctx.synchronize()
        for i in range(BATCH * LATENT):
            grad_z_host[i] = grad_z_host[i] + grad_z_local_host[i]

    # ---- Encoder backward ----
    var grad_z_dev = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
    ctx.enqueue_copy(grad_z_dev, grad_z_host)
    var grad_z_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](grad_z_dev.unsafe_ptr())
    var grad_obs_dev = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
    var grad_obs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](grad_obs_dev.unsafe_ptr())
    enc.zero_grads(ctx)
    var enc_grads_v = enc.grads_view()
    Network[EncModel, EncOpt].backward_gpu[BATCH](
        ctx, grad_z_t, grad_obs_t,
        enc.params_view(), enc.model_state_view(),
        enc_cache_t, enc_grads_v, enc_ws,
    )
    var enc_grads_host = ctx.enqueue_create_host_buffer[dtype](
        EncModel.PARAM_SIZE
    )
    ctx.enqueue_copy(enc_grads_host, enc.grads_buf)
    ctx.synchronize()
    var enc_gn = _l2_norm_host(enc_grads_host, EncModel.PARAM_SIZE)
    return GradReachResult(grad_norm=enc_gn, snap=enc_grads_host^)


def main() raises:
    seed(0xCAFE99)
    print("=" * 70)
    print("TD-MPC2 Test 3 GPU — Encoder gradient reach")
    print("=" * 70)
    var passed = 0
    var total = 0

    with DeviceContext() as ctx:
        # ── Build all networks ──
        var enc_cpu = NetworkState[EncModel, EncOpt]()
        enc_cpu.initialize[Normal[0.0, 0.02]]()
        var dyn_cpu = NetworkState[DynModel, WMOpt]()
        dyn_cpu.initialize[Normal[0.0, 0.02]]()
        var rew_cpu = NetworkState[RewModel, WMOpt]()
        rew_cpu.initialize[Normal[0.0, 0.02]]()
        var q1_cpu = NetworkState[QModel, WMOpt]()
        q1_cpu.initialize[Normal[0.0, 0.02]]()
        var term_cpu = NetworkState[TermModel, WMOpt]()
        term_cpu.initialize[Normal[0.0, 0.02]]()

        var enc = GPUNetworkState[EncModel, EncOpt, dtype](ctx)
        enc.upload_from(enc_cpu, ctx)
        var dyn = GPUNetworkState[DynModel, WMOpt, dtype](ctx)
        dyn.upload_from(dyn_cpu, ctx)
        var rew = GPUNetworkState[RewModel, WMOpt, dtype](ctx)
        rew.upload_from(rew_cpu, ctx)
        var q1 = GPUNetworkState[QModel, WMOpt, dtype](ctx)
        q1.upload_from(q1_cpu, ctx)
        var term = GPUNetworkState[TermModel, WMOpt, dtype](ctx)
        term.upload_from(term_cpu, ctx)

        # Workspace buffers per model.
        comptime ENC_WS_SIZE = (
            BATCH * EncModel.WORKSPACE_SIZE_PER_SAMPLE
            if EncModel.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )
        comptime DYN_WS_SIZE = (
            BATCH * DynModel.WORKSPACE_SIZE_PER_SAMPLE
            if DynModel.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )
        comptime REW_WS_SIZE = (
            BATCH * RewModel.WORKSPACE_SIZE_PER_SAMPLE
            if RewModel.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )
        comptime Q_WS_SIZE = (
            BATCH * QModel.WORKSPACE_SIZE_PER_SAMPLE
            if QModel.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )
        comptime TERM_WS_SIZE = (
            BATCH * TermModel.WORKSPACE_SIZE_PER_SAMPLE
            if TermModel.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )
        var enc_ws = ctx.enqueue_create_buffer[dtype](ENC_WS_SIZE)
        var dyn_ws = ctx.enqueue_create_buffer[dtype](DYN_WS_SIZE)
        var rew_ws = ctx.enqueue_create_buffer[dtype](REW_WS_SIZE)
        var q_ws = ctx.enqueue_create_buffer[dtype](Q_WS_SIZE)
        var term_ws = ctx.enqueue_create_buffer[dtype](TERM_WS_SIZE)

        # ── Build batch on host, upload ──
        var obs0_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
        var obs1_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
        var act_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ACT)
        for i in range(BATCH * OBS):
            obs0_host[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
            obs1_host[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
        for i in range(BATCH * ACT):
            act_host[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
        var obs0_dev = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
        var obs1_dev = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
        ctx.enqueue_copy(obs0_dev, obs0_host)
        ctx.enqueue_copy(obs1_dev, obs1_host)

        # ── 3a–3d & 3e ──
        print()
        print("--- Per-head encoder gradient norms (loss in isolation) ---")
        var r_cons = _backward_through_gpu(
            ctx, 0, obs0_dev, obs1_dev, act_host,
            enc, dyn, rew, q1, term,
            enc_ws, dyn_ws, rew_ws, q_ws, term_ws,
        )
        var r_rew = _backward_through_gpu(
            ctx, 1, obs0_dev, obs1_dev, act_host,
            enc, dyn, rew, q1, term,
            enc_ws, dyn_ws, rew_ws, q_ws, term_ws,
        )
        var r_q = _backward_through_gpu(
            ctx, 2, obs0_dev, obs1_dev, act_host,
            enc, dyn, rew, q1, term,
            enc_ws, dyn_ws, rew_ws, q_ws, term_ws,
        )
        var r_term = _backward_through_gpu(
            ctx, 3, obs0_dev, obs1_dev, act_host,
            enc, dyn, rew, q1, term,
            enc_ws, dyn_ws, rew_ws, q_ws, term_ws,
        )
        var r_all = _backward_through_gpu(
            ctx, 4, obs0_dev, obs1_dev, act_host,
            enc, dyn, rew, q1, term,
            enc_ws, dyn_ws, rew_ws, q_ws, term_ws,
        )
        print("    consistency-only |∇enc| =", r_cons.grad_norm)
        print("    reward-only      |∇enc| =", r_rew.grad_norm)
        print("    Q1-only          |∇enc| =", r_q.grad_norm)
        print("    termination-only |∇enc| =", r_term.grad_norm)
        print("    all combined     |∇enc| =", r_all.grad_norm)

        _expect(
            r_cons.grad_norm > 0.0,
            "3a — consistency loss reaches encoder",
            passed,
            total,
        )
        _expect(
            r_rew.grad_norm > 0.0,
            "3b — reward loss reaches encoder",
            passed,
            total,
        )
        _expect(
            r_q.grad_norm > 0.0,
            "3c — Q1 loss reaches encoder",
            passed,
            total,
        )
        _expect(
            r_term.grad_norm > 0.0,
            "3d — termination loss reaches encoder",
            passed,
            total,
        )

        # 3e — linearity. Compare L2-relative error of (sum of per-head)
        # vs combined.
        var sum_a: Float64 = 0.0
        var sum_diff_sq: Float64 = 0.0
        for i in range(EncModel.PARAM_SIZE):
            var s = (
                Float64(r_cons.snap[i])
                + Float64(r_rew.snap[i])
                + Float64(r_q.snap[i])
                + Float64(r_term.snap[i])
            )
            var a = Float64(r_all.snap[i])
            var d = _abs(s - a)
            sum_diff_sq += d * d
            sum_a += a * a
        var rel_l2_4way = sqrt(sum_diff_sq) / sqrt(sum_a + 1e-12)
        print("    4-way L2 rel = ", rel_l2_4way)

        # which=5: cons+rew combined
        var r_cr = _backward_through_gpu(
            ctx, 5, obs0_dev, obs1_dev, act_host,
            enc, dyn, rew, q1, term,
            enc_ws, dyn_ws, rew_ws, q_ws, term_ws,
        )
        var sum_a_cr: Float64 = 0.0
        var sum_diff_cr: Float64 = 0.0
        for i in range(EncModel.PARAM_SIZE):
            var s = Float64(r_cons.snap[i]) + Float64(r_rew.snap[i])
            var a = Float64(r_cr.snap[i])
            var d = _abs(s - a)
            sum_diff_cr += d * d
            sum_a_cr += a * a
        var rel_l2_cr = sqrt(sum_diff_cr) / sqrt(sum_a_cr + 1e-12)
        print("    cons+rew L2 rel        =", rel_l2_cr)

        # which=6: cons+rew+term combined (deterministic — no Dropout).
        # Q is excluded because the QFirstLayer has Dropout(p=0.01,
        # training=True) which generates a different mask each forward
        # call, breaking strict linear superposition. That's expected
        # stochastic behavior, not a bug.
        var r_crt = _backward_through_gpu(
            ctx, 6, obs0_dev, obs1_dev, act_host,
            enc, dyn, rew, q1, term,
            enc_ws, dyn_ws, rew_ws, q_ws, term_ws,
        )
        var sum_a_crt: Float64 = 0.0
        var sum_diff_crt: Float64 = 0.0
        for i in range(EncModel.PARAM_SIZE):
            var s = (
                Float64(r_cons.snap[i])
                + Float64(r_rew.snap[i])
                + Float64(r_term.snap[i])
            )
            var a = Float64(r_crt.snap[i])
            var d = _abs(s - a)
            sum_diff_crt += d * d
            sum_a_crt += a * a
        var rel_l2_crt = sqrt(sum_diff_crt) / sqrt(sum_a_crt + 1e-12)
        print("    cons+rew+term L2 rel   =", rel_l2_crt)

        _expect(
            rel_l2_cr < 0.001,
            "3e — cons+rew per-head sum matches combined (L2 rel < 0.1%)",
            passed,
            total,
        )
        _expect(
            rel_l2_crt < 0.001,
            (
                "3f — cons+rew+term per-head sum matches combined"
                " (L2 rel < 0.1%; Q excluded due to stochastic dropout)"
            ),
            passed,
            total,
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
