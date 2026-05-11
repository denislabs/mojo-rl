"""TD-MPC2 — consistency loss in isolation, GPU (Test 1 of 5, GPU port).

Mirrors test_tdmpc2_consistency_isolation.mojo but uses real GPU paths
(forward_gpu_with_cache, backward_gpu, GPUNetworkState) so it exercises
the same kernels production training uses.

For simplicity, MSE loss + grad_z_pred are computed on host after
copying z_pred / z_target back, since BATCH * LATENT is tiny.
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
comptime BATCH = 64
comptime ZA = LATENT + ACT
comptime NUM_STEPS = 1000
comptime LOG_EVERY = 100

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
comptime ENC_WS = (
    BATCH * EncModel.WORKSPACE_SIZE_PER_SAMPLE
    if EncModel.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
)
comptime DYN_WS = (
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


def _l2_norm_host(
    h: HostBuffer[dtype], n: Int
) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v * v
    return sqrt(s)


def _batch_std_host(
    h: HostBuffer[dtype],
    batch: Int,
    feat: Int,
) -> Float64:
    var sum_std: Float64 = 0.0
    for k in range(feat):
        var mean: Float64 = 0.0
        var sumsq: Float64 = 0.0
        for b in range(batch):
            var v = Float64(h[b * feat + k])
            mean += v
            sumsq += v * v
        mean /= Float64(batch)
        var v = (sumsq / Float64(batch)) - mean * mean
        if v < 0.0:
            v = 0.0
        sum_std += sqrt(v)
    return sum_std / Float64(feat)


def main() raises:
    seed(0xC0FFEE)
    print("=" * 70)
    print("TD-MPC2 Test 1 GPU — Consistency loss in isolation (GPU paths)")
    print("=" * 70)
    var passed = 0
    var total = 0

    with DeviceContext() as ctx:
        # ── Build CPU init + upload to GPU ──
        var enc_cpu = NetworkState[EncModel, EncOpt]()
        enc_cpu.initialize[Normal[0.0, 0.02]]()
        var dyn_cpu = NetworkState[DynModel, DynOpt]()
        dyn_cpu.initialize[Normal[0.0, 0.02]]()

        var enc_g = GPUNetworkState[EncModel, EncOpt, dtype](ctx)
        enc_g.upload_from(enc_cpu, ctx)
        var dyn_g = GPUNetworkState[DynModel, DynOpt, dtype](ctx)
        dyn_g.upload_from(dyn_cpu, ctx)

        # Workspace
        var enc_ws = ctx.enqueue_create_buffer[dtype](ENC_WS)
        var dyn_ws = ctx.enqueue_create_buffer[dtype](DYN_WS)

        # ── Build fixed dataset on host, then push to device ──
        var W = InlineArray[Float64, OBS * ACT](uninitialized=True)
        for i in range(OBS * ACT):
            W[i] = random_float64() * 2.0 - 1.0
        var obs_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
        var act_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ACT)
        var obs_next_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * OBS
        )
        for b in range(BATCH):
            for d in range(OBS):
                obs_host[b * OBS + d] = Scalar[dtype](
                    random_float64() * 2.0 - 1.0
                )
            for d in range(ACT):
                act_host[b * ACT + d] = Scalar[dtype](
                    random_float64() * 2.0 - 1.0
                )
            for d in range(OBS):
                var s = Float64(obs_host[b * OBS + d])
                var s_next = s
                for k in range(ACT):
                    s_next += (
                        0.1 * W[d * ACT + k]
                        * Float64(act_host[b * ACT + k])
                    )
                obs_next_host[b * OBS + d] = Scalar[dtype](s_next)

        var obs_dev = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
        var act_dev = ctx.enqueue_create_buffer[dtype](BATCH * ACT)
        var obs_next_dev = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
        ctx.enqueue_copy(obs_dev, obs_host)
        ctx.enqueue_copy(act_dev, act_host)
        ctx.enqueue_copy(obs_next_dev, obs_next_host)

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](obs_dev.unsafe_ptr())
        var obs_next_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](obs_next_dev.unsafe_ptr())

        # ── Persistent device buffers ──
        var z_t_dev = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
        var z_target_dev = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
        var z_pred_dev = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
        var enc_cache_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * EncModel.CACHE_SIZE
        )
        var dyn_cache_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * DynModel.CACHE_SIZE
        )
        var za_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var grad_z_pred_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * LATENT
        )
        var grad_za_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var grad_z_dev = ctx.enqueue_create_buffer[dtype](BATCH * LATENT)
        var grad_obs_dev = ctx.enqueue_create_buffer[dtype](BATCH * OBS)

        # Host-side scratch.
        var z_t_host = ctx.enqueue_create_host_buffer[dtype](BATCH * LATENT)
        var z_target_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        var z_pred_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        var za_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
        var grad_z_pred_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        var grad_za_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
        var grad_z_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * LATENT
        )
        var enc_grads_host = ctx.enqueue_create_host_buffer[dtype](
            EncModel.PARAM_SIZE
        )
        var dyn_grads_host = ctx.enqueue_create_host_buffer[dtype](
            DynModel.PARAM_SIZE
        )

        var z_t_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_t_dev.unsafe_ptr())
        var z_target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_target_dev.unsafe_ptr())
        var z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_pred_dev.unsafe_ptr())
        var enc_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, EncModel.CACHE_SIZE),
            MutAnyOrigin,
        ](enc_cache_dev.unsafe_ptr())
        var dyn_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ](dyn_cache_dev.unsafe_ptr())
        var za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](za_dev.unsafe_ptr())
        var grad_z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_pred_dev.unsafe_ptr())
        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za_dev.unsafe_ptr())
        var grad_z_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_dev.unsafe_ptr())
        var grad_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](grad_obs_dev.unsafe_ptr())

        # ── Track properties ──
        var initial_loss: Float64 = -1.0
        var final_loss: Float64 = 0.0
        var min_std_pred: Float64 = 1e30
        var min_std_target: Float64 = 1e30
        var enc_zero_grad_steps = 0
        var dyn_zero_grad_steps = 0

        print("[step]  loss      std(z_pred) std(z_target) |∇enc|     |∇dyn|")
        for step in range(NUM_STEPS):
            # encoder forward (s_t) with cache
            Network[EncModel, EncOpt].forward_gpu_with_cache[BATCH](
                ctx, obs_t, z_t_t,
                enc_g.params_view(), enc_g.model_state_view(),
                enc_cache_t, enc_ws,
            )
            # encoder forward (s_{t+1}) — stop-grad target, no cache
            Network[EncModel, EncOpt].forward_gpu[BATCH](
                ctx, obs_next_t, z_target_t,
                enc_g.params_view(), enc_g.model_state_view(), enc_ws,
            )
            # Build za on host (z_t + a) then upload
            ctx.enqueue_copy(z_t_host, z_t_dev)
            ctx.synchronize()
            for b in range(BATCH):
                for k in range(LATENT):
                    za_host[b * ZA + k] = z_t_host[b * LATENT + k]
                for k in range(ACT):
                    za_host[b * ZA + LATENT + k] = act_host[b * ACT + k]
            ctx.enqueue_copy(za_dev, za_host)

            # dynamics forward with cache
            Network[DynModel, DynOpt].forward_gpu_with_cache[BATCH](
                ctx, za_t, z_pred_t,
                dyn_g.params_view(), dyn_g.model_state_view(),
                dyn_cache_t, dyn_ws,
            )
            # download z_pred + z_target → host, compute MSE loss + grad
            ctx.enqueue_copy(z_pred_host, z_pred_dev)
            ctx.enqueue_copy(z_target_host, z_target_dev)
            ctx.synchronize()

            var loss: Float64 = 0.0
            var sc = 2.0 / Float64(BATCH * LATENT)
            for b in range(BATCH):
                for k in range(LATENT):
                    var diff = (
                        Float64(z_pred_host[b * LATENT + k])
                        - Float64(z_target_host[b * LATENT + k])
                    )
                    loss += diff * diff
                    grad_z_pred_host[b * LATENT + k] = Scalar[dtype](
                        sc * diff
                    )
            loss /= Float64(BATCH * LATENT)

            var std_pred = _batch_std_host(z_pred_host, BATCH, LATENT)
            var std_target = _batch_std_host(z_target_host, BATCH, LATENT)
            if std_pred < min_std_pred:
                min_std_pred = std_pred
            if std_target < min_std_target:
                min_std_target = std_target
            if step == 0:
                initial_loss = loss
            final_loss = loss

            ctx.enqueue_copy(grad_z_pred_dev, grad_z_pred_host)

            # dynamics backward → grad_za, dyn grads accumulated
            dyn_g.zero_grads(ctx)
            var dyn_grads_v = dyn_g.grads_view()
            Network[DynModel, DynOpt].backward_gpu[BATCH](
                ctx, grad_z_pred_t, grad_za_t,
                dyn_g.params_view(), dyn_g.model_state_view(),
                dyn_cache_t, dyn_grads_v, dyn_ws,
            )
            # extract grad_z from grad_za[:, :LATENT] on host
            ctx.enqueue_copy(grad_za_host, grad_za_dev)
            ctx.synchronize()
            for b in range(BATCH):
                for k in range(LATENT):
                    grad_z_host[b * LATENT + k] = grad_za_host[b * ZA + k]
            ctx.enqueue_copy(grad_z_dev, grad_z_host)

            # encoder backward
            enc_g.zero_grads(ctx)
            var enc_grads_v = enc_g.grads_view()
            Network[EncModel, EncOpt].backward_gpu[BATCH](
                ctx, grad_z_t, grad_obs_t,
                enc_g.params_view(), enc_g.model_state_view(),
                enc_cache_t, enc_grads_v, enc_ws,
            )

            # download grads to host for norm checks
            ctx.enqueue_copy(enc_grads_host, enc_g.grads_buf)
            ctx.enqueue_copy(dyn_grads_host, dyn_g.grads_buf)
            ctx.synchronize()

            var enc_gn = _l2_norm_host(enc_grads_host, EncModel.PARAM_SIZE)
            var dyn_gn = _l2_norm_host(dyn_grads_host, DynModel.PARAM_SIZE)
            if enc_gn == 0.0:
                enc_zero_grad_steps += 1
            if dyn_gn == 0.0:
                dyn_zero_grad_steps += 1

            enc_g.optimizer_step(ctx)
            dyn_g.optimizer_step(ctx)

            if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
                print(
                    "[" + String(step) + "]",
                    String(loss)[byte=:8],
                    "  ",
                    String(std_pred)[byte=:8],
                    "  ",
                    String(std_target)[byte=:8],
                    "  ",
                    String(enc_gn)[byte=:8],
                    "  ",
                    String(dyn_gn)[byte=:8],
                )

        # ── Summary ──
        print()
        print("Initial loss:  ", initial_loss)
        print("Final loss:    ", final_loss)
        print("Reduction:     ", final_loss / initial_loss)
        print("min std(z_pred)  =", min_std_pred)
        print("min std(z_target)=", min_std_target)
        print("encoder zero-grad steps  =", enc_zero_grad_steps)
        print("dynamics zero-grad steps =", dyn_zero_grad_steps)

        _expect(
            final_loss < 0.5 * initial_loss,
            "1a — final loss < 0.5 * initial loss (loss decreasing)",
            passed,
            total,
        )
        _expect(
            min_std_pred > 0.05,
            "1b — std(z_pred) stays > 0.05",
            passed,
            total,
        )
        _expect(
            min_std_target > 0.05,
            "1c — std(z_target) stays > 0.05",
            passed,
            total,
        )
        _expect(
            enc_zero_grad_steps == 0,
            "1d — encoder grad-norm > 0 every step",
            passed,
            total,
        )
        _expect(
            dyn_zero_grad_steps == 0,
            "1e — dynamics grad-norm > 0 every step",
            passed,
            total,
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
