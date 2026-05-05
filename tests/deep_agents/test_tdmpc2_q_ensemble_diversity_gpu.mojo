"""TD-MPC2 — Q-ensemble independence, GPU (Test 4 of 5, GPU port).

Mirrors test_tdmpc2_q_ensemble_diversity.mojo but uses GPU forward/backward
paths so we exercise the same kernels production training uses.
"""

from std.math import sqrt, exp
from std.random import seed, random_float64
from std.gpu.host import DeviceContext, DeviceBuffer

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
comptime BATCH = 16
comptime ZA = LATENT + ACT
comptime NUM_Q = 5
comptime TRAIN_STEPS = 200

comptime ENC_LR = 9e-5
comptime WM_LR = 3e-4

comptime WM = WorldModel[
    OBS_DIM=OBS,
    ACTION_DIM=ACT,
    LATENT_DIM=LATENT,
    MLP_DIM=MLP,
    ENC_DIM=ENC,
    NUM_BINS=BINS,
    NUM_Q=NUM_Q,
    SIMPLEX_DIM=SIMPLEX,
    ENC_LR=ENC_LR,
    WM_LR=WM_LR,
]
comptime QModel = WM.QModel
comptime WMOpt = Adam[LR=WM_LR]
comptime Q_WS_PER_SAMPLE = QModel.WORKSPACE_SIZE_PER_SAMPLE


def _expect(cond: Bool, label: String, mut passed: Int, mut total: Int):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _decode_value(
    logits_host: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bins: InlineArray[Float32, BINS],
) -> Float64:
    var max_l: Float64 = -1e30
    for k in range(BINS):
        var v = Float64(logits_host[k])
        if v > max_l:
            max_l = v
    var sum_exp: Float64 = 0.0
    for k in range(BINS):
        sum_exp += exp(Float64(logits_host[k]) - max_l)
    var v_sym: Float64 = 0.0
    for k in range(BINS):
        var p = exp(Float64(logits_host[k]) - max_l) / sum_exp
        v_sym += p * Float64(bins[k])
    var aps = v_sym if v_sym >= 0.0 else -v_sym
    return (exp(aps) - 1.0) if v_sym >= 0.0 else -(exp(aps) - 1.0)


def _gpu_forward_q(
    ctx: DeviceContext,
    mut q: GPUNetworkState[QModel, WMOpt, dtype],
    za_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ],
    out_dev: DeviceBuffer[dtype],
    ws: DeviceBuffer[dtype],
) raises:
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
    ](out_dev.unsafe_ptr())
    Network[QModel, WMOpt].forward_gpu[BATCH](
        ctx, za_dev_t, out_t, q.params_view(), q.model_state_view(), ws
    )


def _gpu_train_step(
    ctx: DeviceContext,
    mut q: GPUNetworkState[QModel, WMOpt, dtype],
    za_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ],
    grad_logits_dev: DeviceBuffer[dtype],
    cache_dev: DeviceBuffer[dtype],
    out_dev: DeviceBuffer[dtype],
    grad_za_dev: DeviceBuffer[dtype],
    ws: DeviceBuffer[dtype],
) raises:
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
    ](out_dev.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, QModel.CACHE_SIZE),
        MutAnyOrigin,
    ](cache_dev.unsafe_ptr())
    Network[QModel, WMOpt].forward_gpu_with_cache[BATCH](
        ctx,
        za_dev_t,
        out_t,
        q.params_view(),
        q.model_state_view(),
        cache_t,
        ws,
    )
    var grad_logits_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
    ](grad_logits_dev.unsafe_ptr())
    var grad_za_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ](grad_za_dev.unsafe_ptr())
    q.zero_grads(ctx)
    var grads_v = q.grads_view()
    Network[QModel, WMOpt].backward_gpu[BATCH](
        ctx,
        grad_logits_t,
        grad_za_t,
        q.params_view(),
        q.model_state_view(),
        cache_t,
        grads_v,
        ws,
    )
    q.optimizer_step(ctx)


def main() raises:
    seed(0xCAFE13)
    print("=" * 70)
    print("TD-MPC2 Test 4 GPU — Q-ensemble independence (GPU paths)")
    print("=" * 70)

    var passed = 0
    var total = 0

    with DeviceContext() as ctx:
        # ── Build CPU init with per-Q seeds, then upload to GPU. ──
        var q1c = NetworkState[QModel, WMOpt]()
        q1c.initialize[Normal[0.0, 0.02, SEED=101]]()
        var q2c = NetworkState[QModel, WMOpt]()
        q2c.initialize[Normal[0.0, 0.02, SEED=102]]()
        var q3c = NetworkState[QModel, WMOpt]()
        q3c.initialize[Normal[0.0, 0.02, SEED=103]]()
        var q4c = NetworkState[QModel, WMOpt]()
        q4c.initialize[Normal[0.0, 0.02, SEED=104]]()
        var q5c = NetworkState[QModel, WMOpt]()
        q5c.initialize[Normal[0.0, 0.02, SEED=105]]()

        var q1g = GPUNetworkState[QModel, WMOpt, dtype](ctx)
        q1g.upload_from(q1c, ctx)
        var q2g = GPUNetworkState[QModel, WMOpt, dtype](ctx)
        q2g.upload_from(q2c, ctx)
        var q3g = GPUNetworkState[QModel, WMOpt, dtype](ctx)
        q3g.upload_from(q3c, ctx)
        var q4g = GPUNetworkState[QModel, WMOpt, dtype](ctx)
        q4g.upload_from(q4c, ctx)
        var q5g = GPUNetworkState[QModel, WMOpt, dtype](ctx)
        q5g.upload_from(q5c, ctx)

        # Workspace buffer (one is enough — forward is sequential).
        comptime WS_TOTAL = (
            BATCH * Q_WS_PER_SAMPLE if Q_WS_PER_SAMPLE > 0 else 1
        )
        var ws = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

        # Bins for value decode.
        var bin_step = 20.0 / Float64(BINS - 1)
        var bins = InlineArray[Float32, BINS](uninitialized=True)
        for i in range(BINS):
            bins[i] = Float32(-10.0 + Float64(i) * bin_step)

        # ── Two batches: train + held-out ──
        var za_train_host = ctx.enqueue_create_host_buffer[dtype](BATCH * ZA)
        var za_holdout_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * ZA
        )
        for i in range(BATCH * ZA):
            za_train_host[i] = Scalar[dtype](
                random_float64() * 0.5 - 0.25
            )
            za_holdout_host[i] = Scalar[dtype](
                random_float64() * 0.5 - 0.25
            )
        var za_train_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        var za_holdout_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)
        ctx.enqueue_copy(za_train_dev, za_train_host)
        ctx.enqueue_copy(za_holdout_dev, za_holdout_host)
        var za_train_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](za_train_dev.unsafe_ptr())
        var za_holdout_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](za_holdout_dev.unsafe_ptr())

        # Output buffer reused across the 5 Qs.
        var out_dev = ctx.enqueue_create_buffer[dtype](BATCH * BINS)
        var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * BINS)
        # Aggregate logits for all 5 Qs on host (NUM_Q * BATCH * BINS).
        var all_logits = ctx.enqueue_create_host_buffer[dtype](
            NUM_Q * BATCH * BINS
        )

        # ── 4a — std across Q logits at init ──
        print()
        print("--- 4a. Per-sample std across 5 Q logits at init (GPU) ---")
        # inlined "forward all 5 Qs into all_logits[0..NUM_Q*BATCH*BINS]"
        _gpu_forward_q(ctx, q1g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[0 * BATCH * BINS + i] = out_host[i]
        _gpu_forward_q(ctx, q2g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[1 * BATCH * BINS + i] = out_host[i]
        _gpu_forward_q(ctx, q3g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[2 * BATCH * BINS + i] = out_host[i]
        _gpu_forward_q(ctx, q4g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[3 * BATCH * BINS + i] = out_host[i]
        _gpu_forward_q(ctx, q5g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[4 * BATCH * BINS + i] = out_host[i]

        var mean_std_logits: Float64 = 0.0
        for b in range(BATCH):
            for k in range(BINS):
                var mean: Float64 = 0.0
                for q in range(NUM_Q):
                    mean += Float64(
                        all_logits[q * BATCH * BINS + b * BINS + k]
                    )
                mean /= Float64(NUM_Q)
                var var_q: Float64 = 0.0
                for q in range(NUM_Q):
                    var d = Float64(
                        all_logits[q * BATCH * BINS + b * BINS + k]
                    ) - mean
                    var_q += d * d
                var_q /= Float64(NUM_Q)
                mean_std_logits += sqrt(var_q)
        mean_std_logits /= Float64(BATCH * BINS)
        print("    init mean per-sample std across Q logits =", mean_std_logits)
        _expect(
            mean_std_logits > 1e-3,
            "4a — init logit std > 1e-3",
            passed,
            total,
        )

        # ── 4b — decoded value range at init ──
        print()
        print("--- 4b. Decoded value range across 5 Qs at init (GPU) ---")
        var mean_range_init: Float64 = 0.0
        var per_q_logits = ctx.enqueue_create_host_buffer[dtype](BINS)
        for b in range(BATCH):
            var vmin: Float64 = 1e30
            var vmax: Float64 = -1e30
            for q in range(NUM_Q):
                for k in range(BINS):
                    per_q_logits[k] = all_logits[
                        q * BATCH * BINS + b * BINS + k
                    ]
                var v = _decode_value(per_q_logits.unsafe_ptr(), bins)
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
            mean_range_init += vmax - vmin
        mean_range_init /= Float64(BATCH)
        print(
            "    init mean (max-min) decoded Q across 5 nets =",
            mean_range_init,
        )
        _expect(
            mean_range_init > 1e-3,
            "4b — init decoded Q range > 1e-3",
            passed,
            total,
        )

        # ── Train all 5 Qs with the same synthetic gradient ──
        print()
        print(
            "--- Training 5 Qs on shared synthetic targets",
            TRAIN_STEPS,
            "steps (GPU) ---",
        )
        var grad_logits_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * BINS
        )
        for i in range(BATCH * BINS):
            var t = (i * 17 + 31) % 11
            grad_logits_host[i] = Scalar[dtype](
                (Float64(t) - 5.0) / Float64(BATCH * BINS)
            )
        var grad_logits_dev = ctx.enqueue_create_buffer[dtype](BATCH * BINS)
        ctx.enqueue_copy(grad_logits_dev, grad_logits_host)

        var cache_dev = ctx.enqueue_create_buffer[dtype](
            BATCH * QModel.CACHE_SIZE
        )
        var grad_za_dev = ctx.enqueue_create_buffer[dtype](BATCH * ZA)

        for _ in range(TRAIN_STEPS):
            _gpu_train_step(
                ctx, q1g, za_train_t, grad_logits_dev, cache_dev,
                out_dev, grad_za_dev, ws,
            )
            _gpu_train_step(
                ctx, q2g, za_train_t, grad_logits_dev, cache_dev,
                out_dev, grad_za_dev, ws,
            )
            _gpu_train_step(
                ctx, q3g, za_train_t, grad_logits_dev, cache_dev,
                out_dev, grad_za_dev, ws,
            )
            _gpu_train_step(
                ctx, q4g, za_train_t, grad_logits_dev, cache_dev,
                out_dev, grad_za_dev, ws,
            )
            _gpu_train_step(
                ctx, q5g, za_train_t, grad_logits_dev, cache_dev,
                out_dev, grad_za_dev, ws,
            )
        ctx.synchronize()

        # ── 4c — held-out decoded value range after training ──
        print()
        print("--- 4c. Decoded Q range on HELD-OUT batch after training ---")
        _gpu_forward_q(ctx, q1g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[0 * BATCH * BINS + i] = out_host[i]
        _gpu_forward_q(ctx, q2g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[1 * BATCH * BINS + i] = out_host[i]
        _gpu_forward_q(ctx, q3g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[2 * BATCH * BINS + i] = out_host[i]
        _gpu_forward_q(ctx, q4g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[3 * BATCH * BINS + i] = out_host[i]
        _gpu_forward_q(ctx, q5g, za_holdout_t, out_dev, ws)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(BATCH * BINS):
            all_logits[4 * BATCH * BINS + i] = out_host[i]
        var mean_range_train: Float64 = 0.0
        for b in range(BATCH):
            var vmin: Float64 = 1e30
            var vmax: Float64 = -1e30
            for q in range(NUM_Q):
                for k in range(BINS):
                    per_q_logits[k] = all_logits[
                        q * BATCH * BINS + b * BINS + k
                    ]
                var v = _decode_value(per_q_logits.unsafe_ptr(), bins)
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
            mean_range_train += vmax - vmin
        mean_range_train /= Float64(BATCH)
        print(
            "    post-train mean (max-min) decoded Q on held-out =",
            mean_range_train,
        )
        _expect(
            mean_range_train > 1e-3,
            "4c — held-out decoded Q range > 1e-3 after training",
            passed,
            total,
        )

        # ── 4d — pairwise param distinctness ──
        print()
        print("--- 4d. Pairwise param distinctness (GPU) ---")
        # Download params from each Q to compare.
        q1g.download_to(q1c, ctx)
        q2g.download_to(q2c, ctx)
        q3g.download_to(q3c, ctx)
        q4g.download_to(q4c, ctx)
        q5g.download_to(q5c, ctx)
        ctx.synchronize()
        var qs = InlineArray[
            UnsafePointer[Scalar[dtype], MutAnyOrigin], NUM_Q
        ](uninitialized=True)
        qs[0] = q1c.params
        qs[1] = q2c.params
        qs[2] = q3c.params
        qs[3] = q4c.params
        qs[4] = q5c.params

        var all_distinct = True
        var min_pair_diff: Float64 = 1e30
        var max_pair_diff: Float64 = 0.0
        for i in range(NUM_Q):
            for j in range(i + 1, NUM_Q):
                var d_sum: Float64 = 0.0
                for k in range(QModel.PARAM_SIZE):
                    var d = Float64(qs[i][k]) - Float64(qs[j][k])
                    d_sum += d * d
                var d_norm = sqrt(d_sum)
                if d_norm > max_pair_diff:
                    max_pair_diff = d_norm
                if d_norm < min_pair_diff:
                    min_pair_diff = d_norm
                if d_norm == 0.0:
                    all_distinct = False
        print(
            "    min pairwise |Δparams| =", min_pair_diff,
            "  max pairwise |Δparams| =", max_pair_diff,
        )
        _expect(
            all_distinct,
            "4d — no pair of Q networks bitwise identical",
            passed,
            total,
        )
        _expect(
            min_pair_diff > 1e-3,
            "4d.b — min pairwise param distance > 1e-3",
            passed,
            total,
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
