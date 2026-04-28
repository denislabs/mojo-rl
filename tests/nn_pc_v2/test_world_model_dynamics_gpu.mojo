"""World-model dynamics (GPU) — Step 3 of PCN_WORLD_MODEL_ROADMAP.md.

GPU port of `test_world_model_dynamics.mojo`. Same architecture, same toy env,
same pass criteria (mean rel err <10%, var rel err <30%).

Uses the new `compute_grads_only_mcpc_gpu` and `generate_samples_gpu`
primitives from pc_trainer.mojo.

Note on performance: with BATCH=32 in training, kernel launch overhead
dominates the small per-kernel work. Training is slower than CPU. The eval
phase (EVAL_BATCH=500 imagined rollouts × T_GENERATE=500 SGLD steps) is
where the GPU genuinely wins — that's the workload that justifies a GPU
port for this kind of model.

Run:
    pixi run -e apple  mojo run -I . tests/nn_pc_v2/test_world_model_dynamics_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn_pc_v2/test_world_model_dynamics_gpu.mojo
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.math import sqrt, log, cos, pi
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.experimental.nn_pc_v2 import (
    PCBlock,
    PCSequential,
    PCTanh,
    PCTrainer,
)


comptime BATCH = 32
comptime EVAL_BATCH = 500
comptime HIDDEN = 32
comptime ACTION_DIM = 1
comptime DATA_DIM = 1
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 5
comptime EPOCHS = 160
comptime N_BATCHES_PER_EPOCH = 50
comptime T_MIXING = 50
comptime T_SAMPLING = 1
comptime T_GENERATE = 500
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.003
comptime ACTION_STEP: Float64 = 0.1
comptime SIGMA_ENV: Float64 = 0.05
comptime SGLD_NOISE_VAR: Float64 = 0.0008

comptime N_MC = 10000

comptime NET = PCSequential[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],
    PCBlock[HIDDEN, DATA_DIM, PCTanh],
]
comptime TRAINER = PCTrainer[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],
    PCBlock[HIDDEN, DATA_DIM, PCTanh],
    dtype=dtype,
]
comptime OPT = Adam[LR=ADAM_LR]


fn _set_action_target_kernel[
    BATCH: Int,
    AUG_DIM: Int,
    HIDDEN: Int,
    DATA_DIM: Int,
    KDT: DType,
](
    actions_slice: LayoutTensor[KDT, Layout.row_major(BATCH), MutAnyOrigin],
    states_slice: LayoutTensor[KDT, Layout.row_major(BATCH), MutAnyOrigin],
    x_in: LayoutTensor[KDT, Layout.row_major(BATCH, AUG_DIM), MutAnyOrigin],
    y_tgt: LayoutTensor[KDT, Layout.row_major(BATCH, DATA_DIM), MutAnyOrigin],
):
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    x_in[b, HIDDEN] = rebind[Scalar[KDT]](actions_slice[b])
    y_tgt[b, 0] = rebind[Scalar[KDT]](states_slice[b])


fn _set_prev_hidden_kernel[
    BATCH: Int, AUG_DIM: Int, HIDDEN: Int, LATENT_DIM: Int, KDT: DType,
](
    latents: LayoutTensor[
        KDT, Layout.row_major(BATCH, LATENT_DIM), MutAnyOrigin
    ],
    x_in: LayoutTensor[KDT, Layout.row_major(BATCH, AUG_DIM), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * HIDDEN:
        return
    var b = idx // HIDDEN
    var k = idx % HIDDEN
    x_in[b, k] = rebind[Scalar[KDT]](latents[b, k])


fn _set_action_eval_kernel[
    BATCH: Int, AUG_DIM: Int, HIDDEN: Int, KDT: DType,
](
    x_in: LayoutTensor[KDT, Layout.row_major(BATCH, AUG_DIM), MutAnyOrigin],
    action_value: Scalar[KDT],
):
    """Set x_in[:, HIDDEN] = action_value for all batch elements (broadcast)."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    x_in[b, HIDDEN] = action_value


fn _zero_x_in_kernel[
    BATCH: Int, AUG_DIM: Int, KDT: DType,
](
    x_in: LayoutTensor[KDT, Layout.row_major(BATCH, AUG_DIM), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * AUG_DIM:
        return
    var b = idx // AUG_DIM
    var k = idx % AUG_DIM
    x_in[b, k] = Scalar[KDT](0)


def _gauss_n01(mut rng: PhiloxRandom) -> Float64:
    var u1 = Float64(rng.step_uniform()[0])
    var u2 = Float64(rng.step_uniform()[0])
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def _sample_action(mut rng: PhiloxRandom) -> Float64:
    var u = Float64(rng.step_uniform()[0])
    return -1.0 if u < 0.5 else 1.0


def main() raises:
    print("=" * 60)
    print("World-model dynamics (GPU) — roadmap Step 3")
    print("=" * 60)
    print("  arch       : PCBlock[", AUG_DIM, ",", HIDDEN, ",PCTanh] → PCBlock[", HIDDEN, ",", DATA_DIM, ",PCTanh]")
    print("  PARAM_SIZE :", NET.PARAM_SIZE, "  LATENT_DIM:", NET.LATENT_DIM)
    print("  BATCH(train)=", BATCH, "  EVAL_BATCH=", EVAL_BATCH, "  SEQ_LEN=", SEQ_LEN)
    print("  EPOCHS=", EPOCHS, "  T_MIXING=", T_MIXING, "  T_GENERATE=", T_GENERATE)
    print("  env σ_env  :", SIGMA_ENV, "  SGLD noise_var=", SGLD_NOISE_VAR)

    var ctx = DeviceContext()

    # ── Init params on host then upload ───────────────────────────────────────
    var params_init_host = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    for i in range(NET.PARAM_SIZE):
        params_init_host.unsafe_ptr()[i] = Scalar[dtype](0)
    var params_init_t = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](params_init_host.unsafe_ptr())
    NET.initialize_params[Xavier[], dtype](params_init_t)

    var params_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    ctx.enqueue_copy(params_dbuf, params_init_host)

    # Training scratch (BATCH=32)
    var grads_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var mu_eps_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var dx_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var noise_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var x_in_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * AUG_DIM)
    var y_tgt_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DATA_DIM)

    var params_t = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](params_dbuf)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](grads_dbuf)
    var lat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](lat_dbuf)
    var mu_eps_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_OUT_DIM), MutAnyOrigin
    ](mu_eps_dbuf)
    var a_below_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](a_below_dbuf)
    var z_below_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](z_below_dbuf)
    var dx_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](dx_dbuf)
    var noise_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](noise_dbuf)
    var x_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, AUG_DIM), MutAnyOrigin
    ](x_in_dbuf)
    var y_tgt_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DATA_DIM), MutAnyOrigin
    ](y_tgt_dbuf)

    # Adam state on GPU
    var opt_state_dbuf = ctx.enqueue_create_buffer[dtype](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var opt_global_dbuf = ctx.enqueue_create_buffer[dtype](OPT.GLOBAL_STATE_SIZE)
    var opt_state_init = ctx.enqueue_create_host_buffer[dtype](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    for i in range(NET.PARAM_SIZE * OPT.STATE_PER_PARAM):
        opt_state_init.unsafe_ptr()[i] = Scalar[dtype](0)
    ctx.enqueue_copy(opt_state_dbuf, opt_state_init)
    var opt_global_init = ctx.enqueue_create_host_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    opt_global_init.unsafe_ptr()[0] = Scalar[dtype](0)
    opt_global_init.unsafe_ptr()[1] = Scalar[dtype](1.0)
    ctx.enqueue_copy(opt_global_dbuf, opt_global_init)

    var opt_state_t = LayoutTensor[
        dtype,
        Layout.row_major(NET.PARAM_SIZE, OPT.STATE_PER_PARAM),
        MutAnyOrigin,
    ](opt_state_dbuf)
    var opt_global_t = LayoutTensor[
        dtype, Layout.row_major(OPT.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](opt_global_dbuf)

    # Per-batch rollout buffers (column-major time, contiguous per-step slice).
    var actions_host = ctx.enqueue_create_host_buffer[dtype](SEQ_LEN * BATCH)
    var states_host = ctx.enqueue_create_host_buffer[dtype]((SEQ_LEN + 1) * BATCH)
    var actions_dbuf = ctx.enqueue_create_buffer[dtype](SEQ_LEN * BATCH)
    var states_dbuf = ctx.enqueue_create_buffer[dtype]((SEQ_LEN + 1) * BATCH)

    ctx.synchronize()

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | wall_t (s)")
    print("  ------+-----------")

    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var step_num: Int = 0
    var noise_offset: UInt64 = 1_000_000
    var philox_seed: UInt64 = 42
    comptime PHILOX_BUMP_PER_TRAIN_STEP = UInt64(
        BATCH * NET.LATENT_DIM * (T_MIXING + T_SAMPLING) * 2
    )

    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            # Generate stochastic rollouts on host.
            for b in range(BATCH):
                states_host.unsafe_ptr()[0 * BATCH + b] = Scalar[dtype](0.0)
            for t in range(SEQ_LEN):
                for b in range(BATCH):
                    var a = _sample_action(rng)
                    actions_host.unsafe_ptr()[t * BATCH + b] = Scalar[dtype](a)
                    var noise = SIGMA_ENV * _gauss_n01(rng)
                    var s_prev = Float64(
                        states_host.unsafe_ptr()[t * BATCH + b]
                    )
                    states_host.unsafe_ptr()[(t + 1) * BATCH + b] = Scalar[
                        dtype
                    ](s_prev + ACTION_STEP * a + noise)

            ctx.enqueue_copy(actions_dbuf, actions_host)
            ctx.enqueue_copy(states_dbuf, states_host)

            # Reset prev_hidden = 0
            comptime zero_k = _zero_x_in_kernel[BATCH, AUG_DIM, dtype]
            var zero_threads = BATCH * AUG_DIM
            var zero_blocks = (zero_threads + TPB - 1) // TPB
            ctx.enqueue_function[zero_k, zero_k](
                x_in_t,
                grid_dim=(zero_blocks,), block_dim=(TPB,),
            )

            for t in range(1, SEQ_LEN + 1):
                var actions_slice = LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ](actions_dbuf.unsafe_ptr() + (t - 1) * BATCH)
                var states_slice = LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ](states_dbuf.unsafe_ptr() + t * BATCH)

                comptime sat_k = _set_action_target_kernel[
                    BATCH, AUG_DIM, HIDDEN, DATA_DIM, dtype
                ]
                var sat_blocks = (BATCH + TPB - 1) // TPB
                ctx.enqueue_function[sat_k, sat_k](
                    actions_slice, states_slice, x_in_t, y_tgt_t,
                    grid_dim=(sat_blocks,), block_dim=(TPB,),
                )

                TRAINER.compute_grads_only_mcpc_gpu[BATCH](
                    ctx,
                    params_t, grads_t, lat_t,
                    mu_eps_t, a_below_t, z_below_t, dx_t, noise_t,
                    x_in_t, y_tgt_t,
                    T_mixing=T_MIXING,
                    T_sampling=T_SAMPLING,
                    lr_x=Scalar[dtype](LR_X),
                    noise_var=Scalar[dtype](SGLD_NOISE_VAR),
                    seed=philox_seed,
                    offset_base=noise_offset,
                )
                noise_offset += PHILOX_BUMP_PER_TRAIN_STEP

                step_num += 1
                OPT.step_gpu[NET.PARAM_SIZE, dtype](
                    ctx, params_t, grads_t, opt_state_t, opt_global_t, step_num
                )

                comptime sph_k = _set_prev_hidden_kernel[
                    BATCH, AUG_DIM, HIDDEN, NET.LATENT_DIM, dtype
                ]
                var sph_threads = BATCH * HIDDEN
                var sph_blocks = (sph_threads + TPB - 1) // TPB
                ctx.enqueue_function[sph_k, sph_k](
                    lat_t, x_in_t,
                    grid_dim=(sph_blocks,), block_dim=(TPB,),
                )

        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == EPOCHS - 1:
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - t0) / 1e9
            print("    ", epoch, "  ", String(elapsed)[byte=:7])

    ctx.synchronize()
    var total_train_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_train_t, "s")

    # ── MC ground truth on host ──────────────────────────────────────────────
    print("\n  [eval] Monte Carlo ground truth: N_MC=", N_MC)
    var mc_mean = ctx.enqueue_create_host_buffer[dtype](SEQ_LEN + 1)
    var mc_var = ctx.enqueue_create_host_buffer[dtype](SEQ_LEN + 1)
    for t in range(SEQ_LEN + 1):
        mc_mean.unsafe_ptr()[t] = Scalar[dtype](0)
        mc_var.unsafe_ptr()[t] = Scalar[dtype](0)

    var mc_rng = PhiloxRandom(seed=UInt64(2026), offset=UInt64(0))
    var mc_states_host = ctx.enqueue_create_host_buffer[dtype](
        N_MC * (SEQ_LEN + 1)
    )
    for n in range(N_MC):
        mc_states_host.unsafe_ptr()[n * (SEQ_LEN + 1) + 0] = Scalar[dtype](0.0)
        for t in range(SEQ_LEN):
            var noise = SIGMA_ENV * _gauss_n01(mc_rng)
            var s_prev = Float64(
                mc_states_host.unsafe_ptr()[n * (SEQ_LEN + 1) + t]
            )
            mc_states_host.unsafe_ptr()[n * (SEQ_LEN + 1) + t + 1] = Scalar[
                dtype
            ](s_prev + ACTION_STEP * 1.0 + noise)
    for t in range(SEQ_LEN + 1):
        var sum_v: Float64 = 0
        for n in range(N_MC):
            sum_v += Float64(
                mc_states_host.unsafe_ptr()[n * (SEQ_LEN + 1) + t]
            )
        mc_mean.unsafe_ptr()[t] = Scalar[dtype](sum_v / Float64(N_MC))
        var sum_sq: Float64 = 0
        for n in range(N_MC):
            var d = (
                Float64(mc_states_host.unsafe_ptr()[n * (SEQ_LEN + 1) + t])
                - Float64(mc_mean.unsafe_ptr()[t])
            )
            sum_sq += d * d
        mc_var.unsafe_ptr()[t] = Scalar[dtype](sum_sq / Float64(N_MC))

    # ── Imagined rollouts on GPU (EVAL_BATCH=500) ────────────────────────────
    print("\n  [eval] generating", EVAL_BATCH, "imagined rollouts via generate_samples_gpu per step")

    var eval_lat_dbuf = ctx.enqueue_create_buffer[dtype](
        EVAL_BATCH * NET.LATENT_DIM
    )
    var eval_mu_eps_dbuf = ctx.enqueue_create_buffer[dtype](
        EVAL_BATCH * NET.SCRATCH_OUT_DIM
    )
    var eval_a_below_dbuf = ctx.enqueue_create_buffer[dtype](
        EVAL_BATCH * NET.SCRATCH_IN_DIM
    )
    var eval_z_below_dbuf = ctx.enqueue_create_buffer[dtype](
        EVAL_BATCH * NET.SCRATCH_IN_DIM
    )
    var eval_dx_dbuf = ctx.enqueue_create_buffer[dtype](
        EVAL_BATCH * NET.LATENT_DIM
    )
    var eval_noise_dbuf = ctx.enqueue_create_buffer[dtype](
        EVAL_BATCH * NET.LATENT_DIM
    )
    var eval_x_in_dbuf = ctx.enqueue_create_buffer[dtype](EVAL_BATCH * AUG_DIM)
    var eval_y_dummy_dbuf = ctx.enqueue_create_buffer[dtype](
        EVAL_BATCH * DATA_DIM
    )
    var eval_sample_dbuf = ctx.enqueue_create_buffer[dtype](
        EVAL_BATCH * DATA_DIM
    )

    var eval_lat_t = LayoutTensor[
        dtype, Layout.row_major(EVAL_BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](eval_lat_dbuf)
    var eval_mu_eps_t = LayoutTensor[
        dtype,
        Layout.row_major(EVAL_BATCH, NET.SCRATCH_OUT_DIM),
        MutAnyOrigin,
    ](eval_mu_eps_dbuf)
    var eval_a_below_t = LayoutTensor[
        dtype,
        Layout.row_major(EVAL_BATCH, NET.SCRATCH_IN_DIM),
        MutAnyOrigin,
    ](eval_a_below_dbuf)
    var eval_z_below_t = LayoutTensor[
        dtype,
        Layout.row_major(EVAL_BATCH, NET.SCRATCH_IN_DIM),
        MutAnyOrigin,
    ](eval_z_below_dbuf)
    var eval_dx_t = LayoutTensor[
        dtype, Layout.row_major(EVAL_BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](eval_dx_dbuf)
    var eval_noise_t = LayoutTensor[
        dtype, Layout.row_major(EVAL_BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](eval_noise_dbuf)
    var eval_x_in_t = LayoutTensor[
        dtype, Layout.row_major(EVAL_BATCH, AUG_DIM), MutAnyOrigin
    ](eval_x_in_dbuf)
    var eval_y_dummy_t = LayoutTensor[
        dtype, Layout.row_major(EVAL_BATCH, DATA_DIM), MutAnyOrigin
    ](eval_y_dummy_dbuf)
    var eval_sample_t = LayoutTensor[
        dtype, Layout.row_major(EVAL_BATCH, DATA_DIM), MutAnyOrigin
    ](eval_sample_dbuf)

    # Zero eval_x_in to start
    comptime zero_k_eval = _zero_x_in_kernel[EVAL_BATCH, AUG_DIM, dtype]
    var zero_threads_eval = EVAL_BATCH * AUG_DIM
    var zero_blocks_eval = (zero_threads_eval + TPB - 1) // TPB
    ctx.enqueue_function[zero_k_eval, zero_k_eval](
        eval_x_in_t,
        grid_dim=(zero_blocks_eval,), block_dim=(TPB,),
    )

    var imagined = ctx.enqueue_create_host_buffer[dtype](EVAL_BATCH * SEQ_LEN)
    var sample_host = ctx.enqueue_create_host_buffer[dtype](EVAL_BATCH * DATA_DIM)

    var eval_offset: UInt64 = 5_000_000_000
    var eval_seed: UInt64 = 99
    comptime PHILOX_BUMP_PER_GEN_STEP = UInt64(
        EVAL_BATCH * NET.LATENT_DIM * T_GENERATE * 2
    )

    var t_eval0 = perf_counter_ns()

    for t in range(1, SEQ_LEN + 1):
        # Set action slot (broadcast +1 across all 500 imagined rollouts).
        comptime sa_k = _set_action_eval_kernel[
            EVAL_BATCH, AUG_DIM, HIDDEN, dtype
        ]
        var sa_blocks = (EVAL_BATCH + TPB - 1) // TPB
        ctx.enqueue_function[sa_k, sa_k](
            eval_x_in_t, Scalar[dtype](1.0),
            grid_dim=(sa_blocks,), block_dim=(TPB,),
        )

        TRAINER.generate_samples_gpu[EVAL_BATCH](
            ctx,
            params_t, eval_lat_t,
            eval_mu_eps_t, eval_a_below_t, eval_z_below_t,
            eval_dx_t, eval_noise_t,
            eval_x_in_t, eval_y_dummy_t, eval_sample_t,
            T=T_GENERATE,
            lr_x=Scalar[dtype](LR_X),
            noise_var=Scalar[dtype](SGLD_NOISE_VAR),
            seed=eval_seed,
            offset_base=eval_offset,
        )
        eval_offset += PHILOX_BUMP_PER_GEN_STEP

        # Download samples and store in imagined[n, t-1]
        ctx.enqueue_copy(sample_host, eval_sample_dbuf)
        ctx.synchronize()
        for n in range(EVAL_BATCH):
            imagined.unsafe_ptr()[n * SEQ_LEN + (t - 1)] = sample_host.unsafe_ptr()[
                n * DATA_DIM
            ]

        # prev_hidden = z_t (post-SGLD latent) — copy lat → x_in[:, 0:HIDDEN].
        comptime sph_k_eval = _set_prev_hidden_kernel[
            EVAL_BATCH, AUG_DIM, HIDDEN, NET.LATENT_DIM, dtype
        ]
        var sph_threads_eval = EVAL_BATCH * HIDDEN
        var sph_blocks_eval = (sph_threads_eval + TPB - 1) // TPB
        ctx.enqueue_function[sph_k_eval, sph_k_eval](
            eval_lat_t, eval_x_in_t,
            grid_dim=(sph_blocks_eval,), block_dim=(TPB,),
        )

    ctx.synchronize()
    var eval_t = Float64(perf_counter_ns() - t_eval0) / 1e9
    print("  total eval time :", eval_t, "s")

    # ── Compare imagined stats to MC truth ───────────────────────────────────
    print("\n  step | imagined mean | true mean | rel-err |  imag var | true var | rel-err")
    print("  -----+---------------+-----------+---------+-----------+----------+---------")

    var mean_rel_err_total: Float64 = 0.0
    var var_rel_err_total: Float64 = 0.0
    for t in range(1, SEQ_LEN + 1):
        var sum_v: Float64 = 0
        for n in range(EVAL_BATCH):
            sum_v += Float64(imagined.unsafe_ptr()[n * SEQ_LEN + (t - 1)])
        var imag_mean = sum_v / Float64(EVAL_BATCH)

        var sum_sq: Float64 = 0
        for n in range(EVAL_BATCH):
            var d = (
                Float64(imagined.unsafe_ptr()[n * SEQ_LEN + (t - 1)])
                - imag_mean
            )
            sum_sq += d * d
        var imag_var = sum_sq / Float64(EVAL_BATCH)

        var true_mean = Float64(mc_mean.unsafe_ptr()[t])
        var true_var = Float64(mc_var.unsafe_ptr()[t])

        var mean_err = imag_mean - true_mean
        if mean_err < 0:
            mean_err = -mean_err
        var mean_rel = mean_err / true_mean if true_mean != 0.0 else mean_err

        var var_err = imag_var - true_var
        if var_err < 0:
            var_err = -var_err
        var var_rel = var_err / true_var if true_var != 0.0 else var_err

        mean_rel_err_total += mean_rel
        var_rel_err_total += var_rel

        print(
            "    ", t,
            "    ", String(imag_mean)[byte=:9],
            "    ", String(true_mean)[byte=:7],
            "  ", String(mean_rel)[byte=:7],
            "    ", String(imag_var)[byte=:9],
            "  ", String(true_var)[byte=:7],
            "  ", String(var_rel)[byte=:7],
        )

    var avg_mean_rel = mean_rel_err_total / Float64(SEQ_LEN)
    var avg_var_rel = var_rel_err_total / Float64(SEQ_LEN)

    print("\n  avg mean rel err :", avg_mean_rel)
    print("  avg var  rel err :", avg_var_rel)

    var pass_mean = avg_mean_rel < 0.10
    var pass_var = avg_var_rel < 0.30

    if pass_mean and pass_var:
        print("\n  [PASS] world-model dynamics GPU: mean ≤10%, var ≤30%")
    else:
        if not pass_mean:
            print("\n  [FAIL] avg mean rel err", avg_mean_rel, "≥ 0.10")
        if not pass_var:
            print("\n  [FAIL] avg var  rel err", avg_var_rel, "≥ 0.30")
        raise Error("world-model dynamics GPU test failed")

    print("=== Done ===")
