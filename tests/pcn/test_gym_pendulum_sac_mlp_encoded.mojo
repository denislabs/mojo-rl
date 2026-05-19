"""Gymnasium Pendulum-v1 SAC — Phase-2 MLP-encoded (gym diagnostic).

Identical to `test_pendulum_sac_mlp_encoded.mojo` but uses GymPendulumEnv
at SAC time. Encoder is still trained on inlined Pendulum physics (same
recipe — both envs share Gymnasium's analytical Euler equations), so the
encoder weights match the native variant. Only the SAC env differs.

Float64 throughout (GymPendulumEnv hardcodes Float64).

Run:
    pixi run mojo run -I . tests/pcn/test_gym_pendulum_sac_mlp_encoded.mojo
"""

from std.math import sqrt, log, cos, sin, tanh, pi
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.scheduler import CosineWarmupSchedule
from mojo_rl.experimental.pcn import (
    EncoderWrappedEnv,
    PCEncoder,
    clip_grad_norm,
)
from mojo_rl.envs.gymnasium.gymnasium_classic_control import GymPendulumEnv
from mojo_rl.deep_agents.core.agents import DeepSACAgent

comptime dtype: DType = DType.float64


# Pendulum physics (Gymnasium defaults).
comptime PEND_G: Float64 = 10.0
comptime PEND_L: Float64 = 1.0
comptime PEND_M: Float64 = 1.0
comptime PEND_DT: Float64 = 0.05
comptime PEND_MAX_SPEED: Float64 = 8.0
comptime PEND_MAX_TORQUE: Float64 = 2.0

# Architecture — same as PCN-encoded variant.
comptime BATCH = 32
comptime HIDDEN = 64
comptime ACTION_DIM = 1
comptime OBS_DIM = 3
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 20
comptime K_BPTT = SEQ_LEN

comptime ENC_EPOCHS = 100
comptime N_BATCHES_PER_EPOCH = 100
comptime WARMUP_EPOCHS = 5
comptime LR_MIN_SCALE: Float64 = 0.1
comptime ADAM_LR: Float64 = 0.001
comptime GRAD_CLIP_NORM: Float64 = 1.0

comptime ENC_INPUT_DIM = HIDDEN + ACTION_DIM + OBS_DIM
comptime ENC_HIDDEN_DIM = 64
comptime ENC_OUTPUT_DIM = HIDDEN
comptime ENC = PCEncoder[ENC_INPUT_DIM, ENC_HIDDEN_DIM, ENC_OUTPUT_DIM]
comptime ENC_PARAM_SIZE = ENC.PARAM_SIZE

# BPTT layer sizes (single Linear+tanh per module).
comptime T_PARAM_SIZE = AUG_DIM * HIDDEN + HIDDEN
comptime D_PARAM_SIZE = HIDDEN * OBS_DIM + OBS_DIM
comptime T_W_OFFSET = 0
comptime T_B_OFFSET = AUG_DIM * HIDDEN
comptime D_W_OFFSET = 0
comptime D_B_OFFSET = HIDDEN * OBS_DIM

comptime OPT = Adam[LR=ADAM_LR]
comptime SCHED = CosineWarmupSchedule[
    WARMUP_EPOCHS=WARMUP_EPOCHS, MIN_SCALE=LR_MIN_SCALE
]

# SAC hyperparameters — must match baseline + PCN-encoded.
comptime SAC_NUM_STEPS = 40_000
comptime SAC_MAX_STEPS = 200
comptime SAC_WARMUP_STEPS = 1000
comptime SAC_PRINT_EVERY = 20


def _angle_normalize(t: Float64) -> Float64:
    var x = (t + pi) - 2.0 * pi * Float64(Int((t + pi) / (2.0 * pi)))
    if x < 0.0:
        x += 2.0 * pi
    return x - pi


def _step_pendulum(
    mut theta: Float64, mut theta_dot: Float64, torque: Float64
) -> Tuple[Float64, Float64]:
    var u = torque
    if u > PEND_MAX_TORQUE:
        u = PEND_MAX_TORQUE
    elif u < -PEND_MAX_TORQUE:
        u = -PEND_MAX_TORQUE
    var theta_acc = (3.0 * PEND_G) / (2.0 * PEND_L) * sin(theta) + (
        3.0 / (PEND_M * PEND_L * PEND_L)
    ) * u
    var new_dot = theta_dot + theta_acc * PEND_DT
    if new_dot > PEND_MAX_SPEED:
        new_dot = PEND_MAX_SPEED
    elif new_dot < -PEND_MAX_SPEED:
        new_dot = -PEND_MAX_SPEED
    var new_theta = _angle_normalize(theta + new_dot * PEND_DT)
    return (new_theta, new_dot)


def _gen_rollout_into[
    SEQ_LEN_T: Int
](
    mut rng: PhiloxRandom,
    actions_buf: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    obs_buf: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    actions_offset: Int,
    obs_offset: Int,
):
    var u0 = Float64(rng.step_uniform()[0])
    var u1 = Float64(rng.step_uniform()[0])
    var theta = (u0 * 2.0 - 1.0) * pi
    var theta_dot = (u1 * 2.0 - 1.0) * 1.0
    obs_buf[obs_offset + 0] = Scalar[dtype](cos(theta))
    obs_buf[obs_offset + 1] = Scalar[dtype](sin(theta))
    obs_buf[obs_offset + 2] = Scalar[dtype](theta_dot / PEND_MAX_SPEED)
    for t in range(SEQ_LEN_T):
        var ua = Float64(rng.step_uniform()[0])
        var torque_norm = ua * 2.0 - 1.0
        var torque = torque_norm * PEND_MAX_TORQUE
        actions_buf[actions_offset + t] = Scalar[dtype](torque_norm)
        var stepped = _step_pendulum(theta, theta_dot, torque)
        theta = stepped[0]
        theta_dot = stepped[1]
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 0] = Scalar[dtype](cos(theta))
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 1] = Scalar[dtype](sin(theta))
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 2] = Scalar[dtype](
            theta_dot / PEND_MAX_SPEED
        )


def _xavier_init_layer(
    params: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    in_dim: Int,
    out_dim: Int,
    seed: UInt64,
):
    var rng = PhiloxRandom(seed=seed, offset=UInt64(0))
    var bound = sqrt(Float64(6.0) / Float64(in_dim + out_dim))
    for i in range(in_dim * out_dim):
        var u = Float64(rng.step_uniform()[0])
        params[i] = Scalar[dtype]((u * 2.0 - 1.0) * bound)
    for j in range(out_dim):
        params[in_dim * out_dim + j] = Scalar[dtype](0.0)


def _lt_forward[
    BATCH_T: Int, IN: Int, OUT: Int
](
    params: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    x: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    mu: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
):
    for s in range(BATCH_T):
        for i in range(IN):
            a[s * IN + i] = Scalar[dtype](tanh(Float64(x[s * IN + i])))
        for j in range(OUT):
            var sum_j = Float64(params[IN * OUT + j])
            for i in range(IN):
                sum_j += Float64(a[s * IN + i]) * Float64(params[i * OUT + j])
            mu[s * OUT + j] = Scalar[dtype](sum_j)


def _lt_backward_accum[
    BATCH_T: Int, IN: Int, OUT: Int
](
    params: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    d_mu: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    d_W: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    d_b: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    d_x: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
):
    for s in range(BATCH_T):
        for j in range(OUT):
            d_b[j] = Scalar[dtype](Float64(d_b[j]) + Float64(d_mu[s * OUT + j]))
        for i in range(IN):
            var a_i = Float64(a[s * IN + i])
            for j in range(OUT):
                var idx = i * OUT + j
                d_W[idx] = Scalar[dtype](
                    Float64(d_W[idx]) + a_i * Float64(d_mu[s * OUT + j])
                )
        for i in range(IN):
            var d_a_i: Float64 = 0
            for j in range(OUT):
                d_a_i += Float64(params[i * OUT + j]) * Float64(
                    d_mu[s * OUT + j]
                )
            var a_i = Float64(a[s * IN + i])
            d_x[s * IN + i] = Scalar[dtype](d_a_i * (1.0 - a_i * a_i))


def main() raises:
    print("=" * 60)
    print("Pendulum SAC — Phase-2 MLP-encoded (BPTT-trained encoder)")
    print("=" * 60)
    print("  Encoder    : MLP+BPTT (K=", K_BPTT, ")")
    print("  Wrapper    : EncoderWrappedEnv → SAC sees", HIDDEN, "-dim latent")
    print("  Enc epochs :", ENC_EPOCHS)
    print("  SAC eps    :", SAC_NUM_STEPS, " steps/ep:", SAC_MAX_STEPS)

    # ────────────────────────────────────────────────────────────────────────
    # PHASE A — Train encoder via K-step BPTT (transition + decoder discarded
    # after training). Encoder params are what feeds Phase B.
    # ────────────────────────────────────────────────────────────────────────

    var T_params_buf = alloc[Scalar[dtype]](T_PARAM_SIZE)
    var T_grads_buf = alloc[Scalar[dtype]](T_PARAM_SIZE)
    var T_opt_state_buf = alloc[Scalar[dtype]](
        T_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var T_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE)
    memset(T_params_buf, 0, T_PARAM_SIZE)
    memset(T_grads_buf, 0, T_PARAM_SIZE)
    memset(T_opt_state_buf, 0, T_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(T_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    _xavier_init_layer(T_params_buf, AUG_DIM, HIDDEN, UInt64(7))
    var T_params = LayoutTensor[
        dtype, Layout.row_major(T_PARAM_SIZE), MutAnyOrigin
    ](T_params_buf)
    var T_grads = LayoutTensor[
        dtype, Layout.row_major(T_PARAM_SIZE), MutAnyOrigin
    ](T_grads_buf)
    var T_opt_state = LayoutTensor[
        dtype,
        Layout.row_major(T_PARAM_SIZE, OPT.STATE_PER_PARAM),
        MutAnyOrigin,
    ](T_opt_state_buf)
    var T_opt_global = LayoutTensor[
        dtype, Layout.row_major(OPT.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](T_opt_global_buf)

    var D_params_buf = alloc[Scalar[dtype]](D_PARAM_SIZE)
    var D_grads_buf = alloc[Scalar[dtype]](D_PARAM_SIZE)
    var D_opt_state_buf = alloc[Scalar[dtype]](
        D_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var D_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE)
    memset(D_params_buf, 0, D_PARAM_SIZE)
    memset(D_grads_buf, 0, D_PARAM_SIZE)
    memset(D_opt_state_buf, 0, D_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(D_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    _xavier_init_layer(D_params_buf, HIDDEN, OBS_DIM, UInt64(8))
    var D_params = LayoutTensor[
        dtype, Layout.row_major(D_PARAM_SIZE), MutAnyOrigin
    ](D_params_buf)
    var D_grads = LayoutTensor[
        dtype, Layout.row_major(D_PARAM_SIZE), MutAnyOrigin
    ](D_grads_buf)
    var D_opt_state = LayoutTensor[
        dtype,
        Layout.row_major(D_PARAM_SIZE, OPT.STATE_PER_PARAM),
        MutAnyOrigin,
    ](D_opt_state_buf)
    var D_opt_global = LayoutTensor[
        dtype, Layout.row_major(OPT.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](D_opt_global_buf)

    var enc_params_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE)
    var enc_grads_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE)
    var enc_opt_state_buf = alloc[Scalar[dtype]](
        ENC_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var enc_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE)
    memset(enc_params_buf, 0, ENC_PARAM_SIZE)
    memset(enc_grads_buf, 0, ENC_PARAM_SIZE)
    memset(enc_opt_state_buf, 0, ENC_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(enc_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var enc_params = LayoutTensor[
        dtype, Layout.row_major(ENC_PARAM_SIZE), MutAnyOrigin
    ](enc_params_buf)
    var enc_grads = LayoutTensor[
        dtype, Layout.row_major(ENC_PARAM_SIZE), MutAnyOrigin
    ](enc_grads_buf)
    var enc_opt_state = LayoutTensor[
        dtype,
        Layout.row_major(ENC_PARAM_SIZE, OPT.STATE_PER_PARAM),
        MutAnyOrigin,
    ](enc_opt_state_buf)
    var enc_opt_global = LayoutTensor[
        dtype, Layout.row_major(OPT.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](enc_opt_global_buf)
    ENC.xavier_init[dtype](enc_params, UInt64(123))

    # Encoder scratch (BATCH for training, BATCH=1 views built by the wrapper).
    var enc_input_buf = alloc[Scalar[dtype]](BATCH * ENC_INPUT_DIM)
    var enc_hpre_buf = alloc[Scalar[dtype]](BATCH * ENC_HIDDEN_DIM)
    var enc_hact_buf = alloc[Scalar[dtype]](BATCH * ENC_HIDDEN_DIM)
    var enc_output_buf = alloc[Scalar[dtype]](BATCH * ENC_OUTPUT_DIM)
    var enc_dz_buf = alloc[Scalar[dtype]](BATCH * ENC_OUTPUT_DIM)
    var enc_input = LayoutTensor[
        dtype, Layout.row_major(BATCH, ENC_INPUT_DIM), MutAnyOrigin
    ](enc_input_buf)
    var enc_hpre = LayoutTensor[
        dtype, Layout.row_major(BATCH, ENC_HIDDEN_DIM), MutAnyOrigin
    ](enc_hpre_buf)
    var enc_hact = LayoutTensor[
        dtype, Layout.row_major(BATCH, ENC_HIDDEN_DIM), MutAnyOrigin
    ](enc_hact_buf)
    var enc_output = LayoutTensor[
        dtype, Layout.row_major(BATCH, ENC_OUTPUT_DIM), MutAnyOrigin
    ](enc_output_buf)
    var enc_dz = LayoutTensor[
        dtype, Layout.row_major(BATCH, ENC_OUTPUT_DIM), MutAnyOrigin
    ](enc_dz_buf)

    # BPTT cache.
    var cache_x_aug_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * AUG_DIM)
    var cache_a_x_aug_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * AUG_DIM)
    var cache_mu_z_next_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * HIDDEN)
    var cache_a_z_next_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * HIDDEN)
    var cache_mu_obs_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * OBS_DIM)
    var d_mu_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM)
    var d_mu_z_next_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var d_z_from_next_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var d_x_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM)

    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN)
    var obs_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1) * OBS_DIM)

    print("\n  --- Phase A: encoder + transition + decoder via K-step BPTT ---")
    print("  epoch | mean_loss | wall_t (s)")
    print("  ------+-----------+------------")
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var step_num: Int = 0
    var t_enc0 = perf_counter_ns()

    for epoch in range(ENC_EPOCHS):
        var lr_scale = SCHED.lr_scale_at(epoch, ENC_EPOCHS)
        var last_loss: Float64 = 0.0
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            for b in range(BATCH):
                _gen_rollout_into[SEQ_LEN](
                    rng,
                    actions_buf,
                    obs_buf,
                    b * SEQ_LEN,
                    b * (SEQ_LEN + 1) * OBS_DIM,
                )

            # t=0 encoder forward (z_0 = enc(0, 0, obs_0)).
            for b in range(BATCH):
                for j in range(HIDDEN):
                    enc_input_buf[b * ENC_INPUT_DIM + j] = Scalar[dtype](0.0)
                enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = Scalar[dtype](0.0)
                for d in range(OBS_DIM):
                    enc_input_buf[
                        b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
                    ] = obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + d]
            ENC.forward[BATCH, dtype](
                enc_params, enc_input, enc_hpre, enc_hact, enc_output
            )

            # Forward: roll z forward through K_BPTT transitions.
            for k in range(K_BPTT):
                for b in range(BATCH):
                    if k == 0:
                        for j in range(HIDDEN):
                            cache_x_aug_buf[
                                k * BATCH * AUG_DIM + b * AUG_DIM + j
                            ] = enc_output_buf[b * ENC_OUTPUT_DIM + j]
                    else:
                        for j in range(HIDDEN):
                            cache_x_aug_buf[
                                k * BATCH * AUG_DIM + b * AUG_DIM + j
                            ] = cache_mu_z_next_buf[
                                (k - 1) * BATCH * HIDDEN + b * HIDDEN + j
                            ]
                    cache_x_aug_buf[
                        k * BATCH * AUG_DIM + b * AUG_DIM + HIDDEN
                    ] = actions_buf[b * SEQ_LEN + k]
                _lt_forward[BATCH, AUG_DIM, HIDDEN](
                    T_params_buf,
                    cache_x_aug_buf + k * BATCH * AUG_DIM,
                    cache_a_x_aug_buf + k * BATCH * AUG_DIM,
                    cache_mu_z_next_buf + k * BATCH * HIDDEN,
                )
                _lt_forward[BATCH, HIDDEN, OBS_DIM](
                    D_params_buf,
                    cache_mu_z_next_buf + k * BATCH * HIDDEN,
                    cache_a_z_next_buf + k * BATCH * HIDDEN,
                    cache_mu_obs_buf + k * BATCH * OBS_DIM,
                )

            var sum_sq: Float64 = 0.0
            for k in range(K_BPTT):
                for b in range(BATCH):
                    for d in range(OBS_DIM):
                        var diff = Float64(
                            cache_mu_obs_buf[
                                k * BATCH * OBS_DIM + b * OBS_DIM + d
                            ]
                        ) - Float64(
                            obs_buf[
                                b * (SEQ_LEN + 1) * OBS_DIM
                                + (k + 1) * OBS_DIM
                                + d
                            ]
                        )
                        sum_sq += diff * diff
            last_loss = 0.5 * sum_sq / Float64(BATCH * K_BPTT)

            memset(T_grads_buf, 0, T_PARAM_SIZE)
            memset(D_grads_buf, 0, D_PARAM_SIZE)
            memset(d_z_from_next_buf, 0, BATCH * HIDDEN)

            for k_rev in range(K_BPTT):
                var k = K_BPTT - 1 - k_rev
                for b in range(BATCH):
                    for d in range(OBS_DIM):
                        d_mu_obs_buf[b * OBS_DIM + d] = Scalar[dtype](
                            Float64(
                                cache_mu_obs_buf[
                                    k * BATCH * OBS_DIM + b * OBS_DIM + d
                                ]
                            )
                            - Float64(
                                obs_buf[
                                    b * (SEQ_LEN + 1) * OBS_DIM
                                    + (k + 1) * OBS_DIM
                                    + d
                                ]
                            )
                        )
                _lt_backward_accum[BATCH, HIDDEN, OBS_DIM](
                    D_params_buf,
                    cache_a_z_next_buf + k * BATCH * HIDDEN,
                    d_mu_obs_buf,
                    D_grads_buf + D_W_OFFSET,
                    D_grads_buf + D_B_OFFSET,
                    d_mu_z_next_buf,
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        d_mu_z_next_buf[b * HIDDEN + j] = Scalar[dtype](
                            Float64(d_mu_z_next_buf[b * HIDDEN + j])
                            + Float64(d_z_from_next_buf[b * HIDDEN + j])
                        )
                _lt_backward_accum[BATCH, AUG_DIM, HIDDEN](
                    T_params_buf,
                    cache_a_x_aug_buf + k * BATCH * AUG_DIM,
                    d_mu_z_next_buf,
                    T_grads_buf + T_W_OFFSET,
                    T_grads_buf + T_B_OFFSET,
                    d_x_aug_buf,
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        d_z_from_next_buf[b * HIDDEN + j] = d_x_aug_buf[
                            b * AUG_DIM + j
                        ]

            # Encoder backward at k=0.
            for b in range(BATCH):
                for j in range(HIDDEN):
                    enc_dz_buf[b * ENC_OUTPUT_DIM + j] = d_z_from_next_buf[
                        b * HIDDEN + j
                    ]
            ENC.backward[BATCH, dtype](
                enc_params, enc_input, enc_hact, enc_dz, enc_grads
            )

            clip_grad_norm[T_PARAM_SIZE, dtype](T_grads, GRAD_CLIP_NORM)
            clip_grad_norm[D_PARAM_SIZE, dtype](D_grads, GRAD_CLIP_NORM)
            clip_grad_norm[ENC_PARAM_SIZE, dtype](enc_grads, GRAD_CLIP_NORM)
            step_num += 1
            OPT.step[T_PARAM_SIZE, dtype](
                T_params,
                T_grads,
                T_opt_state,
                T_opt_global,
                step_num,
                lr_scale=lr_scale,
            )
            OPT.step[D_PARAM_SIZE, dtype](
                D_params,
                D_grads,
                D_opt_state,
                D_opt_global,
                step_num,
                lr_scale=lr_scale,
            )
            OPT.step[ENC_PARAM_SIZE, dtype](
                enc_params,
                enc_grads,
                enc_opt_state,
                enc_opt_global,
                step_num,
                lr_scale=lr_scale,
            )

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == ENC_EPOCHS - 1:
            var elapsed = Float64(perf_counter_ns() - t_enc0) / 1e9
            print(
                "    ep=",
                epoch,
                "  loss=",
                last_loss,
                "  lr_scale=",
                lr_scale,
                "  wall=",
                elapsed,
                "s",
            )

    var enc_train_t = Float64(perf_counter_ns() - t_enc0) / 1e9
    print("  encoder train wall:", enc_train_t, "s")

    # ────────────────────────────────────────────────────────────────────────
    # PHASE B — Wrap GymPendulumEnv with the trained encoder + run SAC.
    # ────────────────────────────────────────────────────────────────────────

    print("\n  --- Phase B: SAC on encoded latent ---")

    var w_enc_input = alloc[Scalar[dtype]](ENC_INPUT_DIM)
    var w_enc_hpre = alloc[Scalar[dtype]](ENC_HIDDEN_DIM)
    var w_enc_hact = alloc[Scalar[dtype]](ENC_HIDDEN_DIM)
    var w_enc_output = alloc[Scalar[dtype]](ENC_OUTPUT_DIM)
    var w_prev_z = alloc[Scalar[dtype]](HIDDEN)
    var w_prev_action = alloc[Scalar[dtype]](ACTION_DIM)
    memset(w_prev_z, 0, HIDDEN)
    memset(w_prev_action, 0, ACTION_DIM)

    var w_obs_div = alloc[Scalar[dtype]](OBS_DIM)
    w_obs_div[0] = Scalar[dtype](1.0)
    w_obs_div[1] = Scalar[dtype](1.0)
    w_obs_div[2] = Scalar[dtype](PEND_MAX_SPEED)
    var w_act_div = alloc[Scalar[dtype]](ACTION_DIM)
    w_act_div[0] = Scalar[dtype](PEND_MAX_TORQUE)

    var base_env = GymPendulumEnv(render_mode="")
    var base_env_ptr = rebind[
        UnsafePointer[GymPendulumEnv, origin=MutAnyOrigin]
    ](UnsafePointer(to=base_env))
    var wrapped = EncoderWrappedEnv[
        GymPendulumEnv, HIDDEN, ENC_HIDDEN_DIM, ACTION_DIM, OBS_DIM
    ](
        base_env=base_env_ptr,
        enc_params=enc_params_buf,
        enc_input=w_enc_input,
        enc_hpre=w_enc_hpre,
        enc_hact=w_enc_hact,
        enc_output=w_enc_output,
        prev_z=w_prev_z,
        prev_action=w_prev_action,
        obs_divisor=w_obs_div,
        action_divisor=w_act_div,
    )

    var agent = DeepSACAgent[
        obs_dim=HIDDEN,
        action_dim=ACTION_DIM,
        hidden_dim=64,
        buffer_capacity=50000,
        batch_size=64,
        actor_lr=0.0003,
        critic_lr=0.0003,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        alpha=0.1,
        auto_alpha=True,
        alpha_lr=0.0001,
    )

    var t_sac0 = perf_counter_ns()
    var metrics = agent.train(
        wrapped,
        num_steps=SAC_NUM_STEPS,
        max_steps_per_episode=SAC_MAX_STEPS,
        warmup_steps=SAC_WARMUP_STEPS,
        train_every=1,
        verbose=True,
        print_every=SAC_PRINT_EVERY,
        environment_name="Pendulum (MLP-encoded)",
    )
    var sac_train_t = Float64(perf_counter_ns() - t_sac0) / 1e9

    print("\n  === per-episode returns (CSV: ep,return,steps) ===")
    var rewards = metrics.get_rewards()
    var steps = metrics.get_steps()
    for i in range(len(rewards)):
        print("  CSV:", i, ",", rewards[i], ",", steps[i])

    print("\n  === Phase-2 MLP-encoded summary ===")
    print("  Encoder train wall :", enc_train_t, "s")
    print("  SAC train wall     :", sac_train_t, "s")
    print("  Total wall         :", enc_train_t + sac_train_t, "s")
    print("  Final α            :", String(agent.alpha)[byte=:6])
    print("  Last-20 avg        :", metrics.mean_reward_last_n(20))

    # cleanup
    T_params_buf.free()
    T_grads_buf.free()
    T_opt_state_buf.free()
    T_opt_global_buf.free()
    D_params_buf.free()
    D_grads_buf.free()
    D_opt_state_buf.free()
    D_opt_global_buf.free()
    enc_params_buf.free()
    enc_grads_buf.free()
    enc_opt_state_buf.free()
    enc_opt_global_buf.free()
    enc_input_buf.free()
    enc_hpre_buf.free()
    enc_hact_buf.free()
    enc_output_buf.free()
    enc_dz_buf.free()
    cache_x_aug_buf.free()
    cache_a_x_aug_buf.free()
    cache_mu_z_next_buf.free()
    cache_a_z_next_buf.free()
    cache_mu_obs_buf.free()
    d_mu_obs_buf.free()
    d_mu_z_next_buf.free()
    d_z_from_next_buf.free()
    d_x_aug_buf.free()
    actions_buf.free()
    obs_buf.free()
    w_enc_input.free()
    w_enc_hpre.free()
    w_enc_hact.free()
    w_enc_output.free()
    w_prev_z.free()
    w_prev_action.free()
    w_obs_div.free()
    w_act_div.free()
    print("=== Done ===")
