"""Pendulum world model — amortized PC (Tschantz 2023 hybrid PC).

Replaces the iterative SGLD inference with an encoder + few refinement steps:
    z_init = encoder([z_{t-1}, a_{t-1}, s_t])    # learned MLP, feedforward
    z_t    = z_init refined with K=5 SGLD steps  # local correction
    W's    updated via PC's local rule at refined z_t  (no backprop through W)
    encoder W's updated via MSE(encoder_out, refined_z_t.detach())  (standard backprop)

This addresses the two failure modes that vanilla iterative-PC hit on Pendulum:
  - **Lazy fit** is gone: encoder is forced to be a tight forward predictor.
  - **Recurrent inference instability** is gone: with K=5 instead of T_INFER=50,
    latents don't have time to diverge even if W's spectral radius drifts past 1.

PC's distinctives are preserved: local learning rule for the world model W's,
iterative refinement (just shorter), Bayesian framing.

Encoder is a 2-layer MLP, hand-rolled in this test file (test-local). If this
works, promote to a framework primitive.

Run:
    pixi run mojo run -I . tests/nn_pc_v2/test_pendulum_amortized_pc.mojo
"""

from std.math import sqrt, log, cos, sin, tanh, pi
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.optimizer.adamw import AdamW
from mojo_rl.nn.training.scheduler import CosineWarmupSchedule
from mojo_rl.experimental.nn_pc_v2 import (
    PCBlock,
    PCSequential,
    PCTanh,
    PCTrainer,
)


# Pendulum physics
comptime PEND_G: Float64 = 10.0
comptime PEND_L: Float64 = 1.0
comptime PEND_M: Float64 = 1.0
comptime PEND_DT: Float64 = 0.05
comptime PEND_MAX_SPEED: Float64 = 8.0
comptime PEND_MAX_TORQUE: Float64 = 2.0

# World-model architecture (PC)
comptime BATCH = 32
comptime HIDDEN = 64
comptime ACTION_DIM = 1
comptime OBS_DIM = 3
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 20
comptime EVAL_HORIZON = 10
comptime EPOCHS = 100
comptime N_BATCHES_PER_EPOCH = 100
comptime T_REFINE = 10
comptime WARMUP_EPOCHS = 5
comptime LR_MIN_SCALE: Float64 = 0.1                  # K SGLD refinement steps after encoder init
comptime LR_X: Float64 = 0.01
comptime ADAM_LR_PC: Float64 = 0.001
comptime ADAM_LR_ENC: Float64 = 0.001
comptime GRAD_CLIP_NORM: Float64 = 1.0
comptime PC_WEIGHT_DECAY: Float64 = 0.01    # AdamW weight decay — regularizes decoder W to prevent unbounded ω predictions

# Encoder architecture (test-local 2-layer MLP)
comptime ENC_INPUT_DIM = HIDDEN + ACTION_DIM + OBS_DIM   # [prev_z, action, obs]
comptime ENC_HIDDEN_DIM = 64
comptime ENC_OUTPUT_DIM = HIDDEN
comptime ENC_W1_SIZE = ENC_INPUT_DIM * ENC_HIDDEN_DIM
comptime ENC_B1_SIZE = ENC_HIDDEN_DIM
comptime ENC_W2_SIZE = ENC_HIDDEN_DIM * ENC_OUTPUT_DIM
comptime ENC_B2_SIZE = ENC_OUTPUT_DIM
comptime ENC_PARAM_SIZE = ENC_W1_SIZE + ENC_B1_SIZE + ENC_W2_SIZE + ENC_B2_SIZE
comptime ENC_W1_OFFSET = 0
comptime ENC_B1_OFFSET = ENC_W1_SIZE
comptime ENC_W2_OFFSET = ENC_W1_SIZE + ENC_B1_SIZE
comptime ENC_B2_OFFSET = ENC_W1_SIZE + ENC_B1_SIZE + ENC_W2_SIZE

# Eval
comptime N_EVAL_TRAJ = 32

comptime NET = PCSequential[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],
    PCBlock[HIDDEN, OBS_DIM, PCTanh],
]
comptime TRAINER = PCTrainer[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],
    PCBlock[HIDDEN, OBS_DIM, PCTanh],
    dtype=dtype,
]
comptime OPT_PC = Adam[LR=ADAM_LR_PC]
comptime OPT_ENC = Adam[LR=ADAM_LR_ENC]
comptime SCHED = CosineWarmupSchedule[
    WARMUP_EPOCHS=WARMUP_EPOCHS, MIN_SCALE=LR_MIN_SCALE
]


fn _angle_normalize(t: Float64) -> Float64:
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
    var theta_acc = (
        (3.0 * PEND_G) / (2.0 * PEND_L) * sin(theta)
        + (3.0 / (PEND_M * PEND_L * PEND_L)) * u
    )
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
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 2] = Scalar[dtype](theta_dot / PEND_MAX_SPEED)


# ── Encoder MLP: enc_input → tanh(W1·x + b1) → W2·h + b2 ─────────────────────

fn _xavier_init_encoder(
    enc_params: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    seed: UInt64,
):
    """Xavier-uniform init for encoder W1, W2 (zeros for b1, b2)."""
    # W1: fan_in = ENC_INPUT_DIM, fan_out = ENC_HIDDEN_DIM
    var rng1 = PhiloxRandom(seed=seed, offset=UInt64(0))
    var bound1 = sqrt(6.0 / Float64(ENC_INPUT_DIM + ENC_HIDDEN_DIM))
    for i in range(ENC_W1_SIZE):
        var u = Float64(rng1.step_uniform()[0])
        enc_params[ENC_W1_OFFSET + i] = Scalar[dtype]((u * 2.0 - 1.0) * bound1)
    # b1 = 0
    for i in range(ENC_B1_SIZE):
        enc_params[ENC_B1_OFFSET + i] = Scalar[dtype](0.0)
    # W2: fan_in = ENC_HIDDEN_DIM, fan_out = ENC_OUTPUT_DIM
    var rng2 = PhiloxRandom(seed=seed + UInt64(1), offset=UInt64(0))
    var bound2 = sqrt(6.0 / Float64(ENC_HIDDEN_DIM + ENC_OUTPUT_DIM))
    for i in range(ENC_W2_SIZE):
        var u = Float64(rng2.step_uniform()[0])
        enc_params[ENC_W2_OFFSET + i] = Scalar[dtype]((u * 2.0 - 1.0) * bound2)
    # b2 = 0
    for i in range(ENC_B2_SIZE):
        enc_params[ENC_B2_OFFSET + i] = Scalar[dtype](0.0)


def _encoder_forward[
    BATCH_T: Int
](
    enc_params: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    enc_input: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    hidden_pre: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    hidden_act: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    output: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
):
    """Forward: input → tanh(W1 · input + b1) → W2 · h + b2 → output.

    Caches hidden_pre and hidden_act for the backward pass.
    Layouts: enc_input [BATCH_T, ENC_INPUT_DIM], hidden_pre/hidden_act
    [BATCH_T, ENC_HIDDEN_DIM], output [BATCH_T, ENC_OUTPUT_DIM].
    """
    for b in range(BATCH_T):
        # h_pre = W1·input + b1 ;  h_act = tanh(h_pre)
        for j in range(ENC_HIDDEN_DIM):
            var s = Float64(enc_params[ENC_B1_OFFSET + j])
            for i in range(ENC_INPUT_DIM):
                s += Float64(enc_input[b * ENC_INPUT_DIM + i]) * Float64(
                    enc_params[ENC_W1_OFFSET + i * ENC_HIDDEN_DIM + j]
                )
            hidden_pre[b * ENC_HIDDEN_DIM + j] = Scalar[dtype](s)
            hidden_act[b * ENC_HIDDEN_DIM + j] = Scalar[dtype](tanh(s))
        # output = W2·h_act + b2
        for j in range(ENC_OUTPUT_DIM):
            var s = Float64(enc_params[ENC_B2_OFFSET + j])
            for i in range(ENC_HIDDEN_DIM):
                s += Float64(hidden_act[b * ENC_HIDDEN_DIM + i]) * Float64(
                    enc_params[ENC_W2_OFFSET + i * ENC_OUTPUT_DIM + j]
                )
            output[b * ENC_OUTPUT_DIM + j] = Scalar[dtype](s)


def _encoder_backward[
    BATCH_T: Int
](
    enc_params: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    enc_input: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    hidden_act: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    dz_out: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    enc_grads: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
):
    """Backward: given dL/dz (= encoder_out − refined_z), accumulate into enc_grads.

    Standard MLP chain rule. enc_grads is written (not accumulated — caller
    zeroes it before calling).
    """
    for i in range(ENC_PARAM_SIZE):
        enc_grads[i] = Scalar[dtype](0.0)

    # dh_act, dh_pre per-batch buffers (small, use stack-like Lists)
    var dh_act_list = List[Float64](capacity=ENC_HIDDEN_DIM)
    var dh_pre_list = List[Float64](capacity=ENC_HIDDEN_DIM)
    for _ in range(ENC_HIDDEN_DIM):
        dh_act_list.append(0.0)
        dh_pre_list.append(0.0)

    for b in range(BATCH_T):
        # Init dh_act = 0
        for j in range(ENC_HIDDEN_DIM):
            dh_act_list[j] = 0.0

        # dW2[i, j] += h_act[b, i] * dz[b, j];  db2[j] += dz[b, j]
        # dh_act[i] += W2[i, j] * dz[b, j]
        for j in range(ENC_OUTPUT_DIM):
            var dz_bj = Float64(dz_out[b * ENC_OUTPUT_DIM + j])
            enc_grads[ENC_B2_OFFSET + j] = Scalar[dtype](
                Float64(enc_grads[ENC_B2_OFFSET + j]) + dz_bj
            )
            for i in range(ENC_HIDDEN_DIM):
                enc_grads[ENC_W2_OFFSET + i * ENC_OUTPUT_DIM + j] = Scalar[
                    dtype
                ](
                    Float64(
                        enc_grads[ENC_W2_OFFSET + i * ENC_OUTPUT_DIM + j]
                    )
                    + Float64(hidden_act[b * ENC_HIDDEN_DIM + i]) * dz_bj
                )
                dh_act_list[i] += (
                    Float64(
                        enc_params[ENC_W2_OFFSET + i * ENC_OUTPUT_DIM + j]
                    )
                    * dz_bj
                )

        # dh_pre[j] = dh_act[j] * (1 - tanh²(h_pre)) = dh_act[j] * (1 - h_act²)
        for j in range(ENC_HIDDEN_DIM):
            var ha = Float64(hidden_act[b * ENC_HIDDEN_DIM + j])
            dh_pre_list[j] = dh_act_list[j] * (1.0 - ha * ha)

        # dW1[i, j] += input[b, i] * dh_pre[j];  db1[j] += dh_pre[j]
        for j in range(ENC_HIDDEN_DIM):
            enc_grads[ENC_B1_OFFSET + j] = Scalar[dtype](
                Float64(enc_grads[ENC_B1_OFFSET + j]) + dh_pre_list[j]
            )
            for i in range(ENC_INPUT_DIM):
                enc_grads[ENC_W1_OFFSET + i * ENC_HIDDEN_DIM + j] = Scalar[
                    dtype
                ](
                    Float64(
                        enc_grads[ENC_W1_OFFSET + i * ENC_HIDDEN_DIM + j]
                    )
                    + Float64(enc_input[b * ENC_INPUT_DIM + i])
                    * dh_pre_list[j]
                )


def _clip_grad_norm_ptr(
    grads: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    size: Int,
    max_norm: Float64,
):
    """Global L2-norm gradient clipping on a flat pointer."""
    var sum_sq: Float64 = 0
    for i in range(size):
        var g = Float64(grads[i])
        sum_sq += g * g
    var norm = sqrt(sum_sq)
    if norm > max_norm:
        var scale = Scalar[dtype](max_norm / norm)
        for i in range(size):
            grads[i] = grads[i] * scale


def main() raises:
    print("=" * 60)
    print("Pendulum world model — amortized PC")
    print("=" * 60)
    print("  PC arch    : PCBlock[", AUG_DIM, ",", HIDDEN, ",PCTanh] → PCBlock[", HIDDEN, ",", OBS_DIM, ",PCTanh]")
    print("  PC params  :", NET.PARAM_SIZE)
    print("  Enc arch   : MLP[", ENC_INPUT_DIM, "→", ENC_HIDDEN_DIM, "→", ENC_OUTPUT_DIM, "]")
    print("  Enc params :", ENC_PARAM_SIZE)
    print("  BATCH=", BATCH, " SEQ_LEN=", SEQ_LEN, " EPOCHS=", EPOCHS, " T_REFINE=", T_REFINE)
    print("  PC_OPT=Adam(lr=", ADAM_LR_PC, ")  ENC_OPT=Adam(lr=", ADAM_LR_ENC, ")")
    print("  LR schedule: cosine warmup (W=", WARMUP_EPOCHS, ", min=", LR_MIN_SCALE, ")")
    print("  LR_X=", LR_X, "  GRAD_CLIP=", GRAD_CLIP_NORM)

    # ── PC params + Adam state ────────────────────────────────────────────────
    var pc_params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    var pc_grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    var pc_opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT_PC.STATE_PER_PARAM)
    var pc_opt_global_buf = alloc[Scalar[dtype]](OPT_PC.GLOBAL_STATE_SIZE)
    memset(pc_params_buf, 0, NET.PARAM_SIZE)
    memset(pc_grads_buf, 0, NET.PARAM_SIZE)
    memset(pc_opt_state_buf, 0, NET.PARAM_SIZE * OPT_PC.STATE_PER_PARAM)
    memset(pc_opt_global_buf, 0, OPT_PC.GLOBAL_STATE_SIZE)
    var pc_params = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](pc_params_buf)
    var pc_grads = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](pc_grads_buf)
    var pc_opt_state = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE, OPT_PC.STATE_PER_PARAM), MutAnyOrigin
    ](pc_opt_state_buf)
    var pc_opt_global = LayoutTensor[
        dtype, Layout.row_major(OPT_PC.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](pc_opt_global_buf)
    NET.initialize_params[Xavier[], dtype](pc_params)

    # ── Encoder params + Adam state ───────────────────────────────────────────
    var enc_params_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE)
    var enc_grads_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE)
    var enc_opt_state_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM)
    var enc_opt_global_buf = alloc[Scalar[dtype]](OPT_ENC.GLOBAL_STATE_SIZE)
    memset(enc_params_buf, 0, ENC_PARAM_SIZE)
    memset(enc_grads_buf, 0, ENC_PARAM_SIZE)
    memset(enc_opt_state_buf, 0, ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM)
    memset(enc_opt_global_buf, 0, OPT_ENC.GLOBAL_STATE_SIZE)
    _xavier_init_encoder(enc_params_buf, UInt64(123))

    var enc_grads = LayoutTensor[
        dtype, Layout.row_major(ENC_PARAM_SIZE), MutAnyOrigin
    ](enc_grads_buf)
    var enc_params = LayoutTensor[
        dtype, Layout.row_major(ENC_PARAM_SIZE), MutAnyOrigin
    ](enc_params_buf)
    var enc_opt_state = LayoutTensor[
        dtype, Layout.row_major(ENC_PARAM_SIZE, OPT_ENC.STATE_PER_PARAM), MutAnyOrigin
    ](enc_opt_state_buf)
    var enc_opt_global = LayoutTensor[
        dtype, Layout.row_major(OPT_ENC.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](enc_opt_global_buf)

    # ── PC scratch ────────────────────────────────────────────────────────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM)
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM)
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM)
    memset(lat_buf, 0, BATCH * NET.LATENT_DIM)

    var latents = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](lat_buf)
    var mu_eps_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_OUT_DIM), MutAnyOrigin
    ](mu_eps_buf_raw)
    var a_below_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](a_below_buf_raw)
    var z_below_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](z_below_buf_raw)
    var dx_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](dx_buf_raw)

    var x_in_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM)
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM)
    memset(x_in_buf, 0, BATCH * AUG_DIM)
    memset(y_tgt_buf, 0, BATCH * OBS_DIM)
    var x_in = LayoutTensor[
        dtype, Layout.row_major(BATCH, AUG_DIM), MutAnyOrigin
    ](x_in_buf)
    var y_target = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
    ](y_tgt_buf)

    # ── Encoder scratch ──────────────────────────────────────────────────────
    var enc_input_buf = alloc[Scalar[dtype]](BATCH * ENC_INPUT_DIM)
    var enc_hpre_buf = alloc[Scalar[dtype]](BATCH * ENC_HIDDEN_DIM)
    var enc_hact_buf = alloc[Scalar[dtype]](BATCH * ENC_HIDDEN_DIM)
    var enc_output_buf = alloc[Scalar[dtype]](BATCH * ENC_OUTPUT_DIM)
    var enc_dz_buf = alloc[Scalar[dtype]](BATCH * ENC_OUTPUT_DIM)

    # ── Per-rollout actions/states scratch ───────────────────────────────────
    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN)
    var obs_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1) * OBS_DIM)

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_step_loss | wall_t (s)")
    print("  ------+----------------+------------")
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var pc_step_num: Int = 0
    var enc_step_num: Int = 0
    var prev_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)  # carries z_{t-1} between time steps
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        # Cosine warmup schedule: linear 0→1 over WARMUP_EPOCHS, then cosine
        # decay to LR_MIN_SCALE. Flattens late-epoch updates so Adam can't
        # destabilize the recurrent training.
        var lr_scale = SCHED.lr_scale_at(epoch, EPOCHS)
        var last_loss: Float64 = 0.0
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            # Generate BATCH rollouts on host.
            for b in range(BATCH):
                _gen_rollout_into[SEQ_LEN](
                    rng,
                    actions_buf,
                    obs_buf,
                    b * SEQ_LEN,
                    b * (SEQ_LEN + 1) * OBS_DIM,
                )

            # prev_z = 0 at the start of each rollout.
            memset(prev_z_buf, 0, BATCH * HIDDEN)

            # Per time step t = 0..SEQ_LEN: encode → refine → update W's.
            # At t=0 there's no prior action; use action=0 in the encoder input.
            for t in range(0, SEQ_LEN + 1):
                # Build encoder input: [prev_z, action_{t-1} or 0, obs_t]
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        enc_input_buf[
                            b * ENC_INPUT_DIM + j
                        ] = prev_z_buf[b * HIDDEN + j]
                    var act_val = Scalar[dtype](0.0) if t == 0 else actions_buf[
                        b * SEQ_LEN + (t - 1)
                    ]
                    enc_input_buf[
                        b * ENC_INPUT_DIM + HIDDEN
                    ] = act_val
                    for d in range(OBS_DIM):
                        enc_input_buf[
                            b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
                        ] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]
                    # PC trainer's x_in = [prev_z, action]
                    for j in range(HIDDEN):
                        x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
                    x_in_buf[b * AUG_DIM + HIDDEN] = act_val
                    # PC trainer's y_target = obs_t
                    for d in range(OBS_DIM):
                        y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]

                # Encoder forward → write encoder_out into latents.
                _encoder_forward[BATCH](
                    enc_params_buf,
                    enc_input_buf,
                    enc_hpre_buf,
                    enc_hact_buf,
                    enc_output_buf,
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        lat_buf[b * NET.LATENT_DIM + j] = enc_output_buf[
                            b * ENC_OUTPUT_DIM + j
                        ]

                # T_REFINE SGLD steps + PC W gradients (skip init_latents).
                var result = TRAINER.compute_grads_from_latents[BATCH](
                    pc_params, pc_grads, latents,
                    mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
                    x_in, y_target,
                    T_infer=T_REFINE,
                    lr_x=Scalar[dtype](LR_X),
                )
                _clip_grad_norm_ptr(pc_grads_buf, NET.PARAM_SIZE, GRAD_CLIP_NORM)
                pc_step_num += 1
                OPT_PC.step[NET.PARAM_SIZE, dtype](
                    pc_params, pc_grads, pc_opt_state, pc_opt_global,
                    pc_step_num, lr_scale=lr_scale,
                )
                last_loss = result.output_loss_final

                # Encoder gradient: dz = encoder_out - settled_z (stop-gradient on settled).
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        enc_dz_buf[b * ENC_OUTPUT_DIM + j] = (
                            enc_output_buf[b * ENC_OUTPUT_DIM + j]
                            - lat_buf[b * NET.LATENT_DIM + j]
                        )
                _encoder_backward[BATCH](
                    enc_params_buf,
                    enc_input_buf,
                    enc_hact_buf,
                    enc_dz_buf,
                    enc_grads_buf,
                )
                _clip_grad_norm_ptr(enc_grads_buf, ENC_PARAM_SIZE, GRAD_CLIP_NORM)
                enc_step_num += 1
                OPT_ENC.step[ENC_PARAM_SIZE, dtype](
                    enc_params, enc_grads, enc_opt_state, enc_opt_global,
                    enc_step_num, lr_scale=lr_scale,
                )

                # prev_z = settled z_t (post-refinement).
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        prev_z_buf[b * HIDDEN + j] = lat_buf[
                            b * NET.LATENT_DIM + j
                        ]

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            var elapsed = Float64(perf_counter_ns() - t0) / 1e9
            print(
                "    ep=", epoch,
                "  loss=", last_loss,
                "  lr_scale=", lr_scale,
                "  wall=", elapsed, "s",
            )

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ── Per-block PC param views for eval feedforward ─────────────────────────
    comptime offset_b1 = NET._param_offset[1]()
    var params_b0 = LayoutTensor[
        dtype, Layout.row_major(NET.block_types[0].PARAM_SIZE), MutAnyOrigin
    ](pc_params_buf)
    var params_b1 = LayoutTensor[
        dtype, Layout.row_major(NET.block_types[1].PARAM_SIZE), MutAnyOrigin
    ](pc_params_buf + offset_b1)

    # Eval feedforward scratch
    var z_pred_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var a_z_pred_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM)
    var s_pred_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM)
    var a_s_pred_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var z_pred = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](z_pred_buf)
    var a_z_pred = LayoutTensor[
        dtype, Layout.row_major(BATCH, AUG_DIM), MutAnyOrigin
    ](a_z_pred_buf)
    var s_pred = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
    ](s_pred_buf)
    var a_s_pred = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](a_s_pred_buf)

    # ── Generate held-out trajectories ───────────────────────────────────────
    var eval_rng = PhiloxRandom(seed=UInt64(2026), offset=UInt64(0))
    for b in range(N_EVAL_TRAJ):
        _gen_rollout_into[SEQ_LEN](
            eval_rng,
            actions_buf,
            obs_buf,
            b * SEQ_LEN,
            b * (SEQ_LEN + 1) * OBS_DIM,
        )

    # ── Eval mode 1: 1-step teacher-forced prediction ────────────────────────
    # At each step, encode/filter z_t against actual s_t, predict s_{t+1} via
    # feedforward block_0+block_1 (PC recurrence). Compare to actual s_{t+1}.
    print("\n  === Mode 1: 1-step teacher-forced prediction ===")
    print("  Per-dim cos/sin/ω: model_total | persist_total | model[per-dim] | persist[per-dim]")

    memset(prev_z_buf, 0, BATCH * HIDDEN)

    # Initial filter: encode + refine at t=0 (bootstrap).
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            enc_input_buf[b * ENC_INPUT_DIM + j] = prev_z_buf[b * HIDDEN + j]
        enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = Scalar[dtype](0.0)
        for d in range(OBS_DIM):
            enc_input_buf[
                b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
            ] = obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + 0 + d]
        for j in range(HIDDEN):
            x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
        x_in_buf[b * AUG_DIM + HIDDEN] = Scalar[dtype](0.0)
        for d in range(OBS_DIM):
            y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                b * (SEQ_LEN + 1) * OBS_DIM + 0 + d
            ]
    _encoder_forward[BATCH](
        enc_params_buf, enc_input_buf, enc_hpre_buf, enc_hact_buf, enc_output_buf
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            lat_buf[b * NET.LATENT_DIM + j] = enc_output_buf[b * ENC_OUTPUT_DIM + j]
    _ = TRAINER.compute_grads_from_latents[BATCH](
        pc_params, pc_grads, latents,
        mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
        x_in, y_target,
        T_infer=T_REFINE,
        lr_x=Scalar[dtype](LR_X),
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            prev_z_buf[b * HIDDEN + j] = lat_buf[b * NET.LATENT_DIM + j]

    var mse_1step_total: Float64 = 0
    var mse_persist_total: Float64 = 0

    for t in range(1, SEQ_LEN + 1):
        # Build x_in = [prev_z, action_{t-1}]
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
            x_in_buf[b * AUG_DIM + HIDDEN] = actions_buf[
                b * SEQ_LEN + (t - 1)
            ]

        # 1-step prediction via PC feedforward (no settle): z_pred = block_0(x_in); s_pred = block_1(z_pred)
        NET.block_types[0].predict[BATCH, dtype](
            x_in, params_b0, z_pred, a_z_pred
        )
        NET.block_types[1].predict[BATCH, dtype](
            z_pred, params_b1, s_pred, a_s_pred
        )

        # MSE
        var step_mse_0: Float64 = 0
        var step_mse_1: Float64 = 0
        var step_mse_2: Float64 = 0
        var step_persist_0: Float64 = 0
        var step_persist_1: Float64 = 0
        var step_persist_2: Float64 = 0
        for b in range(N_EVAL_TRAJ):
            var s_true_0 = Float64(obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + 0])
            var s_true_1 = Float64(obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + 1])
            var s_true_2 = Float64(obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + 2])
            var s_prev_0 = Float64(obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + (t - 1) * OBS_DIM + 0])
            var s_prev_1 = Float64(obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + (t - 1) * OBS_DIM + 1])
            var s_prev_2 = Float64(obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + (t - 1) * OBS_DIM + 2])
            var d0 = Float64(s_pred_buf[b * OBS_DIM + 0]) - s_true_0
            var d1 = Float64(s_pred_buf[b * OBS_DIM + 1]) - s_true_1
            var d2 = Float64(s_pred_buf[b * OBS_DIM + 2]) - s_true_2
            step_mse_0 += d0 * d0
            step_mse_1 += d1 * d1
            step_mse_2 += d2 * d2
            var p0 = s_prev_0 - s_true_0
            var p1 = s_prev_1 - s_true_1
            var p2 = s_prev_2 - s_true_2
            step_persist_0 += p0 * p0
            step_persist_1 += p1 * p1
            step_persist_2 += p2 * p2
        step_mse_0 /= Float64(N_EVAL_TRAJ)
        step_mse_1 /= Float64(N_EVAL_TRAJ)
        step_mse_2 /= Float64(N_EVAL_TRAJ)
        step_persist_0 /= Float64(N_EVAL_TRAJ)
        step_persist_1 /= Float64(N_EVAL_TRAJ)
        step_persist_2 /= Float64(N_EVAL_TRAJ)
        var mse_step = step_mse_0 + step_mse_1 + step_mse_2
        var persist_step = step_persist_0 + step_persist_1 + step_persist_2
        mse_1step_total += mse_step
        mse_persist_total += persist_step
        # Untruncated print — Mojo renders very small Float64 in scientific
        # notation (e.g. "1.12e-04"); truncating to fixed bytes can hide the
        # exponent suffix and badly mislead diagnosis.
        print(
            "    t=", t,
            " | total=", mse_step,
            " | persist=", persist_step,
            " | per-dim=[", step_mse_0, ", ", step_mse_1, ", ", step_mse_2, "]",
            " | persist=[", step_persist_0, ", ", step_persist_1, ", ", step_persist_2, "]",
        )

        # Filter z_t: encode against actual s_t, refine, save.
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                enc_input_buf[b * ENC_INPUT_DIM + j] = prev_z_buf[b * HIDDEN + j]
            enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = actions_buf[
                b * SEQ_LEN + (t - 1)
            ]
            for d in range(OBS_DIM):
                enc_input_buf[
                    b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
                ] = obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d]
            for d in range(OBS_DIM):
                y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                    b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                ]
        _encoder_forward[BATCH](
            enc_params_buf, enc_input_buf, enc_hpre_buf, enc_hact_buf, enc_output_buf
        )
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                lat_buf[b * NET.LATENT_DIM + j] = enc_output_buf[
                    b * ENC_OUTPUT_DIM + j
                ]
        _ = TRAINER.compute_grads_from_latents[BATCH](
            pc_params, pc_grads, latents,
            mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
            x_in, y_target,
            T_infer=T_REFINE,
            lr_x=Scalar[dtype](LR_X),
        )
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                prev_z_buf[b * HIDDEN + j] = lat_buf[b * NET.LATENT_DIM + j]

    var avg_mse_1step = mse_1step_total / Float64(SEQ_LEN)
    var avg_mse_persist = mse_persist_total / Float64(SEQ_LEN)
    print("\n  avg 1-step MSE :", avg_mse_1step)
    print("  avg persist MSE:", avg_mse_persist)
    print("  ratio (model / persist):", avg_mse_1step / avg_mse_persist if avg_mse_persist > 0 else 1.0)

    # ── Eval mode 2: open-loop multi-step prediction ─────────────────────────
    print("\n  === Mode 2: open-loop multi-step prediction ===")
    print("  step | mse_total (open-loop) | mse_persist | ratio")
    print("  -----+-----------------------+-------------+------")

    memset(prev_z_buf, 0, BATCH * HIDDEN)

    # Initial filter at t=0
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            enc_input_buf[b * ENC_INPUT_DIM + j] = prev_z_buf[b * HIDDEN + j]
        enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = Scalar[dtype](0.0)
        for d in range(OBS_DIM):
            enc_input_buf[
                b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
            ] = obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + 0 + d]
        for j in range(HIDDEN):
            x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
        x_in_buf[b * AUG_DIM + HIDDEN] = Scalar[dtype](0.0)
        for d in range(OBS_DIM):
            y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                b * (SEQ_LEN + 1) * OBS_DIM + 0 + d
            ]
    _encoder_forward[BATCH](
        enc_params_buf, enc_input_buf, enc_hpre_buf, enc_hact_buf, enc_output_buf
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            lat_buf[b * NET.LATENT_DIM + j] = enc_output_buf[b * ENC_OUTPUT_DIM + j]
    _ = TRAINER.compute_grads_from_latents[BATCH](
        pc_params, pc_grads, latents,
        mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
        x_in, y_target,
        T_infer=T_REFINE,
        lr_x=Scalar[dtype](LR_X),
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            prev_z_buf[b * HIDDEN + j] = lat_buf[b * NET.LATENT_DIM + j]

    var mse_openloop_total: Float64 = 0
    var mse_openloop_persist: Float64 = 0

    for t in range(1, EVAL_HORIZON + 1):
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
            x_in_buf[b * AUG_DIM + HIDDEN] = actions_buf[
                b * SEQ_LEN + (t - 1)
            ]
        NET.block_types[0].predict[BATCH, dtype](
            x_in, params_b0, z_pred, a_z_pred
        )
        NET.block_types[1].predict[BATCH, dtype](
            z_pred, params_b1, s_pred, a_s_pred
        )

        var step_mse: Float64 = 0
        var step_persist: Float64 = 0
        for b in range(N_EVAL_TRAJ):
            for d in range(OBS_DIM):
                var s_true = Float64(obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d])
                var s_prev = Float64(obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + (t - 1) * OBS_DIM + d])
                var d_m = Float64(s_pred_buf[b * OBS_DIM + d]) - s_true
                step_mse += d_m * d_m
                var d_p = s_prev - s_true
                step_persist += d_p * d_p
        step_mse /= Float64(N_EVAL_TRAJ)
        step_persist /= Float64(N_EVAL_TRAJ)
        mse_openloop_total += step_mse
        mse_openloop_persist += step_persist
        print(
            "    ", t,
            "  ", String(step_mse)[byte=:9],
            "  ", String(step_persist)[byte=:9],
            "  ", String(step_mse / step_persist if step_persist > 0 else 1.0)[byte=:6],
        )

        # Open-loop: prev_z = z_pred (predicted, NOT filtered).
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                prev_z_buf[b * HIDDEN + j] = z_pred_buf[b * HIDDEN + j]

    var avg_mse_openloop = mse_openloop_total / Float64(EVAL_HORIZON)
    var avg_mse_openloop_persist = mse_openloop_persist / Float64(EVAL_HORIZON)
    print("\n  avg open-loop MSE :", avg_mse_openloop)
    print("  avg open-loop persist:", avg_mse_openloop_persist)
    print("  ratio (model / persist):", avg_mse_openloop / avg_mse_openloop_persist if avg_mse_openloop_persist > 0 else 1.0)

    # ── Pass criteria ─────────────────────────────────────────────────────────
    print("\n  === Summary ===")
    var ratio_1step = avg_mse_1step / avg_mse_persist if avg_mse_persist > 0 else 1.0
    var ratio_openloop = avg_mse_openloop / avg_mse_openloop_persist if avg_mse_openloop_persist > 0 else 1.0
    print("  1-step MSE ratio  :", ratio_1step, "  (want < 0.5)")
    print("  open-loop MSE ratio:", ratio_openloop, "  (want < 1.0)")

    var pass_1step = ratio_1step < 0.5
    var pass_openloop = ratio_openloop < 1.0

    if pass_1step and pass_openloop:
        print("\n  [PASS] amortized PC: 1-step MSE < 0.5×persistence, open-loop beats persistence")
    else:
        if not pass_1step:
            print("\n  [FAIL] 1-step MSE ratio ≥ 0.5")
        if not pass_openloop:
            print("\n  [FAIL] open-loop MSE ratio ≥ 1.0 — model worse than persistence")
        raise Error("amortized PC test failed")

    # cleanup
    pc_params_buf.free()
    pc_grads_buf.free()
    pc_opt_state_buf.free()
    pc_opt_global_buf.free()
    enc_params_buf.free()
    enc_grads_buf.free()
    enc_opt_state_buf.free()
    enc_opt_global_buf.free()
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
    x_in_buf.free()
    y_tgt_buf.free()
    enc_input_buf.free()
    enc_hpre_buf.free()
    enc_hact_buf.free()
    enc_output_buf.free()
    enc_dz_buf.free()
    actions_buf.free()
    obs_buf.free()
    prev_z_buf.free()
    z_pred_buf.free()
    a_z_pred_buf.free()
    s_pred_buf.free()
    a_s_pred_buf.free()
    print("=== Done ===")
