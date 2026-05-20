"""Pendulum SAC — Phase-2b PCN-init fine-tuned.

1. Train PCN encoder via Exp-3 procedure (PC weight rule, no SGLD).
2. Construct SAC with EncoderPrefixSACConfig (custom 4-layer architecture
   with a LinearTanh -> Linear prefix matching the PCEncoder shape).
3. Inject the trained encoder's W1[obs_slice], b1, W2, b2 into the
   first 2 layers of SAC's actor and both critics. The prefix is
   trainable — SAC gradients flow back through it during training.

Paired with `test_pendulum_sac_random_finetune.mojo` (same architecture,
random Xavier init, no encoder pretraining) as the architectural control.
Together they isolate "does PCN pretraining help SAC over plain random
init under matched architecture".

Same SAC hyperparameters as the Phase-2 baseline / PCN-encoded /
MLP-encoded tests (hidden=64, twin Q, auto-α, action_scale=2.0).

See `docs/PCN_MBRL_DESIGN.md` Phase-2b for context.

Run:
    pixi run mojo run -I . tests/pcn/test_pendulum_sac_pcn_finetune.mojo
"""

from std.math import sqrt, log, cos, sin, tanh, pi
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.scheduler import CosineWarmupSchedule
from mojo_rl.experimental.pcn import (
    EncoderPrefixSACConfig,
    PCBlock,
    PCEncoder,
    PCSequential,
    PCTanh,
    clip_grad_norm,
    inject_encoder_into_actor,
    inject_encoder_into_critic,
)
from mojo_rl.envs import PendulumEnv
from mojo_rl.deep_agents.core.agents import GenericOffPolicyAgent
from mojo_rl.deep_agents.core.training.offpolicy_train import (
    run_offpolicy_continuous_train,
)


# Pendulum physics — Gymnasium defaults, matches existing PCN tests.
comptime PEND_G: Float64 = 10.0
comptime PEND_L: Float64 = 1.0
comptime PEND_M: Float64 = 1.0
comptime PEND_DT: Float64 = 0.05
comptime PEND_MAX_SPEED: Float64 = 8.0
comptime PEND_MAX_TORQUE: Float64 = 2.0

# Encoder/world-model architecture — matches Phase-2 PCN-encoded test.
comptime BATCH = 32
comptime HIDDEN = 64
comptime ACTION_DIM = 1
comptime OBS_DIM = 3
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 20
comptime ENC_EPOCHS = 100
comptime N_BATCHES_PER_EPOCH = 100
comptime WARMUP_EPOCHS = 5
comptime LR_MIN_SCALE: Float64 = 0.1
comptime ADAM_LR_PC: Float64 = 0.001
comptime ADAM_LR_ENC: Float64 = 0.001
comptime GRAD_CLIP_NORM: Float64 = 1.0

comptime ENC_INPUT_DIM = HIDDEN + ACTION_DIM + OBS_DIM
comptime ENC_HIDDEN_DIM = 64
comptime ENC_OUTPUT_DIM = HIDDEN
comptime ENC = PCEncoder[ENC_INPUT_DIM, ENC_HIDDEN_DIM, ENC_OUTPUT_DIM]
comptime ENC_PARAM_SIZE = ENC.PARAM_SIZE

comptime BLOCK0 = PCBlock[AUG_DIM, HIDDEN, PCTanh]
comptime BLOCK1 = PCBlock[HIDDEN, OBS_DIM, PCTanh]
comptime NET = PCSequential[BLOCK0, BLOCK1]
comptime OPT_PC = Adam[LR=ADAM_LR_PC]
comptime OPT_ENC = Adam[LR=ADAM_LR_ENC]
comptime SCHED = CosineWarmupSchedule[
    WARMUP_EPOCHS=WARMUP_EPOCHS, MIN_SCALE=LR_MIN_SCALE
]

# SAC hyperparameters — match Phase-2 baseline.
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
    """Encoder training rollout — random-policy, normalized obs+action."""
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


def main() raises:
    print("=" * 60)
    print("Pendulum SAC — Phase-2b (PCN-init fine-tuned)")
    print("=" * 60)
    print("  Encoder    : PCN (Exp-3: per-step PC weight rule, no SGLD)")
    print("  SAC arch   : EncoderPrefix (LinearTanh -> Linear -> LinearReLU -> heads)")
    print("  Enc epochs :", ENC_EPOCHS, "  SAC eps:", SAC_NUM_STEPS)

    # ────────────────────────────────────────────────────────────────────────
    # PHASE A — Train encoder via Exp-3 procedure.
    # ────────────────────────────────────────────────────────────────────────

    var pc_params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    var pc_grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    var pc_opt_state_buf = alloc[Scalar[dtype]](
        NET.PARAM_SIZE * OPT_PC.STATE_PER_PARAM
    )
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
        dtype,
        Layout.row_major(NET.PARAM_SIZE, OPT_PC.STATE_PER_PARAM),
        MutAnyOrigin,
    ](pc_opt_state_buf)
    var pc_opt_global = LayoutTensor[
        dtype, Layout.row_major(OPT_PC.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](pc_opt_global_buf)
    NET.initialize_params[Xavier[], dtype](pc_params)

    comptime offset_b1 = NET._param_offset[1]()
    var params_b0 = LayoutTensor[
        dtype, Layout.row_major(BLOCK0.PARAM_SIZE), MutAnyOrigin
    ](pc_params_buf)
    var params_b1 = LayoutTensor[
        dtype, Layout.row_major(BLOCK1.PARAM_SIZE), MutAnyOrigin
    ](pc_params_buf + offset_b1)
    var grads_b0 = LayoutTensor[
        dtype, Layout.row_major(BLOCK0.PARAM_SIZE), MutAnyOrigin
    ](pc_grads_buf)
    var grads_b1 = LayoutTensor[
        dtype, Layout.row_major(BLOCK1.PARAM_SIZE), MutAnyOrigin
    ](pc_grads_buf + offset_b1)

    var enc_params_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE)
    var enc_grads_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE)
    var enc_opt_state_buf = alloc[Scalar[dtype]](
        ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM
    )
    var enc_opt_global_buf = alloc[Scalar[dtype]](OPT_ENC.GLOBAL_STATE_SIZE)
    memset(enc_params_buf, 0, ENC_PARAM_SIZE)
    memset(enc_grads_buf, 0, ENC_PARAM_SIZE)
    memset(enc_opt_state_buf, 0, ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM)
    memset(enc_opt_global_buf, 0, OPT_ENC.GLOBAL_STATE_SIZE)
    var enc_params = LayoutTensor[
        dtype, Layout.row_major(ENC_PARAM_SIZE), MutAnyOrigin
    ](enc_params_buf)
    var enc_grads = LayoutTensor[
        dtype, Layout.row_major(ENC_PARAM_SIZE), MutAnyOrigin
    ](enc_grads_buf)
    var enc_opt_state = LayoutTensor[
        dtype,
        Layout.row_major(ENC_PARAM_SIZE, OPT_ENC.STATE_PER_PARAM),
        MutAnyOrigin,
    ](enc_opt_state_buf)
    var enc_opt_global = LayoutTensor[
        dtype, Layout.row_major(OPT_ENC.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](enc_opt_global_buf)
    ENC.xavier_init[dtype](enc_params, UInt64(123))

    var x_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM)
    var a_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM)
    var mu_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var eps_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var z_init_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var a_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var mu_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM)
    var eps_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM)
    var y_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM)
    var pull_back_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var gated_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var x_aug = LayoutTensor[
        dtype, Layout.row_major(BATCH, AUG_DIM), MutAnyOrigin
    ](x_aug_buf)
    var a_aug = LayoutTensor[
        dtype, Layout.row_major(BATCH, AUG_DIM), MutAnyOrigin
    ](a_aug_buf)
    var mu_z = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](mu_z_buf)
    var z_init = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](z_init_buf)
    var a_z = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](a_z_buf)
    var mu_obs = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
    ](mu_obs_buf)
    var y_obs = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
    ](y_obs_buf)
    var eps_z = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](eps_z_buf)
    var eps_obs = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
    ](eps_obs_buf)
    var pull_back_out = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](pull_back_buf)
    var gated = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin
    ](gated_buf)

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

    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN)
    var obs_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1) * OBS_DIM)
    var prev_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)

    print("\n  --- Phase A: encoder training (Exp-3) ---")
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var pc_step_num: Int = 0
    var enc_step_num: Int = 0
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
            memset(prev_z_buf, 0, BATCH * HIDDEN)
            for t in range(0, SEQ_LEN + 1):
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        enc_input_buf[b * ENC_INPUT_DIM + j] = prev_z_buf[
                            b * HIDDEN + j
                        ]
                    var act_val = (
                        Scalar[dtype](0.0) if t
                        == 0 else actions_buf[b * SEQ_LEN + (t - 1)]
                    )
                    enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = act_val
                    for d in range(OBS_DIM):
                        enc_input_buf[
                            b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
                        ] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]
                    for j in range(HIDDEN):
                        x_aug_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
                    x_aug_buf[b * AUG_DIM + HIDDEN] = act_val
                    for d in range(OBS_DIM):
                        y_obs_buf[b * OBS_DIM + d] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]

                ENC.forward[BATCH, dtype](
                    enc_params, enc_input, enc_hpre, enc_hact, enc_output
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        z_init_buf[b * HIDDEN + j] = enc_output_buf[
                            b * ENC_OUTPUT_DIM + j
                        ]

                BLOCK0.predict[BATCH, dtype](x_aug, params_b0, mu_z, a_aug)
                BLOCK0.eps_compute[BATCH, dtype](z_init, mu_z, eps_z)
                BLOCK1.predict[BATCH, dtype](z_init, params_b1, mu_obs, a_z)
                BLOCK1.eps_compute[BATCH, dtype](y_obs, mu_obs, eps_obs)

                var sum_sq: Float64 = 0.0
                for b in range(BATCH):
                    for d in range(OBS_DIM):
                        var e = Float64(eps_obs_buf[b * OBS_DIM + d])
                        sum_sq += e * e
                last_loss = 0.5 * sum_sq / Float64(BATCH)

                memset(pc_grads_buf, 0, NET.PARAM_SIZE)
                BLOCK0.weight_grad[BATCH, dtype](eps_z, a_aug, grads_b0)
                BLOCK1.weight_grad[BATCH, dtype](eps_obs, a_z, grads_b1)

                BLOCK1.pull_back[BATCH, dtype](
                    eps_obs, params_b1, pull_back_out
                )
                BLOCK1.act_derivative_mul[BATCH, dtype](
                    z_init, pull_back_out, gated
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        enc_dz_buf[b * ENC_OUTPUT_DIM + j] = Scalar[dtype](
                            Float64(eps_z_buf[b * HIDDEN + j])
                            - Float64(gated_buf[b * HIDDEN + j])
                        )
                ENC.backward[BATCH, dtype](
                    enc_params, enc_input, enc_hact, enc_dz, enc_grads
                )

                clip_grad_norm[NET.PARAM_SIZE, dtype](pc_grads, GRAD_CLIP_NORM)
                pc_step_num += 1
                OPT_PC.step[NET.PARAM_SIZE, dtype](
                    pc_params,
                    pc_grads,
                    pc_opt_state,
                    pc_opt_global,
                    pc_step_num,
                    lr_scale=lr_scale,
                )
                clip_grad_norm[ENC_PARAM_SIZE, dtype](enc_grads, GRAD_CLIP_NORM)
                enc_step_num += 1
                OPT_ENC.step[ENC_PARAM_SIZE, dtype](
                    enc_params,
                    enc_grads,
                    enc_opt_state,
                    enc_opt_global,
                    enc_step_num,
                    lr_scale=lr_scale,
                )

                for b in range(BATCH):
                    for j in range(HIDDEN):
                        prev_z_buf[b * HIDDEN + j] = z_init_buf[b * HIDDEN + j]

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == ENC_EPOCHS - 1:
            var elapsed = Float64(perf_counter_ns() - t_enc0) / 1e9
            print(
                "    ep=", epoch, "  loss=", last_loss,
                "  lr_scale=", lr_scale, "  wall=", elapsed, "s",
            )

    var enc_train_t = Float64(perf_counter_ns() - t_enc0) / 1e9
    print("  encoder train wall:", enc_train_t, "s")

    # ────────────────────────────────────────────────────────────────────────
    # PHASE B — Construct SAC with EncoderPrefixSACConfig, inject PCN weights,
    # train end-to-end with gradients flowing back through the prefix.
    # ────────────────────────────────────────────────────────────────────────

    print("\n  --- Phase B: SAC w/ EncoderPrefix + PCN warm-start ---")

    var env = PendulumEnv[dtype]()

    comptime AgentType = GenericOffPolicyAgent[
        EncoderPrefixSACConfig[
            OBS=OBS_DIM,
            ACT=ACTION_DIM,
            HIDDEN=HIDDEN,
            CAP=50000,
            BS=64,
            actor_lr=0.0003,
            critic_lr=0.0003,
            action_scale=2.0,
        ]
    ]
    var agent = AgentType(
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        alpha=0.1,
        auto_alpha=True,
        alpha_lr=0.0001,
    )

    # Build cpu_state externally (instead of letting agent.train() build a
    # fresh one internally and discard our injection). Inject encoder weights
    # into THIS cpu_state, then call run_offpolicy_continuous_train directly.
    var cpu_state = AgentType.CPUStateType()

    # Inject encoder weights into actor + both critics (online + target).
    # obs_slice_offset = ENC_IN - OBS = 68 - 3 = 65 (input is [prev_z, action, obs]).
    comptime OBS_SLICE_OFF = ENC_INPUT_DIM - OBS_DIM

    inject_encoder_into_actor[
        OBS=OBS_DIM, HID=HIDDEN, ENC_IN=ENC_INPUT_DIM, dtype=dtype
    ](
        cpu_state.actor.online.params,
        enc_params_buf,
        OBS_SLICE_OFF,
    )

    cpu_state.actor.copy_target_from_online()

    for i in range(2):
        inject_encoder_into_critic[
            OBS=OBS_DIM,
            ACT=ACTION_DIM,
            HID=HIDDEN,
            ENC_IN=ENC_INPUT_DIM,
            dtype=dtype,
        ](
            cpu_state.critics.pairs[i].online.params,
            enc_params_buf,
            OBS_SLICE_OFF,
        )
        cpu_state.critics.pairs[i].copy_target_from_online()

    print("  injected encoder weights into actor + critic1 + critic2 (online+target)")

    var t_sac0 = perf_counter_ns()
    var metrics = run_offpolicy_continuous_train(
        agent,
        cpu_state,
        env,
        num_steps=SAC_NUM_STEPS,
        max_steps_per_episode=SAC_MAX_STEPS,
        warmup_steps=SAC_WARMUP_STEPS,
        train_every=1,
        verbose=True,
        print_every=SAC_PRINT_EVERY,
        environment_name="Pendulum (PCN-init FT)",
    )
    var sac_train_t = Float64(perf_counter_ns() - t_sac0) / 1e9

    print("\n  === per-episode returns (CSV: ep,return,steps) ===")
    var rewards = metrics.get_rewards()
    var steps = metrics.get_steps()
    for i in range(len(rewards)):
        print("  CSV:", i, ",", rewards[i], ",", steps[i])

    print("\n  === Phase-2b summary (PCN-init FT) ===")
    print("  Encoder train wall :", enc_train_t, "s")
    print("  SAC train wall     :", sac_train_t, "s")
    print("  Total wall         :", enc_train_t + sac_train_t, "s")
    print("  Final α            :", String(agent.alpha)[byte=:6])
    print("  Last-20 avg        :", metrics.mean_reward_last_n(20))
    print("=== Done ===")

    pc_params_buf.free()
    pc_grads_buf.free()
    pc_opt_state_buf.free()
    pc_opt_global_buf.free()
    enc_params_buf.free()
    enc_grads_buf.free()
    enc_opt_state_buf.free()
    enc_opt_global_buf.free()
    x_aug_buf.free()
    a_aug_buf.free()
    mu_z_buf.free()
    eps_z_buf.free()
    z_init_buf.free()
    a_z_buf.free()
    mu_obs_buf.free()
    eps_obs_buf.free()
    y_obs_buf.free()
    pull_back_buf.free()
    gated_buf.free()
    enc_input_buf.free()
    enc_hpre_buf.free()
    enc_hact_buf.free()
    enc_output_buf.free()
    enc_dz_buf.free()
    actions_buf.free()
    obs_buf.free()
    prev_z_buf.free()
