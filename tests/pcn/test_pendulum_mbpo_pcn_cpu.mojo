"""PCN-MBPO CPU on Pendulum — Phase-3 main experiment.

Glues together:
- `DeepSACAgent` for the SAC side (actor + twin critics + auto-α).
- `PCDynamicsEnsemble` (NUM_ENSEMBLE=3) for the world model — trained
  with the Phase-1-baseline procedure (SGLD inference + PC weight rule).
- A custom MBPO training loop: warmup → per-step env+sac+(periodic
  dynamics-train + synth rollouts) → eval per epoch.

Targets a fair comparison with `test_mbpo_pendulum_cpu.mojo` (vanilla
MBPO + 3-net Swish ensemble) and `test_pendulum_sac_baseline.mojo`
(SAC raw obs). Same SAC hyperparameters across all three.

Predicts (delta_obs, reward) to keep dynamics targets in a small range
that PCN's tanh-bounded layers can represent comfortably. Inputs to the
dynamics are also normalized (theta_dot/8, torque/2) to match the scale
the existing PCN Pendulum tests use.

Run:
    pixi run mojo run -I . tests/pcn/test_pendulum_mbpo_pcn_cpu.mojo
"""

from std.math import sqrt, log
from std.memory import alloc, memset
from std.random import random_float64, seed
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.experimental.pcn import (
    PCDynamics,
    PCDynamicsEnsemble,
)
from mojo_rl.envs import PendulumEnv
from mojo_rl.deep_agents.core.agents import DeepSACAgent


# Pendulum constants.
comptime PEND_MAX_SPEED: Float64 = 8.0
comptime PEND_MAX_TORQUE: Float64 = 2.0

# Sizes — match the vanilla MBPO CPU control test.
comptime OBS_DIM = 3
comptime ACTION_DIM = 1
comptime SAC_HIDDEN = 64
comptime SAC_BATCH = 64
comptime SAC_BUFFER_CAP = 50_000

# Dynamics ensemble.
comptime DYN_HIDDEN = 200
comptime NUM_ENSEMBLE = 3
comptime NUM_ELITES = 2

# PCN training.
comptime DYN_BATCH = 64
comptime T_INFER = 10
comptime LR_X: Float64 = 0.01
comptime DYN_LR: Float64 = 0.001
comptime DYN_GRAD_CLIP: Float64 = 1.0

# MBPO loop.
comptime NUM_EPOCHS = 5
comptime STEPS_PER_EPOCH = 1000
comptime MAX_STEPS_PER_EPISODE = 200
comptime WARMUP_STEPS = 1000
comptime MODEL_TRAIN_FREQ = 250
comptime DYN_TRAIN_BATCHES = 30   # number of minibatches per dynamics retrain
comptime NUM_ROLLOUTS_PER_RETRAIN = 100  # synth (s, a, s', r) per dynamics retrain
comptime ROLLOUT_LENGTH = 1
comptime SAC_UPDATES_PER_STEP = 10
comptime EVAL_EPISODES = 5

# Type aliases.
comptime DYN = PCDynamics[OBS_DIM, ACTION_DIM, DYN_HIDDEN, dtype]
comptime ENS = PCDynamicsEnsemble[
    OBS_DIM, ACTION_DIM, DYN_HIDDEN, NUM_ENSEMBLE, NUM_ELITES, dtype
]
comptime DYN_OPT = Adam[LR=DYN_LR]


# =============================================================================
# Manual circular real buffer (real obs only — used for dynamics training).
# DeepSACAgent has its own internal mixed buffer; this one is independent
# so the dynamics never sees synth transitions.
# =============================================================================


fn real_buf_add(
    buf_obs: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    buf_action: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    buf_next: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    buf_reward: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    mut size: Int,
    mut write_idx: Int,
    cap: Int,
    obs: List[Float64],
    action: List[Float64],
    next_obs: List[Float64],
    reward: Float64,
):
    var idx = write_idx
    for d in range(OBS_DIM):
        buf_obs[idx * OBS_DIM + d] = Scalar[dtype](obs[d])
        buf_next[idx * OBS_DIM + d] = Scalar[dtype](next_obs[d])
    for d in range(ACTION_DIM):
        buf_action[idx * ACTION_DIM + d] = Scalar[dtype](action[d])
    buf_reward[idx] = Scalar[dtype](reward)
    write_idx = (write_idx + 1) % cap
    if size < cap:
        size += 1


# =============================================================================
# Build (s, a) and target (delta_obs, reward) minibatches from real buffer.
# Inputs to dynamics are NORMALIZED (matches encoder training scales used in
# the existing PCN Pendulum tests):
#   s' = [s[0], s[1], s[2]/8.0]      (theta_dot/MAX_SPEED)
#   a' = a / 2.0                     (torque/MAX_TORQUE)
# Targets are also normalized so PCTanh-bounded outputs cover the range.
#   target = [(next_obs - obs)/scale, reward/REWARD_SCALE]
# Caller un-normalizes during rollout.
# =============================================================================


comptime REWARD_SCALE: Float64 = 10.0  # Pendulum reward is in [-16ish, 0]


fn obs_normalize(obs: List[Float64]) -> List[Float64]:
    var n = List[Float64](capacity=OBS_DIM)
    n.append(obs[0])                       # cos θ ∈ [-1, 1]
    n.append(obs[1])                       # sin θ ∈ [-1, 1]
    n.append(obs[2] / PEND_MAX_SPEED)      # θ_dot/8 ∈ [-1, 1]
    return n^


fn action_normalize(action: List[Float64]) -> List[Float64]:
    var n = List[Float64](capacity=ACTION_DIM)
    n.append(action[0] / PEND_MAX_TORQUE)  # torque/2 ∈ [-1, 1]
    return n^


def build_dyn_batch[
    BATCH: Int
](
    mut rng: PhiloxRandom,
    buf_obs: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    buf_action: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    buf_next: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    buf_reward: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    buf_size: Int,
    s_a_buf: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
    target_buf: UnsafePointer[Scalar[dtype], origin=MutAnyOrigin],
):
    """Sample BATCH random transitions and lay them out for `compute_grads_batch`.

    s_a   ← [obs_norm | action_norm]
    target ← [delta_obs_norm | reward / REWARD_SCALE]
    """
    for b in range(BATCH):
        var u = Float64(rng.step_uniform()[0])
        var idx = Int(u * Float64(buf_size)) % buf_size
        # obs_norm + action_norm into s_a row.
        s_a_buf[b * DYN.AUG_DIM + 0] = buf_obs[idx * OBS_DIM + 0]
        s_a_buf[b * DYN.AUG_DIM + 1] = buf_obs[idx * OBS_DIM + 1]
        s_a_buf[b * DYN.AUG_DIM + 2] = Scalar[dtype](
            Float64(buf_obs[idx * OBS_DIM + 2]) / PEND_MAX_SPEED
        )
        s_a_buf[b * DYN.AUG_DIM + OBS_DIM + 0] = Scalar[dtype](
            Float64(buf_action[idx * ACTION_DIM + 0]) / PEND_MAX_TORQUE
        )
        # target = (delta_obs_norm, reward/scale).
        var d0 = Float64(buf_next[idx * OBS_DIM + 0]) - Float64(
            buf_obs[idx * OBS_DIM + 0]
        )
        var d1 = Float64(buf_next[idx * OBS_DIM + 1]) - Float64(
            buf_obs[idx * OBS_DIM + 1]
        )
        var d2 = (
            Float64(buf_next[idx * OBS_DIM + 2])
            - Float64(buf_obs[idx * OBS_DIM + 2])
        ) / PEND_MAX_SPEED
        target_buf[b * DYN.READOUT + 0] = Scalar[dtype](d0)
        target_buf[b * DYN.READOUT + 1] = Scalar[dtype](d1)
        target_buf[b * DYN.READOUT + 2] = Scalar[dtype](d2)
        target_buf[b * DYN.READOUT + 3] = Scalar[dtype](
            Float64(buf_reward[idx]) / REWARD_SCALE
        )


def main() raises:
    seed(42)
    print("=" * 70)
    print("PCN-MBPO CPU on Pendulum — Phase-3 main")
    print("=" * 70)
    print("  SAC arch       : hidden=", SAC_HIDDEN, " batch=", SAC_BATCH)
    print("  PCN dynamics   :", NUM_ENSEMBLE, " nets,", NUM_ELITES, " elites")
    print("                   2-layer PCBlock chain, hidden=", DYN_HIDDEN)
    print("                   T_infer=", T_INFER, " lr_x=", LR_X)
    print("  Buffers (real) : SAC=", SAC_BUFFER_CAP, " dyn=", SAC_BUFFER_CAP)
    print("  Training       :", NUM_EPOCHS, " epochs ×", STEPS_PER_EPOCH, " steps")
    print("  Warmup         :", WARMUP_STEPS, " env steps")
    print("  Model freq     : every", MODEL_TRAIN_FREQ, " steps;",
          DYN_TRAIN_BATCHES, " minibatches per retrain")
    print("  Rollouts       :", NUM_ROLLOUTS_PER_RETRAIN, " synth tuples per retrain,",
          " length=", ROLLOUT_LENGTH)
    print("  SAC updates    :", SAC_UPDATES_PER_STEP, " per env step")
    print()

    var env = PendulumEnv[DType.float64]()

    # SAC agent — same config as `test_pendulum_sac_baseline.mojo`.
    var agent = DeepSACAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=SAC_HIDDEN,
        buffer_capacity=SAC_BUFFER_CAP,
        batch_size=SAC_BATCH,
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

    # ── PCN ensemble buffers ─────────────────────────────────────────────────
    var dyn_params_buf = alloc[Scalar[dtype]](ENS.TOTAL_PARAM_SIZE)
    var dyn_grads_buf = alloc[Scalar[dtype]](ENS.TOTAL_PARAM_SIZE)
    var dyn_opt_state_buf = alloc[Scalar[dtype]](
        ENS.TOTAL_PARAM_SIZE * DYN_OPT.STATE_PER_PARAM
    )
    var dyn_opt_global_buf = alloc[Scalar[dtype]](
        NUM_ENSEMBLE * DYN_OPT.GLOBAL_STATE_SIZE
    )
    memset(dyn_params_buf, 0, ENS.TOTAL_PARAM_SIZE)
    memset(dyn_grads_buf, 0, ENS.TOTAL_PARAM_SIZE)
    memset(
        dyn_opt_state_buf, 0,
        ENS.TOTAL_PARAM_SIZE * DYN_OPT.STATE_PER_PARAM,
    )
    memset(
        dyn_opt_global_buf, 0, NUM_ENSEMBLE * DYN_OPT.GLOBAL_STATE_SIZE
    )
    ENS.init_all(dyn_params_buf, base_seed=UInt64(7))

    # SGLD scratch (DYN_BATCH).
    var lat_buf = alloc[Scalar[dtype]](DYN_BATCH * DYN.SCRATCH_LAT)
    var mu_eps_buf_raw = alloc[Scalar[dtype]](DYN_BATCH * DYN.SCRATCH_OUT)
    var a_below_buf_raw = alloc[Scalar[dtype]](DYN_BATCH * DYN.SCRATCH_IN)
    var z_below_buf_raw = alloc[Scalar[dtype]](DYN_BATCH * DYN.SCRATCH_IN)
    var dx_buf_raw = alloc[Scalar[dtype]](DYN_BATCH * DYN.SCRATCH_LAT)
    memset(lat_buf, 0, DYN_BATCH * DYN.SCRATCH_LAT)
    memset(mu_eps_buf_raw, 0, DYN_BATCH * DYN.SCRATCH_OUT)
    memset(a_below_buf_raw, 0, DYN_BATCH * DYN.SCRATCH_IN)
    memset(z_below_buf_raw, 0, DYN_BATCH * DYN.SCRATCH_IN)
    memset(dx_buf_raw, 0, DYN_BATCH * DYN.SCRATCH_LAT)
    var latents = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.SCRATCH_LAT), MutAnyOrigin
    ](lat_buf)
    var mu_eps = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.SCRATCH_OUT), MutAnyOrigin
    ](mu_eps_buf_raw)
    var a_below = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.SCRATCH_IN), MutAnyOrigin
    ](a_below_buf_raw)
    var z_below = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.SCRATCH_IN), MutAnyOrigin
    ](z_below_buf_raw)
    var dx = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.SCRATCH_LAT), MutAnyOrigin
    ](dx_buf_raw)

    # Dynamics input/target batch buffers.
    var s_a_buf_t = alloc[Scalar[dtype]](DYN_BATCH * DYN.AUG_DIM)
    var target_buf_t = alloc[Scalar[dtype]](DYN_BATCH * DYN.READOUT)
    var s_a_t = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.AUG_DIM), MutAnyOrigin
    ](s_a_buf_t)
    var target_t = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.READOUT), MutAnyOrigin
    ](target_buf_t)

    # Predict scratch (BATCH=1 — synth rollouts are one-at-a-time).
    var p_a_aug_buf = alloc[Scalar[dtype]](1 * DYN.AUG_DIM)
    var p_z_buf = alloc[Scalar[dtype]](1 * DYN.HIDDEN_DIM)
    var p_a_z_buf = alloc[Scalar[dtype]](1 * DYN.HIDDEN_DIM)
    var p_out_buf = alloc[Scalar[dtype]](1 * DYN.READOUT)
    var p_s_a_buf = alloc[Scalar[dtype]](1 * DYN.AUG_DIM)
    var p_a_aug = LayoutTensor[
        dtype, Layout.row_major(1, DYN.AUG_DIM), MutAnyOrigin
    ](p_a_aug_buf)
    var p_z = LayoutTensor[
        dtype, Layout.row_major(1, DYN.HIDDEN_DIM), MutAnyOrigin
    ](p_z_buf)
    var p_a_z = LayoutTensor[
        dtype, Layout.row_major(1, DYN.HIDDEN_DIM), MutAnyOrigin
    ](p_a_z_buf)
    var p_out = LayoutTensor[
        dtype, Layout.row_major(1, DYN.READOUT), MutAnyOrigin
    ](p_out_buf)
    var p_s_a = LayoutTensor[
        dtype, Layout.row_major(1, DYN.AUG_DIM), MutAnyOrigin
    ](p_s_a_buf)

    # Eval scratch (BATCH=DYN_BATCH — for `eval_member_loss`).
    var e_a_aug_buf = alloc[Scalar[dtype]](DYN_BATCH * DYN.AUG_DIM)
    var e_z_buf = alloc[Scalar[dtype]](DYN_BATCH * DYN.HIDDEN_DIM)
    var e_a_z_buf = alloc[Scalar[dtype]](DYN_BATCH * DYN.HIDDEN_DIM)
    var e_out_buf = alloc[Scalar[dtype]](DYN_BATCH * DYN.READOUT)
    var e_a_aug = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.AUG_DIM), MutAnyOrigin
    ](e_a_aug_buf)
    var e_z = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.HIDDEN_DIM), MutAnyOrigin
    ](e_z_buf)
    var e_a_z = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.HIDDEN_DIM), MutAnyOrigin
    ](e_a_z_buf)
    var e_out = LayoutTensor[
        dtype, Layout.row_major(DYN_BATCH, DYN.READOUT), MutAnyOrigin
    ](e_out_buf)

    # ── Real-only buffer (for dynamics training) ─────────────────────────────
    var rb_obs = alloc[Scalar[dtype]](SAC_BUFFER_CAP * OBS_DIM)
    var rb_action = alloc[Scalar[dtype]](SAC_BUFFER_CAP * ACTION_DIM)
    var rb_next = alloc[Scalar[dtype]](SAC_BUFFER_CAP * OBS_DIM)
    var rb_reward = alloc[Scalar[dtype]](SAC_BUFFER_CAP)
    var rb_size: Int = 0
    var rb_widx: Int = 0

    # ── Training loop ────────────────────────────────────────────────────────
    print("Starting CPU training...")
    print("-" * 70)
    var t0 = perf_counter_ns()
    var rng = PhiloxRandom(seed=UInt64(11), offset=UInt64(0))
    var dyn_step_nums = List[Int](capacity=NUM_ENSEMBLE)
    for _ in range(NUM_ENSEMBLE):
        dyn_step_nums.append(0)
    var elite_indices = List[Int](capacity=NUM_ELITES)
    for _ in range(NUM_ELITES):
        elite_indices.append(0)

    # Allocate the SAC agent's CPU state (network weights + replay buffer).
    var cpu_state = agent.make_cpu_state()

    # Warmup: random actions to fill real buffer + SAC buffer.
    var warmup_obs_raw = env.reset_obs_list()
    var warmup_obs = List[Float64]()
    for i in range(len(warmup_obs_raw)):
        warmup_obs.append(Float64(warmup_obs_raw[i]))
    for _ in range(WARMUP_STEPS):
        var action = agent.random_action[DType.float64]()
        var result = env.step_continuous_vec(action)
        var next_obs = List[Float64]()
        for i in range(len(result[0])):
            next_obs.append(Float64(result[0][i]))
        var reward = Float64(result[1])
        var done = result[2]
        agent.store_transition[DType.float64](
            cpu_state, warmup_obs, action, reward, next_obs, done
        )
        real_buf_add(
            rb_obs, rb_action, rb_next, rb_reward,
            rb_size, rb_widx, SAC_BUFFER_CAP,
            warmup_obs, action, next_obs, reward,
        )
        if done:
            warmup_obs_raw = env.reset_obs_list()
            warmup_obs = List[Float64]()
            for i in range(len(warmup_obs_raw)):
                warmup_obs.append(Float64(warmup_obs_raw[i]))
        else:
            warmup_obs = next_obs^

    print("Warmup complete:", WARMUP_STEPS, " env steps; real buf size:", rb_size)
    print()

    # Episode tracking.
    var ep_obs_raw = env.reset_obs_list()
    var ep_obs = List[Float64]()
    for i in range(len(ep_obs_raw)):
        ep_obs.append(Float64(ep_obs_raw[i]))
    var ep_reward: Float64 = 0.0
    var ep_steps = 0
    var total_env_steps = 0

    var loss_history = List[Float64]()
    for epoch in range(NUM_EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            var action = agent.select_action[DType.float64](cpu_state, ep_obs)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]
            ep_steps += 1
            ep_reward += reward
            agent.store_transition[DType.float64](
                cpu_state, ep_obs, action, reward, next_obs,
                done and (ep_steps < MAX_STEPS_PER_EPISODE),
            )
            real_buf_add(
                rb_obs, rb_action, rb_next, rb_reward,
                rb_size, rb_widx, SAC_BUFFER_CAP,
                ep_obs, action, next_obs, reward,
            )
            total_env_steps += 1

            # Periodic dynamics retrain + synth rollouts.
            if (
                total_env_steps % MODEL_TRAIN_FREQ == 0
                and rb_size >= DYN_BATCH
            ):
                # Train each ensemble member on DYN_TRAIN_BATCHES minibatches.
                var avg_loss: Float64 = 0.0
                for m in range(NUM_ENSEMBLE):
                    var member_loss: Float64 = 0.0
                    for _ in range(DYN_TRAIN_BATCHES):
                        build_dyn_batch[DYN_BATCH](
                            rng, rb_obs, rb_action, rb_next, rb_reward,
                            rb_size,
                            s_a_buf_t, target_buf_t,
                        )
                        var loss = ENS.train_member[DYN_BATCH, DYN_OPT](
                            m, s_a_t, target_t,
                            dyn_params_buf, dyn_grads_buf,
                            dyn_opt_state_buf, dyn_opt_global_buf,
                            latents, mu_eps, a_below, z_below, dx,
                            dyn_step_nums[m],
                            T_infer=T_INFER,
                            lr_x=Scalar[dtype](LR_X),
                            grad_clip_norm=DYN_GRAD_CLIP,
                        )
                        member_loss += loss
                    avg_loss += member_loss / Float64(DYN_TRAIN_BATCHES)
                avg_loss /= Float64(NUM_ENSEMBLE)
                loss_history.append(avg_loss)

                # Re-evaluate elites (single batch holdout).
                build_dyn_batch[DYN_BATCH](
                    rng, rb_obs, rb_action, rb_next, rb_reward,
                    rb_size,
                    s_a_buf_t, target_buf_t,
                )
                var holdout_losses = List[Float64](capacity=NUM_ENSEMBLE)
                for m in range(NUM_ENSEMBLE):
                    var L = ENS.eval_member_loss[DYN_BATCH](
                        m, s_a_t, target_t, dyn_params_buf,
                        e_a_aug, e_z, e_a_z, e_out,
                    )
                    holdout_losses.append(L)
                ENS.select_elites(holdout_losses, elite_indices)

                # Generate synth rollouts: pick random real start states,
                # roll forward via random elite, store in agent's SAC buffer.
                for _ in range(NUM_ROLLOUTS_PER_RETRAIN):
                    var u = Float64(rng.step_uniform()[0])
                    var src_idx = Int(u * Float64(rb_size)) % rb_size
                    var sim_obs = List[Float64](capacity=OBS_DIM)
                    for d in range(OBS_DIM):
                        sim_obs.append(
                            Float64(rb_obs[src_idx * OBS_DIM + d])
                        )
                    for _ in range(ROLLOUT_LENGTH):
                        var sim_act = agent.select_action[
                            DType.float64
                        ](cpu_state, sim_obs)
                        # Build normalized (s, a) for predict.
                        p_s_a_buf[0] = Scalar[dtype](sim_obs[0])
                        p_s_a_buf[1] = Scalar[dtype](sim_obs[1])
                        p_s_a_buf[2] = Scalar[dtype](
                            sim_obs[2] / PEND_MAX_SPEED
                        )
                        p_s_a_buf[OBS_DIM] = Scalar[dtype](
                            sim_act[0] / PEND_MAX_TORQUE
                        )
                        # Pick random elite.
                        var ue = Float64(rng.step_uniform()[0])
                        var elite_pos = Int(
                            ue * Float64(NUM_ELITES)
                        ) % NUM_ELITES
                        var elite_m = elite_indices[elite_pos]
                        ENS.predict_member[1](
                            elite_m, p_s_a, dyn_params_buf,
                            p_a_aug, p_z, p_a_z, p_out,
                        )
                        # Un-normalize prediction.
                        var d0 = Float64(p_out_buf[0])
                        var d1 = Float64(p_out_buf[1])
                        var d2 = Float64(p_out_buf[2]) * PEND_MAX_SPEED
                        var rsim = Float64(p_out_buf[3]) * REWARD_SCALE
                        var sim_next = List[Float64](capacity=OBS_DIM)
                        sim_next.append(sim_obs[0] + d0)
                        sim_next.append(sim_obs[1] + d1)
                        sim_next.append(sim_obs[2] + d2)
                        # Store synth transition in agent buffer (mark
                        # not-done; horizon=1 always).
                        agent.store_transition[DType.float64](
                            cpu_state, sim_obs, sim_act, rsim, sim_next, False
                        )
                        sim_obs = sim_next^

            # SAC updates.
            for _ in range(SAC_UPDATES_PER_STEP):
                _ = agent.do_cpu_train_step(cpu_state)

            # Episode boundary handling.
            if done or ep_steps >= MAX_STEPS_PER_EPISODE:
                ep_obs_raw = env.reset_obs_list()
                ep_obs = List[Float64]()
                for i in range(len(ep_obs_raw)):
                    ep_obs.append(Float64(ep_obs_raw[i]))
                ep_reward = 0.0
                ep_steps = 0
            else:
                ep_obs = next_obs^

        # Eval at end of epoch: greedy policy.
        var eval_total: Float64 = 0.0
        for _ in range(EVAL_EPISODES):
            var eo_raw = env.reset_obs_list()
            var eo = List[Float64]()
            for i in range(len(eo_raw)):
                eo.append(Float64(eo_raw[i]))
            var ereward: Float64 = 0.0
            for _ in range(MAX_STEPS_PER_EPISODE):
                var ea = agent.select_greedy_action(cpu_state, eo)
                var er = env.step_continuous_vec(ea)
                var en = List[Float64]()
                for i in range(len(er[0])):
                    en.append(Float64(er[0][i]))
                ereward += Float64(er[1])
                if er[2]:
                    break
                eo = en^
            eval_total += ereward
        var avg_eval = eval_total / Float64(EVAL_EPISODES)
        print(
            "Epoch", epoch + 1, " | Eval reward:", avg_eval,
            " | Env steps:", total_env_steps,
            " | Alpha:", String(agent.alpha)[byte=:6],
            " | Last dyn loss:",
            loss_history[len(loss_history) - 1] if len(loss_history) > 0 else 0.0,
        )

    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    print("-" * 70)
    print()
    print("=== PCN-MBPO Pendulum CPU summary ===")
    print("  Total env steps :", NUM_EPOCHS * STEPS_PER_EPOCH)
    print("  Wall time       :", elapsed, "s")
    print("  Final α         :", String(agent.alpha)[byte=:6])
    print("=== Done ===")

    # Cleanup.
    dyn_params_buf.free()
    dyn_grads_buf.free()
    dyn_opt_state_buf.free()
    dyn_opt_global_buf.free()
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
    s_a_buf_t.free()
    target_buf_t.free()
    p_a_aug_buf.free()
    p_z_buf.free()
    p_a_z_buf.free()
    p_out_buf.free()
    p_s_a_buf.free()
    e_a_aug_buf.free()
    e_z_buf.free()
    e_a_z_buf.free()
    e_out_buf.free()
    rb_obs.free()
    rb_action.free()
    rb_next.free()
    rb_reward.free()
