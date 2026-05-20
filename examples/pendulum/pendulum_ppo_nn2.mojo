"""PPO training on Pendulum V1 — end-to-end with the nn2 framework.

Phase 6 keystone: proves nn2 can express PPO. Uses nn2 primitives
directly (Sequential, Linear, Tanh, GaussianHead, Adam, MSELoss,
PPOActorLoss), hand-rolled training loop. No nn2 Trainer (PPO has two
nets with separate optimizers — fits poorly under Trainer[NET, OPT, LOSS]).

Architecture (CleanRL-style PPO continuous):
    Actor:  Linear[3,64] → Tanh → Linear[64,64] → Tanh → GaussianHead[64,1]
            (state-dep mu head + state-indep log_std vector)
    Critic: Linear[3,64] → Tanh → Linear[64,64] → Tanh → Linear[64,1]

Pendulum V1:
    Obs: [cos(θ), sin(θ), θ_dot] (3D continuous)
    Act: torque ∈ [-2, 2] (1D continuous, env clamps at boundary)
    Reward: -(θ² + 0.1·θ_dot² + 0.001·torque²) per step, episode = 200 steps
    Solved at ≈ -200 (random ≈ -1600, optimal ≈ -150).

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_ppo_nn2.mojo
"""

from std.math import log as flog, exp as fexp, sqrt as fsqrt
from std.memory import alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.gaussian_head import GaussianHead
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.mse import MSELoss
from mojo_rl.nn2.loss.ppo_actor_loss import PPOActorLoss
from mojo_rl.nn2.training.episode_tracker import EpisodeTracker
from mojo_rl.nn2.training.gae import compute_gae, normalize_in_place
from mojo_rl.nn2.random.box_muller import box_muller_normal

from mojo_rl.envs.pendulum import PendulumEnv


# ──────────────────────────────────────────────────────────────────────────
# Hyperparameters (CleanRL PPO Pendulum-v1 defaults)
# ──────────────────────────────────────────────────────────────────────────

comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64

comptime ROLLOUT_LEN = 2048  # CleanRL default
comptime MINIBATCH = 64
comptime N_MINIBATCHES = ROLLOUT_LEN // MINIBATCH  # 32
comptime N_EPOCHS = 10
comptime TOTAL_TIMESTEPS = 200_000  # CleanRL Pendulum typically converges 200-500k

comptime GAMMA: Scalar[DT] = 0.99
comptime GAE_LAMBDA: Scalar[DT] = 0.95
comptime CLIP_EPS: Scalar[DT] = 0.2
comptime ENTROPY_COEF: Scalar[DT] = 0.0
comptime ACTOR_LR: Scalar[DT] = 3e-4
comptime CRITIC_LR: Scalar[DT] = 1e-3
comptime LOG_STD_INIT: Scalar[DT] = -0.5  # std ≈ 0.6

comptime MAX_TORQUE: Scalar[DT] = 2.0


# ──────────────────────────────────────────────────────────────────────────
# Network type aliases
# ──────────────────────────────────────────────────────────────────────────

comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT_DIM],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _gaussian_log_prob(
    action: Scalar[DT], mu: Scalar[DT], log_std: Scalar[DT]
) -> Scalar[DT]:
    """log p(action | N(mu, exp(log_std))), unbounded Gaussian."""
    var std = fexp(log_std)
    var z = (action - mu) / (std + Scalar[DT](1e-6))
    return Scalar[DT](-0.5) * (
        Scalar[DT](1.8378770664093453)  # log(2π)
        + Scalar[DT](2.0) * log_std
        + z * z
    )


def _clamp_log_std(ls: Scalar[DT]) -> Scalar[DT]:
    if ls < -5.0:
        return -5.0
    elif ls > 2.0:
        return 2.0
    else:
        return ls


# ──────────────────────────────────────────────────────────────────────────
# GAE: compute advantages + returns in-place from a rollout.
# ──────────────────────────────────────────────────────────────────────────


def _compute_gae(
    rewards: UnsafePointer[Scalar[DT], MutAnyOrigin],
    values: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dones: UnsafePointer[Scalar[DT], MutAnyOrigin],
    next_value: Scalar[DT],
    advantages: UnsafePointer[Scalar[DT], MutAnyOrigin],
    returns: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Standard GAE: walk backward through the rollout.

    NOTE: For Pendulum specifically, `dones` is set at the time-limit
    truncation (step 200), not on a real terminal. We still bootstrap
    across episode boundaries — this matches Gymnasium's TimeLimit
    semantics. Within one rollout, the next-state value across an
    episode reset is from `values[t+1]` of the new episode's first
    step, which is what we want (or the bootstrap from next_value at
    the rollout boundary)."""
    var last_gae: Scalar[DT] = 0.0
    for t in range(ROLLOUT_LEN - 1, -1, -1):
        # Treat done as truncation: always bootstrap (Pendulum has no
        # real terminal). For envs with real terminals, swap back to
        # `1.0 - dones[t]`.
        var nonterm: Scalar[DT] = 1.0
        var nv: Scalar[DT]
        if t == ROLLOUT_LEN - 1:
            nv = next_value
        else:
            nv = values[t + 1]
        var delta = rewards[t] + GAMMA * nv * nonterm - values[t]
        last_gae = delta + GAMMA * GAE_LAMBDA * nonterm * last_gae
        advantages[t] = last_gae
        returns[t] = last_gae + values[t]


def _normalize_advantages(
    advantages: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Subtract mean, divide by std + 1e-8."""
    var sum: Scalar[DT] = 0.0
    for t in range(ROLLOUT_LEN):
        sum += advantages[t]
    var mean = sum / Scalar[DT](ROLLOUT_LEN)
    var sq: Scalar[DT] = 0.0
    for t in range(ROLLOUT_LEN):
        var d = advantages[t] - mean
        sq += d * d
    var std = fsqrt(sq / Scalar[DT](ROLLOUT_LEN))
    for t in range(ROLLOUT_LEN):
        advantages[t] = (advantages[t] - mean) / (std + Scalar[DT](1e-8))


# ──────────────────────────────────────────────────────────────────────────
# Fisher-Yates shuffle on Int32 indices.
# ──────────────────────────────────────────────────────────────────────────


def _shuffle_indices(indices: UnsafePointer[Int32, MutAnyOrigin]):
    for t in range(ROLLOUT_LEN - 1, 0, -1):
        var j = Int(random_float64() * Float64(t + 1))
        if j > t:
            j = t
        var tmp = indices[t]
        indices[t] = indices[j]
        indices[j] = tmp


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn2 PPO Continuous — Pendulum V1 (CPU)")
    print("=" * 70)
    print("Hyperparameters:")
    print("  OBS_DIM=", OBS_DIM, " ACT_DIM=", ACT_DIM, " HIDDEN=", HIDDEN)
    print(
        "  ROLLOUT_LEN=", ROLLOUT_LEN, " MINIBATCH=", MINIBATCH,
        " N_EPOCHS=", N_EPOCHS,
    )
    print(
        "  ACTOR_LR=", ACTOR_LR, " CRITIC_LR=", CRITIC_LR,
        " CLIP_EPS=", CLIP_EPS,
    )
    print(
        "  GAMMA=", GAMMA, " GAE_LAMBDA=", GAE_LAMBDA,
        " ENTROPY_COEF=", ENTROPY_COEF,
    )
    print(
        "  TOTAL_TIMESTEPS=", TOTAL_TIMESTEPS,
        " LOG_STD_INIT=", LOG_STD_INIT,
    )
    print()

    # ── Build nets, optimizers, losses ──────────────────────────────────
    var actor = ActorNet.make[target="cpu", INIT=Xavier]()
    var critic = CriticNet.make[target="cpu", INIT=Xavier]()

    # Override the GaussianHead's log_std init.
    # The actor's last child is the GaussianHead; reach in and set.
    var ls_view = TileTensor(
        actor.children[4].log_std, row_major[ACT_DIM]()
    )
    for k in range(ACT_DIM):
        ls_view[k] = LOG_STD_INIT

    var actor_opt = Adam.make[target="cpu", M=ActorNet](
        actor, lr=ACTOR_LR
    )
    var critic_opt = Adam.make[target="cpu", M=CriticNet](
        critic, lr=CRITIC_LR
    )

    var ppo_loss = PPOActorLoss[ACT_DIM].make["cpu"](
        clip_eps=CLIP_EPS, entropy_coef=ENTROPY_COEF
    )
    var mse_loss = MSELoss[1].make["cpu"]()

    # ── Rollout buffers ─────────────────────────────────────────────────
    var obs_buf = alloc[Scalar[DT]](ROLLOUT_LEN * OBS_DIM)
    var act_buf = alloc[Scalar[DT]](ROLLOUT_LEN * ACT_DIM)
    var olp_buf = alloc[Scalar[DT]](ROLLOUT_LEN)
    var rew_buf = alloc[Scalar[DT]](ROLLOUT_LEN)
    var val_buf = alloc[Scalar[DT]](ROLLOUT_LEN)
    var done_buf = alloc[Scalar[DT]](ROLLOUT_LEN)
    var adv_buf = alloc[Scalar[DT]](ROLLOUT_LEN)
    var ret_buf = alloc[Scalar[DT]](ROLLOUT_LEN)
    var term_buf = alloc[Scalar[DT]](ROLLOUT_LEN)
    for _t in range(ROLLOUT_LEN):
        term_buf[_t] = 0.0

    # ── Single-step scratch (BATCH=1 actor/critic forward) ──────────────
    var ob1 = alloc[Scalar[DT]](OBS_DIM)
    var ao1 = alloc[Scalar[DT]](2 * ACT_DIM)
    var v1 = alloc[Scalar[DT]](1)
    var z_scratch = alloc[Scalar[DT]](ACT_DIM)

    # ── Minibatch scratch (BATCH=MINIBATCH train) ───────────────────────
    var mb_obs = alloc[Scalar[DT]](MINIBATCH * OBS_DIM)
    var mb_act = alloc[Scalar[DT]](MINIBATCH * ACT_DIM)
    var mb_olp = alloc[Scalar[DT]](MINIBATCH)
    var mb_adv = alloc[Scalar[DT]](MINIBATCH)
    var mb_ret = alloc[Scalar[DT]](MINIBATCH * 1)
    var mb_ao = alloc[Scalar[DT]](MINIBATCH * 2 * ACT_DIM)
    var mb_go = alloc[Scalar[DT]](MINIBATCH * 2 * ACT_DIM)
    var mb_gi = alloc[Scalar[DT]](MINIBATCH * OBS_DIM)
    var mb_v = alloc[Scalar[DT]](MINIBATCH * 1)
    var mb_gv = alloc[Scalar[DT]](MINIBATCH * 1)

    # ── Index buffer for shuffling ──────────────────────────────────────
    var indices = alloc[Int32](ROLLOUT_LEN)

    # ── Env ─────────────────────────────────────────────────────────────
    var env = PendulumEnv[DT]()
    _ = env.reset()

    # Tracking
    var tracker = EpisodeTracker.new(
        window_size=10, initial_fill=Scalar[DT](-1600.0)
    )

    var total_steps: Int = 0
    var rollout_idx: Int = 0
    var t_start = perf_counter_ns()

    while total_steps < TOTAL_TIMESTEPS:
        rollout_idx += 1

        # ─────────────────────────────────────────────────────────────
        # Phase 1: rollout
        # ─────────────────────────────────────────────────────────────
        var obs_self = env.get_obs_list()
        for t in range(ROLLOUT_LEN):
            # Store obs.
            for d in range(OBS_DIM):
                obs_buf[t * OBS_DIM + d] = obs_self[d]
                ob1[d] = obs_self[d]

            # Actor forward (BATCH=1) → ao1 = [mu | log_std].
            var ob1_t = TileTensor(ob1, row_major[1, OBS_DIM]())
            var ao1_t = TileTensor(ao1, row_major[1, 2 * ACT_DIM]())
            actor.forward["cpu", 1](ob1_t, ao1_t)

            # Sample action + compute log_prob.
            box_muller_normal(z_scratch, ACT_DIM)
            var action_t: Scalar[DT] = 0.0
            var lp_total: Scalar[DT] = 0.0
            for j in range(ACT_DIM):
                var mu = ao1[j]
                var ls = _clamp_log_std(ao1[ACT_DIM + j])
                var sample = mu + fexp(ls) * z_scratch[j]
                action_t = sample  # ACT_DIM=1: just store once
                act_buf[t * ACT_DIM + j] = sample
                lp_total += _gaussian_log_prob(sample, mu, ls)
            olp_buf[t] = lp_total

            # Critic forward (BATCH=1) → v1.
            var v1_t = TileTensor(v1, row_major[1, 1]())
            critic.forward["cpu", 1](ob1_t, v1_t)
            val_buf[t] = v1[0]

            # Env step (clamp action to torque range).
            var torque = action_t
            if torque > MAX_TORQUE:
                torque = MAX_TORQUE
            elif torque < -MAX_TORQUE:
                torque = -MAX_TORQUE
            var step_res = env.step_continuous(torque)
            obs_self = step_res[0].copy()
            var reward = step_res[1]
            var done = step_res[2]
            rew_buf[t] = reward
            done_buf[t] = 1.0 if done else 0.0
            tracker.add_reward(reward)
            total_steps += 1

            if done:
                tracker.end_episode()
                _ = env.reset()
                obs_self = env.get_obs_list()

        # ─────────────────────────────────────────────────────────────
        # Phase 2: bootstrap + GAE
        # ─────────────────────────────────────────────────────────────
        for d in range(OBS_DIM):
            ob1[d] = obs_self[d]
        var ob1_t = TileTensor(ob1, row_major[1, OBS_DIM]())
        var v1_t = TileTensor(v1, row_major[1, 1]())
        critic.forward["cpu", 1](ob1_t, v1_t)
        var next_value = v1[0]
        compute_gae(
            ROLLOUT_LEN, rew_buf, val_buf, term_buf, next_value,
            GAMMA, GAE_LAMBDA, adv_buf, ret_buf,
        )
        # NOTE: advantage normalization is done per-minibatch (CleanRL style),
        # not here — moved inside the inner update loop below.

        # ─────────────────────────────────────────────────────────────
        # Phase 3: K-epoch minibatch updates
        # ─────────────────────────────────────────────────────────────
        for t in range(ROLLOUT_LEN):
            indices[t] = Int32(t)

        var actor_loss_sum: Scalar[DT] = 0.0
        var critic_loss_sum: Scalar[DT] = 0.0
        var update_count: Int = 0

        for _epoch in range(N_EPOCHS):
            _shuffle_indices(indices)
            for mb in range(N_MINIBATCHES):
                # Gather minibatch.
                for k in range(MINIBATCH):
                    var src = Int(indices[mb * MINIBATCH + k])
                    for d in range(OBS_DIM):
                        mb_obs[k * OBS_DIM + d] = obs_buf[src * OBS_DIM + d]
                    for j in range(ACT_DIM):
                        mb_act[k * ACT_DIM + j] = act_buf[src * ACT_DIM + j]
                    mb_olp[k] = olp_buf[src]
                    mb_adv[k] = adv_buf[src]
                    mb_ret[k] = ret_buf[src]
                # Per-minibatch advantage normalization (CleanRL style).
                normalize_in_place(MINIBATCH, mb_adv)

                # ── Actor update ─────────────────────────────────────
                var mb_obs_t = TileTensor(mb_obs, row_major[MINIBATCH, OBS_DIM]())
                var mb_ao_t = TileTensor(mb_ao, row_major[MINIBATCH, 2 * ACT_DIM]())
                var mb_go_t = TileTensor(mb_go, row_major[MINIBATCH, 2 * ACT_DIM]())
                var mb_gi_t = TileTensor(mb_gi, row_major[MINIBATCH, OBS_DIM]())

                actor.forward["cpu", MINIBATCH](mb_obs_t, mb_ao_t)
                var mb_act_t = TileTensor(mb_act, row_major[MINIBATCH, ACT_DIM]())
                var mb_olp_t = TileTensor(mb_olp, row_major[MINIBATCH]())
                var mb_adv_t = TileTensor(mb_adv, row_major[MINIBATCH]())
                # Loss is informational here — we still call forward to
                # ensure it stays consistent for instrumentation; backward
                # produces grad_actor_output.
                var actor_loss_v = ppo_loss.forward["cpu", MINIBATCH](
                    mb_ao_t, mb_act_t, mb_olp_t, mb_adv_t
                )
                actor_loss_sum += actor_loss_v
                ppo_loss.backward["cpu", MINIBATCH](
                    mb_ao_t, mb_act_t, mb_olp_t, mb_adv_t, mb_go_t
                )
                actor_opt.zero_grad["cpu", M=ActorNet](actor)
                actor.backward["cpu", MINIBATCH](mb_go_t, mb_gi_t)
                actor_opt.step["cpu", M=ActorNet](actor)

                # ── Critic update ────────────────────────────────────
                var mb_v_t = TileTensor(mb_v, row_major[MINIBATCH, 1]())
                var mb_gv_t = TileTensor(mb_gv, row_major[MINIBATCH, 1]())
                var mb_gi_critic = TileTensor(
                    mb_gi, row_major[MINIBATCH, OBS_DIM]()
                )
                var mb_ret_t = TileTensor(mb_ret, row_major[MINIBATCH, 1]())

                critic.forward["cpu", MINIBATCH](mb_obs_t, mb_v_t)
                var critic_loss_v = mse_loss.forward["cpu", MINIBATCH](
                    mb_v_t, mb_ret_t
                )
                critic_loss_sum += critic_loss_v
                update_count += 1
                mse_loss.backward["cpu", MINIBATCH](mb_ret_t, mb_gv_t)
                critic_opt.zero_grad["cpu", M=CriticNet](critic)
                critic.backward["cpu", MINIBATCH](mb_gv_t, mb_gi_critic)
                critic_opt.step["cpu", M=CriticNet](critic)

        # ─────────────────────────────────────────────────────────────
        # Logging
        # ─────────────────────────────────────────────────────────────
        var mean_ret = tracker.mean_return()
        var ep_count = tracker.ep_count
        var elapsed = Float64(perf_counter_ns() - t_start) / 1e9

        # Read current log_std for trace.
        var cur_ls = actor.children[4].log_std[0]

        var mean_actor_loss = actor_loss_sum / Scalar[DT](update_count)
        var mean_critic_loss = critic_loss_sum / Scalar[DT](update_count)

        print(
            "[rollout ", rollout_idx, "] steps=", total_steps,
            "  episodes=", ep_count,
            "  mean_ret(10)=", mean_ret,
            "  log_std=", cur_ls,
            "  actor_L=", mean_actor_loss,
            "  critic_L=", mean_critic_loss,
            "  elapsed=", elapsed, "s",
        )

    # ── Final report ────────────────────────────────────────────────────
    print("=" * 70)
    print("Training complete.")
    var final_mean = tracker.mean_return()
    print("Final mean ep return (last 10): ", final_mean)
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)

    obs_buf.free()
    act_buf.free()
    olp_buf.free()
    rew_buf.free()
    val_buf.free()
    done_buf.free()
    adv_buf.free()
    ret_buf.free()
    ob1.free()
    ao1.free()
    v1.free()
    mb_obs.free()
    mb_act.free()
    mb_olp.free()
    mb_adv.free()
    mb_ret.free()
    mb_ao.free()
    mb_go.free()
    mb_gi.free()
    mb_v.free()
    mb_gv.free()
    indices.free()
