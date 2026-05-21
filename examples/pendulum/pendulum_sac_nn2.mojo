"""SAC training on Pendulum V1 — end-to-end with the nn2 framework.

Phase 7 keystone, Phase 8.1 refactor: extracted boilerplate into
`nn2/data/CPUReplay`, `nn2/core/OnlineTargetPair`, `nn2/random/box_muller_normal`,
`nn2/training/EpisodeTracker`. What remains in this file is the SAC
algorithm itself (≈ 250 LOC vs Phase 7's 674).

Architecture (CleanRL-style SAC continuous):
    Actor:    StochasticActor[3, 1,
                  Linear[3, HIDDEN], ReLU,
                  Linear[HIDDEN, HIDDEN], ReLU]
              → output [BATCH, 2]: [mu | log_std]
    Critic_i: Linear[4, HIDDEN] → ReLU → Linear[HIDDEN, HIDDEN]
              → ReLU → Linear[HIDDEN, 1]
    log_alpha: trainable scalar (init log(0.2)), own Adam state.

Pendulum's `done` flag is a step-200 *truncation*, not a terminal —
we hard-code `nonterm = 1.0` in the critic target bootstrap. See
feedback_ppo_pendulum_timelimit_gae memory.

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_sac_nn2.mojo
"""

from std.math import exp as fexp, log as flog, sqrt as fsqrt
from std.memory import alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.mse import MSELoss
from mojo_rl.nn2.loss.sac_actor_loss import (
    squashed_gaussian_sample,
    sac_actor_backward,
    sac_actor_loss_value,
)
from mojo_rl.nn2.core.online_target_pair import OnlineTargetPair
from mojo_rl.nn2.data.cpu_replay import CPUReplay
from mojo_rl.nn2.random.box_muller import box_muller_normal
from mojo_rl.nn2.training.episode_tracker import EpisodeTracker

from mojo_rl.envs.pendulum import PendulumEnv


# ──────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────

comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime SA_DIM = OBS_DIM + ACT_DIM

comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime LEARNING_STARTS = 1_000
comptime TOTAL_TIMESTEPS = 30_000

comptime GAMMA: Scalar[DT] = 0.99
comptime TAU: Scalar[DT] = 0.005
comptime ACTOR_LR: Scalar[DT] = 3e-4
comptime CRITIC_LR: Scalar[DT] = 1e-3
comptime ALPHA_LR: Scalar[DT] = 3e-4
comptime INIT_ALPHA: Scalar[DT] = 0.2
comptime TARGET_ENTROPY: Scalar[DT] = -1.0
comptime ACTION_SCALE: Scalar[DT] = 2.0


# ──────────────────────────────────────────────────────────────────────────
# Network types
# ──────────────────────────────────────────────────────────────────────────

comptime ActorNet = StochasticActor[
    OBS_DIM, ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]

comptime CriticNet = Sequential[
    Linear[SA_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


# ──────────────────────────────────────────────────────────────────────────
# SAC-specific helpers (stay inline — used only here).
# ──────────────────────────────────────────────────────────────────────────


def _concat_sa[B: Int](
    obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
    act: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_sa: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """sa[b, :OBS] = obs[b]; sa[b, OBS:] = act[b]."""
    for b in range(B):
        for d in range(OBS_DIM):
            out_sa[b * SA_DIM + d] = obs[b * OBS_DIM + d]
        for j in range(ACT_DIM):
            out_sa[b * SA_DIM + OBS_DIM + j] = act[b * ACT_DIM + j]


def _accum_grad_action[B: Int](
    grad_sa: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad_action_acc: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Accumulate grad_action_acc[b, j] += grad_sa[b, OBS+j]."""
    for b in range(B):
        for j in range(ACT_DIM):
            grad_action_acc[b * ACT_DIM + j] = (
                grad_action_acc[b * ACT_DIM + j]
                + grad_sa[b * SA_DIM + OBS_DIM + j]
            )


@fieldwise_init
struct ScalarAdam(Movable & ImplicitlyDestructible):
    """Single-scalar Adam for log_alpha (entropy temperature)."""

    var value: Scalar[DT]
    var m: Scalar[DT]
    var v: Scalar[DT]
    var t: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]

    @staticmethod
    def new(initial: Scalar[DT], lr: Scalar[DT]) -> Self:
        return Self(
            value=initial, m=0.0, v=0.0, t=0,
            lr=lr, beta1=0.9, beta2=0.999, eps=1e-8,
        )

    def step(mut self, grad: Scalar[DT]):
        self.t += 1
        var one: Scalar[DT] = 1.0
        self.m = self.beta1 * self.m + (one - self.beta1) * grad
        self.v = self.beta2 * self.v + (one - self.beta2) * grad * grad
        var b1t = self.beta1
        var b2t = self.beta2
        for _ in range(self.t - 1):
            b1t *= self.beta1
            b2t *= self.beta2
        var m_hat = self.m / (one - b1t)
        var v_hat = self.v / (one - b2t)
        self.value = self.value - self.lr * m_hat / (fsqrt(v_hat) + self.eps)


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn2 SAC Continuous — Pendulum V1 (CPU)")
    print("=" * 70)
    print(
        "  OBS=", OBS_DIM, " ACT=", ACT_DIM, " HIDDEN=", HIDDEN,
        " BATCH=", BATCH,
    )
    print(
        "  LRs: actor=", ACTOR_LR, " critic=", CRITIC_LR,
        " alpha=", ALPHA_LR, " GAMMA=", GAMMA, " TAU=", TAU,
    )
    print(
        "  REPLAY=", REPLAY_CAPACITY, " LEARNING_STARTS=", LEARNING_STARTS,
        " STEPS=", TOTAL_TIMESTEPS,
    )
    print()

    # ── Nets + optimizers ──────────────────────────────────────────────
    var actor = ActorNet.make[target="cpu", INIT=Xavier]()
    var pair1 = OnlineTargetPair[CriticNet].make[target="cpu", INIT=Xavier]()
    var pair2 = OnlineTargetPair[CriticNet].make[target="cpu", INIT=Xavier]()

    var actor_opt = Adam.make[target="cpu", M=ActorNet](actor)
    actor_opt.lr = ACTOR_LR
    var critic1_opt = Adam.make[target="cpu", M=CriticNet](pair1.online)
    critic1_opt.lr = CRITIC_LR
    var critic2_opt = Adam.make[target="cpu", M=CriticNet](pair2.online)
    critic2_opt.lr = CRITIC_LR
    var alpha_opt = ScalarAdam.new(flog(INIT_ALPHA), ALPHA_LR)
    var mse_loss = MSELoss[1].make["cpu"]()

    # ── Replay + per-episode return tracker ────────────────────────────
    var buf = CPUReplay[OBS_DIM, ACT_DIM, REPLAY_CAPACITY].new()
    var tracker = EpisodeTracker.new(
        window_size=10, initial_fill=Scalar[DT](-1250.0)
    )

    # ── Scratch (single-step actor for env interaction) ────────────────
    var ob1 = alloc[Scalar[DT]](OBS_DIM)
    var ao1 = alloc[Scalar[DT]](2 * ACT_DIM)
    var z1 = alloc[Scalar[DT]](ACT_DIM)
    var a1 = alloc[Scalar[DT]](ACT_DIM)
    var lp1 = alloc[Scalar[DT]](1)

    # ── Scratch (minibatch) ────────────────────────────────────────────
    var mb_s = alloc[Scalar[DT]](BATCH * OBS_DIM)
    var mb_a = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_r = alloc[Scalar[DT]](BATCH)
    var mb_sp = alloc[Scalar[DT]](BATCH * OBS_DIM)
    var mb_d = alloc[Scalar[DT]](BATCH)
    var mb_sa = alloc[Scalar[DT]](BATCH * SA_DIM)
    var mb_ao_s = alloc[Scalar[DT]](BATCH * 2 * ACT_DIM)
    var mb_ao_sp = alloc[Scalar[DT]](BATCH * 2 * ACT_DIM)
    var mb_z_s = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_z_sp = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_act_s = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_act_sp = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_lp_s = alloc[Scalar[DT]](BATCH)
    var mb_lp_sp = alloc[Scalar[DT]](BATCH)
    var mb_q1 = alloc[Scalar[DT]](BATCH * 1)
    var mb_q2 = alloc[Scalar[DT]](BATCH * 1)
    var mb_q1_tgt = alloc[Scalar[DT]](BATCH * 1)
    var mb_q2_tgt = alloc[Scalar[DT]](BATCH * 1)
    var mb_y = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_q1 = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_q2 = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_sa1 = alloc[Scalar[DT]](BATCH * SA_DIM)
    var mb_grad_sa2 = alloc[Scalar[DT]](BATCH * SA_DIM)
    var mb_grad_action = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_grad_ao = alloc[Scalar[DT]](BATCH * 2 * ACT_DIM)
    var mb_grad_obs_unused = alloc[Scalar[DT]](BATCH * OBS_DIM)
    var min_q_buf = alloc[Scalar[DT]](BATCH)

    # Single-transition scratch for buf.add (action + next-obs).
    var act_scratch = alloc[Scalar[DT]](ACT_DIM)
    var sp_scratch = alloc[Scalar[DT]](OBS_DIM)

    # ── Env ────────────────────────────────────────────────────────────
    var env = PendulumEnv[DT]()
    _ = env.reset()
    var obs_self = env.get_obs_list()

    var t_start = perf_counter_ns()
    var actor_L_accum: Scalar[DT] = 0.0
    var critic_L_accum: Scalar[DT] = 0.0
    var alpha_accum: Scalar[DT] = 0.0
    var update_count: Int = 0
    comptime LOG_EVERY = 1_000

    var step: Int = 0
    while step < TOTAL_TIMESTEPS:
        # ── Act (single-step actor for env interaction) ────────────────
        for d in range(OBS_DIM):
            ob1[d] = obs_self[d]

        var torque: Scalar[DT] = 0.0
        if step < LEARNING_STARTS:
            torque = Scalar[DT](2.0 * random_float64() - 1.0) * ACTION_SCALE
        else:
            var ob1_t = TileTensor(ob1, row_major[1, OBS_DIM]())
            var ao1_t = TileTensor(ao1, row_major[1, 2 * ACT_DIM]())
            actor.forward["cpu", 1](ob1_t, ao1_t)
            box_muller_normal(z1, ACT_DIM)
            var z1_t = TileTensor(z1, row_major[1, ACT_DIM]())
            var a1_t = TileTensor(a1, row_major[1, ACT_DIM]())
            var lp1_t = TileTensor(lp1, row_major[1]())
            squashed_gaussian_sample[ACT_DIM, 1](
                ao1_t, z1_t, ACTION_SCALE, a1_t, lp1_t
            )
            torque = a1[0]
        if torque > ACTION_SCALE:
            torque = ACTION_SCALE
        elif torque < -ACTION_SCALE:
            torque = -ACTION_SCALE

        var step_res = env.step_continuous(torque)
        var next_obs = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        tracker.add_reward(reward)

        act_scratch[0] = torque
        for i in range(OBS_DIM):
            sp_scratch[i] = next_obs[i]
        buf.add(
            ob1, act_scratch, reward, sp_scratch,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )

        if done:
            tracker.end_episode()
            _ = env.reset()
            obs_self = env.get_obs_list()
        else:
            obs_self = next_obs.copy()

        step += 1

        # ── Train ──────────────────────────────────────────────────────
        if step < LEARNING_STARTS or buf.size < BATCH:
            if step % LOG_EVERY == 0:
                var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
                print(
                    "[warmup ", step, "] mean_ret(10)=", tracker.mean_return(),
                    " ep=", tracker.ep_count, " elapsed=", elapsed, "s",
                )
            continue

        buf.sample(BATCH, mb_s, mb_a, mb_r, mb_sp, mb_d)
        var alpha = fexp(alpha_opt.value)

        # ── Critic target y = r + γ·(min Q' − α·log_prob') ─────────────
        var mb_sp_t = TileTensor(mb_sp, row_major[BATCH, OBS_DIM]())
        var mb_ao_sp_t = TileTensor(mb_ao_sp, row_major[BATCH, 2 * ACT_DIM]())
        actor.forward["cpu", BATCH](mb_sp_t, mb_ao_sp_t)
        box_muller_normal(mb_z_sp, BATCH * ACT_DIM)
        var mb_z_sp_t = TileTensor(mb_z_sp, row_major[BATCH, ACT_DIM]())
        var mb_act_sp_t = TileTensor(mb_act_sp, row_major[BATCH, ACT_DIM]())
        var mb_lp_sp_t = TileTensor(mb_lp_sp, row_major[BATCH]())
        squashed_gaussian_sample[ACT_DIM, BATCH](
            mb_ao_sp_t, mb_z_sp_t, ACTION_SCALE, mb_act_sp_t, mb_lp_sp_t
        )
        _concat_sa[BATCH](mb_sp, mb_act_sp, mb_sa)
        var mb_sa_t = TileTensor(mb_sa, row_major[BATCH, SA_DIM]())
        var mb_q1_tgt_t = TileTensor(mb_q1_tgt, row_major[BATCH, 1]())
        var mb_q2_tgt_t = TileTensor(mb_q2_tgt, row_major[BATCH, 1]())
        pair1.target_net.forward["cpu", BATCH](mb_sa_t, mb_q1_tgt_t)
        pair2.target_net.forward["cpu", BATCH](mb_sa_t, mb_q2_tgt_t)
        for b in range(BATCH):
            var qmin = mb_q1_tgt[b] if mb_q1_tgt[b] < mb_q2_tgt[b] else mb_q2_tgt[b]
            # Pendulum: time-limit truncation → always bootstrap.
            var nonterm: Scalar[DT] = 1.0
            mb_y[b] = mb_r[b] + GAMMA * nonterm * (qmin - alpha * mb_lp_sp[b])

        # ── Critic update: Q1, Q2 forward + MSE loss + backward ────────
        _concat_sa[BATCH](mb_s, mb_a, mb_sa)
        var mb_q1_t = TileTensor(mb_q1, row_major[BATCH, 1]())
        var mb_q2_t = TileTensor(mb_q2, row_major[BATCH, 1]())
        critic1_opt.zero_grad["cpu", M=CriticNet](pair1.online)
        critic2_opt.zero_grad["cpu", M=CriticNet](pair2.online)
        pair1.online.forward["cpu", BATCH](mb_sa_t, mb_q1_t)
        pair2.online.forward["cpu", BATCH](mb_sa_t, mb_q2_t)
        var mb_y_t = TileTensor(mb_y, row_major[BATCH, 1]())
        var loss1 = mse_loss.forward["cpu", BATCH](mb_q1_t, mb_y_t)
        var mb_grad_q1_t = TileTensor(mb_grad_q1, row_major[BATCH, 1]())
        mse_loss.backward["cpu", BATCH](mb_y_t, mb_grad_q1_t)
        var mb_grad_sa1_t = TileTensor(mb_grad_sa1, row_major[BATCH, SA_DIM]())
        pair1.online.backward["cpu", BATCH](mb_grad_q1_t, mb_grad_sa1_t)
        critic1_opt.step["cpu", M=CriticNet](pair1.online)

        var loss2 = mse_loss.forward["cpu", BATCH](mb_q2_t, mb_y_t)
        var mb_grad_q2_t = TileTensor(mb_grad_q2, row_major[BATCH, 1]())
        mse_loss.backward["cpu", BATCH](mb_y_t, mb_grad_q2_t)
        var mb_grad_sa2_t = TileTensor(mb_grad_sa2, row_major[BATCH, SA_DIM]())
        pair2.online.backward["cpu", BATCH](mb_grad_q2_t, mb_grad_sa2_t)
        critic2_opt.step["cpu", M=CriticNet](pair2.online)
        critic_L_accum += loss1 + loss2

        # ── Actor update: re-sample, critic min, sac_actor_backward ────
        var mb_s_t = TileTensor(mb_s, row_major[BATCH, OBS_DIM]())
        var mb_ao_s_t = TileTensor(mb_ao_s, row_major[BATCH, 2 * ACT_DIM]())
        actor_opt.zero_grad["cpu", M=ActorNet](actor)
        actor.forward["cpu", BATCH](mb_s_t, mb_ao_s_t)
        box_muller_normal(mb_z_s, BATCH * ACT_DIM)
        var mb_z_s_t = TileTensor(mb_z_s, row_major[BATCH, ACT_DIM]())
        var mb_act_s_t = TileTensor(mb_act_s, row_major[BATCH, ACT_DIM]())
        var mb_lp_s_t = TileTensor(mb_lp_s, row_major[BATCH]())
        squashed_gaussian_sample[ACT_DIM, BATCH](
            mb_ao_s_t, mb_z_s_t, ACTION_SCALE, mb_act_s_t, mb_lp_s_t
        )
        _concat_sa[BATCH](mb_s, mb_act_s, mb_sa)
        pair1.online.forward["cpu", BATCH](mb_sa_t, mb_q1_t)
        pair2.online.forward["cpu", BATCH](mb_sa_t, mb_q2_t)

        var inv_batch: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BATCH)
        for b in range(BATCH):
            var q1b = mb_q1[b]
            var q2b = mb_q2[b]
            if q1b < q2b:
                mb_grad_q1[b] = -inv_batch; mb_grad_q2[b] = 0.0
                min_q_buf[b] = q1b
            else:
                mb_grad_q1[b] = 0.0; mb_grad_q2[b] = -inv_batch
                min_q_buf[b] = q2b
        # Phase 8.2: critics are frozen during the actor update — call
        # `backward[mode="input_only"]` so grad_action flows back but
        # grad_w/grad_b on the critics are not touched. (Previously this
        # called `backward` which wrote critic grads that were then thrown
        # away at the next zero_grad — wasted work + footgun if zero_grad
        # were ever skipped.)
        pair1.online.backward["cpu", BATCH, mode="input_only"](mb_grad_q1_t, mb_grad_sa1_t)
        pair2.online.backward["cpu", BATCH, mode="input_only"](mb_grad_q2_t, mb_grad_sa2_t)
        for k in range(BATCH * ACT_DIM):
            mb_grad_action[k] = 0.0
        _accum_grad_action[BATCH](mb_grad_sa1, mb_grad_action)
        _accum_grad_action[BATCH](mb_grad_sa2, mb_grad_action)

        var mb_grad_action_t = TileTensor(
            mb_grad_action, row_major[BATCH, ACT_DIM]()
        )
        var mb_grad_ao_t = TileTensor(
            mb_grad_ao, row_major[BATCH, 2 * ACT_DIM]()
        )
        sac_actor_backward[ACT_DIM, BATCH](
            mb_ao_s_t, mb_z_s_t, mb_grad_action_t,
            alpha, ACTION_SCALE, mb_grad_ao_t,
        )
        var mb_min_q_t = TileTensor(min_q_buf, row_major[BATCH]())
        actor_L_accum += sac_actor_loss_value[BATCH](
            mb_lp_s_t, mb_min_q_t, alpha
        )
        var mb_grad_obs_t = TileTensor(
            mb_grad_obs_unused, row_major[BATCH, OBS_DIM]()
        )
        actor.backward["cpu", BATCH](mb_grad_ao_t, mb_grad_obs_t)
        actor_opt.step["cpu", M=ActorNet](actor)

        # ── Alpha update + Polyak target soft-update ───────────────────
        var lp_mean: Scalar[DT] = 0.0
        for b in range(BATCH):
            lp_mean += mb_lp_s[b]
        lp_mean = lp_mean / Scalar[DT](BATCH)
        alpha_opt.step(-(lp_mean + TARGET_ENTROPY))
        alpha_accum += fexp(alpha_opt.value)
        pair1.polyak_step["cpu"](TAU)
        pair2.polyak_step["cpu"](TAU)
        update_count += 1

        # ── Periodic log ───────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            var ucf = Scalar[DT](1.0) / Scalar[DT](
                update_count if update_count > 0 else 1
            )
            print(
                "[step ", step, "] mean_ret(10)=", tracker.mean_return(),
                " ep=", tracker.ep_count, " alpha=", alpha_accum * ucf,
                " actor_L=", actor_L_accum * ucf,
                " critic_L=", critic_L_accum * ucf,
                " elapsed=", elapsed, "s",
            )
            actor_L_accum = 0.0; critic_L_accum = 0.0
            alpha_accum = 0.0; update_count = 0

    # ── Final report ───────────────────────────────────────────────────
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
