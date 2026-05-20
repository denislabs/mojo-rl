"""SAC training on Pendulum V1 — composed-form actor loss via `SACActorLoss`.

Phase 9A validating user. Same algorithm + hyperparameters + RNG seed as
`pendulum_sac_nn2.mojo`, but the SAC actor update is now a single call
into `SACActorLoss.forward_backward(...)` — the block hides:

  - the chain Modules (rsample, scale, elem_min, sub),
  - all intermediate forward/backward scratch (~26 buffers),
  - the gradient seed (1/BATCH),
  - the actor optimizer zero_grad + step,
  - the per-sample log_prob → mean reduction for the α update.

This is the abstraction Phase 8.4 set up but did not actually deliver —
the original composed validator was ~626 LOC, *more* than the free-
function form. With the loss-block in place this file lands under
the free-function form's ~469 LOC.

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_sac_nn2_composed.mojo

Expected: mean ep return (last 10) lands in the -170 range (same seed=42
result as the Phase 8.3 free-function form and the Phase 8.4 inline
composed form — bit-identical chain).
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
from mojo_rl.nn2.loss.sac_actor_loss import squashed_gaussian_sample
from mojo_rl.nn2.loss.sac_actor_loss_block import SACActorLoss
from mojo_rl.nn2.core.online_target_pair import OnlineTargetPair
from mojo_rl.nn2.data.cpu_replay import CPUReplay
from mojo_rl.nn2.random.box_muller import box_muller_normal
from mojo_rl.nn2.training.episode_tracker import EpisodeTracker

from mojo_rl.envs.pendulum import PendulumEnv


# ──────────────────────────────────────────────────────────────────────────
# Hyperparameters (identical to free-function form for bit-level parity).
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
# SAC-specific helpers — only the *data*-side concat (replay action,
# from a different source than actor π(s)) remains inline.
# ──────────────────────────────────────────────────────────────────────────


def _concat_sa[B: Int](
    obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
    act: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_sa: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    for b in range(B):
        for d in range(OBS_DIM):
            out_sa[b * SA_DIM + d] = obs[b * OBS_DIM + d]
        for j in range(ACT_DIM):
            out_sa[b * SA_DIM + OBS_DIM + j] = act[b * ACT_DIM + j]


@fieldwise_init
struct ScalarAdam(Movable & ImplicitlyDestructible):
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
    print("nn2 SAC Continuous (Phase 9A loss-block) — Pendulum V1 (CPU)")
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

    # ── Nets + optimizers ─────────────────────────────────────────────
    var actor = ActorNet.make[target="cpu", INIT=Xavier]()
    var pair1 = OnlineTargetPair[CriticNet].make[target="cpu", INIT=Xavier]()
    var pair2 = OnlineTargetPair[CriticNet].make[target="cpu", INIT=Xavier]()

    var actor_opt = Adam.make[target="cpu", M=ActorNet](actor, lr=ACTOR_LR)
    var critic1_opt = Adam.make[target="cpu", M=CriticNet](
        pair1.online, lr=CRITIC_LR
    )
    var critic2_opt = Adam.make[target="cpu", M=CriticNet](
        pair2.online, lr=CRITIC_LR
    )
    var alpha_opt = ScalarAdam.new(flog(INIT_ALPHA), ALPHA_LR)
    var mse_loss = MSELoss[1].make["cpu"]()

    # ── Phase 9A: the composed-form actor loss as a single block.
    # Owns chain Modules + scratch + grad seed + opt step internally.
    var actor_loss = SACActorLoss[ActorNet, CriticNet, BATCH].make["cpu"](
        action_scale=ACTION_SCALE
    )

    # ── Replay + tracker ──────────────────────────────────────────────
    var buf = CPUReplay[OBS_DIM, ACT_DIM, REPLAY_CAPACITY].new()
    var tracker = EpisodeTracker.new(
        window_size=10, initial_fill=Scalar[DT](-1250.0)
    )

    # ── Scratch (single-step env interaction; shares actor_loss.rsample) ─
    var ob1 = alloc[Scalar[DT]](OBS_DIM)
    var ao1 = alloc[Scalar[DT]](2 * ACT_DIM)
    var alp1 = alloc[Scalar[DT]](ACT_DIM + 1)

    # ── Scratch (critic update minibatch — actor update uses block) ───
    var mb_s = alloc[Scalar[DT]](BATCH * OBS_DIM)
    var mb_a = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_r = alloc[Scalar[DT]](BATCH)
    var mb_sp = alloc[Scalar[DT]](BATCH * OBS_DIM)
    var mb_d = alloc[Scalar[DT]](BATCH)
    var mb_sa = alloc[Scalar[DT]](BATCH * SA_DIM)

    # Target-side: free-function squashed_gaussian_sample for the y target
    # (no gradient flows here; the loss block is only for the actor update).
    var mb_ao_sp = alloc[Scalar[DT]](BATCH * 2 * ACT_DIM)
    var mb_z_sp = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_act_sp = alloc[Scalar[DT]](BATCH * ACT_DIM)
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

    var act_scratch = alloc[Scalar[DT]](ACT_DIM)
    var sp_scratch = alloc[Scalar[DT]](OBS_DIM)

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
        # ── Act (single-step π via actor.forward + block.rsample) ─────
        for d in range(OBS_DIM):
            ob1[d] = obs_self[d]

        var torque: Scalar[DT] = 0.0
        if step < LEARNING_STARTS:
            torque = Scalar[DT](2.0 * random_float64() - 1.0) * ACTION_SCALE
        else:
            var ob1_t = TileTensor(ob1, row_major[1, OBS_DIM]())
            var ao1_t = TileTensor(ao1, row_major[1, 2 * ACT_DIM]())
            actor.forward["cpu", 1](ob1_t, ao1_t)
            var alp1_t = TileTensor(alp1, row_major[1, ACT_DIM + 1]())
            actor_loss.rsample.forward["cpu", 1](ao1_t, alp1_t)
            torque = alp1[0]
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

        # ── Train ────────────────────────────────────────────────────
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

        # ── Critic target y = r + γ·(min Q' − α·log_prob') ──────────
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
            var nonterm: Scalar[DT] = 1.0   # Pendulum truncation: always bootstrap
            mb_y[b] = mb_r[b] + GAMMA * nonterm * (qmin - alpha * mb_lp_sp[b])

        # ── Critic update (unchanged) ───────────────────────────────
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

        # ── Actor update — single block call ─────────────────────────
        var actor_res = actor_loss.forward_backward["cpu", OPT=Adam](
            actor, actor_opt, pair1.online, pair2.online, mb_s, alpha,
        )
        actor_L_accum += actor_res.loss

        # ── α update + Polyak ───────────────────────────────────────
        alpha_opt.step(-(actor_res.log_prob_mean + TARGET_ENTROPY))
        alpha_accum += fexp(alpha_opt.value)
        pair1.polyak_step["cpu"](TAU)
        pair2.polyak_step["cpu"](TAU)
        update_count += 1

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

    print("=" * 70)
    print("Training complete (Phase 9A loss-block).")
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
