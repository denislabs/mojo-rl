"""SAC training on Pendulum V1 — actor loss in *composed-Module* form.

Phase 8.4 validating user. Same algorithm + hyperparameters + RNG seed as
`pendulum_sac_nn2.mojo`, but the SAC actor loss is now a chain of Module
primitives instead of the free-function `squashed_gaussian_sample` +
`sac_actor_backward` pair:

    L = mean_b ( α · log_prob_b - min(Q1(s,a), Q2(s,a)) )
                                              │
        ┌──────────────────────────────────────┴──┐
        │ Composed-form chain (per actor update): │
        ├──────────────────────────────────────────┤
        │ actor(s)            → ao = [mu | log_std]                       │
        │ rsample(ao)         → [action | log_prob]                       │
        │ split action / log_prob                                         │
        │ concat(s, action)   → sa                                        │
        │ critic1_sg(sa)      → q1     (StopGradParams)                   │
        │ critic2_sg(sa)      → q2     (StopGradParams)                   │
        │ pack [q1 | q2]      → q12                                       │
        │ elem_min(q12)       → min_q                                     │
        │ scale.multiplier=α                                              │
        │ scale(log_prob)     → α·log_prob                                │
        │ pack [α·log_prob | min_q] → packed_loss                         │
        │ sub(packed_loss)    → α·log_prob − min_q   (loss per batch)     │
        │ training-loop sum / BATCH → scalar L                            │
        └──────────────────────────────────────────────────────────────────┘

Backward seeds `grad_loss_per_b = 1/BATCH` and walks the chain in reverse
via each Module's `.backward`. Because the critics are wrapped in
`StopGradParams`, their grad_w / grad_b is not touched — only grad_input
flows through (Phase 8.2 contract). The end-of-chain `rsample.backward`
consumes packed `[grad_action | grad_log_prob]` and produces grad_ao,
which `actor.backward` walks into the actor parameter grads.

Phase 8.4 ships this CPU-only. Exit criterion: final mean ep return
within seed-noise of the free-function form (-170.26 EXCELLENT).

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_sac_nn2_composed.mojo
"""

from std.math import exp as fexp, log as flog, sqrt as fsqrt
from std.memory import alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.combinators.stop_grad_params import StopGradParams
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.primitives.rsample import RSample
from mojo_rl.nn2.primitives.scale import Scale
from mojo_rl.nn2.primitives.elem_min import ElemMin
from mojo_rl.nn2.primitives.sub import Sub
from mojo_rl.nn2.initializer import Xavier, Zero
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.mse import MSELoss
from mojo_rl.nn2.core.online_target_pair import OnlineTargetPair
from mojo_rl.nn2.data.cpu_replay import CPUReplay
from mojo_rl.nn2.random.box_muller import box_muller_normal
from mojo_rl.nn2.training.episode_tracker import EpisodeTracker
from mojo_rl.nn2.loss.sac_actor_loss import squashed_gaussian_sample

from mojo_rl.envs.pendulum import PendulumEnv


# ──────────────────────────────────────────────────────────────────────────
# Hyperparameters (identical to free-function form)
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
# SAC-specific helpers (data-side concat stays inline)
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


def _composed_actor_forward[BATCH: Int](
    mut actor: ActorNet,
    mut actor_opt: Adam,
    mut critic1: CriticNet,
    mut critic2: CriticNet,
    mut rsample: RSample[ACT_DIM],
    mut scale: Scale[1],
    mut elem_min: ElemMin[1],
    mut sub_op: Sub[1],
    alpha: Scalar[DT],
    mb_s: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_ao_s: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_alp_s: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_act_s: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_lp_s: UnsafePointer[Scalar[DT], MutAnyOrigin],
    lp_for_alpha: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_sa: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_q1: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_q2: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_q12: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_min_q: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_alpha_lp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_packed_loss: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_loss_per_b: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises -> Scalar[DT]:
    """Composed-form SAC actor FORWARD chain. Returns scalar loss for logging.

    Steps: zero_grad → actor.forward → rsample.forward → slice action/lp
        → concat_sa → critic1.forward → critic2.forward → pack [q1|q2]
        → elem_min → scale → pack [αlp|min_q] → sub → mean.
    """
    var mb_s_t = TileTensor(mb_s, row_major[BATCH, OBS_DIM]())
    var mb_ao_s_t = TileTensor(mb_ao_s, row_major[BATCH, 2 * ACT_DIM]())
    actor_opt.zero_grad["cpu", M=ActorNet](actor)
    actor.forward["cpu", BATCH](mb_s_t, mb_ao_s_t)

    var mb_alp_s_t = TileTensor(mb_alp_s, row_major[BATCH, ACT_DIM + 1]())
    rsample.forward["cpu", BATCH](mb_ao_s_t, mb_alp_s_t)

    for b in range(BATCH):
        for j in range(ACT_DIM):
            mb_act_s[b * ACT_DIM + j] = mb_alp_s[b * (ACT_DIM + 1) + j]
        mb_lp_s[b] = mb_alp_s[b * (ACT_DIM + 1) + ACT_DIM]
        lp_for_alpha[b] = mb_lp_s[b]

    _concat_sa[BATCH](mb_s, mb_act_s, mb_sa)
    var mb_sa_t = TileTensor(mb_sa, row_major[BATCH, SA_DIM]())
    var mb_q1_t = TileTensor(mb_q1, row_major[BATCH, 1]())
    var mb_q2_t = TileTensor(mb_q2, row_major[BATCH, 1]())
    critic1.forward["cpu", BATCH](mb_sa_t, mb_q1_t)
    critic2.forward["cpu", BATCH](mb_sa_t, mb_q2_t)

    for b in range(BATCH):
        mb_q12[b * 2 + 0] = mb_q1[b]
        mb_q12[b * 2 + 1] = mb_q2[b]
    var mb_q12_t = TileTensor(mb_q12, row_major[BATCH, 2]())
    var mb_min_q_t = TileTensor(mb_min_q, row_major[BATCH, 1]())
    elem_min.forward["cpu", BATCH](mb_q12_t, mb_min_q_t)

    scale.multiplier = alpha
    var mb_lp_s_t = TileTensor(mb_lp_s, row_major[BATCH, 1]())
    var mb_alpha_lp_t = TileTensor(mb_alpha_lp, row_major[BATCH, 1]())
    scale.forward["cpu", BATCH](mb_lp_s_t, mb_alpha_lp_t)

    for b in range(BATCH):
        mb_packed_loss[b * 2 + 0] = mb_alpha_lp[b]
        mb_packed_loss[b * 2 + 1] = mb_min_q[b]
    var mb_packed_loss_t = TileTensor(mb_packed_loss, row_major[BATCH, 2]())
    var mb_loss_per_b_t = TileTensor(mb_loss_per_b, row_major[BATCH, 1]())
    sub_op.forward["cpu", BATCH](mb_packed_loss_t, mb_loss_per_b_t)

    var loss_scalar: Scalar[DT] = 0.0
    for b in range(BATCH):
        loss_scalar += mb_loss_per_b[b]
    return loss_scalar / Scalar[DT](BATCH)


def _composed_actor_backward[BATCH: Int](
    mut actor: ActorNet,
    mut actor_opt: Adam,
    mut critic1: CriticNet,
    mut critic2: CriticNet,
    mut rsample: RSample[ACT_DIM],
    mut scale: Scale[1],
    mut elem_min: ElemMin[1],
    mut sub_op: Sub[1],
    mb_grad_loss_per_b: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_packed_loss: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_alpha_lp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_min_q: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_lp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_q12: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_q1: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_q2: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_sa1: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_sa2: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_action_sum: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_alp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_ao: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mb_grad_obs_unused: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    """Composed-form SAC actor BACKWARD chain + optimizer step.

    Seeds grad_loss_per_b = 1/BATCH and walks the chain in reverse.
    """
    var inv_batch: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    for b in range(BATCH):
        mb_grad_loss_per_b[b] = inv_batch
    var mb_grad_loss_per_b_t = TileTensor(
        mb_grad_loss_per_b, row_major[BATCH, 1]()
    )
    var mb_grad_packed_loss_t = TileTensor(
        mb_grad_packed_loss, row_major[BATCH, 2]()
    )
    sub_op.backward["cpu", BATCH](mb_grad_loss_per_b_t, mb_grad_packed_loss_t)

    for b in range(BATCH):
        mb_grad_alpha_lp[b] = mb_grad_packed_loss[b * 2 + 0]
        mb_grad_min_q[b] = mb_grad_packed_loss[b * 2 + 1]
    var mb_grad_alpha_lp_t = TileTensor(mb_grad_alpha_lp, row_major[BATCH, 1]())
    var mb_grad_lp_t = TileTensor(mb_grad_lp, row_major[BATCH, 1]())
    scale.backward["cpu", BATCH](mb_grad_alpha_lp_t, mb_grad_lp_t)

    var mb_grad_min_q_t = TileTensor(mb_grad_min_q, row_major[BATCH, 1]())
    var mb_grad_q12_t = TileTensor(mb_grad_q12, row_major[BATCH, 2]())
    elem_min.backward["cpu", BATCH](mb_grad_min_q_t, mb_grad_q12_t)

    for b in range(BATCH):
        mb_grad_q1[b] = mb_grad_q12[b * 2 + 0]
        mb_grad_q2[b] = mb_grad_q12[b * 2 + 1]
    var mb_grad_q1_t = TileTensor(mb_grad_q1, row_major[BATCH, 1]())
    var mb_grad_q2_t = TileTensor(mb_grad_q2, row_major[BATCH, 1]())
    var mb_grad_sa1_t = TileTensor(mb_grad_sa1, row_major[BATCH, SA_DIM]())
    var mb_grad_sa2_t = TileTensor(mb_grad_sa2, row_major[BATCH, SA_DIM]())
    critic1.backward_input["cpu", BATCH](mb_grad_q1_t, mb_grad_sa1_t)
    critic2.backward_input["cpu", BATCH](mb_grad_q2_t, mb_grad_sa2_t)

    for b in range(BATCH):
        for j in range(ACT_DIM):
            mb_grad_action_sum[b * ACT_DIM + j] = (
                mb_grad_sa1[b * SA_DIM + OBS_DIM + j]
                + mb_grad_sa2[b * SA_DIM + OBS_DIM + j]
            )

    for b in range(BATCH):
        for j in range(ACT_DIM):
            mb_grad_alp[b * (ACT_DIM + 1) + j] = (
                mb_grad_action_sum[b * ACT_DIM + j]
            )
        mb_grad_alp[b * (ACT_DIM + 1) + ACT_DIM] = mb_grad_lp[b]

    var mb_grad_alp_t = TileTensor(mb_grad_alp, row_major[BATCH, ACT_DIM + 1]())
    var mb_grad_ao_t = TileTensor(mb_grad_ao, row_major[BATCH, 2 * ACT_DIM]())
    rsample.backward["cpu", BATCH](mb_grad_alp_t, mb_grad_ao_t)

    var mb_grad_obs_t = TileTensor(
        mb_grad_obs_unused, row_major[BATCH, OBS_DIM]()
    )
    actor.backward["cpu", BATCH](mb_grad_ao_t, mb_grad_obs_t)
    actor_opt.step["cpu", M=ActorNet](actor)


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
    print("nn2 SAC Continuous (COMPOSED-FORM actor loss) — Pendulum V1 (CPU)")
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

    # ── Nets + optimizers (unchanged from free-function form) ─────────
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

    # ── Phase 8.4 composed-form actor-loss Modules ────────────────────
    # rsample owns its z_cache; scale carries α; elem_min owns the min-mask;
    # sub is stateless. All persistent across actor updates.
    var rsample = RSample[ACT_DIM].make[target="cpu", INIT=Zero]()
    rsample.action_scale = ACTION_SCALE
    var scale = Scale[1].make[target="cpu", INIT=Zero]()
    var elem_min = ElemMin[1].make[target="cpu", INIT=Zero]()
    var sub_op = Sub[1].make[target="cpu", INIT=Zero]()
    # Single-sample rsample for env-interaction inference (uses BATCH=1).
    # We reuse the same RSample struct; its cache grows lazily on first
    # BATCH=1 forward, then on first BATCH=256 forward.

    # ── Replay + tracker (unchanged) ──────────────────────────────────
    var buf = CPUReplay[OBS_DIM, ACT_DIM, REPLAY_CAPACITY].new()
    var tracker = EpisodeTracker.new(
        window_size=10, initial_fill=Scalar[DT](-1250.0)
    )

    # ── Scratch (single-step env interaction) ─────────────────────────
    var ob1 = alloc[Scalar[DT]](OBS_DIM)
    var ao1 = alloc[Scalar[DT]](2 * ACT_DIM)
    var alp1 = alloc[Scalar[DT]](ACT_DIM + 1)   # packed [action | log_prob]

    # ── Scratch (minibatch) ───────────────────────────────────────────
    var mb_s = alloc[Scalar[DT]](BATCH * OBS_DIM)
    var mb_a = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_r = alloc[Scalar[DT]](BATCH)
    var mb_sp = alloc[Scalar[DT]](BATCH * OBS_DIM)
    var mb_d = alloc[Scalar[DT]](BATCH)
    var mb_sa = alloc[Scalar[DT]](BATCH * SA_DIM)

    var mb_ao_s = alloc[Scalar[DT]](BATCH * 2 * ACT_DIM)
    var mb_alp_s = alloc[Scalar[DT]](BATCH * (ACT_DIM + 1))   # [action | log_prob]
    var mb_act_s = alloc[Scalar[DT]](BATCH * ACT_DIM)         # extracted action
    var mb_lp_s = alloc[Scalar[DT]](BATCH * 1)                # extracted log_prob

    # Target-side: still uses free-function squashed_gaussian_sample for
    # the y target (no gradient flows here, no need for Module form).
    var mb_ao_sp = alloc[Scalar[DT]](BATCH * 2 * ACT_DIM)
    var mb_z_sp = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_alp_sp = alloc[Scalar[DT]](BATCH * (ACT_DIM + 1))

    var mb_q1 = alloc[Scalar[DT]](BATCH * 1)
    var mb_q2 = alloc[Scalar[DT]](BATCH * 1)
    var mb_q1_tgt = alloc[Scalar[DT]](BATCH * 1)
    var mb_q2_tgt = alloc[Scalar[DT]](BATCH * 1)
    var mb_y = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_q1 = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_q2 = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_sa1 = alloc[Scalar[DT]](BATCH * SA_DIM)
    var mb_grad_sa2 = alloc[Scalar[DT]](BATCH * SA_DIM)

    # Phase 8.4 composed-form scratch.
    var mb_q12 = alloc[Scalar[DT]](BATCH * 2)            # packed [q1 | q2]
    var mb_min_q = alloc[Scalar[DT]](BATCH * 1)
    var mb_alpha_lp = alloc[Scalar[DT]](BATCH * 1)
    var mb_packed_loss = alloc[Scalar[DT]](BATCH * 2)    # [α·lp | min_q]
    var mb_loss_per_b = alloc[Scalar[DT]](BATCH * 1)

    var mb_grad_loss_per_b = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_packed_loss = alloc[Scalar[DT]](BATCH * 2)
    var mb_grad_alpha_lp = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_min_q = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_lp = alloc[Scalar[DT]](BATCH * 1)
    var mb_grad_q12 = alloc[Scalar[DT]](BATCH * 2)
    var mb_grad_action_sum = alloc[Scalar[DT]](BATCH * ACT_DIM)
    var mb_grad_alp = alloc[Scalar[DT]](BATCH * (ACT_DIM + 1))  # [grad_action | grad_lp]
    var mb_grad_ao = alloc[Scalar[DT]](BATCH * 2 * ACT_DIM)
    var mb_grad_obs_unused = alloc[Scalar[DT]](BATCH * OBS_DIM)
    var lp_for_alpha = alloc[Scalar[DT]](BATCH * 1)

    # Single-transition scratch.
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
        # ── Act ──────────────────────────────────────────────────────
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
            rsample.forward["cpu", 1](ao1_t, alp1_t)
            torque = alp1[0]      # action[0] from packed [action | log_prob]
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

        # ── Critic target y = r + γ·(min Q' − α·log_prob') ─────────
        var mb_sp_t = TileTensor(mb_sp, row_major[BATCH, OBS_DIM]())
        var mb_ao_sp_t = TileTensor(mb_ao_sp, row_major[BATCH, 2 * ACT_DIM]())
        actor.forward["cpu", BATCH](mb_sp_t, mb_ao_sp_t)
        box_muller_normal(mb_z_sp, BATCH * ACT_DIM)
        # Use free-function squashed_gaussian_sample for the target (no grad).
        # We could equally use a *second* RSample Module here, but for
        # parity with `pendulum_sac_nn2.mojo` and to keep this example
        # focused on the actor loss, we keep this path inline.
        var mb_z_sp_t = TileTensor(mb_z_sp, row_major[BATCH, ACT_DIM]())
        var mb_alp_sp_t = TileTensor(mb_alp_sp, row_major[BATCH, ACT_DIM + 1]())
        # squashed_gaussian_sample wants [BATCH, ACT] action + [BATCH] log_prob.
        # Reuse mb_alp_sp as the action buffer and write log_prob into a
        # separate sub-buffer.
        var mb_act_sp_buf = alloc[Scalar[DT]](BATCH * ACT_DIM)
        var mb_lp_sp_buf = alloc[Scalar[DT]](BATCH)
        var mb_act_sp_t = TileTensor(mb_act_sp_buf, row_major[BATCH, ACT_DIM]())
        var mb_lp_sp_t = TileTensor(mb_lp_sp_buf, row_major[BATCH]())
        squashed_gaussian_sample[ACT_DIM, BATCH](
            mb_ao_sp_t, mb_z_sp_t, ACTION_SCALE, mb_act_sp_t, mb_lp_sp_t
        )
        _concat_sa[BATCH](mb_sp, mb_act_sp_buf, mb_sa)
        var mb_sa_t = TileTensor(mb_sa, row_major[BATCH, SA_DIM]())
        var mb_q1_tgt_t = TileTensor(mb_q1_tgt, row_major[BATCH, 1]())
        var mb_q2_tgt_t = TileTensor(mb_q2_tgt, row_major[BATCH, 1]())
        pair1.target_net.forward["cpu", BATCH](mb_sa_t, mb_q1_tgt_t)
        pair2.target_net.forward["cpu", BATCH](mb_sa_t, mb_q2_tgt_t)
        for b in range(BATCH):
            var qmin = mb_q1_tgt[b] if mb_q1_tgt[b] < mb_q2_tgt[b] else mb_q2_tgt[b]
            var nonterm: Scalar[DT] = 1.0   # Pendulum truncation; always bootstrap
            mb_y[b] = mb_r[b] + GAMMA * nonterm * (qmin - alpha * mb_lp_sp_buf[b])
        mb_act_sp_buf.free()
        mb_lp_sp_buf.free()

        # ── Critic update (identical to free-function form) ────────
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

        # ── COMPOSED-FORM actor update (extracted helpers) ──────────
        # See feedback_lewm_eval_block_compile_explosion / feedback_mojo_
        # function_inline_call_explosion — the ~25-step composed chain
        # exceeds Mojo nightly's ~20-sequential-call inlining threshold
        # when kept inline; the helpers below stay well under it.
        var loss_scalar = _composed_actor_forward[BATCH](
            actor, actor_opt, pair1.online, pair2.online,
            rsample, scale, elem_min, sub_op, alpha,
            mb_s, mb_ao_s, mb_alp_s, mb_act_s, mb_lp_s, lp_for_alpha,
            mb_sa, mb_q1, mb_q2,
            mb_q12, mb_min_q, mb_alpha_lp, mb_packed_loss, mb_loss_per_b,
        )
        actor_L_accum += loss_scalar

        _composed_actor_backward[BATCH](
            actor, actor_opt, pair1.online, pair2.online,
            rsample, scale, elem_min, sub_op,
            mb_grad_loss_per_b, mb_grad_packed_loss,
            mb_grad_alpha_lp, mb_grad_min_q, mb_grad_lp,
            mb_grad_q12, mb_grad_q1, mb_grad_q2,
            mb_grad_sa1, mb_grad_sa2, mb_grad_action_sum,
            mb_grad_alp, mb_grad_ao, mb_grad_obs_unused,
        )

        # ── Alpha update + Polyak ─────────────────────────────────
        var lp_mean: Scalar[DT] = 0.0
        for b in range(BATCH):
            lp_mean += lp_for_alpha[b]
        lp_mean = lp_mean / Scalar[DT](BATCH)
        alpha_opt.step(-(lp_mean + TARGET_ENTROPY))
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
    print("Training complete (COMPOSED-FORM actor loss).")
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
