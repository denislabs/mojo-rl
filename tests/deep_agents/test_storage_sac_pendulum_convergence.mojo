"""SAC convergence on Pendulum-v1 — Stage-5 payoff gate (CPU).

Assembles EVERY migrated storage SAC block in a direct train loop on the real
Pendulum env and checks the policy actually learns: greedy-eval mean return must
improve from random (~-1200) to clearly-learning (< -400; solved ~= -169).
Validates the full pipeline end-to-end — TargetYBlock (ComputeGraph+ExternalRef),
TwinCriticStep, SACActorLoss (ComputeGraph+ExternalRef), AlphaUpdateStep,
PolyakStep, OnlineTargetPair, CPUReplay → TrainerState — composes and converges.

This is the algorithm gate; the driver-conforming SACTrainer assembly + GPU run
are the next step. Pendulum: OBS=3, ACT=1, torque in [-2,2] (action_scale=2).

Run: pixi run mojo run -I . tests/deep_agents/test_storage_sac_pendulum_convergence.mojo
"""

from std.math import exp as fexp, log as flog, tanh as ftanh
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU
from mojo_rl.nn.storage.primitives.rsample import RSample
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.core.initializer import Xavier, Zero
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.optimizer.scalar_adam import ScalarAdam

from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.core.online_target_pair import OnlineTargetPair
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.training.blocks.twin_critic_step import TwinCriticStep
from mojo_rl.deep_agents.training.blocks.polyak_step import PolyakStep
from mojo_rl.deep_agents.sac.target_y_block import TargetYBlock
from mojo_rl.deep_agents.sac.actor_loss import SACActorLoss
from mojo_rl.deep_agents.sac.blocks.alpha_update_step import AlphaUpdateStep
from mojo_rl.deep_agents.data.cpu_replay import CPUReplay

from mojo_rl.envs.pendulum.pendulum_v1 import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime SA = OBS + ACT
comptime H = 128
comptime BATCH = 128
comptime CAP = 100_000
comptime ASCALE = Scalar[DT](2.0)   # torque in [-2, 2]
comptime GAMMA = Scalar[DT](0.99)
comptime TAU = Scalar[DT](0.005)
comptime ACTOR = StochasticActor[OBS, ACT, LinearReLU[OBS, H], LinearReLU[H, H]]
comptime CRITIC = Sequential[LinearReLU[SA, H], LinearReLU[H, H], Linear[H, 1]]


def _greedy_eval(mut env: PendulumEnv[DT], mut actor: ACTOR, episodes: Int) raises -> Scalar[DT]:
    var ao = Tensor.alloc(2 * ACT)
    var ob = Tensor.alloc(OBS)
    var total: Scalar[DT] = 0
    for _ in range(episodes):
        var obs = env.reset_obs_list()
        var ep: Scalar[DT] = 0
        for _step in range(200):
            for d in range(OBS):
                ob.data[d] = obs[d]
            actor.forward["cpu", 1](TensorRefs[1](ob), ao)
            var a = ftanh(ao.data[0]) * ASCALE   # greedy = mean, squashed
            var r = env.step_continuous(a)
            ep += r[1]
            obs = r[0].copy()
            if r[2]:
                break
        total += ep
    return total / Scalar[DT](episodes)


def main() raises:
    seed(42)
    print("=" * 60)
    print("SAC Pendulum-v1 convergence (storage blocks, CPU)")
    print("=" * 60)

    var env = PendulumEnv[DT]()
    var actor = ACTOR.make["cpu", Xavier]()
    var pair1 = OnlineTargetPair[CRITIC].make["cpu", Xavier]()
    var pair2 = OnlineTargetPair[CRITIC].make["cpu", Xavier]()
    var actor_opt = Adam(lr=3e-4)
    var c1_opt = Adam(lr=1e-3)
    var c2_opt = Adam(lr=1e-3)
    var alpha_opt = ScalarAdam.new(flog(Scalar[DT](0.2)), Scalar[DT](3e-4))

    var ty = TargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT].make["cpu"](
        action_scale=ASCALE, gamma=GAMMA
    )
    var twin = TwinCriticStep[OBS, ACT, BATCH, CRITIC].make["cpu"]()
    var aloss = SACActorLoss[ACTOR, CRITIC, BATCH].make["cpu"](action_scale=ASCALE)
    var alpha_blk = AlphaUpdateStep[OBS, ACT, BATCH].make(Scalar[DT](-Float64(ACT)))
    var polyak = PolyakStep[OBS, ACT, BATCH, CRITIC].make(TAU)
    var replay = CPUReplay[OBS, ACT, CAP].make()
    var state = TrainerState[OBS, ACT, BATCH].make["cpu"]()

    # select-action rsample (separate from the loss graphs' own rsamples).
    var sel = RSample[ACT].make["cpu", Zero]()
    sel.action_scale = ASCALE
    var ob_t = Tensor.alloc(OBS)
    var ao_t = Tensor.alloc(2 * ACT)
    var alp_t = Tensor.alloc(ACT + 1)

    comptime TOTAL = 12_000
    comptime LEARN_START = 500
    var obs = env.reset_obs_list()
    var ep_ret: Scalar[DT] = 0
    var ep_count = 0
    var cl_acc: Scalar[DT] = 0
    var al_acc: Scalar[DT] = 0
    var n_upd = 0

    print("eval @0 (random):", _greedy_eval(env, actor, 5))
    obs = env.reset_obs_list()

    for step in range(TOTAL):
        # ── select action ──────────────────────────────────────────────
        var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
        if step < LEARN_START:
            for j in range(ACT):
                action[j] = Scalar[DT](2.0 * random_float64() - 1.0) * ASCALE
        else:
            for d in range(OBS):
                ob_t.data[d] = obs[d]
            actor.forward["cpu", 1](TensorRefs[1](ob_t), ao_t)
            sel.forward["cpu", 1](TensorRefs[1](ao_t), alp_t)
            for j in range(ACT):
                var a = alp_t.data[j]
                if a > ASCALE: a = ASCALE
                elif a < -ASCALE: a = -ASCALE
                action[j] = a

        # ── env step ────────────────────────────────────────────────────
        var res = env.step_continuous(action[0])
        var next_obs = res[0].copy()
        var reward = res[1]
        var trunc = res[2]   # Pendulum: time-limit truncation (no termination)
        ep_ret += reward
        # store with done=0 (Pendulum never terminates → keep bootstrap).
        replay.add(obs, action, reward, next_obs, Scalar[DT](0.0))
        obs = next_obs.copy()
        if trunc:
            obs = env.reset_obs_list()
            ep_count += 1
            ep_ret = 0

        # ── train ───────────────────────────────────────────────────────
        if step >= LEARN_START and replay.count() >= BATCH:
            state.step_idx = step
            state.alpha = fexp(alpha_opt.value)
            replay.sample_into[BATCH](state)
            ty.step["cpu"](state, actor, pair1.target_net, pair2.target_net)
            twin.step["cpu"](state, pair1.online, c1_opt, pair2.online, c2_opt)
            var out = aloss.forward_backward["cpu"](
                actor, actor_opt, pair1.online, pair2.online, state.mb_s,
                state.alpha,
            )
            state.log_prob_mean = out.log_prob_mean
            alpha_blk.step["cpu"](state, alpha_opt)
            polyak.step["cpu"](state, pair1, pair2)
            cl_acc += state.critic_loss
            al_acc += out.loss
            n_upd += 1

        if (step + 1) % 3000 == 0:
            var ev = _greedy_eval(env, actor, 5)
            var nd = Scalar[DT](n_upd) if n_upd > 0 else Scalar[DT](1)
            print("step", step + 1, " eval", ev,
                  " critic_loss", cl_acc / nd, " actor_loss", al_acc / nd,
                  " alpha", fexp(alpha_opt.value))
            cl_acc = 0; al_acc = 0; n_upd = 0
            obs = env.reset_obs_list()

    var final_eval = _greedy_eval(env, actor, 10)
    print("FINAL greedy eval(10):", final_eval)
    assert_true(
        final_eval > Scalar[DT](-400.0),
        "SAC learns Pendulum (eval return > -400; solved ~ -169)",
    )
    print("SAC PENDULUM CONVERGENCE OK")
