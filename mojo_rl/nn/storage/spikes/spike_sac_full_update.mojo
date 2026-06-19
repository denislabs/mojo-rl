"""Full coupled SAC update — Stage-5 de-risk gate (CPU).

The existing spikes prove the critic step (q→target_y fit) and the actor step
(lower α·logπ−min_q) IN ISOLATION with FROZEN critics. This spike wires the
COMPLETE SAC update together on a fixed synthetic batch — the assembly the real
storage SACTrainer will have — and asserts it is numerically stable:

  per step:
    1. target_y   : a',logp' ~ π(s') ; min over TARGET critics ; sac_target_y
    2. critic step : q_i = Q_i(s, a_replay) ; MSE(q_i, y) ; Adam(critic_i)
    3. actor step  : a~π(s) ; min over ONLINE critics ; loss=mean(α·logp−min_q) ;
                     route grad to actor params ; Adam(actor)   (critics frozen)
    4. alpha step  : grad = −(mean_logp + H_target) ; ScalarAdam(log_alpha)
    5. polyak      : target_i ← τ·online_i  (Module.polyak_from)

Networks reuse the proven spike topology: actor = ComputeGraph[1, trunk,
mu, ls, Concat2, RSample]; critics = ComputeGraph[2, Concat2, Linear, ReLU,
Linear]. Target critics start identical to the online twins (same Deterministic
init), then track via polyak.

This is NOT an MDP-convergence test (the batch is fixed) — it is the stability /
assembly gate: over many steps every quantity must stay finite, α must stay
positive, and the online↔target gap must stay bounded (polyak tracking).

Run: pixi run mojo run -I . mojo_rl/nn/storage/spikes/spike_sac_full_update.mojo
"""

from std.math import exp as fexp, log as flog, isnan, isinf

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.primitives.concat import Concat2
from mojo_rl.nn.storage.primitives.rsample import RSample
from mojo_rl.nn.storage.primitives.binary_elementwise import BinaryElemMin
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.optimizer.scalar_adam import ScalarAdam
from mojo_rl.nn.storage.loss.mse_loss import MSELoss
from mojo_rl.nn.storage.loss.sac import sac_target_y
from mojo_rl.nn.storage.core.initializer import Deterministic


comptime B = 16
comptime S = 3       # obs dim (Pendulum-like: cos, sin, thetadot)
comptime A = 1       # action dim
comptime H = 32
comptime SA = S + A
comptime AOUT = A + 1  # rsample packed [action | log_prob]


def _finite(t: Tensor, n: Int) raises -> Bool:
    for i in range(n):
        var v = t.data[i]
        if isnan(v) or isinf(v):
            return False
    return True


def main() raises:
    comptime GAMMA = Scalar[DT](0.99)
    comptime TAU = Scalar[DT](0.005)
    comptime H_TARGET = Scalar[DT](-Float64(A))  # -ACT heuristic

    # ── actor: obs → trunk → {mu, ls} → concat → rsample ───────────────
    var actor = ComputeGraph[
        1,
        Sequential[Linear[S, H], ReLU[H]],
        Linear[H, A],
        Linear[H, A],
        Concat2[A, A],
        RSample[A],
    ].make["cpu", Deterministic]()
    var a_edges = List[List[Int]]()
    a_edges.append([0])
    a_edges.append([1])
    a_edges.append([1])
    a_edges.append([2, 3])
    a_edges.append([4])

    # ── twin online + target critics (same init → identical at start) ──
    comptime CriticG = ComputeGraph[
        2, Concat2[S, A], Linear[SA, H], ReLU[H], Linear[H, 1]
    ]
    var c1 = CriticG.make["cpu", Deterministic]()
    var c2 = CriticG.make["cpu", Deterministic]()
    var c1t = CriticG.make["cpu", Deterministic]()
    var c2t = CriticG.make["cpu", Deterministic]()
    # perturb c2 (and its target) so min(q1,q2) is non-trivial
    var cap = Scalar[DT](0.05)
    for i in range(SA * H):
        c2.children[1].weight.val.data[i] += cap
        c2t.children[1].weight.val.data[i] += cap
    var c_edges = List[List[Int]]()
    c_edges.append([0, 1])
    c_edges.append([2])
    c_edges.append([3])
    c_edges.append([4])

    var minop = BinaryElemMin[1].make["cpu", Deterministic]()
    var mse = MSELoss[1].make_cpu()

    var actor_opt = Adam(lr=3e-4)
    var c1_opt = Adam(lr=1e-3)
    var c2_opt = Adam(lr=1e-3)
    var alpha_opt = ScalarAdam.new(flog(Scalar[DT](0.2)), Scalar[DT](3e-4))

    # ── fixed synthetic batch (s, a_replay, r, s', done) ───────────────
    var s_pack = TensorPack[1](); s_pack[0].ensure(B * S)
    var sp_pack = TensorPack[1](); sp_pack[0].ensure(B * S)
    var a_replay = Tensor.alloc(B * A)
    var r = Tensor.alloc(B)
    var done = Tensor.alloc(B)
    for i in range(B * S):
        s_pack[0].data[i] = Scalar[DT]((i % 7) - 3) * 0.2
        sp_pack[0].data[i] = Scalar[DT]((i % 5) - 2) * 0.25
    for i in range(B * A):
        a_replay.data[i] = Scalar[DT]((i % 3) - 1) * 0.5
    for b in range(B):
        r.data[b] = Scalar[DT]((b % 4) - 2) * 0.3
        done.data[b] = Scalar[DT](1.0) if (b % 8 == 7) else Scalar[DT](0.0)

    # ── working tensors ────────────────────────────────────────────────
    var nact = Tensor.alloc(B * AOUT)          # next-state actor out
    var next_action = TensorPack[1](); next_action[0].ensure(B * A)
    var next_logp = Tensor.alloc(B)
    var min_qt = Tensor.alloc(B)
    var y = Tensor.alloc(B)
    var qt = TensorPack[2](); qt[0].ensure(B); qt[1].ensure(B)

    var sp_critic_in = TensorPack[2]()         # (s', a') for target critics
    sp_critic_in[0].ensure(B * S); sp_critic_in[1].ensure(B * A)
    var s_critic_in = TensorPack[2]()          # (s, a_replay) for online critics
    s_critic_in[0].ensure(B * S); s_critic_in[1].ensure(B * A)
    for i in range(B * S):
        s_critic_in[0].data[i] = s_pack[0].data[i]
    for i in range(B * A):
        s_critic_in[1].data[i] = a_replay.data[i]

    var q1 = Tensor.alloc(B); var q2 = Tensor.alloc(B)
    var grad_q = Tensor.alloc(B)
    var cgin = TensorPack[2]()

    # actor-step working tensors
    var act_out = Tensor.alloc(B * AOUT)
    var pact = TensorPack[1](); pact[0].ensure(B * A)
    var aq = TensorPack[2](); aq[0].ensure(B); aq[1].ensure(B)
    var minq = Tensor.alloc(B)
    var grad_min = Tensor.alloc(B)
    var gq = TensorPack[2]()
    var ga0 = Tensor.alloc(B * A)
    var grad_actout = Tensor.alloc(B * AOUT)
    var actor_gin = TensorPack[1]()
    var actor_cin = TensorPack[2]()            # (s, a~π) for online critics
    actor_cin[0].ensure(B * S); actor_cin[1].ensure(B * A)
    for i in range(B * S):
        actor_cin[0].data[i] = s_pack[0].data[i]

    comptime STEPS = 500
    var ok = True
    var last_critic: Scalar[DT] = 0
    var last_actor: Scalar[DT] = 0
    var max_gap: Scalar[DT] = 0

    for step in range(STEPS):
        var alpha = fexp(alpha_opt.value)

        # ── 1. target_y (detached: no grad through actor/target critics) ─
        actor.forward[B](a_edges, sp_pack, nact)
        for b in range(B):
            for j in range(A):
                next_action[0].data[b * A + j] = nact.data[b * AOUT + j]
            next_logp.data[b] = nact.data[b * AOUT + A]
        for i in range(B * S):
            sp_critic_in[0].data[i] = sp_pack[0].data[i]
        for i in range(B * A):
            sp_critic_in[1].data[i] = next_action[0].data[i]
        c1t.forward[B](c_edges, sp_critic_in, qt[0])
        c2t.forward[B](c_edges, sp_critic_in, qt[1])
        minop.forward["cpu", B](TensorRefs[2](qt[0], qt[1]), min_qt)
        sac_target_y["cpu", B](r, done, min_qt, next_logp, GAMMA, alpha, y, None)

        # ── 2. critic step (replay action) ──────────────────────────────
        c1.forward[B](c_edges, s_critic_in, q1)
        c2.forward[B](c_edges, s_critic_in, q2)
        var cl1 = mse.forward["cpu", B](q1, y, None)
        var cl2 = mse.forward["cpu", B](q2, y, None)
        last_critic = cl1 + cl2
        mse.vjp["cpu", B](q1, y, grad_q, None)
        c1.zero_grad["cpu"](None); c1.vjp[B](c_edges, grad_q, cgin)
        c1_opt.begin_step(); c1.for_each_param["cpu"](c1_opt, None)
        mse.vjp["cpu", B](q2, y, grad_q, None)
        c2.zero_grad["cpu"](None); c2.vjp[B](c_edges, grad_q, cgin)
        c2_opt.begin_step(); c2.for_each_param["cpu"](c2_opt, None)

        # ── 3. actor step (fresh policy action, online critics frozen) ──
        actor.forward[B](a_edges, s_pack, act_out)
        for i in range(B * S):
            actor_cin[0].data[i] = s_pack[0].data[i]
        for b in range(B):
            for j in range(A):
                actor_cin[1].data[b * A + j] = act_out.data[b * AOUT + j]
        c1.forward[B](c_edges, actor_cin, aq[0])
        c2.forward[B](c_edges, actor_cin, aq[1])
        minop.forward["cpu", B](TensorRefs[2](aq[0], aq[1]), minq)
        var aloss: Scalar[DT] = 0
        var logp_sum: Scalar[DT] = 0
        for b in range(B):
            var lp = act_out.data[b * AOUT + A]
            aloss += alpha * lp - minq.data[b]
            logp_sum += lp
        aloss = aloss / Scalar[DT](B)
        last_actor = aloss
        var logp_mean = logp_sum / Scalar[DT](B)
        # route grad: grad_min=-1/B → min.vjp → critics.vjp → grad wrt action
        for b in range(B):
            grad_min.data[b] = Scalar[DT](-1.0) / Scalar[DT](B)
        minop.vjp["cpu", B](
            TensorRefs[2](aq[0], aq[1]), grad_min, TensorRefs[2](gq[0], gq[1])
        )
        c1.zero_grad["cpu"](None); c2.zero_grad["cpu"](None)
        c1.vjp[B](c_edges, gq[0], cgin)
        for i in range(B * A):
            ga0.data[i] = cgin[1].data[i]
        c2.vjp[B](c_edges, gq[1], cgin)
        for b in range(B):
            for j in range(A):
                grad_actout.data[b * AOUT + j] = (
                    ga0.data[b * A + j] + cgin[1].data[b * A + j]
                )
            grad_actout.data[b * AOUT + A] = alpha / Scalar[DT](B)
        actor.zero_grad["cpu"](None)
        actor.vjp[B](a_edges, grad_actout, actor_gin)
        actor_opt.begin_step(); actor.for_each_param["cpu"](actor_opt, None)

        # ── 4. alpha step ───────────────────────────────────────────────
        var alpha_grad = -(logp_mean + H_TARGET)
        alpha_opt.step(alpha_grad)

        # ── 5. polyak: target ← online ──────────────────────────────────
        c1t.polyak_from["cpu"](c1, TAU, None)
        c2t.polyak_from["cpu"](c2, TAU, None)

        # ── stability checks ────────────────────────────────────────────
        if not (_finite(q1, B) and _finite(q2, B) and _finite(y, B)
                and _finite(act_out, B * AOUT)):
            ok = False
            print("  NON-FINITE at step", step)
            break
        if isnan(alpha) or isinf(alpha) or alpha <= Scalar[DT](0.0):
            ok = False
            print("  BAD alpha at step", step, ":", alpha)
            break
        # online↔target gap (first online/target critic, layer-1 weights)
        var gap: Scalar[DT] = 0
        for i in range(SA * H):
            var dlt = c1.children[1].weight.val.data[i] - (
                c1t.children[1].weight.val.data[i]
            )
            var ad = dlt if dlt >= Scalar[DT](0.0) else -dlt
            if ad > gap:
                gap = ad
        if gap > max_gap:
            max_gap = gap
        if step % 100 == 0:
            print(
                "step", step, " critic_loss", last_critic,
                " actor_loss", last_actor, " alpha", alpha, " gap", gap,
            )

    print("final critic_loss", last_critic, " actor_loss", last_actor)
    print("final alpha", fexp(alpha_opt.value), " max online-target gap", max_gap)
    if ok and max_gap < Scalar[DT](5.0):
        print("FULL SAC UPDATE OK — coupled loop stable, polyak tracks, alpha>0")
    else:
        print("FULL SAC UPDATE FAILED (ok=", ok, " max_gap=", max_gap, ")")
