"""SAC actor step + twin critic — capstone assembly gate (CPU).

Actor = ComputeGraph[1, Sequential[Linear,ReLU] (trunk), Linear (mu), Linear (ls),
        Concat2(mu,ls), RSample]  →  [action | log_prob]   (fan-out: trunk feeds
        both heads — the Parallel combinator we DON'T need).
Twin critic = two ComputeGraph[2, Concat2(s,a), Linear, ReLU, Linear(q)]; the
        actor uses min(q1,q2) via BinaryElemMin.
Actor loss = mean_b( α·log_prob_b − min_q_b ); α fixed.

Backward routing (manual chain at the action boundary):
  grad_min = −1/B  →  BinaryElemMin.vjp → (gq1, gq2)
  critic_i.vjp(gq_i) → grad wrt its action input  →  sum = grad_action
  grad over rsample out = [grad_action | α/B]  →  actor.vjp → actor params.
Frozen critics (their grads are zeroed, not applied). Loss must fall as the
actor raises min_q while managing entropy.

Run: pixi run mojo run -I . mojo_rl/nn/storage/spikes/spike_sac_actor.mojo
"""

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


def main() raises:
    comptime B = 8
    comptime S = 4       # state/obs dim
    comptime A = 2       # action dim
    comptime H = 32
    comptime SA = S + A
    comptime AOUT = A + 1   # rsample packed [action | log_prob]
    comptime ALPHA = Scalar[DT](0.2)

    # ── actor: obs → trunk → {mu, ls} → concat → rsample ───────────────
    var actor = ComputeGraph[
        1,
        Sequential[Linear[S, H], ReLU[H]],   # node0 trunk
        Linear[H, A],                        # node1 mu head
        Linear[H, A],                        # node2 log_std head
        Concat2[A, A],                       # node3 [mu|ls]
        RSample[A],                          # node4 [action|log_prob]
    ].make_cpu()
    var a_edges = List[List[Int]]()
    a_edges.append([0])      # trunk(obs)
    a_edges.append([1])      # mu(trunk)
    a_edges.append([1])      # ls(trunk)  — fan-out
    a_edges.append([2, 3])   # concat(mu, ls)
    a_edges.append([4])      # rsample(concat)

    # ── two frozen critics (slightly perturbed so min is non-trivial) ──
    var c1 = ComputeGraph[2, Concat2[S, A], Linear[SA, H], ReLU[H], Linear[H, 1]].make_cpu()
    var c2 = ComputeGraph[2, Concat2[S, A], Linear[SA, H], ReLU[H], Linear[H, 1]].make_cpu()
    var c_edges = List[List[Int]]()
    c_edges.append([0, 1])
    c_edges.append([2])
    c_edges.append([3])
    c_edges.append([4])
    # perturb c2's first Linear weights so q1 != q2
    var cap = Scalar[DT](0.05)
    for i in range(SA * H):
        c2.children[1].weight.val.data[i] += cap

    var minop = BinaryElemMin[1].make_cpu()

    # obs batch (fixed).
    var obs = TensorPack[1]()
    obs[0].ensure(B * S)
    for i in range(B * S):
        obs[0].data[i] = Scalar[DT]((i % 7) - 3) * 0.2

    var act_out = Tensor.alloc(B * AOUT)
    var action = Tensor.alloc(B * A)
    var qp = TensorPack[2]()   # q1, q2 share one origin (§B0: min's inputs)
    qp[0].ensure(B * 1)
    qp[1].ensure(B * 1)
    var minq = Tensor.alloc(B * 1)
    var grad_min = Tensor.alloc(B * 1)
    var gq = TensorPack[2]()      # min vjp → (gq1, gq2)
    var grad_actout = Tensor.alloc(B * AOUT)
    var actor_gin = TensorPack[1]()
    var c_in = TensorPack[2]()    # critic inputs {obs, action}
    c_in[0].ensure(B * S)
    c_in[1].ensure(B * A)
    var c_gin = TensorPack[2]()   # critic vjp grad_inputs

    var opt = Adam(lr=0.01)

    comptime STEPS = 200
    comptime WIN = 25
    var first_sum: Scalar[DT] = 0   # mean over first WIN steps
    var last_sum: Scalar[DT] = 0    # mean over last WIN steps
    for step in range(STEPS):
        actor.zero_grad["cpu"](None)
        # forward actor → [action | log_prob]
        actor.forward[B](a_edges, obs, act_out)
        for b in range(B):
            for j in range(A):
                action.data[b * A + j] = act_out.data[b * AOUT + j]
        # critic forward (s = obs, a = action) → q1, q2 → min
        for i in range(B * S):
            c_in[0].data[i] = obs[0].data[i]
        for i in range(B * A):
            c_in[1].ensure(B * A)
            c_in[1].data[i] = action.data[i]
        c1.forward[B](c_edges, c_in, qp[0])
        c2.forward[B](c_edges, c_in, qp[1])
        minop.forward["cpu", B](TensorRefs[2].of2(qp[0], qp[1]), minq)
        var loss: Scalar[DT] = 0.0
        for b in range(B):
            loss += ALPHA * act_out.data[b * AOUT + A] - minq.data[b]
        loss = loss / Scalar[DT](B)
        if step < WIN:
            first_sum += loss
        if step >= STEPS - WIN:
            last_sum += loss
        if step % 40 == 0:
            print("step", step, " actor_loss", loss)
        # backward: grad_min = -1/B → split to critics → grad wrt action
        for b in range(B):
            grad_min.data[b] = Scalar[DT](-1.0) / Scalar[DT](B)
        minop.vjp["cpu", B](TensorRefs[2].of2(qp[0], qp[1]), grad_min, TensorRefs[2].of2(gq[0], gq[1]))
        c1.zero_grad["cpu"](None)
        c2.zero_grad["cpu"](None)
        c1.vjp[B](c_edges, gq[0], c_gin)
        var ga0 = Tensor.alloc(B * A)
        for i in range(B * A):
            ga0.data[i] = c_gin[1].data[i]
        c2.vjp[B](c_edges, gq[1], c_gin)
        # grad over rsample out: [grad_action | α/B]
        for b in range(B):
            for j in range(A):
                grad_actout.data[b * AOUT + j] = ga0.data[b * A + j] + c_gin[1].data[b * A + j]
            grad_actout.data[b * AOUT + A] = ALPHA / Scalar[DT](B)
        actor.vjp[B](a_edges, grad_actout, actor_gin)
        opt.begin_step()
        actor.for_each_param["cpu"](opt, None)

    var first_mean = first_sum / Scalar[DT](WIN)
    var last_mean = last_sum / Scalar[DT](WIN)
    print("windowed mean actor_loss:", first_mean, "->", last_mean)
    if last_mean < first_mean:
        print("SAC ACTOR STEP OK — actor lowers (α·logπ − min_q); loss", first_mean, "->", last_mean)
    else:
        print("SAC ACTOR STEP WEAK (", first_mean, "->", last_mean, ")")
