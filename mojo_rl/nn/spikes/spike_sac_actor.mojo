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
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.concat import Concat2
from mojo_rl.nn.primitives.rsample import RSample
from mojo_rl.nn.primitives.binary_elementwise import BinaryElemMin
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import InputSlot, Node
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.initializer import Deterministic


def main() raises:
    comptime B = 8
    comptime S = 4  # state/obs dim
    comptime A = 2  # action dim
    comptime H = 32
    comptime SA = S + A
    comptime AOUT = A + 1  # rsample packed [action | log_prob]
    comptime ALPHA = Scalar[DT](0.2)

    # ── actor: obs → trunk → {mu, ls} → concat → rsample ───────────────
    var actor = ComputeGraph[
        InputSlot["obs", S],
        Node["trunk", Sequential[Linear[S, H], ReLU[H]], "obs"],  # trunk
        Node["mu", Linear[H, A], "trunk"],          # mu head
        Node["ls", Linear[H, A], "trunk"],          # log_std head (fan-out)
        Node["concat", Concat2[A, A], "mu", "ls"],  # [mu|ls]
        Node["rsample", RSample[A], "concat"],      # [action|log_prob]
    ].make["cpu", Deterministic]()

    # ── two frozen critics (slightly perturbed so min is non-trivial) ──
    comptime CriticG = ComputeGraph[
        InputSlot["s", S],
        InputSlot["a", A],
        Node["concat", Concat2[S, A], "s", "a"],
        Node["l1", Linear[SA, H], "concat"],   # children[3]
        Node["relu", ReLU[H], "l1"],
        Node["q", Linear[H, 1], "relu"],
    ]
    var c1 = CriticG.make["cpu", Deterministic]()
    var c2 = CriticG.make["cpu", Deterministic]()
    # perturb c2's first Linear weights so q1 != q2 (children[3].op = l1 Linear)
    var cap = Scalar[DT](0.05)
    for i in range(SA * H):
        c2.children[3].op.weight.val.data[i] += cap

    var minop = BinaryElemMin[1].make["cpu", Deterministic]()

    # obs batch (fixed).
    var obs = Tensor.alloc(B * S)
    for i in range(B * S):
        obs.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    actor.set_input["obs", B](obs)

    var act_out = Tensor.alloc(B * AOUT)
    var action = Tensor.alloc(B * A)
    var qp = TensorPack[2]()  # q1, q2 share one origin (§B0: min's inputs)
    qp[0].ensure(B * 1)
    qp[1].ensure(B * 1)
    var minq = Tensor.alloc(B * 1)
    var grad_min = Tensor.alloc(B * 1)
    var gq = TensorPack[2]()  # min vjp → (gq1, gq2)
    var grad_actout = Tensor.alloc(B * AOUT)
    var c_s = Tensor.alloc(B * S)  # critic state input
    var c_a = Tensor.alloc(B * A)  # critic action input

    var opt = Adam(lr=0.01)

    comptime STEPS = 200
    comptime WIN = 25
    var first_sum: Scalar[DT] = 0  # mean over first WIN steps
    var last_sum: Scalar[DT] = 0  # mean over last WIN steps
    for step in range(STEPS):
        actor.zero_grad["cpu"](None)
        # forward actor → [action | log_prob]
        actor.forward[B](act_out)
        for b in range(B):
            for j in range(A):
                action.data[b * A + j] = act_out.data[b * AOUT + j]
        # critic forward (s = obs, a = action) → q1, q2 → min
        for i in range(B * S):
            c_s.data[i] = obs.data[i]
        for i in range(B * A):
            c_a.data[i] = action.data[i]
        c1.set_input["s", B](c_s)
        c1.set_input["a", B](c_a)
        c2.set_input["s", B](c_s)
        c2.set_input["a", B](c_a)
        c1.forward[B](qp[0])
        c2.forward[B](qp[1])
        minop.forward["cpu", B](TensorRefs[2](qp[0], qp[1]), minq)
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
        minop.vjp["cpu", B](
            TensorRefs[2](qp[0], qp[1]), grad_min, TensorRefs[2](gq[0], gq[1])
        )
        c1.zero_grad["cpu"](None)
        c2.zero_grad["cpu"](None)
        c1.vjp[B](gq[0])
        var ga0 = Tensor.alloc(B * A)
        for i in range(B * A):
            ga0.data[i] = c1.grad_input["a"]().data[i]
        c2.vjp[B](gq[1])
        # grad over rsample out: [grad_action | α/B]
        for b in range(B):
            for j in range(A):
                grad_actout.data[b * AOUT + j] = (
                    ga0.data[b * A + j] + c2.grad_input["a"]().data[b * A + j]
                )
            grad_actout.data[b * AOUT + A] = ALPHA / Scalar[DT](B)
        actor.vjp[B](grad_actout)
        opt.begin_step()
        actor.for_each_param["cpu"](opt, None)

    var first_mean = first_sum / Scalar[DT](WIN)
    var last_mean = last_sum / Scalar[DT](WIN)
    print("windowed mean actor_loss:", first_mean, "->", last_mean)
    if last_mean < first_mean:
        print(
            "SAC ACTOR STEP OK — actor lowers (α·logπ − min_q); loss",
            first_mean,
            "->",
            last_mean,
        )
    else:
        print("SAC ACTOR STEP WEAK (", first_mean, "->", last_mean, ")")
