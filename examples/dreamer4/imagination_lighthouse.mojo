"""Dreamer 4 imagination-RL lighthouse — greedy return beats the BC prior (CPU).

    pixi run mojo run -I . examples/dreamer4/imagination_lighthouse.mojo

The Phase-4 gate (paper §3.3): *imagination training improves the greedy return
over the behavior-cloning prior*. We demonstrate it in a CONTROLLED world model
rather than on real Pong — deliberately, and for the same reason the Phase-3
content-path test was synthetic: the collected Pong buffer carries NO rewards
(imagination RL maximizes a learned reward model, so a reward signal is
required), and real-Pong BC is data-SNR-limited to the class prior. A
controlled world isolates the imagination-RL mechanism cleanly and decisively.

The world (T = 2: one clean context frame, one imagined frame): there is one
GOOD action g. The action sampled at the context state becomes the action token
of the imagined frame; it flows through the transformer into that frame's agent
token h₁, which the reward head reads. The reward is +1 if that action was g,
else −1. The optimal policy always picks g.

Pipeline (all CPU):
  Stage 1  — supervised: train the transformer + action-MLP + reward head so
             r̂(h₁) predicts the action-determined reward. This is the ONLY
             world-model training the gate needs: the reward depends on the
             action token, not on the (untrained) flow-predicted latents, so we
             train on the SAME latent distribution the rollout produces (context
             frame = 0.1, imagined frame ≈ 0). Then FREEZE.
  snapshot — copy the BC policy (Xavier-init near-uniform) as the frozen
             behavioral prior for the PMPO reverse-KL.
  Stage 2  — imagination RL: `imag_train_step` repeatedly. The value head learns
             λ-returns; PMPO upweights the action that earned positive advantage
             (g); the reverse-KL keeps the policy reasonable.

GATE: the fraction of states whose GREEDY action equals g rises from the prior
to ≈1 — the policy learns, purely in imagination, to take the rewarding action.
(The flow head stays untrained — generated latents collapse toward 0 — but the
action→h→reward path the gate exercises is fully trained, so the signal is
honest.)
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao
from mojo_rl.deep_agents.dreamerv3.twohot import (
    symexp_twohot_bins, twohot_loss, twohot_loss_backward, twohot_pred,
)
from mojo_rl.deep_agents.dreamerv3.dists_discrete import cat_argmax


# tiny deterministic RNG (xorshift64*) for fresh action-sampling uniforms
struct Rng(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed | 1

    def uniform(mut self) -> Float64:
        var x = self.s
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.s = x
        return Float64((x * 0x2545F4914F6CDD1D) >> 11) * (1.0 / 9007199254740992.0)


comptime DSP = 4
comptime NSP = 4
comptime D = 16
comptime NH = 2
comptime T = 2                          # 1 context state + 1 imagined frame
comptime NREG = 2
comptime HID = 32
comptime DEPTH = 2
comptime KMAX = 4
comptime NAGENT = 1
comptime NTASK = 1
comptime HHID = 32
comptime NACT = 3
comptime NBINS = 41
comptime NMTP = 1
comptime B = 6                          # two sequences per action
comptime B_SELF = 2
comptime ADIM = NACT
comptime AHID = 2 * D
comptime K_IMAG = 2
comptime NCTX = 1
comptime ND = NSP * DSP
comptime AGD = NAGENT * D
comptime BF = B * T
comptime TM1 = T - 1
comptime RLOG = NMTP * NBINS
comptime GOOD = 1                       # the rewarding action

comptime Agent = Dreamer4Agent[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
    NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
    True, ADIM, AHID, K_IMAG, NCTX,
]


def _greedy_good_fraction(
    plog: Pointer[Scalar[DT], MutAnyOrigin]
) -> Float64:
    """Fraction of acting states (0..T-2) whose greedy dist-0 action == GOOD."""
    var n_good = 0
    var n = 0
    for b in range(B):
        for t in range(TM1):
            var base = (b * T + t) * (NMTP * NACT)   # dist-0 block
            if cat_argmax[NACT](plog, base) == GOOD:
                n_good += 1
            n += 1
    return Float64(n_good) / Float64(n)


def main() raises:
    print("=" * 70)
    print("Dreamer 4 imagination-RL lighthouse — greedy return beats BC prior")
    print("=" * 70)

    var agent = Agent.make["cpu", Xavier](None)
    var bins = Tensor.alloc(NBINS)
    symexp_twohot_bins[NBINS](_mao(bins.data.unsafe_ptr()), lo=Scalar[DT](-9.0))

    var task_ids = Tensor.alloc(B)
    for b in range(B):
        task_ids.data[b] = Scalar[DT](0.0)
    var agp = _mao(agent.agent_in.unsafe_ptr())
    agent.te.embed_into["cpu", B, T](_mao(task_ids.data.unsafe_ptr()), agp)

    # ── Stage 1: supervised reward path (dyn transformer+act-MLP + reward) ──
    # Match the rollout's latent distribution: frame 0 (context) = 0.1, the
    # imagined frame 1 ≈ 0; the imagined frame's action token = a[b] (cover all
    # actions). Train r̂(h₁) = +1 if a[b] == GOOD else −1.
    print("- Stage 1: train world-model reward path (dyn + reward head)")
    var dopt = Adam(lr=Scalar[DT](3e-3))
    var ropt = Adam(lr=Scalar[DT](3e-3))

    var zfix = Tensor.alloc(BF * ND)
    var act_oh = Tensor.alloc(BF * ADIM)
    var act_mask = Tensor.alloc(BF * ADIM)
    var r_tgt = Tensor.alloc(B)                       # reward target on frame 1
    for i in range(BF * ADIM):
        act_mask.data[i] = Scalar[DT](1.0)
    for i in range(BF * ADIM):
        act_oh.data[i] = Scalar[DT](0.0)
    for b in range(B):
        var f0 = (b * T + 0) * ND
        var f1 = (b * T + 1) * ND
        for i in range(ND):
            zfix.data[f0 + i] = Scalar[DT](0.1)      # context frame
            zfix.data[f1 + i] = Scalar[DT](0.0)      # imagined frame ≈ 0
        var a = b % NACT
        for k in range(ADIM):                        # frame-1 action token = a
            act_oh.data[(b * T + 1) * ADIM + k] = Scalar[DT](1.0 if k == a else 0.0)
        r_tgt.data[b] = Scalar[DT](1.0 if a == GOOD else -1.0)

    var sigc = Tensor.alloc(BF)
    var stepc = Tensor.alloc(BF)
    for i in range(BF):
        sigc.data[i] = Scalar[DT](Float64(KMAX - 1))
        stepc.data[i] = Scalar[DT](2.0)              # EMAX = log2(KMAX)

    var zhat = Tensor.alloc(BF * ND)
    var rlog = Tensor.alloc(BF * RLOG)
    var grl = Tensor.alloc(BF * RLOG)
    var grad_hi = Tensor.alloc(BF * AGD)             # rh.vjp grad_input = grad wrt h
    var gzero = Tensor.alloc(BF * ND)
    var gzt = Tensor.alloc(BF * ND)
    for i in range(BF * ND):
        gzero.data[i] = Scalar[DT](0.0)

    var first_r: Float64 = 0.0
    var last_r: Float64 = 0.0
    for step in range(800):
        dopt.zero_grad["cpu"](agent.dyn, None)
        ropt.zero_grad["cpu"](agent.rh, None)
        agent.dyn.set_indices(
            _mao(sigc.data.unsafe_ptr()), _mao(stepc.data.unsafe_ptr()), BF
        )
        agent.dyn.set_actions(
            _mao(act_oh.data.unsafe_ptr()), _mao(act_mask.data.unsafe_ptr()), BF
        )
        agent.dyn.set_agent_in(agp, BF)
        agent.dyn.forward["cpu", BF](TensorRefs[1](zfix), zhat, None)
        var ht = agent.dyn.agent_out_ptr_cpu()
        # h_t [BF, AGD] from the dynamics forward → reward head forward. Wrap it
        # in a borrowing input `Tensor` for the storage Module surface.
        var ht_t = Tensor.alloc(BF * AGD)
        for i in range(BF * AGD):
            ht_t.data[i] = ht[i]
        agent.rh.forward["cpu", BF](TensorRefs[1](ht_t), rlog, None)
        for i in range(BF * RLOG):
            grl.data[i] = Scalar[DT](0.0)
        var rloss: Float64 = 0.0
        for b in range(B):                           # loss on the imagined frame
            var bt = b * T + 1
            rloss += Float64(
                twohot_loss[NBINS](
                    _mao(rlog.data.unsafe_ptr()), bt * RLOG,
                    _mao(bins.data.unsafe_ptr()), r_tgt.data[b]
                )
            )
            twohot_loss_backward[NBINS](
                _mao(rlog.data.unsafe_ptr()), bt * RLOG,
                _mao(bins.data.unsafe_ptr()), r_tgt.data[b],
                Scalar[DT](1.0), _mao(grl.data.unsafe_ptr())
            )
        agent.rh.vjp["cpu", BF](
            TensorRefs[1](ht_t), grl, TensorRefs[1](grad_hi), None
        )
        agent.dyn.set_grad_h(_mao(grad_hi.data.unsafe_ptr()), BF)
        agent.dyn.vjp["cpu", BF](TensorRefs[1](zfix), gzero, TensorRefs[1](gzt), None)
        dopt.step["cpu"](agent.dyn, None)
        ropt.step["cpu"](agent.rh, None)
        if step == 0:
            first_r = rloss
        last_r = rloss
        if step % 160 == 0:
            print("   reward step", step, " twohot CE =", rloss)
    print("   reward CE", first_r, "->", last_r)

    # sanity: predicted reward per imagined-frame action
    print("   r̂ per action:", end=" ")
    for b in range(B):
        if b < NACT:
            var bt = b * T + 1
            print(
                "a", b % NACT, "=",
                Float64(twohot_pred[NBINS](
                    _mao(rlog.data.unsafe_ptr()), bt * RLOG,
                    _mao(bins.data.unsafe_ptr())
                )), end="  "
            )
    print()

    # ── snapshot the BC-prior policy ─────────────────────────────────────
    agent.snapshot_prior()

    var ctx = Tensor.alloc(B * NCTX * ND)
    for i in range(B * NCTX * ND):
        ctx.data[i] = Scalar[DT](0.1)
    var u01 = Tensor.alloc(B * T)
    var znoise = Tensor.alloc(B * T * ND)
    for i in range(B * T):
        u01.data[i] = Scalar[DT](0.5)
    for i in range(B * T * ND):
        znoise.data[i] = Scalar[DT](0.0)             # match the trained frame-1 ≈ 0

    # ── Stage 2: imagination RL (only policy + value heads train) ─────────
    # FRESH action-sampling uniforms each step: the rollout must explore so the
    # value head learns the EXPECTED return (the PMPO baseline); with a fixed
    # seed the value would fit one realized action and the advantage vanishes.
    print("- Stage 2: imagination RL (PMPO + value TD; transformer frozen)")
    var opt = Adam(lr=Scalar[DT](3e-2))
    var rng = Rng(20260607)

    var base_frac: Float64 = -1.0
    var first_p: Float64 = 0.0
    var last_p: Float64 = 0.0
    for step in range(300):
        for i in range(B * T):
            u01.data[i] = Scalar[DT](rng.uniform())
        opt.zero_grad["cpu"](agent, None)
        var losses = agent.imag_train_step(
            _mao(ctx.data.unsafe_ptr()),
            _mao(u01.data.unsafe_ptr()),
            _mao(znoise.data.unsafe_ptr()),
            _mao(task_ids.data.unsafe_ptr()),
            _mao(bins.data.unsafe_ptr()),
        )
        if step == 0:
            base_frac = _greedy_good_fraction(agent.imag_policy_logits_ptr())
            first_p = losses[1]
        opt.step["cpu"](agent, None)
        last_p = losses[1]
        if step % 50 == 0:
            print(
                "   imag step", step, " value =", losses[0], " policy =",
                losses[1], " greedy(good) =",
                _greedy_good_fraction(agent.imag_policy_logits_ptr()),
            )

    var _f = agent.imag_train_step(
        _mao(ctx.data.unsafe_ptr()),
        _mao(u01.data.unsafe_ptr()),
        _mao(znoise.data.unsafe_ptr()),
        _mao(task_ids.data.unsafe_ptr()),
        _mao(bins.data.unsafe_ptr()),
    )
    var final_frac = _greedy_good_fraction(agent.imag_policy_logits_ptr())

    print("-" * 70)
    print("  PMPO policy loss   first =", first_p, "  final =", last_p)
    print("  greedy(good) prior =", base_frac, "  ->  final =", final_frac)
    print("  (optimal = 1.0 ; uniform prior ≈", 1.0 / Float64(NACT), ")")

    assert_true(
        final_frac > base_frac + 0.25,
        "imagination training must raise the greedy-good fraction over the prior",
    )
    assert_true(final_frac > 0.6, "policy should mostly pick the rewarding action")

    print("=" * 70)
    print("IMAGINATION LIGHTHOUSE PASSED — the policy learns, purely in")
    print("imagination, to take the rewarding action (greedy return > BC prior).")
    print("=" * 70)
