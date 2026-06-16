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

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer import Adam
from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamerv3.twohot import (
    symexp_twohot_bins, twohot_loss, twohot_loss_backward, twohot_pred,
)
from mojo_rl.deep_agents.dreamerv3.dists_discrete import cat_argmax


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


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
    plog: UnsafePointer[Scalar[DT], MutAnyOrigin]
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

    var agent = Agent.make[target="cpu", INIT=Xavier]()
    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    var task_ids = _alloc(B)
    for b in range(B):
        task_ids[b] = Scalar[DT](0.0)
    var agp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        agent.agent_in.unsafe_ptr()
    )
    agent.te.embed_into["cpu", B, T](task_ids, agp)

    # ── Stage 1: supervised reward path (dyn transformer+act-MLP + reward) ──
    # Match the rollout's latent distribution: frame 0 (context) = 0.1, the
    # imagined frame 1 ≈ 0; the imagined frame's action token = a[b] (cover all
    # actions). Train r̂(h₁) = +1 if a[b] == GOOD else −1.
    print("- Stage 1: train world-model reward path (dyn + reward head)")
    var dopt = Adam.make["cpu", M=Agent.DYN](agent.dyn)
    dopt.lr = Scalar[DT](3e-3)
    var ropt = Adam.make["cpu", M=Agent.RH](agent.rh)
    ropt.lr = Scalar[DT](3e-3)

    var zfix = _alloc(BF * ND)
    var act_oh = _alloc(BF * ADIM)
    var act_mask = _alloc(BF * ADIM)
    var r_tgt = _alloc(B)                       # reward target on frame 1
    for i in range(BF * ADIM):
        act_mask[i] = Scalar[DT](1.0)
    for i in range(BF * ADIM):
        act_oh[i] = Scalar[DT](0.0)
    for b in range(B):
        var f0 = (b * T + 0) * ND
        var f1 = (b * T + 1) * ND
        for i in range(ND):
            zfix[f0 + i] = Scalar[DT](0.1)      # context frame
            zfix[f1 + i] = Scalar[DT](0.0)      # imagined frame ≈ 0
        var a = b % NACT
        for k in range(ADIM):                   # frame-1 action token = a
            act_oh[(b * T + 1) * ADIM + k] = Scalar[DT](1.0 if k == a else 0.0)
        r_tgt[b] = Scalar[DT](1.0 if a == GOOD else -1.0)

    var sigc = _alloc(BF)
    var stepc = _alloc(BF)
    for i in range(BF):
        sigc[i] = Scalar[DT](Float64(KMAX - 1))
        stepc[i] = Scalar[DT](2.0)               # EMAX = log2(KMAX)

    var zfix_t = TileTensor(zfix, row_major[BF, ND]())
    var zhat = _alloc(BF * ND)
    var zhat_t = TileTensor(zhat, row_major[BF, ND]())
    var rlog = _alloc(BF * RLOG)
    var rlog_t = TileTensor(rlog, row_major[BF, RLOG]())
    var grl = _alloc(BF * RLOG)
    var grl_t = TileTensor(grl, row_major[BF, RLOG]())
    var grad_hi = _alloc(BF * AGD)             # rh.vjp grad_input = grad wrt h
    var grad_hi_t = TileTensor(grad_hi, row_major[BF, AGD]())
    var gzero = _alloc(BF * ND)
    var gzero_t = TileTensor(gzero, row_major[BF, ND]())
    var gzt = _alloc(BF * ND)
    var gzt_t = TileTensor(gzt, row_major[BF, ND]())
    for i in range(BF * ND):
        gzero[i] = Scalar[DT](0.0)

    var first_r: Float64 = 0.0
    var last_r: Float64 = 0.0
    for step in range(800):
        dopt.zero_grad["cpu"](agent.dyn)
        ropt.zero_grad["cpu"](agent.rh)
        agent.dyn.set_indices(sigc, stepc, BF)
        agent.dyn.set_actions(act_oh, act_mask, BF)
        agent.dyn.set_agent_in(agp, BF)
        agent.dyn.forward["cpu", BF](zfix_t, output=zhat_t)
        var ht = agent.dyn.agent_out_ptr_cpu()
        var ht_t = TileTensor(ht, row_major[BF, AGD]())
        agent.rh.forward["cpu", BF](ht_t, output=rlog_t)
        for i in range(BF * RLOG):
            grl[i] = Scalar[DT](0.0)
        var rloss: Float64 = 0.0
        for b in range(B):                       # loss on the imagined frame
            var bt = b * T + 1
            rloss += Float64(twohot_loss[NBINS](rlog, bt * RLOG, bins, r_tgt[b]))
            twohot_loss_backward[NBINS](
                rlog, bt * RLOG, bins, r_tgt[b], Scalar[DT](1.0), grl
            )
        agent.rh.vjp["cpu", BF, mode="all"](grl_t, grad_hi_t)
        agent.dyn.set_grad_h(grad_hi, BF)
        agent.dyn.vjp["cpu", BF](gzero_t, gzt_t)
        dopt.step["cpu"](agent.dyn)
        ropt.step["cpu"](agent.rh)
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
                Float64(twohot_pred[NBINS](rlog, bt * RLOG, bins)), end="  "
            )
    print()

    # ── snapshot the BC-prior policy ─────────────────────────────────────
    agent.snapshot_prior()

    var ctx = _alloc(B * NCTX * ND)
    for i in range(B * NCTX * ND):
        ctx[i] = Scalar[DT](0.1)
    var u01 = _alloc(B * T)
    var znoise = _alloc(B * T * ND)
    for i in range(B * T):
        u01[i] = Scalar[DT](0.5)
    for i in range(B * T * ND):
        znoise[i] = Scalar[DT](0.0)              # match the trained frame-1 ≈ 0

    # ── Stage 2: imagination RL (only policy + value heads train) ─────────
    # FRESH action-sampling uniforms each step: the rollout must explore so the
    # value head learns the EXPECTED return (the PMPO baseline); with a fixed
    # seed the value would fit one realized action and the advantage vanishes.
    print("- Stage 2: imagination RL (PMPO + value TD; transformer frozen)")
    var opt = Adam.make["cpu", M=Agent](agent)
    opt.lr = Scalar[DT](3e-2)
    var rng = Rng(20260607)

    var base_frac: Float64 = -1.0
    var first_p: Float64 = 0.0
    var last_p: Float64 = 0.0
    for step in range(300):
        for i in range(B * T):
            u01[i] = Scalar[DT](rng.uniform())
        opt.zero_grad["cpu"](agent)
        var losses = agent.imag_train_step(ctx, u01, znoise, task_ids, bins)
        if step == 0:
            base_frac = _greedy_good_fraction(agent.imag_policy_logits_ptr())
            first_p = losses[1]
        opt.step["cpu"](agent)
        last_p = losses[1]
        if step % 50 == 0:
            print(
                "   imag step", step, " value =", losses[0], " policy =",
                losses[1], " greedy(good) =",
                _greedy_good_fraction(agent.imag_policy_logits_ptr()),
            )

    var _f = agent.imag_train_step(ctx, u01, znoise, task_ids, bins)
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
