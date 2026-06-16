"""Dreamer4Agent — frame-content→action path isolation (Phase 3.7 diagnostic).

    pixi run mojo run -I . tests/nn/test_dreamer4_agent_content.mojo

The other agent test (test_dreamer4_agent) clones a TASK-determined action
(constant per sequence) — that only exercises the clean task-embedding→action
path. THIS test makes the action a deterministic function of the per-FRAME
latent CONTENT (varying within a sequence), with a SINGLE task, so the task
embedding carries no signal: the policy can only succeed if h_t (the agent
token, which attends to the frame's spatial latents) carries the within-sequence
frame content to the logits.

Setup: a FIXED batch of distinct random latents z1[b,t]; action[b,t] = argmax
over NACT contiguous latent-dim groups of z1[b,t] (a robust content function).
Overfit the agent (joint with the video loss) and check the greedy policy
recovers the per-frame actions FAR above the majority-class prior. If this
passes, the agent→policy path carries frame content — so the real Pong BC
failure is purely the tokenizer averaging away the ~1px ball, not this path.
"""

from std.memory import alloc
from std.math import sin
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer import Adam
from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamerv3.dists_discrete import cat_argmax
from mojo_rl.deep_agents.dreamerv3.twohot import symexp_twohot_bins


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


comptime DSP = 4
comptime NSP = 4
comptime D = 16
comptime NH = 2
comptime T = 3
comptime NREG = 2
comptime HID = 32
comptime DEPTH = 2
comptime KMAX = 4
comptime NAGENT = 1
comptime NTASK = 1
comptime HHID = 32
comptime NACT = 3
comptime NBINS = 41
comptime NMTP = 1            # predict ONLY the current frame's action
comptime B = 3
comptime B_SELF = 1
comptime BF = B * T
comptime ND = NSP * DSP
comptime PLOG = NMTP * NACT

comptime Agent = Dreamer4Agent[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
    NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
]


def _content_action(z1: UnsafePointer[Scalar[DT], MutAnyOrigin], bt: Int) -> Int:
    # argmax over NACT contiguous groups of the frame's latent (robust, content)
    var grp = ND // NACT
    var best = 0
    var bv = Scalar[DT](-1e30)
    for c in range(NACT):
        var s = Scalar[DT](0.0)
        for d in range(grp):
            s += z1[bt * ND + c * grp + d]
        if s > bv:
            bv = s
            best = c
    return best


def main() raises:
    print("=" * 70)
    print("Dreamer4Agent — frame-content→action path (Phase 3.7 diagnostic)")
    print("=" * 70)

    var agent = Agent.make[target="cpu", INIT=Xavier]()
    var optim = Adam.make["cpu", M=Agent](agent)
    optim.lr = Scalar[DT](1e-3)

    # FIXED batch: distinct per-(b,t) latents + fixed noise (deterministic BC).
    # Small amplitude (tanh-latent scale) keeps the flow loss well-conditioned.
    var z1 = _alloc(BF * ND)
    var z0 = _alloc(BF * ND)
    for i in range(BF * ND):
        z1[i] = Scalar[DT](0.3 * sin(0.3 + 0.7 * Float64(i)))
        z0[i] = Scalar[DT](0.25 * sin(2.1 + 1.3 * Float64(i)))

    var sigma = _alloc(BF)
    var sigma_idx = _alloc(BF)
    var step_idx = _alloc(BF)
    for bt in range(BF):
        sigma[bt] = Scalar[DT](0.5)
        sigma_idx[bt] = Scalar[DT](2.0)
        step_idx[bt] = Scalar[DT](1.0)

    var task_ids = _alloc(B)
    for b in range(B):
        task_ids[b] = Scalar[DT](0.0)          # SINGLE task ⇒ no task signal

    # per-FRAME content-determined action (varies within a sequence)
    var actions = _alloc(BF)
    var rewards = _alloc(BF)
    var class_count = InlineArray[Int, NACT](fill=0)
    for bt in range(BF):
        var k = _content_action(z1, bt)
        actions[bt] = Scalar[DT](Float64(k))
        class_count[k] += 1
        rewards[bt] = Scalar[DT](0.0)
    var maj = 0
    for c in range(1, NACT):
        if class_count[c] > class_count[maj]:
            maj = c
    var prior = Float64(class_count[maj]) / Float64(BF)

    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    var first_bc: Float64 = 0.0
    var last_bc: Float64 = 0.0
    for step in range(600):
        optim.zero_grad["cpu"](agent)
        # do_boot=False: the bootstrap two-step term is the unstable part of the
        # video loss and is irrelevant to validating the BC path; the empirical
        # flow term alone keeps the world model well-behaved here.
        var losses = agent.bc_train_step(
            z1, z0, sigma, sigma_idx, step_idx, False,
            task_ids, actions, rewards, bins,
            policy_weight=Scalar[DT](1.0), reward_weight=Scalar[DT](0.0),
        )
        optim.step["cpu"](agent)
        if step == 0:
            first_bc = losses[1]
        last_bc = losses[1]
        if step % 120 == 0:
            print("   step", step, " video =", losses[0], " bc =", losses[1])

    # refresh logits with final params
    var _r = agent.bc_train_step(
        z1, z0, sigma, sigma_idx, step_idx, False,
        task_ids, actions, rewards, bins,
        policy_weight=Scalar[DT](1.0), reward_weight=Scalar[DT](0.0),
    )
    var plog = agent.policy_logits_ptr()
    var n_correct = 0
    for bt in range(BF):
        if cat_argmax[NACT](plog, bt * PLOG) == Int(Float64(actions[bt]) + 0.5):
            n_correct += 1
    var acc = Float64(n_correct) / Float64(BF)

    print("-" * 70)
    print("  BC loss   first =", first_bc, "  final =", last_bc)
    print("  per-frame content accuracy =", acc, " (", n_correct, "/", BF, ")")
    print("  majority-class prior       =", prior, " (action", maj, ")")

    # The policy must recover the per-frame, content-determined actions far
    # above the prior — proving h_t carries within-sequence frame content.
    assert_true(last_bc < 0.4 * first_bc, "BC loss must collapse")
    assert_true(acc > 0.8, "policy recovers per-frame content actions")
    assert_true(acc > prior + 0.2, "content signal, not the class prior")

    print("=" * 70)
    print("PATH ISOLATED — the agent-token → policy path carries within-sequence")
    print("frame content (single task). Real Pong BC is tokenizer-limited, not")
    print("an architecture gap.")
    print("=" * 70)
