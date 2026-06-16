"""Dreamer4Agent — frame-content→REWARD path isolation (Phase 4 diagnostic).

    pixi run mojo run -I . tests/nn/test_dreamer4_reward_content.mojo

The reward-bearing end-to-end run (`examples/dreamer4/pong_reward_end2end.mojo`)
finds the reward head SNR-limited on the Pong PIXEL buffer — Pong's reward fires
on the ~1px ball-at-paddle event that the tokenizer (recon ≈23 dB) blurs away,
the same wall the BC lighthouse hit. This test isolates the reward path from
that pixel-SNR limit, exactly as `test_dreamer4_agent_content` did for the
policy path: it makes the reward a deterministic function of the per-FRAME
latent CONTENT and trains it through the agent's reward head + h_t.

If the reward model recovers the content-determined reward FAR below the
mean-reward baseline, then the reward-bearing path (h_t → reward head, eq. 9
twohot) carries real-magnitude reward when the signal is in the latent — so the
real-Pong shortfall is the tokenizer, not the architecture. This is the reward
analogue of the policy content-path test, and the validated half of the
reward-bearing buffer deliverable.
"""

from std.memory import alloc
from std.math import sin, abs, tanh
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer import Adam
from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamerv3.twohot import symexp_twohot_bins, twohot_pred


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
comptime NMTP = 1            # predict ONLY the current frame's reward
comptime B = 3
comptime B_SELF = 1
comptime BF = B * T
comptime ND = NSP * DSP
comptime RLOG = NMTP * NBINS

comptime Agent = Dreamer4Agent[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
    NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
]


def _content_reward(z1: UnsafePointer[Scalar[DT], MutAnyOrigin], bt: Int) -> Float64:
    # continuous reward determined by the frame's latent halves (varies within a
    # sequence): tanh(Σ first half − Σ second half), range ≈ [−1, 1].
    var half = ND // 2
    var s0 = Scalar[DT](0.0)
    var s1 = Scalar[DT](0.0)
    for d in range(half):
        s0 += z1[bt * ND + d]
    for d in range(half, ND):
        s1 += z1[bt * ND + d]
    return Float64(tanh(1.5 * (s0 - s1)))


def main() raises:
    print("=" * 70)
    print("Dreamer4Agent — frame-content→REWARD path (Phase 4 diagnostic)")
    print("=" * 70)

    var agent = Agent.make[target="cpu", INIT=Xavier]()
    var optim = Adam.make["cpu", M=Agent](agent)
    optim.lr = Scalar[DT](1e-3)

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
        task_ids[b] = Scalar[DT](0.0)

    # per-FRAME content-determined REWARD (varies within a sequence)
    var actions = _alloc(BF)
    var rewards = _alloc(BF)
    var sum_true: Float64 = 0.0
    for bt in range(BF):
        actions[bt] = Scalar[DT](0.0)                # unused (policy_weight=0)
        var r = _content_reward(z1, bt)
        rewards[bt] = Scalar[DT](r)
        sum_true += r
    var mean_r = sum_true / Float64(BF)

    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    var first_bc: Float64 = 0.0
    var last_bc: Float64 = 0.0
    for step in range(600):
        optim.zero_grad["cpu"](agent)
        var losses = agent.bc_train_step(
            z1, z0, sigma, sigma_idx, step_idx, False,
            task_ids, actions, rewards, bins,
            policy_weight=Scalar[DT](0.0), reward_weight=Scalar[DT](1.0),
        )
        optim.step["cpu"](agent)
        if step == 0:
            first_bc = losses[1]
        last_bc = losses[1]
        if step % 120 == 0:
            print("   step", step, " video =", losses[0], " reward CE =",
                  losses[1])

    # refresh reward logits with final params
    var _r = agent.bc_train_step(
        z1, z0, sigma, sigma_idx, step_idx, False,
        task_ids, actions, rewards, bins,
        policy_weight=Scalar[DT](0.0), reward_weight=Scalar[DT](1.0),
    )
    var rlog = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        agent.rlog.unsafe_ptr()
    )
    var mae_model: Float64 = 0.0
    var mae_mean: Float64 = 0.0
    for bt in range(BF):
        var pr = Float64(twohot_pred[NBINS](rlog, bt * RLOG, bins))
        var tr = Float64(rewards[bt])
        mae_model += abs(pr - tr)
        mae_mean += abs(mean_r - tr)
    mae_model /= Float64(BF)
    mae_mean /= Float64(BF)

    print("-" * 70)
    print("  reward CE  first =", first_bc, "  final =", last_bc)
    print("  reward MAE  model =", mae_model, "   mean-baseline =", mae_mean)

    # The reward model must recover the content-determined reward FAR below the
    # mean-reward baseline — proving h_t carries within-sequence frame content
    # to the reward head.
    assert_true(last_bc < 0.6 * first_bc, "reward CE must collapse")
    assert_true(mae_model < 0.3 * mae_mean, "reward model beats the mean baseline")

    print("=" * 70)
    print("REWARD PATH ISOLATED — the agent-token → reward-head path carries")
    print("within-sequence frame content (eq. 9 twohot). Real-Pong reward is")
    print("tokenizer-SNR-limited, not an architecture gap.")
    print("=" * 70)
