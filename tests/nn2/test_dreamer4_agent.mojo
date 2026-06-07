"""Dreamer4Agent — end-to-end BC training step integration (Phase 3.7).

    pixi run mojo run -I . tests/nn2/test_dreamer4_agent.mojo

Drives the full facade `bc_train_step` on a synthetic dataset: the
shortcut-forcing video-prediction loss and the MTP behavior-cloning loss train
JOINTLY through one composite Adam over {dynamics, task embedder, policy head,
reward head}. Validates the whole wiring — TaskEmbedder → agent tokens →
dynamics forwards → h_t → heads → BC loss → grad_h → dyn.vjp (video + BC grads
together) → task-embedder grad — by checking:
  - the video-prediction loss stays finite and trends down,
  - the BC loss collapses, and
  - the greedy policy (argmax distance-0 logits) recovers the dataset actions.

Targets are task-determined (action/reward depend on the sequence's task id),
the clean BC capability: read the task from h_t and predict the dataset action.
"""

from std.memory import alloc
from std.math import sin
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.deep_agents2.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents2.dreamerv3.dists_discrete import cat_argmax
from mojo_rl.deep_agents2.dreamerv3.twohot import symexp_twohot_bins


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


comptime DSP = 4
comptime NSP = 4
comptime D = 8
comptime NH = 2
comptime T = 3
comptime NREG = 2
comptime HID = 16
comptime DEPTH = 2
comptime KMAX = 4
comptime NAGENT = 1
comptime NTASK = 4
comptime HHID = 16
comptime NACT = 3
comptime NBINS = 41
comptime NMTP = 2
comptime B = 3
comptime B_SELF = 1
comptime BF = B * T
comptime ND = NSP * DSP
comptime PLOG = NMTP * NACT

comptime Agent = Dreamer4Agent[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
    NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
]


def main() raises:
    print("=" * 70)
    print("Dreamer4Agent — joint BC + video-prediction step (Phase 3.7)")
    print("=" * 70)

    var agent = Agent.make[target="cpu", INIT=Xavier]()
    var optim = Adam.make["cpu", M=Agent](agent)
    optim.lr = Scalar[DT](2e-3)

    # ── synthetic dataset ───────────────────────────────────────────────
    var z1 = _alloc(BF * ND)
    var z0 = _alloc(BF * ND)
    for i in range(BF * ND):
        z1[i] = Scalar[DT](0.4 * sin(0.3 + 0.5 * Float64(i)))
        z0[i] = Scalar[DT](0.3 * sin(2.1 + 0.9 * Float64(i)))

    # shortcut sampling on the step grid: σ=0.5 (j=1), step=1 ⇒ σ_plus=0.75<1
    var sigma = _alloc(BF)
    var sigma_idx = _alloc(BF)
    var step_idx = _alloc(BF)
    for bt in range(BF):
        sigma[bt] = Scalar[DT](0.5)
        sigma_idx[bt] = Scalar[DT](2.0)
        step_idx[bt] = Scalar[DT](1.0)

    var task_ids = _alloc(B)
    for b in range(B):
        task_ids[b] = Scalar[DT](Float64(b))

    # task-determined targets (constant over the window)
    var actions = _alloc(BF)
    var rewards = _alloc(BF)
    for b in range(B):
        for j in range(T):
            var bt = b * T + j
            actions[bt] = Scalar[DT](Float64((b * 2) % NACT))
            rewards[bt] = Scalar[DT](0.4 * Float64(b - 1))

    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    # ── train ───────────────────────────────────────────────────────────
    var first_v: Float64 = 0.0
    var last_v: Float64 = 0.0
    var first_bc: Float64 = 0.0
    var last_bc: Float64 = 0.0
    for step in range(400):
        optim.zero_grad["cpu"](agent)
        var losses = agent.bc_train_step(
            z1, z0, sigma, sigma_idx, step_idx, step >= 30,
            task_ids, actions, rewards, bins,
        )
        optim.step["cpu"](agent)
        if step == 0:
            first_v = losses[0]
            first_bc = losses[1]
        last_v = losses[0]
        last_bc = losses[1]
        if step % 80 == 0:
            print("   step", step, " video =", losses[0], " bc =", losses[1])
    print("   video loss", first_v, "->", last_v)
    print("   bc    loss", first_bc, "->", last_bc)

    # ── eval: refresh logits with final params, check greedy accuracy ────
    var _refresh = agent.bc_train_step(
        z1, z0, sigma, sigma_idx, step_idx, False,
        task_ids, actions, rewards, bins,
    )
    var plog = agent.policy_logits_ptr()
    var n_correct = 0
    for b in range(B):
        for j in range(T):
            var bt = b * T + j
            var k = Int(Float64(actions[bt]) + 0.5)
            if cat_argmax[NACT](plog, bt * PLOG) == k:   # distance-0 block
                n_correct += 1
    print("   greedy action accuracy =", n_correct, "/", BF)

    assert_true(last_v < first_v, "video-prediction loss must decrease")
    assert_true(last_v < 1e3, "video-prediction loss must stay finite")
    assert_true(last_bc < 0.5 * first_bc, "BC loss must collapse")
    assert_true(n_correct == BF, "greedy policy recovers dataset actions")

    print("=" * 70)
    print("ALL PASSED — Dreamer4Agent BC integration (Phase 3.7)")
    print("=" * 70)
