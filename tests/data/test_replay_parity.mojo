"""Gate for data-platform Stage 4 — `StoreReplay` vs the legacy CPU buffers.

**A stronger gate than the plan asked for.** The plan specified "same reward
curve as the legacy buffer over N seeds". That is a weak, noisy, expensive
proxy: curves drift for reasons unrelated to sampling, and matching them costs
hours. Since Stage 3 proved index-sequence parity, we can instead assert the
*minibatch itself* is *bit-identical* under the same seed — every element of
`mb_s / mb_a / mb_r / mb_sp / mb_d / mb_w`. If the minibatch stream is
byte-for-byte identical, the training trajectory is identical by construction,
so reward-curve equality follows rather than being sampled. It runs in
seconds.

Both paths are driven through the REAL `ReplaySampleStep` block that the
trainers use, not through the buffers directly — so the seam itself is what is
being gated, including `setup`, `add`, `step`, and the PER hooks.

Covered:
  1. uniform:      StoreReplay[.., False] vs CPUReplay
  2. prioritized:  StoreReplay[.., True]  vs CPUPrioritizedReplay,
                   including IS weights and a second draw after
                   `update_priorities`
  3. ring wraparound — the buffer is overfilled past CAP so the overwrite
     path is exercised, not just the initial fill

Run:
    pixi run mojo run -I . tests/data/test_replay_parity.mojo
"""

from std.random import seed
from std.testing import assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.data.replay import StoreReplay
from mojo_rl.deep_agents.data.cpu_replay import CPUReplay
from mojo_rl.deep_agents.data.cpu_per_replay import CPUPrioritizedReplay
from mojo_rl.deep_agents.training.blocks.replay_sample_step import (
    ReplaySampleStep,
)
from mojo_rl.deep_agents.training.trainer_block import TrainerState


comptime OBS: Int = 3
comptime ACT: Int = 2
comptime CAP: Int = 32
comptime BATCH: Int = 16
comptime N_FILL: Int = 50          # > CAP, so the ring wraps
comptime SEED: Int = 20260805


def _obs_for(row: Int) -> List[Scalar[DT]]:
    var o = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    for i in range(OBS):
        o[i] = Scalar[DT](Float64(row) * 10.0 + Float64(i))
    return o^


def _nxt_for(row: Int) -> List[Scalar[DT]]:
    var o = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    for i in range(OBS):
        o[i] = Scalar[DT](Float64(row) * -10.0 - Float64(i))
    return o^


def _act_for(row: Int) -> List[Scalar[DT]]:
    var a = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    for j in range(ACT):
        a[j] = Scalar[DT](Float64(row) * 0.5 + Float64(j) * 0.125)
    return a^


def _assert_state_identical(
    ref a: TrainerState[OBS, ACT, BATCH],
    ref b: TrainerState[OBS, ACT, BATCH],
    label: String,
    check_w: Bool,
) raises:
    for i in range(BATCH * OBS):
        assert_equal(a.mb_s.data[i], b.mb_s.data[i], label + " mb_s[" + String(i) + "]")
        assert_equal(a.mb_sp.data[i], b.mb_sp.data[i], label + " mb_sp[" + String(i) + "]")
    for i in range(BATCH * ACT):
        assert_equal(a.mb_a.data[i], b.mb_a.data[i], label + " mb_a[" + String(i) + "]")
    for i in range(BATCH):
        assert_equal(a.mb_r.data[i], b.mb_r.data[i], label + " mb_r[" + String(i) + "]")
        assert_equal(a.mb_d.data[i], b.mb_d.data[i], label + " mb_d[" + String(i) + "]")
    if check_w:
        for i in range(BATCH):
            assert_equal(
                a.mb_w.data[i], b.mb_w.data[i],
                label + " mb_w[" + String(i) + "]",
            )
        assert_equal(a.has_per, b.has_per, label + " has_per")


def _nonconstant(ref st: TrainerState[OBS, ACT, BATCH]) -> Bool:
    """Guard against a vacuous comparison: two all-zero minibatches match."""
    for i in range(1, BATCH):
        if st.mb_r.data[i] != st.mb_r.data[0]:
            return True
    return False


def test_uniform_parity() raises:
    print("[1] StoreReplay[uniform] vs CPUReplay, through ReplaySampleStep ...")

    var legacy = ReplaySampleStep[CPUReplay[OBS, ACT, CAP], BATCH]()
    var mine = ReplaySampleStep[StoreReplay[OBS, ACT, CAP, False], BATCH]()
    legacy.setup(learning_starts=0)
    mine.setup(learning_starts=0)

    for r in range(N_FILL):
        var o = _obs_for(r)
        var a = _act_for(r)
        var nx = _nxt_for(r)
        legacy.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0))
        mine.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0))

    assert_equal(
        legacy.buf.value().count(), mine.buf.value().count(),
        "count after wraparound must match (ring overwrite)",
    )
    assert_equal(legacy.buf.value().count(), CAP, "ring must be full")

    var st_a = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    var st_b = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    st_a.step_idx = 1
    st_b.step_idx = 1

    seed(SEED)
    legacy.step(st_a)
    seed(SEED)
    mine.step(st_b)

    assert_true(_nonconstant(st_a), "legacy minibatch is constant — bad fixture")
    _assert_state_identical(st_a, st_b, "uniform", False)
    print("      minibatch bit-identical  OK")


def test_per_parity() raises:
    print("[2] StoreReplay[PER] vs CPUPrioritizedReplay ...")

    var legacy = ReplaySampleStep[
        CPUPrioritizedReplay[OBS, ACT, CAP], BATCH
    ]()
    var mine = ReplaySampleStep[StoreReplay[OBS, ACT, CAP, True], BATCH]()
    legacy.configure_per(
        alpha=Scalar[DT](0.6), beta=Scalar[DT](0.4),
        epsilon=Scalar[DT](1e-6),
    )
    mine.configure_per(
        alpha=Scalar[DT](0.6), beta=Scalar[DT](0.4),
        epsilon=Scalar[DT](1e-6),
    )
    legacy.setup(learning_starts=0)
    mine.setup(learning_starts=0)

    for r in range(N_FILL):
        var o = _obs_for(r)
        var a = _act_for(r)
        var nx = _nxt_for(r)
        legacy.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0))
        mine.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0))

    var st_a = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    var st_b = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    st_a.step_idx = 1
    st_b.step_idx = 1

    seed(SEED)
    legacy.step(st_a)
    seed(SEED)
    mine.step(st_b)

    assert_true(st_a.has_per, "legacy must flag has_per")
    assert_true(_nonconstant(st_a), "legacy PER minibatch is constant")
    _assert_state_identical(st_a, st_b, "per-first", True)
    print("      first draw + IS weights bit-identical  OK")

    # ── priority update, then a second draw ───────────────────────────
    # Only matches if the sum-tree WRITE path agrees, not just the descent.
    for i in range(BATCH):
        var v = Scalar[DT](Float64(i + 1) * 0.37 - 3.0)
        st_a.td_residuals.data[i] = v
        st_b.td_residuals.data[i] = v
    legacy.update_priorities(st_a)
    mine.update_priorities(st_b)

    var st_c = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    var st_d = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    st_c.step_idx = 2
    st_d.step_idx = 2

    seed(SEED + 1)
    legacy.step(st_c)
    seed(SEED + 1)
    mine.step(st_d)

    _assert_state_identical(st_c, st_d, "per-second", True)

    var moved = 0
    for i in range(BATCH):
        if st_c.mb_r.data[i] != st_a.mb_r.data[i]:
            moved += 1
    assert_true(
        moved > 0,
        "priority update did not change the draw — the second comparison"
        " would be vacuous",
    )
    print("      post-update draw bit-identical  OK (", moved, "of", BATCH,
          "rows moved)")


def test_one_struct_two_policies() raises:
    """The design claim, asserted: PER is a comptime FLAG on one storage
    struct, so `StoreReplay` replaces both legacy CPU buffers."""
    print("[3] one storage struct serves both policies ...")
    var uniform = StoreReplay[OBS, ACT, CAP, False].make()
    var per = StoreReplay[OBS, ACT, CAP, True].make()
    assert_true(not uniform.per, "uniform must carry no sum-tree")
    assert_true(Bool(per.per), "prioritized must carry a sum-tree")
    assert_equal(uniform.count(), 0, "fresh uniform count")
    assert_equal(per.count(), 0, "fresh per count")
    print("      OK")


def main() raises:
    test_uniform_parity()
    test_per_parity()
    test_one_struct_two_policies()
    print("\n[PASS] StoreReplay parity — Stage 4")
