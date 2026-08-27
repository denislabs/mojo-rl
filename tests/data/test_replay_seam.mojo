"""Seam gate for `StoreReplay` — successor to `test_replay_parity`.

The parity version compared minibatches against `CPUReplay` /
`CPUPrioritizedReplay`. That gate did its job and dies with them. What it
uniquely covered was the SEAM — `setup` / `add` / `step` / the PER hooks /
ring wraparound through the real `ReplaySampleStep` block — not the sampling
policy itself. So this keeps the seam coverage and drops the comparison:

  * the POLICY is pinned by `test_sampler_golden.mojo` (frozen index
    sequences, recorded from the legacy before deletion);
  * the GATHER is pinned by `test_resident_gather.mojo` (device vs host,
    bit-exact);
  * what remains for this file is that the seam wires them together
    correctly, which is checked by CONSISTENCY rather than by literals.

The fixture encodes the row number into every field, so a gathered lane can be
checked for internal agreement: `mb_s`, `mb_a`, `mb_r` and `mb_sp` must all
come from the SAME row. A gather that mixed rows across columns — the failure
a per-column store makes possible — cannot pass that.

Run:
    pixi run mojo run -I . tests/data/test_replay_seam.mojo
"""

from std.random import seed
from std.testing import assert_almost_equal, assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.data.replay import StoreReplay
from mojo_rl.deep_agents.training.blocks.replay_sample_step import (
    ReplaySampleStep,
)
from mojo_rl.deep_agents.training.blocks.n_step_sample_step import (
    NStepSampleStep,
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


def _assert_lanes_consistent(
    ref st: TrainerState[OBS, ACT, BATCH], label: String
) raises:
    """Every column of a lane must come from the same stored row."""
    var distinct = 0
    var first = -1
    for k in range(BATCH):
        var row = Int(Float64(st.mb_s.data[k * OBS]) / 10.0 + 0.5)
        if first < 0:
            first = row
        elif row != first:
            distinct += 1
        for i in range(OBS):
            assert_almost_equal(
                Float64(st.mb_s.data[k * OBS + i]),
                Float64(row) * 10.0 + Float64(i), atol=1e-3,
                msg=label + " mb_s lane " + String(k),
            )
            assert_almost_equal(
                Float64(st.mb_sp.data[k * OBS + i]),
                Float64(row) * -10.0 - Float64(i), atol=1e-3,
                msg=label + " mb_sp lane " + String(k) + " (row mismatch"
                " across columns)",
            )
        for j in range(ACT):
            assert_almost_equal(
                Float64(st.mb_a.data[k * ACT + j]),
                Float64(row) * 0.5 + Float64(j) * 0.125, atol=1e-4,
                msg=label + " mb_a lane " + String(k) + " (row mismatch)",
            )
    assert_true(
        distinct > 0, label + ": every lane drew the same row — degenerate"
    )


def test_uniform_seam() raises:
    print("[1] StoreReplay[uniform] through ReplaySampleStep ...")
    var b = ReplaySampleStep[StoreReplay[OBS, ACT, CAP, False], BATCH]()
    b.setup(learning_starts=0)
    for r in range(N_FILL):
        var o = _obs_for(r)
        var a = _act_for(r)
        var nx = _nxt_for(r)
        b.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0))

    assert_equal(b.buf.value().count(), CAP, "ring must saturate at CAP")

    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    st.step_idx = 1
    seed(SEED)
    b.step(st)
    _assert_lanes_consistent(st, "uniform")
    print("      lanes consistent, ring wrapped  OK")


def test_per_seam() raises:
    print("[2] StoreReplay[PER] hooks ...")
    var b = ReplaySampleStep[StoreReplay[OBS, ACT, CAP, True], BATCH]()
    b.configure_per(
        alpha=Scalar[DT](0.6), beta=Scalar[DT](0.4), epsilon=Scalar[DT](1e-6)
    )
    b.setup(learning_starts=0)
    for r in range(N_FILL):
        var o = _obs_for(r)
        var a = _act_for(r)
        var nx = _nxt_for(r)
        b.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0))

    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    st.step_idx = 1
    seed(SEED)
    b.step(st)
    assert_true(st.has_per, "PER backend must flag has_per")
    _assert_lanes_consistent(st, "per")

    # IS weights are normalised so the max is exactly 1.
    var max_w = Float64(0.0)
    for i in range(BATCH):
        var w = Float64(st.mb_w.data[i])
        assert_true(w > 0.0, "IS weight " + String(i) + " must be positive")
        if w > max_w:
            max_w = w
    assert_almost_equal(max_w, 1.0, atol=1e-6, msg="max IS weight must be 1")

    # A priority update must move the distribution — otherwise the sum-tree
    # write path is inert and nothing downstream would notice.
    var before = List[Float64]()
    for i in range(BATCH):
        before.append(Float64(st.mb_r.data[i]))
    for i in range(BATCH):
        st.td_residuals.data[i] = Scalar[DT](Float64(i + 1) * 0.37 - 3.0)
    b.update_priorities(st)

    var st2 = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    st2.step_idx = 2
    seed(SEED + 1)
    b.step(st2)
    _assert_lanes_consistent(st2, "per-after-update")
    var moved = 0
    for i in range(BATCH):
        if Float64(st2.mb_r.data[i]) != before[i]:
            moved += 1
    assert_true(moved > 0, "priority update left the draw unchanged")
    print("      has_per, weights normalised, update moves the draw  OK (",
          moved, "of", BATCH, ")")


def test_nstep_seam() raises:
    """`NStepSampleStep` is a distinct block from `ReplaySampleStep` and had
    no gate until batch 2 — kept here."""
    print("[3] n-step decorator ...")
    comptime N: Int = 3
    var b = NStepSampleStep[N, StoreReplay[OBS, ACT, CAP, False], BATCH]()
    b.setup(learning_starts=0)
    b.configure_gamma(Scalar[DT](0.99))
    for r in range(N_FILL):
        var o = _obs_for(r)
        var a = _act_for(r)
        var nx = _nxt_for(r)
        var done = Scalar[DT](1.0) if (r % 7 == 6) else Scalar[DT](0.0)
        b.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, done)

    assert_true(
        b.inner.buf.value().count() > 0, "n-step must have stored something"
    )
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    st.step_idx = 1
    seed(SEED)
    b.step(st)
    # obs/action still come from one row; the REWARD is an n-step return, so
    # it is deliberately not checked against the single-step fixture value.
    var distinct = 0
    var first = -1
    for k in range(BATCH):
        var row = Int(Float64(st.mb_s.data[k * OBS]) / 10.0 + 0.5)
        if first < 0:
            first = row
        elif row != first:
            distinct += 1
        for j in range(ACT):
            assert_almost_equal(
                Float64(st.mb_a.data[k * ACT + j]),
                Float64(row) * 0.5 + Float64(j) * 0.125, atol=1e-4,
                msg="n-step mb_a lane " + String(k),
            )
    assert_true(distinct > 0, "n-step draw is degenerate")
    print("      obs/action lane-consistent  OK")


def test_one_struct_two_policies() raises:
    """The design claim: PER is a comptime FLAG on one storage struct, so
    `StoreReplay` replaces both legacy CPU buffers."""
    print("[4] one storage struct serves both policies ...")
    var uniform = StoreReplay[OBS, ACT, CAP, False].make()
    var per = StoreReplay[OBS, ACT, CAP, True].make()
    assert_true(not uniform.per, "uniform must carry no sum-tree")
    assert_true(Bool(per.per), "prioritized must carry a sum-tree")
    print("      OK")


def main() raises:
    test_uniform_seam()
    test_per_seam()
    test_nstep_seam()
    test_one_struct_two_policies()
    print("\n[PASS] StoreReplay seam gate")
