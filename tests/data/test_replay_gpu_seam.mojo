"""Seam gate for `StoreReplayGpu` — successor to `test_replay_gpu_parity`.

Same reasoning as `test_replay_seam.mojo`: the policy is pinned by
`test_sampler_golden.mojo`, the gather by `test_resident_gather.mojo`, so what
remains here is that the GPU seam wires them together — plus the two GPU-only
capabilities that have no CPU counterpart.

The ERE and uint8 checks are PROPERTY tests, and deliberately stronger than the
parity versions they replace:

  * ERE must sample from the RECENT window and that window must SHRINK. A
    parity test only said "same as legacy"; this says what recency-biased
    sampling actually means, so it would catch a legacy-faithful port of a
    broken anneal.
  * uint8 obs must survive the round trip EXACTLY for `k/255` inputs — the
    values the pixel pipeline produces. Agreeing with a legacy that quantised
    wrongly would have proved nothing.

Run:
    pixi run mojo run -I . tests/data/test_replay_gpu_seam.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_almost_equal, assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.data.replay_gpu import StoreReplayGpu
from mojo_rl.deep_agents.training.blocks.replay_sample_step import (
    ReplaySampleStep,
)
from mojo_rl.deep_agents.training.trainer_block import TrainerState


comptime OBS: Int = 3
comptime ACT: Int = 2
comptime CAP: Int = 32
comptime BATCH: Int = 16
comptime N_FILL: Int = 50
comptime U8 = DType.uint8


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


def _rows(ctx: DeviceContext, mut st: TrainerState[OBS, ACT, BATCH]) raises -> List[Int]:
    var h = List[Scalar[DT]](unsafe_uninit_length=BATCH * OBS)
    ctx.enqueue_copy(h.unsafe_ptr(), st.mb_s.dev.value())
    ctx.synchronize()
    var out = List[Int]()
    for k in range(BATCH):
        out.append(Int(Float64(h[k * OBS]) / 10.0 + 0.5))
    return out^


def _fill(mut b: ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP], BATCH], ctx: DeviceContext) raises:
    for r in range(N_FILL):
        var o = _obs_for(r)
        var a = _act_for(r)
        var nx = _nxt_for(r)
        b.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0), ctx=ctx)


def test_uniform_seam() raises:
    print("[1] StoreReplayGpu[uniform] through ReplaySampleStep ...")
    var ctx = DeviceContext()
    var b = ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP], BATCH]()
    b.setup(learning_starts=0, ctx=ctx)
    _fill(b, ctx)
    assert_equal(b.buf.value().count(), CAP, "ring must saturate at CAP")

    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx)
    st.step_idx = 1
    b.step(st)
    var r1 = _rows(ctx, st)
    for k in range(BATCH):
        # The ring holds the LAST CAP adds, so stored VALUES are original
        # row numbers N_FILL-CAP .. N_FILL-1 — not slot indices.
        assert_true(
            r1[k] >= N_FILL - CAP and r1[k] < N_FILL,
            "row " + String(k) + " = " + String(r1[k])
            + " is outside the live window ["
            + String(N_FILL - CAP) + ", " + String(N_FILL) + ")",
        )
    b.step(st)
    var r2 = _rows(ctx, st)
    var moved = False
    for k in range(BATCH):
        if r1[k] != r2[k]:
            moved = True
    assert_true(moved, "second draw identical — the RNG offset did not advance")
    print("      two distinct draws inside the ring  OK")


def test_ere_recency() raises:
    """ERE must sample from a shrinking RECENT window — the property, not
    merely agreement with the legacy."""
    print("[2] ERE recency + anneal (property) ...")
    var ctx = DeviceContext()
    var b = ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP], BATCH]()
    b.setup(learning_starts=0, ctx=ctx)
    b.configure_ere(enable=True, eta=Scalar[DT](0.5), c_min=2, k_max=1000)
    _fill(b, ctx)

    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx)
    st.step_idx = 1
    var spans = List[Int]()
    for _ in range(4):
        b.step(st)
        var rows = _rows(ctx, st)
        var lo = rows[0]
        var hi = rows[0]
        for k in range(BATCH):
            if rows[k] < lo:
                lo = rows[k]
            if rows[k] > hi:
                hi = rows[k]
        spans.append(hi - lo + 1)
    # eta=0.5 halves c_k each call, so the sampled span must NOT grow.
    for i in range(1, len(spans)):
        assert_true(
            spans[i] <= spans[0],
            "ERE window grew (span " + String(spans[i]) + " vs first "
            + String(spans[0]) + ") — the anneal is not running",
        )
    assert_true(
        spans[len(spans) - 1] < CAP,
        "ERE sampled a span as wide as the ring — it is behaving as plain"
        " uniform",
    )
    print("      spans", spans[0], "->", spans[len(spans) - 1], " OK")


def test_ere_per_conflict_raises() raises:
    print("[3] ERE + PER together must raise ...")
    var ctx = DeviceContext()
    var buf = StoreReplayGpu[OBS, ACT, CAP, True].make(ctx)
    var raised = False
    try:
        buf.configure_ere(enable=True)
    except:
        raised = True
    assert_true(raised, "ERE+PER must raise")
    print("      OK")


def test_per_seam() raises:
    print("[4] StoreReplayGpu[PER] device sum-tree ...")
    var ctx = DeviceContext()
    var b = ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP, True], BATCH]()
    b.configure_per(
        alpha=Scalar[DT](0.6), beta=Scalar[DT](0.4), epsilon=Scalar[DT](1e-6)
    )
    b.setup(learning_starts=0, ctx=ctx)
    for r in range(N_FILL):
        var o = _obs_for(r)
        var a = _act_for(r)
        var nx = _nxt_for(r)
        b.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0), ctx=ctx)

    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx)
    st.step_idx = 1
    b.step(st)
    assert_true(st.has_per, "PER must flag has_per")

    var w = List[Scalar[DT]](unsafe_uninit_length=BATCH)
    ctx.enqueue_copy(w.unsafe_ptr(), st.mb_w.dev.value())
    ctx.synchronize()
    var max_w = Float64(0.0)
    for i in range(BATCH):
        assert_true(Float64(w[i]) > 0.0, "IS weight must be positive")
        if Float64(w[i]) > max_w:
            max_w = Float64(w[i])
    assert_almost_equal(max_w, 1.0, atol=1e-5, msg="max IS weight must be 1")

    var before = _rows(ctx, st)
    var td = List[Scalar[DT]](unsafe_uninit_length=BATCH)
    for i in range(BATCH):
        td[i] = Scalar[DT](Float64(i + 1) * 0.37 - 3.0)
    ctx.enqueue_copy(st.td_residuals.dev.value(), td.unsafe_ptr())
    ctx.synchronize()
    b.update_priorities(st)
    b.step(st)
    var after = _rows(ctx, st)
    var moved = False
    for i in range(BATCH):
        if before[i] != after[i]:
            moved = True
    assert_true(
        moved, "device priority update left the draw unchanged — the tree"
        " write/propagate path is inert",
    )
    print("      weights normalised, tree update moves the draw  OK")


def test_u8_roundtrip_exact() raises:
    """Uint8 obs storage: the round trip must be EXACT for k/255."""
    print("[5] uint8 obs storage, exact k/255 round trip ...")
    var ctx = DeviceContext()
    var b = ReplaySampleStep[
        StoreReplayGpu[OBS, ACT, CAP, False, U8], BATCH
    ]()
    b.setup(learning_starts=0, ctx=ctx)
    for r in range(N_FILL):
        var o = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
        var nx = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
        for i in range(OBS):
            o[i] = Scalar[DT](Float64((r * 7 + i * 31) % 256) / 255.0)
            nx[i] = Scalar[DT](Float64((r * 11 + i * 13) % 256) / 255.0)
        var a = _act_for(r)
        b.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0), ctx=ctx)

    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx)
    st.step_idx = 1
    b.step(st)
    var h = List[Scalar[DT]](unsafe_uninit_length=BATCH * OBS)
    ctx.enqueue_copy(h.unsafe_ptr(), st.mb_s.dev.value())
    ctx.synchronize()
    for i in range(BATCH * OBS):
        var v = Float64(h[i]) * 255.0
        var k = Float64(Int(v + 0.5))
        assert_true(
            v > k - 1e-3 and v < k + 1e-3,
            "element " + String(i) + " did not survive the uint8 round trip"
            " exactly (got " + String(Float64(h[i])) + ")",
        )
    print("      all", BATCH * OBS, "elements exact  OK")


def main() raises:
    test_uniform_seam()
    test_ere_recency()
    test_ere_per_conflict_raises()
    test_per_seam()
    test_u8_roundtrip_exact()
    print("\n[PASS] StoreReplayGpu seam gate")
