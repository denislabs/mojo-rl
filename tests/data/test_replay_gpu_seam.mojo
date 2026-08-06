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

from std.gpu.host import DeviceBuffer, DeviceContext
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
comptime N_ENVS: Int = 4
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


def _dev_from(ctx: DeviceContext, ref vals: List[Scalar[DT]]) raises -> DeviceBuffer[DT]:
    var b = ctx.enqueue_create_buffer[DT](len(vals))
    ctx.enqueue_copy(b, vals.unsafe_ptr())
    return b^


def _host(ctx: DeviceContext, ref b: DeviceBuffer[DT], n: Int) raises -> List[Scalar[DT]]:
    var h = List[Scalar[DT]](unsafe_uninit_length=n)
    ctx.enqueue_copy(h.unsafe_ptr(), b)
    ctx.synchronize()
    return h^


def test_add_batch_and_raw_gather() raises:
    """The two DEVICE-SOURCE surfaces the tests above never touch:
    `add_batch[N_ENVS]` (the GPU-batched driver's store) and
    `sample` / `sample_range` (MBPO's raw-DeviceBuffer gather).

    Both take their buffers as READ-ONLY arguments, which is exactly what
    made them fail to instantiate: a `LayoutTensor` built from a borrowed
    `DeviceBuffer` carries an immutable origin and does not convert to the
    kernel's `MutAnyOrigin` parameter. Being generic, neither body was
    compiled at all until a call site instantiated it — no gate did, so the
    defect surfaced only in the SAC batched physics3d smoke test.

    `sample_range(row, row+1)` has span 1, so the drawn index is `row`
    deterministically: that turns the random sampler into an exact
    gather-by-index and lets this check VALUES, not just that it compiles.
    """
    print("[6] add_batch[N_ENVS] + raw-buffer gather ...")
    var ctx = DeviceContext()
    var buf = StoreReplayGpu[OBS, ACT, CAP].make(ctx)

    comptime ROUNDS = 2
    for rd in range(ROUNDS):
        var ho = List[Scalar[DT]](length=N_ENVS * OBS, fill=Scalar[DT](0))
        var ha = List[Scalar[DT]](length=N_ENVS * ACT, fill=Scalar[DT](0))
        var hr = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0))
        var hn = List[Scalar[DT]](length=N_ENVS * OBS, fill=Scalar[DT](0))
        var hd = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0))
        for e in range(N_ENVS):
            var row = rd * N_ENVS + e
            var o = _obs_for(row)
            var nx = _nxt_for(row)
            var a = _act_for(row)
            for i in range(OBS):
                ho[e * OBS + i] = o[i]
                hn[e * OBS + i] = nx[i]
            for j in range(ACT):
                ha[e * ACT + j] = a[j]
            hr[e] = Scalar[DT](Float64(row) * 0.25)
            hd[e] = Scalar[DT](Float64(row % 2))
        var so = _dev_from(ctx, ho)
        var sa = _dev_from(ctx, ha)
        var sr = _dev_from(ctx, hr)
        var sn = _dev_from(ctx, hn)
        var sd = _dev_from(ctx, hd)
        buf.add_batch[N_ENVS](ctx, so, sa, sr, sn, sd)
        ctx.synchronize()

    comptime STORED = ROUNDS * N_ENVS
    assert_equal(buf.count(), STORED, "add_batch must advance size by N_ENVS")

    var mb_s = ctx.enqueue_create_buffer[DT](OBS)
    var mb_a = ctx.enqueue_create_buffer[DT](ACT)
    var mb_r = ctx.enqueue_create_buffer[DT](1)
    var mb_sp = ctx.enqueue_create_buffer[DT](OBS)
    var mb_d = ctx.enqueue_create_buffer[DT](1)
    for row in range(STORED):
        buf.sample_range[1](ctx, row, row + 1, mb_s, mb_a, mb_r, mb_sp, mb_d)
        var gs = _host(ctx, mb_s, OBS)
        var ga = _host(ctx, mb_a, ACT)
        var gr = _host(ctx, mb_r, 1)
        var gn = _host(ctx, mb_sp, OBS)
        var gd = _host(ctx, mb_d, 1)
        var eo = _obs_for(row)
        var en = _nxt_for(row)
        var ea = _act_for(row)
        for i in range(OBS):
            assert_almost_equal(
                Float64(gs[i]), Float64(eo[i]), atol=1e-6,
                msg="obs[" + String(i) + "] of stored row " + String(row),
            )
            assert_almost_equal(
                Float64(gn[i]), Float64(en[i]), atol=1e-6,
                msg="next_obs[" + String(i) + "] of stored row " + String(row),
            )
        for j in range(ACT):
            assert_almost_equal(
                Float64(ga[j]), Float64(ea[j]), atol=1e-6,
                msg="act[" + String(j) + "] of stored row " + String(row),
            )
        assert_almost_equal(
            Float64(gr[0]), Float64(row) * 0.25, atol=1e-6,
            msg="reward of stored row " + String(row),
        )
        assert_almost_equal(
            Float64(gd[0]), Float64(row % 2), atol=1e-6,
            msg="done of stored row " + String(row),
        )
    print("      all", STORED, "batched rows round-trip exactly  OK")

    # `sample` (uniform, no range) — the other raw-buffer entry point.
    var us = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var ua = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var ur = ctx.enqueue_create_buffer[DT](N_ENVS)
    var un = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var ud = ctx.enqueue_create_buffer[DT](N_ENVS)
    buf.sample[N_ENVS](ctx, us, ua, ur, un, ud)
    var hs = _host(ctx, us, N_ENVS * OBS)
    for k in range(N_ENVS):
        # obs[0] of stored row r is r*10, so the decoded row must be live.
        var r = Int(Float64(hs[k * OBS]) / 10.0 + 0.5)
        assert_true(
            r >= 0 and r < STORED,
            "uniform raw draw returned row " + String(r) + " outside [0, "
            + String(STORED) + ")",
        )
        assert_almost_equal(
            Float64(hs[k * OBS]), Float64(r) * 10.0, atol=1e-6,
            msg="raw gather produced a row that was never stored",
        )
    print("      uniform raw gather stays inside the live ring  OK")


def main() raises:
    test_uniform_seam()
    test_ere_recency()
    test_ere_per_conflict_raises()
    test_per_seam()
    test_u8_roundtrip_exact()
    test_add_batch_and_raw_gather()
    print("\n[PASS] StoreReplayGpu seam gate")
