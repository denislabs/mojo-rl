"""The pinned DEMO prefix — RLPD's symmetric sampling over one storage.

    pixi run mojo run -I . tests/data/test_replay_demo_prefix.mojo             # CPU leg
    pixi run -e apple mojo run -I . tests/data/test_replay_demo_prefix.mojo    # + GPU leg

`StoreReplay.pin_demo_prefix` / `StoreReplayGpu.pin_demo_prefix` declare the
rows added so far as demonstrations. What this pins:

  1. THE PREFIX SURVIVES THE RING. Online adds — one at a time and, on the
     GPU, `add_batch` — wrap inside `[demo_n, CAP)`; every demo row is still
     where it was after 3x the online capacity has been written.
  2. HALF OF EVERY BATCH IS DEMO, HALF IS ONLINE. Lanes `[0, B/2)` carry demo
     markers, lanes `[B/2, B)` online markers from the LIVE window of the
     ring (the last `CAP - demo_n` online rows), and the draws move.
  3. NO ONLINE ROW YET -> every lane is a demo row (the batch is still full).
  4. `pin_demo_prefix` REFUSES a count that is not what was added.

The markers: a demo row's `obs[0]` is `1000 + r`, an online row's is `r`,
so a lane's provenance is legible from the minibatch alone.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceBuffer, DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.data.replay import StoreReplay
from mojo_rl.data.replay_gpu import StoreReplayGpu
from mojo_rl.deep_agents.training.blocks.replay_sample_step import (
    ReplaySampleStep,
)
from mojo_rl.deep_agents.training.trainer_block import TrainerState


comptime OBS: Int = 3
comptime ACT: Int = 2
comptime CAP: Int = 64
comptime BATCH: Int = 16
comptime N_DEMO: Int = 20
comptime N_ONLINE: Int = 3 * (CAP - N_DEMO) + 7   # wraps the online ring 3x
comptime N_ENVS: Int = 4
comptime DEMO_MARK: Float64 = 1000.0


def _obs(mark: Float64) -> List[Scalar[DT]]:
    var o = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    o[0] = Scalar[DT](mark)
    for i in range(1, OBS):
        o[i] = Scalar[DT](Float64(i))
    return o^


def _act() -> List[Scalar[DT]]:
    return List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.5))


def _check_batch(
    ref marks: List[Float64], online_lo: Int, online_hi: Int, what: String
) raises:
    """Lanes [0, B/2) demo, [B/2, B) online in [online_lo, online_hi)."""
    for k in range(BATCH // 2):
        assert_true(
            marks[k] >= DEMO_MARK and marks[k] < DEMO_MARK + Float64(N_DEMO),
            what + ": lane " + String(k) + " = " + String(marks[k])
            + " is not a demo row",
        )
    for k in range(BATCH // 2, BATCH):
        assert_true(
            marks[k] >= Float64(online_lo) and marks[k] < Float64(online_hi),
            what + ": lane " + String(k) + " = " + String(marks[k])
            + " is outside the live online window ["
            + String(online_lo) + ", " + String(online_hi) + ")",
        )


# ── CPU ────────────────────────────────────────────────────────────────────


def _cpu_marks(ref st: TrainerState[OBS, ACT, BATCH]) -> List[Float64]:
    var out = List[Float64]()
    for k in range(BATCH):
        out.append(Float64(st.mb_s.data[k * OBS]))
    return out^


def test_cpu() raises:
    print("[cpu] StoreReplay demo prefix ...")
    var b = ReplaySampleStep[StoreReplay[OBS, ACT, CAP], BATCH]()
    b.setup(learning_starts=0)
    for r in range(N_DEMO):
        b.add(_obs(DEMO_MARK + Float64(r)), _act(), Scalar[DT](1), _obs(0), Scalar[DT](0))

    # 4. refuses a count that is not what was added
    var refused = False
    try:
        b.pin_demo_prefix(N_DEMO - 1)
    except:
        refused = True
    assert_true(refused, "pin_demo_prefix must refuse n != rows added")
    b.pin_demo_prefix(N_DEMO)
    assert_equal(b.demo_count(), N_DEMO)

    # 3. no online row yet: every lane is a demo row
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    st.step_idx = 1
    b.step(st)
    assert_true(st.did_step, "a full prefix is enough for a batch")
    var m0 = _cpu_marks(st)
    for k in range(BATCH):
        assert_true(m0[k] >= DEMO_MARK, "lane " + String(k) + " is not demo before any online row")

    # 1. the prefix survives 3x the online capacity
    for r in range(N_ONLINE):
        b.add(_obs(Float64(r)), _act(), Scalar[DT](0), _obs(0), Scalar[DT](0))
    ref buf = b.buf.value()
    assert_equal(buf.count(), CAP, "the ring saturates at CAP")
    for r in range(N_DEMO):
        assert_equal(
            Float64(buf.obs[r * OBS]), DEMO_MARK + Float64(r),
            "demo row " + String(r) + " was overwritten",
        )
    # the online region holds exactly the last CAP - N_DEMO online rows
    var lo = N_ONLINE - (CAP - N_DEMO)
    var seen = List[Bool](length=N_ONLINE, fill=False)
    for slot in range(N_DEMO, CAP):
        var v = Int(Float64(buf.obs[slot * OBS]))
        assert_true(v >= lo and v < N_ONLINE, "online slot holds " + String(v))
        seen[v] = True
    for r in range(lo, N_ONLINE):
        assert_true(seen[r], "online row " + String(r) + " missing from the ring")

    # 2. half demo, half online, and the draws move
    b.step(st)
    var m1 = _cpu_marks(st)
    _check_batch(m1, lo, N_ONLINE, "cpu draw 1")
    b.step(st)
    var m2 = _cpu_marks(st)
    _check_batch(m2, lo, N_ONLINE, "cpu draw 2")
    var moved = False
    for k in range(BATCH):
        if m1[k] != m2[k]:
            moved = True
    assert_true(moved, "two draws identical")
    print("      prefix intact after " + String(N_ONLINE) + " online adds; batches split " + String(BATCH // 2) + "/" + String(BATCH - BATCH // 2) + "  OK")


# ── GPU ────────────────────────────────────────────────────────────────────


def _gpu_marks(
    ctx: DeviceContext, mut st: TrainerState[OBS, ACT, BATCH]
) raises -> List[Float64]:
    var h = List[Scalar[DT]](unsafe_uninit_length=BATCH * OBS)
    ctx.enqueue_copy(h.unsafe_ptr(), st.mb_s.dev.value())
    ctx.synchronize()
    var out = List[Float64]()
    for k in range(BATCH):
        out.append(Float64(h[k * OBS]))
    return out^


def test_gpu() raises:
    print("[gpu] StoreReplayGpu demo prefix ...")
    var ctx = DeviceContext()
    var b = ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP], BATCH]()
    b.setup(learning_starts=0, ctx=ctx)
    for r in range(N_DEMO):
        b.add(_obs(DEMO_MARK + Float64(r)), _act(), Scalar[DT](1), _obs(0), Scalar[DT](0), ctx=ctx)
    var refused = False
    try:
        b.pin_demo_prefix(N_DEMO + 1, ctx=ctx)
    except:
        refused = True
    assert_true(refused, "pin_demo_prefix must refuse n != rows added")
    b.pin_demo_prefix(N_DEMO, ctx=ctx)
    assert_equal(b.demo_count(), N_DEMO)

    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx)
    st.step_idx = 1
    b.step(st)
    var m0 = _gpu_marks(ctx, st)
    for k in range(BATCH):
        assert_true(m0[k] >= DEMO_MARK, "lane " + String(k) + " is not demo before any online row")

    # online rows: half through `add`, half through `add_batch[N_ENVS]`
    var n_single = N_ONLINE // 2
    for r in range(n_single):
        b.add(_obs(Float64(r)), _act(), Scalar[DT](0), _obs(0), Scalar[DT](0), ctx=ctx)
    var n_batched = ((N_ONLINE - n_single) // N_ENVS) * N_ENVS
    var so = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var sa = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var sr = ctx.enqueue_create_buffer[DT](N_ENVS)
    var sn = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var sd = ctx.enqueue_create_buffer[DT](N_ENVS)
    var ho = List[Scalar[DT]](length=N_ENVS * OBS, fill=Scalar[DT](0))
    var hz = List[Scalar[DT]](length=N_ENVS * OBS, fill=Scalar[DT](0))
    var r0 = n_single
    while r0 < n_single + n_batched:
        for e in range(N_ENVS):
            ho[e * OBS] = Scalar[DT](Float64(r0 + e))
        ctx.enqueue_copy(so, ho.unsafe_ptr())
        ctx.enqueue_copy(sa, hz.unsafe_ptr())
        ctx.enqueue_copy(sr, hz.unsafe_ptr())
        ctx.enqueue_copy(sn, hz.unsafe_ptr())
        ctx.enqueue_copy(sd, hz.unsafe_ptr())
        ctx.synchronize()
        b.add_batch_gpu[N_ENVS](ctx, so, sa, sr, sn, sd)
        r0 += N_ENVS
    var n_total = n_single + n_batched
    ctx.synchronize()
    assert_equal(b.buf.value().count(), CAP, "the ring saturates at CAP")

    var hb = List[Scalar[DT]](unsafe_uninit_length=CAP * OBS)
    ctx.enqueue_copy(hb.unsafe_ptr(), b.buf.value().obs)
    ctx.synchronize()
    for r in range(N_DEMO):
        assert_equal(
            Float64(hb[r * OBS]), DEMO_MARK + Float64(r),
            "demo row " + String(r) + " was overwritten",
        )
    var lo = n_total - (CAP - N_DEMO)
    var seen = List[Bool](length=n_total, fill=False)
    for slot in range(N_DEMO, CAP):
        var v = Int(Float64(hb[slot * OBS]))
        assert_true(v >= lo and v < n_total, "online slot holds " + String(v))
        seen[v] = True
    for r in range(lo, n_total):
        assert_true(seen[r], "online row " + String(r) + " missing from the ring")

    b.step(st)
    var m1 = _gpu_marks(ctx, st)
    _check_batch(m1, lo, n_total, "gpu draw 1")
    b.step(st)
    var m2 = _gpu_marks(ctx, st)
    _check_batch(m2, lo, n_total, "gpu draw 2")
    var moved = False
    for k in range(BATCH):
        if m1[k] != m2[k]:
            moved = True
    assert_true(moved, "two draws identical")
    print("      prefix intact after " + String(n_total) + " online rows (add + add_batch); batches split  OK")


def main() raises:
    test_cpu()
    comptime if has_accelerator():
        test_gpu()
    else:
        print("[gpu] skipped — no accelerator (run with -e apple / -e nvidia)")
    print("REPLAY DEMO PREFIX OK")
