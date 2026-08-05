"""Gate for 4b — `StoreReplayGpu` vs `GPUReplay`, bit-identical.

Same standard as the CPU gate: the minibatch is compared element-by-element
after a device gather, driven through the real `ReplaySampleStep` block. The
device sampler shares `GPUReplay`'s seed and offset-advance, so a migrated
call site draws the *same* Philox sequence — this asserts it.

Covered:
  1. single `add` path, uniform sample, bit-identical minibatch
  2. a SECOND consecutive draw, so a wrong RNG-offset advance cannot pass
  3. ring wraparound (filled past CAP)
  4. `configure_ere(enable=True)` RAISES rather than silently sampling
     uniformly

Run:
    pixi run mojo run -I . tests/data/test_replay_gpu_parity.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.data.replay_gpu import StoreReplayGpu
from mojo_rl.deep_agents.data.gpu_replay import GPUReplay
from mojo_rl.deep_agents.training.blocks.replay_sample_step import (
    ReplaySampleStep,
)
from mojo_rl.deep_agents.training.trainer_block import TrainerState


comptime OBS: Int = 3
comptime ACT: Int = 2
comptime CAP: Int = 32
comptime BATCH: Int = 16
comptime N_FILL: Int = 50          # > CAP so the ring wraps


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


def _readback(
    ctx: DeviceContext, mut st: TrainerState[OBS, ACT, BATCH]
) raises -> List[Scalar[DT]]:
    """mb_s, mb_a, mb_r, mb_sp, mb_d concatenated on the host."""
    var out = List[Scalar[DT]]()
    var s = List[Scalar[DT]](unsafe_uninit_length=BATCH * OBS)
    var a = List[Scalar[DT]](unsafe_uninit_length=BATCH * ACT)
    var r = List[Scalar[DT]](unsafe_uninit_length=BATCH)
    var sp = List[Scalar[DT]](unsafe_uninit_length=BATCH * OBS)
    var d = List[Scalar[DT]](unsafe_uninit_length=BATCH)
    ctx.enqueue_copy(s.unsafe_ptr(), st.mb_s.dev.value())
    ctx.enqueue_copy(a.unsafe_ptr(), st.mb_a.dev.value())
    ctx.enqueue_copy(r.unsafe_ptr(), st.mb_r.dev.value())
    ctx.enqueue_copy(sp.unsafe_ptr(), st.mb_sp.dev.value())
    ctx.enqueue_copy(d.unsafe_ptr(), st.mb_d.dev.value())
    ctx.synchronize()
    for i in range(len(s)):
        out.append(s[i])
    for i in range(len(a)):
        out.append(a[i])
    for i in range(len(r)):
        out.append(r[i])
    for i in range(len(sp)):
        out.append(sp[i])
    for i in range(len(d)):
        out.append(d[i])
    return out^


def test_gpu_parity() raises:
    print("[1] StoreReplayGpu vs GPUReplay, through ReplaySampleStep ...")
    var ctx = DeviceContext()

    var legacy = ReplaySampleStep[GPUReplay[OBS, ACT, CAP], BATCH]()
    var mine = ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP], BATCH]()
    legacy.setup(learning_starts=0, ctx=ctx)
    mine.setup(learning_starts=0, ctx=ctx)

    for r in range(N_FILL):
        var o = _obs_for(r)
        var a = _act_for(r)
        var nx = _nxt_for(r)
        legacy.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0), ctx=ctx)
        mine.add(o, a, Scalar[DT](Float64(r) * 0.25), nx, Scalar[DT](0), ctx=ctx)

    assert_equal(
        legacy.buf.value().count(), mine.buf.value().count(),
        "count after wraparound must match",
    )
    assert_equal(legacy.buf.value().count(), CAP, "ring must be full")

    var st_a = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx)
    var st_b = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx)
    st_a.step_idx = 1
    st_b.step_idx = 1

    # ── first draw ────────────────────────────────────────────────────
    legacy.step(st_a)
    mine.step(st_b)
    var ga = _readback(ctx, st_a)
    var gb = _readback(ctx, st_b)
    assert_equal(len(ga), len(gb), "readback size")

    var nonconstant = False
    for i in range(1, BATCH):
        if ga[i] != ga[0]:
            nonconstant = True
    assert_true(nonconstant, "legacy minibatch is constant — bad fixture")

    for i in range(len(ga)):
        assert_equal(ga[i], gb[i], "first draw element " + String(i))
    print("      first draw bit-identical (", len(ga), "elements )  OK")

    # ── second draw: catches a wrong RNG-offset advance ────────────────
    legacy.step(st_a)
    mine.step(st_b)
    var g2a = _readback(ctx, st_a)
    var g2b = _readback(ctx, st_b)
    for i in range(len(g2a)):
        assert_equal(
            g2a[i], g2b[i],
            "SECOND draw element " + String(i) + " (RNG offset advance)",
        )
    var moved = False
    for i in range(len(ga)):
        if g2a[i] != ga[i]:
            moved = True
    assert_true(moved, "second draw identical to first — RNG did not advance")
    print("      second draw bit-identical  OK")


def test_ere_raises() raises:
    """ERE is an unported index policy; silently sampling uniformly would
    change training with no signal."""
    print("[2] configure_ere(enable=True) must raise ...")
    var ctx = DeviceContext()
    var buf = StoreReplayGpu[OBS, ACT, CAP].make(ctx)
    var raised = False
    try:
        buf.configure_ere(enable=True)
    except:
        raised = True
    assert_true(raised, "ERE must raise, not no-op")
    # Disabled is fine.
    buf.configure_ere(enable=False)
    print("      OK")


def main() raises:
    test_gpu_parity()
    test_ere_raises()
    print("\n[PASS] StoreReplayGpu parity — 4b")
