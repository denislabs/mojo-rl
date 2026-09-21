"""`EpisodeRecorder`, `Handover`, `CtrlRange` — the recorders' shared rules.

    pixi run mojo run -I . tests/deep_agents/test_episode_recorder.mojo

The hold rule ends an episode exactly at `hold_steps` and a broken hold
restarts the count; a failure is dropped unless `keep_failures`; a discard
drops even a success; the file is rewritten after every kept episode and
holds only kept ones; intervened rows are counted and flagged. `Handover`
passes control only after `min_steps`. `CtrlRange` round-trips and clamps.
"""

from std.testing import assert_equal, assert_true, assert_false

from noeira.deep_agents.demos.file import read_demo_file
from noeira.deep_agents.demos.recorder import EpisodeRecorder, Handover
from noeira.deep_agents.demos.ctrl_range import CtrlRange

comptime OBS = 3
comptime ACT = 2
comptime PATH = "/tmp/noeira_test_episode_recorder.demo"


def _step(
    mut rec: EpisodeRecorder, k: Int, holds: Bool, intervened: Bool = False
) raises -> Bool:
    var o = List[Float64](length=OBS, fill=Float64(k))
    var n = List[Float64](length=OBS, fill=Float64(k + 1))
    var a = List[Float64](length=ACT, fill=0.5)
    return rec.record(o, a, 1.0, n, holds, intervened)


def main() raises:
    var rec = EpisodeRecorder(OBS, ACT, String(PATH), False, 3)

    # episode 1: hold broken once, then held 3 -> success at exactly the 3rd
    rec.begin()
    assert_false(_step(rec, 0, True))
    assert_false(_step(rec, 1, False), "a broken hold restarts the count")
    assert_false(_step(rec, 2, True))
    assert_false(_step(rec, 3, True, intervened=True))
    assert_true(_step(rec, 4, True, intervened=True), "held 3 in a row")
    assert_equal(rec.rows, 5)
    assert_equal(rec.intervened_rows, 2)
    assert_equal(rec.ret, 5.0)
    assert_true(rec.end(success=True))
    var f = read_demo_file(String(PATH))
    assert_equal(f.count(), 5, "the file is rewritten after a kept episode")
    assert_equal(f.n_intervened(), 2)

    # episode 2: a failure, failures not kept -> dropped, file unchanged
    rec.begin()
    _ = _step(rec, 10, False)
    assert_false(rec.end(success=False))
    assert_equal(read_demo_file(String(PATH)).count(), 5)

    # episode 3: a success, discarded -> dropped
    rec.begin()
    _ = _step(rec, 20, True)
    assert_false(rec.end(success=True, discard=True))
    # episode 4: no rows -> dropped even when "successful"
    rec.begin()
    assert_false(rec.end(success=True))
    assert_equal(rec.n_saved, 1)
    assert_equal(rec.n_dropped, 3)
    assert_equal(read_demo_file(String(PATH)).n_episodes(), 1)

    # keep_failures
    var keep = EpisodeRecorder(OBS, ACT, String(PATH), True, 3)
    keep.begin()
    _ = _step(keep, 0, False)
    assert_true(keep.end(success=False), "a failure is kept with keep_failures")
    var g = read_demo_file(String(PATH))
    assert_equal(g.n_episodes(), 1)
    assert_equal(g.n_successes(), 0)

    # Handover
    var h = Handover(20, 140)
    assert_false(h.arrived_at(18, True), "not before min_steps")
    assert_true(h.arrived_at(19, True), "the 20th step may hand over")
    assert_false(h.arrived_at(50, False))

    # CtrlRange
    var lo = List[Float64]()
    var hi = List[Float64]()
    lo.append(-1.5)
    hi.append(0.5)
    lo.append(2.0)
    hi.append(2.0)
    var c = CtrlRange(lo^, hi^)
    assert_equal(c.normalize(0, -1.5), -1.0)
    assert_equal(c.normalize(0, 0.5), 1.0)
    assert_equal(c.normalize(0, -0.5), 0.0)
    assert_equal(c.normalize(0, 9.0), 1.0, "clamped")
    assert_equal(c.normalize(0, -9.0), -1.0, "clamped")
    assert_equal(c.normalize(1, 2.0), 0.0, "a zero-width range maps to 0")
    assert_equal(c.denormalize(0, c.normalize(0, -0.25)), -0.25)
    print("episode recorder: hold rule, keep/drop/discard, rewrite, handover, ctrl range")
