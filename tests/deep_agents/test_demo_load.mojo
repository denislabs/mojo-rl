"""Demos into a replay's pinned prefix — `demos.load_demos_into_replay`.

    pixi run mojo run -I . tests/deep_agents/test_demo_load.mojo

The golden `.demo` (3 episodes: a failure; a success with intervened rows; a
success) into a CPU `ReplaySampleStep`, under each filter: the kept-row
count, the pinned prefix, the mean reward. Then the refusals: a file of
another width, and a filter that keeps nothing.
"""

from std.testing import assert_equal, assert_true

from noeira.data.replay import StoreReplay
from noeira.deep_agents.training.blocks.replay_sample_step import (
    ReplaySampleStep,
)
from noeira.deep_agents.demos.file import (
    DemoSet, read_demo_file, write_demo_file,
)
from noeira.deep_agents.demos.filter import DemoFilter
from noeira.deep_agents.demos.load import load_demos_into_replay

comptime OBS = 4
comptime ACT = 3
comptime CAP = 64
comptime BATCH = 8
comptime GOLDEN = "tests/fixtures/demo_v1_small.demo"
comptime NO_SUCCESS = "/tmp/noeira_test_demo_load_fail.demo"


def _expected(f: DemoFilter) raises -> Tuple[Int, Float64]:
    var ds = read_demo_file(String(GOLDEN))
    var n = 0
    var s = 0.0
    for r in range(ds.count()):
        if f.keeps(ds, r):
            n += 1
            s += Float64(ds.rew[r])
    return (n, s / Float64(n))


def _load(f: DemoFilter, path: String) raises -> Tuple[Int, Int, Float64, Int]:
    var b = ReplaySampleStep[StoreReplay[OBS, ACT, CAP], BATCH]()
    b.setup(learning_starts=0)
    var paths = List[String]()
    paths.append(path)
    var rep = load_demos_into_replay(b, paths, f, None, "test")
    return (rep.n_rows, rep.n_file_rows, rep.mean_reward, b.demo_count())


def main() raises:
    for k in range(3):
        var f = DemoFilter(k)
        var want = _expected(f)
        var got = _load(f, String(GOLDEN))
        assert_equal(got[0], want[0], f.name() + ": kept rows")
        assert_equal(got[1], 21, f.name() + ": rows in the file")
        assert_equal(got[2], want[1], f.name() + ": mean reward")
        assert_equal(got[3], want[0], f.name() + ": the pinned prefix is every kept row")
        print("  ", f.name(), ": ", got[0], "rows pinned, mean reward", got[2])
    assert_equal(_expected(DemoFilter(DemoFilter.INTERVENED))[0], 4)

    assert_equal(DemoFilter.parse("success").kind, DemoFilter.SUCCESS)
    var refused = False
    try:
        _ = DemoFilter.parse("successful", "sac task")
    except e:
        refused = String(e).startswith("sac task: --demo-filter must be")
    assert_true(refused, "an unknown filter is refused, naming the flag")

    # a file whose widths are not the block's
    var wide = DemoSet(OBS + 1, ACT)
    wide.begin_episode()
    var o = List[Float64](length=OBS + 1, fill=0.0)
    var o_next = List[Float64](length=OBS + 1, fill=0.0)
    var a = List[Float64](length=ACT, fill=0.0)
    wide.add(o, a, 1.0, o_next, 1.0)
    wide.end_episode(success=True)
    write_demo_file(String(NO_SUCCESS), wide)
    refused = False
    try:
        _ = _load(DemoFilter(), String(NO_SUCCESS))
    except e:
        refused = "recorded on another family" in String(e)
    assert_true(refused, "a file of another width is refused")

    # a filter that keeps nothing: one failed episode, filter=success
    var fail = DemoSet(OBS, ACT)
    fail.begin_episode()
    var o4 = List[Float64](length=OBS, fill=0.5)
    var o4_next = List[Float64](length=OBS, fill=0.5)
    fail.add(o4, a, 0.25, o4_next, 1.0)
    fail.end_episode(success=False)
    write_demo_file(String(NO_SUCCESS), fail)
    refused = False
    try:
        _ = _load(DemoFilter(DemoFilter.SUCCESS), String(NO_SUCCESS))
    except e:
        refused = "kept none" in String(e)
    assert_true(refused, "a filter that keeps nothing is refused")
    print("demo load: 3 filters pinned exactly, 3 refusals")
