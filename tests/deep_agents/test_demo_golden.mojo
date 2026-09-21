"""`.demo` GOLDEN FILE — MRLDEMO1 stays readable, byte for byte.

    pixi run mojo run -I . tests/deep_agents/test_demo_golden.mojo
    pixi run mojo run -I . tests/deep_agents/test_demo_golden.mojo --write   # re-bless (never casually)

`tests/fixtures/demo_v1_small.demo` was written by the recorder's code BEFORE
the demos/ refactor. Recordings made then are on disk and in projects, and
`--demos` must keep reading them: this gate reads the committed file, checks
every field against the set it was made from, and checks that writing that
set today produces the SAME BYTES. A format change fails here first.
"""

from std.sys import argv
from std.testing import assert_equal, assert_true

from noeira.deep_agents.demos.file import (
    DemoSet, read_demo_file, write_demo_file,
)

comptime OBS = 4
comptime ACT = 3
comptime FIXTURE = "tests/fixtures/demo_v1_small.demo"
comptime TMP = "/tmp/noeira_test_demo_golden.demo"


def _set() raises -> DemoSet:
    """Three episodes: a failure, a success with intervened rows, a success.
    Values are exact in float32 (multiples of 1/8), so no rounding can hide."""
    var d = DemoSet(OBS, ACT)
    var k = 0
    for ep in range(3):
        d.begin_episode()
        var n = 4 + ep * 3
        for t in range(n):
            var o = List[Float64]()
            var no = List[Float64]()
            for i in range(OBS):
                o.append(Float64(k * OBS + i) * 0.125 - 2.0)
                no.append(Float64((k + 1) * OBS + i) * 0.125 - 2.0)
            var a = List[Float64]()
            for j in range(ACT):
                a.append(Float64((k + j) % 16) * 0.125 - 1.0)
            var done = 1.0 if t == n - 1 else 0.0
            d.add(o, a, Float64(k) * 0.25, no, done, intervened=(ep == 1 and t >= 3))
            k += 1
        d.end_episode(success=(ep != 0))
    return d^


def _bytes(path: String) raises -> List[UInt8]:
    with open(path, "r") as fh:
        return fh.read_bytes()


def main() raises:
    var args = argv()
    if len(args) > 1 and String(args[1]) == "--write":
        var d = _set()
        write_demo_file(String(FIXTURE), d)
        print("wrote", FIXTURE)
        return

    var want = _set()
    var got = read_demo_file(String(FIXTURE))
    assert_equal(got.obs_dim, OBS)
    assert_equal(got.act_dim, ACT)
    assert_equal(got.count(), want.count(), "row count")
    assert_equal(got.n_episodes(), 3)
    assert_equal(got.n_successes(), 2)
    assert_equal(got.n_intervened(), want.n_intervened())
    assert_true(want.n_intervened() > 0, "the fixture carries intervened rows")
    for r in range(want.count()):
        assert_equal(got.row_intervened(r), want.row_intervened(r))
        assert_equal(got.row_success(r), want.row_success(r))
    for i in range(len(want.obs)):
        assert_equal(got.obs[i], want.obs[i])
        assert_equal(got.nobs[i], want.nobs[i])
    for i in range(len(want.act)):
        assert_equal(got.act[i], want.act[i])
    for i in range(len(want.rew)):
        assert_equal(got.rew[i], want.rew[i])
        assert_equal(got.done[i], want.done[i])
    for e in range(3):
        assert_equal(got.ep_start[e], want.ep_start[e])
        assert_equal(got.ep_len[e], want.ep_len[e])
        assert_equal(got.ep_success[e], want.ep_success[e])

    write_demo_file(String(TMP), want)
    var a = _bytes(String(FIXTURE))
    var b = _bytes(String(TMP))
    assert_equal(len(a), len(b), "today's writer, same size")
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error("byte " + String(i) + " differs from the golden file")
    print("demo golden: MRLDEMO1 read field by field, rewritten byte for byte (",
          len(a), "bytes,", want.count(), "rows )")
