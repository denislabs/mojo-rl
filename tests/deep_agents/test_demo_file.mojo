"""`.demo` round trip — what the recorder writes is what `--demos` reads.

    pixi run mojo run -I . tests/deep_agents/test_demo_file.mojo

Three episodes (one discarded, one failed-and-kept, one success with
intervened rows), written and read back: every float, every flag, the table.
An OPEN episode must not reach the file.
"""

from std.testing import assert_equal, assert_true

from mojo_rl.deep_agents.data.demo_file import (
    DemoSet, read_demo_file, write_demo_file, DEMO_FLAG_INTERVENED,
    DEMO_FLAG_SUCCESS,
)

comptime OBS = 5
comptime ACT = 2
comptime PATH = "/tmp/mojo_rl_test_demo_file.demo"


def _obs(k: Int) -> List[Float64]:
    var o = List[Float64]()
    for i in range(OBS):
        o.append(Float64(k) * 1.5 + Float64(i) * 0.25 - 3.0)
    return o^


def _act(k: Int) -> List[Float64]:
    var a = List[Float64]()
    for j in range(ACT):
        a.append(Float64(k) * -0.125 + Float64(j))
    return a^


def main() raises:
    var d = DemoSet(OBS, ACT)
    # episode A: discarded
    d.begin_episode()
    for k in range(4):
        d.add(_obs(k), _act(k), 0.1, _obs(k + 1), 0.0)
    d.discard_episode()
    assert_equal(d.count(), 0, "a discarded episode leaves no rows")
    # episode B: a failure, kept
    d.begin_episode()
    for k in range(10, 13):
        d.add(_obs(k), _act(k), Float64(k) * 0.01, _obs(k + 1), 0.0)
    d.end_episode(success=False)
    # episode C: a success with two intervened rows
    d.begin_episode()
    for k in range(20, 25):
        d.add(_obs(k), _act(k), Float64(k) * 0.01, _obs(k + 1), 0.0, intervened=(k >= 23))
    d.end_episode(success=True)
    # episode D: OPEN — must not be written
    d.begin_episode()
    d.add(_obs(99), _act(99), 9.0, _obs(100), 1.0)
    assert_equal(d.open_rows(), 1)

    write_demo_file(String(PATH), d)
    var r = read_demo_file(String(PATH))
    assert_equal(r.obs_dim, OBS)
    assert_equal(r.act_dim, ACT)
    assert_equal(r.count(), 8, "3 + 5 closed rows; the open one stays out")
    assert_equal(r.n_episodes(), 2)
    assert_equal(r.n_successes(), 1)
    assert_equal(r.n_intervened(), 2)
    assert_equal(r.ep_start[0], 0)
    assert_equal(r.ep_len[0], 3)
    assert_true(not r.ep_success[0])
    assert_equal(r.ep_start[1], 3)
    assert_equal(r.ep_len[1], 5)
    assert_true(r.ep_success[1])
    # the values, exactly (float32 both sides)
    var ks = List[Int]()
    for k in range(10, 13):
        ks.append(k)
    for k in range(20, 25):
        ks.append(k)
    var o = List[Float32](length=OBS, fill=0)
    var a = List[Float32](length=ACT, fill=0)
    var n = List[Float32](length=OBS, fill=0)
    for row in range(8):
        var k = ks[row]
        r.row_obs[DType.float32](row, o)
        r.row_act[DType.float32](row, a)
        r.row_next_obs[DType.float32](row, n)
        var eo = _obs(k)
        var ea = _act(k)
        var en = _obs(k + 1)
        for i in range(OBS):
            assert_equal(o[i], Float32(eo[i]), "obs row " + String(row))
            assert_equal(n[i], Float32(en[i]), "next_obs row " + String(row))
        for j in range(ACT):
            assert_equal(a[j], Float32(ea[j]), "act row " + String(row))
        assert_equal(r.rew[row], Float32(Float64(k) * 0.01), "reward row " + String(row))
        assert_equal(r.done[row], Float32(0))
        assert_equal(r.row_success(row), row >= 3, "success flag row " + String(row))
        assert_equal(r.row_intervened(row), k >= 23, "intervened flag row " + String(row))
    print("  " + r.summary())
    print("DEMO FILE OK")
