"""Unit tests for MCTSExampleReplay — record/sample pairing integrity + ring.

Run:
    pixi run mojo run -I . tests/deep_agents2/test_zero_example_replay.mojo
"""

from std.memory import alloc
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.zero.example_replay import MCTSExampleReplay


def test_record_len_and_ring() raises:
    comptime OBS = 4
    comptime TGT = 3
    comptime CAP = 8
    var rb = MCTSExampleReplay[OBS, TGT, CAP]()
    var o = alloc[Scalar[DT]](OBS)
    var t = alloc[Scalar[DT]](TGT)
    for k in range(5):
        for j in range(OBS):
            o[j] = Scalar[DT](k)
        for j in range(TGT):
            t[j] = Scalar[DT](k)
        rb.record(o, t)
    assert_equal(len(rb), 5)
    # Overflow capacity → size clamps at CAP.
    for k in range(5, 20):
        for j in range(OBS):
            o[j] = Scalar[DT](k)
        for j in range(TGT):
            t[j] = Scalar[DT](k)
        rb.record(o, t)
    assert_equal(len(rb), CAP)
    o.free()
    t.free()


def test_sample_pairing() raises:
    # obs row = constant b; target = [one-hot at b%ACT | z=b*0.1]. Sampling
    # must preserve the (obs,target) pairing: the obs constant determines both
    # the policy argmax and the z value of the SAME sampled row.
    comptime OBS = 27
    comptime ACT = 9
    comptime W = ACT + 1
    comptime CAP = 64
    comptime N = 10
    comptime SB = 16
    var rb = MCTSExampleReplay[OBS, W, CAP]()
    var o = alloc[Scalar[DT]](OBS)
    var t = alloc[Scalar[DT]](W)
    for b in range(N):
        for j in range(OBS):
            o[j] = Scalar[DT](b)
        for a in range(ACT):
            t[a] = Scalar[DT](1.0 if a == (b % ACT) else 0.0)
        t[ACT] = Scalar[DT](Float64(b) * 0.1)
        rb.record(o, t)

    var so = alloc[Scalar[DT]](SB * OBS)
    var st = alloc[Scalar[DT]](SB * W)
    rb.sample_batch[SB](so, st)
    for i in range(SB):
        var bval = Int(Float64(so[i * OBS]))
        assert_true(bval >= 0 and bval < N, "sampled obs out of range")
        # obs row is uniform → every cell equals bval
        for j in range(OBS):
            assert_equal(Int(Float64(so[i * OBS + j])), bval)
        var amax = 0
        for a in range(1, ACT):
            if st[i * W + a] > st[i * W + amax]:
                amax = a
        assert_equal(amax, bval % ACT)  # policy paired with obs
        assert_true(
            abs(Float64(st[i * W + ACT]) - Float64(bval) * 0.1) < 1e-4,
            "z target not paired with obs",
        )
    so.free()
    st.free()
    o.free()
    t.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
