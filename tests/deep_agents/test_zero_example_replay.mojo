"""Unit tests for MCTSExampleReplay — record/sample pairing integrity + ring.

Run:
    pixi run mojo run -I . tests/deep_agents/test_zero_example_replay.mojo
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.zero.example_replay import MCTSExampleReplay


def test_record_len_and_ring() raises:
    comptime OBS = 4
    comptime TGT = 3
    comptime CAP = 8
    var rb = MCTSExampleReplay[OBS, TGT, CAP]()
    var o = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var t = List[Scalar[DT]](length=TGT, fill=Scalar[DT](0))
    for k in range(5):
        for j in range(OBS):
            o[j] = Scalar[DT](k)
        for j in range(TGT):
            t[j] = Scalar[DT](k)
        rb.record(o, 0, t, 0)
    assert_equal(len(rb), 5)
    # Overflow capacity → size clamps at CAP.
    for k in range(5, 20):
        for j in range(OBS):
            o[j] = Scalar[DT](k)
        for j in range(TGT):
            t[j] = Scalar[DT](k)
        rb.record(o, 0, t, 0)
    assert_equal(len(rb), CAP)
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
    var o = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var t = List[Scalar[DT]](length=W, fill=Scalar[DT](0))
    for b in range(N):
        for j in range(OBS):
            o[j] = Scalar[DT](b)
        for a in range(ACT):
            t[a] = Scalar[DT](1.0 if a == (b % ACT) else 0.0)
        t[ACT] = Scalar[DT](Float64(b) * 0.1)
        rb.record(o, 0, t, 0)

    # Sampling now lands straight in storage `Tensor`s (`sample_batch_tensors`),
    # the same bridge the selfplay drivers use.
    var so = Tensor.alloc(SB * OBS)
    var st = Tensor.alloc(SB * W)
    rb.sample_batch_tensors[SB](so, st)
    for i in range(SB):
        var bval = Int(Float64(so.data[i * OBS]))
        assert_true(bval >= 0 and bval < N, "sampled obs out of range")
        # obs row is uniform → every cell equals bval
        for j in range(OBS):
            assert_equal(Int(Float64(so.data[i * OBS + j])), bval)
        var amax = 0
        for a in range(1, ACT):
            if st.data[i * W + a] > st.data[i * W + amax]:
                amax = a
        assert_equal(amax, bval % ACT)  # policy paired with obs
        assert_true(
            abs(Float64(st.data[i * W + ACT]) - Float64(bval) * 0.1) < 1e-4,
            "z target not paired with obs",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
