"""Golden gate for the index policies — successor to `test_sampler_parity`.

The parity version compared each policy against the legacy buffer it replaced.
That was the right gate for the migration, but it dies with the legacy code.
These literals are the SAME sequences, extracted from the legacy buffers before
they were deleted (`scratchpad/dump_golden.mojo`, 2026-08-05), so the
protection survives the deletion.

**Index sequences are the right thing to freeze.** They ARE the policy. The
gather that turns them into a minibatch is separately gated by
`test_resident_gather.mojo` (device-vs-host, bit-exact), so freezing indices
plus a gated gather covers the whole path without pinning megabytes of
minibatch data.

⚠ If a change here fails, that is the gate working: it means the sampled
sequence moved. Do not "re-bless" the literals without knowing why — a sampler
that is subtly *differently* random still trains, just worse, and only visibly
so several algorithms later. Re-blessing is only correct for a deliberate,
documented policy change.

Run:
    pixi run mojo run -I . tests/data/test_sampler_golden.mojo
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_almost_equal, assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.data import (
    PrioritizedSampler,
    SequenceWindowSampler,
    UniformDeviceSampler,
    UniformSampler,
)


comptime SEED: Int = 20260805
comptime CAP: Int = 64
comptime N_FILL: Int = 50
comptime BATCH: Int = 32


# ── goldens, recorded from the legacy buffers before deletion ─────────────

def GOLDEN_UNIFORM_HOST() -> List[Int]:
    return [
        41, 25, 40, 39, 33, 24, 4, 3, 44, 21, 8, 21, 12, 46, 43, 18,
        18, 15, 29, 4, 18, 31, 32, 12, 27, 26, 43, 28, 6, 34, 49, 1,
    ]

def GOLDEN_PER_HOST_1() -> List[Int]:
    return [
        1, 2, 4, 5, 7, 8, 9, 11, 13, 14, 15, 17, 19, 21, 23, 24,
        25, 27, 29, 29, 31, 33, 35, 36, 38, 39, 41, 43, 43, 46, 48, 48,
    ]

def GOLDEN_PER_HOST_2() -> List[Int]:
    return [
        0, 1, 3, 4, 5, 6, 7, 9, 11, 13, 16, 18, 18, 23, 25, 27,
        30, 31, 32, 33, 35, 36, 38, 39, 41, 41, 42, 43, 46, 47, 48, 48,
    ]

def GOLDEN_PER_HOST_W2() -> List[Float64]:
    return [
        0.8663377165794373, 0.5997189283370972, 0.8663377165794373,
        0.6253156065940857, 0.6404967308044434, 0.8663377165794373,
        0.6577981114387512, 0.7014602422714233, 0.7300891876220703,
        0.7660112977027893, 0.8663377165794373, 0.8663377165794373,
        0.8663377165794373, 1.0, 0.8149774670600891, 0.7671163082122803,
        0.8663377165794373, 0.6784096360206604, 0.8663377165794373,
        0.6583003401756287, 0.6409342288970947, 0.6257020235061646,
        0.612172544002533, 0.6000301837921143, 0.5890376567840576,
        0.5890376567840576, 0.8663377165794373, 0.5698100328445435,
        0.5613167881965637, 0.8663377165794373, 0.5461019277572632,
        0.5461019277572632,
    ]

def GOLDEN_SEQ_STARTS() -> List[Int]:
    return [36, 22, 35, 34, 29, 21, 3, 3]

def GOLDEN_UNIFORM_DEV_1() -> List[Int]:
    return [
        24, 47, 30, 23, 38, 26, 49, 46, 36, 42, 48, 9, 7, 9, 14, 15,
        44, 20, 41, 30, 45, 33, 35, 7, 18, 18, 18, 35, 19, 22, 12, 6,
    ]

def GOLDEN_UNIFORM_DEV_2() -> List[Int]:
    return [
        38, 8, 21, 12, 46, 43, 37, 24, 9, 13, 27, 32, 28, 20, 15, 33,
        22, 5, 30, 26, 11, 46, 6, 41, 21, 43, 13, 36, 4, 7, 43, 3,
    ]


def _check(ref got: List[Scalar[DType.int32]], ref want: List[Int], label: String) raises:
    assert_equal(len(got), len(want), label + ": length")
    for i in range(len(want)):
        assert_equal(
            Int(got[i]), want[i],
            label + " index " + String(i) + " moved (see the re-bless warning"
            " in this file's docstring)",
        )


def test_uniform_host_golden() raises:
    print("[1] uniform (host) vs golden ...")
    seed(SEED)
    var s = UniformSampler(N_FILL)
    var got = s.draw(BATCH)
    _check(got.host, GOLDEN_UNIFORM_HOST(), "uniform-host")
    print("      OK")


def test_per_host_golden() raises:
    print("[2] prioritized (host) vs golden ...")
    var p = PrioritizedSampler(
        CAP, alpha=Scalar[DT](0.6), beta=Scalar[DT](0.4),
        epsilon=Scalar[DT](1e-6),
    )
    for r in range(N_FILL):
        p.note_added(r)

    seed(SEED)
    var d1 = p.draw(BATCH)
    _check(d1.host, GOLDEN_PER_HOST_1(), "per-host draw 1")
    # Equal priorities on the first draw => uniform IS weights, all 1.0 after
    # normalisation. Asserted as a structural property rather than a literal.
    for i in range(BATCH):
        assert_almost_equal(
            Float64(p.last_weights[i]), 1.0, atol=1e-9,
            msg="per IS weight " + String(i) + " on the uniform-priority draw",
        )

    var td = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    for k in range(BATCH):
        td[k] = Scalar[DT](Float64(k + 1) * 0.37 - 5.0)
    p.update_priorities(td)

    seed(SEED + 1)
    var d2 = p.draw(BATCH)
    _check(d2.host, GOLDEN_PER_HOST_2(), "per-host draw 2 (after update)")
    # NOW the weights are non-trivial — this is where a broken tree shows up.
    for i in range(BATCH):
        assert_almost_equal(
            Float64(p.last_weights[i]), GOLDEN_PER_HOST_W2()[i], atol=1e-6,
            msg="per IS weight " + String(i) + " after the priority update",
        )
    print("      draws + post-update IS weights OK")


def test_sequence_golden() raises:
    print("[3] sequence-window starts vs golden ...")
    comptime T: Int = 6
    seed(SEED)
    var s = SequenceWindowSampler(N_FILL, T)
    var got = s.draw_starts(8)
    _check(got.host, GOLDEN_SEQ_STARTS(), "sequence starts")
    assert_equal(s.n_valid(), N_FILL - T, "n_valid must stay size - T")
    print("      OK")


def test_uniform_device_golden() raises:
    print("[4] uniform (device) vs golden, two batches ...")
    var ctx = DeviceContext()
    var s = UniformDeviceSampler(
        N_FILL, seed=UInt64(0xC0FFEE_DECADE_0042), offset=UInt64(0)
    )
    var d1 = s.draw(ctx, BATCH)
    _check(d1.host, GOLDEN_UNIFORM_DEV_1(), "uniform-device batch 1")
    var d2 = s.draw(ctx, BATCH)
    _check(d2.host, GOLDEN_UNIFORM_DEV_2(), "uniform-device batch 2")
    print("      OK (2nd batch pins the RNG offset advance)")


def main() raises:
    test_uniform_host_golden()
    test_per_host_golden()
    test_sequence_golden()
    test_uniform_device_golden()
    print("\n[PASS] sampler golden gate")
