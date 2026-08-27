"""Frame↔patch (temporal_patchify / temporal_unpatchify) CPU roundtrip (Phase 1).

Patchify then unpatchify must recover the original frames exactly, and a
single known patch must map to the right pixels (F.unfold ordering).
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamer4.patchify import (
    temporal_patchify, temporal_unpatchify,
)


def _alloc(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    print("=" * 70)
    print("temporal_patchify roundtrip (Phase 1)")
    print("=" * 70)
    comptime BT = 2          # B*T frames
    comptime C = 3
    comptime H = 8
    comptime W = 6
    comptime PATCH = 2
    comptime PH = H // PATCH
    comptime PW = W // PATCH
    comptime NP = PH * PW
    comptime DP = C * PATCH * PATCH
    comptime VN = BT * C * H * W
    comptime PN = BT * NP * DP

    var video = _alloc(VN)
    var patches = _alloc(PN)
    var recon = _alloc(VN)
    for i in range(VN):
        video[i] = Scalar[DT](Float64(i) * 0.01 - 1.0)

    temporal_patchify[BT, C, H, W, PATCH](video, patches)

    # spot check: patch (pr=0,pc=0), c=1, ky=1, kx=0 → pixel (h=1,w=0)
    var np_idx = 0  # pr*PW+pc = 0
    var dp = 1 * PATCH * PATCH + 1 * PATCH + 0
    var got = Float64(patches[0 * NP * DP + np_idx * DP + dp])
    var want = Float64(video[0 * C * H * W + 1 * H * W + 1 * W + 0])
    assert_true(abs(got - want) < 1e-7, "patchify ordering spot check")

    temporal_unpatchify[BT, C, H, W, PATCH](patches, recon)
    var max_err: Float64 = 0.0
    for i in range(VN):
        var e = abs(Float64(recon[i]) - Float64(video[i]))
        if e > max_err:
            max_err = e
    print("   roundtrip max err =", max_err)
    assert_true(max_err == 0.0, "patchify∘unpatchify must be identity")

    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
