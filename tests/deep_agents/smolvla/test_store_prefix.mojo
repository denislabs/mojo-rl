# +--------------------------------------------------------------------------+ #
# | A store row and a live observation must fill the SAME KV cache
# +--------------------------------------------------------------------------+ #
"""The last join between training and deployment, end to end.

    pixi run -e apple mojo run -I . \\
        tests/deep_agents/smolvla/test_store_prefix.mojo

Two gates already say the pixels agree: `test_store_vs_camera_frame.mojo` for
one frame through the two SigLIP doors, `test_store_observation.mojo` for the
N_CAM assembly around them. Neither runs a single weight.

This runs the whole prefix — SigLIP tower, PixelShuffle, connector, language
embedding, `state_proj`, then sixteen VLM layers into the KV cache — from a
store row and from the equivalent camera capture, and demands the two caches
be **bit-identical**. That is the artefact the denoising step actually reads,
so it is the thing that has to match; agreement earlier in the chain is
necessary and does not establish it, because the prefix also folds in the
state and the language tokens, and either of those could enter differently on
the training side.

⚠ `build_prefix` is called by BOTH paths here, which is the point rather than
a weakness: it was factored out of `select_action` precisely so training and
inference cannot describe a prefix differently. What this leg checks is that
the INPUTS a store row produces are the inputs a camera produces — and legs
[2] and [3] make sure that claim has content.

⚠ **A 2/2/2 fixture and seeded weights.** 2 VLM layers, 2 expert, 2 vision,
the checkpoint's real WIDTHS. Numerical parity against `lerobot` is
`test_parity_vs_hf.mojo`'s job and needs the download; this is about wiring,
and wiring is depth-independent.
"""

from std.math import abs
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.policy import SmolVLAPolicy
from mojo_rl.deep_agents.smolvla.normalize import SmolVLAStats
from mojo_rl.deep_agents.smolvla.observation import (
    fill_camera_images, fill_store_images,
)

comptime N_CAM = 2
comptime N_LANG = 6
comptime CHUNK = 4
comptime STEPS = 2
comptime B = 1
comptime LAYERS = 2
comptime VIS_LAYERS = 2
comptime SRC_W = 64
comptime SRC_H = 48
comptime PX = SRC_W * SRC_H
comptime PER_CAM = 3 * PX

comptime Pol = SmolVLAPolicy[
    N_CAM, N_LANG, CHUNK, STEPS, B, LAYERS, VIS_LAYERS
]


def _stats() raises -> SmolVLAStats:
    var s = SmolVLAStats()
    for j in range(6):
        s.state_mean.append(Float32(j) * 0.5)
        s.state_std.append(Float32(1.0) + Float32(j) * 0.1)
        s.action_mean.append(Float32(-j) * 0.25)
        s.action_std.append(Float32(2.0) + Float32(j) * 0.05)
    return s^


def main() raises:
    print("=" * 70)
    print("store row vs live observation -> the KV cache")
    print("=" * 70)
    print("  P =", N_CAM, "x 64 +", N_LANG, "+ 1 =", Pol.P,
          " layers", LAYERS, "/", VIS_LAYERS)

    var d = DeviceContext()
    var pol = Pol.make["gpu", Deterministic](Optional(d))
    pol.stats = _stats()

    # ── the same physical scene, in both representations ─────────────────
    var row = List[Scalar[DType.uint8]](unsafe_uninit_length=N_CAM * PER_CAM)
    for cam in range(N_CAM):
        for y in range(SRC_H):
            for x in range(SRC_W):
                var i = cam * PER_CAM + y * SRC_W + x
                row[i] = Scalar[DType.uint8](
                    ((x * 3 + y * 5 + cam * 71) % 251)
                )
                row[PX + i] = Scalar[DType.uint8](
                    ((x * 7 + y * 2 + cam * 31) % 251)
                )
                row[2 * PX + i] = Scalar[DType.uint8](
                    ((x + y * 11 + cam * 13) % 251)
                )
    var frames = List[List[UInt8]]()
    var widths = List[Int]()
    var heights = List[Int]()
    for cam in range(N_CAM):
        var f = List[UInt8](unsafe_uninit_length=PER_CAM)
        for i in range(PX):
            f[i * 3 + 0] = UInt8(Int(row[cam * PER_CAM + 2 * PX + i]))
            f[i * 3 + 1] = UInt8(Int(row[cam * PER_CAM + PX + i]))
            f[i * 3 + 2] = UInt8(Int(row[cam * PER_CAM + i]))
        frames.append(f^)
        widths.append(SRC_W)
        heights.append(SRC_H)

    var lang = List[Int]()
    for t in range(N_LANG):
        lang.append(55 + t * 37)
    var state = List[Float32]()
    for j in range(6):
        state.append(Float32(j) * 1.7 - 2.0)

    # ── [1] the two caches are bit-identical ─────────────────────────────
    var img_c = Tensor.alloc(N_CAM * 3 * 512 * 512)
    var sc1 = List[Float32]()
    fill_camera_images["gpu", N_CAM](
        frames, widths, heights, True, img_c, sc1, Optional(d)
    )
    pol.build_prefix["gpu"](img_c, lang, state, Optional(d))
    d.synchronize()
    pol.cache.k.download(d)
    pol.cache.v.download(d)
    var k_cam = List[Scalar[DT]](unsafe_uninit_length=Pol.Cache.TOTAL)
    var v_cam = List[Scalar[DT]](unsafe_uninit_length=Pol.Cache.TOTAL)
    for i in range(Pol.Cache.TOTAL):
        k_cam[i] = pol.cache.k.data[i]
        v_cam[i] = pol.cache.v.data[i]

    var img_s = Tensor.alloc(N_CAM * 3 * 512 * 512)
    var sc2 = List[Float32]()
    fill_store_images["gpu", N_CAM](
        row, SRC_W, SRC_H, img_s, sc2, Optional(d)
    )
    pol.build_prefix["gpu"](img_s, lang, state, Optional(d))
    d.synchronize()
    pol.cache.k.download(d)
    pol.cache.v.download(d)

    var diff = 0
    var worst = Scalar[DT](0)
    for i in range(Pol.Cache.TOTAL):
        if pol.cache.k.data[i] != k_cam[i]:
            diff += 1
        if pol.cache.v.data[i] != v_cam[i]:
            diff += 1
        var dk = abs(pol.cache.k.data[i] - k_cam[i])
        var dv = abs(pol.cache.v.data[i] - v_cam[i])
        if dk > worst:
            worst = dk
        if dv > worst:
            worst = dv
    print("  [1] store cache vs camera cache: compared",
          2 * Pol.Cache.TOTAL, " differing", diff, " worst", worst)
    assert_true(
        diff == 0,
        "the cache a store row produces differs from the one the same scene"
        " produces live — training and deployment condition on different"
        " prefixes",
    )

    # ── [2] the cache is not trivially constant ──────────────────────────
    # ⚠ Leg [1] would pass on two all-zero caches, and a prefix that never ran
    # is exactly that.
    var nz = 0
    var nan = 0
    for i in range(Pol.Cache.TOTAL):
        if pol.cache.k.data[i] != Scalar[DT](0):
            nz += 1
        var y = pol.cache.k.data[i]
        if y != y:
            nan += 1
    print("  [2] cache K: nonzero", nz, "of", Pol.Cache.TOTAL, " nan", nan)
    assert_true(nan == 0, "the prefix produced NaN")
    assert_true(
        nz * 10 > Pol.Cache.TOTAL * 9,
        "most of the cache is zero — the prefill did not run",
    )

    # ── [3] a DIFFERENT scene gives a different cache ────────────────────
    # ⚠ What says leg [1] is comparing two runs of a live computation rather
    # than two reads of a stale buffer.
    var row2 = List[Scalar[DType.uint8]](unsafe_uninit_length=N_CAM * PER_CAM)
    for i in range(N_CAM * PER_CAM):
        row2[i] = Scalar[DType.uint8]((Int(row[i]) + 97) % 251)
    var img_2 = Tensor.alloc(N_CAM * 3 * 512 * 512)
    var sc3 = List[Float32]()
    fill_store_images["gpu", N_CAM](
        row2, SRC_W, SRC_H, img_2, sc3, Optional(d)
    )
    pol.build_prefix["gpu"](img_2, lang, state, Optional(d))
    d.synchronize()
    pol.cache.k.download(d)
    var moved = 0
    for i in range(Pol.Cache.TOTAL):
        if pol.cache.k.data[i] != k_cam[i]:
            moved += 1
    print("  [3] a different scene moves", moved, "of", Pol.Cache.TOTAL,
          "cache entries")
    assert_true(
        moved * 2 > Pol.Cache.TOTAL,
        "changing every pixel barely changed the cache — leg [1] is"
        " comparing a buffer nobody wrote",
    )

    print()
    print("PASSED — one prefix, whichever side the pixels came from")
