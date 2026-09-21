# +--------------------------------------------------------------------------+ #
# | The vision cache: a prefix from a cached row is THE prefix, bit for bit
# +--------------------------------------------------------------------------+ #
"""`build_prefix_from_segment` against `build_prefix`, through a real file.

    pixi run -e apple mojo run -I . \\
        tests/deep_agents/smolvla/test_vision_cache.mojo

The fine-tune's vision cache (`vision_cache.mojo`) replaces the SigLIP tower,
the shuffle and the connector with a 491 KB read. That is only a saving if it
is not also a change: the claim is that the prefix buffer and the prefill
output are IDENTICAL whichever door they came through, and "identical" here
is bit-identical — the cached row is the same fp32 numbers the tower wrote,
and `run_tail` and the prefill are the same code either way.

  [1] the fresh door: `build_prefix` on a frame -> prefix buffer A, prefill
      output PA.
  [2] the segment: `embed_images` + `image_segment` on the same frame; its
      first IMG_SEG floats equal A's, bit for bit.
  [3] the file: the segment written with `VisionCache.create/write_row`,
      the file closed, reopened with `open`, read back with `read_row` —
      equal to what was written, every float.
  [4] the cached door: `build_prefix_from_segment` on the read-back row ->
      prefix buffer B == A and prefill PB == PA, bit for bit. This is the
      gate.
  [5] NOT VACUOUS: one float of the row perturbed, the cached door run again
      -> the prefix differs at exactly that slot AND the prefill output
      differs. Without this, [4] would also pass on a `build_prefix_from_
      segment` that ignored its argument and re-ran the tower.
  [6] the file's contract: a header with the wrong row count is REFUSED;
      `rows_done` is a resume point — 2 of 3 rows written, closed, reopened
      reports 2, and row 2 is unreadable rather than zeros.

⚠ A SHALLOW FIXTURE, like `test_policy.mojo`: 2 VLM / 2 expert / 2 vision
layers, deterministic weights. Bit-identity through the SAME kernels does not
get harder with depth; what depth costs is 3.2 GB of policy. ONE policy.
"""

from std.math import abs
from std.os import remove
from std.os.path import exists
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.deep_agents.smolvla.policy import SmolVLAPolicy
from noeira.deep_agents.smolvla.normalize import SmolVLAStats
from noeira.deep_agents.smolvla.tasks import TaskTokens
from noeira.deep_agents.smolvla.observation import fill_camera_images
from noeira.deep_agents.smolvla.vision_cache import VisionCache
from noeira.vision.resize_pad import SIGLIP_INPUT

comptime TABLE = "tools/vla/smolvla_tasks_record-test_20260828_092736.tsv"
comptime N_CAM = 2
comptime N_LANG = 6
comptime CHUNK = 50
comptime STEPS = 10
comptime VLM_LAYERS = 2
comptime VIS_LAYERS = 2
comptime Pol = SmolVLAPolicy[
    N_CAM, N_LANG, CHUNK, STEPS, 1, VLM_LAYERS, VIS_LAYERS
]
comptime CAM_W = 640
comptime CAM_H = 480
comptime RDIM = 6
comptime SEG = Pol.Prefix.IMG_SEG
comptime OUT_N = Pol.Prefix.OUT_N
comptime CACHE_PATH = "/tmp/smolvla_vision_cache_test.bin"


def robot_stats() raises -> SmolVLAStats:
    var s = SmolVLAStats()
    var m: List[Float32] = [16.64, -29.97, 31.07, 73.73, 41.12, 26.27]
    var d: List[Float32] = [21.00, 54.38, 51.43, 17.93, 18.72, 9.21]
    for i in range(RDIM):
        s.state_mean.append(m[i])
        s.state_std.append(d[i])
        s.action_mean.append(m[i])
        s.action_std.append(d[i])
    return s^


def make_frames(seed: Int) raises -> List[List[UInt8]]:
    var out = List[List[UInt8]]()
    for c in range(N_CAM):
        var f = List[UInt8](unsafe_uninit_length=CAM_W * CAM_H * 3)
        for i in range(len(f)):
            f[i] = UInt8((i * 7 + c * 53 + seed * 101) % 256)
        out.append(f^)
    return out^


def snapshot(mut t: Tensor, n: Int, d: DeviceContext) raises -> List[Float32]:
    t.download(d)
    var out = List[Float32](unsafe_uninit_length=n)
    for i in range(n):
        out[i] = Float32(t.data[i])
    return out^


def count_diff(ref a: List[Float32], ref b: List[Float32]) raises -> Int:
    assert_equal(len(a), len(b), "snapshots differ in length")
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def main() raises:
    print("=" * 70)
    print("SmolVLA vision cache — cached door vs fresh door, bit for bit")
    print("=" * 70)
    if exists(String(CACHE_PATH)):
        remove(String(CACHE_PATH))

    var tasks = TaskTokens(String(TABLE))
    var ids = tasks.for_index(0)
    assert_equal(len(ids), N_LANG, "instruction table vs N_LANG")

    var d = DeviceContext()
    var pol = Pol.make["gpu", Deterministic](Optional(d))
    pol.stats = robot_stats()
    var widths: List[Int] = [CAM_W, CAM_W]
    var heights: List[Int] = [CAM_H, CAM_H]
    var scratch = List[Float32]()
    var frames = make_frames(1)
    var images = Tensor()
    fill_camera_images["gpu", N_CAM, SIGLIP_INPUT](
        frames, widths, heights, False, images, scratch, Optional(d)
    )
    var pose: List[Float32] = [12.0, -40.0, 22.0, 70.0, 35.0, 20.0]
    comptime PN = Pol.P * Pol.W

    # [1] the fresh door
    pol.build_prefix["gpu"](images, ids, pose, Optional(d))
    var a = snapshot(pol.prefix_buf, OUT_N, d)
    var pa = snapshot(pol.prefill_out, PN, d)
    var nz = 0
    for i in range(SEG):
        if a[i] != 0.0:
            nz += 1
    assert_true(nz > SEG // 2, "the image segment is mostly zeros — no tower?")
    print("  [1] fresh prefix:", OUT_N, "floats, image segment", SEG,
          "of which nonzero", nz)

    # [2] the segment alone equals the prefix's head
    var seg = List[Float32]()
    pol.embed_images["gpu"](images, Optional(d))
    pol.image_segment["gpu"](seg, Optional(d))
    assert_equal(len(seg), SEG, "image_segment length")
    var d2 = 0
    for i in range(SEG):
        if seg[i] != a[i]:
            d2 += 1
    assert_equal(d2, 0, "embed_images + image_segment != build_prefix's head")
    print("  [2] image_segment == prefix[0:SEG]: 0 of", SEG, "differ")

    # [3] through the file
    var vc = VisionCache.create(String(CACHE_PATH), 3, SEG)
    vc.write_row(0, seg)
    vc.close()
    var vo = VisionCache.open(String(CACHE_PATH), 3, SEG)
    assert_equal(vo.rows_done, 1, "rows_done after one write + close")
    var back = List[Float32]()
    vo.read_row(0, back)
    assert_equal(count_diff(seg, back), 0, "file round trip changed floats")
    print("  [3] file round trip: 0 of", SEG, "differ, rows_done", vo.rows_done)

    # [4] the cached door — THE gate
    # Disturb the buffers first, so equality below is not the previous answer
    # still sitting there.
    var junk = List[Float32](length=SEG, fill=Float32(-7.5))
    pol.build_prefix_from_segment["gpu"](junk, ids, pose, Optional(d))
    pol.build_prefix_from_segment["gpu"](back, ids, pose, Optional(d))
    var b = snapshot(pol.prefix_buf, OUT_N, d)
    var pb = snapshot(pol.prefill_out, PN, d)
    var d_pre = count_diff(a, b)
    var d_pfl = count_diff(pa, pb)
    assert_equal(d_pre, 0, "cached-door prefix differs from the fresh door")
    assert_equal(d_pfl, 0, "cached-door prefill differs from the fresh door")
    print("  [4] cached door: prefix 0 of", OUT_N, "differ, prefill 0 of",
          PN, "differ — BIT-IDENTICAL")

    # [5] not vacuous
    var pert = back.copy()
    pert[SEG // 3] += 1.0
    pol.build_prefix_from_segment["gpu"](pert, ids, pose, Optional(d))
    var c = snapshot(pol.prefix_buf, OUT_N, d)
    var pc = snapshot(pol.prefill_out, PN, d)
    var d5 = count_diff(a, c)
    var d5p = count_diff(pa, pc)
    assert_equal(d5, 1, "one perturbed float should change exactly one slot")
    assert_true(c[SEG // 3] != a[SEG // 3], "the perturbed slot did not move")
    assert_true(d5p > 0, "a changed prefix left the prefill unchanged — the"
                " cached door is not feeding the tower")
    print("  [5] one float perturbed: prefix differs at 1 slot, prefill at",
          d5p, "of", PN)

    # [6] the file's contract
    var refused = False
    try:
        var bad = VisionCache.open(String(CACHE_PATH), 4, SEG)
        _ = bad.rows_done
    except:
        refused = True
    assert_true(refused, "a cache for 3 rows was accepted for 4")
    vo.write_row(1, seg)
    vo.close()
    var v3 = VisionCache.open(String(CACHE_PATH), 3, SEG)
    assert_equal(v3.rows_done, 2, "resume point after 2 of 3 rows")
    var past = False
    try:
        v3.read_row(2, back)
    except:
        past = True
    assert_true(past, "an unwritten row was readable")
    var wrong_order = False
    try:
        v3.write_row(0, seg)
    except:
        wrong_order = True
    assert_true(wrong_order, "an out-of-order write was accepted")
    v3.write_row(2, seg)
    assert_true(v3.complete(), "3 of 3 written but not complete")
    v3.close()
    remove(String(CACHE_PATH))
    print("  [6] wrong row count refused; resume point 2/3; unwritten row"
          " unreadable; out-of-order write refused; 3/3 complete")

    print("")
    print("PASSED — the cached door is the fresh door, bit for bit")
