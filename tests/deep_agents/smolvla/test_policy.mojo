"""The assembled policy: two camera frames and a pose to a chunk of angles.

Every component here has its own gate. What none of them can see is the
wiring — and the wiring failures all produce a correctly shaped chunk of
plausible joint angles:

  1. **`P` derived once.** `SmolVLAPolicy` computes `N_CAM*64 + N_LANG + 1` and
     hands the same integer to the cache, the prefill mask and both denoise
     masks. Checked against the instruction table's own token count.
  2. **The output is in ROBOT units.** A missing unnormalisation returns values
     near +-3 that look exactly like a normalised chunk; the dataset's stds are
     ~10..54, so real angles are an order of magnitude larger.
  3. **The cache belongs to ONE observation.** A second, DIFFERENT observation
     must give a different chunk. Prefilling once and sampling forever is the
     failure this whole object exists to prevent, and it is invisible: the
     policy keeps emitting smooth, believable motion for the scene it saw
     first.
  4. **The same observation reproduces exactly** — the complement of 3. If 3
     passes because something is drifting rather than because the cache was
     refilled, this fails.
  5. **No stats means a raised error, not an unnormalised state.** Feeding raw
     joint angles (tens) where the tower expects ~+-2 is finite and wrong.

Deterministic weights, no download: this gates the assembly, not the numbers.
Parity against the real checkpoint is `test_parity_vs_hf`.

⚠⚠ **ONE POLICY. NEVER TWO.** `Tensor` holds host AND device copies, so a
single `SmolVLAPolicy` at real shapes is 402,737,376 params x 4 bytes x 2 =
**3.2 GB**. An earlier draft of this file built three — one to vary the action
stats, one to test the missing-stats error — for **9.7 GB**, which on a 16 GiB
machine took the whole laptop down with it, mid-compile. Both variations need
nothing but a different 6-element stats list, so both mutate the ONE policy and
restore it. If you need another configuration, change `pol.stats`, not `Pol`.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_policy.mojo
"""

from std.math import abs
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.policy import SmolVLAPolicy
from mojo_rl.deep_agents.smolvla.normalize import SmolVLAStats
from mojo_rl.deep_agents.smolvla.observation import fill_camera_images
from mojo_rl.deep_agents.smolvla.tasks import TaskTokens
from mojo_rl.vision.resize_pad import SIGLIP_INPUT

comptime TABLE = "tools/vla/smolvla_tasks_record-test_20260828_092736.tsv"
comptime N_CAM = 2
comptime N_LANG = 6
comptime CHUNK = 50
comptime STEPS = 10
comptime Pol = SmolVLAPolicy[N_CAM, N_LANG, CHUNK, STEPS, 1]
comptime CAM_W = 640
comptime CAM_H = 480
comptime RDIM = 6


def robot_stats() raises -> SmolVLAStats:
    """The recording's own scale, hardcoded so this gate needs no dataset."""
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


def main() raises:
    print("=" * 70)
    print("SmolVLA assembled policy")
    print("=" * 70)

    var tasks = TaskTokens(String(TABLE))
    var ids = tasks.for_index(0)
    assert_equal(
        len(ids), N_LANG,
        "the instruction table and the policy's N_LANG disagree — every mask"
        " in the prefill is built from that number",
    )
    print("  [1] P =", N_CAM, "x 64 +", len(ids), "+ 1 =", Pol.P)
    assert_equal(Pol.P, N_CAM * 64 + N_LANG + 1, "P was not derived")

    var d = DeviceContext()
    print("      building the policy (16 VLM + 16 expert + 12 vision layers)…")
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
    print("      images:", N_CAM, "x 3 x", SIGLIP_INPUT, "x", SIGLIP_INPUT)

    var pose: List[Float32] = [12.0, -40.0, 22.0, 70.0, 35.0, 20.0]

    comptime XN = CHUNK * 32
    var noise = Tensor.alloc(XN)
    for i in range(XN):
        noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
    noise.upload(d)

    var act = List[Float32]()
    pol.select_action["gpu"](images, ids, pose, noise, act, Optional(d))

    assert_equal(
        len(act), CHUNK * RDIM,
        "the chunk kept the padded dims — 32 wide instead of 6",
    )
    var nan = 0
    var lo = act[0]
    var hi = act[0]
    var big = 0
    for i in range(len(act)):
        if act[i] != act[i]:
            nan += 1
        if act[i] < lo:
            lo = act[i]
        if act[i] > hi:
            hi = act[i]
        if abs(act[i]) > 5.0:
            big += 1
    print("  [2] chunk", CHUNK, "x", RDIM, ": nan", nan, " min", lo,
          " max", hi)
    assert_equal(nan, 0, "the policy produced NaN")
    assert_true(hi - lo > 1e-4, "the chunk is constant")

    # ⚠ "the values are big, so they must be unnormalised" DOES NOT DECIDE.
    # With seeded weights the chunk is already far outside +-5 before any
    # scaling, so that test passes whether or not the stats were applied. It
    # was in this file and it was vacuous.
    #
    # What decides: run again with the ACTION stds DOUBLED and nothing else
    # touched. The network computes exactly the same chunk, so every output
    # must satisfy  out2 - mean == 2 * (out1 - mean),  per dimension. That
    # catches the stats not being applied, being applied once instead of
    # per-element, and being indexed by the wrong dimension.
    # ⚠ The SAME policy with its action_std doubled — NOT a second policy.
    # See the header: one instantiation is 3.2 GB.
    var saved_std = List[Float32]()
    for i in range(RDIM):
        saved_std.append(pol.stats.action_std[i])
        pol.stats.action_std[i] = pol.stats.action_std[i] * 2.0
    var noise_b = Tensor.alloc(XN)
    for i in range(XN):
        noise_b.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
    noise_b.upload(d)
    var act_b = List[Float32]()
    pol.select_action["gpu"](images, ids, pose, noise_b, act_b, Optional(d))
    for i in range(RDIM):
        pol.stats.action_std[i] = saved_std[i]  # restore BEFORE checks [3]/[4]

    var worst: Float32 = 0.0
    var checked = 0
    for t in range(CHUNK):
        for i in range(RDIM):
            var m = pol.stats.action_mean[i]
            var want = 2.0 * (act[t * RDIM + i] - m) + m
            var e = abs(act_b[t * RDIM + i] - want)
            var scale = abs(want - m) + 1.0
            if e / scale > worst:
                worst = e / scale
            checked += 1
    print("      doubling only action_std: compared", checked,
          " worst relative error", worst)
    assert_true(
        checked == len(act),
        "the scale check covered nothing — it must see every value",
    )
    assert_true(
        worst < 1e-4,
        "doubling action_std did not double the output's deviation from the"
        " mean — the dataset scale is not being applied per dimension",
    )
    print("      ", big, "of", len(act), "outside +-5 (informational only)")

    # [3] a DIFFERENT observation must give a DIFFERENT chunk
    var frames2 = make_frames(9)
    var images2 = Tensor()
    var scratch2 = List[Float32]()
    fill_camera_images["gpu", N_CAM, SIGLIP_INPUT](
        frames2, widths, heights, False, images2, scratch2, Optional(d)
    )
    var pose2: List[Float32] = [30.0, -10.0, 40.0, 80.0, 50.0, 30.0]
    var noise2 = Tensor.alloc(XN)
    for i in range(XN):
        noise2.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
    noise2.upload(d)
    var act2 = List[Float32]()
    pol.select_action["gpu"](images2, ids, pose2, noise2, act2, Optional(d))
    var differ = 0
    for i in range(len(act)):
        if abs(act[i] - act2[i]) > 1e-4:
            differ += 1
    print("  [3] a new observation: differing", differ, "/", len(act))
    assert_true(
        differ > len(act) // 4,
        "a different scene gave the same chunk — the KV cache was not refilled,"
        " so the policy is acting on the observation before this one",
    )

    # [4] the SAME observation reproduces exactly
    var noise3 = Tensor.alloc(XN)
    for i in range(XN):
        noise3.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
    noise3.upload(d)
    var act3 = List[Float32]()
    pol.select_action["gpu"](images, ids, pose, noise3, act3, Optional(d))
    var drift = 0
    for i in range(len(act)):
        if act[i] != act3[i]:
            drift += 1
    print("  [4] rerun of observation 1: differing", drift, "/", len(act))
    assert_equal(
        drift, 0,
        "the same observation gave a different chunk — state is leaking"
        " between calls",
    )

    # [5] no stats must raise — again by emptying THIS policy's stats, not by
    # building another one.
    # Emptying `state_mean` is what `select_action`'s guard reads
    # (`stats.state_dim() == 0`). Done by clearing the list rather than moving
    # the field out, so there is no partially-initialised struct to reason
    # about.
    var keep_mean = List[Float32]()
    var keep_std = List[Float32]()
    for i in range(RDIM):
        keep_mean.append(pol.stats.state_mean[i])
        keep_std.append(pol.stats.state_std[i])
    pol.stats.state_mean.clear()
    pol.stats.state_std.clear()
    var raised = False
    var act4 = List[Float32]()
    try:
        pol.select_action["gpu"](images, ids, pose, noise, act4, Optional(d))
    except:
        raised = True
    for i in range(RDIM):
        pol.stats.state_mean.append(keep_mean[i])
        pol.stats.state_std.append(keep_std[i])
    assert_true(
        raised,
        "select_action without stats must raise — raw joint angles are tens"
        " where the tower expects ~+-2",
    )
    print("  [5] select_action without stats raises")

    print()
    print("PASSED — raw frames + pose ->", len(act), "joint angles")
