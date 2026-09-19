# +--------------------------------------------------------------------------+ #
# | How fast the DEMONSTRATIONS move, in the units the deploy report prints
# +--------------------------------------------------------------------------+ #
"""The per-step action change in a recording, to compare against a running
policy's `step within chunk`.

    # on whichever box has a store of the recording — no import needed
    mojo run -I . tools/so101/demo_step_stats.mojo \\
        --store ~/.cache/mojo_rl/act_so101/so101-tower__cube-in-bowl_480x640.h5

    # the ACT-sized store of the SAME recording answers identically
    mojo run -I . tools/so101/demo_step_stats.mojo --store <...>_240x320.h5 \\
        --height 240 --width 320

⚠⚠ WHY THIS EXISTS. The armed SmolVLA run moved "faster than teleop", and the
deploy report alone cannot say whether that is a mis-scaled policy or a
correctly scaled one executing a discontinuous plan. Its two rows were:

    step within chunk = 4.99 deg mean, 17.9 worst
    step at handover  = 37.92 deg mean, 116.4 worst

The handover number is explained (we skip ~22 of 50 steps, so the arm is sent
to where a newer trajectory says it already is). The WITHIN-chunk number has
no reference — 4.99 deg per 1/30 s is only "fast" against something, and the
only honest something is the data the policy was fitted to.

So this prints the same statistic over the recording: the Euclidean norm of
`action[t+1] - action[t]` across the joints, in degrees, per 1/30 s step,
skipping episode boundaries (where the difference is a teleport between
takes, not motion).

If the demonstrations sit near 5 deg, the policy reproduces their speed and
the arm's haste is the handover alone. If they sit near 1 deg, the policy is
moving several times faster than anything it was shown, and the place to look
is the normalisation statistics — not the control loop.
"""

from std.math import sqrt
from std.sys import argv

from mojo_rl.deep_agents.act.config import SO101_FPS
from mojo_rl.deep_agents.act.data import ACTDataset
from mojo_rl.nn.constants import DT
from mojo_rl.utils.fmt import fixed, pad_left

comptime QPOS = 6
comptime ADIM = 6
comptime N_CAM = 2
comptime N_BUCKETS = 12

# ⚠ THE IMAGE SIZE IS A COMPTIME PARAMETER OF THE DATASET, and the two stores
# of one recording differ only in it: 240x320 for ACT, 480x640 for SmolVLA.
# The action column is identical in both — actions do not depend on how the
# frames were resized — so this measures the same thing either way and takes
# whichever store a machine happens to have, rather than asking for a 9 GB
# import of one it does not.
comptime ACT_H = 240
comptime ACT_W = 320
comptime VLA_H = 480
comptime VLA_W = 640


def _opt(ref args: List[String], flag: String, dflt: String) -> String:
    for i in range(len(args)):
        if String(args[i]) == flag and i + 1 < len(args):
            return String(args[i + 1])
    return dflt


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var path = _opt(args, String("--store"), String(""))
    if path == "":
        raise Error(
            "demo_step_stats: --store <the .h5> is required (the ACT-sized"
            " store is fine — the action column does not depend on the image"
            " size)"
        )

    print("=" * 70)
    print("Demonstration step size — " + path)
    print("=" * 70)

    # ⚠ THE IMAGES ARE NOT NEEDED and they are most of the file, so the
    # resident cap is set to zero: the column stays on disk and this runs in
    # seconds on a board with the store on a slow disk.
    var height = atol(_opt(args, String("--height"), String("480")))
    var width = atol(_opt(args, String("--width"), String("640")))
    if not (
        (height == ACT_H and width == ACT_W)
        or (height == VLA_H and width == VLA_W)
    ):
        raise Error(
            "demo_step_stats: --height/--width must be 240x320 or 480x640,"
            " got " + String(height) + "x" + String(width)
            + " — those are the two stores an SO-101 recording is imported to"
        )

    var action_raw = List[Scalar[DT]]()
    var starts = List[Int]()
    var lengths = List[Int]()
    var n = 0
    var n_ep = 0
    if height == ACT_H:
        var ds = ACTDataset[QPOS, ADIM, N_CAM, ACT_H, ACT_W](path^, 0, 0)
        n = ds.n_rows()
        n_ep = ds.n_episodes()
        action_raw = ds.action_raw.copy()
        for e in range(n_ep):
            starts.append(ds.store.episodes.start_of(e))
            lengths.append(ds.store.episodes.length_of(e))
    else:
        var ds = ACTDataset[QPOS, ADIM, N_CAM, VLA_H, VLA_W](path^, 0, 0)
        n = ds.n_rows()
        n_ep = ds.n_episodes()
        action_raw = ds.action_raw.copy()
        for e in range(n_ep):
            starts.append(ds.store.episodes.start_of(e))
            lengths.append(ds.store.episodes.length_of(e))
    print("  " + String(n) + " rows, " + String(n_ep) + " episodes, "
          + String(SO101_FPS) + " Hz, images " + String(height) + "x"
          + String(width) + " (not read)")

    var sum_step = 0.0
    var worst = 0.0
    var counted = 0
    var per_joint = List[Float64](length=ADIM, fill=0.0)
    var hist = List[Int](length=N_BUCKETS, fill=0)

    for e in range(n_ep):
        var start = starts[e]
        var length = lengths[e]
        # ⚠ WITHIN an episode only. Across a boundary the difference is the
        # arm being reset between takes, which is not motion the policy is
        # asked to reproduce.
        for t in range(length - 1):
            var r0 = (start + t) * ADIM
            var r1 = (start + t + 1) * ADIM
            var acc = 0.0
            for j in range(ADIM):
                var d = Float64(action_raw[r1 + j]) - Float64(
                    action_raw[r0 + j]
                )
                acc += d * d
                per_joint[j] += abs(d)
            var st = sqrt(acc)
            sum_step += st
            counted += 1
            if st > worst:
                worst = st
            var b = Int(st)
            if b >= N_BUCKETS:
                b = N_BUCKETS - 1
            hist[b] += 1

    if counted == 0:
        raise Error("demo_step_stats: no consecutive pairs — empty store?")

    # ⚠⚠ THE SECOND STATISTIC, AND IT SEPARATES TWO FAILURES THAT LOOK ALIKE.
    # A trajectory that JITTERS has a large per-step delta and goes nowhere; a
    # trajectory SCALED UP has a large per-step delta and arrives somewhere
    # three times too far away. Path length over one chunk's worth of steps,
    # against the straight line between its ends, tells them apart.
    comptime WIN = 50
    var sum_path = 0.0
    var sum_net = 0.0
    var windows = 0
    for e in range(n_ep):
        var start = starts[e]
        var length = lengths[e]
        var w = 0
        while w + WIN < length:
            var path = 0.0
            for t in range(w, w + WIN):
                var acc = 0.0
                for j in range(ADIM):
                    var d = Float64(
                        action_raw[(start + t + 1) * ADIM + j]
                    ) - Float64(action_raw[(start + t) * ADIM + j])
                    acc += d * d
                path += sqrt(acc)
            var net = 0.0
            for j in range(ADIM):
                var d = Float64(
                    action_raw[(start + w + WIN) * ADIM + j]
                ) - Float64(action_raw[(start + w) * ADIM + j])
                net += d * d
            sum_path += path
            sum_net += sqrt(net)
            windows += 1
            w += WIN

    var mean = sum_step / Float64(counted)
    print("")
    print("  step (all joints, euclidean) = " + fixed(mean, 2)
          + " deg mean, " + fixed(worst, 1) + " worst   ("
          + String(counted) + " steps)")
    print("  that is " + fixed(mean * Float64(SO101_FPS), 1)
          + " deg/s at " + String(SO101_FPS) + " Hz")
    print("")
    print("  per joint, mean |delta| per step:")
    for j in range(ADIM):
        print("     joint " + String(j) + "  "
              + fixed(per_joint[j] / Float64(counted), 3) + " deg")
    print("")
    print("  distribution (deg per step):")
    for b in range(N_BUCKETS):
        var share = 100.0 * Float64(hist[b]) / Float64(counted)
        var bar = String("")
        for _ in range(Int(share / 2.0)):
            bar += "#"
        var label = (
            String(b) + "-" + String(b + 1) if b < N_BUCKETS - 1
            else String(">=") + String(N_BUCKETS - 1)
        )
        print("     " + pad_left(label, 5) + "  " + pad_left(
            fixed(share, 1), 5
        ) + "%  " + bar)
    # ⚠ HOW OFTEN A JOINT REVERSES, which separates a high-frequency tremor
    # from a slow wobble. Demonstrated motion changes direction when the task
    # does; noise changes direction about half the time, and only the first
    # kind survives a low-pass filter unharmed.
    var flips = 0
    var flip_of = 0
    for e in range(n_ep):
        var start = starts[e]
        var length = lengths[e]
        for j in range(ADIM):
            for t in range(1, length - 1):
                var d0 = Float64(
                    action_raw[(start + t) * ADIM + j]
                ) - Float64(action_raw[(start + t - 1) * ADIM + j])
                var d1 = Float64(
                    action_raw[(start + t + 1) * ADIM + j]
                ) - Float64(action_raw[(start + t) * ADIM + j])
                if d0 * d1 < 0.0:
                    flips += 1
                flip_of += 1
    if flip_of > 0:
        print("  direction reversals = "
              + fixed(100.0 * Float64(flips) / Float64(flip_of), 1)
              + "% of steps   (a tremor is ~50%)")

    if windows > 0:
        var mpath = sum_path / Float64(windows)
        var mnet = sum_net / Float64(windows)
        print("")
        print("  over " + String(WIN) + "-step windows (one chunk): path "
              + fixed(mpath, 1) + " deg, net " + fixed(mnet, 1)
              + " deg, wiggle " + fixed(mpath / mnet if mnet > 0.0 else 0.0, 2)
              + "x   (" + String(windows) + " windows)")
    print("")
    print("  ⚠ compare with the deploy report's `step within chunk`. A policy"
          " that matches this")
    print("    is moving at demonstration speed, whatever it looks like from"
          " across the desk.")
