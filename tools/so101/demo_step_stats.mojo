# +--------------------------------------------------------------------------+ #
# | How fast the DEMONSTRATIONS move, in the units the deploy report prints
# +--------------------------------------------------------------------------+ #
"""The per-step action change in a recording, to compare against a running
policy's `step within chunk`.

    pixi run -e jetson mojo run -I . tools/so101/demo_step_stats.mojo \\
        --store ~/.cache/mojo_rl/act_so101/so101-tower__cube-in-bowl_240x320.h5

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
comptime IMG_H = 240
comptime IMG_W = 320
comptime N_BUCKETS = 12


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
    var ds = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](path^, 0, 0)
    var n = ds.n_rows()
    var n_ep = ds.n_episodes()
    print("  " + String(n) + " rows, " + String(n_ep) + " episodes, "
          + String(SO101_FPS) + " Hz")

    var sum_step = 0.0
    var worst = 0.0
    var counted = 0
    var per_joint = List[Float64](length=ADIM, fill=0.0)
    var hist = List[Int](length=N_BUCKETS, fill=0)

    for e in range(n_ep):
        var start = ds.store.episodes.start_of(e)
        var length = ds.store.episodes.length_of(e)
        # ⚠ WITHIN an episode only. Across a boundary the difference is the
        # arm being reset between takes, which is not motion the policy is
        # asked to reproduce.
        for t in range(length - 1):
            var r0 = (start + t) * ADIM
            var r1 = (start + t + 1) * ADIM
            var acc = 0.0
            for j in range(ADIM):
                var d = Float64(ds.action_raw[r1 + j]) - Float64(
                    ds.action_raw[r0 + j]
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
    print("")
    print("  ⚠ compare with the deploy report's `step within chunk`. A policy"
          " that matches this")
    print("    is moving at demonstration speed, whatever it looks like from"
          " across the desk.")
