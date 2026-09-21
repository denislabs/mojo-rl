#!/usr/bin/env python
"""Compare leader and follower calibrations and report the teleop mapping.

Both arms run MotorNormMode.DEGREES, where
    deg      = (raw - mid) * 360 / 4095          with mid = (range_min + range_max) / 2
    raw_goal = deg * 4095 / 360 + mid
so the follower goal is  raw_L + (mid_F - mid_L)  -- a pure tick offset. Any
difference between the two recorded mid-points is a permanent joint desync, and
it also eats into what the follower can still reach.
"""

import json
from pathlib import Path

CAL = Path.home() / ".cache/huggingface/lerobot/calibration"
RES = 4096
DEG = 360 / (RES - 1)


def load(p):
    return json.loads(p.read_text())


def main(follower_id="noeira_follower_arm", leader_id="noeira_leader_arm"):
    f = load(CAL / "robots/so_follower" / f"{follower_id}.json")
    ld = load(CAL / "teleoperators/so_leader" / f"{leader_id}.json")

    hdr = (f"{'MOTOR':<14}{'leader span':>13}{'follower span':>15}{'desync':>10}"
           f"{'lost @min':>11}{'lost @max':>11}")
    print(hdr)
    print("-" * len(hdr))
    for m in f:
        if m == "wrist_roll":
            print(f"{m:<14}{'full turn':>13}{'full turn':>15}{'-':>10}{'-':>11}{'-':>11}")
            continue
        if m == "gripper":
            # RANGE_0_100, not DEGREES: the mapping rescales min..max onto min..max,
            # so the two spans are matched end-to-end by construction -- no desync.
            lmin, lmax = ld[m]["range_min"], ld[m]["range_max"]
            fmin, fmax = f[m]["range_min"], f[m]["range_max"]
            print(f"{m:<14}{(lmax - lmin) * DEG:>10.0f} deg{(fmax - fmin) * DEG:>12.0f} deg"
                  f"{'n/a':>10}{'n/a':>11}{'n/a':>11}  (0-100%% rescale, self-matching)")
            continue
        lmin, lmax = ld[m]["range_min"], ld[m]["range_max"]
        fmin, fmax = f[m]["range_min"], f[m]["range_max"]
        mid_l, mid_f = (lmin + lmax) / 2, (fmin + fmax) / 2
        shift = mid_f - mid_l                      # follower_goal = leader_raw + shift

        # where the leader's extremes land on the follower, and what the follower clips
        lost_min = max(0, fmin - (lmin + shift))   # follower bottoms out early
        lost_max = max(0, (lmax + shift) - fmax)   # follower tops out early
        flag = "  <-- " + ("desync" if abs(shift) > 60 else "") if abs(shift) > 60 else ""
        print(f"{m:<14}{(lmax - lmin) * DEG:>10.0f} deg{(fmax - fmin) * DEG:>12.0f} deg"
              f"{shift * DEG:>+9.1f} deg{lost_min * DEG:>10.0f} deg{lost_max * DEG:>10.0f} deg{flag}")

    print("\nspan mismatch = one arm was not swept to its real end stops during calibration.")
    print("desync        = constant angular offset between leader and follower.")
    print("lost @min/max = travel the follower can no longer reach, whatever the leader does.")


if __name__ == "__main__":
    main()
