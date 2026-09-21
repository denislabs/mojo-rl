#!/usr/bin/env python
"""Measure one SO-101 joint's travel in RAW encoder space, and/or park it at 2047.

  sweep  : torque OFF, homing offset temporarily 0, you move the joint by hand
           end-to-end. Reports the true reachable interval of the 12-bit absolute
           encoder and whether it crosses the 4095 -> 0 seam.
  center : torque ON, drives the servo to raw 2047 so you can re-index the horn.

Both modes restore the saved Homing_Offset / position limits on exit.
"""

import argparse
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.utils import enter_pressed, move_cursor_up

RES = 4096
IDS = {
    "shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
    "wrist_flex": 4, "wrist_roll": 5, "gripper": 6,
}


def open_bus(port, motor):
    bus = FeetechMotorsBus(port=port, motors={motor: Motor(IDS[motor], "sts3215", MotorNormMode.DEGREES)})
    bus.connect()
    return bus


def save(bus, m):
    return {k: bus.read(k, m, normalize=False)
            for k in ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit", "Torque_Enable")}


def restore(bus, m, s):
    bus.write("Torque_Enable", m, 0, normalize=False)
    for k in ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit"):
        bus.write(k, m, s[k], normalize=False)
    print(f"\nrestored {m}: offset={s['Homing_Offset']} "
          f"limits=[{s['Min_Position_Limit']}, {s['Max_Position_Limit']}], torque off")


def unclamp(bus, m):
    """Expose the bare encoder: no offset, no position limits."""
    bus.write("Torque_Enable", m, 0, normalize=False)
    bus.write("Homing_Offset", m, 0, normalize=False)
    bus.write("Min_Position_Limit", m, 0, normalize=False)
    bus.write("Max_Position_Limit", m, RES - 1, normalize=False)


def sweep(bus, m):
    print(f"\nHOLD THE ARM so it cannot fall -- torque on '{m}' is about to be released.")
    input("Press ENTER when you are holding it....")
    unclamp(bus, m)

    prev = bus.read("Present_Position", m, normalize=False)
    lo = hi = acc = 0          # unwrapped coordinate, relative to `prev`
    origin, wraps = prev, 0

    print(f"\nMove '{m}' slowly through its ENTIRE range, both directions. Press ENTER to stop...\n")
    while True:
        cur = bus.read("Present_Position", m, normalize=False)
        d = cur - prev
        if d > RES // 2:            # 0 -> 4095
            d -= RES
            wraps += 1
        elif d < -RES // 2:         # 4095 -> 0
            d += RES
            wraps += 1
        acc += d
        prev = cur
        lo, hi = min(lo, acc), max(hi, acc)

        print(f"  raw encoder : {cur:>5}   (0..4095, seam at 4095/0)")
        print(f"  unwrapped   : {origin + acc:>5}   span so far {hi - lo:>5} ticks = {(hi - lo) * 360 / (RES - 1):5.1f} deg")
        print(f"  seam crossed: {wraps} time(s){'   <-- travel spans the seam' if wraps else ''}      ")
        if enter_pressed():
            break
        move_cursor_up(3)
        time.sleep(0.02)

    raw_lo, raw_hi = origin + lo, origin + hi
    mid = (raw_lo + raw_hi) / 2
    print("\n" + "=" * 72)
    print(f"true travel (unwrapped) : {raw_lo} .. {raw_hi}   "
          f"({(hi - lo) * 360 / (RES - 1):.1f} deg, {hi - lo} ticks)")
    print(f"mid-travel sits at raw  : {mid:.0f}      (it should sit at 2047)")
    if raw_lo < 0 or raw_hi > RES - 1:
        err = mid - 2047
        print(f"\n  ** the joint's travel crosses the 4095/0 encoder seam **")
        print(f"  re-index the output horn by {abs(err) * 360 / (RES - 1):.0f} deg "
              f"({abs(err):.0f} ticks) {'CW' if err > 0 else 'CCW'} in encoder terms")
        print(f"  STS3215 spline = 25 teeth = 14.4 deg/tooth -> "
              f"{abs(err) * 360 / (RES - 1) / 14.4:.1f} teeth")
    else:
        margin = min(raw_lo, RES - 1 - raw_hi)
        print(f"\n  travel is clear of the seam (margin {margin} ticks = {margin * 360 / (RES - 1):.0f} deg)")
    print("=" * 72)


def center(bus, m):
    print(f"\nDETACH '{m}' from its bracket first -- it is about to be driven to raw 2047.")
    if input("Type 'yes' to energise the servo: ").strip().lower() != "yes":
        return
    unclamp(bus, m)
    bus.write("Goal_Position", m, 2047, normalize=False)
    bus.write("Torque_Enable", m, 1, normalize=False)
    print("\nHolding raw 2047. Re-index the horn so the joint sits at MID-travel, then press ENTER...\n")
    while True:
        print(f"  raw encoder: {bus.read('Present_Position', m, normalize=False):>5}   ")
        if enter_pressed():
            break
        move_cursor_up(1)
        time.sleep(0.05)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sweep", "center"])
    p.add_argument("--port", default="/dev/tty.usbmodem5B8E1139971")
    p.add_argument("--motor", default="wrist_flex", choices=list(IDS))
    a = p.parse_args()

    bus = open_bus(a.port, a.motor)
    saved = save(bus, a.motor)
    print(f"saved {a.motor}: offset={saved['Homing_Offset']} "
          f"limits=[{saved['Min_Position_Limit']}, {saved['Max_Position_Limit']}]")
    try:
        (sweep if a.mode == "sweep" else center)(bus, a.motor)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            restore(bus, a.motor, saved)
        finally:
            bus.disconnect(disable_torque=False)
