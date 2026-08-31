#!/usr/bin/env python
"""Read-only health dump for an SO-101 leader/follower pair.

Writes NOTHING to the servos and does not touch torque, so it is safe to run
while the arms are powered and holding a pose.
"""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
RES = 4096  # sts3215 encoder counts

# Feetech SMS/STS "servo status" byte @ addr 65 (same bit layout as
# Unloading_Condition @19 and LED_Alarm_Condition @20).
STATUS_BITS = [
    (0x01, "VOLTAGE"),
    (0x02, "SENSOR"),
    (0x04, "TEMPERATURE"),
    (0x08, "CURRENT"),
    (0x10, "ANGLE"),
    (0x20, "OVERLOAD"),
]


def bits(v):
    on = [name for mask, name in STATUS_BITS if v & mask]
    return f"0x{v:02X} [{'|'.join(on) if on else 'ok'}]"


def dump(port, label):
    bus = FeetechMotorsBus(
        port=port,
        motors={n: Motor(i, "sts3215", MotorNormMode.DEGREES) for n, i in MOTORS.items()},
    )
    bus.connect()
    print(f"\n{'=' * 100}\n{label}  ({port})\n{'=' * 100}")

    hdr = f"{'MOTOR':<14}{'id':>3}{'pres':>7}{'goal':>7}{'ofs':>7}{'lo':>6}{'hi':>6}{'load':>7}{'V':>6}{'°C':>5}{'trq':>4}{'mode':>5}  status"
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for name in MOTORS:
        r = lambda k: bus.read(k, name, normalize=False)  # noqa: E731
        d = dict(
            pres=r("Present_Position"),
            goal=r("Goal_Position"),
            ofs=r("Homing_Offset"),
            lo=r("Min_Position_Limit"),
            hi=r("Max_Position_Limit"),
            load=r("Present_Load"),
            volt=r("Present_Voltage"),
            temp=r("Present_Temperature"),
            trq=r("Torque_Enable"),
            mode=r("Operating_Mode"),
            status=r("Status"),
            unload=r("Unloading_Condition"),
            led=r("LED_Alarm_Condition"),
            maxtrq=r("Max_Torque_Limit"),
            trqlim=r("Torque_Limit"),
            prot_t=r("Protective_Torque"),
            over_t=r("Overload_Torque"),
            prot_ms=r("Protection_Time"),
            fw=(r("Firmware_Major_Version"), r("Firmware_Minor_Version")),
        )
        rows[name] = d
        print(
            f"{name:<14}{MOTORS[name]:>3}{d['pres']:>7}{d['goal']:>7}{d['ofs']:>7}{d['lo']:>6}{d['hi']:>6}"
            f"{d['load']:>7}{d['volt'] / 10:>6.1f}{d['temp']:>5}{d['trq']:>4}{d['mode']:>5}  {bits(d['status'])}"
        )

    # --- encoder-wrap analysis -------------------------------------------------
    # Feetech firmware computes  Present_Position = Actual_Position - Homing_Offset,
    # where Actual_Position is the 12-bit absolute encoder (0..4095) fixed by how the
    # output horn is bolted on. If the joint's travel spans the 4095->0 seam, the
    # servo cannot follow a goal across it.
    print(f"\n{'MOTOR':<14}{'raw@lo':>8}{'raw@hi':>8}{'travel°':>9}   verdict")
    print("-" * 64)
    for name, d in rows.items():
        if name == "wrist_roll":
            print(f"{name:<14}{'-':>8}{'-':>8}{'-':>9}   (full-turn joint, not range-limited)")
            continue
        raw_lo = d["lo"] + d["ofs"]
        raw_hi = d["hi"] + d["ofs"]
        span = (d["hi"] - d["lo"]) * 360 / (RES - 1)
        crosses = not (0 <= raw_lo <= RES - 1 and 0 <= raw_hi <= RES - 1)
        margin = min(raw_lo, RES - 1 - raw_hi)
        verdict = (
            f"** SEAM INSIDE TRAVEL ** raw range {raw_lo}..{raw_hi} escapes 0..4095"
            if crosses
            else f"ok (margin to seam: {margin} ticks = {margin * 360 / (RES - 1):.0f} deg)"
        )
        print(f"{name:<14}{raw_lo:>8}{raw_hi:>8}{span:>9.1f}   {verdict}")

    print("\nprotection / alarm config (should be identical across all 6):")
    for name, d in rows.items():
        print(
            f"  {name:<14} fw={d['fw'][0]}.{d['fw'][1]}  unloading_cond={bits(d['unload'])} "
            f"led_alarm={bits(d['led'])} max_torque={d['maxtrq']} torque_limit={d['trqlim']} "
            f"protective_torque={d['prot_t']} overload_torque={d['over_t']} protection_time={d['prot_ms']}"
        )

    bus.disconnect(disable_torque=False)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--follower", default="/dev/tty.usbmodem5B8E1139971")
    p.add_argument("--leader", default="/dev/tty.usbmodem5B910455171")
    a = p.parse_args()
    dump(a.follower, "FOLLOWER")
    dump(a.leader, "LEADER")
