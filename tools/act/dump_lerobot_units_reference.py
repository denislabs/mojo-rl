"""Pin lerobot's tick<->units arithmetic, taken from lerobot's OWN SOURCE TEXT.

    pixi run python tools/act/dump_lerobot_units_reference.py
    ... --lerobot ../lerobot/lerobot/src/lerobot/motors/motors_bus.py

⚠⚠ **THE EXPRESSIONS ARE EXTRACTED, NOT TRANSCRIBED, AND THAT IS THE WHOLE
POINT.** A hand-copied formula in a gate proves that two of my transcriptions
agree with each other. This reads the exact lines out of
`MotorsBus._normalize` / `_unnormalize` and evaluates them, so the fixture is
lerobot's arithmetic. If lerobot changes those lines, the extraction FAILS
LOUDLY and someone has to look — which is the outcome you want, because the
alternative is a robot quietly speaking last year's units.

⚠ lerobot cannot simply be imported here: it is not in the pixi env, and adding
it pulls torch and draccus for six lines of arithmetic. Reading the source is
cheaper and is a stronger gate than a transcription either way.

WHICH MODE THE SO-101 USES IS A RECORDING-TIME FLAG, NOT A FACT ABOUT THE ARM.
`so_follower.py` reads:

    norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)

and `config_so_follower.py` has `use_degrees: bool = True` ("for backward
compatibility with previous policies/datasets"). So the DEFAULT is DEGREES —
but a dataset recorded with the flag flipped carries different numbers under
the same column names, and nothing in the file says which. See
`--check-dataset`.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/fixtures/robot/lerobot_units.txt"
DEFAULT_SRC = ROOT.parent / "lerobot/lerobot/src/lerobot/motors/motors_bus.py"

# (name, regex capturing the right-hand side) — anchored on the assignment so a
# renamed variable or a changed constant cannot match by accident.
WANTED = {
    "norm_m100_100": r"norm = (\(\(\(bounded_val - min_\) / \(max_ - min_\)\) \* 200\) - 100)",
    "norm_0_100": r"norm = (\(\(bounded_val - min_\) / \(max_ - min_\)\) \* 100)",
    "norm_degrees": r"normalized_values\[id_\] = (\(val - mid\) \* 360 / max_res)",
    "unnorm_m100_100": r"unnormalized_values\[id_\] = (int\(\(\(bounded_val \+ 100\) / 200\) \* \(max_ - min_\) \+ min_\))",
    "unnorm_0_100": r"unnormalized_values\[id_\] = (int\(\(bounded_val / 100\) \* \(max_ - min_\) \+ min_\))",
    "unnorm_degrees": r"unnormalized_values\[id_\] = (int\(\(val \* max_res / 360\) \+ mid\))",
}

# A real SO-101 follower calibration shape: five swept joints and one gripper,
# plus `wrist_roll` carrying lerobot's UNLIMITED marker (0, 4095).
JOINTS = [
    ("shoulder_pan", 812, 3170, "body"),
    ("shoulder_lift", 720, 3260, "body"),
    ("elbow_flex", 900, 3100, "body"),
    ("wrist_flex", 780, 3250, "body"),
    ("wrist_roll", 0, 4095, "body"),
    ("gripper", 2020, 3480, "gripper"),
]
RESOLUTION = 4096


def extract(src: Path) -> dict:
    text = src.read_text()
    out = {}
    for name, pat in WANTED.items():
        m = re.search(pat, text)
        if not m:
            print(f"could not find `{name}` in {src}.\n"
                  "  lerobot's normalization has changed — read it and update "
                  "both this script and mojo_rl/robot/so101/arm.mojo.",
                  file=sys.stderr)
            raise SystemExit(1)
        out[name] = m.group(1)
    return out


def check_dataset(root: Path) -> int:
    """Say which norm mode a recorded dataset implies, from its own stats.

    ⚠⚠ THE DATASET IS THE ONLY WITNESS. Both modes write `*.pos` columns and
    neither records which was used. `RANGE_M100_100` CLAMPS to [-100, 100] by
    construction, so any value outside that range is proof of `DEGREES`. The
    converse is NOT proof: an operator who never reached an end stop produces
    in-range values under either mode, and this says so rather than guessing.
    """
    import json
    stats = root / "meta" / "stats.json"
    if not stats.exists():
        print(f"no {stats}", file=sys.stderr)
        return 1
    st = json.loads(stats.read_text())
    verdict = 1
    for col in ("observation.state", "action"):
        if col not in st:
            continue
        mn, mx = st[col]["min"], st[col]["max"]
        outside = [i for i in range(len(mn)) if mn[i] < -100.0 or mx[i] > 100.0]
        print(f"{col}:")
        for i in range(len(mn)):
            flag = "  <-- outside +-100" if i in outside else ""
            print(f"  [{i}] {mn[i]:8.2f} .. {mx[i]:8.2f}{flag}")
        if outside:
            print(f"  => DEGREES. Joints {outside} exceed +-100, which "
                  "RANGE_M100_100 cannot produce.")
            verdict = 0
        else:
            print("  => INCONCLUSIVE. Everything is inside +-100, which BOTH "
                  "modes can produce.\n"
                  "     Check the recording config's `use_degrees` "
                  "(default True) before deploying.")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--check-dataset", type=Path, default=None,
                    help="a LeRobot dataset root; report which norm mode it implies")
    args = ap.parse_args()

    if args.check_dataset is not None:
        return check_dataset(args.check_dataset)
    if not args.lerobot.exists():
        print(f"no lerobot source at {args.lerobot}", file=sys.stderr)
        return 1

    expr = extract(args.lerobot)
    print(f"extracted 6 expressions from {args.lerobot}")
    for k, v in expr.items():
        print(f"  {k:18} {v}")

    lines = [f"# generated from {args.lerobot}", f"resolution {RESOLUTION}"]
    for name, min_, max_, kind in JOINTS:
        lines.append(f"joint {name} {min_} {max_} {kind}")

    max_res = RESOLUTION - 1
    # Ticks spread over and BEYOND the calibrated range: the clamped modes and
    # the unclamped one only disagree outside it, which is exactly where the
    # 50-demo dataset's shoulder_lift lives.
    for name, min_, max_, kind in JOINTS:
        mid = (min_ + max_) / 2
        for val in (min_ - 300, min_, min_ + 1, (min_ + max_) // 2, max_ - 1,
                    max_, max_ + 300):
            bounded_val = min(max_, max(min_, val))
            deg = eval(expr["norm_degrees"])                       # noqa: S307
            m100 = eval(expr["norm_m100_100"])                     # noqa: S307
            p100 = eval(expr["norm_0_100"])                        # noqa: S307
            lines.append(f"norm {name} {val} {deg!r} {m100!r} {p100!r}")

    # And the inverse, from units back to ticks.
    for name, min_, max_, kind in JOINTS:
        mid = (min_ + max_) / 2
        vals = ((-181.0, -90.0, -0.5, 0.0, 12.34, 90.0, 181.0) if kind == "body"
                else (-10.0, 0.0, 1.0, 37.5, 99.9, 100.0, 110.0))
        for val in vals:
            if kind == "body":
                back = eval(expr["unnorm_degrees"])                # noqa: S307
            else:
                bounded_val = min(100.0, max(0.0, val))
                back = eval(expr["unnorm_0_100"])                  # noqa: S307
            lines.append(f"unnorm {name} {val!r} {back}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
