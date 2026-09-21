#!/usr/bin/env python3
"""A LeRobot-shaped `stats.json` for SmolVLA, from an ACT run's exact `norm.json`.

    python3 tools/vla/smolvla_stats_from_act_norm.py \
        --norm projects/so101-tower/policies/act.norm.json \
        --out  projects/so101-tower/datasets/cube-in-bowl/meta/smolvla_stats.json

Standard library only, on purpose: it runs on a rented box without a new pixi
environment.

⚠⚠ WHY NOT THE RECORDING'S OWN `meta/stats.json`. Two reasons, both measured on
`cube-in-bowl`:

1. It carries NO `std` at all — only min/max/mean — and
   `SmolVLAStats.from_stats_json` requires one.
2. Its mean is the recorder's per-episode AGGREGATE over every episode it
   wrote, INCLUDING the ones discarded afterwards (56 recorded, 50 kept). It
   disagrees with the exact mean by ~1 degree on several joints.

`norm.json` is computed by `ACTDataset` over exactly the rows of the store —
the 50 kept episodes, 19 365 frames — which are exactly the rows a SmolVLA
store imported from the same recording holds. Normalising with statistics of
different rows than the ones trained on is a scale error with no symptom but a
worse policy.

⚠ The state/action statistics do not depend on the image size, so a norm.json
from a 240x320 ACT store is exact for a 480x640 SmolVLA store of the SAME
recording. It is NOT exact for a different recording — `--expect-rows` exists
to catch exactly that.

⚠ std CONVENTION: this repo's is the sample std, LeRobot's the population std.
They differ by sqrt(N/(N-1)) — at N = 19 365 that is 1.0000258, 0.003 %, and not
worth a conversion that could itself be wrong.
"""

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", required=True, help="an ACT norm.json")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--expect-rows",
        type=int,
        default=0,
        help="refuse unless norm.json covers exactly this many frames",
    )
    a = ap.parse_args()

    with open(a.norm) as f:
        n = json.load(f)

    if a.expect_rows and n.get("n_rows") != a.expect_rows:
        print(
            f"refusing: {a.norm} covers {n.get('n_rows')} rows, expected"
            f" {a.expect_rows} — it is not this recording",
            file=sys.stderr,
        )
        return 1

    def block(prefix: str) -> dict:
        out = {}
        for k in ("mean", "std", "min", "max"):
            v = n[f"{prefix}_{k}"]
            if not isinstance(v, list) or not v:
                raise SystemExit(f"{a.norm}: '{prefix}_{k}' is missing or empty")
            out[k] = [float(x) for x in v]
        dims = {len(x) for x in out.values()}
        if len(dims) != 1:
            raise SystemExit(f"{a.norm}: {prefix} vectors disagree in length {dims}")
        # A zero std means a joint never moved; normalising by it would send
        # that column to ~1e8. Refuse here rather than in the activations.
        for j, s in enumerate(out["std"]):
            if s <= 0.0:
                raise SystemExit(f"{a.norm}: {prefix} joint {j} has std {s}")
        return out

    stats = {"action": block("action"), "observation.state": block("qpos")}
    with open(a.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(
        f"wrote {a.out}  ({n.get('n_rows')} rows, {n.get('n_episodes')} episodes,"
        f" {len(stats['action']['mean'])}-dim action/state, from {n.get('store')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
