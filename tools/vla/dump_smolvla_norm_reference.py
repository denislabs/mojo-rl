#!/usr/bin/env python3
"""Dump the reference state/action normalisation, for gating the Mojo boundary.

    pixi run -e act-ref python tools/vla/dump_smolvla_norm_reference.py \
        --stats ~/.cache/huggingface/lerobot/DenisLabs/record-test_20260828_092736/meta/stats.json \
        --out /tmp/vla_norm

Runs against the REAL `meta/stats.json` of the recording, because the thing most
worth pinning is not the arithmetic -- it is two lines -- but the SOURCE of the
numbers and the ORDER of the steps:

  * the stats must come from the dataset, not be recomputed. Ours differ from
    LeRobot's by exactly sqrt(N/(N-1)) on every std (sample vs population).
  * the padded dims must end at exactly 0, and the inverse must DROP them
    rather than unnormalise them -- the model emits values there that mean
    nothing, and `x*std+mean` on them yields plausible joint angles.

⚠ What this does NOT cover: `normalize_processor.py` divides by `std + 1e-8`
while its inverse multiplies by bare `std`. At float32 that difference is
1.35e-08 against a 3.8e-06 ulp -- unobservable. No output here can distinguish
the variants, and none pretends to.

Output is `refload.mojo`'s format (`manifest.txt` + `<name>.bin` float32 C order).
"""

import argparse
import json
import os

import numpy as np

DEFAULT_DATASET = os.path.expanduser(
    "~/.cache/huggingface/lerobot/DenisLabs/record-test_20260828_092736"
)

EPS = np.float32(1e-8)
MAX_STATE = 32
MAX_ACTION = 32
CHUNK = 50


class Dump:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.lines = []

    def add(self, name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        a.tofile(os.path.join(self.root, name + ".bin"))
        self.lines.append(f"{name}\t{','.join(str(d) for d in a.shape)}")
        print(f"  {name:20s} {str(a.shape):14s} min {a.min():+.6f} max {a.max():+.6f}")

    def close(self):
        with open(os.path.join(self.root, "manifest.txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=os.path.join(DEFAULT_DATASET, "meta", "stats.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = os.path.expanduser(a.stats)
    st = json.load(open(src))
    s_mean = np.array(st["observation.state"]["mean"], dtype=np.float32)
    s_std = np.array(st["observation.state"]["std"], dtype=np.float32)
    a_mean = np.array(st["action"]["mean"], dtype=np.float32)
    a_std = np.array(st["action"]["std"], dtype=np.float32)
    d_s, d_a = len(s_mean), len(a_mean)
    print(f"stats: state dim {d_s}, action dim {d_a}")

    d = Dump(a.out)
    # the parsed stats themselves — so the Mojo JSON read is checked, not assumed
    d.add("state_mean", s_mean)
    d.add("state_std", s_std)
    d.add("action_mean", a_mean)
    d.add("action_std", a_std)

    # a raw pose with structure, well outside +-1 so a missing normalisation
    # cannot look like a normalised value
    raw = (s_mean + s_std * np.array([1.7, -0.9, 0.4, 2.1, -1.3, 0.6][:d_s],
                                     dtype=np.float32)).astype(np.float32)
    d.add("state_raw", raw)

    # THE REFERENCE ORDER: normalise, then zero-pad.
    norm = ((raw - s_mean) / (s_std + EPS)).astype(np.float32)
    padded = np.zeros(MAX_STATE, dtype=np.float32)
    padded[:d_s] = norm
    d.add("state_norm", padded)

    # an action chunk as the model emits it, and its inverse
    rng = np.random.default_rng(11)
    chunk = rng.standard_normal((CHUNK, MAX_ACTION)).astype(np.float32)
    d.add("action_chunk", chunk)
    # ⚠ inverse branch: `tensor * std + mean`, NO eps. And the padded dims are
    # DROPPED, not unnormalised — the model emits values there that mean nothing.
    d.add("action_out", (chunk[:, :d_a] * a_std + a_mean).astype(np.float32))

    # Copy the source stats beside the dump so the Mojo gate reads THE SAME
    # file without hardcoding a path under someone's home directory.
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(st, f)

    d.close()
    print(f"\nwritten to {a.out}  (stats.json copied from {src})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
