#!/usr/bin/env python3
"""Print what a `.demo` file holds — episodes, returns, flags, action ranges.

    pixi run python tools/demo/demo_stats.py projects/so101-tower/demos/*.demo
    pixi run python tools/demo/demo_stats.py FILE --episodes      # one line per episode

The format is `mojo_rl/deep_agents/data/demo_file.mojo`'s (version 1); this
is a READER for a human, not the loader the trainer uses — that one is Mojo
and gated by `tests/deep_agents/test_demo_file.mojo`. Numpy only.
"""

import struct
import sys

import numpy as np

MAGIC = b"MRLDEMO1"
FLAG_INTERVENED = 1
FLAG_SUCCESS = 2


def read_demo(path):
    with open(path, "rb") as f:
        b = f.read()
    if b[:8] != MAGIC:
        raise SystemExit(f"{path}: not a .demo (magic {b[:8]!r})")
    version, obs_dim, act_dim = struct.unpack_from("<III", b, 8)
    (n,) = struct.unpack_from("<Q", b, 20)
    (n_ep,) = struct.unpack_from("<I", b, 28)
    if version != 1:
        raise SystemExit(f"{path}: version {version}, this reader is 1")
    off = 32
    row_f32 = obs_dim + act_dim + 1 + obs_dim + 1
    row_bytes = 4 * row_f32 + 4
    rows = np.frombuffer(b, dtype=np.uint8, count=n * row_bytes, offset=off)
    rows = rows.reshape(n, row_bytes)
    floats = rows[:, : 4 * row_f32].copy().view("<f4").reshape(n, row_f32)
    flags = rows[:, 4 * row_f32 :].copy().view("<u4").reshape(n)
    off += n * row_bytes
    eps = []
    for _ in range(n_ep):
        s, l, ok = struct.unpack_from("<QQI", b, off)
        off += 20
        eps.append((s, l, bool(ok)))
    o = 0
    obs = floats[:, o : o + obs_dim]
    o += obs_dim
    act = floats[:, o : o + act_dim]
    o += act_dim
    rew = floats[:, o]
    o += 1
    nobs = floats[:, o : o + obs_dim]
    o += obs_dim
    done = floats[:, o]
    return dict(
        obs_dim=obs_dim, act_dim=act_dim, obs=obs, act=act, rew=rew,
        nobs=nobs, done=done, flags=flags, episodes=eps,
    )


def main(argv):
    per_episode = "--episodes" in argv
    paths = [a for a in argv[1:] if not a.startswith("--")]
    if not paths:
        print(__doc__)
        return 2
    for path in paths:
        d = read_demo(path)
        eps = d["episodes"]
        n = len(d["rew"])
        n_ok = sum(1 for e in eps if e[2])
        n_int = int((d["flags"] & FLAG_INTERVENED).astype(bool).sum())
        print(f"{path}")
        print(f"  obs {d['obs_dim']}  act {d['act_dim']}  rows {n}  episodes {len(eps)}"
              f"  successful {n_ok}  intervened rows {n_int}")
        if n:
            r = d["rew"]
            print(f"  reward   mean {r.mean():.4f}  min {r.min():.4f}  max {r.max():.4f}"
                  f"  rows>1.24 (jaws on brick) {(r > 1.24).sum()}"
                  f"  rows>1.5 (rung paid) {(r > 1.5).sum()}")
            a = d["act"]
            print(f"  action   min {np.round(a.min(0), 2).tolist()}")
            print(f"           max {np.round(a.max(0), 2).tolist()}")
            print(f"  done     nonzero rows {(d['done'] != 0).sum()}")
            lens = [e[1] for e in eps]
            if lens:
                print(f"  episode length  mean {np.mean(lens):.1f}  min {min(lens)}  max {max(lens)}")
        if per_episode:
            for k, (s, l, ok) in enumerate(eps):
                r = d["rew"][s : s + l]
                fl = d["flags"][s : s + l]
                print(f"    ep {k:3d}  start {s:6d}  len {l:4d}  {'SUCCESS' if ok else 'fail   '}"
                      f"  return {r.sum():8.2f}  mean r {r.mean():.3f}"
                      f"  intervened {(fl & FLAG_INTERVENED).astype(bool).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
