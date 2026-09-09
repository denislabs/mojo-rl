"""LAFAN1 (reference dump) -> a `TrajectoryStore` `.h5` the Mojo side reads.

    pixi run python tools/g1/lafan_to_store.py \
        --npz /path/to/lafan_reference.npz --out lafan_g1_50hz.h5

Input is `tools/g1/lafan_reference_dump.py`'s output — the reference's OWN
loader and observation builder run on every clip — so this file adds no
arithmetic: it lays the columns out in `TrajectoryStoreWriter`'s format
(`mojo_rl/data/store.mojo`), one episode per clip, and writes the manifest
byte-for-byte the way `tools/act/lerobot_v3_to_store.py::encode_manifest`
does (imported, not copied).

Columns (rank-2, one row per 50 Hz frame):

    qpos          f4 (36)   [root_pos 3, root_quat WXYZ 4, dof_pos 29] —
                            MuJoCo's qpos layout for `unitree_g1.xml`, so a
                            row drops straight into `Phyics3dEnv.set_state`
    qvel          f4 (35)   [root lin vel WORLD 3, root ang vel WORLD 3,
                            dof_vel 29] — see the note below
    state         f4 (64)   the reference's proprio `state`, unscaled
    privileged    f4 (463)  the reference's `max_local_self`
    body_pos      f4 (93)   31 bodies x 3, world, skeleton order + head_link
    body_quat     f4 (124)  31 bodies x 4, XYZW as the reference stores it
    body_vel      f4 (93)
    body_ang_vel  f4 (93)
    motion_id     i4        clip index in the pickle's key order (rank-1, like `task`)

⚠ THE ROOT VELOCITIES ARE THE REFERENCE'S FILTERED FINITE DIFFERENCES IN THE
WORLD FRAME (`_compute_velocity` / `_compute_angular_velocity`, Gaussian
sigma 2 at 30 fps, then blended to 50 Hz). MuJoCo's free-joint `qvel[3:6]` is
the angular velocity in the BODY frame, and its `qvel[0:3]` is exact, not
filtered. So `qvel` here is the reference's kinematic estimate laid out in
MuJoCo's slots, not a state the simulator produced; a reset from this row
followed by one step will not reproduce the next row exactly, and that is
the reference's property too (its reference-state init uses the same
numbers). The angular part is left in the WORLD frame deliberately — the
reference's `state[61:64]` is `body_ang_vel_t[:, 0]`, world frame, and the
env's live `base_ang_vel` is body frame, which the paper's obs pipeline does
not reconcile either. `docs/BFM_ZERO_G1_REPRODUCTION.md` §10 records this.

⚠ `state[0:29]` is `dof_pos - default_dof_pos` and `state[61:64]` is NOT yet
scaled by 0.25; the env applies `obs_scales` to the live observation and the
expert rows are built from `raw_obs` AFTER scaling in the reference
(`humanoidverse_isaac.py:404-450`), so the consumer that pairs expert rows
with live rows must apply the same scale. Recorded, not silently applied.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "act"))
from lerobot_v3_to_store import encode_manifest  # noqa: E402

ENV_ID = "unitree_g1_lafan1_50hz"


def source_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    ep_len = z["ep_len"].astype(np.int64)
    n_rows = int(ep_len.sum())
    n_ep = int(ep_len.shape[0])
    ep_offset = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)

    root_pos = z["root_pos"]
    root_quat = quat_xyzw_to_wxyz(z["root_quat_xyzw"])
    dof_pos = z["dof_pos"]
    dof_vel = z["dof_vel"]
    body_vel = z["body_vel"]
    body_ang = z["body_ang_vel"]
    qpos = np.concatenate([root_pos, root_quat, dof_pos], axis=1).astype(np.float32)
    qvel = np.concatenate([body_vel[:, 0], body_ang[:, 0], dof_vel], axis=1).astype(np.float32)
    assert qpos.shape[1] == 36 and qvel.shape[1] == 35, (qpos.shape, qvel.shape)
    motion_id = np.repeat(np.arange(n_ep, dtype=np.int32), ep_len)  # rank-1, like `task`

    cols = [
        ("qpos", qpos),
        ("qvel", qvel),
        ("state", z["state"].astype(np.float32)),
        ("privileged", z["privileged"].astype(np.float32)),
        ("body_pos", z["body_pos"].reshape(n_rows, -1).astype(np.float32)),
        ("body_quat", z["body_quat_xyzw"].reshape(n_rows, -1).astype(np.float32)),
        ("body_vel", body_vel.reshape(n_rows, -1).astype(np.float32)),
        ("body_ang_vel", body_ang.reshape(n_rows, -1).astype(np.float32)),
        ("motion_id", motion_id),
    ]
    for name, arr in cols:
        assert arr.shape[0] == n_rows, (name, arr.shape, n_rows)

    manifest_cols = [
        (name, "float32" if arr.dtype == np.float32 else "int32",
         (int(arr.shape[1]),) if arr.ndim == 2 else ())
        for name, arr in cols
    ]
    manifest = encode_manifest(
        ENV_ID, n_rows, n_ep, seed=0, source_commit=source_commit(),
        columns=manifest_cols,
    )

    out = Path(args.out)
    tmp = out.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as f:
        for name, arr in cols:
            chunk = (min(4096, n_rows),) + tuple(arr.shape[1:])
            f.create_dataset(
                name, data=arr, dtype="f4" if arr.dtype == np.float32 else "i4",
                chunks=chunk, maxshape=(None,) + tuple(arr.shape[1:]),
            )
        f.create_dataset("ep_len", data=ep_len)
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset(
            "__manifest__", data=np.frombuffer(manifest.encode("utf-8"), dtype=np.uint8)
        )
        # Side tables, not manifest columns: the clip names and the defaults
        # the state column was built with.
        f.create_dataset("motion_key", data=np.asarray([str(k) for k in z["key"]], dtype="S"))
        f.create_dataset("default_dof_pos", data=z["default_dof_pos"].astype(np.float32))
        f.create_dataset("env_dt", data=np.asarray(float(z["env_dt"])))
    os.replace(tmp, out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"wrote {out}: {n_rows} rows, {n_ep} episodes, {len(cols)} columns, sha256 {h}")
    print(manifest)


if __name__ == "__main__":
    main()
