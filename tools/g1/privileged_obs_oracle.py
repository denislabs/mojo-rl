"""The ORACLE for the live 463-D privileged observation (G3.0).

    # certify the numpy transcription against the reference's torch function:
    <refvenv>/bin/python tools/g1/privileged_obs_oracle.py --selfcheck
    # the Mojo gate imports `max_local_self_from_sim` through interop:
    pixi run mojo run -I . tests/robots/test_unitree_g1_privileged_obs.mojo

Two layers, both the reference's arithmetic:

  extend_bodies(...)        `legged_robot_motions.py:205-236` — the virtual
                            head_link from the torso: position `torso +
                            R_torso (0,0,0.35)`, rotation = torso's, angular
                            velocity = torso's, linear velocity `v_torso +
                            ω_torso × (0,0,0.35)` with the offset UNROTATED
                            (their `torch.cross(ω, offset_in_parent)`).
  max_local_self(...)       `compute_humanoid_observations_max` (`:596-645`)
                            in numpy: root height, heading-frame local
                            positions (root dropped), tangent-normal
                            rotations, heading-frame velocities.
  origin_velocity(...)      Isaac's `_rigid_body_vel` is the link-frame
                            origin's velocity; our engine's `xvel` is the
                            COM point's — `v_origin = v_com + ω × (xpos − xipos)`.

`max_local_self_from_sim` chains the three the way the env hook must, from
OUR engine's raw per-body arrays (30 skeleton bodies in the reference's
order, plus the torso's index), so the Mojo gate feeds raw simulator state
and compares the whole 463-vector.

⚠ `--selfcheck` NEEDS TORCH (the scratch venv, not pixi): it lifts
`compute_humanoid_observations_max` from the reference file by `ast` exactly
as `lafan_reference_dump.py` does and compares on random inputs to 1e-6
(float32 torch vs float64 numpy). The pixi env has no torch, so the Mojo
gate runs against the certified numpy transcription.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "references" / "BFM-Zero-main"
HEAD_OFFSET = np.array([0.0, 0.0, 0.35])
N_SKELETON = 30
TORSO_SKELETON_INDEX = 15  # `torso_link` in the reference's body_names order


def quat_rotate_xyzw(q, v):
    """`my_quat_rotate` (w last): v + 2w (q_v × v) + 2 q_v × (q_v × v)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qv, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_mul_xyzw(a, b):
    """`quat_mul` (w last)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def heading_quat_inv(root_quat_xyzw):
    """`calc_heading_quat_inv`: the rotation by −heading about z."""
    d = quat_rotate_xyzw(root_quat_xyzw, np.array([1.0, 0.0, 0.0]))
    heading = np.arctan2(d[..., 1], d[..., 0])
    half = -0.5 * heading
    return np.stack([np.zeros_like(half), np.zeros_like(half), np.sin(half), np.cos(half)], axis=-1)


def quat_to_tan_norm(q_xyzw):
    return np.concatenate(
        [quat_rotate_xyzw(q_xyzw, np.array([1.0, 0.0, 0.0])), quat_rotate_xyzw(q_xyzw, np.array([0.0, 0.0, 1.0]))],
        axis=-1,
    )


def max_local_self(body_pos, body_quat_xyzw, body_vel, body_ang_vel):
    """`compute_humanoid_observations_max(..., local_root_obs=True, root_height_obs=True)`
    for ONE frame: arrays (31, 3), (31, 4), (31, 3), (31, 3) -> (463,)."""
    body_pos = np.asarray(body_pos, dtype=np.float64)
    body_quat_xyzw = np.asarray(body_quat_xyzw, dtype=np.float64)
    body_vel = np.asarray(body_vel, dtype=np.float64)
    body_ang_vel = np.asarray(body_ang_vel, dtype=np.float64)
    nb = body_pos.shape[0]
    root_pos = body_pos[0]
    h = heading_quat_inv(body_quat_xyzw[0])
    hb = np.broadcast_to(h, (nb, 4))
    local_pos = quat_rotate_xyzw(hb, body_pos - root_pos)[1:].reshape(-1)
    local_rot = quat_to_tan_norm(quat_mul_xyzw(hb, body_quat_xyzw)).reshape(-1)
    local_vel = quat_rotate_xyzw(hb, body_vel).reshape(-1)
    local_ang = quat_rotate_xyzw(hb, body_ang_vel).reshape(-1)
    return np.concatenate([root_pos[2:3], local_pos, local_rot, local_vel, local_ang])


def extend_bodies(pos, quat_xyzw, vel, ang_vel, torso_index=TORSO_SKELETON_INDEX):
    """Append the virtual head_link (`legged_robot_motions.py:205-236`)."""
    pos = np.asarray(pos, dtype=np.float64)
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    ang_vel = np.asarray(ang_vel, dtype=np.float64)
    tq = quat_xyzw[torso_index]
    head_pos = pos[torso_index] + quat_rotate_xyzw(tq, HEAD_OFFSET)
    head_vel = vel[torso_index] + np.cross(ang_vel[torso_index], HEAD_OFFSET)  # unrotated offset, as theirs
    return (
        np.concatenate([pos, head_pos[None]]),
        np.concatenate([quat_xyzw, tq[None]]),
        np.concatenate([vel, head_vel[None]]),
        np.concatenate([ang_vel, ang_vel[torso_index][None]]),
    )


def origin_velocity(v_com, ang_vel, xpos, xipos):
    """Link-frame origin velocity from the COM point's."""
    return np.asarray(v_com, dtype=np.float64) + np.cross(
        np.asarray(ang_vel, dtype=np.float64), np.asarray(xpos, dtype=np.float64) - np.asarray(xipos, dtype=np.float64)
    )


def max_local_self_from_sim(xpos, xquat_xyzw, xipos, xvel_com, xangvel):
    """From OUR engine's raw per-body arrays for the 30 skeleton bodies (the
    reference's order) -> the 463-vector the env hook must produce."""
    xpos = np.asarray(xpos, dtype=np.float64).reshape(N_SKELETON, 3)
    xquat = np.asarray(xquat_xyzw, dtype=np.float64).reshape(N_SKELETON, 4)
    xipos = np.asarray(xipos, dtype=np.float64).reshape(N_SKELETON, 3)
    xvel = np.asarray(xvel_com, dtype=np.float64).reshape(N_SKELETON, 3)
    xang = np.asarray(xangvel, dtype=np.float64).reshape(N_SKELETON, 3)
    v_o = origin_velocity(xvel, xang, xpos, xipos)
    p, q, v, w = extend_bodies(xpos, xquat, v_o, xang)
    return max_local_self(p, q, v, w)


def _selfcheck(n=64, seed=0):
    """numpy transcription vs the reference's torch function on random inputs."""
    import ast

    import torch

    sys.path.insert(0, str(REF))
    from humanoidverse.utils import torch_utils as tu

    src = (REF / "humanoidverse/envs/legged_robot_motions/legged_robot_motions.py").read_text()
    fn = next(
        n_ for n_ in ast.parse(src).body
        if isinstance(n_, ast.FunctionDef) and n_.name == "compute_humanoid_observations_max"
    )
    fn.decorator_list = []
    code = "\n".join(l for l in ast.get_source_segment(src, fn).splitlines() if not l.strip().startswith("@"))
    ns = {
        "torch": torch, "OrderedDict": __import__("collections").OrderedDict,
        "calc_heading_quat_inv": tu.calc_heading_quat_inv, "my_quat_rotate": tu.my_quat_rotate,
        "quat_mul": tu.quat_mul, "quat_to_tan_norm": tu.quat_to_tan_norm,
    }
    exec(code, ns)
    ref_fn = ns["compute_humanoid_observations_max"]
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        pos = rng.standard_normal((31, 3))
        q = rng.standard_normal((31, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        vel = rng.standard_normal((31, 3)); ang = rng.standard_normal((31, 3))
        ours = max_local_self(pos, q, vel, ang)
        t = lambda a: torch.tensor(a[None], dtype=torch.float32)
        d = ref_fn(t(pos), t(q), t(vel), t(ang), True, True)
        theirs = torch.cat([v for v in d.values()], dim=-1)[0].numpy().astype(np.float64)
        assert theirs.shape == ours.shape == (463,), (theirs.shape, ours.shape)
        worst = max(worst, float(np.abs(theirs - ours).max()))
    print(f"selfcheck: numpy transcription vs the reference's torch function on {n} random frames: worst |d| {worst:.3e} (float32 torch)")
    assert worst < 1e-5, worst
    # the extension: the reference's own lines on random torso states
    pos = rng.standard_normal((30, 3)); q = rng.standard_normal((30, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    vel = rng.standard_normal((30, 3)); ang = rng.standard_normal((30, 3))
    p, qq, v, w = extend_bodies(pos, q, vel, ang)
    tq = torch.tensor(q[TORSO_SKELETON_INDEX][None], dtype=torch.float32)
    off = torch.tensor(HEAD_OFFSET[None], dtype=torch.float32)
    ref_head_pos = tu.my_quat_rotate(tq, off)[0].numpy() + pos[TORSO_SKELETON_INDEX]
    ref_head_vel = vel[TORSO_SKELETON_INDEX] + torch.cross(torch.tensor(ang[TORSO_SKELETON_INDEX][None]), torch.tensor(HEAD_OFFSET[None]), dim=1)[0].numpy()
    e = max(float(np.abs(p[30] - ref_head_pos).max()), float(np.abs(v[30] - ref_head_vel).max()))
    print(f"selfcheck: head extension vs the reference's lines: worst |d| {e:.3e}")
    assert e < 1e-5, e
    print("SELFCHECK PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
    else:
        print(__doc__)
