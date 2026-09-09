"""Run BFM-Zero's OWN motion library and observation builder on LAFAN1, dump the result.

    <refvenv>/bin/python tools/g1/lafan_reference_dump.py \
        --pkl references/BFM-Zero-main/humanoidverse/data/lafan_29dof.pkl \
        --out /path/to/lafan_reference.npz [--max-motions N]

This is the ORACLE for the G1 rung of `docs/BFM_ZERO_G1_REPRODUCTION.md`:
it is the reference's code, not a transcription of it. It builds
`MotionLibRobot` from the motion block of `g1_29dof_hard_waist.yaml`, loads
every clip the way `load_motions_for_training()` does (axis-angle -> joint
angles by `pose.sum(-1)`, torch FK on the 30-body skeleton + the virtual
`head_link` 0.35 m above `torso_link`, velocities by central differences at
the clip's own fps then a Gaussian filter of sigma 2, joint velocities by
one-sided differences), samples every clip at the environment's 0.02 s the
way `load_expert_trajectories_from_motion_lib` does (`get_motion_state` at
`arange(ceil(len / dt)) * dt`: linear blend of positions and velocities,
slerp of rotations), and computes the two observations the FB nets consume:

    state            (T, 64)   [dof_pos - default, dof_vel, projected gravity,
                               root angular velocity]  — NOTE: unscaled here,
                               the env applies the 0.25 on the angular velocity
                               and the 4x history separately
    privileged_state (T, 463)  `compute_humanoid_observations_max` over the 31
                               bodies with `root_height_obs=True`

⚠ `compute_humanoid_observations_max` lives in `legged_robot_motions.py`,
whose module imports the simulator stack. Its SOURCE is extracted from the
file and executed against the reference's own `torch_utils` helpers, so the
function that runs is theirs byte for byte without importing the module.

⚠ NEEDS TORCH. The pixi env has none; build a scratch venv with CPU torch,
joblib, scipy, lxml, easydict, loguru, rich, hydra-core, omegaconf, numpy.

Dumped per clip (float32 unless stated), concatenated over clips in the
pickle's key order with `ep_len`:
    key            str
    fps            the clip's own fps (30)
    n_src_frames   frames in the clip at its own fps
    root_pos       (T, 3)      world, from the extended FK (body 0 = pelvis)
    root_quat_xyzw (T, 4)
    dof_pos        (T, 29)     raw joint angles (NOT default-subtracted)
    dof_vel        (T, 29)
    body_pos       (T, 31, 3)  world, skeleton order + head_link last
    body_quat_xyzw (T, 31, 4)
    body_vel       (T, 31, 3)
    body_ang_vel   (T, 31, 3)
    state          (T, 64)
    privileged     (T, 463)
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "references" / "BFM-Zero-main"
ORIG_CWD = Path.cwd()  # resolve the CLI paths against where the user ran from
sys.path.insert(0, str(REF))
os.chdir(REF)  # the motion cfg's asset paths are relative to the reference root

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from easydict import EasyDict  # noqa: E402

from humanoidverse.utils import torch_utils as tu  # noqa: E402
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot  # noqa: E402

ROBOT_YAML = REF / "humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml"
OBS_SRC = REF / "humanoidverse/envs/legged_robot_motions/legged_robot_motions.py"
ENV_DT = 0.02  # `sim.fps 200` / `control_decimation 4`


def reference_obs_builder():
    """`compute_humanoid_observations_max`, extracted from the file it lives in.

    Executed with the reference's `torch_utils` helpers in scope, exactly the
    names the function body uses (`calc_heading_quat_inv`, `my_quat_rotate`,
    `quat_mul`, `quat_to_tan_norm`). The `@torch.jit.script` decorator is
    dropped: scripting needs the module's typed imports, and eager execution
    of the same source is the same arithmetic.
    """
    src = OBS_SRC.read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_humanoid_observations_max"
    )
    fn.decorator_list = []
    code = ast.get_source_segment(src, fn)
    # strip the decorator line(s) the segment may still carry
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("@"))
    ns = {
        "torch": torch,
        "OrderedDict": __import__("collections").OrderedDict,
        "calc_heading_quat_inv": tu.calc_heading_quat_inv,
        "my_quat_rotate": tu.my_quat_rotate,
        "quat_mul": tu.quat_mul,
        "quat_to_tan_norm": tu.quat_to_tan_norm,
    }
    exec(code, ns)
    return ns["compute_humanoid_observations_max"]


def motion_cfg(pkl: Path):
    y = yaml.safe_load(open(ROBOT_YAML))
    m = y["robot"]["motion"]
    cfg = EasyDict(m)
    cfg.motion_file = str(pkl)
    cfg.step_dt = ENV_DT
    cfg.asset = EasyDict(cfg.asset)
    cfg.extend_config = [EasyDict(e) for e in cfg.extend_config]
    return cfg, y


def default_dof_pos(y):
    names = y["robot"]["dof_names"]
    d = {k.strip(): v for k, v in y["robot"]["init_state"]["default_joint_angles"].items()}
    return torch.tensor([float(d[n]) for n in names], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-motions", type=int, default=None)
    args = ap.parse_args()
    pkl = (ORIG_CWD / args.pkl).resolve()
    out = (ORIG_CWD / args.out).resolve()

    cfg, y = motion_cfg(pkl)
    obs_fn = reference_obs_builder()
    default = default_dof_pos(y)
    gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)

    lib = MotionLibRobot(cfg, num_envs=1, device="cpu")
    lib.load_motions_for_training()
    n = lib._num_unique_motions
    if args.max_motions is not None:
        n = min(n, args.max_motions)
    print(f"reference motion lib: {lib._num_unique_motions} motions, dumping {n}")

    cols = {k: [] for k in (
        "root_pos", "root_quat_xyzw", "dof_pos", "dof_vel", "body_pos",
        "body_quat_xyzw", "body_vel", "body_ang_vel", "state", "privileged",
    )}
    keys, fps, n_src, ep_len = [], [], [], []
    for i in range(n):
        length = float(lib._motion_lengths[i])
        times = torch.arange(int(np.ceil(length / ENV_DT))) * ENV_DT
        ids = torch.full((times.shape[0],), i, dtype=torch.long)
        res = lib.get_motion_state(ids, times)
        body_pos = res["rg_pos_t"]
        body_rot = res["rg_rot_t"]
        body_vel = res["body_vel_t"]
        body_ang = res["body_ang_vel_t"]
        obs = obs_fn(body_pos, body_rot, body_vel, body_ang, True, True)
        privileged = torch.cat([v for v in obs.values()], dim=-1)
        base_quat = body_rot[:, 0]
        dof_pos = res["dof_pos"]
        dof_vel = res["dof_vel"]
        ang_vel = body_ang[:, 0]
        proj_g = tu.quat_rotate_inverse(base_quat, gravity.repeat(body_pos.shape[0], 1), w_last=True)
        state = torch.cat([dof_pos - default, dof_vel, proj_g, ang_vel], dim=-1)
        assert state.shape[1] == 64 and privileged.shape[1] == 463, (state.shape, privileged.shape)
        T = state.shape[0]
        for k, v in (
            ("root_pos", body_pos[:, 0]), ("root_quat_xyzw", base_quat),
            ("dof_pos", dof_pos), ("dof_vel", dof_vel), ("body_pos", body_pos),
            ("body_quat_xyzw", body_rot), ("body_vel", body_vel),
            ("body_ang_vel", body_ang), ("state", state), ("privileged", privileged),
        ):
            cols[k].append(v.detach().cpu().numpy().astype(np.float32))
        keys.append(str(lib._motion_data_keys[i]))
        fps.append(float(lib._motion_fps[i]))
        n_src.append(int(lib._motion_num_frames[i]))
        ep_len.append(T)
        print(f"  [{i:3d}] {keys[-1]:32s} src {n_src[-1]:6d} @ {fps[-1]:.0f} fps -> {T:6d} rows at 50 Hz")

    out_arrays = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    out_arrays["ep_len"] = np.asarray(ep_len, dtype=np.int64)
    out_arrays["fps"] = np.asarray(fps, dtype=np.float64)
    out_arrays["n_src_frames"] = np.asarray(n_src, dtype=np.int64)
    out_arrays["key"] = np.asarray(keys)
    out_arrays["default_dof_pos"] = default.numpy()
    out_arrays["env_dt"] = np.asarray(ENV_DT)
    np.savez_compressed(out, **out_arrays)
    print(f"wrote {out}: {int(out_arrays['ep_len'].sum())} rows over {n} clips")


if __name__ == "__main__":
    main()
