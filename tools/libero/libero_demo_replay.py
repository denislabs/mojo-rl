#!/usr/bin/env python
"""L4's gate: one LIBERO demo, three sims, one controller transcription.

    pixi run libero-replay <demo.hdf5> --demo 0 --out <dir>
    pixi run mojo run -I . examples/tasks/libero_demo_replay.mojo <dir>/demo_0_dump.txt <dir>/demo_0_ours.txt
    pixi run libero-replay <demo.hdf5> --demo 0 --out <dir> --compare

The demo files are HF `yifengzhu-hf/LIBERO-datasets` (one HDF5 per task,
50 demos each, 0.4-1 GB); `references/libero_demos/libero_goal/
turn_on_the_stove_demo.hdf5` (426 MB, gitignored) is the one L4 was gated
on. Needs `references/robosuite-1.4.0/robosuite/` (see `RS14`).

Measured 2026-09-13 on that demo (80 policy steps, 2000 substeps):

    ours vs MuJoCo 3.12 on our scene   grip site 1.5e-05 m max, 1.6e-06 mean;
                                        full qpos 2.0e-04 max; gripper 1.5e-05
    MuJoCo on our scene vs their model  2.2e-08 m (the two models are the same)
    ours vs the 2022 recording          1.59e-02 m at step 30 — IDENTICAL to what
                                        MuJoCo 3.12 itself shows against it

Columns (the `_a_third_column_makes_a_sim_to_sim_gate_decisive` shape):

    recorded   the file's `states` / `robot_states`, produced by robosuite
               1.4.0 + LIBERO on the MuJoCo of 2022 with THEIR merged model
    theirs     the same actions, replayed here with a verbatim transcription
               of the 1.4.0 OSC_POSE + gripper ramp + 25-substep loop, on
               MuJoCo 3.12 with THEIR merged model (asset paths rewritten)
    mujoco     the same transcription on OUR composed `libero_goal` scene,
               MuJoCo 3.12
    ours       OUR engine + OUR controller (`osc_pose.mojo`) on our scene

`recorded - theirs` is the MuJoCo-version gap under their own model (the
noise floor nobody can beat without their binary); `mujoco - theirs` is
the model gap (our vendored Panda + generated objects vs their merged
XML); `ours - mujoco` is the engine + transcription gap, on the SAME
model, and is the number this file exists to print. Residuals are per
policy step, in metres at the grip site and radians on the arm.

The initial state is `states[0]` (time 0.25 s — after LIBERO's five
zero-action settle steps), converted into our joint order by NAME.
"""

import argparse
import json
import math
import os
import re
import sys

import h5py
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUR_SCENE = os.path.join(ROOT, "noeira", "tasks", "scenes", "libero_goal.xml")
PACK = os.path.join(ROOT, "noeira", "tasks", "libero", "assets")
# robosuite 1.4.0's package tree (the version LIBERO pins): its `models/assets`
# holds the Panda / gripper / mount meshes the demo's merged model names, and
# `controllers/` is the source this file transcribes. `references/` is
# gitignored; put the sdist's `robosuite/` there (PyPI robosuite-1.4.0.tar.gz).
RS14 = os.environ.get(
    "ROBOSUITE_140", os.path.join(ROOT, "references", "robosuite-1.4.0", "robosuite")
)

SUBSTEPS = 25
TIMESTEP = 0.002


# ── the 1.4.0 controller, transcribed ─────────────────────────────────────


def quat2mat_f32(q_xyzw):
    """`transform_utils.quat2mat` VERBATIM, float32 cast included."""
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(q_xyzw).copy().astype(np.float32)[inds]
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def axisangle2quat(vec):
    angle = np.linalg.norm(vec)
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / angle
    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def orientation_error(desired, current):
    rc1, rc2, rc3 = current[0:3, 0], current[0:3, 1], current[0:3, 2]
    rd1, rd2, rd3 = desired[0:3, 0], desired[0:3, 1], desired[0:3, 2]
    return 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))


def opspace_matrices(mass_matrix, J_full, J_pos, J_ori):
    mass_matrix_inv = np.linalg.inv(mass_matrix)
    lambda_full_inv = J_full @ mass_matrix_inv @ J_full.T
    lambda_pos_inv = J_pos @ mass_matrix_inv @ J_pos.T
    lambda_ori_inv = J_ori @ mass_matrix_inv @ J_ori.T
    lambda_full = np.linalg.pinv(lambda_full_inv)
    lambda_pos = np.linalg.pinv(lambda_pos_inv)
    lambda_ori = np.linalg.pinv(lambda_ori_inv)
    Jbar = (mass_matrix_inv @ J_full.T) @ lambda_full
    nullspace_matrix = np.eye(J_full.shape[-1]) - Jbar @ J_full
    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


class Osc14:
    """`OperationalSpaceController` (1.4.0), fixed impedance, delta input,
    no interpolator, uncoupled — LIBERO's `osc_pose.json`."""

    def __init__(self, m, d, site, qpos_idx, qvel_idx, tmin, tmax):
        self.m, self.d = m, d
        self.site = site
        self.qpos_idx = np.array(qpos_idx)
        self.qvel_idx = np.array(qvel_idx)
        self.tmin, self.tmax = np.array(tmin), np.array(tmax)
        self.kp = np.ones(6) * 150.0
        self.kd = 2 * np.sqrt(self.kp) * 1.0
        self.output_max = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        self.output_min = -self.output_max
        self.input_max = np.ones(6)
        self.input_min = -np.ones(6)
        self.update()
        self.initial_joint = self.joint_pos.copy()
        self.goal_ori = np.array(self.ee_ori_mat)
        self.goal_pos = np.array(self.ee_pos)

    def update(self):
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        self.ee_pos = np.array(d.site_xpos[self.site])
        self.ee_ori_mat = np.array(d.site_xmat[self.site].reshape(3, 3))
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
        self.ee_pos_vel = jacp @ d.qvel
        self.ee_ori_vel = jacr @ d.qvel
        self.joint_pos = np.array(d.qpos[self.qpos_idx])
        self.joint_vel = np.array(d.qvel[self.qvel_idx])
        self.J_pos = jacp[:, self.qvel_idx]
        self.J_ori = jacr[:, self.qvel_idx]
        self.J_full = np.vstack([self.J_pos, self.J_ori])
        M = np.zeros((m.nv, m.nv))
        # MuJoCo 3.12's `mj_fullM(m, d, dst)`; 2.x took `d.qM` as the source
        mujoco.mj_fullM(m, d, M)
        self.mass_matrix = M[self.qvel_idx, :][:, self.qvel_idx]
        self.torque_compensation = np.array(d.qfrc_bias[self.qvel_idx])

    def scale_action(self, action):
        scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
        out_t = (self.output_max + self.output_min) / 2.0
        in_t = (self.input_max + self.input_min) / 2.0
        action = np.clip(action, self.input_min, self.input_max)
        return (action - in_t) * scale + out_t

    def set_goal(self, action):
        self.update()
        scaled = self.scale_action(np.array(action, dtype=float))
        bools = [0.0 if math.isclose(e, 0.0) else 1.0 for e in scaled[3:]]
        if sum(bools) > 0.0:
            rot_err = quat2mat_f32(axisangle2quat(scaled[3:]))
            self.goal_ori = np.dot(rot_err, self.ee_ori_mat)
        self.goal_pos = self.ee_pos + scaled[:3]

    def run_controller(self):
        self.update()
        position_error = self.goal_pos - self.ee_pos
        vel_pos_error = -self.ee_pos_vel
        desired_force = position_error * self.kp[0:3] + vel_pos_error * self.kd[0:3]
        ori_error = orientation_error(self.goal_ori, self.ee_ori_mat)
        vel_ori_error = -self.ee_ori_vel
        desired_torque = ori_error * self.kp[3:6] + vel_ori_error * self.kd[3:6]
        lambda_full, lambda_pos, lambda_ori, nullspace = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )
        wrench = np.concatenate([lambda_pos @ desired_force, lambda_ori @ desired_torque])
        torques = self.J_full.T @ wrench + self.torque_compensation
        joint_kp = 10.0
        joint_kv = np.sqrt(joint_kp) * 2
        pose_torques = self.mass_matrix @ (
            joint_kp * (self.initial_joint - self.joint_pos) - joint_kv * self.joint_vel
        )
        torques = torques + nullspace.T @ pose_torques
        self.torques = torques
        return np.clip(torques, self.tmin, self.tmax)


class GripperRamp:
    def __init__(self, ctrl_ranges):
        self.current = np.zeros(2)
        self.bias = 0.5 * (ctrl_ranges[:, 1] + ctrl_ranges[:, 0])
        self.weight = 0.5 * (ctrl_ranges[:, 1] - ctrl_ranges[:, 0])

    def step(self, a):
        self.current = np.clip(self.current + np.array([-1.0, 1.0]) * 0.01 * np.sign(a), -1.0, 1.0)
        return self.bias + self.weight * self.current


# ── models ────────────────────────────────────────────────────────────────


def rewrite_paths(xml):
    """The merged model names the recorder's absolute paths.

    ⚠ AND IT LOST `inertiagrouprange="0 0"`. The attribute is in robosuite
    1.4.0's `base.xml` and is what the recording compiled under, but the
    string in the HDF5 is `mj_saveLastXML`'s output, which does not write it
    (its compiler tag is `angle meshdir autolimits`). Reloaded as saved, the
    `.msh` visual meshes (group 1, `density`) weigh in: the bowl 0.046 kg
    instead of 0.0056, the plate 0.030 instead of 0.0115, the table 127 kg
    instead of 60. Restored here so the reference leg compiles the model the
    recorder ran, not the one the saver wrote.
    """
    xml = xml.replace("/Users/yifengz/workspace/robosuite-master/robosuite", RS14)
    xml = xml.replace("/Users/yifengz/workspace/libero-dev/chiliocosm/assets", PACK)
    xml = xml.replace('<compiler angle="radian" meshdir="meshes/" autolimits="true"/>',
                      '<compiler angle="radian" meshdir="meshes/" autolimits="true" inertiagrouprange="0 0"/>', 1)
    if 'inertiagrouprange="0 0"' not in xml:
        sys.exit("could not restore inertiagrouprange on the merged model's compiler tag")
    left = re.findall(r'file="(/Users/yifengz[^"]+)"', xml)
    if left:
        sys.exit(f"unrewritten asset paths remain: {left[:3]}")
    return xml


def load_their_model(xml, out_dir):
    path = os.path.join(out_dir, "their_model.xml")
    with open(path, "w") as f:
        f.write(rewrite_paths(xml))
    return mujoco.MjModel.from_xml_path(path)


def names(m, kind, n):
    return [mujoco.mj_id2name(m, kind, i) for i in range(n)]


def robot_refs(m, prefix, gprefix):
    """Arm joints, gripper joints, grip site, actuators, torque limits."""
    jn = names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt)
    arm = [jn.index(f"{prefix}joint{i}") for i in range(1, 8)]
    grip = [jn.index(f"{gprefix}finger_joint1"), jn.index(f"{gprefix}finger_joint2")]
    qpos_idx = [int(m.jnt_qposadr[j]) for j in arm]
    qvel_idx = [int(m.jnt_dofadr[j]) for j in arm]
    gq = [int(m.jnt_qposadr[j]) for j in grip]
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{gprefix}grip_site")
    an = names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu)
    torq = [an.index(f"{prefix}torq_j{i}") for i in range(1, 8)]
    gact = [an.index(f"{gprefix}gripper_finger_joint1"), an.index(f"{gprefix}gripper_finger_joint2")]
    tmin = m.actuator_ctrlrange[torq, 0]
    tmax = m.actuator_ctrlrange[torq, 1]
    return dict(qpos_idx=qpos_idx, qvel_idx=qvel_idx, gq=gq, site=site, torq=torq,
                gact=gact, tmin=tmin, tmax=tmax, gripper_ranges=m.actuator_ctrlrange[gact])


def run_leg(m, qpos0, qvel0, actions, refs, trace=0):
    d = mujoco.MjData(m)
    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    mujoco.mj_forward(m, d)
    osc = Osc14(m, d, refs["site"], refs["qpos_idx"], refs["qvel_idx"], refs["tmin"], refs["tmax"])
    grip = GripperRamp(refs["gripper_ranges"])
    out = []
    for t in range(len(actions)):
        a = actions[t]
        for s in range(SUBSTEPS):
            mujoco.mj_forward(m, d)
            if s == 0:
                osc.set_goal(a[:6])
            tau = osc.run_controller()
            g = grip.step(a[6])
            sub = t * SUBSTEPS + s
            if sub < trace:
                f = lambda v: " ".join(repr(float(x)) for x in np.atleast_1d(v))
                print(f"TRACE {sub} tau {f(tau)} grip {f(g)} q {f(osc.joint_pos)} qd {f(osc.joint_vel)}"
                      f" fq {f(d.qpos[refs['gq']])} ee {f(osc.ee_pos)} goal {f(osc.goal_pos)}"
                      f" bias0 {f(osc.torque_compensation[:2])} M00 {f([osc.mass_matrix[0,0], osc.mass_matrix[1,1]])} ncon {d.ncon}")
            d.ctrl[:] = 0.0
            d.ctrl[refs["torq"]] = tau
            d.ctrl[refs["gact"]] = g
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        q = np.array(d.qpos)
        p = np.array(d.site_xpos[refs["site"]])
        R = np.array(d.site_xmat[refs["site"]].reshape(3, 3))
        out.append((q, p, R, np.array(d.qpos[refs["gq"]])))
    return out


def quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", action="store_true", help="also read <out>/demo_<i>_ours.txt")
    a = ap.parse_args()
    out_dir = a.out or os.path.dirname(os.path.abspath(a.hdf5))
    os.makedirs(out_dir, exist_ok=True)

    f = h5py.File(a.hdf5, "r")
    g = f["data"][f"demo_{a.demo}"]
    actions = np.array(g["actions"])
    states = np.array(g["states"])
    robot_states = np.array(g["robot_states"])
    xml = g.attrs["model_file"]
    T = len(actions)
    print(f"demo_{a.demo}: {T} steps, states {states.shape}, task '{json.loads(f['data'].attrs['problem_info'])['language_instruction']}'")

    # ── their model, ours ────────────────────────────────────────────────
    mt = load_their_model(xml, out_dir)
    mo = mujoco.MjModel.from_xml_path(OUR_SCENE)
    assert mt.nq == mo.nq and mt.nv == mo.nv, (mt.nq, mo.nq, mt.nv, mo.nv)
    assert states.shape[1] == 1 + mt.nq + mt.nv
    their_j = names(mt, mujoco.mjtObj.mjOBJ_JOINT, mt.njnt)
    our_j = names(mo, mujoco.mjtObj.mjOBJ_JOINT, mo.njnt)

    def to_ours(name):
        if name.startswith("robot0_"):
            return "robot_" + name[len("robot0_"):]
        if name.startswith("gripper0_"):
            return "robot_" + name[len("gripper0_"):]
        return name

    qpos_t = states[0][1:1 + mt.nq]
    qvel_t = states[0][1 + mt.nq:]
    qpos_o = np.zeros(mo.nq)
    qvel_o = np.zeros(mo.nv)
    for j, nm in enumerate(their_j):
        oj = our_j.index(to_ours(nm))
        nqj = int(mo.jnt_qposadr[oj + 1] - mo.jnt_qposadr[oj]) if oj + 1 < mo.njnt else mo.nq - int(mo.jnt_qposadr[oj])
        nvj = int(mo.jnt_dofadr[oj + 1] - mo.jnt_dofadr[oj]) if oj + 1 < mo.njnt else mo.nv - int(mo.jnt_dofadr[oj])
        ta, oa = int(mt.jnt_qposadr[j]), int(mo.jnt_qposadr[oj])
        tv, ov = int(mt.jnt_dofadr[j]), int(mo.jnt_dofadr[oj])
        qpos_o[oa:oa + nqj] = qpos_t[ta:ta + nqj]
        qvel_o[ov:ov + nvj] = qvel_t[tv:tv + nvj]
    print(f"  joint order: theirs {their_j[:3]}..., ours {our_j[:3]}... — converted by name")

    # ── the fixtures, where THIS demo had them ───────────────────────────
    # LIBERO draws each fixture's xy inside its region rect at every reset
    # (`MultiRegionRandomSampler`), a +-1 cm jitter that is a MODEL field
    # (`body_pos`), not part of `states`. The family's static slot sits at
    # the rect centre; the demo's merged XML carries the drawn pose, and the
    # stove button is 9 mm from where our scene has it. Both legs and the
    # dump take the recorded placement, so the model gap is measured with
    # the fixtures where the recorder put them.
    their_b = names(mt, mujoco.mjtObj.mjOBJ_BODY, mt.nbody)
    our_b = names(mo, mujoco.mjtObj.mjOBJ_BODY, mo.nbody)
    fixtures = []
    for i, nm_ in enumerate(their_b):
        if not nm_.endswith("_main") or mt.body_parentid[i] != 0:
            continue
        if mt.body_jntnum[i] != 0:
            continue  # a free object: its pose is in `states`
        ob_name = nm_[:-5] + "_object"
        if ob_name not in our_b:
            sys.exit(f"fixture {nm_}: no body {ob_name} in our scene")
        j = our_b.index(ob_name)
        pos = np.array(mt.body_pos[i])
        quat = np.array(mt.body_quat[i])
        print(f"  fixture {nm_}: recorded pos {pos.round(4).tolist()} vs family {mo.body_pos[j].round(4).tolist()}")
        mo.body_pos[j] = pos
        mo.body_quat[j] = quat
        fixtures.append((ob_name, pos, quat))

    dump = os.path.join(out_dir, f"demo_{a.demo}_dump.txt")
    with open(dump, "w") as fh:
        fh.write(f"{T}\n")
        fh.write(" ".join(repr(float(x)) for x in qpos_o) + "\n")
        fh.write(" ".join(repr(float(x)) for x in qvel_o) + "\n")
        for t in range(T):
            fh.write(" ".join(repr(float(x)) for x in actions[t]) + "\n")
        for ob_name, pos, quat in fixtures:
            fh.write(f"fixture {ob_name} " + " ".join(repr(float(x)) for x in pos)
                     + " " + " ".join(repr(float(x)) for x in quat) + "\n")
    print(f"  wrote {dump}")

    # ── the two MuJoCo legs ──────────────────────────────────────────────
    rt = robot_refs(mt, "robot0_", "gripper0_")
    ro = robot_refs(mo, "robot_", "robot_")
    trace = int(os.environ.get("REPLAY_TRACE", "0"))
    theirs = run_leg(mt, qpos_t, qvel_t, actions, rt)
    ours_mj = run_leg(mo, qpos_o, qvel_o, actions, ro, trace=trace)

    # recorded, per step t >= 1: states[t] is the state BEFORE actions[t]
    rec_p = robot_states[:, 2:5]
    rec_q = robot_states[:, 5:9]
    rec_arm = states[:, 1 + np.array(rt["qpos_idx"])]

    def report(label, traj, arm_of, site_pos_of, site_rot_of):
        n = min(T - 1, len(traj))
        dp = []
        dq = []
        for t in range(n):
            q, p, R, gq = traj[t]
            dp.append(np.linalg.norm(p - site_pos_of(t + 1)))
            dq.append(np.max(np.abs(arm_of(q) - arm_of_rec(t + 1))))
        dp = np.array(dp)
        dq = np.array(dq)
        i = int(np.argmax(dp))
        print(f"  {label:<22} grip-site |dp| max {dp.max():.4e} m at step {i + 1} (first {dp[0]:.2e}, mean {dp.mean():.2e}); arm |dq| max {dq.max():.4e} rad")
        return dp, dq

    arm_of_rec = lambda t: rec_arm[t]

    print("--- against the RECORDED trajectory (their sim of 2022) ---")
    report("theirs (MuJoCo 3.12)", theirs, lambda q: q[rt["qpos_idx"]], lambda t: rec_p[t], None)
    report("mujoco (our scene)", ours_mj, lambda q: q[ro["qpos_idx"]], lambda t: rec_p[t], None)

    print("--- mujoco on our scene vs theirs on their model (the MODEL gap) ---")
    n = T
    dp = np.array([np.linalg.norm(ours_mj[t][1] - theirs[t][1]) for t in range(n)])
    dq = np.array([np.max(np.abs(ours_mj[t][0][ro["qpos_idx"]] - theirs[t][0][rt["qpos_idx"]])) for t in range(n)])
    print(f"  grip-site |dp| max {dp.max():.4e} m at step {int(np.argmax(dp)) + 1}; arm |dq| max {dq.max():.4e} rad")

    if a.compare:
        ours_path = os.path.join(out_dir, f"demo_{a.demo}_ours.txt")
        rows = [l for l in open(ours_path).read().split("\n") if l.strip()]
        ours = []
        for l in rows:
            left, right = l.split("|")
            q = np.array([float(x) for x in left.split()])
            r = [float(x) for x in right.split()]
            p = np.array(r[0:3])
            R = quat_xyzw_to_mat(r[3:7])
            ours.append((q, p, R, np.array(r[7:9])))
        n = min(len(ours), T)
        print(f"--- OURS (osc_pose.mojo + our engine) vs mujoco on the SAME scene, {n} steps ---")
        dp = np.array([np.linalg.norm(ours[t][1] - ours_mj[t][1]) for t in range(n)])
        dq = np.array([np.max(np.abs(ours[t][0] - ours_mj[t][0])) for t in range(n)])
        da = np.array([rot_angle(ours[t][2], ours_mj[t][2]) for t in range(n)])
        dg = np.array([np.max(np.abs(ours[t][3] - ours_mj[t][3])) for t in range(n)])
        print(f"  grip-site |dp| max {dp.max():.4e} m at step {int(np.argmax(dp)) + 1} (first {dp[0]:.2e}, mean {dp.mean():.2e})")
        print(f"  grip-site angle max {da.max():.4e} rad; FULL qpos |dq| max {dq.max():.4e} at step {int(np.argmax(dq)) + 1}; gripper |dq| max {dg.max():.2e}")
        print("--- OURS vs the RECORDED trajectory ---")
        dpr = np.array([np.linalg.norm(ours[t][1] - rec_p[t + 1]) for t in range(min(n, T - 1))])
        dqr = np.array([np.max(np.abs(ours[t][0][ro["qpos_idx"]] - rec_arm[t + 1])) for t in range(min(n, T - 1))])
        print(f"  grip-site |dp| max {dpr.max():.4e} m at step {int(np.argmax(dpr)) + 1} (first {dpr[0]:.2e}, mean {dpr.mean():.2e}); arm |dq| max {dqr.max():.4e} rad")
        k = len(dpr)
        np.savetxt(os.path.join(out_dir, f"demo_{a.demo}_residuals.txt"),
                   np.stack([dp[:k], da[:k], dq[:k], dpr]).T,
                   header="ours-mujoco |dp| | angle | full |dq| | ours-recorded |dp|")


if __name__ == "__main__":
    main()
