"""Replay a LIBERO demo's actions through OUR Panda + OUR OSC_POSE — L4's leg.

    pixi run mojo run -I . examples/tasks/libero_demo_replay.mojo <dump.txt> <out.txt>

`tools/tasks/libero_demo_replay.py` writes `<dump.txt>` from one demo of a
LIBERO HDF5: the initial `qpos`/`qvel` converted into OUR joint order by
NAME (their order is robot, gripper, objects, fixtures; ours is robot,
gripper, then family slot order) and the `(T, 7)` actions. This runs them
at the benchmark's clocks — 20 Hz policy, 2 ms physics, 25 substeps per
action, `set_goal` on the first substep, the gripper ramp and the
controller `update` on every substep, exactly `Robot.control` — and
writes one line per policy step: the full `qpos`, the grip site's world
position and (x, y, z, w) orientation, the two finger qpos. The tool then
compares that against the file's own `states`/`robot_states` and against
the same transcription run by MuJoCo on the same model.
"""

from std.sys import argv
from std.os import getenv
from std.math import sqrt
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS

from mojo_rl.physics3d.fields import Data, Model, DynDims, DynamicsScratch
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
    spec_fields_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.kinematics.site_frame import site_world_quat_list
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.dynamics.osc_pose import (
    OscPose, OscPoseConfig, PandaGripperRamp, ARM_DOF,
)
from mojo_rl.physics3d.studio.stepping import StudioIntegEll
from mojo_rl.tasks.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS


comptime DT = DType.float64
comptime SCENE = "mojo_rl/tasks/scenes/libero_goal.xml"
comptime SUBSTEPS = 25
comptime ROBOT = "robot_"


def _floats(line: String) raises -> List[Float64]:
    var out = List[Float64]()
    var toks = line.split(" ")
    for k in range(len(toks)):
        var t = String(String(toks[k]).strip())
        if t.byte_length() > 0:
            out.append(Float64(t))
    return out^


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("replay: no '" + want + "' in the composed scene")


def main() raises:
    var a = argv()
    if len(a) < 3:
        raise Error("usage: libero_demo_replay.mojo <dump.txt> <out.txt>")
    var dump_path = String(a[1])
    var out_path = String(a[2])

    # ── the dump ─────────────────────────────────────────────────────────
    var text: String
    with open(dump_path, "r") as fh:
        text = fh.read()
    var lines = text.split("\n")
    var n_steps = Int(String(String(lines[0]).strip()))
    var init_qpos = _floats(String(lines[1]))
    var init_qvel = _floats(String(lines[2]))
    var actions = List[List[Float64]]()
    for t in range(n_steps):
        var row = _floats(String(lines[3 + t]))
        if len(row) != 7:
            raise Error("replay: action line " + String(t) + " has " + String(len(row)) + " numbers")
        actions.append(row^)
    # `fixture <our body> x y z qw qx qy qz`: where THIS demo's fixtures were
    var fix_names = List[String]()
    var fix_vals = List[List[Float64]]()
    for k in range(3 + n_steps, len(lines)):
        var l = String(String(lines[k]).strip())
        if not l.startswith("fixture "):
            continue
        var toks = l.split(" ")
        fix_names.append(String(toks[1]))
        var v = List[Float64]()
        for q in range(2, len(toks)):
            v.append(Float64(String(toks[q])))
        if len(v) != 7:
            raise Error("replay: bad fixture line: " + l)
        fix_vals.append(v^)

    # ── the model ────────────────────────────────────────────────────────
    var fmd = parse_model_runtime(SCENE)
    # ⚠ THE FIXTURE POSES ARE A MODEL FIELD, patched into the flat def
    # before the records are built. `BodyData.pos/quat` is the body's pose
    # in its parent (the world for a slot root), MuJoCo's (w, x, y, z).
    for k in range(len(fix_names)):
        var bi = -1
        for b in range(len(fmd.body_names)):
            if String(fmd.body_names[b]) == fix_names[k]:
                bi = b
        if bi <= 0:
            raise Error("replay: no body '" + fix_names[k] + "' for a fixture line")
        fmd.bodies[bi - 1].pos_x = fix_vals[k][0]
        fmd.bodies[bi - 1].pos_y = fix_vals[k][1]
        fmd.bodies[bi - 1].pos_z = fix_vals[k][2]
        fmd.bodies[bi - 1].quat_w = fix_vals[k][3]
        fmd.bodies[bi - 1].quat_x = fix_vals[k][4]
        fmd.bodies[bi - 1].quat_y = fix_vals[k][5]
        fmd.bodies[bi - 1].quat_z = fix_vals[k][6]
        print("  fixture", fix_names[k], "placed at", fix_vals[k][0], fix_vals[k][1], fix_vals[k][2])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    var scratch = DynamicsScratch[DT, DynDims, 1](dims)
    var integ = StudioIntegEll(dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nact = dims.get_nact()
    if len(init_qpos) != nq or len(init_qvel) != nv:
        raise Error(
            "replay: dump has " + String(len(init_qpos)) + " qpos / "
            + String(len(init_qvel)) + " qvel, the scene has " + String(nq)
            + " / " + String(nv)
        )

    # joint addresses, in joint order
    var qadr_all = List[Int]()
    var dadr_all = List[Int]()
    var qa = 0
    var da = 0
    for i in range(len(fmd.joints)):
        qadr_all.append(qa)
        dadr_all.append(da)
        qa += fmd.joints[i].nq
        da += fmd.joints[i].nv
    var dof = List[Int]()
    var qadr = List[Int]()
    for j in range(ARM_DOF):
        var ji = _index(fmd.joint_names, String(ROBOT) + "joint" + String(j + 1))
        dof.append(dadr_all[ji])
        qadr.append(qadr_all[ji])
    var g1 = _index(fmd.joint_names, String(ROBOT) + "finger_joint1")
    var g2 = _index(fmd.joint_names, String(ROBOT) + "finger_joint2")
    var site = _index(fmd.site_names, String(ROBOT) + "grip_site")
    var site_body = fmd.sites[site].body_id
    var act_idx = List[Int]()
    var tmin = List[Float64]()
    var tmax = List[Float64]()
    for j in range(ARM_DOF):
        var ai = _index(fmd.actuator_names, String(ROBOT) + "torq_j" + String(j + 1))
        act_idx.append(ai)
        tmin.append(fmd.actuators[ai].ctrl_min)
        tmax.append(fmd.actuators[ai].ctrl_max)
    var ga1 = _index(fmd.actuator_names, String(ROBOT) + "gripper_finger_joint1")
    var ga2 = _index(fmd.actuator_names, String(ROBOT) + "gripper_finger_joint2")
    print("replay: nq", nq, "nv", nv, "nact", nact, "| arm dofs", String(dof),
          "| grip site", site, "| steps", n_steps)

    # ── the state ────────────────────────────────────────────────────────
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](init_qpos[i])
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](init_qvel[i])

    var osc = OscPose(dof^, qadr^, site, site_body, tmin^, tmax^, OscPoseConfig())
    osc.update(d, m, scratch)
    osc.reset()
    var grip = PandaGripperRamp(
        fmd.actuators[ga1].ctrl_min, fmd.actuators[ga1].ctrl_max,
        fmd.actuators[ga2].ctrl_min, fmd.actuators[ga2].ctrl_max,
    )
    var ctrl = List[Float64](length=nact, fill=0.0)
    var act = List[Scalar[DT]](length=nact if nact > 0 else 1, fill=Scalar[DT](0))

    # REPLAY_TRACE=<n>: print the first n substeps' controller inputs and
    # outputs, one line each, for a substep-level diff against the MuJoCo leg
    var trace_n = 0
    var tr = getenv("REPLAY_TRACE")
    if tr.byte_length() > 0:
        trace_n = Int(tr)

    var out = String("")
    var sub = 0
    for t in range(n_steps):
        for s in range(SUBSTEPS):
            osc.update(d, m, scratch)
            if s == 0:
                osc.set_goal(actions[t])
            var tau = osc.run()
            var gc = grip.step(actions[t][6])
            if sub < trace_n:
                var l = String("TRACE ") + String(sub) + " tau"
                for j in range(ARM_DOF):
                    l += " " + String(tau[j])
                l += " grip " + String(gc[0]) + " " + String(gc[1]) + " q"
                for j in range(ARM_DOF):
                    l += " " + String(Float64(d.qpos.data[osc.qadr[j]]))
                l += " qd"
                for j in range(ARM_DOF):
                    l += " " + String(Float64(d.qvel.data[osc.dof[j]]))
                l += " fq " + String(Float64(d.qpos.data[qadr_all[g1]])) + " " + String(Float64(d.qpos.data[qadr_all[g2]]))
                l += " ee " + String(osc.ee_pos[0]) + " " + String(osc.ee_pos[1]) + " " + String(osc.ee_pos[2])
                l += " goal " + String(osc.goal_pos[0]) + " " + String(osc.goal_pos[1]) + " " + String(osc.goal_pos[2])
                l += " bias0 " + String(osc.bias[0]) + " " + String(osc.bias[1]) + " M00 " + String(osc.M[0]) + " " + String(osc.M[8])
                l += " ncon " + String(Int(d.meta.data[META_IDX_NUM_CONTACTS]))
                print(l)
            sub += 1
            for i in range(nact):
                ctrl[i] = 0.0
            for j in range(ARM_DOF):
                ctrl[act_idx[j]] = tau[j]
            ctrl[ga1] = gc[0]
            ctrl[ga2] = gc[1]
            for i in range(nv):
                d.qfrc.data[i] = Scalar[DT](0)
            apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
            integ.step["cpu"](d, m)
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        var line = String("")
        for i in range(nq):
            line += String(Float64(d.qpos.data[i])) + " "
        line += "| "
        for k in range(3):
            line += String(Float64(d.site_xpos.data[site * 3 + k])) + " "
        var sq = site_world_quat_list[DT](m.sites.data, d.xquat.data, site_body, site)
        line += String(Float64(sq[0])) + " " + String(Float64(sq[1])) + " "
        line += String(Float64(sq[2])) + " " + String(Float64(sq[3])) + " "
        line += String(Float64(d.qpos.data[qadr_all[g1]])) + " "
        line += String(Float64(d.qpos.data[qadr_all[g2]]))
        out += line + "\n"
        if t == 0 or t == n_steps - 1:
            print("  step", t, "grip site", Float64(d.site_xpos.data[site * 3]),
                  Float64(d.site_xpos.data[site * 3 + 1]),
                  Float64(d.site_xpos.data[site * 3 + 2]), "| tau0..2",
                  tau_str(osc.torques))
    with open(out_path, "w") as fh:
        fh.write(out)
    print("replay: wrote", out_path)


def tau_str(t: List[Float64]) -> String:
    var s = String("")
    for i in range(min(3, len(t))):
        s += String(t[i]) + " "
    return s^
