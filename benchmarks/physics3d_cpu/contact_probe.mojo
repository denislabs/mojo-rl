"""Per-step contact sets of a scene, as body pairs, for a diff against MuJoCo.

    pixi run mojo build -I . -I benchmarks benchmarks/physics3d_cpu/contact_probe.mojo -o <bin>
    <bin> <reassemble3|reassemble5> [steps] [pose_out]   # float64, TASK reset pose, ctrl = 0.1

One line per step: `STEP n ncon=<k> qsum=<checksum> pairs=a-b,a-b,...` with the
pairs sorted, so `benchmarks/physics3d_contact_diff.py` can find the first
step where the multiset differs from MuJoCo's (`physics3d_contact_probe.py`).
Contacts are recorded as BODY pairs (our record has no geom ids), so the
twin maps its geoms to bodies before printing.
"""

from std.sys import argv

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X, CONTACT_IDX_NX, CONTACT_IDX_DIST, CONTACT_IDX_CONDIM,
    CONTACT_IDX_FORCE_N, CONTACT_IDX_FORCE_T1, CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FRAME_T1_X, CONTACT_IDX_FRICTION,
    MODEL_META_IDX_NOSLIP_ITERATIONS, MODEL_META_IDX_SOLVER_TOLERANCE,
    MODEL_META_IDX_CCD_TOLERANCE, MODEL_META_IDX_CCD_ITERATIONS,
    MODEL_BODY_SIZE, BODY_IDX_MASS, BODY_IDX_IXX, BODY_IDX_IPOS_X, BODY_IDX_IQUAT_X,
)
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from physics3d_cpu.harness import write_pose
from mojo_rl.envs.dm_control.manipulation_reassemble3_def import Reassemble3Model
from mojo_rl.envs.dm_control.manipulation_reassemble3_config import Reassemble3Config
from mojo_rl.envs.dm_control.manipulation_reassemble5_def import Reassemble5Model
from mojo_rl.envs.dm_control.manipulation_reassemble5_config import Reassemble5Config

comptime DT = DType.float64


def run[MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig](
    steps: Int, pose_file: String, det_a: Int, det_b: Int, det_steps: Int,
    noslip_off: Bool, tol: Float64, ccd_tol: Float64,
) raises:
    var env = Phyics3dEnv[MODEL, CONFIG, DT, False]()
    _ = env.reset()
    # The env's own reset pose (the task's), written for the twin. From qpos0
    # the bricks coincide at the origin and the comparison is meaningless.
    for i in range(MODEL.NV):
        env.d.qacc_warmstart.data[i] = Scalar[DT](0)
    if pose_file != "":
        write_pose(env, pose_file)
    # `NOSLIP=0` in the environment: run the pass with zero sweeps (the count
    # is read from the model meta; the comptime enable stays), so a step-0
    # difference can be attributed to noslip or to the primal solve.
    if noslip_off:
        env.mf.meta.data[MODEL_META_IDX_NOSLIP_ITERATIONS] = Scalar[DT](0)
    # `TOL=<x>` (8th arg): override `<option tolerance>`, to see how much of a
    # difference against the reference is the solver's residual.
    if tol > 0:
        env.mf.meta.data[MODEL_META_IDX_SOLVER_TOLERANCE] = Scalar[DT](tol)
    # `<ccd_tol>` (9th arg): override `<option ccd_tolerance>` (and give GJK
    # 500 iterations) — the six stud/flange contacts differ from the reference
    # by 0.47 um, below the 1e-6 default, so this tests whether that is all.
    if ccd_tol > 0:
        env.mf.meta.data[MODEL_META_IDX_CCD_TOLERANCE] = Scalar[DT](ccd_tol)
        env.mf.meta.data[MODEL_META_IDX_CCD_ITERATIONS] = Scalar[DT](500)
    # The mass properties of the detail pair's bodies (and of body 21), to
    # compare with MuJoCo's `body_mass` / `body_inertia` / `body_ipos`.
    for b in range(MODEL.NBODY):
        if b != det_a and b != det_b and b != 21:
            continue
        var o = b * MODEL_BODY_SIZE
        var line = String("BODYINFO ") + String(b)
        line += " mass=" + String(Float64(env.mf.bodies.data[o + BODY_IDX_MASS]))
        line += " I="
        for k in range(3):
            line += String(Float64(env.mf.bodies.data[o + BODY_IDX_IXX + k])) + ","
        line += " ipos="
        for k in range(3):
            line += String(Float64(env.mf.bodies.data[o + BODY_IDX_IPOS_X + k])) + ","
        line += " iquat_xyzw="
        for k in range(4):
            line += String(Float64(env.mf.bodies.data[o + BODY_IDX_IQUAT_X + k])) + ","
        print(line)
    var actions = List[Float64]()
    for _ in range(MODEL.ACTION_DIM):
        actions.append(0.1)
    var q0 = String("QPOS0")
    for i in range(MODEL.NQ):
        q0 += " " + String(Float64(env.d.qpos.data[i]))
    print(q0)
    for n in range(steps):
        MODEL.apply_actions[NORMALIZED=False](env.sf, env.d, actions, env.act)
        env.integ_euler.step["cpu"](env.d, env.mf)
        var ncon = Int(env.d.meta.data[META_IDX_NUM_CONTACTS])
        var keys = List[Int]()
        for c in range(ncon):
            var a = Int(env.d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_A])
            var b = Int(env.d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_B])
            # Per-contact detail for one body pair over the first steps.
            if n < det_steps and ((a == det_a and b == det_b) or (a == det_b and b == det_a)):
                var o = c * CONTACT_SIZE
                var line = String("CON step ") + String(n) + " " + String(a) + "-" + String(b)
                line += " dist=" + String(Float64(env.d.contacts.data[o + CONTACT_IDX_DIST]))
                line += " pos="
                for k in range(3):
                    line += String(Float64(env.d.contacts.data[o + CONTACT_IDX_POS_X + k])) + ","
                line += " n="
                for k in range(3):
                    line += String(Float64(env.d.contacts.data[o + CONTACT_IDX_NX + k])) + ","
                line += " dim=" + String(Int(env.d.contacts.data[o + CONTACT_IDX_CONDIM]))
                line += " f=" + String(Float64(env.d.contacts.data[o + CONTACT_IDX_FORCE_N])) + ","
                line += String(Float64(env.d.contacts.data[o + CONTACT_IDX_FORCE_T1])) + ","
                line += String(Float64(env.d.contacts.data[o + CONTACT_IDX_FORCE_T2]))
                line += " t1hint="
                for k in range(3):
                    line += String(Float64(env.d.contacts.data[o + CONTACT_IDX_FRAME_T1_X + k])) + ","
                line += " mu=" + String(Float64(env.d.contacts.data[o + CONTACT_IDX_FRICTION]))
                print(line)
            var lo = a if a < b else b
            var hi = b if a < b else a
            keys.append(lo * 1000 + hi)
        sort(keys)
        var qsum = Float64(0)
        for i in range(MODEL.NQ):
            qsum += Float64(env.d.qpos.data[i]) * Float64(i + 1)
        var s = String("STEP ") + String(n) + " ncon=" + String(ncon) + " qsum=" + String(qsum) + " pairs="
        if n == 0:
            var qa = String("QACC0")
            for i in range(MODEL.NV):
                qa += " " + String(Float64(env.integ_euler.scratch.qacc_constrained.data[i]))
            print(qa)
            var q1 = String("QPOS1")
            for i in range(MODEL.NQ):
                q1 += " " + String(Float64(env.d.qpos.data[i]))
            print(q1)
        for k in range(len(keys)):
            if k > 0:
                s += ","
            s += String(keys[k] // 1000) + "-" + String(keys[k] % 1000)
        print(s)


def main() raises:
    var args = argv()
    var name = String(args[1]) if len(args) > 1 else String("reassemble3")
    var steps = Int(atol(String(args[2]))) if len(args) > 2 else 500
    var pose_file = String(args[3]) if len(args) > 3 else String("")
    # Optional: `<a> <b> <nsteps>` — dump every contact of body pair (a, b)
    # for the first nsteps steps (`CON ...` lines).
    var det_a = Int(atol(String(args[4]))) if len(args) > 4 else -2
    var det_b = Int(atol(String(args[5]))) if len(args) > 5 else -2
    var det_steps = Int(atol(String(args[6]))) if len(args) > 6 else 0
    var noslip_off = len(args) > 7 and String(args[7]) == "noslip0"
    var tol = atof(String(args[8])) if len(args) > 8 else Float64(0)
    var ccd_tol = atof(String(args[9])) if len(args) > 9 else Float64(0)
    if name == "reassemble3":
        run[Reassemble3Model, Reassemble3Config](steps, pose_file, det_a, det_b, det_steps, noslip_off, tol, ccd_tol)
    elif name == "reassemble5":
        run[Reassemble5Model, Reassemble5Config](steps, pose_file, det_a, det_b, det_steps, noslip_off, tol, ccd_tol)
    else:
        print("unknown model:", name)
