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
    steps: Int, pose_file: String
) raises:
    var env = Phyics3dEnv[MODEL, CONFIG, DT, False]()
    _ = env.reset()
    # The env's own reset pose (the task's), written for the twin. From qpos0
    # the bricks coincide at the origin and the comparison is meaningless.
    for i in range(MODEL.NV):
        env.d.qacc_warmstart.data[i] = Scalar[DT](0)
    if pose_file != "":
        write_pose(env, pose_file)
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
            var lo = a if a < b else b
            var hi = b if a < b else a
            keys.append(lo * 1000 + hi)
        sort(keys)
        var qsum = Float64(0)
        for i in range(MODEL.NQ):
            qsum += Float64(env.d.qpos.data[i]) * Float64(i + 1)
        var s = String("STEP ") + String(n) + " ncon=" + String(ncon) + " qsum=" + String(qsum) + " pairs="
        if n == 0:
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
    if name == "reassemble3":
        run[Reassemble3Model, Reassemble3Config](steps, pose_file)
    elif name == "reassemble5":
        run[Reassemble5Model, Reassemble5Config](steps, pose_file)
    else:
        print("unknown model:", name)
