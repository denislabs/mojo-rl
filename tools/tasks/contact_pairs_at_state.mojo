"""OUR CPU NARROW PHASE ON A DUMPED DEVICE POSE — the third leg of a three-way.

    pixi run mojo run -I . tools/tasks/contact_pairs_at_state.mojo \
        libero_living_room_scene3 build/diag/lr3_dev_states.txt

`examples/tasks/libero_family_batched.mojo --dump-state PATH` writes the
DEVICE's own `qpos` at the first step where its contact COUNT differs from the
CPU leg's. This runs our CPU detector — both broadphases — on those exact
words, so the device, our CPU and MuJoCo can be compared AT ONE POSE instead of
at three drifted states. The MuJoCo leg is four lines of h5-free Python
(`mujoco.MjModel.from_xml_path`, set `qpos`, `mj_forward`, count
`d.contact[i]` by body pair).

⚠ WHY THIS EXISTS. A count mismatch between two legs can mean either engine is
wrong, or neither — by the time it shows, the two states have already drifted.
Only the same pose in all three says which. Measured on libero_living_room_
scene3 (Metal, 2026-09-16, after the box/box per-thread-array fix 8fea21fb9):

    lane 0 step 8  table x alphabet_soup   device 6  our CPU 4  MuJoCo 4
    lane 1 step 4  table x ketchup         device 2  our CPU 4  MuJoCo 4
    lane 2 step 2  table x wooden_tray     device 3  our CPU 3  MuJoCo 3

Whole-scene totals: our CPU 21/17/13 == MuJoCo 21/17/13; the device 23/15/13.
So the GPU box/box manifold keeps two EXTRA points in one case and drops two in
another, at a pose where our CPU is exact.
"""
from std.sys import argv
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
)
from mojo_rl.tasks.spec import load_family
from mojo_rl.tasks.family import scene_path

comptime H = DType.float64

def main() raises:
    var a = argv()
    var f = load_family(String("mojo_rl/tasks/families/") + String(a[1]) + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var dims = dims_from_flat(fmd, max_contacts=512, nmesh_verts=65536)
    var m = Model[H, DynDims](dims)
    build_model_runtime[H](fmd, dims, m)
    var d = Data[H, DynDims, 1](dims)
    var nq = dims.get_nq()
    var text: String
    with open(String(a[2]), "r") as fh:
        text = fh.read()
    var lines = text.split("\n")
    for li in range(len(lines)):
        var l = String(String(lines[li]).strip())
        if not l.startswith("QPOS"):
            continue
        var t = l.split(" ")
        var lane = String(t[2])
        var step = String(t[4])
        for k in range(nq):
            d.qpos.data[k] = Scalar[H](Float64(String(t[5 + k])))
        for k in range(dims.get_nv()):
            d.qvel.data[k] = Scalar[H](0)
        forward_kinematics["cpu", H, DynDims, 1](d, m)
        # BOTH broadphases, since the batch takes SAP and the O(N^2) loop is the
        # other producer of the same manifolds.
        detect_contacts_sap["cpu", H, DynDims, 1](d, m)
        var n_sap = Int(d.meta.data[META_IDX_NUM_CONTACTS])
        var out = String("lane ") + lane + " step " + step + " : cpu SAP " + String(n_sap)
        var want_a = -1
        for bi in range(len(fmd.body_names)):
            if fmd.body_names[bi] == "arena_living_room_table_col":
                want_a = bi
        var per = String("")
        var pairs = List[Int]()
        for c in range(n_sap):
            var ba = Int(d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_A])
            var bb = Int(d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_B])
            pairs.append(ba * 4096 + bb if ba <= bb else bb * 4096 + ba)
        detect_contacts["cpu", H, DynDims, 1](d, m)
        out += " | cpu N^2 " + String(Int(d.meta.data[META_IDX_NUM_CONTACTS]))
        # per-pair counts for the SAP list
        var seen = List[Int]()
        for i in range(len(pairs)):
            var dup = False
            for k in range(len(seen)):
                if seen[k] == pairs[i]:
                    dup = True
            if dup:
                continue
            seen.append(pairs[i])
            var n = 0
            for j in range(len(pairs)):
                if pairs[j] == pairs[i]:
                    n += 1
            var ba = pairs[i] // 4096
            var bb = pairs[i] - ba * 4096
            if ba == want_a or bb == want_a:
                per += "  " + fmd.body_names[ba] + " x " + fmd.body_names[bb] + " = " + String(n)
        print(out)
        print("   table pairs:" + per)
