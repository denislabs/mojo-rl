"""OUR CPU NARROW PHASE ON A DUMPED DEVICE POSE — the third leg of a three-way.

    pixi run mojo run -I . tools/tasks/contact_pairs_at_state.mojo \
        libero_living_room_scene3 build/diag/lr3_dev_states.txt

`examples/libero/libero_family_batched.mojo --dump-state PATH` writes the
DEVICE's own `qpos` at the first step where its contact COUNT differs from the
CPU leg's. This runs our CPU detector — both broadphases — on those exact
words, so the device, our CPU and MuJoCo can be compared AT ONE POSE instead of
at three drifted states. The MuJoCo leg is four lines of h5-free Python
(`mujoco.MjModel.from_xml_path`, set `qpos`, `mj_forward`, count
`d.contact[i]` by body pair).

⚠⚠ WHAT THIS CANNOT SHOW, AND AN EARLIER VERSION OF THIS HEADER CLAIMED IT DID.
The numbers below were read as "the GPU box/box manifold keeps two extra points
where our CPU is exact". THAT CONCLUSION IS WITHDRAWN. The batched env's
`d.contacts` is NOT refreshed after a step (`SYNC_FK_AFTER_STEP` re-runs FK and
velocities only), so the device's contact list describes the state BEFORE the
last integration while the dumped `qpos` is post-step — a contact set compared
against a pose one substep later. Run at EQUAL poses
(`tools/tasks/collision_at_pose.mojo`: the same scene, one lane, no stepping)
the GPU agrees with our CPU and with MuJoCo, four points and the same
distances, on exactly these pairs.

So this tool answers "what does our CPU say at this pose", which is one leg of
three; it does not by itself convict the device. Kept because the CPU-vs-MuJoCo
agreement at a dumped pose is worth one command.

MEASURED (Metal, 2026-09-16, the lane-0/1/2 poses of libero_living_room_scene3;
the device column is its LAGGED list and is why the rows differ):

    lane 0 step 8   table x alphabet_soup   our CPU 4   MuJoCo 4   (device 6)
    lane 1 step 4   table x ketchup         our CPU 4   MuJoCo 4   (device 2)
    lane 2 step 2   table x wooden_tray     our CPU 3   MuJoCo 3   (device 3)

"""
from std.sys import argv
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.collision.broadphase_sap import detect_contacts_sap
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X, CONTACT_IDX_DIST, CONTACT_IDX_NX,
)
from noeira.tasks.spec import load_family
from noeira.tasks.family import scene_path

comptime H = DType.float64

def main() raises:
    var a = argv()
    var f = load_family(String(a[1]) if String(a[1]).endswith(".family") else String("noeira/tasks/families/") + String(a[1]) + ".family")
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
        # ⚠ THE POINTS, NOT ONLY THE COUNT: two engines that agree on the pair
        # and differ on the count differ in the MANIFOLD — which vertex was
        # kept or clipped, or whether two points are the same point inside the
        # distinctness tolerance. Only the positions say which.
        if len(a) > 3 and String(a[3]) == "--points":
            detect_contacts_sap["cpu", H, DynDims, 1](d, m)
            var nn = Int(d.meta.data[META_IDX_NUM_CONTACTS])
            for c in range(nn):
                var ba = Int(d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_A])
                var bb = Int(d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_B])
                if ba != want_a and bb != want_a:
                    continue
                var o = c * CONTACT_SIZE
                print("     cpu pt", fmd.body_names[ba], "x", fmd.body_names[bb],
                      " pos", Float64(d.contacts.data[o + CONTACT_IDX_POS_X]),
                      Float64(d.contacts.data[o + CONTACT_IDX_POS_X + 1]),
                      Float64(d.contacts.data[o + CONTACT_IDX_POS_X + 2]),
                      " dist", Float64(d.contacts.data[o + CONTACT_IDX_DIST]),
                      " n", Float64(d.contacts.data[o + CONTACT_IDX_NX]),
                      Float64(d.contacts.data[o + CONTACT_IDX_NX + 1]),
                      Float64(d.contacts.data[o + CONTACT_IDX_NX + 2]))
