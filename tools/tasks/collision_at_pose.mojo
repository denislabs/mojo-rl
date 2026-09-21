"""ONE POSE, BOTH TARGETS, THE WHOLE SCENE — is the GPU collider wrong there?

    # FAMILY is a comptime constant; sed it like the other family drivers
    pixi run mojo run -I . tools/tasks/collision_at_pose.mojo build/diag/lr3_dev_states.txt

Reads a state dumped by `examples/libero/libero_family_batched.mojo --dump-state`
and runs, on the SAME model def the batch uses: FK + both broadphases, on CPU
and on GPU, at that pose. No stepping, one lane.

## ⚠⚠ WHY THE STEP-WISE CONTACT COMPARISON CANNOT ANSWER THIS

`SYNC_FK_AFTER_STEP` re-runs FK and velocities after a step and NOT the
collision, so the batched env's `d.contacts` / `META_IDX_NUM_CONTACTS` describe
the state BEFORE the last integration while its `qpos` is post-step. Comparing
that list against the post-step pose compares a contact set with a pose one
substep later — which is how an earlier version of this investigation
"established" a GPU box/box defect that does not exist. The two legs of a
batch-vs-CPU run are also at genuinely different states by then, so their
counts differ for the ordinary reason.

This file removes both confounds: one pose in, both colliders run on it.

## MEASURED (Metal, libero_living_room_scene3, the lane-0 dumped pose)

    sap cpu  20 contacts | table x soup 4 points, dist -1.359e-05, -6.3e-08
    sap gpu  20 contacts | table x soup 4 points, the SAME four
    n^2 cpu / n^2 gpu     the same four
    MuJoCo (float64) at that pose: 4, dist -1.360e-05 / -8.7e-08

So at equal poses the GPU agrees with our CPU and with MuJoCo, including on the
box/box pairs whose gaps are 1e-08..2e-05 m. The |dq| these families fail their
CPU window on is trajectory divergence between a float32 batch and a float64
reference, amplified by contacts sitting at the margin — not a collider defect.
A 3-box fixture cut from this very pair agrees on all four paths too.
"""

from std.sys import argv
from max.gpu.host import DeviceContext
from noeira.physics3d.fields import Data, Model, Dims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.broadphase_sap import detect_contacts_sap
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST, CONTACT_IDX_POS_X,
)
from noeira.envs.libero.models.libero_living_room_scene3_xml import (
    LiberoLivingRoomScene3Model,
)
from noeira.envs.libero.models.libero_living_room_scene2_xml import (
    LiberoLivingRoomScene2Model,
)
from noeira.envs.libero.models.libero_kitchen_scene5_xml import (
    LiberoKitchenScene5Model,
)

comptime DTYPE = DType.float32
comptime M = LiberoLivingRoomScene3Model
"""⚠ ONE FAMILY PER BUILD, and this line is the knob:

    sed 's/^comptime M = .*/comptime M = LiberoLivingRoomScene2Model/' \\
        tools/tasks/collision_at_pose.mojo > /tmp/cap.mojo

A parametric `_run_all[M: ModelDefFromXML]` does not bind (the struct's own
parameters cannot be inferred from an alias) and `ModelDefLike` does not carry
`make_spec_fields` / `init_fields`, so the concrete alias has to be named
here.""" 
comptime MD = Dims[
    nq=M.NQ, nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM,
    nsite=M.NSITE, max_contacts=M.MAX_CONTACTS, nequality=M.MAX_EQUALITY,
    ntendon=M.MAX_TENDON, nexclude=M.NEXCLUDE, nmesh_verts=65536, npair=M.NPAIR,
    nact=M.NACT, nten=M.NTEN_F, nkey=M.NKEY,
]



# ⚠ THE TWO BODIES TO REPORT, by index in OUR body order — print every pair
# instead if they are not known: the point is the pair the driver named.
comptime TABLE = 19
comptime SOUP = 20

def _show(mut d: Data[DTYPE, MD, 1], tag: String) raises:
    var n = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    var k = 0
    var line = String(tag) + " : ncon " + String(n) + " | table x soup:"
    for c in range(n):
        var o = c * CONTACT_SIZE
        var ba = Int(d.contacts.data[o + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[o + CONTACT_IDX_BODY_B])
        if (ba == TABLE and bb == SOUP) or (ba == SOUP and bb == TABLE):
            k += 1
            line += " [dist " + String(Float64(d.contacts.data[o + CONTACT_IDX_DIST]))
            line += " x " + String(Float64(d.contacts.data[o + CONTACT_IDX_POS_X])) + "]"
    print(line, "->", k, "points")

def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error("usage: collision_at_pose.mojo <dumped-states.txt>")
    var path = String(a[1])
    var text: String
    with open(path, "r") as fh:
        text = fh.read()
    var q = List[Float64]()
    var lines = text.split("\n")
    for i in range(len(lines)):
        var l = String(String(lines[i]).strip())
        if l.startswith("QPOS"):
            var t = l.split(" ")
            for k in range(5, len(t)):
                q.append(Float64(String(t[k])))
            break
    print("qpos words:", len(q), "| nq", M.NQ)
    var ctx = DeviceContext()
    var sf = M.make_spec_fields[DTYPE]()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)

    var dc = Data[DTYPE, MD, 1]()
    M.reset_data(sf, dc)
    for k in range(M.NQ):
        dc.qpos.data[k] = Scalar[DTYPE](q[k])
    forward_kinematics["cpu"](dc, mf)
    detect_contacts_sap["cpu"](dc, mf)
    _show(dc, String("sap cpu"))
    detect_contacts["cpu"](dc, mf)
    _show(dc, String("n^2 cpu"))

    var dg = Data[DTYPE, MD, 1]()
    M.reset_data(sf, dg)
    for k in range(M.NQ):
        dg.qpos.data[k] = Scalar[DTYPE](q[k])
    dg.upload_all(ctx)
    forward_kinematics["gpu"](dg, mf, ctx)
    detect_contacts_sap["gpu"](dg, mf, ctx)
    dg.contacts.download(ctx); dg.meta.download(ctx); ctx.synchronize()
    _show(dg, String("sap gpu"))
    detect_contacts["gpu"](dg, mf, ctx)
    dg.contacts.download(ctx); dg.meta.download(ctx); ctx.synchronize()
    _show(dg, String("n^2 gpu"))

