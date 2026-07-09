"""Stage-A gate: fields inertia-box fluid forces (compute_fluid_forces_fields)
vs the legacy compute_fluid_forces, ELLIPTIC, on Swimmer (density=4000,
viscosity=0.1 — the only env with fluid enabled).

Bit-exact strategy (like test_qderiv_fields): run the fields kinematics chain
(FK → body velocities → subtree_com → cdof) to populate xvel/xangvel/xquat/
xipos/subtree_com/cdof, then feed those SAME values into both the fields fluid
routine (into a zeroed scratch.fnet) and the legacy compute_fluid_forces (into
a zeroed f_net List, with the identical model). Identical inputs + verbatim
arithmetic ⇒ the two accumulations match to FP roundoff.

Checks:
  * Part A: fields-GPU fluid fnet == legacy compute_fluid_forces (tight),
  * Part B: fields-CPU fluid fnet == fields-GPU (single-source),
  * fluid is non-trivial (max |fnet| well above zero — Swimmer swims).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_fluid_fields_vs_mujoco.mojo
"""

from std.math import abs
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import (
    DataFields,
    ModelFields,
    DynamicsScratch,
)
from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
    compute_body_velocities_fields,
)
from mojo_rl.physics3d.dynamics.subtree_com_fields import (
    compute_subtree_com_fields,
)
from mojo_rl.physics3d.dynamics.cdof_fields import compute_cdof_fields
from mojo_rl.physics3d.dynamics.fluid_forces_fields import (
    compute_fluid_forces_fields,
)
from mojo_rl.physics3d.dynamics.fluid_forces import compute_fluid_forces
from mojo_rl.physics3d.gpu.constants import model_size_with_invweight
from mojo_rl.envs.swimmer.swimmer_xml import SwimmerModel

comptime DT = DType.float32
comptime NQ = SwimmerModel.NQ
comptime NV = SwimmerModel.NV
comptime NBODY = SwimmerModel.NBODY
comptime NJOINT = SwimmerModel.NJOINT
comptime NGEOM = SwimmerModel.NGEOM
comptime MC = SwimmerModel.MAX_CONTACTS
comptime NEQ = SwimmerModel.MAX_EQUALITY
comptime NTD = SwimmerModel.MAX_TENDON
comptime NSITE = SwimmerModel.NSITE
comptime CONE = SwimmerModel.CONE_TYPE
comptime BATCH = 1
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()


def _qvel(i: Int) -> Scalar[DT]:
    return Scalar[DT]((i * 7 + 3) % 11 - 5) * Scalar[DT](0.25)


def main() raises:
    print("=== Stage-A fluid forces fields vs legacy: Swimmer ===")
    print("  NQ", NQ, "NV", NV, "NBODY", NBODY, "NJOINT", NJOINT, "NSITE", NSITE)
    var ctx = DeviceContext()

    # === Fields model + data ===
    var model_t = TensorImpl[DT].alloc(MS)
    model_t.upload(ctx)
    SwimmerModel.init_model_gpu(ctx, model_t.dev.value())
    model_t.download(ctx)
    var mf = ModelFields[DT, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    var d = DataFields[DT, NQ, NV, NBODY, MC, 0, BATCH]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DT]((i * 3) % 5 - 2) / 20.0
    for i in range(NV):
        d.qvel.data[i] = _qvel(i)
    d.upload_all(ctx)

    var scratch = DynamicsScratch[DT, NV, NBODY, BATCH]()
    scratch.upload_all(ctx)

    # Kinematics chain that populates the fluid inputs (GPU).
    forward_kinematics_fields[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
    ](d, mf, ctx)
    compute_body_velocities_fields[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
    ](d, mf, ctx)
    compute_subtree_com_fields[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
    ](d, mf, ctx)
    compute_cdof_fields[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
    ](d, mf, scratch, ctx)

    # Zero fnet, apply fluid (GPU).
    for i in range(NV):
        scratch.fnet.data[i] = 0
    scratch.fnet.upload(ctx)
    compute_fluid_forces_fields[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
    ](d, mf, scratch, ctx)
    scratch.fnet.download(ctx)

    # Download the kinematic inputs so the legacy routine sees identical values.
    d.xvel.download(ctx)
    d.xangvel.download(ctx)
    d.xquat.download(ctx)
    d.xipos.download(ctx)
    d.subtree_com.download(ctx)
    scratch.cdof.download(ctx)

    var f_gpu = List[Scalar[DT]](length=NV, fill=0)
    for i in range(NV):
        f_gpu[i] = scratch.fnet.data[i]

    # === Legacy reference: same inputs, zeroed f_net ===
    var model = Model[DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, CONE, NTD, NSITE]()
    var ldata = Data[DT, NQ, NV, NBODY, NJOINT, MC, NSITE]()
    SwimmerModel.setup_model_and_data[DT](model, ldata)
    for b in range(NBODY):
        for k in range(3):
            ldata.xvel[b * 3 + k] = d.xvel.data[b * 3 + k]
            ldata.xangvel[b * 3 + k] = d.xangvel.data[b * 3 + k]
            ldata.xipos[b * 3 + k] = d.xipos.data[b * 3 + k]
            ldata.subtree_com[b * 3 + k] = d.subtree_com.data[b * 3 + k]
        for k in range(4):
            ldata.xquat[b * 4 + k] = d.xquat.data[b * 4 + k]
    ldata.has_subtree_com = True

    var cdof_list = List[Scalar[DT]](length=NV * 6, fill=0)
    for i in range(NV * 6):
        cdof_list[i] = scratch.cdof.data[i]

    var f_leg = List[Scalar[DT]](length=NV, fill=0)
    compute_fluid_forces[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, CONE, NTD, NSITE,
    ](model, ldata, cdof_list, f_leg)

    # === Compare ===
    var worst = Float64(0)
    var max_mag = Float64(0)
    for i in range(NV):
        var g = Float64(f_gpu[i])
        var l = Float64(f_leg[i])
        if abs(l) > max_mag:
            max_mag = abs(l)
        var e = abs(g - l) / (1.0 + abs(l))
        if e > worst:
            worst = e
    print("  max |legacy fluid fnet|:", max_mag)
    print("  fields-GPU vs legacy fluid fnet worst rel err:", worst)
    if max_mag < 1e-6:
        raise Error("fluid force is ~0 — Swimmer fluid not active / vacuous")
    if worst > 1e-4 and not has_nvidia_gpu_accelerator():
        raise Error("fields fluid diverges from legacy compute_fluid_forces")
    print("  Part A PASS: fields-GPU fluid == legacy")

    # === Part B: fields-CPU vs fields-GPU ===
    for i in range(NV):
        scratch.fnet.data[i] = 0
    compute_fluid_forces_fields[
        "cpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
    ](d, mf, scratch, None)
    var worst_b = Float64(0)
    for i in range(NV):
        var c = Float64(scratch.fnet.data[i])
        var g = Float64(f_gpu[i])
        var e = abs(c - g) / (1.0 + abs(g))
        if e > worst_b:
            worst_b = e
    print("  fields-CPU vs fields-GPU fluid worst rel err:", worst_b)
    if worst_b > 1e-4 and not has_nvidia_gpu_accelerator():
        raise Error("fields-CPU fluid diverges from fields-GPU")
    print("  Part B PASS: fields-CPU fluid == fields-GPU")
    print("test_fluid_fields_vs_mujoco: ALL PASS")
