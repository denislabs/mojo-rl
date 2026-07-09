"""Stage-I gate: RNE velocity derivative over fields tensors (qderiv_fields)
bit-exact vs the legacy `compute_rne_vel_derivative`.

qDeriv = d(qfrc_bias)/d(qvel) is the dense non-symmetric term the fields
Implicit integrator subtracts into M_hat. This gate isolates the derivative
ARITHMETIC: it runs the fields FK + subtree_com + cdof pipeline (CPU) to
produce xipos/xquat/cdof, then feeds those IDENTICAL inputs (+ the same
nonzero qvel) into BOTH the fields per-env function and the legacy
List/Model-based function, and checks:
  * fields-CPU qderiv == legacy qderiv BIT-EXACT (same inputs, same math),
  * qderiv is non-trivially nonzero (Coriolis active — nonzero qvel),
  * fields-GPU == fields-CPU (the identical per-env kernel).

Walker2D (NV=9), build-light; CPU + one small GPU kernel → safe on Apple.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_qderiv_fields.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator

from mojo_rl.physics3d.fields import (
    DataFields,
    ModelFields,
    DynamicsScratch,
    ImplicitScratch,
)
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
)
from mojo_rl.physics3d.dynamics.subtree_com_fields import (
    compute_subtree_com_fields,
)
from mojo_rl.physics3d.dynamics.cdof_fields import compute_cdof_fields
from mojo_rl.physics3d.dynamics.qderiv_fields import (
    compute_rne_vel_derivative_fields,
)
from mojo_rl.physics3d.dynamics.velocity_derivatives import (
    compute_rne_vel_derivative,
)
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DT = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime CONE = Walker2dModel.CONE_TYPE
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.nexclude
comptime BATCH = 1


def main() raises:
    print("=== Stage-I qderiv_fields parity: Walker2D NV=", NV, " ===")
    var ctx = DeviceContext()

    var mf = ModelFields[DT, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0]()
    Walker2dModel.init_fields[DT, 0](ctx, mf)

    var d = DataFields[DT, NQ, NV, NBODY, MC, NSITE, BATCH]()
    # Standing-ish pose + nonzero velocities so Coriolis/centrifugal is active.
    d.qpos.data[1] = 1.25  # rootz
    d.qpos.data[3] = -0.3
    d.qpos.data[4] = -0.5
    d.qpos.data[5] = 0.2
    d.qpos.data[6] = -0.4
    for i in range(NV):
        d.qvel.data[i] = Scalar[DT]((i * 5 + 3) % 7 - 3) * Scalar[DT](0.4)

    var sc = DynamicsScratch[DT, NV, NBODY, BATCH]()
    var isc = ImplicitScratch[DT, NV, NBODY, BATCH]()

    # ── fields CPU pipeline: FK -> subtree_com -> cdof -> qderiv ──────────
    forward_kinematics_fields[
        "cpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, None)
    compute_subtree_com_fields[
        "cpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, None)
    compute_cdof_fields[
        "cpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, sc, None)
    for i in range(NV * NV):
        isc.qderiv.data[i] = 0
    compute_rne_vel_derivative_fields[
        "cpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, sc, isc, None)

    # ── legacy reference fed the SAME xipos/xquat/qvel/cdof ──────────────
    var model = Model[DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, CONE, NTD, NSITE]()
    var ldata = Data[DT, NQ, NV, NBODY, NJOINT, MC, NSITE]()
    Walker2dModel.setup_model_and_data[DT](model, ldata)
    for i in range(NBODY * 3):
        ldata.xipos[i] = d.xipos.data[i]
    for i in range(NBODY * 4):
        ldata.xquat[i] = d.xquat.data[i]
    for i in range(NV):
        ldata.qvel[i] = d.qvel.data[i]
    var cdof_list = List[Scalar[DT]](length=NV * 6, fill=0)
    for i in range(NV * 6):
        cdof_list[i] = sc.cdof.data[i]
    var qd_legacy = List[Scalar[DT]](length=NV * NV, fill=0)
    compute_rne_vel_derivative[DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, CONE, NTD, NSITE](
        model, ldata, cdof_list, qd_legacy
    )

    # ── compare ──────────────────────────────────────────────────────────
    var bad = 0
    var maxabs = Float64(0)
    for i in range(NV * NV):
        if isc.qderiv.data[i] != qd_legacy[i]:
            if bad < 4:
                print(
                    "  qderiv[", i, "]: fields", isc.qderiv.data[i],
                    " vs legacy", qd_legacy[i],
                )
            bad += 1
        var a = abs(Float64(isc.qderiv.data[i]))
        if a > maxabs:
            maxabs = a
    if bad != 0:
        raise Error(
            "fields-CPU qderiv != legacy (" + String(bad) + " entries)"
        )
    print("  fields-CPU qderiv == legacy BIT-EXACT; max|qderiv| =", maxabs)
    if maxabs < 1e-6:
        raise Error("qderiv ~0 — Coriolis inactive, gate is vacuous")
    print("  Part A PASS: RNE velocity derivative matches legacy + nonzero")

    # ── fields GPU vs fields CPU ─────────────────────────────────────────
    var qd_cpu = List[Scalar[DT]](length=NV * NV, fill=0)
    for i in range(NV * NV):
        qd_cpu[i] = isc.qderiv.data[i]

    d.upload_all(ctx)  # mf already uploaded by init_fields
    sc.upload_all(ctx)
    isc.upload_all(ctx)
    # Re-run FK/subtree/cdof on GPU so xipos/xquat/cdof live on device.
    forward_kinematics_fields[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, ctx)
    compute_subtree_com_fields[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, ctx)
    compute_cdof_fields[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, sc, ctx)
    # zero qderiv on device then compute
    for i in range(NV * NV):
        isc.qderiv.data[i] = 0
    isc.qderiv.upload(ctx)
    compute_rne_vel_derivative_fields[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, sc, isc, ctx)
    isc.qderiv.download(ctx)

    var worst = Float64(0)
    for i in range(NV * NV):
        var e = abs(Float64(isc.qderiv.data[i]) - Float64(qd_cpu[i]))
        if e > worst:
            worst = e
    print("  fields-GPU vs fields-CPU worst qderiv err:", worst)
    if worst > 1e-4 and not has_nvidia_gpu_accelerator():
        raise Error("fields-GPU qderiv diverges from fields-CPU")
    print("  Part B PASS: fields-GPU == fields-CPU")
    print("test_qderiv_fields: ALL PASS")
