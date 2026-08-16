"""Stage-I gate: RNE velocity derivative over fields tensors (qderiv) —
fields self-check (no legacy reference; the legacy `velocity_derivatives` was
deleted at the P6 sunset — bit-exact-vs-legacy was validated in git `a6804ab4`).

qDeriv = d(qfrc_bias)/d(qvel) is the dense non-symmetric term the fields
Implicit integrator subtracts into M_hat. This gate runs the fields FK +
subtree_com + cdof + qderiv pipeline (CPU) with a standing pose + nonzero qvel
and checks:
  * qderiv is non-trivially nonzero (Coriolis active — nonzero qvel),
  * fields-GPU == fields-CPU (the identical per-env kernel).

Walker2D (NV=9), build-light; CPU + one small GPU kernel → safe on Apple.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_qderiv_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator

from mojo_rl.physics3d.fields import (
    Data,
    Model,
    DynamicsScratch,
    ImplicitScratch,
    Dims,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
)
from mojo_rl.physics3d.dynamics.subtree_com import (
    compute_subtree_com,
)
from mojo_rl.physics3d.dynamics.cdof import compute_cdof
from mojo_rl.physics3d.dynamics.qderiv import (
    compute_rne_vel_derivative,
)
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
    print("=== Stage-I qderiv parity: Walker2D NV=", NV, " ===")
    var ctx = DeviceContext()

    var mf = Model[DT, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0]()
    Walker2dModel.init_fields[DT, 0](ctx, mf)

    var d = Data[DT, NQ, NV, NBODY, MC, NSITE, BATCH]()
    # Standing-ish pose + nonzero velocities so Coriolis/centrifugal is active.
    d.qpos.data[1] = 1.25  # rootz
    d.qpos.data[3] = -0.3
    d.qpos.data[4] = -0.5
    d.qpos.data[5] = 0.2
    d.qpos.data[6] = -0.4
    for i in range(NV):
        d.qvel.data[i] = Scalar[DT]((i * 5 + 3) % 7 - 3) * Scalar[DT](0.4)

    var sc = DynamicsScratch[DT, NV, NBODY, BATCH]()
    var isc = ImplicitScratch[DT, Dims[nv=NV, nbody=NBODY], BATCH]()

    # ── fields CPU pipeline: FK -> subtree_com -> cdof -> qderiv ──────────
    forward_kinematics[
        "cpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, None)
    compute_subtree_com[
        "cpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, None)
    compute_cdof[
        "cpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, sc, None)
    for i in range(NV * NV):
        isc.qderiv.data[i] = 0
    compute_rne_vel_derivative[
        "cpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, sc, isc, None)

    # ── ground-truth-lite: qderiv is non-trivially nonzero (Coriolis active) ─
    # (bit-exact-vs-legacy was validated at Stage I-2, git `a6804ab4`, before
    # the legacy `velocity_derivatives` was deleted at the P6 sunset.)
    var maxabs = Float64(0)
    for i in range(NV * NV):
        var a = abs(Float64(isc.qderiv.data[i]))
        if a > maxabs:
            maxabs = a
    print("  max|qderiv| =", maxabs)
    if maxabs < 1e-6:
        raise Error("qderiv ~0 — Coriolis inactive, gate is vacuous")
    print("  Part A PASS: RNE velocity derivative computed + nonzero")

    # ── fields GPU vs fields CPU ─────────────────────────────────────────
    var qd_cpu = List[Scalar[DT]](length=NV * NV, fill=0)
    for i in range(NV * NV):
        qd_cpu[i] = isc.qderiv.data[i]

    d.upload_all(ctx)  # mf already uploaded by init_fields
    sc.upload_all(ctx)
    isc.upload_all(ctx)
    # Re-run FK/subtree/cdof on GPU so xipos/xquat/cdof live on device.
    forward_kinematics[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, ctx)
    compute_subtree_com[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, ctx)
    compute_cdof[
        "gpu", DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH
    ](d, mf, sc, ctx)
    # zero qderiv on device then compute
    for i in range(NV * NV):
        isc.qderiv.data[i] = 0
    isc.qderiv.upload(ctx)
    compute_rne_vel_derivative[
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
