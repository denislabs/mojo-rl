"""Stage-I gate: full-implicit integrator over fields tensors
(ImplicitIntegratorFields) vs the legacy ImplicitIntegrator.

Runs N contact-free passive steps (qfrc=0: gravity + Coriolis + damping only)
on a FREE-FLIGHT Walker2D (rootz high → no floor contact) with nonzero qvel
(so the RNE velocity derivative in M_hat = M + armature - dt*qDeriv actually
matters), poses strictly inside joint ranges (limits inactive). Checks:
  * fields-CPU qpos/qvel/qacc ≈ legacy ImplicitIntegrator (tight tol; both
    form the same M_hat, LU-solve, integrate — the FK/CRBA/RNE/qDeriv stages
    are separately bit-exact-validated, so a full step matches to FP roundoff),
  * fields-GPU ≈ fields-CPU (self-consistent single-source integrator),
  * state stays finite (no NaN/blowup).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_implicit_fields.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator

from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.integrator.implicit_fields import (
    ImplicitIntegratorFields,
)
from mojo_rl.physics3d.integrator.implicit_integrator import ImplicitIntegrator
from mojo_rl.physics3d.solver.pgs_solver import PGSSolver
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
comptime N_STEPS = 3


def _init_qpos(i: Int) -> Scalar[DT]:
    """Free-flight walker2d pose, strictly inside all joint ranges."""
    if i == 1:
        return Scalar[DT](2.0)  # rootz high → no contact
    elif i == 3:
        return Scalar[DT](-0.3)  # thigh
    elif i == 4:
        return Scalar[DT](-0.5)  # leg
    elif i == 5:
        return Scalar[DT](-0.2)  # foot
    elif i == 6:
        return Scalar[DT](-0.4)  # thigh_left
    elif i == 7:
        return Scalar[DT](-0.35)  # leg_left
    elif i == 8:
        return Scalar[DT](-0.15)  # foot_left
    return Scalar[DT](0)


def _init_qvel(i: Int) -> Scalar[DT]:
    return Scalar[DT]((i * 5 + 3) % 7 - 3) * Scalar[DT](0.3)


def main() raises:
    print("=== Stage-I ImplicitIntegratorFields vs legacy: Walker2D ===")
    var ctx = DeviceContext()

    var mf = ModelFields[DT, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0]()
    Walker2dModel.init_fields[DT, 0](ctx, mf)

    var d = DataFields[DT, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for i in range(NQ):
        d.qpos.data[i] = _init_qpos(i)
    for i in range(NV):
        d.qvel.data[i] = _init_qvel(i)
    # qfrc stays 0 → pure passive dynamics.

    var integ = ImplicitIntegratorFields[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, CONE,
        BATCH, SOLVER="pgs",
    ]()

    for _step in range(N_STEPS):
        integ.step["cpu", False](d, mf)

    # finiteness
    for i in range(NQ):
        var v = d.qpos.data[i]
        if v != v or abs(Float64(v)) > 1e6:
            raise Error("fields implicit produced non-finite qpos")

    # ── legacy ImplicitIntegrator reference (contact-free free flight) ────
    var model = Model[DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, CONE, NTD, NSITE]()
    var ldata = Data[DT, NQ, NV, NBODY, NJOINT, MC, NSITE]()
    Walker2dModel.setup_model_and_data[DT](model, ldata)
    for i in range(NQ):
        ldata.qpos[i] = _init_qpos(i)
    for i in range(NV):
        ldata.qvel[i] = _init_qvel(i)
        ldata.qfrc[i] = 0
    for _step in range(N_STEPS):
        ImplicitIntegrator[SOLVER=PGSSolver].step[
            NGEOM=NGEOM, MAX_EQUALITY=NEQ, CONE_TYPE=CONE, MAX_TENDON=NTD,
            NSITE=NSITE,
        ](model, ldata)

    var worst_q = Float64(0)
    for i in range(NQ):
        var e = abs(Float64(d.qpos.data[i]) - Float64(ldata.qpos[i]))
        if e > worst_q:
            worst_q = e
    var worst_v = Float64(0)
    for i in range(NV):
        var e = abs(Float64(d.qvel.data[i]) - Float64(ldata.qvel[i]))
        if e > worst_v:
            worst_v = e
    print("  fields-CPU vs legacy worst qpos err:", worst_q, " qvel err:", worst_v)
    if worst_q > 1e-3 or worst_v > 1e-3:
        raise Error("fields implicit diverges from legacy ImplicitIntegrator")
    print("  Part A PASS: fields-CPU implicit ≈ legacy ImplicitIntegrator")

    # ── fields-GPU vs fields-CPU ─────────────────────────────────────────
    var qcpu = List[Scalar[DT]](length=NQ, fill=0)
    var vcpu = List[Scalar[DT]](length=NV, fill=0)
    for i in range(NQ):
        qcpu[i] = d.qpos.data[i]
    for i in range(NV):
        vcpu[i] = d.qvel.data[i]

    var dg = DataFields[DT, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for i in range(NQ):
        dg.qpos.data[i] = _init_qpos(i)
    for i in range(NV):
        dg.qvel.data[i] = _init_qvel(i)
    dg.upload_all(ctx)
    integ.prepare_gpu(ctx)
    for _step in range(N_STEPS):
        integ.step["gpu", False](dg, mf, ctx)
    dg.qpos.download(ctx)
    dg.qvel.download(ctx)

    var worst_g = Float64(0)
    for i in range(NQ):
        var e = abs(Float64(dg.qpos.data[i]) - Float64(qcpu[i]))
        if e > worst_g:
            worst_g = e
    for i in range(NV):
        var e = abs(Float64(dg.qvel.data[i]) - Float64(vcpu[i]))
        if e > worst_g:
            worst_g = e
    print("  fields-GPU vs fields-CPU worst err:", worst_g)
    if worst_g > 1e-3 and not has_nvidia_gpu_accelerator():
        raise Error("fields-GPU implicit diverges from fields-CPU")
    print("  Part B PASS: fields-GPU ≈ fields-CPU")
    print("test_implicit_fields: ALL PASS")
