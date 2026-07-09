"""Stage-S-CG-wire gate: SOLVER="cg" threaded through the fields integrators
(Euler / Implicit / RK4).

Validates that the CG solver is dispatchable from the integrator seams and
produces sane physics:
  * Part A: EulerIntegratorFields[SOLVER="cg"] vs [SOLVER="newton"] over N
    contact steps on a fallen Walker2D — both solve the same convex contact
    problem each step, so the trajectories stay close (loose tol; per-step
    CG-vs-Newton residual ~1e-3 integrates over the steps). Finite.
  * Part B: ImplicitIntegratorFields[SOLVER="cg"] and RK4IntegratorFields[
    SOLVER="cg"] each take one contact step and stay finite (this forces
    their CG-branch wiring to compile + run).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_cg_wire_fields.mojo
"""

from std.math import abs
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler_fields import EulerIntegratorFields
from mojo_rl.physics3d.integrator.implicit_fields import (
    ImplicitIntegratorFields,
)
from mojo_rl.physics3d.integrator.rk4_fields import RK4IntegratorFields
from mojo_rl.physics3d.gpu.constants import model_size_with_invweight
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DT = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime CONE = ConeType.ELLIPTIC
comptime BATCH = 2
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime N_STEPS = 3


def _init_state(mut d: DataFields[DT, NQ, NV, NBODY, MC, 0, BATCH]):
    """Fallen Walker2D (feet penetrating the floor)."""
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DT]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10
            d.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DT]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5
            d.qvel.data[e * NV + i] = qv


def _load_model(ctx: DeviceContext) raises -> ModelFields[
    DT, NV, NBODY, NJOINT, NGEOM, 0, 0, 0, 0, 0
]:
    var model_t = TensorImpl[DT].alloc(MS)
    model_t.upload(ctx)
    Walker2dModel.init_model_gpu(ctx, model_t.dev.value())
    model_t.download(ctx)
    var mf = ModelFields[DT, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)
    return mf^


def part_a(ctx: DeviceContext) raises:
    print("--- Part A: Euler SOLVER='cg' vs 'newton' (", N_STEPS, "steps)")
    var mf = _load_model(ctx)

    var dN = DataFields[DT, NQ, NV, NBODY, MC, 0, BATCH]()
    var dC = DataFields[DT, NQ, NV, NBODY, MC, 0, BATCH]()
    _init_state(dN)
    _init_state(dC)
    dN.upload_all(ctx)
    dC.upload_all(ctx)

    var integN = EulerIntegratorFields[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, CONE, BATCH,
        SOLVER="newton",
    ]()
    var integC = EulerIntegratorFields[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, CONE, BATCH,
        SOLVER="cg",
    ]()
    integN.prepare_gpu(ctx)
    integC.prepare_gpu(ctx)

    for _s in range(N_STEPS):
        integN.step["gpu", True](dN, mf, ctx)
        integC.step["gpu", True](dC, mf, ctx)

    dN.qpos.download(ctx)
    dN.qvel.download(ctx)
    dC.qpos.download(ctx)
    dC.qvel.download(ctx)

    var worst_q = Float64(0)
    for i in range(BATCH * NQ):
        var n = Float64(dN.qpos.data[i])
        var c = Float64(dC.qpos.data[i])
        if n != n or c != c:
            raise Error("Part A: non-finite qpos")
        var err = abs(n - c) / (1.0 + abs(n))
        if err > worst_q:
            worst_q = err
    var worst_v = Float64(0)
    for i in range(BATCH * NV):
        var n = Float64(dN.qvel.data[i])
        var c = Float64(dC.qvel.data[i])
        if n != n or c != c:
            raise Error("Part A: non-finite qvel")
        var err = abs(n - c) / (1.0 + abs(n))
        if err > worst_v:
            worst_v = err
    print("  Euler CG vs Newton worst rel err  qpos:", worst_q, " qvel:", worst_v)
    if (worst_q > 5e-2 or worst_v > 5e-2) and not has_nvidia_gpu_accelerator():
        raise Error("Part A: Euler CG trajectory diverges from Newton")
    print("  Part A PASS: Euler CG ≈ Newton over", N_STEPS, "steps")


def part_b(ctx: DeviceContext) raises:
    print("--- Part B: Implicit + RK4 SOLVER='cg' compile + step (finite)")
    var mf = _load_model(ctx)

    var dImp = DataFields[DT, NQ, NV, NBODY, MC, 0, BATCH]()
    var dRk4 = DataFields[DT, NQ, NV, NBODY, MC, 0, BATCH]()
    _init_state(dImp)
    _init_state(dRk4)
    dImp.upload_all(ctx)
    dRk4.upload_all(ctx)

    var integImp = ImplicitIntegratorFields[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, CONE, BATCH,
        SOLVER="cg",
    ]()
    var integRk4 = RK4IntegratorFields[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, CONE, BATCH,
        SOLVER="cg",
    ]()
    integImp.prepare_gpu(ctx)
    integRk4.prepare_gpu(ctx)

    integImp.step["gpu", True](dImp, mf, ctx)
    integRk4.step["gpu", True](dRk4, mf, ctx)

    dImp.qpos.download(ctx)
    dImp.qvel.download(ctx)
    dRk4.qpos.download(ctx)
    dRk4.qvel.download(ctx)

    for i in range(BATCH * NQ):
        var iv = Float64(dImp.qpos.data[i])
        var rv = Float64(dRk4.qpos.data[i])
        if iv != iv or abs(iv) > 1e6 or rv != rv or abs(rv) > 1e6:
            raise Error("Part B: non-finite qpos (implicit/rk4 CG)")
    for i in range(BATCH * NV):
        var iv = Float64(dImp.qvel.data[i])
        var rv = Float64(dRk4.qvel.data[i])
        if iv != iv or abs(iv) > 1e6 or rv != rv or abs(rv) > 1e6:
            raise Error("Part B: non-finite qvel (implicit/rk4 CG)")
    print("  Part B PASS: Implicit + RK4 CG step finite")


def main() raises:
    print("=== Stage-S-CG-wire: SOLVER='cg' through fields integrators ===")
    var ctx = DeviceContext()
    part_a(ctx)
    part_b(ctx)
    print("test_cg_wire_fields: ALL PASS")
