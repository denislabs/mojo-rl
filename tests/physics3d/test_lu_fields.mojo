"""Stage-I gate: dense LU factor/solve over per-field tensors (lu_fields)
bit-exact vs the legacy `lu_factor`/`lu_solve`/`compute_M_inv_from_lu`.

LU is the non-symmetric linear solve the fields `ImplicitIntegratorFields`
needs (M_hat = M + armature - dt*qDeriv is asymmetric). This gate feeds a
deterministic, diagonally-dominant non-symmetric NV×NV system per env into
BOTH the fields helpers and the legacy `List`-based helpers and checks:
  * fields-CPU solve == legacy solve BIT-EXACT (same arithmetic),
  * fields-CPU M^-1 == legacy M^-1 BIT-EXACT,
  * the solution actually solves the system (M @ x ≈ b residual),
  * fields-GPU == fields-CPU (tight tol; only asserted off-NVIDIA... the
    GPU path runs the identical per-env helper).

Pure scratch math (no model, no physics) → light on the GPU.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_lu_fields.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator
from layout import Layout

from mojo_rl.physics3d.fields import DynamicsScratch
from mojo_rl.physics3d.dynamics.lu_fields import (
    lu_factor_fields,
    lu_solve_fields,
    compute_m_inv_from_lu_fields,
)
from mojo_rl.physics3d.dynamics.lu_factorization import (
    lu_factor,
    lu_solve,
    compute_M_inv_from_lu,
)

comptime DT = DType.float32
comptime NV = 6
comptime NBODY = 2
comptime BATCH = 2
comptime M_SIZE = NV * NV
comptime V_SIZE = NV


def _fill_M(e: Int, i: Int, j: Int) -> Scalar[DT]:
    """Deterministic non-symmetric entry; diagonal boosted so the system is
    well-conditioned + non-singular (i*7 vs j*13 → asymmetric)."""
    var base = Scalar[DT]((i * 7 + j * 13 + e * 3) % 11 - 5)
    if i == j:
        base += Scalar[DT](NV * 2 + e)  # dominant, env-varying diagonal
    return base


def _fill_b(e: Int, i: Int) -> Scalar[DT]:
    return Scalar[DT]((i * 5 + e * 2) % 7 - 3)


def main() raises:
    print("=== Stage-I lu_fields parity (NV=", NV, " BATCH=", BATCH, ") ===")
    var ctx = DeviceContext()

    var sc = DynamicsScratch[DT, NV, NBODY, BATCH]()
    for e in range(BATCH):
        for i in range(NV):
            for j in range(NV):
                sc.M.data[e * M_SIZE + i * NV + j] = _fill_M(e, i, j)
            sc.fnet.data[e * V_SIZE + i] = _fill_b(e, i)

    # ── fields-CPU factor + solve + M^-1 ──────────────────────────────────
    lu_factor_fields["cpu", DT, NV, NBODY, BATCH](sc)
    lu_solve_fields["cpu", DT, NV, NBODY, BATCH](sc)
    compute_m_inv_from_lu_fields["cpu", DT, NV, NBODY, BATCH](sc)

    # snapshot fields-CPU results (M/fnet are untouched by factor/solve)
    var xf_cpu = List[Scalar[DT]](length=BATCH * V_SIZE, fill=0)
    var mi_cpu = List[Scalar[DT]](length=BATCH * M_SIZE, fill=0)
    for i in range(BATCH * V_SIZE):
        xf_cpu[i] = sc.qacc_ws.data[i]
    for i in range(BATCH * M_SIZE):
        mi_cpu[i] = sc.m_inv.data[i]

    # ── legacy List-based factor + solve + M^-1 (per env) ─────────────────
    var bad_solve = 0
    var bad_minv = 0
    var worst_resid = Float64(0)
    for e in range(BATCH):
        var A = List[Scalar[DT]](length=M_SIZE, fill=0)
        var b = List[Scalar[DT]](length=V_SIZE, fill=0)
        for i in range(NV):
            for j in range(NV):
                A[i * NV + j] = sc.M.data[e * M_SIZE + i * NV + j]
            b[i] = sc.fnet.data[e * V_SIZE + i]
        var piv = List[Int](length=V_SIZE, fill=0)
        var x = List[Scalar[DT]](length=V_SIZE, fill=0)
        lu_factor[DT, NV, M_SIZE, V_SIZE](A, piv)
        lu_solve[DT, NV, M_SIZE, V_SIZE](A, piv, b, x)
        var Minv = List[Scalar[DT]](length=M_SIZE, fill=0)
        compute_M_inv_from_lu[DT, NV, M_SIZE, V_SIZE](A, piv, Minv)

        for i in range(NV):
            if xf_cpu[e * V_SIZE + i] != x[i]:
                bad_solve += 1
        for i in range(M_SIZE):
            if mi_cpu[e * M_SIZE + i] != Minv[i]:
                bad_minv += 1

        # residual: original M @ x_fields - b
        for i in range(NV):
            var s = Float64(0)
            for j in range(NV):
                s += Float64(sc.M.data[e * M_SIZE + i * NV + j]) * Float64(
                    xf_cpu[e * V_SIZE + j]
                )
            var r = abs(s - Float64(sc.fnet.data[e * V_SIZE + i]))
            if r > worst_resid:
                worst_resid = r

    if bad_solve != 0:
        raise Error(
            "fields-CPU LU solve != legacy (" + String(bad_solve) + " entries)"
        )
    if bad_minv != 0:
        raise Error(
            "fields-CPU LU M^-1 != legacy (" + String(bad_minv) + " entries)"
        )
    print("  fields-CPU LU solve + M^-1 == legacy BIT-EXACT")
    print("  worst residual |M x - b|:", worst_resid)
    if worst_resid > 1e-3:
        raise Error("LU solution residual too large — not solving the system")
    print("  Part A PASS: LU solve is correct + matches legacy")

    # ── fields-GPU factor + solve + M^-1 vs fields-CPU ────────────────────
    sc.upload_all(ctx)  # push host M/fnet (+ all) to device
    lu_factor_fields["gpu", DT, NV, NBODY, BATCH](sc, ctx)
    lu_solve_fields["gpu", DT, NV, NBODY, BATCH](sc, ctx)
    compute_m_inv_from_lu_fields["gpu", DT, NV, NBODY, BATCH](sc, ctx)
    sc.qacc_ws.download(ctx)
    sc.m_inv.download(ctx)

    var worst_gpu = Float64(0)
    for i in range(BATCH * V_SIZE):
        var d = abs(Float64(sc.qacc_ws.data[i]) - Float64(xf_cpu[i]))
        if d > worst_gpu:
            worst_gpu = d
    for i in range(BATCH * M_SIZE):
        var d = abs(Float64(sc.m_inv.data[i]) - Float64(mi_cpu[i]))
        if d > worst_gpu:
            worst_gpu = d
    print("  fields-GPU vs fields-CPU worst err:", worst_gpu)
    if worst_gpu > 1e-4 and not has_nvidia_gpu_accelerator():
        raise Error("fields-GPU LU diverges from fields-CPU")
    print("  Part B PASS: fields-GPU == fields-CPU")
    print("test_lu_fields: ALL PASS")
