"""Bench PGS pyramidal-cone inner loop: scalar vs SIMD.

Mirrors the hot loop in `mojo_rl/physics3d/solver/pgs_solver.mojo`:200-267
(PYRAMIDAL cone path). Constructs a Hopper-sized synthetic ConstraintData
and times 100 PGS iterations of the pyramidal block.

Two variants:
  - scalar: the current production loop (per-element `for i in range(NV)`)
  - simd:   manual `ptr.load[width=W]` over the row, scalar tail

Both produce qacc + lambda_val arrays — bench validates they agree to FP
tolerance, then reports ns/iter for each.

Run:
    pixi run mojo run -I . benchmarks/benchmark_pgs_inner.mojo
"""

from std.math import sqrt
from std.memory import alloc
from std.random import seed, random_float64
from std.sys import simd_width_of
from std.time import perf_counter_ns

from mojo_rl.physics3d.constraints.constraint_data import (
    ConstraintData,
    ConstraintRow,
    CNSTR_NORMAL,
    CNSTR_PYRAMID_EDGE,
)


comptime DTYPE: DType = DType.float32
comptime NV: Int = 11  # Hopper has ~11 DOF
comptime MAX_ROWS: Int = 64
comptime NUM_NORMALS: Int = 5
comptime NUM_FRICTION: Int = 20  # 4 edges × 5 normals (pyramidal)
comptime PGS_ITERATIONS: Int = 100


# -----------------------------------------------------------------------------
# Scalar reference (verbatim port of pgs_solver.mojo:200-267 PYRAMIDAL branch)
# -----------------------------------------------------------------------------
def pgs_pyramidal_scalar(
    mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    mut qacc: List[Scalar[DTYPE]],
    num_normals: Int,
    num_friction: Int,
    friction_start: Int,
):
    for _ in range(PGS_ITERATIONS):
        for normal_r in range(num_normals):
            var a_n = Scalar[DTYPE](0)
            for i in range(NV):
                a_n += constraints.J[normal_r * NV + i] * qacc[i]
            var R_n = (
                Scalar[DTYPE](1.0) / constraints.rows[normal_r].inv_K_imp
                - constraints.rows[normal_r].K
            )
            var residual_n = (
                a_n
                + constraints.rows[normal_r].bias
                + R_n * constraints.rows[normal_r].lambda_val
            )
            var delta_n = -residual_n * constraints.rows[normal_r].inv_K_imp
            var old_lambda_n = constraints.rows[normal_r].lambda_val
            constraints.rows[normal_r].lambda_val = old_lambda_n + delta_n
            if constraints.rows[normal_r].lambda_val < Scalar[DTYPE](0):
                constraints.rows[normal_r].lambda_val = Scalar[DTYPE](0)
            var actual_n = (
                constraints.rows[normal_r].lambda_val - old_lambda_n
            )
            if actual_n != Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += (
                        constraints.MinvJT[normal_r * NV + i] * actual_n
                    )

        for r_off in range(num_friction):
            var r = friction_start + r_off
            var a_edge = Scalar[DTYPE](0)
            for i in range(NV):
                a_edge += constraints.J[r * NV + i] * qacc[i]
            var R_edge = (
                Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp
                - constraints.rows[r].K
            )
            var residual_edge = (
                a_edge
                + constraints.rows[r].bias
                + R_edge * constraints.rows[r].lambda_val
            )
            var delta_edge = -residual_edge * constraints.rows[r].inv_K_imp
            var old_lambda_edge = constraints.rows[r].lambda_val
            constraints.rows[r].lambda_val = old_lambda_edge + delta_edge
            if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                constraints.rows[r].lambda_val = Scalar[DTYPE](0)
            var actual_edge = (
                constraints.rows[r].lambda_val - old_lambda_edge
            )
            if actual_edge != Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += constraints.MinvJT[r * NV + i] * actual_edge


# -----------------------------------------------------------------------------
# SIMD variant: replace inner `for i in range(NV)` with ptr.load[width=W]
# -----------------------------------------------------------------------------
def pgs_pyramidal_simd(
    mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    mut qacc: List[Scalar[DTYPE]],
    num_normals: Int,
    num_friction: Int,
    friction_start: Int,
):
    comptime W = simd_width_of[DTYPE]()
    var J_ptr = constraints.J.unsafe_ptr()
    var MJ_ptr = constraints.MinvJT.unsafe_ptr()
    var qacc_ptr = qacc.unsafe_ptr()

    for _ in range(PGS_ITERATIONS):
        for normal_r in range(num_normals):
            # Dot product: a_n = J[r,:] · qacc
            var row_off = normal_r * NV
            var acc_v = SIMD[DTYPE, W](0)
            var i = 0
            while i + W <= NV:
                acc_v += (
                    J_ptr.load[width=W](row_off + i)
                    * qacc_ptr.load[width=W](i)
                )
                i += W
            var a_n = acc_v.reduce_add()
            while i < NV:
                a_n += J_ptr[row_off + i] * qacc_ptr[i]
                i += 1

            var R_n = (
                Scalar[DTYPE](1.0) / constraints.rows[normal_r].inv_K_imp
                - constraints.rows[normal_r].K
            )
            var residual_n = (
                a_n
                + constraints.rows[normal_r].bias
                + R_n * constraints.rows[normal_r].lambda_val
            )
            var delta_n = -residual_n * constraints.rows[normal_r].inv_K_imp
            var old_lambda_n = constraints.rows[normal_r].lambda_val
            constraints.rows[normal_r].lambda_val = old_lambda_n + delta_n
            if constraints.rows[normal_r].lambda_val < Scalar[DTYPE](0):
                constraints.rows[normal_r].lambda_val = Scalar[DTYPE](0)
            var actual_n = (
                constraints.rows[normal_r].lambda_val - old_lambda_n
            )

            # axpy: qacc += MinvJT[r,:] * actual_n
            if actual_n != Scalar[DTYPE](0):
                var s_v = SIMD[DTYPE, W](actual_n)
                var j = 0
                while j + W <= NV:
                    var q = qacc_ptr.load[width=W](j)
                    var m = MJ_ptr.load[width=W](row_off + j)
                    qacc_ptr.store(j, q + m * s_v)
                    j += W
                while j < NV:
                    qacc_ptr[j] += MJ_ptr[row_off + j] * actual_n
                    j += 1

        for r_off in range(num_friction):
            var r = friction_start + r_off
            var row_off = r * NV
            var acc_v = SIMD[DTYPE, W](0)
            var i = 0
            while i + W <= NV:
                acc_v += (
                    J_ptr.load[width=W](row_off + i)
                    * qacc_ptr.load[width=W](i)
                )
                i += W
            var a_edge = acc_v.reduce_add()
            while i < NV:
                a_edge += J_ptr[row_off + i] * qacc_ptr[i]
                i += 1

            var R_edge = (
                Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp
                - constraints.rows[r].K
            )
            var residual_edge = (
                a_edge
                + constraints.rows[r].bias
                + R_edge * constraints.rows[r].lambda_val
            )
            var delta_edge = -residual_edge * constraints.rows[r].inv_K_imp
            var old_lambda_edge = constraints.rows[r].lambda_val
            constraints.rows[r].lambda_val = old_lambda_edge + delta_edge
            if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                constraints.rows[r].lambda_val = Scalar[DTYPE](0)
            var actual_edge = (
                constraints.rows[r].lambda_val - old_lambda_edge
            )

            if actual_edge != Scalar[DTYPE](0):
                var s_v = SIMD[DTYPE, W](actual_edge)
                var j = 0
                while j + W <= NV:
                    var q = qacc_ptr.load[width=W](j)
                    var m = MJ_ptr.load[width=W](row_off + j)
                    qacc_ptr.store(j, q + m * s_v)
                    j += W
                while j < NV:
                    qacc_ptr[j] += MJ_ptr[row_off + j] * actual_edge
                    j += 1


# -----------------------------------------------------------------------------
# Setup: build a Hopper-sized synthetic ConstraintData with realistic values
# -----------------------------------------------------------------------------
def reset_problem(
    mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    mut qacc: List[Scalar[DTYPE]],
):
    """Re-randomize fields in already-allocated buffers (deterministic via seed).
    """
    constraints.num_normals = NUM_NORMALS
    constraints.num_friction = NUM_FRICTION
    constraints.num_limits = 0
    constraints.num_equality = 0
    constraints.num_rows = NUM_NORMALS + NUM_FRICTION

    for r in range(constraints.num_rows):
        for i in range(NV):
            constraints.J[r * NV + i] = Scalar[DTYPE](
                random_float64(-1.0, 1.0)
            )
            constraints.MinvJT[r * NV + i] = (
                constraints.J[r * NV + i] * Scalar[DTYPE](0.1)
            )
        var K_approx = Scalar[DTYPE](0.0)
        for i in range(NV):
            K_approx += (
                constraints.J[r * NV + i]
                * constraints.MinvJT[r * NV + i]
            )
        if K_approx < Scalar[DTYPE](1e-3):
            K_approx = Scalar[DTYPE](1.0)
        constraints.rows[r].K = K_approx
        constraints.rows[r].inv_K_imp = Scalar[DTYPE](0.5) / K_approx
        constraints.rows[r].bias = Scalar[DTYPE](
            random_float64(-0.5, 0.5)
        )
        constraints.rows[r].lambda_val = Scalar[DTYPE](0)
        if r < NUM_NORMALS:
            constraints.rows[r].constraint_type = CNSTR_NORMAL
        else:
            constraints.rows[r].constraint_type = CNSTR_PYRAMID_EDGE

    qacc.clear()
    for i in range(NV):
        qacc.append(Scalar[DTYPE](random_float64(-2.0, 2.0)))


def main() raises:
    var c_scalar = ConstraintData[DTYPE, MAX_ROWS, NV]()
    var c_simd = ConstraintData[DTYPE, MAX_ROWS, NV]()
    var q_scalar = List[Scalar[DTYPE]]()
    var q_simd = List[Scalar[DTYPE]]()

    seed(42)
    reset_problem(c_scalar, q_scalar)
    seed(42)
    reset_problem(c_simd, q_simd)

    # --- Correctness ---
    pgs_pyramidal_scalar(
        c_scalar, q_scalar, NUM_NORMALS, NUM_FRICTION, NUM_NORMALS
    )
    pgs_pyramidal_simd(
        c_simd, q_simd, NUM_NORMALS, NUM_FRICTION, NUM_NORMALS
    )

    var max_q_diff = Float64(0)
    for i in range(NV):
        var d = Float64(q_scalar[i]) - Float64(q_simd[i])
        if d < 0:
            d = -d
        if d > max_q_diff:
            max_q_diff = d
    var max_l_diff = Float64(0)
    for r in range(NUM_NORMALS + NUM_FRICTION):
        var d = Float64(c_scalar.rows[r].lambda_val) - Float64(
            c_simd.rows[r].lambda_val
        )
        if d < 0:
            d = -d
        if d > max_l_diff:
            max_l_diff = d

    print(
        "Correctness: max_qacc_diff =",
        max_q_diff,
        " max_lambda_diff =",
        max_l_diff,
    )
    print(
        "  (FP reorder noise expected; should be < ~1e-3 for converged"
        " values)"
    )

    # --- Perf: scalar ---
    var iters = 500
    seed(42)
    reset_problem(c_scalar, q_scalar)
    var t0 = perf_counter_ns()
    for _ in range(iters):
        pgs_pyramidal_scalar(
            c_scalar, q_scalar, NUM_NORMALS, NUM_FRICTION, NUM_NORMALS
        )
    var t1 = perf_counter_ns()
    var t_scalar_us = Float64(t1 - t0) / Float64(iters) / 1000.0

    # --- Perf: SIMD ---
    seed(42)
    reset_problem(c_simd, q_simd)
    t0 = perf_counter_ns()
    for _ in range(iters):
        pgs_pyramidal_simd(
            c_simd, q_simd, NUM_NORMALS, NUM_FRICTION, NUM_NORMALS
        )
    t1 = perf_counter_ns()
    var t_simd_us = Float64(t1 - t0) / Float64(iters) / 1000.0

    print(
        "\n",
        PGS_ITERATIONS,
        "iters × (",
        NUM_NORMALS,
        "normals +",
        NUM_FRICTION,
        "edges) at NV =",
        NV,
        ", dtype =",
        DTYPE,
        ", SIMD_W =",
        simd_width_of[DTYPE](),
    )
    print(
        "  scalar = ",
        Float64(Int(t_scalar_us * 100)) / 100.0,
        " us/call",
    )
    print(
        "  SIMD   = ",
        Float64(Int(t_simd_us * 100)) / 100.0,
        " us/call",
    )
    print(
        "  speedup =",
        Float64(Int(t_scalar_us / t_simd_us * 100)) / 100.0,
        "x",
    )
