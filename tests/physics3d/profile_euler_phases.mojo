"""Profile Euler integrator phases individually.

Breaks the monolithic step_kernel_mt into separate kernel launches for each
phase, so nsys can measure FK, CDOF, CRB, MassMatrix, LDL, RNE, etc.
individually.

Uses HalfCheetah configuration (NV=10, NBODY=8, STEP_THREADS=10).

Run:
    pixi run -e nvidia mojo run -I . tests/physics3d/profile_euler_phases.mojo

Profile:
    nsys profile -o euler_phases pixi run -e nvidia mojo run -I . tests/physics3d/profile_euler_phases.mojo
    nsys stats --report cuda_gpu_kern_sum euler_phases.nsys-rep | head -40
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.gpu.constants import (
    TPB,
    state_size,
    model_size_with_invweight,
    integrator_workspace_size,
    ws_cdof_offset,
    ws_crb_offset,
    ws_M_offset,
    ws_L_offset,
    ws_D_offset,
    ws_m_inv_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_qacc_constrained_offset,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from mojo_rl.physics3d.dynamics.jacobian import (
    compute_cdof_gpu,
    compute_composite_inertia_gpu,
)
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full_gpu,
    compute_mass_matrix_full_gpu_mt,
    ldl_factor_gpu,
    compute_M_inv_from_ldl_gpu,
    ldl_solve_workspace_gpu,
)
from mojo_rl.physics3d.dynamics.bias_forces import (
    compute_bias_forces_rne_gpu,
)
from mojo_rl.physics3d.solver.newton_solver import NewtonSolver
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator

from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
    HalfCheetahModel,
)


fn main() raises:
    seed(42)

    comptime dtype = DType.float32
    comptime BATCH = 256
    comptime N_STEPS = 500  # Enough for profiling

    # HalfCheetah dimensions
    comptime NQ = HalfCheetahModel.NQ  # 10
    comptime NV = HalfCheetahModel.NV  # 10
    comptime NBODY = HalfCheetahModel.NBODY  # 8
    comptime NJOINT = HalfCheetahModel.NJOINT  # 10
    comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS
    comptime NGEOM = HalfCheetahModel.NGEOM

    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, 0]()
    comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
    comptime WS_SIZE = integrator_workspace_size[
        NV, NBODY
    ]() + NV * NV + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()

    comptime STEP_THREADS = NV  # 10
    comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
    comptime STEP_ENV_TPB = TPB // STEP_THREADS
    comptime STEP_ENV_BLOCKS = (BATCH + STEP_ENV_TPB - 1) // STEP_ENV_TPB

    print("=" * 60)
    print("Euler Integrator Phase Profiling")
    print("=" * 60)
    print(
        "BATCH="
        + String(BATCH)
        + " NV="
        + String(NV)
        + " NBODY="
        + String(NBODY)
    )
    print("STEP_THREADS=" + String(STEP_THREADS))
    print(
        "ENV_BLOCKS="
        + String(ENV_BLOCKS)
        + " STEP_ENV_BLOCKS="
        + String(STEP_ENV_BLOCKS)
    )
    print("Steps: " + String(N_STEPS))
    print()

    with DeviceContext() as ctx:
        # Allocate buffers
        var state_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
        var model_buf = ctx.enqueue_create_buffer[dtype](MODEL_SIZE)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](BATCH * WS_SIZE)

        # Initialize model (physics params, body masses, joint axes, etc.)
        HalfCheetahModel.init_model_gpu(ctx, model_buf)

        # Reset state (sets qpos to home position, zeros qvel)
        HalfCheetah[dtype].reset_kernel_gpu[BATCH, STATE_SIZE](
            ctx, state_buf
        )

        # Zero workspace
        ctx.enqueue_memset(workspace_buf, 0)
        ctx.synchronize()

        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var model = LayoutTensor[
            dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())
        var workspace = LayoutTensor[
            dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ](workspace_buf.unsafe_ptr())

        # ── Phase kernels ──
        # Each phase is a separate kernel launch so nsys can measure individually

        # Phase 1: Forward Kinematics (serial, 1 thread per env)
        @always_inline
        fn fk_kernel(
            s: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            m: LayoutTensor[
                dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            forward_kinematics_gpu[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
                NGEOM,
            ](env, s, m)

        # Phase 2: Body Velocities (serial)
        @always_inline
        fn vel_kernel(
            s: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            m: LayoutTensor[
                dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            compute_body_velocities_gpu[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
            ](env, s, m)

        # Phase 3: CDOF (serial)
        @always_inline
        fn cdof_kernel(
            s: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            m: LayoutTensor[
                dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            compute_cdof_gpu[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
                WS_SIZE,
            ](env, s, m, w)

        # Phase 4: Composite Rigid Body Inertia (serial)
        @always_inline
        fn crb_kernel(
            s: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            m: LayoutTensor[
                dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            compute_composite_inertia_gpu[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
                WS_SIZE,
            ](env, s, m, w)

        # Phase 5: Mass Matrix (multi-threaded)
        @always_inline
        fn mass_matrix_kernel(
            s: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            m: LayoutTensor[
                dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            var tid = Int(thread_idx.y)
            if env >= BATCH:
                return
            compute_mass_matrix_full_gpu_mt[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
                WS_SIZE,
            ](env, tid, STEP_THREADS, s, m, w)

        # Phase 6: LDL Factorization (serial)
        @always_inline
        fn ldl_kernel(
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            ldl_factor_gpu[dtype, NV, NBODY, BATCH, WS_SIZE](env, w)

        # Phase 7: M_inv from LDL (serial)
        @always_inline
        fn minv_kernel(
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            compute_M_inv_from_ldl_gpu[dtype, NV, NBODY, BATCH, WS_SIZE](env, w)

        # Phase 8: Bias Forces RNE (serial)
        @always_inline
        fn rne_kernel(
            s: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            m: LayoutTensor[
                dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            compute_bias_forces_rne_gpu[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
                WS_SIZE,
            ](env, s, m, w)

        # Phase 9: LDL Solve (serial)
        @always_inline
        fn ldl_solve_kernel(
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            ldl_solve_workspace_gpu[dtype, NV, NBODY, BATCH, WS_SIZE](env, w)

        # ── Warmup: run full step a few times ──
        print("Warming up...")
        for _ in range(10):
            EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                BATCH,
                NGEOM,
                STEP_THREADS=NV,
            ](ctx, state_buf, model_buf, workspace_buf)
        ctx.synchronize()
        print("Warmup done!")
        print()

        # ── Profile each phase separately ──
        print("Profiling individual phases (" + String(N_STEPS) + " steps)...")
        print("-" * 60)

        # Phase 1: FK
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[fk_kernel, fk_kernel](
                state,
                model,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var fk_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("1. FK (forward kinematics):  " + String(fk_us)[:8] + " μs")

        # Phase 2: Body Velocities
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[vel_kernel, vel_kernel](
                state,
                model,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var vel_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("2. Body velocities:          " + String(vel_us)[:8] + " μs")

        # Phase 3: CDOF
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[cdof_kernel, cdof_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var cdof_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("3. CDOF (motion axes):       " + String(cdof_us)[:8] + " μs")

        # Phase 4: CRB
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[crb_kernel, crb_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var crb_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("4. CRB (composite inertia):  " + String(crb_us)[:8] + " μs")

        # Phase 5: Mass Matrix (multi-threaded)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[mass_matrix_kernel, mass_matrix_kernel](
                state,
                model,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var mm_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("5. Mass matrix (MT):         " + String(mm_us)[:8] + " μs")

        # Phase 6: LDL
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[ldl_kernel, ldl_kernel](
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var ldl_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("6. LDL factorization:        " + String(ldl_us)[:8] + " μs")

        # Phase 7: M_inv
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[minv_kernel, minv_kernel](
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var minv_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("7. M_inv from LDL:           " + String(minv_us)[:8] + " μs")

        # Phase 8: RNE
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[rne_kernel, rne_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var rne_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("8. RNE (bias forces):        " + String(rne_us)[:8] + " μs")

        # Phase 9: LDL Solve
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[ldl_solve_kernel, ldl_solve_kernel](
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var solve_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("9. LDL solve:                " + String(solve_us)[:8] + " μs")

        # ── Reference: full monolithic step ──
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                BATCH,
                NGEOM,
                STEP_THREADS=NV,
            ](ctx, state_buf, model_buf, workspace_buf)
        ctx.synchronize()
        t1 = perf_counter_ns()
        var full_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)

        print()
        print("=" * 60)
        var phases_total = (
            fk_us
            + vel_us
            + cdof_us
            + crb_us
            + mm_us
            + ldl_us
            + minv_us
            + rne_us
            + solve_us
        )
        print(
            "Sum of phases:               " + String(phases_total)[:8] + " μs"
        )
        print("Full step (monolithic):      " + String(full_us)[:8] + " μs")
        print("  (includes solver + finalize + contact detection)")
        print()

        # Percentages
        print("Phase breakdown:")
        print("  FK:          " + String(fk_us / phases_total * 100)[:5] + "%")
        print("  Velocities:  " + String(vel_us / phases_total * 100)[:5] + "%")
        print(
            "  CDOF:        " + String(cdof_us / phases_total * 100)[:5] + "%"
        )
        print("  CRB:         " + String(crb_us / phases_total * 100)[:5] + "%")
        print("  Mass Matrix: " + String(mm_us / phases_total * 100)[:5] + "%")
        print("  LDL Factor:  " + String(ldl_us / phases_total * 100)[:5] + "%")
        print(
            "  M_inv:       " + String(minv_us / phases_total * 100)[:5] + "%"
        )
        print("  RNE:         " + String(rne_us / phases_total * 100)[:5] + "%")
        print(
            "  LDL Solve:   " + String(solve_us / phases_total * 100)[:5] + "%"
        )
        print("=" * 60)
