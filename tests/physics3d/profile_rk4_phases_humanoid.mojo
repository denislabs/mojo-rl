"""Profile Euler integrator phases individually.

Breaks the monolithic step_kernel_mt into separate kernel launches for each
phase, so nsys can measure FK, CDOF, CRB, MassMatrix, LDL, RNE, etc.
individually.

Uses Humanoid configuration (NV=23, NBODY=14, STEP_THREADS=23). Same RK4 phase
profiler as the HalfCheetah variant, env swapped — used to reproduce the
compile-time OOM at large dims.

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
    rk4_extra_workspace_size,
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
    forward_kinematics_gpu_mt,
    compute_body_velocities_gpu,
    compute_body_velocities_gpu_mt,
)
from mojo_rl.physics3d.dynamics.jacobian import (
    compute_cdof_gpu,
    compute_cdof_gpu_mt,
    compute_composite_inertia_gpu,
)
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full_gpu,
    compute_mass_matrix_full_gpu_mt,
    ldl_factor_gpu,
    ldl_factor_gpu_mt,
    compute_M_inv_from_ldl_gpu,
    compute_M_inv_from_ldl_gpu_mt,
    ldl_solve_workspace_gpu,
)
from mojo_rl.physics3d.dynamics.bias_forces import (
    compute_bias_forces_rne_gpu,
)
from mojo_rl.physics3d.solver.newton_solver import NewtonSolver
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator

from mojo_rl.envs.humanoid import Humanoid
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel
from mojo_rl.envs.humanoid.humanoid_config import HumanoidConfig


def main() raises:
    seed(42)

    comptime dtype = DType.float32
    comptime BATCH = 256
    comptime N_STEPS = 500  # Enough for profiling

    # Humanoid dimensions (ModelDefFromXML)
    comptime NQ = HumanoidModel.NQ  # 24
    comptime NV = HumanoidModel.NV  # 23
    comptime NBODY = HumanoidModel.NBODY  # 14
    comptime NJOINT = HumanoidModel.NJOINT  # 18
    comptime MAX_CONTACTS = HumanoidModel.MAX_CONTACTS  # 50
    comptime NGEOM = HumanoidModel.NGEOM  # 18
    comptime NSITE = HumanoidModel.NSITE
    comptime MAX_EQUALITY = HumanoidModel.MAX_EQUALITY
    comptime MAX_TENDON = HumanoidModel.MAX_TENDON
    comptime NEXCLUDE = HumanoidModel.nexclude

    # Sizes must match Phyics3dEnv (Humanoid has sites + tendons).
    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
    comptime MODEL_SIZE = model_size_with_invweight[
        NBODY, NJOINT, NV, NGEOM, MAX_EQUALITY, MAX_TENDON, NSITE, NEXCLUDE
    ]()
    comptime WS_SIZE = integrator_workspace_size[
        NV, NBODY
    ]() + NV * NV + NewtonSolver.solver_workspace_size[
        NV, MAX_CONTACTS
    ]() + rk4_extra_workspace_size[NQ, NV]()

    comptime STEP_THREADS = NV  # 10
    comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
    comptime STEP_ENV_TPB = TPB // STEP_THREADS
    comptime STEP_ENV_BLOCKS = (BATCH + STEP_ENV_TPB - 1) // STEP_ENV_TPB

    print("=" * 60)
    print("RK4 Integrator Phase Profiling — Humanoid (NV=23)")
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
    # Exact host-side buffer footprint (device VRAM these will occupy).
    print(
        "Buffers: state "
        + String(Float64(BATCH * STATE_SIZE * 4) / 1.0e6)
        + " MB + model "
        + String(Float64(MODEL_SIZE * 4) / 1.0e6)
        + " MB + workspace "
        + String(Float64(BATCH * WS_SIZE * 4) / 1.0e6)
        + " MB"
    )
    print()

    with DeviceContext() as ctx:
        var (free0, total0) = ctx.get_memory_info()
        print(
            "VRAM @ ctx open:     free "
            + String(Float64(free0) / 1.0e9)
            + " / "
            + String(Float64(total0) / 1.0e9)
            + " GB"
        )

        # Allocate buffers
        var state_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
        var model_buf = ctx.enqueue_create_buffer[dtype](MODEL_SIZE)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](BATCH * WS_SIZE)

        # Initialize model (physics params, body masses, joint axes, etc.)
        HumanoidModel.init_model_gpu(ctx, model_buf)

        # Reset state (sets qpos to home position, zeros qvel)
        Humanoid[dtype].reset_kernel_gpu[BATCH, STATE_SIZE](ctx, state_buf)

        # Zero workspace
        ctx.enqueue_memset(workspace_buf, 0)
        ctx.synchronize()

        var (free1, total1) = ctx.get_memory_info()
        print(
            "VRAM after buffers:  free "
            + String(Float64(free1) / 1.0e9)
            + " GB  (buffers+init used "
            + String(Float64(free0 - free1) / 1.0e6)
            + " MB)"
        )

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
        def fk_kernel(
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

        # Phase 1b: FK level-parallel (RK4_PARALLEL_FK path). Cooperative across
        # STEP_THREADS with internal per-level barriers → 2D launch, called
        # unconditionally, valid_env-guarded. Head-to-head vs serial FK.
        @always_inline
        def fk_mt_kernel(
            s: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            m: LayoutTensor[
                dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            var tid = Int(thread_idx.y)
            var valid_env = env < BATCH
            forward_kinematics_gpu_mt[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
            ](env, tid, STEP_THREADS, valid_env, s, m)

        # Phase 2: Body Velocities (serial)
        @always_inline
        def vel_kernel(
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

        # Phase 2b: Body velocities level-parallel (MT) — head-to-head
        @always_inline
        def vel_mt_kernel(
            s: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            m: LayoutTensor[
                dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            var tid = Int(thread_idx.y)
            var valid_env = env < BATCH
            compute_body_velocities_gpu_mt[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
            ](env, tid, STEP_THREADS, valid_env, s, m)

        # Phase 3: CDOF (serial)
        @always_inline
        def cdof_kernel(
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

        # Phase 3b: CDOF flat-parallel (MT) — head-to-head
        @always_inline
        def cdof_mt_kernel(
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
            var valid_env = env < BATCH
            compute_cdof_gpu_mt[
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
            ](env, tid, STEP_THREADS, valid_env, s, m, w)

        # Phase 4: Composite Rigid Body Inertia (serial)
        @always_inline
        def crb_kernel(
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
        def mass_matrix_kernel(
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

        # Phase 6: LDL factorization (MT — production path, RK4_PARALLEL_LDL).
        # Cooperative across STEP_THREADS with internal per-column barriers, so
        # it must be launched 2D (envs × threads) and called UNCONDITIONALLY
        # (all threads reach the barriers); valid_env guards the writes.
        @always_inline
        def ldl_kernel(
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            var tid = Int(thread_idx.y)
            var valid_env = env < BATCH
            ldl_factor_gpu_mt[dtype, NV, NBODY, BATCH, WS_SIZE](
                env, tid, STEP_THREADS, valid_env, w
            )

        # Phase 7: M_inv from LDL (MT — production path, RK4_PARALLEL_MINV).
        # Columns are independent (no internal barriers); the prior kernel
        # boundary guarantees the LDL factors are ready.
        @always_inline
        def minv_kernel(
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            var tid = Int(thread_idx.y)
            if env >= BATCH:
                return
            compute_M_inv_from_ldl_gpu_mt[dtype, NV, NBODY, BATCH, WS_SIZE](
                env, tid, STEP_THREADS, w
            )

        # Phase 8: Bias Forces RNE (serial)
        @always_inline
        def rne_kernel(
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
        def ldl_solve_kernel(
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            ldl_solve_workspace_gpu[dtype, NV, NBODY, BATCH, WS_SIZE](env, w)

        # ── Warmup: run the individual phase kernels (dependency order). We do
        # NOT launch the full RK4 step_gpu: the fused rk4_stage_kernel inlines
        # every phase's per-thread InlineArrays into one mega-kernel, giving a
        # huge per-thread stack frame; at BATCH=256 its local-memory reservation
        # (frame × launched threads) exceeds device VRAM → CUDA OOM at launch
        # (rk4_integrator.mojo:2317). The per-phase split does not need it.
        print("Warming up phase kernels...")
        for _ in range(3):
            ctx.enqueue_function[fk_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[fk_mt_kernel](
                state,
                model,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
            ctx.enqueue_function[vel_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[vel_mt_kernel](
                state,
                model,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
            ctx.enqueue_function[cdof_kernel](
                state, model, workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[cdof_mt_kernel](
                state,
                model,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
            ctx.enqueue_function[crb_kernel](
                state, model, workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[mass_matrix_kernel](
                state,
                model,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
            ctx.enqueue_function[ldl_kernel](
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
            ctx.enqueue_function[minv_kernel](
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
            ctx.enqueue_function[rne_kernel](
                state, model, workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[ldl_solve_kernel](
                workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
        ctx.synchronize()
        var (free2, total2) = ctx.get_memory_info()
        print("Warmup done!")
        print(
            "VRAM after warmup:   free "
            + String(Float64(free2) / 1.0e9)
            + " GB  (phase kernels reserved "
            + String(Float64(free1 - free2) / 1.0e6)
            + " MB)"
        )
        print()

        # ── Confirmation: full RK4 step with CONE_TYPE=PYRAMIDAL routes to the
        # blocked (shared-memory) solver instead of the serial solve_gpu whose
        # huge per-thread InlineArrays (Je=ME*V_SIZE etc.) OOM the device. With
        # PYRAMIDAL this should run and reserve only a small frame. (Omitting
        # CONE_TYPE → ELLIPTIC default → serial solver → CUDA OOM at BATCH=256.)
        print("Full RK4 step (CONE_TYPE=PYRAMIDAL, blocked solver)...")
        for _ in range(10):
            RK4Integrator[SOLVER=NewtonSolver].step_gpu[
                dtype,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                BATCH,
                NGEOM,
                CONE_TYPE=HumanoidModel.CONE_TYPE,
                STEP_THREADS=NV,
            ](ctx, state_buf, model_buf, workspace_buf)
        ctx.synchronize()
        var (free3, total3) = ctx.get_memory_info()
        print(
            "VRAM after full step: free "
            + String(Float64(free3) / 1.0e9)
            + " GB  (blocked step_gpu reserved "
            + String(Float64(free2 - free3) / 1.0e6)
            + " MB)  <- if this prints, no OOM"
        )
        print()

        # ── Profile each phase separately ──
        print("Profiling individual phases (" + String(N_STEPS) + " steps)...")
        print("-" * 60)

        # Phase 1: FK
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[fk_kernel](
                state,
                model,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var fk_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("1. FK (forward kinematics):  " + String(fk_us)[byte=:8] + " μs")

        # Phase 1b: FK level-parallel (MT) — head-to-head vs serial FK
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[fk_mt_kernel](
                state,
                model,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var fk_mt_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print(
            "1b. FK (MT level-parallel):  "
            + String(fk_mt_us)[byte=:8]
            + " μs  (serial FK = "
            + String(fk_us)[byte=:8]
            + ")"
        )

        # Phase 2: Body Velocities
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[vel_kernel](
                state,
                model,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var vel_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("2. Body velocities:          " + String(vel_us)[byte=:8] + " μs")

        # Phase 2b: Body velocities level-parallel (MT)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[vel_mt_kernel](
                state,
                model,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var vel_mt_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print(
            "2b. Body velocities (MT):    "
            + String(vel_mt_us)[byte=:8]
            + " μs  (serial = "
            + String(vel_us)[byte=:8]
            + ")"
        )

        # Phase 3: CDOF
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[cdof_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var cdof_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print(
            "3. CDOF (motion axes):       " + String(cdof_us)[byte=:8] + " μs"
        )

        # Phase 3b: CDOF flat-parallel (MT)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[cdof_mt_kernel](
                state,
                model,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var cdof_mt_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print(
            "3b. CDOF (MT):               "
            + String(cdof_mt_us)[byte=:8]
            + " μs  (serial = "
            + String(cdof_us)[byte=:8]
            + ")"
        )

        # Phase 4: CRB
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[crb_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var crb_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("4. CRB (composite inertia):  " + String(crb_us)[byte=:8] + " μs")

        # Phase 5: Mass Matrix (multi-threaded)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[mass_matrix_kernel](
                state,
                model,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var mm_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("5. Mass matrix (MT):         " + String(mm_us)[byte=:8] + " μs")

        # Phase 6: LDL (MT — production path)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[ldl_kernel](
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var ldl_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("6. LDL factorization (MT):   " + String(ldl_us)[byte=:8] + " μs")

        # Phase 7: M_inv (MT — production path)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[minv_kernel](
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var minv_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print(
            "7. M_inv from LDL (MT):      " + String(minv_us)[byte=:8] + " μs"
        )

        # Phase 8: RNE
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[rne_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var rne_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("8. RNE (bias forces):        " + String(rne_us)[byte=:8] + " μs")

        # Phase 9: LDL Solve
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[ldl_solve_kernel](
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var solve_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print(
            "9. LDL solve:                " + String(solve_us)[byte=:8] + " μs"
        )

        # NOTE: the full monolithic RK4 step (step_gpu) is intentionally NOT run
        # — its fused rk4_stage_kernel has too large a per-thread frame and OOMs
        # the device at BATCH=256 (see warmup note).

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
            "Sum of phases:               "
            + String(phases_total)[byte=:8]
            + " μs"
        )
        print()

        # Percentages
        print("Phase breakdown:")
        print(
            "  FK:          "
            + String(fk_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print(
            "  Velocities:  "
            + String(vel_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print(
            "  CDOF:        "
            + String(cdof_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print(
            "  CRB:         "
            + String(crb_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print(
            "  Mass Matrix: "
            + String(mm_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print(
            "  LDL Factor:  "
            + String(ldl_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print(
            "  M_inv:       "
            + String(minv_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print(
            "  RNE:         "
            + String(rne_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print(
            "  LDL Solve:   "
            + String(solve_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print("=" * 60)
