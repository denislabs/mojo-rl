"""Profile RK4 forward-dynamics phases individually — Humanoid (NV=23).

Humanoid counterpart of `profile_euler_phases.mojo` (HalfCheetah/Euler). Breaks
the monolithic `rk4_stage_kernel` into separate per-phase kernel launches so we
can measure each phase (FK, velocities, contacts, subtree-COM, CDOF, CRB, mass
matrix, LDL, M_inv, RNE, LDL-solve) in isolation on the *real* large-NV target.

This is the apples-to-apples per-phase split the blocked-solver doc calls for
before investing in Lever 1 (branch/level-parallel forward dynamics): it tells
us whether FK + velocities (the serial root→leaf tree walks) are a slice worth
parallelizing, vs CRB/LDL/M_inv (dense linear algebra) or RNE.

Humanoid uses RK4Integrator[NewtonSolver] with frame_skip=5, so a single
env.step runs 20 solves and 20 forward-dynamics passes (4 RK4 stages × 5
substeps). Each phase below is launched standalone with the same dims/layout the
RK4 stage kernel uses (STEP_THREADS=NV), so the per-phase µs are directly the
per-pass cost; multiply by 20 to weight a full env.step.

NOTE on multi-threaded phases: mass matrix is launched multi-threaded
(`*_gpu_mt`, the one phase already distributed). FK / velocities / CDOF / CRB /
RNE are launched single-thread-per-env here (matching how the *current* stage
kernel runs them — redundantly on all STEP_THREADS, so per-env latency = one
serial walk). LDL / M_inv are launched single-thread here to isolate the raw
serial cost (the stage kernel parallelizes them via RK4_PARALLEL_LDL/MINV).

Run:
    pixi run -e nvidia mojo run -I . tests/physics3d/profile_rk4_phases_humanoid.mojo

nsys (per-kernel):
    nsys profile -o rk4_phases_humanoid pixi run -e nvidia mojo run -I . \
        tests/physics3d/profile_rk4_phases_humanoid.mojo
    nsys stats --report cuda_gpu_kern_sum rk4_phases_humanoid.nsys-rep | head -40
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.gpu.constants import (
    TPB,
    state_size,
    model_size_with_invweight,
    integrator_workspace_size,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from mojo_rl.physics3d.dynamics.jacobian import (
    compute_cdof_gpu,
    compute_composite_inertia_gpu,
    compute_subtree_com_gpu,
)
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full_gpu_mt,
    ldl_factor_gpu,
    compute_M_inv_from_ldl_gpu,
    ldl_solve_workspace_gpu,
)
from mojo_rl.physics3d.dynamics.bias_forces import (
    compute_bias_forces_rne_gpu,
)
from mojo_rl.physics3d.collision.broadphase_sap import (
    detect_contacts_auto_gpu,
)
from mojo_rl.physics3d.solver.newton_solver import NewtonSolver
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator

from mojo_rl.envs.humanoid import Humanoid
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel
from mojo_rl.envs.humanoid.humanoid_config import HumanoidConfig


def main() raises:
    seed(42)

    comptime dtype = DType.float32
    # NOTE: the per-phase % split is BATCH-invariant (every phase scales with
    # BATCH), so a small BATCH is fine for the breakdown — and avoids the CUDA
    # OOM that BATCH=256 hits (256 full-Humanoid workspaces + the fused
    # rk4_stage_kernel's local-memory reservation). Bump back up if the GPU has
    # the headroom (check `nvidia-smi`); the relative split won't change.
    comptime BATCH = 64
    comptime N_STEPS = 200  # RK4 is heavier than Euler; fewer steps suffice

    # When False (default), the full RK4 step_gpu is NEVER launched — only the
    # individual phase kernels are. This avoids reserving the large per-kernel
    # local-memory windows of the 4 fused rk4_stage_kernel instantiations + the
    # Newton solver (the real CUDA-OOM cause; see warmup note). The per-phase
    # split (this profiler's purpose) does not need the full step. Set True only
    # to also measure the monolithic step total, and only if VRAM allows.
    comptime PROFILE_FULL_STEP = False

    # Humanoid dimensions (from HumanoidModel = ModelDefFromXML[...])
    comptime NQ = HumanoidModel.NQ  # 24
    comptime NV = HumanoidModel.NV  # 23
    comptime NBODY = HumanoidModel.NBODY  # 14
    comptime NJOINT = HumanoidModel.NJOINT  # 18
    comptime NGEOM = HumanoidModel.NGEOM  # 18
    comptime MAX_CONTACTS = HumanoidModel.MAX_CONTACTS  # 50
    comptime NSITE = HumanoidModel.NSITE
    comptime MAX_EQUALITY = HumanoidModel.MAX_EQUALITY
    comptime MAX_TENDON = HumanoidModel.MAX_TENDON
    comptime NEXCLUDE = HumanoidModel.nexclude

    # Sizes must match Phyics3dEnv exactly (Humanoid has sites + tendons, so the
    # model/state buffers carry equality/tendon/site/exclude blocks too).
    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
    comptime MODEL_SIZE = model_size_with_invweight[
        NBODY,
        NJOINT,
        NV,
        NGEOM,
        MAX_EQUALITY,
        MAX_TENDON,
        NSITE,
        NEXCLUDE,
    ]()
    # RK4 path needs the extra workspace (INTEGRATOR_WS_EXTRA = NQ + 7*NV).
    comptime WS_SIZE = integrator_workspace_size[
        NV, NBODY
    ]() + NV * NV + NewtonSolver.solver_workspace_size[
        NV, MAX_CONTACTS
    ]() + HumanoidConfig.INTEGRATOR_WS_EXTRA

    comptime STEP_THREADS = NV  # 23
    comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
    comptime STEP_ENV_TPB = TPB // STEP_THREADS
    comptime STEP_ENV_BLOCKS = (BATCH + STEP_ENV_TPB - 1) // STEP_ENV_TPB

    print("=" * 60)
    print("RK4 Forward-Dynamics Phase Profiling — Humanoid")
    print("=" * 60)
    print(
        "BATCH="
        + String(BATCH)
        + " NV="
        + String(NV)
        + " NBODY="
        + String(NBODY)
        + " MAX_CONTACTS="
        + String(MAX_CONTACTS)
    )
    print("STEP_THREADS=" + String(STEP_THREADS))
    print(
        "ENV_BLOCKS="
        + String(ENV_BLOCKS)
        + " STEP_ENV_BLOCKS="
        + String(STEP_ENV_BLOCKS)
    )
    print("Steps: " + String(N_STEPS) + " (per RK4 pass; ×20 for a full step)")
    print()

    # Computed buffer footprint (host-side, exact).
    print("Buffer sizes (per env / total):")
    print(
        "  STATE_SIZE = " + String(STATE_SIZE) + " floats  -> state_buf "
        + String(Float64(BATCH * STATE_SIZE * 4) / 1.0e6) + " MB"
    )
    print(
        "  MODEL_SIZE = " + String(MODEL_SIZE) + " floats  -> model_buf "
        + String(Float64(MODEL_SIZE * 4) / 1.0e6) + " MB"
    )
    print(
        "  WS_SIZE    = " + String(WS_SIZE) + " floats  -> workspace_buf "
        + String(Float64(BATCH * WS_SIZE * 4) / 1.0e6) + " MB"
    )
    print(
        "  TOTAL buffers = "
        + String(
            Float64((BATCH * STATE_SIZE + MODEL_SIZE + BATCH * WS_SIZE) * 4)
            / 1.0e6
        )
        + " MB"
    )
    print()

    with DeviceContext() as ctx:
        var (free0, total0) = ctx.get_memory_info()
        print(
            "VRAM @ ctx open:        free "
            + String(Float64(free0) / 1.0e9) + " / "
            + String(Float64(total0) / 1.0e9) + " GB"
        )

        var state_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
        var model_buf = ctx.enqueue_create_buffer[dtype](MODEL_SIZE)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](BATCH * WS_SIZE)

        HumanoidModel.init_model_gpu(ctx, model_buf)
        Humanoid[dtype].reset_kernel_gpu[BATCH, STATE_SIZE](ctx, state_buf)
        ctx.enqueue_memset(workspace_buf, 0)
        ctx.synchronize()

        var (free1, total1) = ctx.get_memory_info()
        print(
            "VRAM after buffers:     free "
            + String(Float64(free1) / 1.0e9) + " GB  (buffers+init used "
            + String(Float64(free0 - free1) / 1.0e6) + " MB)"
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

        # ── Phase kernels (one launch per phase) ──

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

        @always_inline
        def detect_kernel(
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
            detect_contacts_auto_gpu[
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

        @always_inline
        def com_kernel(
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
            compute_subtree_com_gpu[
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

        @always_inline
        def ldl_kernel(
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            ldl_factor_gpu[dtype, NV, NBODY, BATCH, WS_SIZE](env, w)

        @always_inline
        def minv_kernel(
            w: LayoutTensor[
                dtype, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            compute_M_inv_from_ldl_gpu[dtype, NV, NBODY, BATCH, WS_SIZE](env, w)

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

        # ── Warmup: run each phase kernel in dependency order (populates the
        # workspace + triggers each kernel's first-launch local-memory window).
        # We deliberately do NOT launch the full RK4 step_gpu here: its 4 fused
        # rk4_stage_kernel instantiations + the Newton solver reserve large
        # per-kernel local-memory backing stores (stackframe × max-resident
        # threads), which is the real VRAM consumer behind the CUDA OOM — and
        # none of those kernels is among the phases we measure. The VRAM probe
        # below shows exactly how much the *phase* kernels reserve.
        print("Warming up phase kernels (dependency order)...")
        for _ in range(3):
            ctx.enqueue_function[fk_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[vel_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[detect_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[com_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[cdof_kernel](
                state, model, workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
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
                workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[minv_kernel](
                workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[rne_kernel](
                state, model, workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
            ctx.enqueue_function[ldl_solve_kernel](
                workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
        ctx.synchronize()
        var (free2, total2) = ctx.get_memory_info()
        print(
            "VRAM after phase warmup: free "
            + String(Float64(free2) / 1.0e9)
            + " GB  (phase kernels reserved "
            + String(Float64(free1 - free2) / 1.0e6) + " MB)"
        )

        comptime if PROFILE_FULL_STEP:
            print("Warming up full RK4 step_gpu (PROFILE_FULL_STEP=True)...")
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
                    STEP_THREADS=NV,
                ](ctx, state_buf, model_buf, workspace_buf)
            ctx.synchronize()
            var (free3, total3) = ctx.get_memory_info()
            print(
                "VRAM after full step:    free "
                + String(Float64(free3) / 1.0e9)
                + " GB  (step_gpu kernels reserved "
                + String(Float64(free2 - free3) / 1.0e6) + " MB)"
            )
        print("Warmup done!")
        print()

        print(
            "Profiling individual phases (" + String(N_STEPS) + " launches)..."
        )
        print("-" * 60)

        # Phase 1: FK
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[fk_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var fk_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("1.  FK (forward kinematics): " + String(fk_us)[byte=:8] + " μs")

        # Phase 2: Body velocities
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[vel_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var vel_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("2.  Body velocities:         " + String(vel_us)[byte=:8] + " μs")

        # Phase 3: Contact detection
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[detect_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var detect_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print(
            "3.  Contact detection:       " + String(detect_us)[byte=:8] + " μs"
        )

        # Phase 3a: Subtree COM
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[com_kernel](
                state, model, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var com_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("3a. Subtree COM:             " + String(com_us)[byte=:8] + " μs")

        # Phase 4: CDOF
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
            "4.  CDOF (motion axes):      " + String(cdof_us)[byte=:8] + " μs"
        )

        # Phase 5: CRB
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
        print("5.  CRB (composite inertia): " + String(crb_us)[byte=:8] + " μs")

        # Phase 6: Mass matrix (multi-threaded)
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
        print("6.  Mass matrix (MT):        " + String(mm_us)[byte=:8] + " μs")

        # Phase 7: LDL factorization (serial — isolate raw cost)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[ldl_kernel](
                workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var ldl_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print("7.  LDL factorization:       " + String(ldl_us)[byte=:8] + " μs")

        # Phase 8: M_inv from LDL (serial — isolate raw cost)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[minv_kernel](
                workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var minv_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print(
            "8.  M_inv from LDL:          " + String(minv_us)[byte=:8] + " μs"
        )

        # Phase 9: RNE (bias forces)
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
        print("9.  RNE (bias forces):       " + String(rne_us)[byte=:8] + " μs")

        # Phase 10: LDL solve
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[ldl_solve_kernel](
                workspace, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
            )
        ctx.synchronize()
        t1 = perf_counter_ns()
        var solve_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)
        print(
            "10. LDL solve:               " + String(solve_us)[byte=:8] + " μs"
        )

        # ── Reference: full monolithic RK4 step (gated — see PROFILE_FULL_STEP).
        var full_us = Float64(0)
        comptime if PROFILE_FULL_STEP:
            ctx.synchronize()
            t0 = perf_counter_ns()
            for _ in range(N_STEPS):
                RK4Integrator[SOLVER=NewtonSolver].step_gpu[
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
            full_us = Float64(t1 - t0) / 1000.0 / Float64(N_STEPS)

        print()
        print("=" * 60)
        var phases_total = (
            fk_us
            + vel_us
            + detect_us
            + com_us
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
        comptime if PROFILE_FULL_STEP:
            print(
                "Full RK4 step (×20 passes):  "
                + String(full_us)[byte=:8] + " μs"
            )
            print("  (includes 20× solver + finalize + 4-stage RK4 combine)")
        else:
            print("Full RK4 step:               skipped (PROFILE_FULL_STEP=False)")
        print()

        print("Phase breakdown (% of summed phases):")
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
            "  Contacts:    "
            + String(detect_us / phases_total * 100)[byte=:5]
            + "%"
        )
        print(
            "  SubtreeCOM:  "
            + String(com_us / phases_total * 100)[byte=:5]
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
        print()
        print(
            "FK + Velocities (Lever 1 target) = "
            + String((fk_us + vel_us) / phases_total * 100)[byte=:5]
            + "% of phases"
        )
