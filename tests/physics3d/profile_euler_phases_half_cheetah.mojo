"""Profile EulerIntegrator phases individually — HalfCheetah (NV=9).

Companion to `profile_rk4_phases_humanoid.mojo`. HalfCheetah is the env that uses
EulerIntegrator (1 forward-dynamics eval/step) at small NV (9, shallow tree) and
PYRAMIDAL cone — the regime where the RK4 cooperative `_mt` wins (measured on
Humanoid NV=23) are *expected* to shrink. This profiler answers, before any port:

  1. Which phases dominate the Euler forward-dynamics step at NV=9?
  2. Do the cooperative `_mt` walks actually BEAT their serial form here, or does
     barrier overhead eat the win (the walker2d NV=9 prediction)? — each of
     FK / vel / cdof / RNE / mass-matrix is timed serial AND `_mt`, head-to-head.

It does NOT measure the physics-vs-network split (that needs the full SAC run —
see sac_half_cheetah_profile_graph*.txt). It also does not time contact detection
or the constraint solver — only the dense forward-dynamics phases — matching the
Humanoid profiler's scope.

Run:
    pixi run -e nvidia mojo run -I . tests/physics3d/profile_euler_phases_half_cheetah.mojo

The launch config mirrors production: EulerIntegrator.step_gpu is launched with
STEP_THREADS=NV (half_cheetah_config.mojo:187), so the `_mt` 2D-block launch here
(grid=(STEP_ENV_BLOCKS,1), block=(STEP_ENV_TPB, STEP_THREADS)) is the real path.

NOTE on compile cost: at NV=9 the per-thread frames are small, so compiling all
the `_mt` head-to-heads together is fine (unlike Humanoid NV=23, where >1-2
cooperative `_mt` standalone kernels OOM the host Mojo->PTX compiler — see
project_physics3d_blocked_solver memory). If this ever changes, drop a head-to-head.
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
    compute_mass_matrix_treewalk_gpu_mt,
    ldl_factor_gpu_mt,
    compute_M_inv_from_ldl_gpu_mt,
    ldl_solve_workspace_gpu,
)
from mojo_rl.physics3d.dynamics.bias_forces import (
    compute_bias_forces_rne_gpu,
    compute_bias_forces_rne_gpu_mt,
)
from mojo_rl.physics3d.solver.newton_solver import NewtonSolver

# Legacy engine on purpose: this profiler drives the legacy static GPU kernels
# (reset_kernel_gpu) + legacy solvers/integrators; dies with legacy at P6.
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel


def main() raises:
    seed(42)

    comptime dtype = DType.float32
    comptime BATCH = 256
    comptime N_STEPS = 500

    # HalfCheetah dimensions (ModelDefFromXML) — match Phyics3dEnv exactly.
    comptime NQ = HalfCheetahModel.NQ
    comptime NV = HalfCheetahModel.NV  # 9
    comptime NBODY = HalfCheetahModel.NBODY
    comptime NJOINT = HalfCheetahModel.NJOINT
    comptime MAX_CONTACTS = HalfCheetahModel.MAX_CONTACTS
    comptime NGEOM = HalfCheetahModel.NGEOM
    comptime NSITE = HalfCheetahModel.NSITE
    comptime MAX_EQUALITY = HalfCheetahModel.MAX_EQUALITY
    comptime MAX_TENDON = HalfCheetahModel.MAX_TENDON
    comptime NEXCLUDE = HalfCheetahModel.nexclude

    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
    comptime MODEL_SIZE = model_size_with_invweight[
        NBODY, NJOINT, NV, NGEOM, MAX_EQUALITY, MAX_TENDON, NSITE, NEXCLUDE
    ]()
    # EulerIntegrator: INTEGRATOR_WS_EXTRA = 0 (no rk4_extra), so WS_SIZE matches
    # Phyics3dEnv.STEP_WS_PER_ENV exactly.
    comptime WS_SIZE = integrator_workspace_size[
        NV, NBODY
    ]() + NV * NV + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()

    comptime STEP_THREADS = NV  # production (half_cheetah_config.mojo:187)
    comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
    comptime STEP_ENV_TPB = TPB // STEP_THREADS
    comptime STEP_ENV_BLOCKS = (BATCH + STEP_ENV_TPB - 1) // STEP_ENV_TPB

    print("=" * 60)
    print("Euler Integrator Phase Profiling — HalfCheetah (NV=9)")
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
    print("Steps: " + String(N_STEPS))
    print()

    with DeviceContext() as ctx:
        var state_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
        var model_buf = ctx.enqueue_create_buffer[dtype](MODEL_SIZE)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](BATCH * WS_SIZE)

        HalfCheetahModel.init_model_gpu(ctx, model_buf)
        Phyics3dEnv[HalfCheetahModel, HalfCheetahConfig, dtype].reset_kernel_gpu[BATCH, STATE_SIZE](ctx, state_buf)
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

        # ── Phase kernels (each a separate launch) ──

        # Phase 1: Forward Kinematics — serial
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
                dtype, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, NGEOM,
            ](env, s, m)

        # Phase 1b: Forward Kinematics — MT (level-parallel)
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
                dtype, NQ, NV, NBODY, NJOINT, STATE_SIZE, MODEL_SIZE, BATCH
            ](env, tid, STEP_THREADS, valid_env, s, m)

        # Phase 2: Body Velocities — serial
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
                dtype, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH,
            ](env, s, m)

        # Phase 2b: Body Velocities — MT (level-parallel)
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
                dtype, NQ, NV, NBODY, NJOINT, STATE_SIZE, MODEL_SIZE, BATCH
            ](env, tid, STEP_THREADS, valid_env, s, m)

        # Phase 3: CDOF — serial
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
                dtype, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, s, m, w)

        # Phase 3b: CDOF — MT (flat-parallel)
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
                dtype, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, tid, STEP_THREADS, valid_env, s, m, w)

        # Phase 4: CRB (composite inertia) — serial
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
                dtype, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, s, m, w)

        # Phase 5: Mass Matrix — dense MT (the form Euler currently uses)
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
                dtype, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, tid, STEP_THREADS, s, m, w)

        # Phase 5b: Mass Matrix — tree-walk CRBA (the RK4 win; needs FK+cdof+crb
        # populated first — the warmup runs them in order).
        @always_inline
        def mass_matrix_treewalk_kernel(
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
            compute_mass_matrix_treewalk_gpu_mt[
                dtype, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, tid, STEP_THREADS, valid_env, s, m, w)

        # Phase 6: LDL factorization — MT
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

        # Phase 7: M_inv from LDL — MT
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

        # Phase 8: RNE bias forces — serial
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
                dtype, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, s, m, w)

        # Phase 8b: RNE — MT (cooperative)
        @always_inline
        def rne_mt_kernel(
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
            compute_bias_forces_rne_gpu_mt[
                dtype, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, tid, STEP_THREADS, valid_env, s, m, w)

        # Phase 9: LDL solve — serial
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

        comptime grid1d = (ENV_BLOCKS,)
        comptime block1d = (TPB,)
        comptime grid2d = (STEP_ENV_BLOCKS, 1)
        comptime block2d = (STEP_ENV_TPB, STEP_THREADS)

        # ── Warmup (dependency order so each phase reads valid upstream data) ──
        print("Warming up phase kernels...")
        for _ in range(3):
            ctx.enqueue_function[fk_kernel](
                state, model, grid_dim=grid1d, block_dim=block1d
            )
            ctx.enqueue_function[fk_mt_kernel](
                state, model, grid_dim=grid2d, block_dim=block2d
            )
            ctx.enqueue_function[vel_kernel](
                state, model, grid_dim=grid1d, block_dim=block1d
            )
            ctx.enqueue_function[vel_mt_kernel](
                state, model, grid_dim=grid2d, block_dim=block2d
            )
            ctx.enqueue_function[cdof_kernel](
                state, model, workspace, grid_dim=grid1d, block_dim=block1d
            )
            ctx.enqueue_function[cdof_mt_kernel](
                state, model, workspace, grid_dim=grid2d, block_dim=block2d
            )
            ctx.enqueue_function[crb_kernel](
                state, model, workspace, grid_dim=grid1d, block_dim=block1d
            )
            ctx.enqueue_function[mass_matrix_kernel](
                state, model, workspace, grid_dim=grid2d, block_dim=block2d
            )
            ctx.enqueue_function[mass_matrix_treewalk_kernel](
                state, model, workspace, grid_dim=grid2d, block_dim=block2d
            )
            ctx.enqueue_function[ldl_kernel](
                workspace, grid_dim=grid2d, block_dim=block2d
            )
            ctx.enqueue_function[minv_kernel](
                workspace, grid_dim=grid2d, block_dim=block2d
            )
            ctx.enqueue_function[rne_kernel](
                state, model, workspace, grid_dim=grid1d, block_dim=block1d
            )
            ctx.enqueue_function[rne_mt_kernel](
                state, model, workspace, grid_dim=grid2d, block_dim=block2d
            )
            ctx.enqueue_function[ldl_solve_kernel](
                workspace, grid_dim=grid1d, block_dim=block1d
            )
        ctx.synchronize()
        print("Warmup done!")
        print()

        print("Profiling individual phases (" + String(N_STEPS) + " steps)...")
        print("-" * 60)

        # FK (serial vs MT)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[fk_kernel](
                state, model, grid_dim=grid1d, block_dim=block1d
            )
        ctx.synchronize()
        var fk_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(N_STEPS)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[fk_mt_kernel](
                state, model, grid_dim=grid2d, block_dim=block2d
            )
        ctx.synchronize()
        var fk_mt_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(N_STEPS)

        # Velocities (serial vs MT)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[vel_kernel](
                state, model, grid_dim=grid1d, block_dim=block1d
            )
        ctx.synchronize()
        var vel_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(N_STEPS)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[vel_mt_kernel](
                state, model, grid_dim=grid2d, block_dim=block2d
            )
        ctx.synchronize()
        var vel_mt_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(
            N_STEPS
        )

        # CDOF (serial vs MT)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[cdof_kernel](
                state, model, workspace, grid_dim=grid1d, block_dim=block1d
            )
        ctx.synchronize()
        var cdof_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(
            N_STEPS
        )
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[cdof_mt_kernel](
                state, model, workspace, grid_dim=grid2d, block_dim=block2d
            )
        ctx.synchronize()
        var cdof_mt_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(
            N_STEPS
        )

        # CRB (serial only — dead under treewalk, kept for reference)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[crb_kernel](
                state, model, workspace, grid_dim=grid1d, block_dim=block1d
            )
        ctx.synchronize()
        var crb_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(N_STEPS)

        # Mass matrix (dense MT vs tree-walk CRBA)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[mass_matrix_kernel](
                state, model, workspace, grid_dim=grid2d, block_dim=block2d
            )
        ctx.synchronize()
        var mm_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(N_STEPS)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[mass_matrix_treewalk_kernel](
                state, model, workspace, grid_dim=grid2d, block_dim=block2d
            )
        ctx.synchronize()
        var mm_tw_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(
            N_STEPS
        )

        # LDL factor (MT)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[ldl_kernel](
                workspace, grid_dim=grid2d, block_dim=block2d
            )
        ctx.synchronize()
        var ldl_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(N_STEPS)

        # M_inv (MT)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[minv_kernel](
                workspace, grid_dim=grid2d, block_dim=block2d
            )
        ctx.synchronize()
        var minv_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(
            N_STEPS
        )

        # RNE (serial vs MT)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[rne_kernel](
                state, model, workspace, grid_dim=grid1d, block_dim=block1d
            )
        ctx.synchronize()
        var rne_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(N_STEPS)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[rne_mt_kernel](
                state, model, workspace, grid_dim=grid2d, block_dim=block2d
            )
        ctx.synchronize()
        var rne_mt_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(
            N_STEPS
        )

        # LDL solve (serial)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(N_STEPS):
            ctx.enqueue_function[ldl_solve_kernel](
                workspace, grid_dim=grid1d, block_dim=block1d
            )
        ctx.synchronize()
        var solve_us = Float64(perf_counter_ns() - t0) / 1000.0 / Float64(
            N_STEPS
        )

        # ── Report ──
        print("Phase (best-of serial/MT used in 'production column'):")
        print(
            "1.  FK:            serial "
            + String(fk_us)[byte=:7]
            + "  MT "
            + String(fk_mt_us)[byte=:7]
            + "  ("
            + String(fk_us / fk_mt_us)[byte=:5]
            + "x)"
        )
        print(
            "2.  Velocities:    serial "
            + String(vel_us)[byte=:7]
            + "  MT "
            + String(vel_mt_us)[byte=:7]
            + "  ("
            + String(vel_us / vel_mt_us)[byte=:5]
            + "x)"
        )
        print(
            "3.  CDOF:          serial "
            + String(cdof_us)[byte=:7]
            + "  MT "
            + String(cdof_mt_us)[byte=:7]
            + "  ("
            + String(cdof_us / cdof_mt_us)[byte=:5]
            + "x)"
        )
        print("4.  CRB (serial):  " + String(crb_us)[byte=:7] + " μs")
        print(
            "5.  Mass matrix:   dense "
            + String(mm_us)[byte=:7]
            + "  treewalk "
            + String(mm_tw_us)[byte=:7]
            + "  ("
            + String(mm_us / mm_tw_us)[byte=:5]
            + "x)"
        )
        print("6.  LDL factor MT: " + String(ldl_us)[byte=:7] + " μs")
        print("7.  M_inv MT:      " + String(minv_us)[byte=:7] + " μs")
        print(
            "8.  RNE:           serial "
            + String(rne_us)[byte=:7]
            + "  MT "
            + String(rne_mt_us)[byte=:7]
            + "  ("
            + String(rne_us / rne_mt_us)[byte=:5]
            + "x)"
        )
        print("9.  LDL solve:     " + String(solve_us)[byte=:7] + " μs")
        print("=" * 60)

        # Production-path total: best (MT where it wins) for the parallelizable
        # walks + treewalk MM, plus the already-MT LDL/M_inv + serial solve.
        var best_fk = fk_mt_us if fk_mt_us < fk_us else fk_us
        var best_vel = vel_mt_us if vel_mt_us < vel_us else vel_us
        var best_cdof = cdof_mt_us if cdof_mt_us < cdof_us else cdof_us
        var best_mm = mm_tw_us if mm_tw_us < mm_us else mm_us
        var best_rne = rne_mt_us if rne_mt_us < rne_us else rne_us
        var prod_total = (
            best_fk
            + best_vel
            + best_cdof
            + best_mm
            + ldl_us
            + minv_us
            + best_rne
            + solve_us
        )
        var serial_total = (
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
            "Serial-walks total:        " + String(serial_total)[byte=:8] + " μs"
        )
        print(
            "Best (MT walks) total:     " + String(prod_total)[byte=:8] + " μs"
        )
        print(
            "Potential walk savings:    "
            + String(serial_total - prod_total)[byte=:8]
            + " μs  ("
            + String((serial_total - prod_total) / serial_total * 100)[byte=:5]
            + "% of forward-dynamics phases)"
        )
        print("=" * 60)
        print(
            "NOTE: this is the WITHIN-PHYSICS picture. Whether physics is worth"
        )
        print(
            "optimizing at all needs the physics-vs-network split from a full"
        )
        print("SAC run (sac_half_cheetah_profile_graph*.txt).")
