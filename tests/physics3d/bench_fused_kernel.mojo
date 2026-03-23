"""Benchmark: GPU kernel launch overhead for physics step.

Measures the kernel launch overhead by comparing:
1. The actual 3-kernel physics step (step → solve → finalize)
2. A no-op kernel launched the same number of times
3. Different batch sizes to see if launch overhead dominates

This tells us whether kernel fusion would be worth pursuing.

For HalfCheetah with FRAME_SKIP=5 and ROLLOUT_LEN=512:
- 512 × 5 = 2560 physics substeps per rollout
- 4 kernel launches per substep = 10240 total launches
- Fusion would save 2560-5120 launches depending on approach

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/bench_fused_kernel.mojo
    cd mojo-rl && pixi run -e nvidia mojo run physics3d/tests/bench_fused_kernel.mojo
"""

from std.time import perf_counter_ns
from std.collections import InlineArray
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.gpu.constants import (
    TPB,
    state_size,
    model_size_with_invweight,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    integrator_workspace_size,
)
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from mojo_rl.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS  # 20
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6
comptime NSITE = HalfCheetahModel.NSITE


# =============================================================================
# No-op kernel for measuring pure launch overhead
# =============================================================================


@always_inline
def noop_kernel[
    DTYPE: DType,
    BATCH: Int,
    SIZE: Int,
](buf: LayoutTensor[DTYPE, Layout.row_major(BATCH, SIZE), MutAnyOrigin],):
    """No-op kernel: just reads thread index and exits."""
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    # Touch one element to prevent dead code elimination
    var _ = buf[env, 0]


@always_inline
def noop_2d_kernel[
    DTYPE: DType,
    BATCH: Int,
    SIZE: Int,
](buf: LayoutTensor[DTYPE, Layout.row_major(BATCH, SIZE), MutAnyOrigin],):
    """No-op 2D kernel (solver-shaped grid): reads thread index and exits."""
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    var tid_y = Int(thread_idx.y)
    if env >= BATCH:
        return
    if tid_y == 0:
        var _ = buf[env, 0]


# =============================================================================
# Benchmark functions
# =============================================================================


def bench_noop_launches[
    BATCH: Int
](ctx: DeviceContext, num_launches: Int, launches_per_iter: Int,) raises:
    """Benchmark pure kernel launch overhead with no-op kernels."""
    comptime SIZE = 16
    var buf = ctx.enqueue_create_buffer[DTYPE](BATCH * SIZE)
    var lt = LayoutTensor[DTYPE, Layout.row_major(BATCH, SIZE), MutAnyOrigin](
        buf.unsafe_ptr()
    )

    comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB

    comptime wrapper_1d = noop_kernel[DTYPE, BATCH, SIZE]
    comptime wrapper_2d = noop_2d_kernel[DTYPE, BATCH, SIZE]

    # Solver grid dims (matching HalfCheetah Newton solver)
    comptime THREADS = MAX_CONTACTS  # 20
    comptime SOLVER_ENV_TPB = TPB // THREADS  # 12
    comptime SOLVER_ENV_BLOCKS = (BATCH + SOLVER_ENV_TPB - 1) // SOLVER_ENV_TPB

    # Warmup
    for _ in range(100):
        ctx.enqueue_function[wrapper_1d, wrapper_1d](
            lt, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
        )
    ctx.synchronize()

    # --- Benchmark: N × 1D no-op launches ---
    var t0 = perf_counter_ns()
    for _ in range(num_launches):
        ctx.enqueue_function[wrapper_1d, wrapper_1d](
            lt, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var noop_1d_ms = Float64(t1 - t0) / 1e6

    # --- Benchmark: N × 2D no-op launches (solver-shaped) ---
    var t2 = perf_counter_ns()
    for _ in range(num_launches):
        ctx.enqueue_function[wrapper_2d, wrapper_2d](
            lt,
            grid_dim=(SOLVER_ENV_BLOCKS, 1),
            block_dim=(SOLVER_ENV_TPB, THREADS),
        )
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var noop_2d_ms = Float64(t3 - t2) / 1e6

    # --- Benchmark: N × 3 no-op launches (simulating 3-kernel pipeline) ---
    var t4 = perf_counter_ns()
    for _ in range(num_launches):
        ctx.enqueue_function[wrapper_1d, wrapper_1d](
            lt, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
        )
        ctx.enqueue_function[wrapper_2d, wrapper_2d](
            lt,
            grid_dim=(SOLVER_ENV_BLOCKS, 1),
            block_dim=(SOLVER_ENV_TPB, THREADS),
        )
        ctx.enqueue_function[wrapper_1d, wrapper_1d](
            lt, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
        )
    ctx.synchronize()
    var t5 = perf_counter_ns()
    var noop_3x_ms = Float64(t5 - t4) / 1e6

    print("  No-op launches (batch=" + String(BATCH) + "):")
    print(
        "    1 × 1D launch ×",
        num_launches,
        ":",
        String(noop_1d_ms)[byte=:8],
        "ms (",
        String(noop_1d_ms / Float64(num_launches) * 1000.0)[byte=:6],
        "us/launch)",
    )
    print(
        "    1 × 2D launch ×",
        num_launches,
        ":",
        String(noop_2d_ms)[byte=:8],
        "ms (",
        String(noop_2d_ms / Float64(num_launches) * 1000.0)[byte=:6],
        "us/launch)",
    )
    print(
        "    3 × launches  ×",
        num_launches,
        ":",
        String(noop_3x_ms)[byte=:8],
        "ms (",
        String(noop_3x_ms / Float64(num_launches) * 1000.0)[byte=:6],
        "us/iter, overhead=",
        String((noop_3x_ms - noop_1d_ms) / Float64(num_launches) * 1000.0)[byte=:6],
        "us extra)",
    )
    print()


def bench_physics_step[
    BATCH: Int
](ctx: DeviceContext, num_substeps: Int,) raises:
    """Benchmark the actual 3-kernel physics step."""

    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
    comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
    comptime WS_SIZE = integrator_workspace_size[
        NV, NBODY
    ]() + NV * NV + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()

    # === Allocate GPU buffers ===
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)

    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var ws_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)

    # === Initialize state ===
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for e in range(BATCH):
        var base = e * STATE_SIZE
        state_host[base + qpos_offset[NQ, NV]() + 1] = Scalar[DTYPE](0.7)
        var qfrc_off = qfrc_offset[NQ, NV]()
        state_host[base + qfrc_off + 3] = Scalar[DTYPE](0.5 * 120.0)
        state_host[base + qfrc_off + 4] = Scalar[DTYPE](-0.3 * 90.0)
        state_host[base + qfrc_off + 5] = Scalar[DTYPE](0.2 * 60.0)
        state_host[base + qfrc_off + 6] = Scalar[DTYPE](0.4 * 120.0)
        state_host[base + qfrc_off + 7] = Scalar[DTYPE](-0.1 * 60.0)
        state_host[base + qfrc_off + 8] = Scalar[DTYPE](0.3 * 30.0)

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    # === Warmup ===
    for _ in range(10):
        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM,
            CONE_TYPE=HalfCheetahModel.CONE_TYPE,
        ](ctx, state_buf, model_buf, ws_buf)
    ctx.synchronize()

    # Re-initialize
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    # === Benchmark: physics step ===
    var t0 = perf_counter_ns()
    for _ in range(num_substeps):
        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM,
            CONE_TYPE=HalfCheetahModel.CONE_TYPE,
        ](ctx, state_buf, model_buf, ws_buf)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var physics_ms = Float64(t1 - t0) / 1e6
    var per_step_us = physics_ms / Float64(num_substeps) * 1000.0
    var total_launches = num_substeps * 4

    print(
        "  Physics step (batch="
        + String(BATCH)
        + ", substeps="
        + String(num_substeps)
        + "):"
    )
    print(
        "    Total:      ",
        String(physics_ms)[byte=:8],
        "ms (",
        total_launches,
        "kernel launches)",
    )
    print(
        "    Per substep:",
        String(per_step_us)[byte=:7],
        "us (4 launches)",
    )
    print(
        "    Per launch: ",
        String(physics_ms / Float64(total_launches) * 1000.0)[byte=:7],
        "us (avg across step/solve/finalize)",
    )
    print()


def bench_physics_with_sync[
    BATCH: Int
](ctx: DeviceContext, num_substeps: Int,) raises:
    """Benchmark with synchronize after each kernel to isolate per-kernel timing.
    """

    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
    comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
    comptime WS_SIZE = integrator_workspace_size[
        NV, NBODY
    ]() + NV * NV + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()

    comptime THREADS = NewtonSolver.solver_threads[
        NQ, NV, NBODY, NJOINT, MAX_CONTACTS
    ]()
    comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
    comptime SOLVER_ENV_TPB = TPB // THREADS
    comptime SOLVER_ENV_BLOCKS = (BATCH + SOLVER_ENV_TPB - 1) // SOLVER_ENV_TPB
    comptime SOLVER_THREADS_BLOCKS = (THREADS + THREADS - 1) // THREADS

    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)

    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var ws_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)

    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for e in range(BATCH):
        var base = e * STATE_SIZE
        state_host[base + qpos_offset[NQ, NV]() + 1] = Scalar[DTYPE](0.7)
        var qfrc_off = qfrc_offset[NQ, NV]()
        state_host[base + qfrc_off + 3] = Scalar[DTYPE](60.0)
        state_host[base + qfrc_off + 4] = Scalar[DTYPE](-27.0)
        state_host[base + qfrc_off + 5] = Scalar[DTYPE](12.0)
        state_host[base + qfrc_off + 6] = Scalar[DTYPE](48.0)
        state_host[base + qfrc_off + 7] = Scalar[DTYPE](-6.0)
        state_host[base + qfrc_off + 8] = Scalar[DTYPE](9.0)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    # Warmup
    for _ in range(10):
        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM,
            CONE_TYPE=HalfCheetahModel.CONE_TYPE,
        ](ctx, state_buf, model_buf, ws_buf)
    ctx.synchronize()

    # Re-init
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    # === Per-kernel timing with sync ===
    var state = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var model = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())
    var workspace = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ](ws_buf.unsafe_ptr())

    comptime step_wrapper = EulerIntegrator[SOLVER=NewtonSolver].step_kernel[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
        WS_SIZE,
        NGEOM,
    ]
    comptime contact_wrapper = EulerIntegrator[
        SOLVER=NewtonSolver
    ].contact_detection_kernel[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
        NGEOM,
    ]
    comptime solver_wrapper = NewtonSolver.solve_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        NV,
        BATCH,
        WS_SIZE,
        NGEOM,
        CONE_TYPE=HalfCheetahModel.CONE_TYPE,
    ]
    comptime finalize_wrapper = EulerIntegrator[
        SOLVER=NewtonSolver
    ].step_finalize_kernel[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
        WS_SIZE,
    ]

    var step_ns: Int = 0
    var contact_ns: Int = 0
    var solve_ns: Int = 0
    var finalize_ns: Int = 0

    for _ in range(num_substeps):
        var ts0 = perf_counter_ns()
        ctx.enqueue_function[step_wrapper, step_wrapper](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.synchronize()
        var ts1 = perf_counter_ns()

        ctx.enqueue_function[contact_wrapper, contact_wrapper](
            state,
            model,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.synchronize()
        var ts_contact = perf_counter_ns()

        ctx.enqueue_function[solver_wrapper, solver_wrapper](
            state,
            model,
            workspace,
            grid_dim=(SOLVER_ENV_BLOCKS, SOLVER_THREADS_BLOCKS),
            block_dim=(SOLVER_ENV_TPB, THREADS),
        )
        ctx.synchronize()
        var ts2 = perf_counter_ns()

        ctx.enqueue_function[finalize_wrapper, finalize_wrapper](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.synchronize()
        var ts3 = perf_counter_ns()

        step_ns += Int(ts1 - ts0)
        contact_ns += Int(ts_contact - ts1)
        solve_ns += Int(ts2 - ts_contact)
        finalize_ns += Int(ts3 - ts2)

    var step_ms = Float64(step_ns) / 1e6
    var contact_ms = Float64(contact_ns) / 1e6
    var solve_ms = Float64(solve_ns) / 1e6
    var finalize_ms = Float64(finalize_ns) / 1e6
    var total_ms = step_ms + contact_ms + solve_ms + finalize_ms

    print(
        "  Per-kernel timing with sync (batch="
        + String(BATCH)
        + ", substeps="
        + String(num_substeps)
        + "):"
    )
    print(
        "    Step kernel:     ",
        String(step_ms)[byte=:8],
        "ms (",
        String(step_ms / Float64(num_substeps) * 1000.0)[byte=:7],
        "us/call,",
        String(step_ms / total_ms * 100.0)[byte=:5],
        "%)",
    )
    print(
        "    Contact kernel:  ",
        String(contact_ms)[byte=:8],
        "ms (",
        String(contact_ms / Float64(num_substeps) * 1000.0)[byte=:7],
        "us/call,",
        String(contact_ms / total_ms * 100.0)[byte=:5],
        "%)",
    )
    print(
        "    Solve kernel:    ",
        String(solve_ms)[byte=:8],
        "ms (",
        String(solve_ms / Float64(num_substeps) * 1000.0)[byte=:7],
        "us/call,",
        String(solve_ms / total_ms * 100.0)[byte=:5],
        "%)",
    )
    print(
        "    Finalize kernel: ",
        String(finalize_ms)[byte=:8],
        "ms (",
        String(finalize_ms / Float64(num_substeps) * 1000.0)[byte=:7],
        "us/call,",
        String(finalize_ms / total_ms * 100.0)[byte=:5],
        "%)",
    )
    print(
        "    Total:           ",
        String(total_ms)[byte=:8],
        "ms (",
        String(total_ms / Float64(num_substeps) * 1000.0)[byte=:7],
        "us/substep)",
    )
    print(
        (
            "    NOTE: sync after each kernel adds overhead vs pipelined"
            " execution."
        ),
    )
    print()


def bench_physics_step_mt[
    BATCH: Int, STEP_THREADS: Int
](ctx: DeviceContext, num_substeps: Int,) raises:
    """Benchmark multi-threaded step kernel vs single-threaded."""

    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
    comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
    comptime WS_SIZE = integrator_workspace_size[
        NV, NBODY
    ]() + NV * NV + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()

    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)

    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var ws_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)

    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for e in range(BATCH):
        var base = e * STATE_SIZE
        state_host[base + qpos_offset[NQ, NV]() + 1] = Scalar[DTYPE](0.7)
        var qfrc_off = qfrc_offset[NQ, NV]()
        state_host[base + qfrc_off + 3] = Scalar[DTYPE](0.5 * 120.0)
        state_host[base + qfrc_off + 4] = Scalar[DTYPE](-0.3 * 90.0)
        state_host[base + qfrc_off + 5] = Scalar[DTYPE](0.2 * 60.0)
        state_host[base + qfrc_off + 6] = Scalar[DTYPE](0.4 * 120.0)
        state_host[base + qfrc_off + 7] = Scalar[DTYPE](-0.1 * 60.0)
        state_host[base + qfrc_off + 8] = Scalar[DTYPE](0.3 * 30.0)

    # --- Benchmark: single-threaded (STEP_THREADS=1) ---
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    # Warmup
    for _ in range(10):
        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM,
            CONE_TYPE=HalfCheetahModel.CONE_TYPE,
            STEP_THREADS=1,
        ](ctx, state_buf, model_buf, ws_buf)
    ctx.synchronize()

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(num_substeps):
        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM,
            CONE_TYPE=HalfCheetahModel.CONE_TYPE,
            STEP_THREADS=1,
        ](ctx, state_buf, model_buf, ws_buf)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var st_ms = Float64(t1 - t0) / 1e6
    var st_per_step = st_ms / Float64(num_substeps) * 1000.0

    # --- Benchmark: multi-threaded (STEP_THREADS=NV) ---
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    # Warmup
    for _ in range(10):
        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM,
            CONE_TYPE=HalfCheetahModel.CONE_TYPE,
            STEP_THREADS=STEP_THREADS,
        ](ctx, state_buf, model_buf, ws_buf)
    ctx.synchronize()

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    var t2 = perf_counter_ns()
    for _ in range(num_substeps):
        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM,
            CONE_TYPE=HalfCheetahModel.CONE_TYPE,
            STEP_THREADS=STEP_THREADS,
        ](ctx, state_buf, model_buf, ws_buf)
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var mt_ms = Float64(t3 - t2) / 1e6
    var mt_per_step = mt_ms / Float64(num_substeps) * 1000.0

    var speedup = st_ms / mt_ms

    print(
        "  Multi-thread step kernel (batch="
        + String(BATCH)
        + ", STEP_THREADS="
        + String(STEP_THREADS)
        + ", substeps="
        + String(num_substeps)
        + "):"
    )
    print(
        "    Single-threaded: ",
        String(st_ms)[byte=:8],
        "ms (",
        String(st_per_step)[byte=:7],
        "us/substep)",
    )
    print(
        "    Multi-threaded:  ",
        String(mt_ms)[byte=:8],
        "ms (",
        String(mt_per_step)[byte=:7],
        "us/substep)",
    )
    print(
        "    Speedup:         ",
        String(speedup)[byte=:5],
        "x",
    )
    print()


# =============================================================================
# Main
# =============================================================================


def main() raises:
    print("=" * 70)
    print("Benchmark: GPU Kernel Launch Overhead for Physics Step")
    print("=" * 70)
    print()
    print("Environment: HalfCheetah (NQ=9, NV=9, MAX_CONTACTS=20)")
    print("Solver: Newton (solver_threads=20)")
    print(
        "4 kernel launches per substep: step(1D) + contact(1D) + solve(2D) +"
        " finalize(1D)"
    )
    print()

    with DeviceContext() as ctx:
        # === Section 1: No-op kernel launch overhead ===
        print("-" * 70)
        print("Section 1: Pure kernel launch overhead (no-op kernels)")
        print("-" * 70)
        bench_noop_launches[64](ctx, 10000, 3)
        bench_noop_launches[256](ctx, 10000, 3)
        bench_noop_launches[1024](ctx, 10000, 3)

        # === Section 2: Physics step total time (pipelined, no sync) ===
        print("-" * 70)
        print("Section 2: Physics step (pipelined, sync only at end)")
        print("-" * 70)
        bench_physics_step[64](ctx, 2560)
        bench_physics_step[256](ctx, 2560)
        bench_physics_step[512](ctx, 2560)
        bench_physics_step[1024](ctx, 2560)

        # === Section 3: Per-kernel timing (with sync) ===
        print("-" * 70)
        print("Section 3: Per-kernel timing (sync after each kernel)")
        print(
            "  Shows actual GPU execution time per kernel, but sync adds"
            " overhead."
        )
        print("-" * 70)
        bench_physics_with_sync[256](ctx, 500)
        bench_physics_with_sync[1024](ctx, 500)

        # === Section 4: Multi-thread step kernel comparison ===
        print("-" * 70)
        print(
            "Section 4: Multi-thread step kernel (STEP_THREADS=1 vs"
            " STEP_THREADS=NV)"
        )
        print("-" * 70)
        bench_physics_step_mt[256, NV](ctx, 2560)
        bench_physics_step_mt[512, NV](ctx, 2560)
        bench_physics_step_mt[1024, NV](ctx, 2560)

        # === Section 5: Extrapolation ===
        print("-" * 70)
        print("Section 5: Impact estimate for training (batch=256)")
        print("-" * 70)
        print("  Per rollout: 512 steps × 5 frame_skip = 2560 physics substeps")
        print("  Per rollout: 2560 × 4 launches = 10240 kernel launches")
        print(
            "  If fusion saves 3 launches/substep: 7680 fewer launches/rollout"
        )
        print()

        print("=" * 70)
        print("Done!")
        print("=" * 70)
