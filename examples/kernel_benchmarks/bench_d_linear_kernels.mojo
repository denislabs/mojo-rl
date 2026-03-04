"""Benchmark Group D: Linear layer GPU kernels (tiled matmul).

Each unique (IN_DIM, OUT_DIM, BATCH) produces a distinct Metal/CUDA shader.
Layers used in TDMPC2 WorldModel (HalfCheetah):

  Layer              Forward (no-cache)  Backward (dx + dW + db)
  ─────────────────────────────────────────────────────────────
  Linear[17,   256]  ✓                  ✓   (encoder input)
  Linear[262,  256]  ✓                  ✓   (dynamics/reward/term/Q input: LATENT+ACT)
  Linear[256,  256]  ✓                  ✓   (shared: dynamics proj, many places)
  Linear[256,  101]  ✓                  ✓   (reward/Q output: BINS)
  Linear[256,   12]  ✓                  ✓   (policy output: 2*ACT)
  Linear[256,    1]  ✓                  ✓   (termination output)

Note: forward_gpu_no_cache is used during inference / env collection.
      forward_gpu (with cache) + backward_gpu are used during training.

Run:
    pixi run -e apple mojo build examples/kernel_benchmarks/bench_d_linear_kernels.mojo -o /tmp/bench_d
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer
from nn.constants import dtype
from nn.model.linear import Linear

comptime BATCH: Int = 256


fn trigger_linear[
    IN: Int, OUT: Int
](ctx: DeviceContext, p: UnsafePointer[Scalar[dtype]]) raises:
    """Compile forward_no_cache, forward_with_cache, and backward for Linear[IN,OUT].
    """

    @always_inline
    fn mk(n: Int) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](ctx, p, n, owning=False)

    comptime P_SIZE: Int = IN * OUT + OUT  # weights + bias
    comptime WS: Int = 1  # Linear workspace is unused

    # Forward without cache (inference path)
    Linear[IN, OUT].forward_gpu_no_cache[BATCH](
        ctx,
        mk(BATCH * OUT),  # output
        mk(BATCH * IN),  # input
        mk(P_SIZE),  # params
        mk(WS),  # workspace (unused)
    )

    # Forward with cache (training path — caches input for weight gradient)
    Linear[IN, OUT].forward_gpu[BATCH](
        ctx,
        mk(BATCH * OUT),  # output
        mk(BATCH * IN),  # input
        mk(P_SIZE),  # params
        mk(BATCH * IN),  # cache = saved input [BATCH * IN]
        mk(WS),  # workspace
    )

    # Backward: dx + dW + db
    Linear[IN, OUT].backward_gpu[BATCH](
        ctx,
        mk(BATCH * IN),  # grad_input (output)
        mk(BATCH * OUT),  # grad_output
        mk(P_SIZE),  # params (W)
        mk(BATCH * IN),  # cache (saved input from forward)
        mk(P_SIZE),  # grads (dW + db)
        mk(WS),  # workspace
    )


fn main() raises:
    var ctx = DeviceContext()

    # Allocate a single scratch buffer large enough for all calls
    comptime MAX: Int = BATCH * 512  # 256*512 = 131072 elements, covers all shapes
    var scratch = ctx.enqueue_create_buffer[dtype](MAX)
    var p = scratch.unsafe_ptr()

    print("Linear[17,  256] (encoder input layer)...")
    trigger_linear[17, 256](ctx, p)

    print("Linear[262, 256] (dynamics/reward/Q input: LATENT+ACT)...")
    trigger_linear[262, 256](ctx, p)

    print("Linear[256, 256] (shared projection layer)...")
    trigger_linear[256, 256](ctx, p)

    print("Linear[256, 101] (reward/Q output: BINS)...")
    trigger_linear[256, 101](ctx, p)

    print("Linear[256,  12] (policy output: 2*ACT)...")
    trigger_linear[256, 12](ctx, p)

    print("Linear[256,   1] (termination output)...")
    trigger_linear[256, 1](ctx, p)

    ctx.synchronize()
    print("Group D kernels compiled and ran OK")
