"""Benchmark Group E: NormedLinear + SimNorm + Sigmoid GPU kernels.

NormedLinear[IN,OUT] = Sequential[Linear[IN,OUT], LayerNorm[OUT], Mish[OUT]]
Each unique (IN, OUT) is a distinct Sequential type with its own kernel instantiations.

Layers in TDMPC2 WorldModel:
  NormedLinear[17,   256]  (encoder first block)
  NormedLinear[262,  256]  (dynamics/reward/Q first block: LATENT+ACT)
  NormedLinear[256,  256]  (shared second block in all networks)
  SimNorm[256, 8]          (dynamics output normalization)
  Sigmoid[1]               (termination output activation)

Note: LayerNorm and Mish are compiled inside NormedLinear's Sequential.
      If those are slow, they will show up in this group.

Run:
    pixi run -e apple mojo build examples/kernel_benchmarks/bench_e_normed_linear_kernels.mojo -o /tmp/bench_e
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer
from nn.constants import dtype
from nn.model.normed_linear import NormedLinear
from nn.model.simnorm import SimNorm
from nn.model.sigmoid import Sigmoid

comptime BATCH: Int = 256
comptime LATENT: Int = 256
comptime SIMPLEX: Int = 8


fn trigger_normed_linear[
    IN: Int, OUT: Int
](ctx: DeviceContext, p: UnsafePointer[Scalar[dtype]]) raises:
    """Compile forward_no_cache, forward_with_cache, backward for NormedLinear[IN,OUT].
    """

    @always_inline
    fn mk(n: Int) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](ctx, p, n, owning=False)

    comptime P: Int = NormedLinear[IN, OUT].PARAM_SIZE
    comptime C: Int = NormedLinear[IN, OUT].CACHE_SIZE
    comptime W: Int = NormedLinear[IN, OUT].WORKSPACE_SIZE_PER_SAMPLE
    comptime WS: Int = BATCH * W if W > 0 else 1

    # Forward no-cache (inference)
    NormedLinear[IN, OUT].forward_gpu_no_cache[BATCH](
        ctx,
        mk(BATCH * OUT),  # output
        mk(BATCH * IN),  # input
        mk(P),  # params
        mk(WS),  # workspace
    )

    # Forward with cache (training)
    NormedLinear[IN, OUT].forward_gpu[BATCH](
        ctx,
        mk(BATCH * OUT),  # output
        mk(BATCH * IN),  # input
        mk(P),  # params
        mk(BATCH * C),  # cache
        mk(WS),  # workspace
    )

    # Backward
    NormedLinear[IN, OUT].backward_gpu[BATCH](
        ctx,
        mk(BATCH * IN),  # grad_input
        mk(BATCH * OUT),  # grad_output
        mk(P),  # params
        mk(BATCH * C),  # cache
        mk(P),  # grad_params
        mk(WS),  # workspace
    )


fn trigger_simnorm(ctx: DeviceContext, p: UnsafePointer[Scalar[dtype]]) raises:
    @always_inline
    fn mk(n: Int) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](ctx, p, n, owning=False)

    comptime P: Int = SimNorm[LATENT, SIMPLEX].PARAM_SIZE  # = 0
    comptime C: Int = SimNorm[LATENT, SIMPLEX].CACHE_SIZE
    comptime W: Int = SimNorm[LATENT, SIMPLEX].WORKSPACE_SIZE_PER_SAMPLE
    comptime WS: Int = BATCH * W if W > 0 else 1
    comptime PARAMS: Int = P if P > 0 else 1

    SimNorm[LATENT, SIMPLEX].forward_gpu_no_cache[BATCH](
        ctx, mk(BATCH * LATENT), mk(BATCH * LATENT), mk(PARAMS), mk(WS)
    )
    SimNorm[LATENT, SIMPLEX].forward_gpu[BATCH](
        ctx,
        mk(BATCH * LATENT),
        mk(BATCH * LATENT),
        mk(PARAMS),
        mk(BATCH * C),
        mk(WS),
    )
    SimNorm[LATENT, SIMPLEX].backward_gpu[BATCH](
        ctx,
        mk(BATCH * LATENT),
        mk(BATCH * LATENT),
        mk(PARAMS),
        mk(BATCH * C),
        mk(PARAMS),
        mk(WS),
    )


fn trigger_sigmoid(ctx: DeviceContext, p: UnsafePointer[Scalar[dtype]]) raises:
    @always_inline
    fn mk(n: Int) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](ctx, p, n, owning=False)

    comptime C: Int = Sigmoid[1].CACHE_SIZE
    comptime W: Int = Sigmoid[1].WORKSPACE_SIZE_PER_SAMPLE
    comptime WS: Int = BATCH * W if W > 0 else 1
    comptime PARAMS: Int = 1  # no params but need non-zero buffer

    Sigmoid[1].forward_gpu_no_cache[BATCH](
        ctx, mk(BATCH), mk(BATCH), mk(PARAMS), mk(WS)
    )
    Sigmoid[1].forward_gpu[BATCH](
        ctx, mk(BATCH), mk(BATCH), mk(PARAMS), mk(BATCH * C), mk(WS)
    )
    Sigmoid[1].backward_gpu[BATCH](
        ctx, mk(BATCH), mk(BATCH), mk(PARAMS), mk(BATCH * C), mk(PARAMS), mk(WS)
    )


fn main() raises:
    var ctx = DeviceContext()

    # Allocate scratch large enough for all calls
    # NormedLinear[262,256] cache = BATCH * C which is the largest
    comptime MAX: Int = BATCH * 1024
    var scratch = ctx.enqueue_create_buffer[dtype](MAX)
    var p = scratch.unsafe_ptr()

    print("NormedLinear[17,  256] (encoder first block)...")
    trigger_normed_linear[17, 256](ctx, p)

    print("NormedLinear[262, 256] (dynamics/reward/Q first block)...")
    trigger_normed_linear[262, 256](ctx, p)

    print("NormedLinear[256, 256] (shared second block)...")
    trigger_normed_linear[256, 256](ctx, p)

    print("SimNorm[256, 8] (dynamics output normalization)...")
    trigger_simnorm(ctx, p)

    print("Sigmoid[1] (termination output activation)...")
    trigger_sigmoid(ctx, p)

    ctx.synchronize()
    print("Group E kernels compiled and ran OK")
