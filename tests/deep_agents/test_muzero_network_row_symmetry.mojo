"""Network forward row-symmetry test at BATCH=8.

Bisection step (2026-05-23): the MCTS per-env isolation test
(``test_muzero_gpu_mcts_per_env_isolation.mojo``) shows env_0 diverges
from envs 1..7 (which are bit-identical to each other) even at
BATCH_SIMS=1. The first divergence point is the leaf reward + leaf
value, both of which decode from the dynamics/prediction network output
batched at ``N_ENVS * BATCH_SIMS = 8`` rows. The MCTS kernels look
per-env-isolated by code review.

This test discriminates between two remaining hypotheses:

  A. **Network forward has a row-0 bug at BATCH=8.** Build 8 identical
     input rows, forward through each MuZero net, check whether all
     8 output rows are bit-identical. If row 0 differs from rows 1..7,
     the bug is in the AutoFused / Linear / Mish / MinMaxNorm GPU
     kernels at BATCH=8 — not in MCTS.
  B. **Network is fine; bug is in MCTS scratch / tree init.** All 8
     output rows match → the network forward is symmetric, so the
     row-0 anomaly comes from a buffer that's not properly initialized
     at env-0 slab.

The Stage 1 parity test ran at BATCH=4 and passed; if THIS test fails
at BATCH=8 with row-0 specifically off, that's a comptime-specific
kernel bug not caught by Stage 1.

Usage:
    pixi run -e apple mojo run -I . tests/deep_agents/test_muzero_network_row_symmetry.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs
from std.memory import alloc
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Linear,
    LinearMish,
    Parallel,
    MinMaxNorm,
)


def row_symmetry_check[M: Model, BATCH: Int = 8](
    ctx: DeviceContext, name: String, tol: Float64 = 1e-6
) raises:
    """Forward BATCH identical rows through M.forward_gpu; check rows match."""
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print()
    print(
        "[", name, "] IN=", IN, " OUT=", OUT, " BATCH=", BATCH
    )

    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()

    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    # Build a single row of input + replicate it BATCH times.
    var row_template = alloc[Scalar[dtype]](IN)
    for i in range(IN):
        row_template[i] = Scalar[dtype](
            0.05 + Float64(i % 17) / 17.0 * 0.6 - 0.2
        )

    var input_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    for b in range(BATCH):
        for i in range(IN):
            input_host[b * IN + i] = row_template[i]
    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    ctx.enqueue_copy(input_buf, input_host)

    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * CS if CS > 0 else 1
    )
    var workspace = ctx.enqueue_create_buffer[dtype](
        BATCH * WS if WS > 0 else 1
    )

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_buf.unsafe_ptr())
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CS), MutAnyOrigin
    ](cache_buf.unsafe_ptr())

    M.forward_gpu[BATCH](
        ctx, output_t, input_t, gpu.params_view(),
        gpu.model_state_view(), cache_t, workspace,
    )

    var output_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    ctx.enqueue_copy(output_host, output_buf)
    ctx.synchronize()

    # Compare every row against row 0 element-wise.
    var fail_count = 0
    var max_abs_diff: Float64 = 0.0
    var first_div_row = -1
    var first_div_col = -1
    var first_div_r0_val: Float64 = 0.0
    var first_div_re_val: Float64 = 0.0
    for b in range(1, BATCH):
        for c in range(OUT):
            var v0 = Float64(output_host[0 * OUT + c])
            var vb = Float64(output_host[b * OUT + c])
            var d = abs(v0 - vb)
            if d > max_abs_diff:
                max_abs_diff = d
            if d > tol:
                fail_count += 1
                if first_div_row < 0:
                    first_div_row = b
                    first_div_col = c
                    first_div_r0_val = v0
                    first_div_re_val = vb

    # Print first row's sample for diagnostic context
    print("  row 0 sample (first 4): ", end="")
    for c in range(min(4, OUT)):
        print(Float64(output_host[c]), end=" ")
    print()
    if BATCH > 1:
        print("  row 1 sample (first 4): ", end="")
        for c in range(min(4, OUT)):
            print(Float64(output_host[OUT + c]), end=" ")
        print()

    if fail_count == 0:
        print(
            "  [PASS] all", BATCH - 1,
            "rows match row 0 (max_abs_diff=",
            max_abs_diff,
            ")",
        )
    else:
        print(
            "  [FAIL]",
            fail_count,
            "/",
            (BATCH - 1) * OUT,
            "cells diverge from row 0 (max_abs_diff=",
            max_abs_diff,
            ")",
        )
        print(
            "    first divergence: row",
            first_div_row,
            "col",
            first_div_col,
            ": row_0=",
            first_div_r0_val,
            ", row_e=",
            first_div_re_val,
        )


def main() raises:
    print("=== MuZero Network Row-Symmetry @ BATCH=8 ===")
    print()
    print(
        "Feeds 8 identical input rows through each MuZero net. All"
        " 8 output rows MUST"
    )
    print(
        "be bit-identical (modulo FP reduction noise) since the math is"
        " row-independent."
    )
    print(
        "If row 0 differs from rows 1..7, the GPU forward kernel has a"
        " slot-0 bug."
    )

    var ctx = DeviceContext()

    # CartPole config dims — match the production training agent.
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 128
    comptime HIDDEN = 128
    comptime BINS = 51
    comptime DYN_IN = LATENT + ACT
    comptime PRED_OUT = ACT + BINS

    print("--- Individual layer types ---")
    row_symmetry_check[LinearMish[OBS, HIDDEN], 8](ctx, "LinearMish[4,128]")
    row_symmetry_check[LinearMish[HIDDEN, HIDDEN], 8](
        ctx, "LinearMish[128,128]"
    )
    row_symmetry_check[Linear[HIDDEN, LATENT], 8](ctx, "Linear[128,128]")
    row_symmetry_check[Linear[HIDDEN, ACT], 8](ctx, "Linear[128,2]")
    row_symmetry_check[Linear[HIDDEN, BINS], 8](ctx, "Linear[128,51]")
    row_symmetry_check[MinMaxNorm[LATENT], 8](ctx, "MinMaxNorm[128]")

    print()
    print("--- Composites: Parallel heads ---")
    row_symmetry_check[
        Parallel[Linear[HIDDEN, ACT], Linear[HIDDEN, BINS]], 8
    ](ctx, "Parallel[Linear[128,2], Linear[128,51]] (pred head)")

    print()
    print("--- Full MuZero networks ---")
    comptime RepModel = Sequential[
        LinearMish[OBS, HIDDEN],
        LinearMish[HIDDEN, HIDDEN],
        Linear[HIDDEN, LATENT],
        MinMaxNorm[LATENT],
    ]
    row_symmetry_check[RepModel, 8](ctx, "Full RepModel")

    comptime DynModel = Sequential[
        LinearMish[DYN_IN, HIDDEN],
        LinearMish[HIDDEN, HIDDEN],
        Parallel[
            Sequential[Linear[HIDDEN, LATENT], MinMaxNorm[LATENT]],
            Linear[HIDDEN, BINS],
        ],
    ]
    row_symmetry_check[DynModel, 8](ctx, "Full DynModel")

    comptime PredModel = Sequential[
        LinearMish[LATENT, HIDDEN],
        Parallel[
            Linear[HIDDEN, ACT],
            Linear[HIDDEN, BINS],
        ],
    ]
    row_symmetry_check[PredModel, 8](ctx, "Full PredModel")

    print()
    print("=== Done ===")
    print()
    print(
        "Interpretation: any [FAIL] above with row 0 differing from"
        " rows 1..7"
    )
    print(
        "is the bug source. All [PASS] → the network forward is"
        " symmetric and"
    )
    print(
        "the MCTS row-0 anomaly comes from a buffer (dyn_input,"
        " hidden_states, etc.)"
    )
    print("being read at env-0 slab before being initialized.")
