"""O.1 — OFE composite (GPU) smoke.

GPU-side gate for the new `SkipConcat[Inner]` combinator: build a
single `OFEDenseBlock` (the load-bearing new code path — Linear +
LayerNorm + SiLU under the SkipConcat skip-merge), forward + backward
on device, and verify:

  (1) Skip-path bit-identity: `output[:, 0:IN] == input[:, 0:IN]` on
      device after the SkipConcat forward kernel runs.
  (2) Inner-path is finite (LayerNorm + SiLU produce sane numbers).
  (3) `vjp[mode='all']` and `vjp[mode='input_only']` both run cleanly
      and produce a finite grad_input on device.

The deep `OFEStateBranch6` / `OFEActionBranch6` aren't exercised on
GPU here on purpose — those are 6-deep nested generics, and the Metal
compiler is finicky about that pattern (see
`feedback_metal_nested_generics.md`). They'll be smoked end-to-end
once the trainer wires them up in Phase O.2 on NVIDIA (Apple GPU just
needs to be able to compile the new SkipConcat kernels, which a
single block exercises directly)."""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier

from mojo_rl.deep_agents.redq_ofe.ofe_nets import OFEDenseBlock


comptime BATCH = 4
comptime IN_DIM = 3
comptime PER_UNIT = 2
comptime OUT_DIM = IN_DIM + PER_UNIT


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _is_finite(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Bool:
    for i in range(n):
        if p[i] != p[i]:
            return False
        if _abs(p[i]) > Scalar[DT](1e30):
            return False
    return True


def test_skip_concat_gpu() raises:
    print("--- SkipConcat GPU smoke (single OFEDenseBlock) ---")
    var ctx = DeviceContext()

    seed(42)
    var block = OFEDenseBlock[IN_DIM, PER_UNIT].make[
        target="gpu", INIT=Xavier,
    ](ctx)

    comptime N_X = BATCH * IN_DIM
    comptime N_Y = BATCH * OUT_DIM

    # Host stage.
    var x_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    var y_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var go_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_Y)
    var gi_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_X)
    for i in range(N_X):
        x_h[i] = Scalar[DT](-0.5 + 0.13 * Float64(i))
    for i in range(N_Y):
        go_h[i] = Scalar[DT](0.05 - 0.07 * Float64(i))

    # Device buffers.
    var x_dev = ctx.enqueue_create_buffer[DT](N_X)
    var y_dev = ctx.enqueue_create_buffer[DT](N_Y)
    var go_dev = ctx.enqueue_create_buffer[DT](N_Y)
    var gi_dev = ctx.enqueue_create_buffer[DT](N_X)

    var x_host = ctx.enqueue_create_host_buffer[DT](N_X)
    var go_host = ctx.enqueue_create_host_buffer[DT](N_Y)
    ctx.synchronize()
    for i in range(N_X):
        x_host.unsafe_ptr()[i] = x_h[i]
    for i in range(N_Y):
        go_host.unsafe_ptr()[i] = go_h[i]
    ctx.enqueue_copy(x_dev, x_host)
    ctx.enqueue_copy(go_dev, go_host)

    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = x_dev.unsafe_ptr()
    var y_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = y_dev.unsafe_ptr()
    var go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = go_dev.unsafe_ptr()
    var gi_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = gi_dev.unsafe_ptr()

    var x_t = TileTensor(x_p, row_major[BATCH, IN_DIM]())
    var y_t = TileTensor(y_p, row_major[BATCH, OUT_DIM]())
    var go_t = TileTensor(go_p, row_major[BATCH, OUT_DIM]())
    var gi_t = TileTensor(gi_p, row_major[BATCH, IN_DIM]())

    # ── Forward ────────────────────────────────────────────────────────
    block.forward["gpu", BATCH](x_t, output=y_t)

    var y_host = ctx.enqueue_create_host_buffer[DT](N_Y)
    ctx.enqueue_copy(y_host, y_dev)
    ctx.synchronize()
    for i in range(N_Y):
        y_h[i] = y_host.unsafe_ptr()[i]

    # (1) Skip-path bit-identity on device.
    var max_skip_diff: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(IN_DIM):
            var diff = _abs(y_h[b * OUT_DIM + d] - x_h[b * IN_DIM + d])
            if diff > max_skip_diff:
                max_skip_diff = diff
    print("  fwd skip-path max-diff:", max_skip_diff)
    assert_true(
        max_skip_diff == Scalar[DT](0),
        "GPU SkipConcat must preserve input in first IN columns bit-identically",
    )

    # (2) Inner path finite.
    assert_true(
        _is_finite(y_h + BATCH * IN_DIM, BATCH * PER_UNIT),
        "GPU SkipConcat inner-path output must be finite",
    )

    # ── Backward (mode="all") ──────────────────────────────────────────
    block.zero_grad["gpu"]()
    block.vjp["gpu", BATCH](go_t, gi_t)

    var gi_host = ctx.enqueue_create_host_buffer[DT](N_X)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()
    for i in range(N_X):
        gi_h[i] = gi_host.unsafe_ptr()[i]
    assert_true(
        _is_finite(gi_h, N_X), "GPU grad_input finite (mode='all')",
    )

    # ── Backward (mode="input_only") ───────────────────────────────────
    block.zero_grad["gpu"]()
    block.vjp["gpu", BATCH, mode="input_only"](go_t, gi_t)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()
    for i in range(N_X):
        gi_h[i] = gi_host.unsafe_ptr()[i]
    assert_true(
        _is_finite(gi_h, N_X), "GPU grad_input finite (mode='input_only')",
    )

    x_h.free()
    y_h.free()
    go_h.free()
    gi_h.free()

    print("PASS — GPU SkipConcat (forward skip-identity + both vjp modes).")


def main() raises:
    test_skip_concat_gpu()
    print("=" * 70)
    print("ALL PASS — O.1 OFE composite (GPU)")
    print("=" * 70)
