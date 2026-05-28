"""Box-Muller transform — fill a buffer with i.i.d. N(0, 1) samples.

Box-Muller transform: given two independent U(0, 1) samples u1, u2,
    z = sqrt(-2 ln u1) · cos(2π u2)
is N(0, 1). We clamp u1 ≥ 1e-10 to avoid log(0).

The PPO and SAC examples both used a copy of this in-file; Phase 8.1
extracts it to nn2/random/. Phase 7 used `std.random.random_float64`
as the entropy source — same here. RNG seeding is the caller's
responsibility (`from std.random import seed`).

Block D (2026-05-21): adds `box_muller_normal_gpu` — philox-based GPU
variant. Same `N(0, 1)` output, pair-indexed cos/sin branches per
element. Pattern lifted from `mojo_rl/experimental/pcn/pc_trainer.mojo`
to keep API identical between v1 and nn2 callers.
"""

from std.math import cos as fcos, log as flog, sin as fsin, sqrt as fsqrt, pi
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from ..constants import DT, TPB


def box_muller_normal(out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int):
    """Fill out_ptr[0:n] with iid N(0, 1) samples via Box-Muller.

    Each lane is drawn independently — no batched-pair optimization (the
    classic Box-Muller produces *two* normals per cos/sin pair; we only
    keep the cos branch for simplicity since correlations don't matter
    here). If you need the second normal, call again — RNG advances.
    """
    for i in range(n):
        var u1 = random_float64()
        if u1 < 1e-10:
            u1 = 1e-10
        var u2 = random_float64()
        out_ptr[i] = fsqrt(Scalar[DT](-2.0) * flog(Scalar[DT](u1))) * fcos(
            Scalar[DT](2.0 * pi) * Scalar[DT](u2)
        )


def _box_muller_kernel[N: Int](
    noise: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    seed: UInt64,
    offset_base: UInt64,
):
    """One thread per output element. Pair-indexed cos/sin branches via
    Philox at offsets (offset_base + 2·pair_idx) and (... + 1).

    Float32 internally — Philox produces Float32 uniforms; the math is
    fine at that precision for SAC noise samples (the sample variance
    matters far more than per-sample bit fidelity)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N:
        return
    var pair_idx = UInt64(idx // 2)
    var which_in_pair = idx % 2

    var rng1 = PhiloxRandom(seed=seed, offset=offset_base + pair_idx * 2)
    var rng2 = PhiloxRandom(seed=seed, offset=offset_base + pair_idx * 2 + 1)
    var u1 = Float32(rng1.step_uniform()[0])
    var u2 = Float32(rng2.step_uniform()[0])
    if u1 < Float32(1e-7):
        u1 = Float32(1e-7)
    var r = fsqrt(Float32(-2.0) * flog(u1))
    var two_pi_u2 = Float32(6.283185307179586) * u2
    var z: Float32 = (
        r * fcos(two_pi_u2) if which_in_pair == 0 else r * fsin(two_pi_u2)
    )
    noise[idx] = Scalar[DT](z)


def box_muller_normal_gpu[N: Int](
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    seed: UInt64,
    offset_base: UInt64,
) raises:
    """Fill out_ptr[0:N] with iid N(0, 1) samples on GPU via Philox+BM.

    `seed` + `offset_base` together determine the sequence — callers
    advance `offset_base` by `2·ceil(N/2)` (or equivalently `N` rounded
    up to even) between calls so successive calls draw fresh samples
    without reuse. The Philox stream is deterministic for a given
    (seed, offset_base) pair.
    """
    var noise_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](out_ptr)
    comptime n_blocks = (N + TPB - 1) // TPB
    comptime kernel = _box_muller_kernel[N]
    ctx.enqueue_function[kernel](
        noise_lt, seed, offset_base,
        grid_dim=n_blocks, block_dim=TPB,
    )
