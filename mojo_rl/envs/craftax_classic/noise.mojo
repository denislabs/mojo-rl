"""Perlin / fractal noise for Craftax-Classic world generation.

Ports `references/Craftax-main/craftax/craftax_classic/util/noise.py`.

Algorithm: standard 2D Perlin noise on a (RES_H+1) × (RES_W+1) grid of
random gradient angles, with bilinear blend and quintic ease curve.
Craftax always uses `octaves=1`, so "fractal" reduces to a single Perlin
field followed by per-map min–max normalization to [0, 1].

The min–max normalization is *per realization* (per env), which means
threshold values like 0.7 (water) are relative to the noise range — not
absolute Perlin values. Important for parity with the reference.

Sqrt(2) is dropped: it's a constant pre-factor that the subsequent min–max
normalization removes.
"""

from std.math import sin, cos
from std.random.philox import Random as PhiloxRandom


comptime TWO_PI: Float32 = 6.2831853


@always_inline
def quintic_ease(t: Float32) -> Float32:
    """6t^5 − 15t^4 + 10t^3 ease curve used by Perlin."""
    return (
        t * t * t * (t * (t * Float32(6.0) - Float32(15.0)) + Float32(10.0))
    )


@always_inline
def generate_perlin_noise_2d[
    H: Int, W: Int, RES_H: Int, RES_W: Int
](
    mut rng: PhiloxRandom,
    out_data: UnsafePointer[Float32, MutAnyOrigin],
):
    """Generate a single-octave Perlin noise field of shape (H, W).

    Writes into `out_data[y * W + x]` for all (y, x). Values are NOT
    normalized — call `normalize_inplace` afterwards.
    """
    comptime NUM_ANGLES: Int = (RES_H + 1) * (RES_W + 1)
    comptime D_H: Int = H // RES_H
    comptime D_W: Int = W // RES_W
    comptime INV_D_H: Float32 = Float32(1.0) / Float32(D_H)
    comptime INV_D_W: Float32 = Float32(1.0) / Float32(D_W)
    comptime STRIDE: Int = RES_W + 1

    # Stack-allocate gradient angles, drawn from `rng`.
    var angles = InlineArray[Float32, NUM_ANGLES](fill=Float32(0.0))
    var i = 0
    while i < NUM_ANGLES:
        var u = rng.step_uniform()
        for k in range(4):
            if i < NUM_ANGLES:
                angles[i] = Float32(u[k]) * TWO_PI
                i += 1

    # Compute one Perlin value per tile.
    for y in range(H):
        var cy = y // D_H
        var fy = Float32(y - cy * D_H) * INV_D_H
        var ty = quintic_ease(fy)
        var one_minus_ty = Float32(1.0) - ty
        var fy_m1 = fy - Float32(1.0)
        for x in range(W):
            var cx = x // D_W
            var fx = Float32(x - cx * D_W) * INV_D_W
            var tx = quintic_ease(fx)
            var one_minus_tx = Float32(1.0) - tx
            var fx_m1 = fx - Float32(1.0)

            var a00 = angles[cy * STRIDE + cx]
            var a10 = angles[(cy + 1) * STRIDE + cx]
            var a01 = angles[cy * STRIDE + (cx + 1)]
            var a11 = angles[(cy + 1) * STRIDE + (cx + 1)]

            var n00 = fy * cos(a00) + fx * sin(a00)
            var n10 = fy_m1 * cos(a10) + fx * sin(a10)
            var n01 = fy * cos(a01) + fx_m1 * sin(a01)
            var n11 = fy_m1 * cos(a11) + fx_m1 * sin(a11)

            var n0 = n00 * one_minus_ty + n10 * ty
            var n1 = n01 * one_minus_ty + n11 * ty
            out_data[y * W + x] = n0 * one_minus_tx + n1 * tx


@always_inline
def normalize_inplace[
    H: Int, W: Int
](data: UnsafePointer[Float32, MutAnyOrigin]):
    """Min–max normalize `data` (length H*W) in place to [0, 1].

    Matches reference: `(x - min) / (max - min)`. If max == min, leaves
    values unchanged (degenerate case).
    """
    comptime N: Int = H * W
    var min_v = data[0]
    var max_v = data[0]
    for i in range(1, N):
        var v = data[i]
        if v < min_v:
            min_v = v
        if v > max_v:
            max_v = v
    var range_v = max_v - min_v
    if range_v <= Float32(1e-20):
        return
    var inv = Float32(1.0) / range_v
    for i in range(N):
        data[i] = (data[i] - min_v) * inv


@always_inline
def generate_fractal_noise_2d_normalized[
    H: Int, W: Int, RES_H: Int, RES_W: Int
](
    mut rng: PhiloxRandom,
    out_data: UnsafePointer[Float32, MutAnyOrigin],
):
    """Single-octave fractal noise (= Perlin) + min–max normalize to [0, 1]."""
    generate_perlin_noise_2d[H, W, RES_H, RES_W](rng, out_data)
    normalize_inplace[H, W](out_data)
