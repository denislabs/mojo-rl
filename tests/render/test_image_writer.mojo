"""Smoke test for image_writer PPM output."""

from mojo_rl.render.image_writer import (
    save_ppm,
    save_reconstruction_grid,
    save_image_row,
    save_vector_heatmap,
    save_vector_comparison,
)
from std.math import sin, cos


def test_save_ppm() raises:
    """Write a 64x64 grayscale gradient."""
    var n = 64 * 64
    var data = alloc[Scalar[DType.float32]](n)
    for y in range(64):
        for x in range(64):
            (data + y * 64 + x)[] = Float32(x) / 63.0
    save_ppm("test_gradient.ppm", data, 64, 64, channels=1, vmin=0.0, vmax=1.0)
    data.free()
    print("PASS test_save_ppm")


def test_save_ppm_rgb() raises:
    """Write a 64x64 RGB image (CHW layout)."""
    var n = 3 * 64 * 64
    var data = alloc[Scalar[DType.float32]](n)
    var hw = 64 * 64
    for y in range(64):
        for x in range(64):
            var idx = y * 64 + x
            (data + idx)[] = Float32(x) / 63.0           # R
            (data + hw + idx)[] = Float32(y) / 63.0       # G
            (data + 2 * hw + idx)[] = 0.5                 # B
    save_ppm("test_rgb.ppm", data, 64, 64, channels=3, vmin=0.0, vmax=1.0)
    data.free()
    print("PASS test_save_ppm_rgb")


def test_reconstruction_grid() raises:
    """Write a grid of 4 original/reconstructed 16x16 grayscale pairs."""
    var n_pairs = 4
    var h = 16
    var w = 16
    var img_size = h * w
    var originals = alloc[Scalar[DType.float32]](n_pairs * img_size)
    var reconstructions = alloc[Scalar[DType.float32]](n_pairs * img_size)
    for i in range(n_pairs):
        for y in range(h):
            for x in range(w):
                var idx = i * img_size + y * w + x
                var val = Float32(x + y * 2) / Float32(w + 2 * h)
                (originals + idx)[] = val
                (reconstructions + idx)[] = val * 0.8 + 0.1  # slightly shifted
    save_reconstruction_grid(
        "test_recon_grid.ppm", originals, reconstructions,
        n=n_pairs, height=h, width=w, channels=1,
    )
    originals.free()
    reconstructions.free()
    print("PASS test_reconstruction_grid")


def test_vector_heatmap() raises:
    """Write a heatmap of 8 vectors with dim=17 (like HalfCheetah obs)."""
    var n = 8
    var dim = 17
    var data = alloc[Scalar[DType.float32]](n * dim)
    for i in range(n):
        for d in range(dim):
            (data + i * dim + d)[] = sin(Float32(i * dim + d) * 0.3)
    save_vector_heatmap("test_heatmap.ppm", data, n_rows=n, dim=dim)
    data.free()
    print("PASS test_vector_heatmap")


def test_vector_comparison() raises:
    """Write interleaved orig/recon vector pairs."""
    var n = 4
    var dim = 10
    var originals = alloc[Scalar[DType.float32]](n * dim)
    var reconstructions = alloc[Scalar[DType.float32]](n * dim)
    for i in range(n):
        for d in range(dim):
            var val = sin(Float32(i * dim + d) * 0.5)
            (originals + i * dim + d)[] = val
            (reconstructions + i * dim + d)[] = val + 0.15
    save_vector_comparison(
        "test_vec_compare.ppm", originals, reconstructions,
        n=n, dim=dim,
    )
    originals.free()
    reconstructions.free()
    print("PASS test_vector_comparison")


def test_image_row_digits() raises:
    """Simulate PCN bidirectional digit generation: 10 fake 28x28 images with labels."""
    var n = 10
    var h = 28
    var w = 28
    var img_size = h * w
    var data = alloc[Scalar[DType.float32]](n * img_size)
    for i in range(n):
        for y in range(h):
            for x in range(w):
                # Create a distinctive pattern per "digit"
                var cx = Float32(x) - 14.0
                var cy = Float32(y) - 14.0
                var r2 = cx * cx + cy * cy
                var phase = Float32(i) * 0.7
                var v = 0.5 + 0.5 * sin(r2 * 0.02 + phase) * cos(cx * 0.2 + phase)
                (data + i * img_size + y * w + x)[] = v
    var digit_labels = List[String]()
    for i in range(10):
        digit_labels.append(String(i))
    save_image_row(
        "test_pcn_digits.ppm", data,
        n=n, height=h, width=w, channels=1,
        vmin=0.0, vmax=1.0, pixel_scale=4, labels=digit_labels,
    )
    data.free()
    print("PASS test_image_row_digits")


def main() raises:
    test_save_ppm()
    test_save_ppm_rgb()
    test_reconstruction_grid()
    test_vector_heatmap()
    test_vector_comparison()
    test_image_row_digits()
    print("All image_writer tests passed")
