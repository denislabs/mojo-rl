"""Reusable two-hot CE + decode helpers (storage loss/two_hot.mojo).

Validates the fused two-hot soft-cross-entropy (forward + backward) and the
symexp decode backward that TD-MPC2 (and, on migration, DreamerV3) share:

  1. CE backward vs central finite differences of the CE forward (CPU).
  2. decode backward vs central finite differences of decode_value_batch (CPU).
  3. CPU vs GPU parity for all four (CE fwd/bwd, decode fwd/bwd).

Run:
  pixi run mojo run -I . tests/nn/test_two_hot_ce_storage.mojo
  pixi run -e apple mojo run -I . tests/nn/test_two_hot_ce_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.loss.two_hot import (
    fill_bins,
    decode_value_batch,
    two_hot_ce_loss_batch,
    two_hot_ce_backward_batch,
    decode_value_backward_batch,
    two_hot_ce_fwd_kernel,
    two_hot_ce_bwd_kernel,
    decode_value_fwd_kernel,
    decode_value_bwd_kernel,
)


comptime B = 4
comptime BINS = 7
comptime VMIN = -5
comptime VMAX = 5


def _fill(mut lg: Tensor, mut tg: Tensor) raises:
    for i in range(B * BINS):
        lg.data[i] = Scalar[DT]((i % 5) - 2) * 0.4
    for b in range(B):
        tg.data[b] = Scalar[DT]((b % 4) - 1) * 1.7


def test_ce_backward_fd() raises:
    print("two-hot CE backward vs finite-diff (CPU) ...")
    var lg = Tensor.alloc(B * BINS)
    var tg = Tensor.alloc(B)
    var bins = Tensor.alloc(BINS)
    _fill(lg, tg)
    fill_bins[BINS](Scalar[DT](VMIN), Scalar[DT](VMAX), bins)
    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](1.0)
    var glg = Tensor.alloc(B * BINS)
    two_hot_ce_backward_batch[B, BINS](lg, bins, tg, go, glg)

    var eps = Scalar[DT](1e-3)
    var out_p = Tensor.alloc(B)
    var out_m = Tensor.alloc(B)
    var maxd = Scalar[DT](0)
    for b in range(B):
        for c in range(BINS):
            var idx = b * BINS + c
            var saved = lg.data[idx]
            lg.data[idx] = saved + eps
            two_hot_ce_loss_batch[B, BINS](lg, bins, tg, out_p)
            lg.data[idx] = saved - eps
            two_hot_ce_loss_batch[B, BINS](lg, bins, tg, out_m)
            lg.data[idx] = saved
            var fd = (out_p.data[b] - out_m.data[b]) / (Scalar[DT](2) * eps)
            var d = abs(fd - glg.data[idx])
            if d > maxd:
                maxd = d
    print("  max |fd - analytic| =", maxd)
    assert_true(maxd < Scalar[DT](2e-3), "CE backward FD")
    print("  ok")


def test_decode_backward_fd() raises:
    print("decode backward vs finite-diff (CPU) ...")
    var lg = Tensor.alloc(B * BINS)
    var tg = Tensor.alloc(B)
    var bins = Tensor.alloc(BINS)
    _fill(lg, tg)
    fill_bins[BINS](Scalar[DT](VMIN), Scalar[DT](VMAX), bins)
    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](1.0)
    var glg = Tensor.alloc(B * BINS)
    decode_value_backward_batch[B, BINS](lg, bins, go, glg)

    var eps = Scalar[DT](1e-3)
    var v_p = Tensor.alloc(B)
    var v_m = Tensor.alloc(B)
    var maxd = Scalar[DT](0)
    for b in range(B):
        for c in range(BINS):
            var idx = b * BINS + c
            var saved = lg.data[idx]
            lg.data[idx] = saved + eps
            decode_value_batch[B, BINS](lg, bins, v_p)
            lg.data[idx] = saved - eps
            decode_value_batch[B, BINS](lg, bins, v_m)
            lg.data[idx] = saved
            var fd = (v_p.data[b] - v_m.data[b]) / (Scalar[DT](2) * eps)
            var d = abs(fd - glg.data[idx])
            if d > maxd:
                maxd = d
    print("  max |fd - analytic| =", maxd)
    assert_true(maxd < Scalar[DT](5e-3), "decode backward FD")
    print("  ok")


def test_cpu_gpu_parity() raises:
    print("CPU/GPU parity (CE fwd/bwd, decode fwd/bwd) ...")
    var c = DeviceContext()
    var lg = Tensor.alloc(B * BINS)
    var tg = Tensor.alloc(B)
    var bins = Tensor.alloc(BINS)
    _fill(lg, tg)
    fill_bins[BINS](Scalar[DT](VMIN), Scalar[DT](VMAX), bins)
    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](0.3 + 0.1 * Float64(b))

    # CPU references.
    var ce_cpu = Tensor.alloc(B)
    two_hot_ce_loss_batch[B, BINS](lg, bins, tg, ce_cpu)
    var gce_cpu = Tensor.alloc(B * BINS)
    two_hot_ce_backward_batch[B, BINS](lg, bins, tg, go, gce_cpu)
    var dec_cpu = Tensor.alloc(B)
    decode_value_batch[B, BINS](lg, bins, dec_cpu)
    var gdec_cpu = Tensor.alloc(B * BINS)
    decode_value_backward_batch[B, BINS](lg, bins, go, gdec_cpu)

    # GPU.
    lg.upload(c); tg.upload(c); bins.upload(c); go.upload(c)
    comptime nb = (B + TPB - 1) // TPB
    var ce_g = Tensor.alloc_gpu(c, B)
    c.enqueue_function[two_hot_ce_fwd_kernel[B, BINS, True]](
        lg.lt["gpu", Layout.row_major(B * BINS)](),
        tg.lt["gpu", Layout.row_major(B)](),
        bins.lt["gpu", Layout.row_major(BINS)](),
        ce_g.lt["gpu", Layout.row_major(B)](),
        grid_dim=nb, block_dim=TPB,
    )
    var gce_g = Tensor.alloc_gpu(c, B * BINS)
    c.enqueue_function[two_hot_ce_bwd_kernel[B, BINS, True]](
        go.lt["gpu", Layout.row_major(B)](),
        lg.lt["gpu", Layout.row_major(B * BINS)](),
        tg.lt["gpu", Layout.row_major(B)](),
        bins.lt["gpu", Layout.row_major(BINS)](),
        gce_g.lt["gpu", Layout.row_major(B * BINS)](),
        grid_dim=nb, block_dim=TPB,
    )
    var dec_g = Tensor.alloc_gpu(c, B)
    c.enqueue_function[decode_value_fwd_kernel[B, BINS]](
        lg.lt["gpu", Layout.row_major(B * BINS)](),
        bins.lt["gpu", Layout.row_major(BINS)](),
        dec_g.lt["gpu", Layout.row_major(B)](),
        grid_dim=nb, block_dim=TPB,
    )
    var gdec_g = Tensor.alloc_gpu(c, B * BINS)
    c.enqueue_function[decode_value_bwd_kernel[B, BINS]](
        go.lt["gpu", Layout.row_major(B)](),
        lg.lt["gpu", Layout.row_major(B * BINS)](),
        bins.lt["gpu", Layout.row_major(BINS)](),
        gdec_g.lt["gpu", Layout.row_major(B * BINS)](),
        grid_dim=nb, block_dim=TPB,
    )
    ce_g.download(c); gce_g.download(c); dec_g.download(c); gdec_g.download(c)

    var m = Scalar[DT](0)
    for b in range(B):
        if abs(ce_g.data[b] - ce_cpu.data[b]) > m: m = abs(ce_g.data[b] - ce_cpu.data[b])
        if abs(dec_g.data[b] - dec_cpu.data[b]) > m: m = abs(dec_g.data[b] - dec_cpu.data[b])
    for i in range(B * BINS):
        if abs(gce_g.data[i] - gce_cpu.data[i]) > m: m = abs(gce_g.data[i] - gce_cpu.data[i])
        if abs(gdec_g.data[i] - gdec_cpu.data[i]) > m: m = abs(gdec_g.data[i] - gdec_cpu.data[i])
    print("  max CPU/GPU Δ =", m)
    assert_true(m < Scalar[DT](2e-5), "CPU/GPU parity")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("Reusable two-hot CE + decode helpers")
    print("=" * 60)
    test_ce_backward_fd()
    test_decode_backward_fd()
    test_cpu_gpu_parity()
    print("ALL PASSED")
