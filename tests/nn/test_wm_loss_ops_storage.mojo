"""DreamerV3 WM loss ops on the storage ABI — gates the legacy→storage port of
`mojo_rl/deep_agents/dreamerv3/wm_loss_ops.mojo`.

For each of SymlogMSELoss[OBS] / TwoHotLoss[BINS] / BinaryLoss:
  1. Finite-difference grad check on CPU (vjp vs central-difference of Σ out).
  2. CPU vs GPU parity (forward + vjp), max abs diff < 1e-4.

The two operands of a `TensorRefs[2]` must share one origin, so the input /
grad-input pairs are backed by a `TensorPack[2]` (whose subscript returns a
shared `MutAnyOrigin` ref) — mirroring `test_two_hot_ce_storage.mojo`.

Run:
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . \
      tests/nn/test_wm_loss_ops_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Zero
from mojo_rl.deep_agents.dreamerv3.wm_loss_ops import (
    SymlogMSELoss,
    TwoHotLoss,
    BinaryLoss,
)


comptime OBS = 4
comptime BINS = 7
comptime B = 5


# ──────────────────────────────────────────────────────────────────────
# SymlogMSELoss[OBS] — inputs (pred[B*OBS], target[B*OBS]).
# ──────────────────────────────────────────────────────────────────────


def test_symmse() raises:
    print("SymlogMSELoss[OBS] ...")
    var inp = TensorPack[2]()
    inp[0].ensure(B * OBS)
    inp[1].ensure(B * OBS)
    for i in range(B * OBS):
        inp[0].data[i] = Scalar[DT]((i % 5) - 2) * 0.3
        inp[1].data[i] = Scalar[DT]((i % 3) - 1) * 0.7

    var op = SymlogMSELoss[OBS].make["cpu", Zero]()
    var out = Tensor.alloc(B)
    op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out)

    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](1.0)
    var gpk = TensorPack[2]()
    op.vjp["cpu", B](
        TensorRefs[2](inp[0], inp[1]), go, TensorRefs[2](gpk[0], gpk[1])
    )

    # FD: scalar loss = Σ out; d/d pred[idx].
    var eps = Scalar[DT](1e-3)
    var out_p = Tensor.alloc(B)
    var out_m = Tensor.alloc(B)
    var maxd = Scalar[DT](0)
    for idx in range(B * OBS):
        var saved = inp[0].data[idx]
        inp[0].data[idx] = saved + eps
        op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out_p)
        inp[0].data[idx] = saved - eps
        op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out_m)
        inp[0].data[idx] = saved
        var sp = Scalar[DT](0)
        var sm = Scalar[DT](0)
        for b in range(B):
            sp += out_p.data[b]
            sm += out_m.data[b]
        var fd = (sp - sm) / (Scalar[DT](2) * eps)
        var d = abs(fd - gpk[0].data[idx])
        if d > maxd:
            maxd = d
    print("  max |fd - analytic| =", maxd)
    assert_true(maxd < Scalar[DT](2e-3), "SymlogMSE FD")

    # CPU/GPU parity.
    var c = DeviceContext()
    var gop = SymlogMSELoss[OBS].make["gpu", Zero](Optional(c))
    var ing = TensorPack[2]()
    ing[0].ensure(B * OBS)
    ing[1].ensure(B * OBS)
    for i in range(B * OBS):
        ing[0].data[i] = inp[0].data[i]
        ing[1].data[i] = inp[1].data[i]
    ing[0].upload(c); ing[1].upload(c)
    var outg = Tensor.alloc_gpu(c, B)
    gop.forward["gpu", B](TensorRefs[2](ing[0], ing[1]), outg, Optional(c))
    var gog = Tensor.alloc(B)
    for b in range(B):
        gog.data[b] = Scalar[DT](1.0)
    gog.upload(c)
    var ggk = TensorPack[2]()
    ggk[0].ensure_gpu(c, B * OBS)
    ggk[1].ensure_gpu(c, B * OBS)
    gop.vjp["gpu", B](
        TensorRefs[2](ing[0], ing[1]), gog, TensorRefs[2](ggk[0], ggk[1]),
        Optional(c),
    )
    outg.download(c); ggk[0].download(c)
    var m = Scalar[DT](0)
    for b in range(B):
        if abs(outg.data[b] - out.data[b]) > m: m = abs(outg.data[b] - out.data[b])
    for i in range(B * OBS):
        if abs(ggk[0].data[i] - gpk[0].data[i]) > m: m = abs(ggk[0].data[i] - gpk[0].data[i])
    print("  max CPU/GPU Δ =", m)
    assert_true(m < Scalar[DT](1e-4), "SymlogMSE parity")
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# TwoHotLoss[BINS] — inputs (logits[B*BINS], target[B]).
# ──────────────────────────────────────────────────────────────────────


def test_twohot() raises:
    print("TwoHotLoss[BINS] ...")
    var inp = TensorPack[2]()
    inp[0].ensure(B * BINS)
    inp[1].ensure(B)
    for i in range(B * BINS):
        inp[0].data[i] = Scalar[DT]((i % 5) - 2) * 0.4
    for b in range(B):
        inp[1].data[b] = Scalar[DT]((b % 4) - 1) * 1.7

    var op = TwoHotLoss[BINS].make["cpu", Zero]()
    var out = Tensor.alloc(B)
    op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out)

    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](1.0)
    var gpk = TensorPack[2]()
    op.vjp["cpu", B](
        TensorRefs[2](inp[0], inp[1]), go, TensorRefs[2](gpk[0], gpk[1])
    )

    # FD on logits.
    var eps = Scalar[DT](1e-3)
    var out_p = Tensor.alloc(B)
    var out_m = Tensor.alloc(B)
    var maxd = Scalar[DT](0)
    for idx in range(B * BINS):
        var saved = inp[0].data[idx]
        inp[0].data[idx] = saved + eps
        op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out_p)
        inp[0].data[idx] = saved - eps
        op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out_m)
        inp[0].data[idx] = saved
        var sp = Scalar[DT](0)
        var sm = Scalar[DT](0)
        for b in range(B):
            sp += out_p.data[b]
            sm += out_m.data[b]
        var fd = (sp - sm) / (Scalar[DT](2) * eps)
        var d = abs(fd - gpk[0].data[idx])
        if d > maxd:
            maxd = d
    print("  max |fd - analytic| =", maxd)
    assert_true(maxd < Scalar[DT](2e-3), "TwoHot FD")

    # CPU/GPU parity.
    var c = DeviceContext()
    var gop = TwoHotLoss[BINS].make["gpu", Zero](Optional(c))
    var ing = TensorPack[2]()
    ing[0].ensure(B * BINS)
    ing[1].ensure(B)
    for i in range(B * BINS):
        ing[0].data[i] = inp[0].data[i]
    for b in range(B):
        ing[1].data[b] = inp[1].data[b]
    ing[0].upload(c); ing[1].upload(c)
    var outg = Tensor.alloc_gpu(c, B)
    gop.forward["gpu", B](TensorRefs[2](ing[0], ing[1]), outg, Optional(c))
    var gog = Tensor.alloc(B)
    for b in range(B):
        gog.data[b] = Scalar[DT](1.0)
    gog.upload(c)
    var ggk = TensorPack[2]()
    ggk[0].ensure_gpu(c, B * BINS)
    ggk[1].ensure_gpu(c, B)
    gop.vjp["gpu", B](
        TensorRefs[2](ing[0], ing[1]), gog, TensorRefs[2](ggk[0], ggk[1]),
        Optional(c),
    )
    outg.download(c); ggk[0].download(c)
    var m = Scalar[DT](0)
    for b in range(B):
        if abs(outg.data[b] - out.data[b]) > m: m = abs(outg.data[b] - out.data[b])
    for i in range(B * BINS):
        if abs(ggk[0].data[i] - gpk[0].data[i]) > m: m = abs(ggk[0].data[i] - gpk[0].data[i])
    print("  max CPU/GPU Δ =", m)
    assert_true(m < Scalar[DT](1e-4), "TwoHot parity")
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# BinaryLoss — inputs (logit[B], target[B]).
# ──────────────────────────────────────────────────────────────────────


def test_binary() raises:
    print("BinaryLoss ...")
    var inp = TensorPack[2]()
    inp[0].ensure(B)
    inp[1].ensure(B)
    for b in range(B):
        inp[0].data[b] = Scalar[DT]((b % 5) - 2) * 0.6
        inp[1].data[b] = Scalar[DT](b % 2)

    var op = BinaryLoss.make["cpu", Zero]()
    var out = Tensor.alloc(B)
    op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out)

    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](1.0)
    var gpk = TensorPack[2]()
    op.vjp["cpu", B](
        TensorRefs[2](inp[0], inp[1]), go, TensorRefs[2](gpk[0], gpk[1])
    )

    var eps = Scalar[DT](1e-3)
    var out_p = Tensor.alloc(B)
    var out_m = Tensor.alloc(B)
    var maxd = Scalar[DT](0)
    for idx in range(B):
        var saved = inp[0].data[idx]
        inp[0].data[idx] = saved + eps
        op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out_p)
        inp[0].data[idx] = saved - eps
        op.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out_m)
        inp[0].data[idx] = saved
        var sp = Scalar[DT](0)
        var sm = Scalar[DT](0)
        for b in range(B):
            sp += out_p.data[b]
            sm += out_m.data[b]
        var fd = (sp - sm) / (Scalar[DT](2) * eps)
        var d = abs(fd - gpk[0].data[idx])
        if d > maxd:
            maxd = d
    print("  max |fd - analytic| =", maxd)
    assert_true(maxd < Scalar[DT](2e-3), "Binary FD")

    # CPU/GPU parity.
    var c = DeviceContext()
    var gop = BinaryLoss.make["gpu", Zero](Optional(c))
    var ing = TensorPack[2]()
    ing[0].ensure(B)
    ing[1].ensure(B)
    for b in range(B):
        ing[0].data[b] = inp[0].data[b]
        ing[1].data[b] = inp[1].data[b]
    ing[0].upload(c); ing[1].upload(c)
    var outg = Tensor.alloc_gpu(c, B)
    gop.forward["gpu", B](TensorRefs[2](ing[0], ing[1]), outg, Optional(c))
    var gog = Tensor.alloc(B)
    for b in range(B):
        gog.data[b] = Scalar[DT](1.0)
    gog.upload(c)
    var ggk = TensorPack[2]()
    ggk[0].ensure_gpu(c, B)
    ggk[1].ensure_gpu(c, B)
    gop.vjp["gpu", B](
        TensorRefs[2](ing[0], ing[1]), gog, TensorRefs[2](ggk[0], ggk[1]),
        Optional(c),
    )
    outg.download(c); ggk[0].download(c)
    var m = Scalar[DT](0)
    for b in range(B):
        if abs(outg.data[b] - out.data[b]) > m: m = abs(outg.data[b] - out.data[b])
        if abs(ggk[0].data[b] - gpk[0].data[b]) > m: m = abs(ggk[0].data[b] - gpk[0].data[b])
    print("  max CPU/GPU Δ =", m)
    assert_true(m < Scalar[DT](1e-4), "Binary parity")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("DreamerV3 WM loss ops (storage ABI)")
    print("=" * 60)
    test_symmse()
    test_twohot()
    test_binary()
    print("ALL PASSED")
