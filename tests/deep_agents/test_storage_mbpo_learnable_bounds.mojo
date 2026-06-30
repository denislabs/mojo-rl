"""MBPO learnable logvar bounds — storage DynamicsEnsembleBlock (GPU + CPU).

Targeted gate for the learnable-bounds path swept off `unsafe_ptr` onto `lt_at` /
`create_sub_buffer` (the soft-clamp NLL grad + the per-member bounds Adam step).
Builds a small ensemble, enables learnable bounds, trains a member on synthetic
data, and asserts:
  - the bounds move OFF their inits (+0.5 / -10) — the soft-clamp grad + L2 Adam
    update actually fires (this is the per-member bounds sub-view path),
  - losses + `predict_member` logvars stay finite + within the soft-clamp band,
  - the fixed-clamp default path still runs.
Both GPU (the sub-view/create_sub_buffer surface) and CPU (the `.data` path).

Ported from the legacy `test_mbpo_learnable_bounds` (legacy nets → storage; raw
TileTensor/DeviceBuffer args → storage `Tensor`).

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_mbpo_learnable_bounds.mojo
"""

from std.math import isfinite
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Swish
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.mbpo.dynamics_ensemble_block import (
    DynamicsEnsembleBlock,
)

comptime OBS = 4
comptime ACT = 2
comptime IN_DIM = OBS + ACT          # 6
comptime PRED = 1 + OBS              # 5
comptime OUT = 2 * PRED              # 10
comptime N = 3
comptime ELITES = 2
comptime BATCH = 8
comptime HID = 16

comptime DynNet = Sequential[
    Linear[IN_DIM, HID], Swish[HID], Linear[HID, OUT]
]
comptime Block = DynamicsEnsembleBlock[
    DynNet, N, ELITES, IN_DIM, OUT, BATCH
]


def _fill(mut in_t: Tensor, mut tgt_t: Tensor):
    for i in range(BATCH * IN_DIM):
        in_t.data[i] = Scalar[DT](0.1) * Scalar[DT]((i % 7) - 3)
    for i in range(BATCH * PRED):
        tgt_t.data[i] = Scalar[DT](0.05) * Scalar[DT]((i % 5) - 2)


def main() raises:
    print("=" * 60)
    print("MBPO learnable logvar bounds — storage (GPU + CPU)")
    print("=" * 60)

    with DeviceContext() as ctx:
        var blk = Block.make["gpu", Kaiming](ctx)
        blk.set_lr(Scalar[DT](1e-3))
        blk.enable_learnable_bounds()

        var in_t = Tensor.alloc(BATCH * IN_DIM)
        var tgt_t = Tensor.alloc(BATCH * PRED)
        _fill(in_t, tgt_t)
        in_t.upload(ctx)
        tgt_t.upload(ctx)

        # Bounds BEFORE (member 0): expect +0.5 / -10.
        blk._max_lv.download(ctx)
        blk._min_lv.download(ctx)
        var max0_before = blk._max_lv.data[0]
        var min0_before = blk._min_lv.data[0]
        print("before: max_lv[0] =", max0_before, " min_lv[0] =", min0_before)
        if abs(max0_before - Scalar[DT](0.5)) > Scalar[DT](1e-5):
            raise Error("max_lv not initialised to +0.5")
        if abs(min0_before - Scalar[DT](-10.0)) > Scalar[DT](1e-5):
            raise Error("min_lv not initialised to -10")

        var last_loss = Scalar[DT](0.0)
        for _ in range(60):
            last_loss = blk.train_member_step["gpu"](0, in_t, tgt_t)
        print("member 0 final train loss =", last_loss)
        if not isfinite(last_loss):
            raise Error("train loss is non-finite")

        # Bounds AFTER: max moves DOWN, min UP (L2 pulls them together).
        blk._max_lv.download(ctx)
        blk._min_lv.download(ctx)
        var max0_after = blk._max_lv.data[0]
        var min0_after = blk._min_lv.data[0]
        print("after:  max_lv[0] =", max0_after, " min_lv[0] =", min0_after)
        if abs(max0_after - max0_before) < Scalar[DT](1e-4):
            raise Error("max_lv did not move — bound update never fired")
        if abs(min0_after - min0_before) < Scalar[DT](1e-4):
            raise Error("min_lv did not move — bound update never fired")
        if max0_after >= max0_before:
            raise Error("max_lv should have decreased (L2 pulls it down)")
        if min0_after <= min0_before:
            raise Error("min_lv should have increased (L2 pushes it up)")

        # predict_member finite (mu, lv) with the soft clamp.
        var mu_t = Tensor.alloc_gpu(ctx, BATCH * PRED)
        var lv_t = Tensor.alloc_gpu(ctx, BATCH * PRED)
        blk.predict_member["gpu"](0, in_t, mu_t, lv_t)
        lv_t.download(ctx)
        for i in range(BATCH * PRED):
            var v = lv_t.data[i]
            if not isfinite(v):
                raise Error("predict_member produced non-finite logvar")
            if v > Scalar[DT](1.0) or v < Scalar[DT](-11.0):
                raise Error("clamped logvar out of expected band")

        var hl = blk.eval_member_loss["gpu"](0, in_t, tgt_t)
        print("member 0 holdout loss =", hl)
        if not isfinite(hl):
            raise Error("eval loss non-finite")

        # Fixed-clamp default path still runs (learnable off).
        var blk2 = Block.make["gpu", Kaiming](ctx)
        blk2.set_lr(Scalar[DT](1e-3))
        var l2 = blk2.train_member_step["gpu"](0, in_t, tgt_t)
        if not isfinite(l2):
            raise Error("fixed-clamp path loss non-finite")
        print("fixed-clamp path loss =", l2)

    # ── CPU path (mirror) — exercises the `.data` bounds branch. ──────────
    print("-" * 60)
    print("CPU path")
    var cblk = Block.make["cpu", Kaiming]()
    cblk.set_lr(Scalar[DT](1e-3))
    cblk.enable_learnable_bounds()
    var cin = Tensor.alloc(BATCH * IN_DIM)
    var ctg = Tensor.alloc(BATCH * PRED)
    _fill(cin, ctg)
    var cmax_before = cblk._max_lv.data[0]
    var cmin_before = cblk._min_lv.data[0]
    var closs = Scalar[DT](0.0)
    for _ in range(60):
        closs = cblk.train_member_step["cpu"](0, cin, ctg)
    var cmax_after = cblk._max_lv.data[0]
    var cmin_after = cblk._min_lv.data[0]
    print("cpu loss =", closs, " max:", cmax_before, "->", cmax_after,
          " min:", cmin_before, "->", cmin_after)
    if not isfinite(closs):
        raise Error("cpu train loss non-finite")
    if cmax_after >= cmax_before or cmin_after <= cmin_before:
        raise Error("cpu bounds did not tighten")

    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
