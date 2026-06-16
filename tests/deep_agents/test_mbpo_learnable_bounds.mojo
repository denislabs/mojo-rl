"""Targeted smoke for MBPO learnable logvar bounds (GPU).

Builds a small DynamicsEnsembleBlock, enables learnable bounds, runs a few
train steps on synthetic data, and asserts:
  - the bounds move OFF their inits (+0.5 / -10) — i.e. the soft-clamp grad +
    0.01-L2 Adam update actually fires,
  - losses + predict_member outputs stay finite,
  - the fixed-clamp path (learnable off) still runs (bit-path unchanged).
"""

from std.math import isfinite
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
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
    Linear[IN_DIM, HID],
    Elementwise[HID, SwishOp],
    Linear[HID, OUT],
]

comptime Block = DynamicsEnsembleBlock[
    DynNet, N, ELITES, IN_DIM, OUT, BATCH
]


def main() raises:
    print("=" * 60)
    print("MBPO learnable logvar bounds — GPU smoke")
    print("=" * 60)
    var ctx = DeviceContext()

    var blk = Block.make["gpu", Kaiming](ctx)
    blk.set_lr(Scalar[DT](1e-3))
    blk.enable_learnable_bounds()

    # Synthetic input/target buffers, filled with a deterministic pattern.
    var in_dev = ctx.enqueue_create_buffer[DT](BATCH * IN_DIM)
    var tgt_dev = ctx.enqueue_create_buffer[DT](BATCH * PRED)
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * IN_DIM)
    var tgt_host = ctx.enqueue_create_host_buffer[DT](BATCH * PRED)
    ctx.synchronize()
    for i in range(BATCH * IN_DIM):
        in_host.unsafe_ptr()[i] = Scalar[DT](0.1) * Scalar[DT]((i % 7) - 3)
    for i in range(BATCH * PRED):
        tgt_host.unsafe_ptr()[i] = Scalar[DT](0.05) * Scalar[DT]((i % 5) - 2)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.enqueue_copy(tgt_dev, tgt_host)
    ctx.synchronize()

    var in_t = TileTensor(in_dev.unsafe_ptr(), row_major[BATCH, IN_DIM]())
    var tgt_t = TileTensor(tgt_dev.unsafe_ptr(), row_major[BATCH, PRED]())

    # Read bounds BEFORE training (member 0): expect +0.5 / -10.
    var bnd_host = ctx.enqueue_create_host_buffer[DT](N * PRED)
    ctx.enqueue_copy(bnd_host, blk._max_lv.dev.value())
    ctx.synchronize()
    var max0_before = bnd_host.unsafe_ptr()[0]
    ctx.enqueue_copy(bnd_host, blk._min_lv.dev.value())
    ctx.synchronize()
    var min0_before = bnd_host.unsafe_ptr()[0]
    print("before: max_lv[0] =", max0_before, " min_lv[0] =", min0_before)
    if abs(max0_before - Scalar[DT](0.5)) > Scalar[DT](1e-5):
        raise Error("max_lv not initialised to +0.5")
    if abs(min0_before - Scalar[DT](-10.0)) > Scalar[DT](1e-5):
        raise Error("min_lv not initialised to -10")

    # Train member 0 for a bunch of steps.
    var last_loss = Scalar[DT](0.0)
    for _ in range(60):
        last_loss = blk.train_member_step["gpu"](0, in_t, tgt_t)
    print("member 0 final train loss =", last_loss)
    if not isfinite(last_loss):
        raise Error("train loss is non-finite")

    # Read bounds AFTER: max should have moved DOWN, min UP (L2 pulls them
    # toward each other; data grad also shapes them).
    ctx.enqueue_copy(bnd_host, blk._max_lv.dev.value())
    ctx.synchronize()
    var max0_after = bnd_host.unsafe_ptr()[0]
    ctx.enqueue_copy(bnd_host, blk._min_lv.dev.value())
    ctx.synchronize()
    var min0_after = bnd_host.unsafe_ptr()[0]
    print("after:  max_lv[0] =", max0_after, " min_lv[0] =", min0_after)
    if abs(max0_after - max0_before) < Scalar[DT](1e-4):
        raise Error("max_lv did not move — bound update never fired")
    if abs(min0_after - min0_before) < Scalar[DT](1e-4):
        raise Error("min_lv did not move — bound update never fired")
    if max0_after >= max0_before:
        raise Error("max_lv should have decreased (L2 pulls it down)")
    if min0_after <= min0_before:
        raise Error("min_lv should have increased (L2 pushes it up)")

    # predict_member must produce finite (mu, lv) with the soft clamp.
    var mu_dev = ctx.enqueue_create_buffer[DT](BATCH * PRED)
    var lv_dev = ctx.enqueue_create_buffer[DT](BATCH * PRED)
    var mu_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        mu_dev.unsafe_ptr()
    )
    var lv_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        lv_dev.unsafe_ptr()
    )
    var mu_t = TileTensor(mu_p, row_major[BATCH, PRED]())
    var lv_t = TileTensor(lv_p, row_major[BATCH, PRED]())
    blk.predict_member["gpu"](0, in_t, mu_t, lv_t)
    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * PRED)
    ctx.enqueue_copy(out_host, lv_dev)
    ctx.synchronize()
    for i in range(BATCH * PRED):
        var v = out_host.unsafe_ptr()[i]
        if not isfinite(v):
            raise Error("predict_member produced non-finite logvar")
        # Soft-clamp keeps lv within (min, max) ≈ (-10, 0.5).
        if v > Scalar[DT](1.0) or v < Scalar[DT](-11.0):
            raise Error("clamped logvar out of expected band")

    # eval_member_loss finite.
    var hl = blk.eval_member_loss["gpu"](0, in_t, tgt_t)
    print("member 0 holdout loss =", hl)
    if not isfinite(hl):
        raise Error("eval loss non-finite")

    # Fixed-clamp path still runs (learnable off) — sanity that we didn't
    # break the default.
    var blk2 = Block.make["gpu", Kaiming](ctx)
    blk2.set_lr(Scalar[DT](1e-3))
    var l2 = blk2.train_member_step["gpu"](0, in_t, tgt_t)
    if not isfinite(l2):
        raise Error("fixed-clamp path loss non-finite")
    print("fixed-clamp path loss =", l2)

    # ── CPU path (mirror) — the example is GPU-only, so the CPU branches of
    # train/predict/bounds-step are otherwise never RUN. ──────────────────
    print("-" * 60)
    print("CPU path")
    var cblk = Block.make["cpu", Kaiming]()
    cblk.set_lr(Scalar[DT](1e-3))
    cblk.enable_learnable_bounds()
    var cin = List[Scalar[DT]](length=BATCH * IN_DIM, fill=Scalar[DT](0.0))
    var ctg = List[Scalar[DT]](length=BATCH * PRED, fill=Scalar[DT](0.0))
    for i in range(BATCH * IN_DIM):
        cin[i] = Scalar[DT](0.1) * Scalar[DT]((i % 7) - 3)
    for i in range(BATCH * PRED):
        ctg[i] = Scalar[DT](0.05) * Scalar[DT]((i % 5) - 2)
    var cin_t = TileTensor(cin.unsafe_ptr(), row_major[BATCH, IN_DIM]())
    var ctg_t = TileTensor(ctg.unsafe_ptr(), row_major[BATCH, PRED]())
    var cmax_before = cblk._max_lv.cpu_ptr()[0]
    var cmin_before = cblk._min_lv.cpu_ptr()[0]
    var closs = Scalar[DT](0.0)
    for _ in range(60):
        closs = cblk.train_member_step["cpu"](0, cin_t, ctg_t)
    var cmax_after = cblk._max_lv.cpu_ptr()[0]
    var cmin_after = cblk._min_lv.cpu_ptr()[0]
    print("cpu loss =", closs, " max:", cmax_before, "->", cmax_after,
          " min:", cmin_before, "->", cmin_after)
    if not isfinite(closs):
        raise Error("cpu train loss non-finite")
    if cmax_after >= cmax_before or cmin_after <= cmin_before:
        raise Error("cpu bounds did not tighten")

    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
