"""LeWMTrainer GPU scale check — legacy §10.7 Pong-ViT recipe dims.

Instantiates the trainer at the production recipe
(`LeWMPongViTConfig[batch=16, t=6, depth=6, hidden=128, emb=128]`:
84×84×4 frames, patch=14 → 36 patches, H=3, EMB=128, DEPTH=6) and runs
the device train loop on SYNTHETIC data.

PURPOSE — this is the CUDA-correctness-at-scale check, NOT the paper
convergence reproduction. It confirms every GPU kernel in the JEPA graph
(ViT patch-embed at 84×84, MHA, SIGReg's 9 kernels, the loss reduce,
checkpoint D2H/H2D) compiles + runs on CUDA at production sizes, and that
the collapse probes stay finite/healthy on a big net. The real §10.7
numbers require real Pong frames + the offline data-collection port
(not yet wired into lewm).

Run on NVIDIA:
  pixi run -e nvidia mojo run -I . tests/experimental/lewm/test_trainer_gpu_scale.mojo
(also runs on Apple, slowly — used here to verify the script compiles).
Crank STEPS up on NVIDIA to watch the loss + collapse trend over a longer
horizon.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer


# ── legacy §10.7 Pong-ViT recipe (batch=16, t=6, depth=6, emb=128) ─────
comptime IN_CH = 4
comptime IMG = 84
comptime PATCH = 14          # 84 // 14 == 6  → 36 patches
comptime HIDDEN = 128
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 1
comptime EMB = 128
comptime ENC_PROJ_H = 64
comptime ENC_FF_MULT = 2
comptime T = 6
comptime ACT = 3
comptime SMOOTHED = 16
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 2
comptime PRED_FF = 64
comptime DEPTH = 6
comptime PRED_PROJ_H = 256
comptime SIG_PROJ = 64
comptime SIG_KNOTS = 5
comptime B = 16

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def main() raises:
    print("=" * 70)
    print("LeWMTrainer GPU scale check (Pong-ViT §10.7 dims, synthetic data)")
    print("=" * 70)
    var ctx = DeviceContext()

    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)

    var pix_d = ctx.enqueue_create_buffer[DT](B * PIX)
    var act_d = ctx.enqueue_create_buffer[DT](B * ACTIN)
    var pix_h = ctx.enqueue_create_host_buffer[DT](B * PIX)
    var act_h = ctx.enqueue_create_host_buffer[DT](B * ACTIN)
    ctx.synchronize()
    for k in range(B * PIX):
        pix_h.unsafe_ptr()[k] = _det(k + 1, 1.0)
    for k in range(B * ACTIN):
        act_h.unsafe_ptr()[k] = _det(k + 7, 1.0)
    ctx.enqueue_copy(pix_d, pix_h)
    ctx.enqueue_copy(act_d, act_h)
    ctx.synchronize()

    var pix_t = TileTensor(_p(pix_d), row_major[B, PIX]())
    var act_t = TileTensor(_p(act_d), row_major[B, ACTIN]())

    print("train ... (crank STEPS up on NVIDIA for a longer trend)")
    comptime STEPS = 30
    comptime WINDOW = 10
    var first_win: Scalar[DT] = 0.0
    var last_win: Scalar[DT] = 0.0
    var win_idx = 0
    tr.reset_loss_accum()
    for s in range(STEPS):
        _ = tr.train_step(pix_t, act_t)
        if (s + 1) % WINDOW == 0:
            var wl = tr.read_loss_accum()
            tr.reset_loss_accum()
            var probes = tr.collapse_probes()
            print("   step", s, " window_loss=", wl,
                  " var_min=", probes[0], " gram_off=", probes[1])
            assert_true(not (isnan(wl) or isinf(wl)), "window loss finite")
            if win_idx == 0:
                first_win = wl
            last_win = wl
            win_idx += 1
    print("   first_window=", first_win, " last_window=", last_win)
    assert_true(last_win < first_win, "loss must decrease over training")

    var probes = tr.collapse_probes()
    assert_true(not (isnan(probes[0]) or isinf(probes[0])), "var_min finite")
    assert_true(not (isnan(probes[1]) or isinf(probes[1])), "gram_off finite")

    _ = tr^
    print("=" * 70)
    print("ALL PASSED (CUDA path runs at §10.7 scale)")
    print("=" * 70)
