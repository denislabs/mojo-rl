"""LeWM (nn) — PushT JEPA training at the PAPER-WIDTH recipe (GPU).

Scales the scaled-baseline `lewm_pusht_train_gpu.mojo` up to the legacy
`lewm_pusht_pixel_train_gpu_paper.mojo` width: ViT-Tiny encoder (hidden=192,
3 heads × head_dim 64, 12 layers), wide projectors (2048), wide predictor
FFN (2048), depth 6.

FULLY FAITHFUL NOW. Encoder = paper ViT-Tiny (hidden=192, enc_heads=3 →
head_dim 64, inner 192). Predictor = paper expanded attention via
`MultiHeadAttentionXL` (PRED_HEADS=16 × PRED_DIM_HEAD=64 = 1024 inner ≫ emb
192) — the `PRED_DIM_HEAD` param (last LeWMTrainer param) threads through
ARPredictor → ConditionalTransformerBlock → MultiHeadAttentionXL. Validated by
`tests/nn/test_conditional_block_xl.mojo` (expanded attn, identity-at-init +
grad, CPU+GPU bitwise).

This is a LARGE model (12 enc layers + 6 cond blocks @ 192-d, 224×224, B=16):
expect a long compile and a multi-hour run (legacy estimated ~6-10h @ 32k
steps). STEPS defaults to 8000 — crank for a full run.

Run (NVIDIA; reuses the cached PushT dataset):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_train_gpu_paper.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.pong_data import WindowSource
from mojo_rl.envs.pusht import PushTOfflineSampler


# ── paper-width PushT-ViT recipe (encoder exact; predictor attn non-expanded)
comptime IN_CH = 3
comptime IMG = 224
comptime PATCH = 14            # → 256 patches
comptime HIDDEN = 192          # paper ViT-Tiny
comptime ENC_HEADS = 3         # head_dim = 192/3 = 64 (paper)
comptime ENC_LAYERS = 12       # paper depth
comptime EMB = 192
comptime ENC_PROJ_H = 2048     # paper wide projector
comptime ENC_FF_MULT = 2       # legacy encoder ff_mult (ViT-Tiny FFN = 384)
comptime T = 6
comptime ACT = 10
comptime SMOOTHED = 32
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 16       # paper: 16 heads
comptime PRED_DIM_HEAD = 64    # paper: head_dim 64 → inner = 16·64 = 1024 ≫
                               # emb 192 (EXPANDED attention via
                               # MultiHeadAttentionXL — now fully faithful)
comptime PRED_FF = 2048        # paper wide predictor FFN
comptime DEPTH = 6
comptime PRED_PROJ_H = 2048
comptime SIG_PROJ = 2048       # 2048/EMB(192) ≈ 10.7× over-determined (restores
                               # the baseline's ~11× ratio; paper used 1024 ≈
                               # 5.3× which converges slower — P/D ratio drives
                               # anti-collapse strength, see the Pong sweep)
comptime SIG_KNOTS = 17
comptime B = 16
comptime FRAMESKIP = 5

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

comptime STEPS: Int = 32000    # paper budget (the bigger model needs it)
comptime LOG_EVERY: Int = 200
comptime CKPT_EVERY: Int = 2000  # periodic v3 saves (atomic tmp+rename):
                                 # crash-safe on a 6-10 h run + enables
                                 # intermediate evals on the live ckpt
comptime LAM: Scalar[DT] = 0.09
comptime LR: Scalar[DT] = 1e-3
comptime CKPT_PATH: String = "/tmp/lewm_pusht_paper_world_model.txt"

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu", PRED_DIM_HEAD,
]
comptime Source = WindowSource[
    IMG_DIM, ACT, T, B, "gpu", PushTOfflineSampler, IN_CH, IMG
]


def main() raises:
    print("=" * 70)
    print("LeWM nn — PushT JEPA training (GPU, PAPER-WIDTH)")
    print("=" * 70)
    print("encoder: hidden=", HIDDEN, " heads=", ENC_HEADS, " layers=",
          ENC_LAYERS, " | emb=", EMB, " proj_h=", ENC_PROJ_H)
    print("predictor: depth=", DEPTH, " pred_ff=", PRED_FF, " pred_heads=",
          PRED_HEADS, " head_dim=", PRED_DIM_HEAD, " (expanded attn inner=",
          PRED_HEADS * PRED_DIM_HEAD, "= paper 1024)")
    print()

    var ctx = DeviceContext()
    print("opening PushT expert dataset ...")
    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var src = Source.make(sampler^, ctx=ctx)
    var tr = Trainer.make(lam=LAM, lr=LR, ctx=ctx)

    print("training", STEPS, "steps ...")
    tr.reset_loss_accum()
    for s in range(STEPS):
        src.next_batch()
        var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
        var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
        _ = tr.train_step(pix_t, act_t)
        if (s + 1) % LOG_EVERY == 0:
            var wl = tr.read_loss_accum()
            tr.reset_loss_accum()
            var probes = tr.collapse_probes()
            print("   step", s + 1, "/", STEPS,
                  " loss=", wl, " var_min=", probes[0],
                  " gram_off=", probes[1])
        if (s + 1) % CKPT_EVERY == 0:
            tr.save_params(CKPT_PATH)
            print("   [ckpt] saved @ step", s + 1, "→", CKPT_PATH)

    print()
    print("saving →", CKPT_PATH)
    tr.save_params(CKPT_PATH)
    _ = src^
    _ = tr^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
