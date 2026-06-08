"""LeWM (nn2) — PushT JEPA training at the PAPER-WIDTH recipe (GPU).

Scales the scaled-baseline `lewm2_pusht_train_gpu.mojo` up to the legacy
`lewm_pusht_pixel_train_gpu_paper.mojo` width: ViT-Tiny encoder (hidden=192,
3 heads × head_dim 64, 12 layers), wide projectors (2048), wide predictor
FFN (2048), depth 6.

FIDELITY NOTE — the ENCODER is exactly the paper's ViT-Tiny (hidden=192,
enc_heads=3 → head_dim = 192/3 = 64, inner = 192, standard MHA). The
PREDICTOR is where nn2 deviates: the paper predictor uses EXPANDED attention
(pred_heads=16 × pred_dim_head=64 = 1024 inner dim, decoupled from emb=192).
nn2's MultiHeadAttention ties head_dim = EMB/HEADS (inner = emb), per locked
decision #5 — there's no 1024-wide expansion. We set PRED_HEADS=3 → head_dim
64 (matches the paper's per-head resolution; inner stays 192, not 1024). So
this matches the paper on every capacity axis EXCEPT the predictor's attention
expansion. For a fully-faithful predictor, nn2 needs a MultiHeadAttentionXL
(independent head_dim → QKV Linear[emb, 3·heads·head_dim]); ping to add it.

This is a LARGE model (12 enc layers + 6 cond blocks @ 192-d, 224×224, B=16):
expect a long compile and a multi-hour run (legacy estimated ~6-10h @ 32k
steps). STEPS defaults to 8000 — crank for a full run.

Run (NVIDIA; reuses the cached PushT dataset):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm2_pusht_train_gpu_paper.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.experimental.lewm2.trainer import LeWMTrainer
from mojo_rl.experimental.lewm2.pong_data import WindowSource
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
comptime PRED_HEADS = 3        # head_dim = 192/3 = 64 (paper per-head width;
                               # paper used 16 heads × 64 = 1024 inner — see note)
comptime PRED_FF = 2048        # paper wide predictor FFN
comptime DEPTH = 6
comptime PRED_PROJ_H = 2048
comptime SIG_PROJ = 1024       # paper value. NOTE: 1024/EMB(192) ≈ 5.3× —
                               # HALF the baseline's 1024/96 ≈ 11× over-
                               # determination, so isotropy comes in slower
                               # (an 8000-step run leaves var_min ~0.09,
                               # still rising). For faster/cleaner var_min set
                               # SIG_PROJ=2048 (≈11× again); else use the
                               # paper's 32000 steps. (P/D ratio drives
                               # anti-collapse strength — see the Pong sweep.)
comptime SIG_KNOTS = 17
comptime B = 16
comptime FRAMESKIP = 5

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

comptime STEPS: Int = 8000     # paper used 32000 — at 8000 the paper-width
                               # var_min is still climbing (~0.09); use 32000
                               # and/or SIG_PROJ=2048 for a converged var_min>0.1
comptime LOG_EVERY: Int = 200
comptime LAM: Scalar[DT] = 0.09
comptime LR: Scalar[DT] = 1e-3
comptime CKPT_PATH: String = "/tmp/lewm2_pusht_paper_world_model.txt"

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]
comptime Source = WindowSource[
    IMG_DIM, ACT, T, B, "gpu", PushTOfflineSampler, IN_CH, IMG
]


def main() raises:
    print("=" * 70)
    print("LeWM nn2 — PushT JEPA training (GPU, PAPER-WIDTH)")
    print("=" * 70)
    print("encoder: hidden=", HIDDEN, " heads=", ENC_HEADS, " layers=",
          ENC_LAYERS, " | emb=", EMB, " proj_h=", ENC_PROJ_H)
    print("predictor: depth=", DEPTH, " pred_ff=", PRED_FF, " pred_heads=",
          PRED_HEADS, "(head_dim 64; attn inner=emb, not paper's 1024)")
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

    print()
    print("saving →", CKPT_PATH)
    tr.save_params(CKPT_PATH)
    _ = src^
    _ = tr^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
