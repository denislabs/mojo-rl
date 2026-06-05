"""LeWM (nn2) — train the JEPA world model on real Pong pixels (GPU).

End-to-end real-data driver for the nn2 LeWM port. Loads a Pong offline
buffer (collect it first with `examples/lewm/lewm_pong_collect_buffer.mojo`),
streams length-T windows through `PongWindowSource` into `LeWMTrainer` at the
§10.7 Pong-ViT recipe, and prints the loss + representation-collapse probes.

Pipeline (all real, no synthetic data):
  PongPixelEnv → PongOfflineBuffer (uint8 CHW + actions + dones, on disk)
    → PongWindowSource (sample window → H2D → uint8→fp32 ÷255)
    → LeWMTrainer.train_step (JEPA graph: encoder → AR predictor → MSE + SIGReg)

Recipe (LeWMPongViTConfig[batch=16, t=6, depth=6, hidden=128, emb=128]):
  84×84×4 frames, patch=14 → 36 patches, H=3 context, EMB=128, DEPTH=6.

Run (after collecting the buffer):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm2_pong_train_gpu.mojo

Watch for: loss falling smoothly, var_min rising > 0.1, gram_off < 0.5
(legacy §10.7 healthy-representation thresholds). On Apple it runs but is
slow at this scale — NVIDIA is the intended target.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.experimental.lewm2.trainer import LeWMTrainer
from mojo_rl.experimental.lewm2.pong_data import PongWindowSource
from mojo_rl.envs.arcade_games.pong.offline_buffer import (
    PongOfflineBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)


# ── §10.7 Pong-ViT recipe ──────────────────────────────────────────────
comptime IN_CH = 4
comptime IMG = 84
comptime PATCH = 14          # 84 // 14 == 6 → 36 patches
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
comptime SIG_PROJ = 256       # > D=128: over-determines the latent so SIGReg
                              # can't be gamed by collapsing orthogonal dims
                              # (P=64 was too coarse → real-Pong collapse)
comptime SIG_KNOTS = 5
comptime B = 16

comptime IMG_DIM = IN_CH * IMG * IMG       # 28224 == PONG_FRAME_BYTES
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

# ── run config ──────────────────────────────────────────────────────────
comptime BUFFER_PATH: String = "/tmp/lewm_pong_buffer.bin"
comptime STEPS: Int = 2000
comptime LOG_EVERY: Int = 50
comptime LAM: Scalar[DT] = 0.09
comptime LR: Scalar[DT] = 1e-3
comptime CKPT_PATH: String = "/tmp/lewm2_pong_world_model.txt"

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]
comptime Source = PongWindowSource[IMG_DIM, ACT, T, B, "gpu"]


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def main() raises:
    print("=" * 70)
    print("LeWM nn2 — Pong JEPA world model training (GPU, real pixels)")
    print("=" * 70)
    print("recipe: 84x84x4, patch=14, EMB=", EMB, " DEPTH=", DEPTH,
          " B=", B, " T=", T)
    print("buffer:", BUFFER_PATH)
    print()

    var ctx = DeviceContext()

    print("loading offline buffer ...")
    var buf = PongOfflineBuffer.load(BUFFER_PATH)
    print("   n_frames =", buf.n_frames)

    var src = Source.make(buf^, ctx=ctx)
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
    print("saving world-model checkpoint →", CKPT_PATH)
    tr.save_params(CKPT_PATH)

    _ = src^
    _ = tr^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
