"""LeWM (nn2) — SIGReg λ sweep on real Pong (GPU).

The first real-Pong run (λ=0.09) collapsed: prediction loss fell but
var_min→0.01 / gram_off→0.81 (Pong's redundant frames let the model
satisfy the mean prediction MSE by collapsing the latent). SIGReg needs
more weight on real data than the synthetic scale test suggested. This
sweep trains a FRESH world model per λ (sharing one offline buffer +
window source) and prints the final loss + collapse probes, flagging the
λ values that stay healthy (var_min > 0.1, gram_off < 0.5). Pick the
smallest healthy λ for the full run.

Run (after collecting the buffer):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm2_pong_lambda_sweep_gpu.mojo
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
comptime PATCH = 14
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

comptime BUFFER_PATH: String = "/tmp/lewm_pong_buffer.bin"
comptime STEPS: Int = 1500
comptime LR: Scalar[DT] = 1e-3

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]
comptime Source = PongWindowSource[IMG_DIM, ACT, T, B, "gpu"]


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _run_one(
    mut src: Source, lam: Scalar[DT], ctx: DeviceContext,
) raises -> Tuple[Float64, Float64, Float64]:
    """Train a fresh model at `lam`; return (final_loss, var_min, gram_off)."""
    var tr = Trainer.make(lam=lam, lr=LR, ctx=ctx)
    tr.reset_loss_accum()
    var last_loss: Float64 = 0.0
    for s in range(STEPS):
        src.next_batch()
        var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
        var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
        _ = tr.train_step(pix_t, act_t)
        if (s + 1) % 100 == 0:
            last_loss = Float64(tr.read_loss_accum())
            tr.reset_loss_accum()
    var probes = tr.collapse_probes()
    _ = tr^
    return (last_loss, Float64(probes[0]), Float64(probes[1]))


def main() raises:
    print("=" * 70)
    print("LeWM nn2 — SIGReg λ sweep on real Pong (GPU)")
    print("=" * 70)
    var ctx = DeviceContext()

    var buf = PongOfflineBuffer.load(BUFFER_PATH)
    print("buffer n_frames =", buf.n_frames, "  steps/λ =", STEPS)
    var src = Source.make(buf^, ctx=ctx)

    var lambdas = List[Scalar[DT]](
        Scalar[DT](0.09), Scalar[DT](0.3), Scalar[DT](1.0), Scalar[DT](3.0)
    )
    print()
    print("   λ        loss        var_min     gram_off    healthy?")
    print("   " + "-" * 56)
    for i in range(len(lambdas)):
        var lam = lambdas[i]
        var r = _run_one(src, lam, ctx)
        var healthy = r[1] > 0.1 and r[2] < 0.5
        print("   ", lam, "   ", r[0], "  ", r[1], "  ", r[2], "  ",
              "YES" if healthy else "no")

    _ = src^
    print("=" * 70)
    print("Pick the smallest λ flagged healthy for the full run.")
    print("=" * 70)
