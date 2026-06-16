"""Dreamer 4 tokenizer lighthouse — learn to reconstruct Pong frames (CPU).

    pixi run mojo run -I . examples/dreamer4/pong_tokenizer_lighthouse.mojo

Streams live Pong pixel windows (OnlinePongSampler → WindowSource, CPU), takes
the latest grayscale frame of each 4-stack, box-downscales 84×84 → 32×32,
patchifies (patch 8 → 16 patches × 64), and trains the Dreamer 4 causal
tokenizer with the masked-autoencoding objective. Every few steps it runs a
p=0 (no-mask) pass and reports full-frame reconstruction PSNR.

Gate: full-frame PSNR climbs substantially over training — the tokenizer
learns to encode+reconstruct real Pong frames. Pure CPU.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer import Adam
from mojo_rl.envs.arcade_games.pong.online_sampler import (
    OnlinePongSampler, ScriptedPongPolicy,
)
from mojo_rl.experimental.lewm.pong_data import WindowSource
from mojo_rl.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents.dreamer4.recon_loss import (
    masked_recon_loss, full_recon_psnr,
)
from mojo_rl.deep_agents.dreamer4.patchify import downscale_box, temporal_patchify


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    print("=" * 70)
    print("Dreamer 4 tokenizer lighthouse — Pong reconstruction (CPU)")
    print("=" * 70)

    # ── Pong stream config ──────────────────────────────────────────────
    comptime IN_CH = 4
    comptime IMG = 84
    comptime IMG_DIM = IN_CH * IMG * IMG       # 28224 (CHW 4-stack)
    comptime ACT = 3
    comptime T = 4                              # window length
    comptime B = 4
    comptime BATCH = B * T                      # frames per step

    # ── Tokenizer config (small, CPU-friendly) ──────────────────────────
    comptime TGT = 32                           # downscaled frame size
    comptime PATCH = 8
    comptime NP = (TGT // PATCH) * (TGT // PATCH)   # 16
    comptime DP = PATCH * PATCH                  # 64 (C=1)
    comptime D = 64
    comptime NH = 4
    comptime L = 8                               # latent tokens
    comptime D_BOT = 16
    comptime HID = 256
    comptime DEPTH = 2
    comptime DROP = 0.5                          # MAE drop-rate (train)
    comptime STEPS = 250
    comptime EVAL_EVERY = 25
    comptime LR = Scalar[DT](2e-3)

    comptime FRAME_N = BATCH * TGT * TGT         # C=1
    comptime PATCH_N = BATCH * NP * DP

    # ── Build stream + model ────────────────────────────────────────────
    comptime OnlineBuf = OnlinePongSampler[ScriptedPongPolicy, B, T]
    var src = WindowSource[IMG_DIM, ACT, T, B, "cpu", OnlineBuf].make(
        OnlineBuf.make(ScriptedPongPolicy(eps=0.3))
    )

    var tok = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, DROP, DROP, 7
    ].make[target="cpu", INIT=Xavier]()
    var optim = Adam.make["cpu", M=type_of(tok)](tok)
    optim.lr = LR

    var frames = _alloc(FRAME_N)
    var patches = _alloc(PATCH_N)
    var pred = _alloc(PATCH_N)
    var gpred = _alloc(PATCH_N)
    var gin = _alloc(PATCH_N)
    var pt = TileTensor(patches, row_major[BATCH, NP * DP]())
    var prt = TileTensor(pred, row_major[BATCH, NP * DP]())
    var git = TileTensor(gin, row_major[BATCH, NP * DP]())

    var first_psnr: Float64 = 0.0
    var last_psnr: Float64 = 0.0

    for step in range(STEPS):
        src.next_batch()
        var pix = src.pix_ptr()
        # latest grayscale channel (idx 3) of each stack → downscale → frames
        for b in range(B):
            for t in range(T):
                var bt = b * T + t
                var fsrc = pix + (b * T + t) * IMG_DIM + 3 * IMG * IMG
                downscale_box[IMG, IMG, TGT, TGT](fsrc, frames + bt * TGT * TGT)
        temporal_patchify[BATCH, 1, TGT, TGT, PATCH](frames, patches)

        # train step (MAE masking active)
        optim.zero_grad["cpu"](tok)
        tok.forward["cpu", BATCH](pt, output=prt)
        var mask = tok.mae_mask_ptr()
        var loss = masked_recon_loss[NP, DP, BATCH](pred, patches, mask, gpred)
        var got = TileTensor(gpred, row_major[BATCH, NP * DP]())
        tok.vjp["cpu", BATCH](got, git)
        optim.step["cpu"](tok)
        tok.advance_rng()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            # full-frame eval: p=0 (no masking), PSNR over all patches.
            tok.set_mae_p(0.0, 0.0)
            tok.forward["cpu", BATCH](pt, output=prt)
            var psnr = full_recon_psnr[NP, DP, BATCH](pred, patches)
            tok.set_mae_p(DROP, DROP)
            if step == 0:
                first_psnr = psnr
            last_psnr = psnr
            print(
                "  step", step, " masked_loss =", loss,
                " full PSNR =", psnr, "dB",
            )

    print("-" * 70)
    print("  first PSNR =", first_psnr, "dB   final PSNR =", last_psnr, "dB")
    assert_true(last_psnr > first_psnr + 3.0, "PSNR must climb (>3 dB)")
    print("=" * 70)
    print("LIGHTHOUSE PASSED — tokenizer learns Pong reconstruction")
    print("=" * 70)
