"""Dreamer 4 tokenizer lighthouse — GPU (Pong reconstruction).

    pixi run -e apple  mojo run -I . examples/dreamer4/pong_tokenizer_lighthouse_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/dreamer4/pong_tokenizer_lighthouse_gpu.mojo

Same as the CPU lighthouse but the tokenizer trains on the GPU. Data prep
(stream Pong, take latest grayscale channel, box-downscale 84→32, patchify)
stays on the host; the patches are copied H2D and the model forward/backward
+ Adam run on device. The masked-MSE training gradient is a device kernel
(`masked_recon_grad_gpu`); PSNR eval downloads `pred` and scores on host.

Gate: full-frame PSNR climbs over training, on GPU.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.envs.arcade_games.pong.online_sampler import (
    OnlinePongSampler, ScriptedPongPolicy,
)
from mojo_rl.experimental.lewm2.pong_data import WindowSource
from mojo_rl.deep_agents2.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents2.dreamer4.recon_loss import (
    masked_recon_grad_gpu, full_recon_psnr,
)
from mojo_rl.deep_agents2.dreamer4.patchify import downscale_box, temporal_patchify


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def main() raises:
    print("=" * 70)
    print("Dreamer 4 tokenizer lighthouse — Pong reconstruction (GPU)")
    print("=" * 70)

    comptime IN_CH = 4
    comptime IMG = 84
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime ACT = 3
    comptime T = 4
    comptime B = 4
    comptime BATCH = B * T

    comptime TGT = 32
    comptime PATCH = 8
    comptime NP = (TGT // PATCH) * (TGT // PATCH)   # 16
    comptime DP = PATCH * PATCH                      # 64 (C=1)
    comptime D = 64
    comptime NH = 4
    comptime L = 8
    comptime D_BOT = 16
    comptime HID = 256
    comptime DEPTH = 2
    comptime DROP = 0.5
    comptime STEPS = 150
    comptime EVAL_EVERY = 25
    comptime LR = Scalar[DT](2e-3)

    comptime FRAME_N = BATCH * TGT * TGT
    comptime PATCH_N = BATCH * NP * DP

    var ctx = DeviceContext()
    comptime OnlineBuf = OnlinePongSampler[ScriptedPongPolicy, B, T]
    var src = WindowSource[IMG_DIM, ACT, T, B, "cpu", OnlineBuf].make(
        OnlineBuf.make(ScriptedPongPolicy(eps=0.3))
    )

    var tok = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, DROP, DROP, 7
    ].make[target="gpu", INIT=Xavier](ctx)
    var optim = Adam.make["gpu", M=type_of(tok)](tok, ctx)
    optim.lr = LR

    # host data-prep buffers
    var frames = _alloc(FRAME_N)
    var patches_h = ctx.enqueue_create_host_buffer[DT](PATCH_N)
    var pred_h = ctx.enqueue_create_host_buffer[DT](PATCH_N)
    ctx.synchronize()
    # device buffers
    var patches_d = ctx.enqueue_create_buffer[DT](PATCH_N)
    var pred_d = ctx.enqueue_create_buffer[DT](PATCH_N)
    var gpred_d = ctx.enqueue_create_buffer[DT](PATCH_N)
    var gin_d = ctx.enqueue_create_buffer[DT](PATCH_N)
    var pt = TileTensor(_mao(patches_d), row_major[BATCH, NP * DP]())
    var prt = TileTensor(_mao(pred_d), row_major[BATCH, NP * DP]())
    var got = TileTensor(_mao(gpred_d), row_major[BATCH, NP * DP]())
    var git = TileTensor(_mao(gin_d), row_major[BATCH, NP * DP]())

    var first_psnr: Float64 = 0.0
    var last_psnr: Float64 = 0.0

    for step in range(STEPS):
        src.next_batch()
        var pix = src.pix_ptr()
        for b in range(B):
            for t in range(T):
                var bt = b * T + t
                var fsrc = pix + (b * T + t) * IMG_DIM + 3 * IMG * IMG
                downscale_box[IMG, IMG, TGT, TGT](fsrc, frames + bt * TGT * TGT)
        temporal_patchify[BATCH, 1, TGT, TGT, PATCH](frames, patches_h.unsafe_ptr())
        ctx.enqueue_copy(patches_d, patches_h)

        optim.zero_grad["gpu"](tok)
        tok.forward["gpu", BATCH](pt, output=prt)
        var mask = tok.mae_mask_ptr()  # device keep ptr
        masked_recon_grad_gpu[NP, DP, BATCH](
            _mao(pred_d), _mao(patches_d), mask, _mao(gpred_d), ctx
        )
        tok.vjp["gpu", BATCH](got, git)
        optim.step["gpu"](tok)
        tok.advance_rng()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            tok.set_mae_p(0.0, 0.0)
            tok.forward["gpu", BATCH](pt, output=prt)
            ctx.enqueue_copy(pred_h, pred_d)
            ctx.synchronize()
            var psnr = full_recon_psnr[NP, DP, BATCH](
                pred_h.unsafe_ptr(), patches_h.unsafe_ptr()
            )
            tok.set_mae_p(DROP, DROP)
            if step == 0:
                first_psnr = psnr
            last_psnr = psnr
            print("  step", step, " full PSNR =", psnr, "dB")

    print("-" * 70)
    print("  first PSNR =", first_psnr, "dB   final PSNR =", last_psnr, "dB")
    assert_true(last_psnr > first_psnr + 3.0, "PSNR must climb (>3 dB) on GPU")
    print("=" * 70)
    print("GPU LIGHTHOUSE PASSED — tokenizer learns Pong reconstruction on GPU")
    print("=" * 70)
