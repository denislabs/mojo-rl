"""Dreamer4 tokenizer probe — reproduces the REAL online Stage 1 THROUGH the
replay buffer, and logs the two things the production driver hides:

  * TARGET frame variance (the mean-prediction MSE floor) — is the sampled data
    sane [0.3,1.0] real frames, or degenerate/stale?
  * NO-MASK recon (the frozen tokenizer runs at mae_p=0 downstream, but Stage 1
    only logs the MASKED training proxy) — the number that actually matters for RL.

Isolated tests (bypassing the buffer) reconstruct real frames to MSE ~0.005 on
HELD-OUT frames, but the real run's masked proxy sits ~0.022. This probe routes
through Dreamer4FrameBuffer.add_step_fp32_list + sample_reward_window_batch — the
one component the isolated tests skip — to localize the gap.

Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/car_racing/dreamer4_tokenizer_probe.mojo
"""

from std.gpu.host import DeviceContext
from std.random import seed
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents.dreamer4.frame_buffer import Dreamer4FrameBuffer
from mojo_rl.deep_agents.dreamer4.recon_loss import masked_recon_grad_gpu
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao
from mojo_rl.deep_agents.dreamer4.patchify import downscale_box, temporal_patchify
from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB


def _dptr(mut t: Tensor) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        t.dev.value().unsafe_ptr()
    )


def _win_to_patches[
    B: Int, T: Int, IN_CH: Int, IMG: Int, TGT: Int, PATCH: Int
](
    pix: UnsafePointer[Scalar[DT], MutAnyOrigin],
    frames: UnsafePointer[Scalar[DT], MutAnyOrigin],
    patches: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime BATCH = B * T
    for b in range(B):
        for t in range(T):
            var bt = b * T + t
            var fsrc = pix + bt * IMG_DIM + (IN_CH - 1) * IMG * IMG
            downscale_box[IMG, IMG, TGT, TGT](fsrc, frames + bt * TGT * TGT)
    temporal_patchify[BATCH, 1, TGT, TGT, PATCH](frames, patches)


def main() raises:
    # ── CAPACITY TEST TOGGLE ──
    # The probe showed the tokenizer plateaus at masked≈no-mask≈0.02 on the real
    # (diverse) warmup data — only ~15% of pixel variance, blurry. BIG bumps the
    # tokenizer's INTERNAL depth+width (D/HID/DEPTH). These do NOT touch the
    # L·D_BOT=256 bottleneck, so they cost nothing on the agent side (the agent's
    # ND=NSP·DSP=256 is unchanged). If BIG drops well below 0.02 → depth/width is
    # the lever (raise it in the tokenizer only). If BIG also sticks ~0.02 → the
    # 256-dim bottleneck itself is the limit (that one DOES ripple into the agent).
    comptime BIG = True

    comptime DP = 64
    comptime TOK_D = 256 if BIG else 128
    comptime TOK_NH = 4
    comptime T = 10
    comptime L = 16
    comptime NP = 64
    comptime D_BOT = 16
    comptime TOK_HID = 512 if BIG else 256
    comptime TOK_DEPTH = 4 if BIG else 2
    comptime DROP = 0.5
    comptime B = 8
    comptime BATCH = B * T
    comptime TGT = 64
    comptime PATCH = 8
    comptime IMG = 84
    comptime IN_CH = 4
    comptime NACT = 5
    comptime CAP = 20_000
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime WARMUP = 5_000
    comptime FRAME_REPEAT = 4

    seed(42)
    var ctx = DeviceContext()
    var tok = Dreamer4Tokenizer[
        DP, TOK_D, TOK_NH, T, L, NP, D_BOT, TOK_HID, TOK_DEPTH, DROP, DROP, 7
    ].make["gpu", Xavier](Optional(ctx))
    var buf = Dreamer4FrameBuffer[IN_CH, IMG, IMG, NACT, CAP]()
    var env = CarRacingMB[DT, PIXEL_OBS=True, PIX_RES=IMG]()
    print("config: BIG=", BIG, " TOK_D=", TOK_D, " TOK_HID=", TOK_HID,
          " TOK_DEPTH=", TOK_DEPTH, " (bottleneck L*D_BOT=", L * D_BOT, ")")

    # ── warmup collect THROUGH THE BUFFER (mirrors online.mojo Stage 0) ──
    print("warmup collect through buffer:", WARMUP, "steps")
    var lcg: UInt64 = 12345
    var ob0 = env.reset_obs_list()
    var cur = List[Scalar[DT]](length=IMG_DIM, fill=Scalar[DT](0.0))
    for i in range(IMG_DIM):
        cur[i] = Scalar[DT](Float64(ob0[i]))
    var collected = 0
    while collected < WARMUP:
        lcg = lcg * 6364136223846793005 + 1442695040888963407
        var a = Int((lcg >> 33) % NACT)
        var r_sum: Float64 = 0.0
        var d = False
        var nxt = cur.copy()
        for _ in range(FRAME_REPEAT):
            var res = env.step_obs(a)
            r_sum += Float64(res[1])
            d = res[2]
            for i in range(IMG_DIM):
                nxt[i] = Scalar[DT](Float64(res[0][i]))
            if d:
                break
        buf.add_step_fp32_list(cur, a, d, Scalar[DT](r_sum))
        collected += 1
        if d:
            var no = env.reset_obs_list()
            for i in range(IMG_DIM):
                cur[i] = Scalar[DT](Float64(no[i]))
        else:
            cur = nxt.copy()
    print("buffer count:", buf.count())

    # ── scratch ──
    var pix = Tensor()
    pix.ensure(BATCH * IMG_DIM)
    var act_oh = Tensor()
    act_oh.ensure(BATCH * NACT)
    var rew = Tensor()
    rew.ensure(BATCH)
    var done_b = Tensor()
    done_b.ensure(BATCH)
    var frames = Tensor()
    frames.ensure(BATCH * TGT * TGT)
    var patches = Tensor()
    patches.ensure(BATCH * NP * DP)
    patches.ensure_gpu(ctx, BATCH * NP * DP)
    var pred = Tensor()
    pred.ensure(BATCH * NP * DP)
    pred.ensure_gpu(ctx, BATCH * NP * DP)
    var gpred = Tensor()
    gpred.ensure_gpu(ctx, BATCH * NP * DP)
    var gin = Tensor()
    gin.ensure_gpu(ctx, BATCH * NP * DP)
    var topt = Adam(lr=Scalar[DT](2e-3))

    print("step  masked-MSE   NO-MASK-MSE   target[mean/var/min/max]")
    for s in range(2001):
        buf.sample_reward_window_batch[B, T](
            pix.data.unsafe_ptr(), act_oh.data.unsafe_ptr(),
            rew.data.unsafe_ptr(), done_b.data.unsafe_ptr(),
        )
        _win_to_patches[B, T, IN_CH, IMG, TGT, PATCH](
            _mao(pix.data.unsafe_ptr()), _mao(frames.data.unsafe_ptr()),
            _mao(patches.data.unsafe_ptr()),
        )
        patches.upload(ctx)
        topt.zero_grad["gpu"](tok, Optional(ctx))
        tok.forward["gpu", BATCH](TensorRefs[1](patches), pred, Optional(ctx))
        masked_recon_grad_gpu[NP, DP, BATCH](
            _dptr(pred), _dptr(patches), tok.mae_mask_ptr(), _dptr(gpred), ctx
        )
        tok.vjp["gpu", BATCH](
            TensorRefs[1](patches), gpred, TensorRefs[1](gin), Optional(ctx)
        )
        topt.step["gpu"](tok, Optional(ctx))
        tok.advance_rng()
        if s % 200 == 0:
            pred.download(ctx)
            # masked-proxy MSE (full-frame, mask-on forward) + target stats
            var mse: Float64 = 0.0
            var tsum: Float64 = 0.0
            var tsq: Float64 = 0.0
            var tmin: Float64 = 1e9
            var tmax: Float64 = -1e9
            for i in range(BATCH * NP * DP):
                var tv = Float64(patches.data[i])
                var d2 = Float64(pred.data[i]) - tv
                mse += d2 * d2
                tsum += tv
                tsq += tv * tv
                if tv < tmin:
                    tmin = tv
                if tv > tmax:
                    tmax = tv
            var nn_ = Float64(BATCH * NP * DP)
            mse /= nn_
            var tmean = tsum / nn_
            var tvar = tsq / nn_ - tmean * tmean
            # NO-MASK recon on the SAME batch (what RL actually consumes)
            tok.set_mae_p(0.0, 0.0)
            tok.forward["gpu", BATCH](TensorRefs[1](patches), pred, Optional(ctx))
            tok.set_mae_p(DROP, DROP)
            pred.download(ctx)
            var nm: Float64 = 0.0
            for i in range(BATCH * NP * DP):
                var d3 = Float64(pred.data[i]) - Float64(patches.data[i])
                nm += d3 * d3
            nm /= nn_
            print(s, " ", mse, "  ", nm, "  ",
                  tmean, " / ", tvar, " / ", tmin, " / ", tmax)
