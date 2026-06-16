"""LeWM (nn) — decoder probe on the CLS-TOKEN PushT world model (GPU).

The first validation gate after the CLS retrain: train the reconstruction
decoder on the frozen CLS-token WM and look at the reconstructions — does the
agent/pusher dot now come through (vs the mean-pooled WM, which dropped it)?
If yes, the CLS token captured control-relevant state and a second
closed-loop attempt is warranted.

Identical to `lewm_pusht_decode_gpu.mojo` except the WM uses the CLS encoder
(`EncCLS` passed as LeWMTrainer's trailing ENC param) and the CLS checkpoint.

Run (NVIDIA, after lewm_pusht_train_gpu_paper_cls.mojo):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_decode_gpu_cls.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.encoder import LeWMEncoderCLS
from mojo_rl.experimental.lewm.pong_data import WindowSource
from mojo_rl.experimental.lewm.decoder import patchify, unpatchify
from mojo_rl.experimental.lewm.decoder_trainer import LeWMDecoderTrainer
from mojo_rl.envs.pusht import PushTOfflineSampler
from mojo_rl.render.image_writer import save_reconstruction_grid


# ── CLS WM config (matches lewm_pusht_train_gpu_paper_cls.mojo) ───────
comptime IN_CH = 3
comptime IMG = 224
comptime PATCH = 14
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)
comptime HIDDEN = 192
comptime ENC_HEADS = 3
comptime ENC_LAYERS = 12
comptime EMB = 192
comptime ENC_PROJ_H = 2048
comptime ENC_FF_MULT = 2
comptime T = 6
comptime ACT = 10
comptime SMOOTHED = 32
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 16
comptime PRED_DIM_HEAD = 64
comptime PRED_FF = 2048
comptime DEPTH = 6
comptime PRED_PROJ_H = 2048
comptime SIG_PROJ = 2048
comptime SIG_KNOTS = 17
comptime B = 16
comptime FRAMESKIP = 5

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT
comptime WM_CKPT: String = "/tmp/lewm_pusht_paper_cls_world_model.txt"

comptime PATCH_D = 16
comptime N_Q = (IMG // PATCH_D) * (IMG // PATCH_D)
comptime PATCH_PX = IN_CH * PATCH_D * PATCH_D
comptime DEC_HID = 192
comptime DEC_FF = 4 * DEC_HID
comptime DEC_LAYERS = 4
comptime DEC_BATCH = B * T

comptime STEPS = 20_000
comptime PRINT_EVERY = 500
comptime VIZ_EVERY = 4_000
comptime N_VIZ = 8
comptime DEC_CKPT: String = "/tmp/lewm_pusht_decoder_cls.txt"

comptime EncCLS = LeWMEncoderCLS[
    IN_CH, IMG, PATCH, N_PATCHES, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
    ENC_PROJ_H, ENC_FF_MULT,
]
comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu", PRED_DIM_HEAD, EncCLS,
]
comptime Source = WindowSource[
    IMG_DIM, ACT, T, B, "gpu", PushTOfflineSampler, IN_CH, IMG
]
comptime Decoder = LeWMDecoderTrainer[
    EMB, DEC_HID, N_Q, PATCH_PX, DEC_FF, DEC_LAYERS, DEC_BATCH, "gpu"
]


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def main() raises:
    print("=" * 70)
    print("LeWM nn — decoder probe on the CLS-token PushT WM (GPU)")
    print("=" * 70)
    var ctx = DeviceContext()

    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var src = Source.make(sampler^, ctx=ctx)
    var wm = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading frozen CLS WM", WM_CKPT, "...")
    wm.load_params(WM_CKPT)

    var dec = Decoder.make(lr=Scalar[DT](1e-3), ctx=ctx)
    print("  decoder:", DEC_LAYERS, "layers, hid", DEC_HID, ", queries", N_Q)

    var tgt_dev = ctx.enqueue_create_buffer[DT](DEC_BATCH * N_Q * PATCH_PX)
    var recon_host = ctx.enqueue_create_host_buffer[DT](
        DEC_BATCH * N_Q * PATCH_PX
    )
    var recon_img = ctx.enqueue_create_host_buffer[DT](N_VIZ * IMG_DIM)
    var orig_img = ctx.enqueue_create_host_buffer[DT](N_VIZ * IMG_DIM)
    ctx.synchronize()

    print("training decoder", STEPS, "steps (CLS encoder frozen) ...")
    dec.reset_loss_accum()
    for step in range(1, STEPS + 1):
        src.next_batch()
        var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
        var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
        _ = wm.eval_loss(pix_t, act_t)
        var emb_ptr = wm.graph.node_out_ptr["emb"]()
        var emb_t = TileTensor(emb_ptr, row_major[DEC_BATCH, EMB]())
        patchify["gpu", DEC_BATCH, IN_CH, IMG, PATCH_D](
            ctx, src.pix_ptr(), _p(tgt_dev)
        )
        var tgt_t = TileTensor(
            _p(tgt_dev), row_major[DEC_BATCH, N_Q * PATCH_PX]()
        )
        _ = dec.train_step(emb_t, tgt_t)

        if step % PRINT_EVERY == 0:
            var ml = dec.read_loss_accum()
            dec.reset_loss_accum()
            print("   step", step, "/", STEPS, " recon_mse=", ml)

        if step % VIZ_EVERY == 0 or step == STEPS:
            dec.recon_into(
                emb_t,
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    recon_host.unsafe_ptr()
                ),
            )
            unpatchify["cpu", N_VIZ, IN_CH, IMG, PATCH_D](
                None,
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    recon_host.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    recon_img.unsafe_ptr()
                ),
            )
            # D2H the first N_VIZ original frames (host idiom: UnsafePointer
            # dst + non-owning device src).
            var orig_dev = DeviceBuffer[DT](
                ctx,
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](src.pix_ptr()),
                N_VIZ * IMG_DIM, owning=False,
            )
            ctx.enqueue_copy(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    orig_img.unsafe_ptr()
                ),
                orig_dev,
            )
            ctx.synchronize()
            var path = "/tmp/lewm_pusht_cls_recon_" + String(step) + ".ppm"
            save_reconstruction_grid(
                path,
                orig_img.unsafe_ptr(),
                recon_img.unsafe_ptr(),
                n=N_VIZ, height=IMG, width=IMG, channels=IN_CH,
                vmin=0.0, vmax=1.0,
            )

    dec.save_params(DEC_CKPT)
    print("decoder weights →", DEC_CKPT)
    _ = src^; _ = wm^; _ = dec^
    print("=" * 70)
    print("DONE — compare /tmp/lewm_pusht_cls_recon_*.ppm vs the mean-pool grid")
    print("  (agent dot now visible ⇒ CLS captured the pusher ⇒ retry closed-loop)")
    print("=" * 70)
