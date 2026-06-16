"""End-to-end wiring test for the LeWM decoder probe (toy, Apple GPU).

Mirrors `examples/lewm/lewm_pusht_decode_gpu.mojo` at toy scale to validate
the cross-component data flow the example depends on (the 224² example is an
NVIDIA-only build):

  frozen WM forward → `wm.graph.node_out_ptr["emb"]` → (B,T·EMB)==(B·T,EMB)
    → patchify(pixels) → decoder.train_step(emb, tgt)        (loss decreases)
    → decoder.recon_into → unpatchify → save_reconstruction_grid (file written)

The WM is run once on a fixed synthetic batch (encoder frozen); the decoder
overfits the fixed emb→image mapping.

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm/test_decode_integration.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.decoder import patchify, unpatchify
from mojo_rl.experimental.lewm.decoder_trainer import LeWMDecoderTrainer
from mojo_rl.render.image_writer import save_reconstruction_grid


# toy WM config (mirrors test_trainer_gpu.mojo)
comptime IN_CH = 4
comptime IMG = 8
comptime PATCH = 4
comptime HIDDEN = 8
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 2
comptime EMB = 8
comptime ENC_PROJ_H = 16
comptime ENC_FF_MULT = 2
comptime T = 4
comptime ACT = 3
comptime SMOOTHED = 8
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 2
comptime PRED_FF = 16
comptime DEPTH = 2
comptime PRED_PROJ_H = 16
comptime SIG_PROJ = 8
comptime SIG_KNOTS = 5
comptime B = 4

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

# decoder probe (matches WM EMB; its own patch size)
comptime PATCH_D = 4
comptime N_Q = (IMG // PATCH_D) * (IMG // PATCH_D)   # 4
comptime PATCH_PX = IN_CH * PATCH_D * PATCH_D        # 64
comptime DEC_HID = 8
comptime DEC_FF = 16
comptime DEC_LAYERS = 2
comptime DEC_BATCH = B * T                           # 16
comptime N_VIZ = 3

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]
comptime Decoder = LeWMDecoderTrainer[
    EMB, DEC_HID, N_Q, PATCH_PX, DEC_FF, DEC_LAYERS, DEC_BATCH, "gpu"
]


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int) -> Scalar[DT]:
    return Scalar[DT]((Float64((i * 2654435761) % 1000) / 500.0) - 1.0)


def main() raises:
    print("=" * 70)
    print("LeWM decoder probe — end-to-end wiring (toy, GPU)")
    print("=" * 70)
    var ctx = DeviceContext()

    var wm = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    var dec = Decoder.make(lr=Scalar[DT](3e-3), ctx=ctx)

    var pix_d = ctx.enqueue_create_buffer[DT](B * PIX)
    var act_d = ctx.enqueue_create_buffer[DT](B * ACTIN)
    var tgt_d = ctx.enqueue_create_buffer[DT](DEC_BATCH * N_Q * PATCH_PX)
    var pix_h = ctx.enqueue_create_host_buffer[DT](B * PIX)
    var act_h = ctx.enqueue_create_host_buffer[DT](B * ACTIN)
    ctx.synchronize()
    # fixed synthetic pixels in [0,1] (image-like) + actions
    for k in range(B * PIX):
        pix_h.unsafe_ptr()[k] = (_det(k + 1) + Scalar[DT](1.0)) * Scalar[DT](0.5)
    for k in range(B * ACTIN):
        act_h.unsafe_ptr()[k] = _det(k + 7)
    ctx.enqueue_copy(pix_d, pix_h); ctx.enqueue_copy(act_d, act_h)
    ctx.synchronize()

    var pix_t = TileTensor(_p(pix_d), row_major[B, PIX]())
    var act_t = TileTensor(_p(act_d), row_major[B, ACTIN]())

    # frozen WM forward → emb node; reinterpret (B, T·EMB) as (B·T, EMB)
    _ = wm.eval_loss(pix_t, act_t)
    var emb_ptr = wm.graph.node_out_ptr["emb"]()
    var emb_t = TileTensor(emb_ptr, row_major[DEC_BATCH, EMB]())

    # target patches from the same frames
    patchify["gpu", DEC_BATCH, IN_CH, IMG, PATCH_D](ctx, _p(pix_d), _p(tgt_d))
    var tgt_t = TileTensor(_p(tgt_d), row_major[DEC_BATCH, N_Q * PATCH_PX]())

    # COLD recon_into (no prior train_step) must not crash on an unset `tgt`
    # slot — mirrors the diagnostic/closed-loop usage (load weights → recon).
    var cold = ctx.enqueue_create_host_buffer[DT](DEC_BATCH * N_Q * PATCH_PX)
    ctx.synchronize()
    dec.recon_into(
        emb_t,
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](cold.unsafe_ptr()),
    )
    var cold_fin = True
    for i in range(DEC_BATCH * N_Q * PATCH_PX):
        var v = cold.unsafe_ptr()[i]
        if not (v == v):
            cold_fin = False
    assert_true(cold_fin, "cold recon_into runs (tgt slot bound)")
    print("   cold recon_into ok")

    print("train decoder on frozen emb ...")
    dec.reset_loss_accum()
    for _ in range(150):
        _ = dec.train_step(emb_t, tgt_t)
    var early = dec.read_loss_accum()
    dec.reset_loss_accum()
    for _ in range(150):
        _ = dec.train_step(emb_t, tgt_t)
    var late = dec.read_loss_accum()
    print("   recon_mse window1=", early, " window2=", late)
    assert_true(not (isnan(late) or isinf(late)), "recon loss finite")
    assert_true(late < early * Scalar[DT](0.6), "decoder learns from frozen emb")

    # recon_into → unpatchify → save grid (exercises the viz path)
    var recon_host = ctx.enqueue_create_host_buffer[DT](
        DEC_BATCH * N_Q * PATCH_PX
    )
    var recon_img = ctx.enqueue_create_host_buffer[DT](N_VIZ * IMG_DIM)
    ctx.synchronize()
    dec.recon_into(
        emb_t,
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](recon_host.unsafe_ptr()),
    )
    unpatchify["cpu", N_VIZ, IN_CH, IMG, PATCH_D](
        None,
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](recon_host.unsafe_ptr()),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](recon_img.unsafe_ptr()),
    )
    # originals already in pix_h (host) — first N_VIZ frames, CHW [0,1].
    save_reconstruction_grid(
        String("/tmp/lewm_decode_integration.ppm"),
        pix_h.unsafe_ptr(),
        recon_img.unsafe_ptr(),
        n=N_VIZ, height=IMG, width=IMG, channels=IN_CH, vmin=0.0, vmax=1.0,
    )

    _ = wm^; _ = dec^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
