"""LeWM offline trainer on Pong pixels — GPU (scaled config).

Scaled config — meaningful test of whether the AdaLN-zero + pos embed
architecture can hold non-collapsed embeddings when given enough
headroom. Notable scale-ups from the §10.6 toy GPU run:

  - BATCH 4 → 16 (4× — gets us past Apple's kernel-launch overhead
    threshold; on NVIDIA, batch-amortization dominates)
  - T 4 → 6, H 3 → 4 (longer context + longer prediction window)
  - EMB 32 → 96 (3× spread headroom — addresses the "thin spike fits
    Gaussian" failure mode from §10.5/§10.6)
  - HIDDEN 32 → 96, ENC_HEADS 2 → 4, ENC_LAYERS 1 → 2 (deeper encoder)
  - PROJ_H 64 → 256, SMOOTHED 16 → 32 (projector + action embedder)
  - PRED_HEADS 2 → 4 (MSA in the predictor)

Expected memory:
  - Encoder params ~500k (12× toy); total ~700-800k.
  - Encoder workspace ~550k fp32 / sample × BATCH·T = 96 → ~210 MB.
  - Pixel batch: BATCH·T · 4·84·84 fp32 = 96 · 28k · 4 = 10.8 MB.
  Tractable on any GPU with ≥2 GB free.

If this OOMs on Apple M1 Pro, drop BATCH to 8 (or revert toy config —
see the comment block at the bottom).

Run:
    pixi run -e apple  mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_pixel_train_gpu.mojo

Expected throughput:
    Apple M1 Pro:  ~5-10 it/s  (kernel launches still dominate at small B)
    NVIDIA A100:   ~150+ it/s  (BMM-optimised attention + bigger batch)
"""

from mojo_rl.experimental.lewm.train_offline_gpu import train_lewm_offline_gpu


def main() raises:
    # =========================================================================
    # Scaled config — change `BATCH=8` if Apple Metal OOMs.
    # =========================================================================
    # NOTE on DEPTH: defaults to 1 below (single dual-branch cond_block).
    # To test paper-aligned multi-layer predictor, append `DEPTH=N` after
    # PRED_FF=256 (N=2 ~doubles param count + compile time; paper uses 6).
    # Cold-build time scales roughly linearly with DEPTH (3 min at D=1,
    # ~5-7 min at D=2 on Apple M1 Pro).
    train_lewm_offline_gpu[
        BATCH=16, T=6, H=4, N_PREDS=1,
        IN_CH=4, IMG=84, PATCH=14, N_PATCHES=36,
        HIDDEN=96, ENC_HEADS=4, ENC_LAYERS=2,
        EMB=96, PROJ_H=256,
        ACT=3, SMOOTHED=32,
        PRED_HEADS=4, PRED_FF=256,
    ](
        buffer_path=String("/tmp/lewm_pong_buffer.bin"),
        num_steps=8000,
        log_every=200,
        rng_seed=0xCAFE,
    )

    # =========================================================================
    # Toy config (§10.6 baseline) — use this if the scaled config OOMs:
    #
    # train_lewm_offline_gpu[
    #     BATCH=4, T=4, H=3, N_PREDS=1,
    #     IN_CH=4, IMG=84, PATCH=14, N_PATCHES=36,
    #     HIDDEN=32, ENC_HEADS=2, ENC_LAYERS=1, EMB=32, PROJ_H=64,
    #     ACT=3, SMOOTHED=16,
    #     PRED_HEADS=2, PRED_FF=64,
    # ](
    #     buffer_path=String("/tmp/lewm_pong_buffer.bin"),
    #     num_steps=1000,
    #     log_every=100,
    #     rng_seed=0xCAFE,
    # )
