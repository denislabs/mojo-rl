"""LeWM PushT — nsys profiling driver (paper width + paper batch).

Profiles a 50-step training pass at the paper-aligned config:

  - **Width**: HIDDEN=192, ENC_LAYERS=12, EMB=192, PROJ_H=2048,
    PRED_HEADS=16, PRED_FF=2048 — matches `references/le-wm-main/
    config/train/lewm.yaml` except for `dim_head` (paper uses
    `dim_head=64`, ours is implicit `192/16=12` — see below).
  - **Batch**: 32 (paper is 128, but 128 and 64 both OOM on a 24GB GPU
    at paper width — activations at hidden=192 × 12 enc layers × 256
    patches + depth=6 predictor blocks are too large in fp32).
    Mixed-precision (bf16 activations + fp32 reductions/optimizer) is
    on the roadmap and would unlock paper batch=128; not landed yet
    (separate 1-2 week task). 32 is what fits in fp32 today. The prior
    32k-step training run used BATCH=16 because the sampler was the
    bottleneck; with the sampler now ~20 it/s and GPU-bound, 32 is
    enough to characterize the GPU-bound regime.
  - **T/H/depth**: 6 / 3 / 6 — matches the long + paper-width runs.

`num_steps=50` is enough to clear warmup and produce steady-state
phase averages. `eval_steps=0` skips H6/H7/MPC. No checkpoint write.

Known divergence from paper: predictor MSA `dim_head` is implicit
`HIDDEN/PRED_HEADS=12` here vs paper's explicit `64` (heads=16 ×
dim_head=64 = internal MSA width 1024). That's a separate
architectural change (`MultiHeadAttentionXL` composite + new
`PRED_DIM_HEAD` config field, ~1-2 days) and is NOT in this profile.
The numbers below should match the production training run's per-step
cost minus that MSA capacity gap.

Run (NVIDIA, capture trace + inline stats):

    nsys profile \\
        -t cuda,nvtx,osrt \\
        --stats=true \\
        -o lewm_pusht_paper_profile \\
        pixi run -e nvidia mojo run -I . \\
            examples/lewm/lewm_pusht_pixel_profile_gpu.mojo

Then open `lewm_pusht_paper_profile.nsys-rep` in Nsight Systems.

What to look for to find optimization headroom:

1. **GPU vs sampler split per step**: at BATCH=16 + paper width the
   GPU was ~80% of step time (197ms / 248ms total). At BATCH=128 the
   GPU compute scales sublinearly while sampler stays at ~50ms — GPU
   should now be ~90-95% of step time. If sampler is still a visible
   gap, that's a pipeline-overlap opportunity.
2. **Encoder vs predictor split**: ENC at paper width has 12 layers
   × 256 patches × hidden=192 ≈ dominant compute. If a single layer
   shows up as >20% of step time, that's a fusion target.
3. **AdaLN + MSA + MLP per-block timing**: the predictor has DEPTH=6
   conditional blocks. Each block is AdaLN(modulate) → MSA → AdaLN →
   MLP. At paper width MLP (192→2048→192) likely dominates a block.
4. **Memory bandwidth vs compute**: check the `compute_throughput`
   vs `memory_throughput` columns in nsys's kernel summary. Paper
   width's PRED_FF=2048 GEMMs should be compute-bound. If memory-
   bound, there's room for fusion or pipelining.
5. **Kernel launch overhead**: at BATCH=128 each forward is heavier,
   so launches amortize better than at BATCH=16. Compare `launch %`
   in the GPU summary against the BATCH=16 baseline profile.

For a head-to-head with the BATCH=16 baseline:

    git stash; nsys profile -o lewm_pusht_paper_b16_profile \\
        pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_pixel_profile_gpu.mojo
    git stash pop

(or revert `batch=128` in the config below to `batch=16` temporarily).
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    train_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    train_lewm_offline_gpu_pusht[LeWMPushTViTConfig[
        batch=32, t=6, h=3,
        hidden=192, enc_heads=3, enc_layers=12,
        emb=192, proj_h=2048,
        pred_heads=16, pred_ff=2048,
        depth=6,
    ]](
        num_steps=50,
        log_every=10,
        rng_seed=0xCAFE,
        eval_steps=0,
        eval_samples=0,
        mpc_horizon=0,
        cem_iters=0,
        time_phases=True,
    )
