"""LeWM PushT — nsys profiling driver.

Same comptime config as the scaled training driver
(`lewm_pusht_pixel_train_gpu_v2.mojo`) so the kernel landscape, batch
sizes, and shape-dependent autotuning are identical to production. But:

- `num_steps=50` (not 8000) — long enough to clear warmup and produce
  representative steady-state metrics, short enough to keep the nsys
  trace manageable (~50 MB instead of multi-GB).
- `eval_steps=0` — skip H6/H7/MPC/CEM phases (very different per-step
  shape than training, would dilute the profile).
- No `checkpoint_path` — checkpoint writes add ~2-3s per save that
  would skew per-step measurements.
- `log_every=10` — fewer host-side prints than the default 200.

Run (NVIDIA, capture trace + inline stats):

    nsys profile \\
        -t cuda,nvtx,osrt \\
        --stats=true \\
        -o lewm_pusht_profile \\
        pixi run -e nvidia mojo run -I . \\
            examples/lewm/lewm_pusht_pixel_profile_gpu_v2.mojo

Then open `lewm_pusht_profile.nsys-rep` in Nsight Systems.

What to look for to confirm "data is the bottleneck":

1. **CPU↔GPU gap per step**: in the timeline, the GPU should be idle
   between steps while the CPU runs `sample_batch_fp32`. Each gap is
   one batch's host-side work (HDF5 reads + uint8→fp32 + permute).
2. **`pread64` / `read` syscalls**: with `-t osrt`, the OS runtime
   trace shows HDF5 disk reads. 16 reads per step (one per batch
   element) is the smoking gun.
3. **Memcpy H2D timing**: `enqueue_copy pixels_host→pixels_buf` should
   be fast (~ms) since it's pinned-DMA. If this dominates, the
   redundant host→host memcpy in train_step (line 871) is the culprit.
4. **GPU kernel utilization**: if total GPU-kernel time per step is
   much less than wall time per step, the GPU is sitting idle waiting
   for the CPU pipeline — that's the bottleneck.

For a quick "everything-on-GPU baseline" comparison, also profile the
scaled Pong driver (which uses an in-RAM uint8 buffer, no disk I/O):

    nsys profile -t cuda,nvtx,osrt --stats=true -o lewm_pong_profile \\
        pixi run -e nvidia mojo run -I . \\
            examples/lewm/lewm_pong_pixel_train_gpu_v2.mojo

The CPU↔GPU gap difference between the two profiles is exactly the
PushT-specific cost (disk + larger frame size).
"""

from mojo_rl.experimental.lewm.trainer_struct import (
    train_lewm_offline_gpu_pusht_v2,
)


def main() raises:
    train_lewm_offline_gpu_pusht_v2[
        BATCH=16, T=4, H=3, N_PREDS=1,
        IN_CH=3, IMG=224, PATCH=14, N_PATCHES=256,
        HIDDEN=96, ENC_HEADS=4, ENC_LAYERS=2,
        EMB=96, PROJ_H=256,
        ACT=10, SMOOTHED=32,
        PRED_HEADS=4, PRED_FF=256,
        DEPTH=6,
        FRAMESKIP=5, ACTION_DIM=2,
    ](
        num_steps=50,
        log_every=10,
        rng_seed=0xCAFE,
        eval_steps=0,
        eval_samples=0,
        mpc_horizon=0,
        cem_iters=0,
    )
