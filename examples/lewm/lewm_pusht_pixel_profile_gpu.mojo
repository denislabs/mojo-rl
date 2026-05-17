"""LeWM PushT — nsys profiling driver.

Same CONFIG as the scaled training driver
(`lewm_pusht_pixel_train_gpu.mojo`) so the kernel landscape, batch
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
            examples/lewm/lewm_pusht_pixel_profile_gpu.mojo

Then open `lewm_pusht_profile.nsys-rep` in Nsight Systems.

What to look for to confirm "data is the bottleneck":

1. **CPU↔GPU gap per step**: in the timeline, the GPU should be idle
   between steps while the CPU runs `sample_batch_uint8`. Each gap is
   one batch's host-side work (HDF5 reads + GPU uint8→fp32 conversion).
2. **`pread64` / `read` syscalls**: with `-t osrt`, the OS runtime
   trace shows HDF5 disk reads.
3. **GPU kernel utilization**: if total GPU-kernel time per step is
   much less than wall time per step, the GPU is sitting idle waiting
   for the CPU pipeline — that's the bottleneck.

For a quick "everything-on-GPU baseline" comparison, also profile the
scaled Pong driver (which uses an in-RAM uint8 buffer, no disk I/O):

    nsys profile -t cuda,nvtx,osrt --stats=true -o lewm_pong_profile \\
        pixi run -e nvidia mojo run -I . \\
            examples/lewm/lewm_pong_pixel_train_gpu.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import (
    train_lewm_offline_gpu_pusht,
)
from mojo_rl.experimental.lewm.lewm_config import LeWMPushTViTConfig


def main() raises:
    train_lewm_offline_gpu_pusht[LeWMPushTViTConfig[
        batch=16, t=4, h=3,
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
