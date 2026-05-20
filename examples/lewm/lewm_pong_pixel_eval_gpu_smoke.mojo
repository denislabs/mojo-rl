"""LeWM eval-only driver — Pong, smoke config.

Loads the checkpoint written by `lewm_pong_pixel_train_gpu_smoke.mojo`
and runs only the eval phases (H6 action-shuffle, H7 closed-loop drift,
Phase 4b/4c open-loop MPC+CEM, Phase 4d receding-horizon MPC when
`rh_steps > 0`). No training — should finish in seconds.

CONFIG must match the training driver that wrote the checkpoint.

Run:
    pixi run -e apple mojo run -I . examples/lewm/lewm_pong_pixel_eval_gpu_smoke.mojo
"""

from mojo_rl.experimental.lewm.offline_trainer import eval_lewm_offline_gpu
from mojo_rl.experimental.lewm.lewm_config import LeWMPongViTConfig


def main() raises:
    eval_lewm_offline_gpu[LeWMPongViTConfig[]](
        buffer_path=String("/tmp/lewm_pong_buffer.bin"),
        checkpoint_path=String("/tmp/lewm_pong_smoke.ckpt"),
        eval_steps=3,
        eval_samples=8,
        eval_seed=0xBEEF,
        mpc_horizon=2,
        cem_iters=2,
        cem_samples=16,
        cem_topk=4,
        cem_smoothing=0.5,
        rh_steps=2,
    )
