"""ACT ON LIBERO — the policy's SHAPE, written once and read by both ends.

    from mojo_rl.tasks.libero_act import LiberoActTrainer, LiberoActDataset

`examples/tasks/libero_act_train.mojo` fits it; `examples/tasks/
libero_eval_batched.mojo --act DIR` runs it in the loop with the cameras.
`tasks/bc_policy.mojo` exists for the same reason this file does: a network
each driver spells for itself is `_a_rule_written_inline_twice_drifts`, and
ACT has thirteen parameters to get wrong where the MLP had one.

## What is LIBERO's and what is the paper's

| | here | why |
|---|---|---|
| `QPOS` 9 | the 7 arm joints + 2 finger joints — robosuite's `low_dim` modality, the store's `qpos` column |
| `ADIM` 7 | OSC_POSE: six pose deltas + the gripper word |
| cameras | 2 x 128x128 — `agentview`, `eye_in_hand`, the recording's own |
| `K` 40 | **2.0 s at LIBERO's 20 Hz**. The paper chunks 100 steps at 50 Hz; the SO-101 run chunks 60 at 30 fps. The horizon is the quantity carried over, not the count — see `act/config.RUN_K`. |
| dim / ff / heads / latent / enc / dec | the `RUN_*` block: 256 / 1024 / 8 / 32 / 4 / 1 |
| lr / kl | 1e-5 / 10, both references' values |

⚠ `K` IS DERIVED FROM THE CONTROL CLOCK, not typed: `2 * LIBERO_CONTROL_FREQ`.
A family running at another rate would change it here and nowhere else.

⚠ CHANGING ANY OF THESE INVALIDATES EVERY CHECKPOINT. `ACTTrainer.load`
validates names and sizes; an old file raises rather than loading a different
network. The `norm.json` beside a checkpoint carries the store's statistics
and the eval refuses one whose widths are not this file's.

## The two stores, and which one a checkpoint was fitted on

    build/demos/libero_goal.rendered.h5   OURS   — frame r = state r, our tracer
    build/demos/libero_goal.h5            THEIRS — frame r = state r+1, robosuite's OpenGL

Both carry the same `qpos` / `action` / `task_index`, so a checkpoint from
either loads into the same network. The eval renders with OUR tracer, so a
checkpoint from the recorded store crosses the pixel-domain gap at test time —
that is the ARM that prices the gap, not a mistake, and the eval prints which
store the norm file names so the two cannot be confused in a table.
"""

from mojo_rl.data.libero_demos import (
    CAM_H, CAM_W, N_CAMS, ACTION_DIM, QPOS_DIM,
)
from mojo_rl.deep_agents.act.config import (
    RUN_DIM, RUN_HEADS, RUN_FF, RUN_LATENT, RUN_ENC_LAYERS, RUN_DEC_LAYERS,
    RUN_LR, ACT_DROPOUT, ACT_KL_WEIGHT,
)
from mojo_rl.deep_agents.act.trainer import ACTTrainer
from mojo_rl.deep_agents.act.data import ACTDataset
from mojo_rl.deep_agents.act.data_gpu import ACTDeviceDataset
from mojo_rl.tasks.libero_osc_config import LIBERO_CONTROL_FREQ


comptime LIBERO_ACT_QPOS: Int = QPOS_DIM
comptime LIBERO_ACT_ADIM: Int = ACTION_DIM
comptime LIBERO_ACT_N_CAM: Int = N_CAMS
comptime LIBERO_ACT_IMG_H: Int = CAM_H
comptime LIBERO_ACT_IMG_W: Int = CAM_W
comptime LIBERO_ACT_IMG_ELEMS: Int = N_CAMS * 3 * CAM_H * CAM_W

comptime LIBERO_ACT_HORIZON_S: Int = 2
"""The paper's two seconds."""
comptime LIBERO_ACT_K: Int = LIBERO_ACT_HORIZON_S * LIBERO_CONTROL_FREQ
comptime LIBERO_ACT_DIM: Int = RUN_DIM
comptime LIBERO_ACT_HEADS: Int = RUN_HEADS
comptime LIBERO_ACT_FF: Int = RUN_FF
comptime LIBERO_ACT_LATENT: Int = RUN_LATENT
comptime LIBERO_ACT_N_ENC: Int = RUN_ENC_LAYERS
comptime LIBERO_ACT_N_DEC: Int = RUN_DEC_LAYERS
comptime LIBERO_ACT_LR: Float64 = RUN_LR
comptime LIBERO_ACT_KL: Float64 = ACT_KL_WEIGHT

comptime LIBERO_ACT_STORE_RENDERED = "build/demos/libero_goal.rendered.h5"
comptime LIBERO_ACT_STORE_RECORDED = "build/demos/libero_goal.h5"

comptime LiberoActDataset = ACTDataset[
    LIBERO_ACT_QPOS, LIBERO_ACT_ADIM, LIBERO_ACT_N_CAM,
    LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W,
]
comptime LiberoActDeviceDataset = ACTDeviceDataset[
    LIBERO_ACT_QPOS, LIBERO_ACT_ADIM, LIBERO_ACT_N_CAM,
    LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W,
]
comptime LiberoActTrainer[BATCH: Int, target: StaticString] = ACTTrainer[
    LIBERO_ACT_QPOS, LIBERO_ACT_ADIM, LIBERO_ACT_N_CAM,
    LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W,
    LIBERO_ACT_K, LIBERO_ACT_DIM, LIBERO_ACT_HEADS, LIBERO_ACT_FF,
    LIBERO_ACT_LATENT, LIBERO_ACT_N_ENC, LIBERO_ACT_N_DEC,
    BATCH, ACT_DROPOUT, target,
]
"""The trainer fits it at its batch; the eval runs it at `LANES`."""
