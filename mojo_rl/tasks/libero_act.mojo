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
| `QPOS` 19 | the 7 arm joints + 2 finger joints — robosuite's `low_dim` modality — **plus a one-hot of the task over libero_goal's ten**, the store's `qpos` column |
| `ADIM` 7 | OSC_POSE: six pose deltas + the gripper word |
| cameras | 2 x 128x128 — `agentview`, `eye_in_hand`, the recording's own |
| `K` 40 | **2.0 s at LIBERO's 20 Hz**. The paper chunks 100 steps at 50 Hz; the SO-101 run chunks 60 at 30 fps. The horizon is the quantity carried over, not the count — see `act/config.RUN_K`. |
| dim / ff / heads / latent / enc / dec | the `RUN_*` block: 256 / 1024 / 8 / 32 / 4 / 1 |
| backbone | **ResNet18 cut after layer3** (`ResNet18Layer3Backbone`): 8x8 = 64 tokens per camera at 256 channels, ImageNet layers 1-3. The full trunk gives 4x4 at 128x128 and the fits on it solved only the task whose target never moves (5/20 stove, 0/20 the nine others) |
| lr / kl | 1e-5 / 10, both references' values |

⚠⚠ THE TASK IS AN INPUT BECAUSE THE PICTURE CANNOT CARRY IT. Every
libero_goal task is the same scene at the same layout; the first fit on nine
proprio words (2026-09-18) learned the mean of ten behaviours — position
deltas a tenth of the demonstrations', the gripper at 0.6 — and scored 0/200
with the ensemble on or off. The one-hot is LIBERO's task embedding for a
fixed set of tasks; the eval builds it from the lane's `row_task`.

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
    CAM_H, CAM_W, N_CAMS, ACTION_DIM, QPOS_PROPRIO, QPOS_WORDS,
)
from mojo_rl.deep_agents.act.config import (
    RUN_DIM, RUN_HEADS, RUN_FF, RUN_LATENT, RUN_ENC_LAYERS, RUN_DEC_LAYERS,
    RUN_LR, ACT_DROPOUT, ACT_KL_WEIGHT,
)
from mojo_rl.deep_agents.act.trainer import ACTTrainer
from mojo_rl.nn.models.resnet18 import (
    ResNet18Layer3Backbone, ResNet18L3OutH, ResNet18L3OutW, RESNET18_L3_OUT_CH,
)
from mojo_rl.deep_agents.act.data import ACTDataset
from mojo_rl.deep_agents.act.data_gpu import ACTDeviceDataset
from mojo_rl.tasks.libero_osc_config import LIBERO_CONTROL_FREQ


comptime LIBERO_GOAL_N_TASKS: Int = 10
comptime LIBERO_ACT_PROPRIO: Int = QPOS_PROPRIO
comptime LIBERO_ACT_QPOS: Int = QPOS_WORDS + LIBERO_GOAL_N_TASKS
"""9 proprio words + their 9 one-step differences + the task one-hot. A store
from another suite (or the 19-wide stores of 2026-09-18/19) has another width
and `ACTDataset` refuses it by name and size. The differences are the phase
the picture does not carry — see `libero_demos.mojo`'s header and the
vision-free control at 0.60."""
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
comptime LIBERO_ACT_FEAT_CH: Int = RESNET18_L3_OUT_CH
comptime LIBERO_ACT_OH: Int = ResNet18L3OutH[LIBERO_ACT_IMG_H]
comptime LIBERO_ACT_OW: Int = ResNet18L3OutW[LIBERO_ACT_IMG_W]
comptime LiberoActBackbone = ResNet18Layer3Backbone[
    3, LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W
]
comptime LiberoActTrainer[BATCH: Int, target: StaticString] = ACTTrainer[
    LIBERO_ACT_QPOS, LIBERO_ACT_ADIM, LIBERO_ACT_N_CAM,
    LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W,
    LIBERO_ACT_K, LIBERO_ACT_DIM, LIBERO_ACT_HEADS, LIBERO_ACT_FF,
    LIBERO_ACT_LATENT, LIBERO_ACT_N_ENC, LIBERO_ACT_N_DEC,
    BATCH, ACT_DROPOUT, target,
    FEAT_CH=LIBERO_ACT_FEAT_CH, OH=LIBERO_ACT_OH, OW=LIBERO_ACT_OW,
    BACKBONE=LiberoActBackbone,
]
"""The trainer fits it at its batch; the eval runs it at `LANES`."""
