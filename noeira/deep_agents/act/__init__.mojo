"""ACT — Action Chunking with Transformers (arXiv:2304.13705).

A CVAE whose decoder is a DETR-style transformer: from the current joint
positions and camera images it predicts the next `k` actions in one shot, and
at inference overlapping chunks are combined by an exponentially-weighted
temporal ensemble.

Reference: `references/act-main/` (paper: `docs/ACT.pdf`).
Target robot: SO-ARM101, from a LeRobot v3 dataset.
"""

from .config import (
    ACT_BATCH,
    ACT_CHUNK,
    ACT_DEC_LAYERS,
    ACT_DROPOUT,
    ACT_ENC_LAYERS,
    ACT_EPOCHS,
    ACT_FF,
    ACT_HEADS,
    ACT_HIDDEN,
    ACT_KL_WEIGHT,
    ACT_LATENT,
    ACT_LR,
    ACT_TEMPORAL_ENSEMBLE_M,
    ACT_USE_LAST_HS,
    ACT_WEIGHT_DECAY,
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from .data import ACTDataset
from .inference import TemporalEnsemble, denormalize
from .layers import (
    DETRDecoderLayer,
    DETREncoderLayer,
    DETREncoderLayerMasked,
)
from .loss_graph import ACTLossGraph
from .trainer import (
    ACTMetricAccum,
    ACTStepResult,
    ACTTrainer,
    ACTWindowMetrics,
)
