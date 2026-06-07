"""Dreamer 4 — imagination-training agent inside a scalable world model.

Port of Hafner/Yan/Lillicrap 2025 (`docs/dreamer4.pdf`) onto the nn2
framework; see `docs/DREAMER4_PORT_PLAN.md`. Built on the Phase 0/1
primitives `MaskedAttention` / `ModalitySpaceAttention` (modality-gated space
attention), `TimeAttentionLatents` (causal time attention), `SwiGLU`,
`SpaceTimeTranspose`, and `SinusoidalPosAdd`.
"""

from .blocks import (
    Dreamer4SpaceSub,
    Dreamer4TimeSub,
    Dreamer4FFNSub,
    Dreamer4Block,
    Dreamer4BlockNoTime,
    Dreamer4Stack,
    Dreamer4Decoder,
)
from .encoder import Dreamer4Encoder
from .tokenizer import Dreamer4Tokenizer
from .dynamics import Dreamer4Dynamics
from .shortcut_loss import dynamics_pretrain_loss, ShortcutDynamics
from .ode_sampler import sample_one_timestep
from .task_embedder import TaskEmbedder
from .heads import Dreamer4PolicyHead, Dreamer4RewardHead
from .bc_loss import bc_mtp_loss, bc_n_valid
from .agent import Dreamer4Agent
from .recon_loss import (
    masked_recon_loss, full_recon_psnr, masked_recon_grad_gpu,
)
from .patchify import temporal_patchify, temporal_unpatchify, downscale_box
