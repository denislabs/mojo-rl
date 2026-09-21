"""Dreamer 4 — imagination-training agent inside a scalable world model.

Port of Hafner/Yan/Lillicrap 2025 (`docs/dreamer4.pdf`) onto the nn
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
from .shortcut_loss import dynamics_pretrain_loss, ShortcutDynamics, AgentDynamics
from .ode_sampler import sample_one_timestep
from .imag_rollout import imagine_rollout
from .task_embedder import TaskEmbedder
from .heads import (
    Dreamer4PolicyHead, Dreamer4RewardHead, Dreamer4ValueHead,
    Dreamer4ContinueHead,
)
from .bc_loss import bc_mtp_loss, bc_n_valid
from .imag_rl_loss import (
    lambda_returns,
    value_td_loss_cpu,
    value_td_loss_backward,
    pmpo_policy_loss_cpu,
    pmpo_policy_loss_backward,
    continue_pred,
    continue_bce_loss,
    continue_bce_backward,
)
from .agent import Dreamer4Agent
from .pong_reward_buffer import Dreamer4PongRewardBuffer
from .recon_loss import (
    masked_recon_loss, full_recon_psnr, masked_recon_grad_gpu,
)
from .patchify import temporal_patchify, temporal_unpatchify, downscale_box
