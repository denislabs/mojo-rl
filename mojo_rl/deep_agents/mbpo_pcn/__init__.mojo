"""PCN-MBPO — fork of MBPO with PCN-trained world-model dynamics.

Replaces vanilla MBPO's probabilistic Swish ensemble (Gaussian NLL,
predicted mean+logvar, learnable bounds) with a deterministic PCN
dynamics ensemble trained via Phase-1's per-step local energy weight rule
(SGLD inference for the internal latent z + PC weight gradients per
block). Variance for imagination comes from ensemble disagreement, not
per-network logvar.

SAC side is untouched — actor, critic, replay, CUDA-graph capture all
preserved verbatim from `mojo_rl.deep_agents.core.agents.mbpo_agent`. The
only swaps live in:
- the dynamics struct fields on `*CPUState` / `*GPUState` (PCN instance
  wrappers from `nn_pc_v2`),
- `train_dynamics` / `do_model_rollouts` (CPU + GPU bodies),
- the rollout sample kernel (deterministic — no Gaussian sampling).

See `docs/PCN_MBRL_DESIGN.md` Phase 3 for context.
"""

from .pcn_mbpo_config import PCNMBPOConfig, DefaultPCNMBPOConfig
from .pcn_mbpo_agent import PCNMBPOAgent
