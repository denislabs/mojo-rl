"""DreamerV3 — world-model + actor-critic-in-imagination RL.

Ported against `references/dreamerv3-main/` (Hafner's Nature-2025
reimplementation). See `docs/DREAMERV3_PORTING_PLAN.md`. Built on the
nn2 framework + the PR-1 primitives (GELU / SiLU / RMSNorm / BlockLinear)
and the PR-3 `DreamerOpt` optimizer.
"""

from .lambda_return import lambda_return_cpu, lambda_return_gpu
from .onehot_kl import OneHotKL
from .normalize import PercentileNormalize
from .rssm import RSSM
from .encoder import Encoder
from .decoder import Decoder
from .twohot import (
    twohot_pred, twohot_loss, twohot_loss_backward, symexp_twohot_bins,
)
from .dists import bounded_mean, bounded_std, normal_logp, normal_entropy
from .slow_head import SlowModelHead, polyak_mix
from .imag_loss import imag_loss_cpu, imag_loss_backward
from .repl_loss import repl_loss_cpu, repl_loss_backward
from .heads import RewardHead, ContHead
