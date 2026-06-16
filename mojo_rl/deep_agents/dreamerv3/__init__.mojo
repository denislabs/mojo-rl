"""DreamerV3 — world-model + actor-critic-in-imagination RL.

Ported against `references/dreamerv3-main/` (Hafner's Nature-2025
reimplementation). See `docs/DREAMERV3_PORTING_PLAN.md`. Built on the
nn framework + the PR-1 primitives (GELU / SiLU / RMSNorm / BlockLinear)
and the PR-3 `DreamerOpt` optimizer.
"""

from .onehot_kl import OneHotKL
from .normalize import PercentileNormalize
from .twohot import (
    twohot_pred, twohot_loss, twohot_loss_backward, symexp_twohot_bins,
)
from .dists import bounded_mean, bounded_std, normal_logp, normal_entropy
from .imag_loss import imag_loss_cpu, imag_loss_backward
from .repl_loss import repl_loss_cpu, repl_loss_backward

# The manual rssm/encoder/decoder/heads oracle (PR4/5b hand-written fwd/bwd)
# was retired 2026-05-29 once the ComputeGraph/Sequential versions matched
# the jax fixtures. The world model + actor-critic now live in graph form;
# import them directly from their modules (kept out of this re-export to
# avoid the heavy-struct __init__ compile-hang trap):
#   from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer
#   from mojo_rl.deep_agents.dreamerv3.agent   import DreamerV3Agent
#   from mojo_rl.deep_agents.dreamerv3.nets    import DreamerEncoder, ...
#   from mojo_rl.deep_agents.dreamerv3.wm      import WMCoreGraph, ...
