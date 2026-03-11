"""Shared utilities for RL algorithms.

This module provides common utility functions used across multiple
RL agents, avoiding code duplication and ensuring consistency.

Modules:
    - softmax: Numerically stable softmax for action sampling
    - gae: Generalized Advantage Estimation for policy gradient methods
    - normalization: Advantage normalization and statistics
    - shuffle: Fisher-Yates shuffle for minibatch sampling

Example usage:
    # Import from submodules to avoid naming conflicts
    from mojo_rl.core.utils.softmax import softmax, sample_from_probs
    from mojo_rl.core.utils.gae import compute_gae, compute_returns_from_advantages
    from mojo_rl.core.utils.normalization import normalize, compute_mean_std
    from mojo_rl.core.utils.shuffle import shuffle_indices

    # Or use module aliases
    from mojo_rl.core.utils import softmax as sm
    var probs = sm.softmax(logits)

Note:
    For Gaussian noise generation, use `nn.gpu.random.gaussian_noise()`
    which is co-located with other random utilities.
"""

# Re-export submodules for qualified access (e.g., utils.softmax.softmax)
from .softmax import (
    softmax,
    softmax_inline,
)
from .gae import (
    compute_gae,
    compute_gae_inline,
    compute_nstep_returns,
    compute_td_targets,
)
from .normalization import (
    normalize,
    normalize_inline,
    compute_mean,
    compute_std,
    compute_mean_std,
    RunningMeanStd,
)
from .shuffle import shuffle_indices, shuffle_indices_inline
