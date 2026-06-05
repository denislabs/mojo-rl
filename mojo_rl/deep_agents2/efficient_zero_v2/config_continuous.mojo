"""EfficientZeroV2 continuous config bundle — names every torso from one set of
dims, mirroring `EZV2DiscreteMLPConfig` and the legacy
``EZV2ContinuousMLPConfig``.

A type carrier (never instantiated): it exposes the five EZv2 continuous nets —
the MuZero learned model ``Rep`` / ``Dyn`` (reused verbatim; the continuous
dynamics input is ``[z | action_vector]``) plus the squashed-Gaussian
``Pred`` head (``EZContPredNet``: ``f → [μ_raw | σ_raw | value]``) and the
SimSiam ``Proj`` / ``Predh`` — from one set of compile-time dims, read at the
`EZv2ContinuousAgent` instantiation site:

    comptime Cfg = EZV2ContinuousMLPConfig[
        OBS=3, ACT_DIM=1, LATENT=64, HIDDEN=64, BINS=51,
        PROJ=128, PROJ_HID=128, BOTTLENECK=64,
    ]
    comptime Agent = EZv2ContinuousAgent[
        PendulumEnv[DType.float32],
        Cfg.Rep, Cfg.Dyn, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        Cfg.OBS, Cfg.ACT_DIM, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=32, MAX_NODES=128, K_ROOT=16, K_NON_ROOT=8,
        CAP=50000, B=128, K=5, N=5,
    ]

``Rep``/``Dyn``/``Pred`` share ``LATENT`` and ``BINS`` (categorical reward + value
over ``[v_min, v_max]``); the projector ``Proj`` takes ``LATENT`` in and emits
``PROJ``, which the predictor ``Predh`` round-trips through ``BOTTLENECK``. The
prediction head ``OUT_DIM = 2·ACT_DIM + BINS`` matches the
`SampledGumbelGPUMCTS` ``PredictionGPU`` contract.
"""

from .nets import MZRepNet, MZDynNet, EZProjectorNet, EZPredictorNet
from .nets_continuous import EZContPredNet


struct EZV2ContinuousMLPConfig[
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int = 64,
    HIDDEN: Int = 64,
    BINS: Int = 51,
    PROJ: Int = 128,
    PROJ_HID: Int = 128,
    BOTTLENECK: Int = 64,
]:
    """Standard EZv2 continuous MLP model bundle for proprioceptive (vector)
    observations.

    Read the struct parameters directly (e.g. ``Cfg.OBS``) alongside the net
    aliases below."""

    comptime Rep = MZRepNet[Self.OBS, Self.LATENT, Self.HIDDEN]
    comptime Dyn = MZDynNet[Self.LATENT, Self.ACT_DIM, Self.BINS, Self.HIDDEN]
    comptime Pred = EZContPredNet[
        Self.LATENT, Self.ACT_DIM, Self.BINS, Self.HIDDEN
    ]
    comptime Proj = EZProjectorNet[Self.LATENT, Self.PROJ, Self.PROJ_HID]
    comptime Predh = EZPredictorNet[Self.PROJ, Self.BOTTLENECK]
