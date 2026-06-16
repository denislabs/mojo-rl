"""EfficientZeroV2 discrete config bundle — names every torso from one set of
dims, mirroring the legacy ``EZV2DiscreteMLPConfig``.

A type carrier (never instantiated): it exposes the five EZv2 discrete nets —
the MuZero learned model ``Rep`` / ``Dyn`` / ``Pred`` plus the SimSiam
``Proj`` / ``Predh`` — from one set of compile-time dims, read at the
``EZv2DiscreteAgent`` instantiation site:

    comptime Cfg = EZV2DiscreteMLPConfig[
        OBS=4, ACT=2, LATENT=64, HIDDEN=128, BINS=51,
        PROJ=128, PROJ_HID=128, BOTTLENECK=64,
    ]
    comptime Agent = EZv2DiscreteAgent[
        CartPoleEnv[DType.float64],
        Cfg.Rep, Cfg.Dyn, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=25, MAX_NODES=128, CAP=50000, B=64, K=5, N=10,
    ]

``Rep``/``Dyn``/``Pred`` share ``LATENT`` and ``BINS`` (categorical reward + value
over ``[v_min, v_max]``); the projector ``Proj`` takes ``LATENT`` in and emits
``PROJ``, which the predictor ``Predh`` round-trips through ``BOTTLENECK``.
"""

from .nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)


struct EZV2DiscreteMLPConfig[
    OBS: Int,
    ACT: Int,
    LATENT: Int = 128,
    HIDDEN: Int = 128,
    BINS: Int = 51,
    PROJ: Int = 256,
    PROJ_HID: Int = 256,
    BOTTLENECK: Int = 128,
]:
    """Standard EZv2 discrete MLP model bundle for clean (vector) observations.

    Read the struct parameters directly (e.g. ``Cfg.OBS``) alongside the net
    aliases below."""

    comptime Rep = MZRepNet[Self.OBS, Self.LATENT, Self.HIDDEN]
    comptime Dyn = MZDynNet[Self.LATENT, Self.ACT, Self.BINS, Self.HIDDEN]
    comptime Pred = MZPredNet[Self.LATENT, Self.ACT, Self.BINS, Self.HIDDEN]
    comptime Proj = EZProjectorNet[Self.LATENT, Self.PROJ, Self.PROJ_HID]
    comptime Predh = EZPredictorNet[Self.PROJ, Self.BOTTLENECK]
