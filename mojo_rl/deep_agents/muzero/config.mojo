"""MuZero config bundle — names the three learned-model torsos from one set of
dims, mirroring the legacy ``MuZeroMLPConfig``.

A config is purely a *type carrier*: it exposes ``Rep`` / ``Dyn`` / ``Pred`` (the
`MZRepNet` / `MZDynNet` / `MZPredNet` aliases) plus the scalar dims the self-play
driver needs as explicit comptime params (the driver can't recover them from the
``Module`` trait). It owns no state and is never instantiated — read its members
at the `MuZeroAgent` instantiation site:

    comptime Cfg = MuZeroMLPConfig[OBS=4, ACT=2, LATENT=64, HIDDEN=128, BINS=51]
    comptime Agent = MuZeroAgent[
        "cpu", CartPoleEnv[DType.float64],
        Cfg.Rep, Cfg.Dyn, Cfg.Pred,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=25, MAX_NODES=128, CAP=50000, B=64, K=5, N=10,
    ]

``Rep``/``Dyn``/``Pred`` share ``LATENT`` and ``BINS`` so the categorical reward
(``Dyn``) and value (``Pred``) heads agree with `zero/twohot_targets.mojo` and the
planner support — keep ``v_min``/``v_max`` in sync on the agent.
"""

from .nets import MZRepNet, MZRepNetCNN, MZDynNet, MZPredNet


struct MuZeroMLPConfig[
    OBS: Int,
    ACT: Int,
    LATENT: Int = 128,
    HIDDEN: Int = 128,
    BINS: Int = 51,
]:
    """Standard MuZero MLP model bundle for clean (vector) observations.

    ``OBS`` / ``ACT`` / ``LATENT`` / ``HIDDEN`` / ``BINS`` are the struct
    parameters — read them directly (e.g. ``Cfg.OBS``) alongside the net
    aliases below."""

    comptime Rep = MZRepNet[Self.OBS, Self.LATENT, Self.HIDDEN]
    comptime Dyn = MZDynNet[Self.LATENT, Self.ACT, Self.BINS, Self.HIDDEN]
    comptime Pred = MZPredNet[Self.LATENT, Self.ACT, Self.BINS, Self.HIDDEN]


struct MuZeroCNNConfig[
    FRAMES: Int,
    ACT: Int,
    LATENT: Int = 128,
    HIDDEN: Int = 512,
    BINS: Int = 51,
]:
    """MuZero model bundle for ``FRAMES``×84×84 stacked-frame **pixel** obs.

    The representation ``Rep`` is the Nature-CNN `MZRepNetCNN` (84→20→9→7
    convolutional backbone → latent); ``Dyn`` / ``Pred`` are the same
    latent-space `MZDynNet` / `MZPredNet` as the MLP config — the learned model
    is identical once the observation has been encoded, so only ``Rep`` differs.

    Spatial dims are fixed at 84×84, so ``OBS = FRAMES·84·84`` is derived (the
    self-play driver reads ``Cfg.OBS`` like the MLP config). ``HIDDEN`` defaults
    to 512 (the Nature-CNN projection width); ``LATENT`` / ``BINS`` keep the
    MuZero defaults so reward (``Dyn``) and value (``Pred``) heads agree with
    `zero/twohot_targets.mojo` and the planner support — keep ``v_min``/``v_max``
    in sync on the agent.

        comptime Cfg = MuZeroCNNConfig[FRAMES=4, ACT=3, LATENT=128, BINS=51]
        # Cfg.OBS == 4*84*84 == 28224
    """

    comptime OBS = Self.FRAMES * 84 * 84
    comptime Rep = MZRepNetCNN[Self.FRAMES, Self.LATENT, Self.HIDDEN]
    comptime Dyn = MZDynNet[Self.LATENT, Self.ACT, Self.BINS, Self.HIDDEN]
    comptime Pred = MZPredNet[Self.LATENT, Self.ACT, Self.BINS, Self.HIDDEN]
