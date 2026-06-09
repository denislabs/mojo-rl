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

from .nets import MZRepNet, MZDynNet, MZPredNet


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
