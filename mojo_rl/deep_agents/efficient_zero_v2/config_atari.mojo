"""EfficientZeroV2 **Atari** config bundle — the spatial-latent model carrier.

Sibling of `EZV2DiscreteMLPConfig` (`config.mojo`): a type carrier (never
instantiated) that names the five EZv2 nets from one set of compile-time dims,
swapping in the spatial Atari backbone (`nets_atari.mojo`) while keeping the
flat `LATENT = 64·6·6 = 2304` interface so the agent facade / driver / planner
adapters instantiate with identical type-param structure (only the concrete net
types differ).

Defaults follow `references/EfficientZeroV2-main/ez/config/exp/atari.yaml`:
  * obs   = `FRAMES·3 × 96 × 96` stacked RGB frames (`image_based`, `gray_scale=False`)
  * LATENT = 2304 (spatial [64,6,6], `num_channels=64`, pinned)
  * BINS  = 601 (support range [-300,300], scale 1 — the Atari branch of
            `DiscreteSupport`; the yaml's `bins:51` is the DMC/Gym path only)
  * PROJ / PROJ_HID = 1024 / 1024  (`projection_layers: [1024, 1024]`)
  * BOTTLENECK = 256               (`prjection_head_layers: [256, 1024]`)

    comptime Cfg = EZV2AtariConfig[FRAMES=4, ACT=18]
    comptime Agent = EZv2DiscreteAgent[
        ENV, Cfg.Rep, Cfg.Dyn, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS, ...,
    ]
"""

from .nets import EZProjectorNet, EZPredictorNet
from .nets_atari import (
    EZRepNetResNetAtari, EZDynNetAtari, EZPredNetAtari, EZ_C, EZ_LATENT,
)


struct EZV2AtariConfig[
    FRAMES: Int,
    ACT: Int,
    LATENT: Int = EZ_LATENT,       # 2304 = [64,6,6], pinned
    BINS: Int = 601,
    PROJ: Int = 1024,
    PROJ_HID: Int = 1024,
    BOTTLENECK: Int = 256,
]:
    """EZv2 Atari spatial model bundle (RGB 96×96 pixel obs).

    Read the struct parameters directly (e.g. ``Cfg.OBS``) alongside the net
    aliases below. ``LATENT`` is pinned to the [64,6,6] spatial latent; the
    nets carry the conv geometry internally."""

    comptime IN_CH = Self.FRAMES * 3            # stacked RGB
    comptime OBS = Self.IN_CH * 96 * 96         # rep IN_DIMS[0]
    comptime Rep = EZRepNetResNetAtari[Self.IN_CH, EZ_C]
    comptime Dyn = EZDynNetAtari[Self.ACT, Self.BINS]
    comptime Pred = EZPredNetAtari[Self.ACT, Self.BINS]
    comptime Proj = EZProjectorNet[Self.LATENT, Self.PROJ, Self.PROJ_HID]
    comptime Predh = EZPredictorNet[Self.PROJ, Self.BOTTLENECK]
