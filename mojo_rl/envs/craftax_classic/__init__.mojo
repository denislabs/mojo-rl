"""Craftax-Classic environment.

Mojo port of `craftax_classic` from the Craftax JAX benchmark
(Matthews et al. 2024, ICML). See `docs/CRAFTAX_PORT.md`.

Phase 1: skeleton — trait conformance + no-op step path.
"""

from .craftax_classic import CraftaxClassicEnv, CraftaxState, CraftaxAction
from .craftax_classic_pixel import (
    CraftaxClassicPixelEnv,
    PIXEL_OBS_DIM,
    BLOCK_PIXEL_SIZE,
    INVENTORY_OBS_HEIGHT,
    OBS_PIX_H,
    OBS_PIX_W,
    OBS_CHANNELS,
)
from .constants import (
    MAP_H,
    MAP_W,
    NUM_ACTIONS,
    NUM_ACHIEVEMENTS,
    NUM_BLOCK_TYPES,
    OBS_DIM,
)
from .state import STATE_SIZE
