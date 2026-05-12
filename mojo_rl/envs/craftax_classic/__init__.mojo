"""Craftax-Classic environment.

Mojo port of `craftax_classic` from the Craftax JAX benchmark
(Matthews et al. 2024, ICML). See `docs/CRAFTAX_PORT.md`.

Phase 1: skeleton — trait conformance + no-op step path.
"""

from .craftax_classic import CraftaxClassicEnv, CraftaxState, CraftaxAction
from .constants import (
    MAP_H,
    MAP_W,
    NUM_ACTIONS,
    NUM_ACHIEVEMENTS,
    NUM_BLOCK_TYPES,
    OBS_DIM,
)
from .state import STATE_SIZE
