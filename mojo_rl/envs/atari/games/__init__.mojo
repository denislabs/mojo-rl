"""Game-specific definitions for Atari 2600 games.

Each game module provides functions to extract score, lives, and
terminal status from the Atari 2600's 128-byte RAM.
"""

from .helpers import get_decimal_score, get_decimal_score_2, get_decimal_score_3
from .pong import PongDef
from .breakout import BreakoutDef
from .space_invaders import SpaceInvadersDef
