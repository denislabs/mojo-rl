"""Articulated body chain support for 2D physics.

This module provides utilities for multi-body articulated chains
such as those used in planar locomotion environments (Hopper, Walker2d, HalfCheetah).
"""

from .constants import (
    CHAIN_BODY_COUNT,
    CHAIN_JOINT_COUNT,
    CHAIN_ROOT_IDX,
    CHAIN_HEADER_SIZE,
    LINK_DATA_SIZE,
    LINK_PARENT_IDX,
    LINK_JOINT_IDX,
    LINK_LENGTH,
    LINK_WIDTH,
    DEFAULT_KP,
    DEFAULT_KD,
    DEFAULT_MAX_TORQUE,
    HOPPER_NUM_BODIES,
    HOPPER_NUM_JOINTS,
    HOPPER_OBS_DIM,
    HOPPER_ACTION_DIM,
    WALKER_NUM_BODIES,
    WALKER_NUM_JOINTS,
    WALKER_OBS_DIM,
    WALKER_ACTION_DIM,
    CHEETAH_NUM_BODIES,
    CHEETAH_NUM_JOINTS,
    CHEETAH_OBS_DIM,
    CHEETAH_ACTION_DIM,
)

from .chain import (
    LinkDef,
    compute_link_inertia,
    ArticulatedChain,
)
