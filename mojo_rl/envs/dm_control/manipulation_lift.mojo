"""`dm_control` `manipulation/lift_*_features` env facades.

    lift_large_box = DMLiftLargeBox[DTYPE]

⚠ `lift_brick_features` is the SAME `Lift` task class with a Duplo prop, which
is a different model (62 geoms of studs instead of one box) and a different
workspace (`_DUPLO_WORKSPACE` drops the brick from z = 0 rather than placing it
at rest). It is not this config with a flag.
"""

from .manipulation_lift_box_def import LiftLargeBoxModel
from .manipulation_lift_box_config import LiftLargeBoxConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMLiftLargeBox[DTYPE: DType = DType.float64] = Phyics3dEnv[
    LiftLargeBoxModel, LiftLargeBoxConfig, DTYPE, False
]
