"""`dm_control` `pendulum` domain (task: swingup).

Reference: references/dm_control-main/dm_control/suite/pendulum.py + .xml
"""

from .pendulum import DMPendulum, DMPendulumBatched
from .pendulum_config import DMPendulumConfig, COSINE_BOUND, ANGLE_BOUND_DEG
from .pendulum_xml import DMPendulumModel, POLE_BODY_IDX
