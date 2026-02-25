"""Ant Environment Package.

MuJoCo-style Ant using the physics3d Generalized Coordinates engine.

The Ant is a 3D quadruped (free joint root) with 4 legs, each having
a hip hinge and ankle hinge joint. 13 bodies, 9 joints, 8 actuators.

Example usage:
    from envs.ant import Ant
    from core import ContAction

    var env = Ant()
    var state = env.reset()

    var action = ContAction[8]()
    var result = env.step(action)
"""

from .ant import Ant
from .ant_config import AntConfig
from .ant_xml import AntModel
from .curriculum import AntCurriculum
