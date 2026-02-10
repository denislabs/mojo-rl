"""Robot definition package for compile-time body/joint specifications.

Provides BodySpec and JointSpec traits with concrete implementations
(CapsuleBody, SphereBody, BoxBody, HingeJoint, SlideJoint) and a
RobotDef compositor for composing full robot definitions.
"""

from .body_spec import BodySpec, CapsuleBody, SphereBody, BoxBody
from .joint_spec import JointSpec, HingeJoint, SlideJoint
from .robot_def import Bodies, Joints, RobotDef
