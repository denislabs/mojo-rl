"""Per-field tensor containers — the physics3d engine state.

`Data` — batched per-field simulation state (qpos/qvel/FK products/contacts/
meta/mocap, one packed tensor per region). `Model` — static model config as
one packed record tensor per family (bodies/joints/geoms/equality/...).
`SpecFields` — the actuation records, a bundle of its own because the
actuation kernels are the only readers. Plus the integrator scratch
containers. The flat slab era (state slab + model slab + offset tables) ended
at the fields sunset.
"""

from .data import Data
from .model import Model
from .spec_fields import (
    SpecFields,
    actuator_column,
    act_tendon_column,
    joint_limit_column,
)
from .dynamics_scratch import DynamicsScratch
from .contact_scratch import ContactScratch
from .rk4_scratch import Rk4Scratch
from .implicit_scratch import ImplicitScratch
