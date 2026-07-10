from .constraint_data import (
    ConstraintData,
    ConstraintRow,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
    CNSTR_FRICTION_TORSION,
    CNSTR_FRICTION_ROLL1,
    CNSTR_FRICTION_ROLL2,
    CNSTR_PYRAMID_EDGE,
    CNSTR_EQUALITY_CONNECT,
    CNSTR_EQUALITY_WELD,
)
# Legacy slab constraint builders (constraint_builder[_gpu]) were deleted at the
# P6 fields sunset — the fields path builds constraints inside
# `contact_solve` / `equality_tendon` / `limits`.
# `ConstraintData` (above) is kept: the shared `traits/solver` trait (used by
# physics2d) references it.
