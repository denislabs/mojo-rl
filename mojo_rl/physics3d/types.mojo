"""Physics3D shared type leaves.

The legacy struct-of-Lists `Model`/`Data` (MuJoCo-style CPU engine state) were
deleted at the G4 fields sunset — the engine state is `fields.ModelFields` /
`fields.DataFields` (packed per-record tensors). What remains here are the
tiny shared leaves: `_max_one`, the equality-constraint type codes, and
`ConeType`.
"""


from .joint_types import JointDef, JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from .joint_types import get_joint_qpos_size, get_joint_qvel_size

# Actuator dynamics type constants
@always_inline
def _max_one[n: Int]() -> Int:
    """Clamp a comptime size to >= 1 (zero-entity buffers still allocate)."""
    return n if n > 0 else 1


# Equality constraint types (MuJoCo mjtEq subset)
comptime EQ_CONNECT: Int = 0  # Point-to-point ball joint (3 position rows)
comptime EQ_WELD: Int = 1  # Rigid attachment (3 position + 3 orientation rows)
comptime EQ_TENDON: Int = 2  # Fixed tendon (1 bilateral row)


struct ConeType:
    comptime PYRAMIDAL: Int = 0
    comptime ELLIPTIC: Int = 1
