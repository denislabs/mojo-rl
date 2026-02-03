"""Debug state size computation."""

from physics3d_v2.gpu.constants import (
    compute_state_size,
    JOINT_STATE_SIZE,
    SLIDE_JOINT_STATE_SIZE,
    BODY_STATE_SIZE,
    METADATA_SIZE,
    CONTACT_STATE_SIZE,
)
from envs.hopper_3d import Hopper3D

alias HopperEnv = Hopper3D[DType.float32]


fn main():
    print("GPU Constants:")
    print("  JOINT_STATE_SIZE:", JOINT_STATE_SIZE)
    print("  SLIDE_JOINT_STATE_SIZE:", SLIDE_JOINT_STATE_SIZE)
    print("  BODY_STATE_SIZE:", BODY_STATE_SIZE)
    print("  CONTACT_STATE_SIZE:", CONTACT_STATE_SIZE)
    print("  METADATA_SIZE:", METADATA_SIZE)
    print()
    print("Hopper3D constants:")
    print("  NUM_BODIES:", HopperEnv.NUM_BODIES)
    print("  MAX_CONTACTS:", HopperEnv.MAX_CONTACTS)
    print("  NUM_HINGE_JOINTS:", HopperEnv.NUM_HINGE_JOINTS)
    print("  NUM_SLIDE_JOINTS:", HopperEnv.NUM_SLIDE_JOINTS)
    print("  STATE_SIZE:", HopperEnv.STATE_SIZE)
    print()

    var expected = compute_state_size[
        HopperEnv.NUM_BODIES,
        HopperEnv.MAX_CONTACTS,
        HopperEnv.NUM_HINGE_JOINTS,
        HopperEnv.NUM_SLIDE_JOINTS,
    ]()
    print("Expected STATE_SIZE from compute_state_size():", expected)
    print()

    var manual = (
        HopperEnv.NUM_BODIES * BODY_STATE_SIZE
        + HopperEnv.MAX_CONTACTS * CONTACT_STATE_SIZE
        + HopperEnv.NUM_HINGE_JOINTS * JOINT_STATE_SIZE
        + HopperEnv.NUM_SLIDE_JOINTS * SLIDE_JOINT_STATE_SIZE
        + METADATA_SIZE
    )
    print("Manual calculation:", manual)
    print("  Bodies: ", HopperEnv.NUM_BODIES, "*", BODY_STATE_SIZE, "=", HopperEnv.NUM_BODIES * BODY_STATE_SIZE)
    print("  Contacts:", HopperEnv.MAX_CONTACTS, "*", CONTACT_STATE_SIZE, "=", HopperEnv.MAX_CONTACTS * CONTACT_STATE_SIZE)
    print("  Hinge:  ", HopperEnv.NUM_HINGE_JOINTS, "*", JOINT_STATE_SIZE, "=", HopperEnv.NUM_HINGE_JOINTS * JOINT_STATE_SIZE)
    print("  Slide:  ", HopperEnv.NUM_SLIDE_JOINTS, "*", SLIDE_JOINT_STATE_SIZE, "=", HopperEnv.NUM_SLIDE_JOINTS * SLIDE_JOINT_STATE_SIZE)
    print("  Meta:   ", METADATA_SIZE)
