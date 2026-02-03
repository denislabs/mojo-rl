"""GPU utilities for Generalized Coordinates engine.

This module provides GPU buffer layout constants and utilities for the GC engine.
"""

from .constants import (
    # GPU configuration
    TPB,
    TILE,
    # State buffer offsets
    gc_qpos_offset,
    gc_qvel_offset,
    gc_qacc_offset,
    gc_qfrc_offset,
    gc_xpos_offset,
    gc_xquat_offset,
    gc_xvel_offset,
    gc_xangvel_offset,
    gc_contacts_offset,
    gc_contact_offset,
    gc_metadata_offset,
    gc_state_size,
    # Contact layout
    GC_CONTACT_SIZE,
    GC_CONTACT_IDX_BODY_A,
    GC_CONTACT_IDX_BODY_B,
    GC_CONTACT_IDX_POS_X,
    GC_CONTACT_IDX_POS_Y,
    GC_CONTACT_IDX_POS_Z,
    GC_CONTACT_IDX_NX,
    GC_CONTACT_IDX_NY,
    GC_CONTACT_IDX_NZ,
    GC_CONTACT_IDX_DIST,
    GC_CONTACT_IDX_IMPULSE_N,
    GC_CONTACT_IDX_IMPULSE_T1,
    GC_CONTACT_IDX_IMPULSE_T2,
    # Metadata layout
    GC_METADATA_SIZE,
    GC_META_IDX_NUM_CONTACTS,
    # Model buffer layout
    GC_MODEL_BODY_SIZE,
    GC_BODY_IDX_MASS,
    GC_BODY_IDX_INV_MASS,
    gc_model_body_offset,
    GC_MODEL_JOINT_SIZE,
    gc_model_joint_offset,
    gc_model_metadata_offset,
    gc_model_size,
    # Geometry types
    GC_GEOM_PLANE,
    GC_GEOM_SPHERE,
    GC_GEOM_CAPSULE,
    GC_GEOM_BOX,
    # Joint types
    GC_JNT_FREE,
    GC_JNT_BALL,
    GC_JNT_SLIDE,
    GC_JNT_HINGE,
)

from .buffer_utils import (
    create_gc_state_buffer,
    create_gc_model_buffer,
    copy_model_to_buffer,
    copy_data_to_buffer,
    copy_buffer_to_data,
    free_gc_buffer,
)
