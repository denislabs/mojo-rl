"""Physics3D v2 GPU constants - Flat buffer layout for GPU kernels.

Since we can't use structs inside GPU kernels, all state is stored in
a flat LayoutTensor[BATCH, STATE_SIZE]. This file defines the offsets
for each field within the state buffer.

Layout per environment (single body):
  [0-6]:   qpos    = [x, y, z, qx, qy, qz, qw]     (7 floats)
  [7-12]:  qvel    = [vx, vy, vz, wx, wy, wz]      (6 floats)
  [13-18]: qacc    = [ax, ay, az, αx, αy, αz]      (6 floats)
  [19-24]: qfrc    = [fx, fy, fz, τx, τy, τz]      (6 floats)
  [25-27]: xpos    = [x, y, z] (world frame)       (3 floats)
  [28-35]: contact = [active, depth, nx, ny, nz, px, py, pz] (8 floats)

  Total: 36 floats per environment
"""

# =============================================================================
# GPU Configuration
# =============================================================================

comptime TPB: Int = 256  # Threads per block (optimal for most GPUs)


# =============================================================================
# State Buffer Layout - Section Offsets
# =============================================================================

# Position (generalized coordinates) - 7 floats
comptime QPOS_OFFSET: Int = 0
comptime QPOS_SIZE: Int = 7

# Velocity (generalized velocities) - 6 floats
comptime QVEL_OFFSET: Int = QPOS_OFFSET + QPOS_SIZE  # 7
comptime QVEL_SIZE: Int = 6

# Acceleration (computed) - 6 floats
comptime QACC_OFFSET: Int = QVEL_OFFSET + QVEL_SIZE  # 13
comptime QACC_SIZE: Int = 6

# Applied forces/torques - 6 floats
comptime QFRC_OFFSET: Int = QACC_OFFSET + QACC_SIZE  # 19
comptime QFRC_SIZE: Int = 6

# World-frame position (computed from qpos) - 3 floats
comptime XPOS_OFFSET: Int = QFRC_OFFSET + QFRC_SIZE  # 25
comptime XPOS_SIZE: Int = 3

# Contact information - 8 floats
comptime CONTACT_OFFSET: Int = XPOS_OFFSET + XPOS_SIZE  # 28
comptime CONTACT_SIZE: Int = 8

# Total state size per environment
comptime STATE_SIZE: Int = CONTACT_OFFSET + CONTACT_SIZE  # 36


# =============================================================================
# Individual Field Indices (absolute offsets within state)
# =============================================================================

# qpos fields [0-6]
comptime IDX_X: Int = 0
comptime IDX_Y: Int = 1
comptime IDX_Z: Int = 2
comptime IDX_QX: Int = 3
comptime IDX_QY: Int = 4
comptime IDX_QZ: Int = 5
comptime IDX_QW: Int = 6

# qvel fields [7-12]
comptime IDX_VX: Int = 7
comptime IDX_VY: Int = 8
comptime IDX_VZ: Int = 9
comptime IDX_WX: Int = 10
comptime IDX_WY: Int = 11
comptime IDX_WZ: Int = 12

# qacc fields [13-18]
comptime IDX_AX: Int = 13
comptime IDX_AY: Int = 14
comptime IDX_AZ: Int = 15
comptime IDX_ALPHA_X: Int = 16
comptime IDX_ALPHA_Y: Int = 17
comptime IDX_ALPHA_Z: Int = 18

# qfrc fields [19-24]
comptime IDX_FX: Int = 19
comptime IDX_FY: Int = 20
comptime IDX_FZ: Int = 21
comptime IDX_TAU_X: Int = 22
comptime IDX_TAU_Y: Int = 23
comptime IDX_TAU_Z: Int = 24

# xpos fields [25-27]
comptime IDX_XPOS_X: Int = 25
comptime IDX_XPOS_Y: Int = 26
comptime IDX_XPOS_Z: Int = 27

# contact fields [28-35]
comptime IDX_CONTACT_ACTIVE: Int = 28  # 0.0 = inactive, 1.0 = active
comptime IDX_CONTACT_DEPTH: Int = 29
comptime IDX_CONTACT_NX: Int = 30
comptime IDX_CONTACT_NY: Int = 31
comptime IDX_CONTACT_NZ: Int = 32
comptime IDX_CONTACT_PX: Int = 33
comptime IDX_CONTACT_PY: Int = 34
comptime IDX_CONTACT_PZ: Int = 35


# =============================================================================
# Geometry Types
# =============================================================================

comptime GEOM_PLANE: Int = 0
comptime GEOM_SPHERE: Int = 1


# =============================================================================
# Physics Defaults
# =============================================================================

comptime DEFAULT_GRAVITY_Z: Float32 = -9.81
comptime DEFAULT_TIMESTEP: Float32 = 0.01
comptime DEFAULT_RESTITUTION: Float32 = 0.0
comptime DEFAULT_BAUMGARTE: Float32 = 0.2
comptime DEFAULT_SLOP: Float32 = 0.001
