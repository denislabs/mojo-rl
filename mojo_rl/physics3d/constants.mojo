"""Physics3D constants.

This module defines compile-time constants for the physics engine.
"""


# GPU kernel configuration
comptime TILE: Int = 16  # Optimal for Apple Silicon
comptime TPB: Int = 256  # Threads per block


struct PhysicsConstants[DTYPE: DType]:
    # Physics defaults
    comptime DEFAULT_GRAVITY_Z: Scalar[Self.DTYPE] = -9.81
    comptime DEFAULT_TIMESTEP: Scalar[Self.DTYPE] = 0.01


# Geometry types
comptime GEOM_PLANE: Int = 0
comptime GEOM_SPHERE: Int = 1
comptime GEOM_CAPSULE: Int = 2
comptime GEOM_BOX: Int = 3
comptime GEOM_CYLINDER: Int = 4
comptime GEOM_MESH: Int = 5
# `ellipsoid` used to fall through to GEOM_SPHERE SILENTLY (no `ellipsoid`
# case in `_geom_type_from_str`, whose default is sphere). Harmless while every
# ellipsoid in the repo carried `mass="0"` with contacts disabled — swimmer's
# head, finger's touch SITES — and load-bearing the moment fish arrived, whose
# tail and fins ARE ellipsoids with density-derived mass: a sphere of radius
# size[0] gave tail1 1/128th of its mass and each fin 26x too much.
#
# INERTIA ONLY. There is no ellipsoid narrow phase; `init_fields` raises if an
# ellipsoid geom can actually collide, rather than silently colliding it as a
# sphere. See `geom_volume` / `geom_inertia`.
comptime GEOM_ELLIPSOID: Int = 6

# `<geom type="hfield">` — a HEIGHTFIELD, and until it existed this fell
# through `_geom_type_from_str`'s `return _GEOM_SPHERE  # default` and collided
# as a BALL of radius `size[0]`. Measured on
# `google_barkour_vb/scene_hfield_mjx`: MuJoCo emitted 8 contacts and we
# emitted 4, on 6 different body pairs, 2.219e-01 apart in depth and 81.1 deg
# apart in normal.
#
# ⚠ THE NUMBER IS OURS, NOT MuJoCo'S. `mjtGeom` puts HFIELD at 1 and PLANE at
# 0; this enum has never matched it (SPHERE is 1 here and 2 there), so every
# comparison against `m.geom_type` goes through the parser's mapping. Appending
# keeps every stored model file readable.
comptime GEOM_HFIELD: Int = 7


@always_inline
def mj_geom_type_rank(t: Int) -> Int:
    """`mjtGeom`'s ordinal for one of OUR `GEOM_*` ids — the pair sort key.

    ⚠⚠ `pushPairArena` SORTS BY MuJoCo's TYPE ID, AND OURS IS NOT MuJoCo's.
    `engine_collision_driver.c:489` canonicalises every candidate pair with

        if (m->geom_type[g1] > m->geom_type[g2]) { swap }

    against `mjtGeom` — PLANE 0, HFIELD 1, SPHERE 2, CAPSULE 3, ELLIPSOID 4,
    CYLINDER 5, BOX 6, MESH 7. The enum above is PLANE 0, SPHERE 1, CAPSULE 2,
    BOX 3, CYLINDER 4, MESH 5, ELLIPSOID 6, HFIELD 7, and the note on
    `GEOM_HFIELD` already says it "has never matched it".

    THE TWO ARE NOT A MONOTONE REMAPPING OF EACH OTHER, so comparing our raw
    ids orders **10 of the 28 unordered type pairs the OPPOSITE way** from the
    reference: box/cylinder, ellipsoid/{box, cylinder, mesh} and hfield/{sphere,
    capsule, box, cylinder, ellipsoid, mesh}. Reachable in up to 28 Menagerie
    scenes (box/cylinder alone).

    ⚠ THAT IS NOT COSMETIC. `mjc_ccd`'s multi-contact is NOT symmetric in its
    two objects: it takes the REFERENCE face from obj1 and clips obj2's face
    against it, so running a pair in the other order returns the same
    PENETRATION DEPTH and different WITNESS POSITIONS.

    ⚠ AN IF-CHAIN, NOT A TABLE. A per-thread `Array` indexed by a RUNTIME
    value reads back the wrong value on Metal, with no crash — four instances
    in this engine already (`87960e10`, `836a65ff`). This is called from inside
    the collision kernels with a runtime type.
    """
    if t == GEOM_PLANE:
        return 0
    if t == GEOM_HFIELD:
        return 1
    if t == GEOM_SPHERE:
        return 2
    if t == GEOM_CAPSULE:
        return 3
    if t == GEOM_ELLIPSOID:
        return 4
    if t == GEOM_CYLINDER:
        return 5
    if t == GEOM_BOX:
        return 6
    return 7  # GEOM_MESH

# `kAngleTol` in `mjCMesh::MakePolygons` (`user_mesh.cc:2905`): the bucket width,
# in radians, used to decide that two hull triangles are coplanar and belong to
# the same polygon. Faces whose normals differ by less than this merge.
comptime MESH_POLY_ANGLE_TOL: Float64 = 0.01


# =============================================================================
# Sensors — `mjtSensor`, `mjtObj`, `mjtDataType`, `mjtStage`
# =============================================================================
#
# ⚠⚠ THESE MATCH MuJoCo'S NUMBERING EXACTLY, and that is a deliberate break
# with `GEOM_*` above — which documents at `ray_geom` that "this tree's enum
# and `mjtGeom` have never agreed", and pays for it with a translation at every
# boundary. There is no such history here and no reason to invent one: the
# sensor table is compared field-for-field against `m.sensor_type` /
# `m.sensor_objtype` / `m.sensor_datatype` in the gate, so agreeing with the
# oracle IS the design. Verified against the 3.12.0 runtime, not transcribed
# from the header: `mjtype.h:328-397`, and `test_sensor_table_vs_mujoco`
# re-reads every one of them off a live `MjModel`.
#
# Only the types this loader MODELS are named. The rest are refused by name in
# `_fill_sensors` rather than given a constant they would never be compared
# against — a named constant for an unimplemented type is exactly the
# accept-and-ignore shape the AUD-23 scan exists to kill.
comptime SENS_TOUCH: Int = 0
comptime SENS_ACCELEROMETER: Int = 1
comptime SENS_VELOCIMETER: Int = 2
comptime SENS_GYRO: Int = 3
comptime SENS_FORCE: Int = 4
comptime SENS_TORQUE: Int = 5
comptime SENS_RANGEFINDER: Int = 7
comptime SENS_JOINTPOS: Int = 9
comptime SENS_JOINTVEL: Int = 10
comptime SENS_TENDONPOS: Int = 11
comptime SENS_ACTUATORPOS: Int = 13
comptime SENS_JOINTACTFRC: Int = 16
comptime SENS_FRAMEPOS: Int = 26
comptime SENS_FRAMEQUAT: Int = 27
comptime SENS_FRAMEXAXIS: Int = 28
comptime SENS_FRAMEYAXIS: Int = 29
comptime SENS_FRAMEZAXIS: Int = 30
comptime SENS_FRAMELINVEL: Int = 31
comptime SENS_FRAMEANGVEL: Int = 32
comptime SENS_SUBTREECOM: Int = 35
comptime SENS_SUBTREELINVEL: Int = 36

# `mjtObj` — the object a sensor is attached to (`mjtype.h:291-298`, re-read
# off `mujoco.mjtObj` in the gate).
#
# ⚠⚠ `BODY` AND `XBODY` ARE DIFFERENT FRAMES OF THE SAME BODY, and mixing
# them is silent. `mjOBJ_XBODY` is the body's own frame (`xpos`/`xquat`);
# `mjOBJ_BODY` is its INERTIAL frame (`xipos`, and `xquat * body_iquat`). They
# coincide only when the body's centre of mass sits at its origin with the
# principal axes aligned — true of a centred sphere, false of most links. The
# `<sensor objtype=>` keyword for the first is the string "xbody" and for the
# second "body", which reads backwards and is MuJoCo's spelling all the same
# (`frameobj_map`, xml/generated/mjcf_map.h:318).
comptime SENSOBJ_UNKNOWN: Int = 0
comptime SENSOBJ_BODY: Int = 1
comptime SENSOBJ_XBODY: Int = 2
comptime SENSOBJ_JOINT: Int = 3
comptime SENSOBJ_GEOM: Int = 5
comptime SENSOBJ_SITE: Int = 6
comptime SENSOBJ_CAMERA: Int = 7
comptime SENSOBJ_TENDON: Int = 18
comptime SENSOBJ_ACTUATOR: Int = 19

# `mjtDataType` — decides how `cutoff` clamps (`engine_sensor.c:198-224`):
# REAL clips to [-cutoff, +cutoff], POSITIVE takes `min(cutoff, x)`.
#
# ⚠ RANGEFINDER IS `REAL`, NOT `POSITIVE`, and the audit's AUD-47 said
# otherwise. Checked in all three trees (`sensorDatatype`, user_objects.cc):
# TOUCH and INSIDESITE are the ONLY positive types in 3.10, 3.11 and 3.12
# alike, so this is not a release change that moved under the audit — it was
# wrong when written. The two rules agree on a hit (both cap at `cutoff`) and
# differ only on the `-1` miss when `cutoff < 1`, which is why it survived.
comptime SENSDATA_REAL: Int = 0
comptime SENSDATA_POSITIVE: Int = 1
comptime SENSDATA_AXIS: Int = 2
comptime SENSDATA_QUATERNION: Int = 3

# `mjtStage` — which of `mj_sensorPos` / `mj_sensorVel` / `mj_sensorAcc`
# evaluates the sensor (`engine_forward.c:1797, 1814, 1832`). The numbering is
# MuJoCo's `mjSTAGE_*`, so NONE is 0 and POS is 1.
comptime SENSSTAGE_NONE: Int = 0
comptime SENSSTAGE_POS: Int = 1
comptime SENSSTAGE_VEL: Int = 2
comptime SENSSTAGE_ACC: Int = 3
