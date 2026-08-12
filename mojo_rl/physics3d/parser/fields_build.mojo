"""Spec-direct Model build: FlatModelDef → packed record tensors (G4).

Replaces the legacy two-hop build (`FlatModelDef.setup_model` → CPU `Model` →
`Model.load_from_model`) with ONE pass that writes the packed record
tensors directly. Every derived quantity the legacy path computed is ported
verbatim:

  * body inv mass / inv inertia (`Model.set_body`: unguarded 1/x),
  * body_rootid / body_weldid (parent-chain walks),
  * joint qpos_adr / dof_adr allocation (running per-type counters),
    hinge/slide axis normalization, per-type defaults (tau_limit 1000 for
    hinge/slide, 0 for free), qpos0 = joint `ref` value,
  * per-joint solref/solimp limit fallbacks (parsed value if >= 0, else the
    MuJoCo model defaults) + the model-level limit sync from joint 0,
  * geom rbound per type (planes 1e10, capsule r+hl, cylinder/box norms),
  * mesh convex-hull loading (STL → dedup → hull) with shared-mesh remap,
  * `<compiler inertiafromgeom>` (modes 1=true / 2=auto) via the ported
    `_inertia_from_geoms_staging` + `<compiler settotalmass>` rescale
    (applied only when inertiafromgeom is active — legacy behavior),
  * equality records with the legacy hardcoded solimp[3]=0.5 / solimp[4]=2.0
    (the parsed values were dropped by `add_connect/weld_constraint` too).

INTENTIONAL FIX vs legacy: `mf.sites` records are now populated (body id +
local pos). `load_from_model` never wrote them — sites were silently
all-zero on the fields path.

invweight0 is NOT computed here — `init_fields` runs the fields-native
`compute_invweight0` (G1) after this build.

Only the BODY mass/inertia block is staged in host Lists (inertiafromgeom +
settotalmass mutate it); all other records are written straight into `mf`.
"""

from std.collections import InlineArray
from std.math import sqrt

from mojo_rl.physics3d.joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_SLIDE,
    JNT_HINGE,
)
from mojo_rl.physics3d.constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_MESH,
    GEOM_ELLIPSOID,
)
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.collision.convex_hull import (
    load_mesh_hull,
    compute_bounding_radius_at,
)
from mojo_rl.physics3d.model.inertia_from_geom import (
    geom_effective_mass,
    geom_inertia,
    globalinertia,
    offcenter,
    eig3_symmetric,
    quat_to_mat,
    MJ_DEFAULT_DENSITY,
    _mulquat as _mulquat_xyzw,
)
from mojo_rl.physics3d.model.mesh_inertia import (
    MeshInertia,
    mesh_inertia_from_file,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_EQ_SIZE,
    MODEL_SITE_SIZE,
    MODEL_MESH_META_SIZE,
    MODEL_MESH_POLY_SIZE,
    MESH_META_IDX_POLYADR,
    MESH_META_IDX_POLYNUM,
    MESH_POLY_IDX_VERTADR,
    MESH_POLY_IDX_VERTNUM,
    MESH_POLY_IDX_NX,
    MESH_POLY_IDX_NY,
    MESH_POLY_IDX_NZ,
    mesh_max_poly,
    mesh_max_polyvert,
    MAX_GPU_MESHES,
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_INV_IXX,
    BODY_IDX_INV_IYY,
    BODY_IDX_INV_IZZ,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_PARENT,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_ROOTID,
    BODY_IDX_WELDID,
    BODY_IDX_MOCAP,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TAU_LIMIT,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_SOLIMP_LIMIT_3,
    JOINT_IDX_SOLIMP_LIMIT_4,
    JOINT_IDX_QPOS0,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    MODEL_META_IDX_IMPRATIO,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
    MODEL_META_IDX_NEXCLUDE,
    MODEL_META_IDX_NPAIR,
    MODEL_PAIR_SIZE,
    PAIR_IDX_GEOM1,
    PAIR_IDX_GEOM2,
    PAIR_IDX_CONDIM,
    PAIR_IDX_FRICTION,
    PAIR_IDX_FRICTION_SPIN,
    PAIR_IDX_FRICTION_ROLL,
    PAIR_IDX_SOLREF_0,
    PAIR_IDX_SOLREF_1,
    PAIR_IDX_SOLIMP_0,
    PAIR_IDX_SOLIMP_1,
    PAIR_IDX_SOLIMP_2,
    PAIR_IDX_SOLIMP_3,
    PAIR_IDX_SOLIMP_4,
    PAIR_IDX_MARGIN,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_RBOUND,
    GEOM_IDX_PRIORITY,
    GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2,
    GEOM_IDX_SOLIMP_3,
    GEOM_IDX_SOLIMP_4,
    GEOM_IDX_MARGIN,
    GEOM_IDX_MESH_ID,
    EQ_IDX_TYPE,
    EQ_IDX_OBJTYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
    EQ_IDX_TORQUESCALE,
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    MODEL_TENDON_SIZE,
    TENDON_MAX_WRAPS,
    TENDON_IDX_KIND,
    TENDON_IDX_IS_EQUALITY,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_COEF_0,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_NUM_SITES,
    TENDON_IDX_SITE_0,
    TENDON_IDX_LIMITED,
    TENDON_IDX_RANGE_MIN,
    TENDON_IDX_RANGE_MAX,
    TENDON_IDX_MARGIN,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    TENDON_IDX_SOLREF_LIM_0,
    TENDON_IDX_SOLREF_LIM_1,
    TENDON_IDX_SOLIMP_LIM_0,
    TENDON_IDX_SOLIMP_LIM_1,
    TENDON_IDX_SOLIMP_LIM_2,
    TENDON_IDX_SOLIMP_LIM_3,
    TENDON_IDX_SOLIMP_LIM_4,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    SITE_IDX_TYPE,
    SITE_IDX_SIZE_0,
    SITE_IDX_SIZE_1,
    SITE_IDX_SIZE_2,
    SITE_IDX_QUAT_W,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
)
from .flat_model import FlatModelDef, _EQ_CONNECT, _EQ_WELD, _EQ_JOINT


def _jnt_qpos_size(jnt_type: Int) -> Int:
    if jnt_type == JNT_FREE:
        return 7
    elif jnt_type == JNT_BALL:
        return 4
    return 1


def _jnt_qvel_size(jnt_type: Int) -> Int:
    if jnt_type == JNT_FREE:
        return 6
    elif jnt_type == JNT_BALL:
        return 3
    return 1


def _geom_mass_and_inertia[
    DTYPE: DType
](
    g: Int,
    geom_type: Int,
    stored_mass: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    geom_mesh_vol: List[Scalar[DTYPE]],
    geom_mesh_eig: List[Scalar[DTYPE]],
) -> InlineArray[Scalar[DTYPE], 4]:
    """(mass, Ixx, Iyy, Izz) for one geom, meshes included.

    ⚠ A MESH's moments are `eigval * (mass / volume)`, and both halves matter:
    the eigenvalues are UNITLESS (length^5, volume-weighted), and the volume is
    the one MuJoCo recomputes in its inertia pass, not the one from its
    centre-of-mass pass. `geom_inertia`'s `else` branch returns (0, 0, 0) for a
    mesh — silently — which is what this exists to stop.
    """
    var out = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    if geom_type == GEOM_MESH:
        var vol = geom_mesh_vol[g]
        if vol <= Scalar[DTYPE](1e-30):
            return out^
        # mass= wins; otherwise density * volume, the same rule
        # `geom_effective_mass` applies to every other type.
        var m = stored_mass
        if m < Scalar[DTYPE](0):
            m = Scalar[DTYPE](MJ_DEFAULT_DENSITY) * vol
        var dens = m / vol
        out[0] = m
        out[1] = geom_mesh_eig[g * 3 + 0] * dens
        out[2] = geom_mesh_eig[g * 3 + 1] * dens
        out[3] = geom_mesh_eig[g * 3 + 2] * dens
        return out^
    var m2 = geom_effective_mass[DTYPE](
        geom_type, stored_mass, radius, half_length, half_x, half_y, half_z
    )
    var d = geom_inertia[DTYPE](
        geom_type, m2, radius, half_length, half_x, half_y, half_z
    )
    out[0] = m2
    out[1] = d[0]
    out[2] = d[1]
    out[3] = d[2]
    return out^


def _inertia_from_geoms_staging[
    DTYPE: DType,
    NBODY: Int,
    NGEOM: Int,
    INERTIA_GROUP_MIN: Int,
    INERTIA_GROUP_MAX: Int,
    AUTO_MODE: Bool,
](
    geoms: List[Scalar[DTYPE]],  # packed [NGEOM * MODEL_GEOM_SIZE] records
    bodies: List[Scalar[DTYPE]],  # packed [NBODY * MODEL_BODY_SIZE] records
    geom_mass: List[Scalar[DTYPE]],  # build-only (-1 = use density*volume)
    geom_group: List[Int],  # build-only (inertiagrouprange filter)
    # Mesh geoms only. `geom_inertia` takes scalar dims and cannot reach a
    # vertex list, so a mesh's volume and UNITLESS principal moments are
    # resolved during the geom pass and read back here. Zero for every
    # non-mesh geom, and for a mesh whose frame was never computed (see the
    # cache filter in `build_model_fields_from_flat`).
    geom_mesh_vol: List[Scalar[DTYPE]],
    geom_mesh_eig: List[Scalar[DTYPE]],  # [NGEOM * 3]
    body_has_explicit_inertia: List[Bool],
    mut body_mass: List[Scalar[DTYPE]],
    mut body_inv_mass: List[Scalar[DTYPE]],
    mut body_inertia: List[Scalar[DTYPE]],
    mut body_inv_inertia: List[Scalar[DTYPE]],
    mut body_ipos: List[Scalar[DTYPE]],
    mut body_iquat: List[Scalar[DTYPE]],
):
    """`compute_inertia_from_geoms` (MuJoCo inertiafromgeom) ported onto the
    build staging arrays + packed geom records — arithmetic verbatim from the
    legacy Model-typed routine (deleted at G4)."""
    for body_id in range(1, NBODY):
        comptime if AUTO_MODE:
            if body_has_explicit_inertia[body_id]:
                continue

        var bo = body_id * MODEL_BODY_SIZE
        var total_mass = Scalar[DTYPE](0)
        var num_contributing = 0

        for g in range(NGEOM):
            var go = g * MODEL_GEOM_SIZE
            var ggrp = geom_group[g]
            if ggrp < INERTIA_GROUP_MIN or ggrp > INERTIA_GROUP_MAX:
                continue
            if Int(geoms[go + GEOM_IDX_BODY]) == body_id:
                var gmi0 = _geom_mass_and_inertia[DTYPE](
                    g,
                    Int(geoms[go + GEOM_IDX_TYPE]),
                    geom_mass[g],
                    geoms[go + GEOM_IDX_RADIUS],
                    geoms[go + GEOM_IDX_HALF_LENGTH],
                    geoms[go + GEOM_IDX_HALF_X],
                    geoms[go + GEOM_IDX_HALF_Y],
                    geoms[go + GEOM_IDX_HALF_Z],
                    geom_mesh_vol,
                    geom_mesh_eig,
                )
                var gm = gmi0[0]
                if gm > Scalar[DTYPE](1e-10):
                    num_contributing += 1
                    total_mass += gm

        if num_contributing == 0:
            # ⚠ A BODY WITH NOTHING TO WEIGH HAS ZERO MASS, NOT THE DEFAULT ONE.
            #
            # This used to `continue`, leaving `BodyData`'s constructor defaults
            # (mass = 1.0, diaginertia = 0.01) standing. MuJoCo assigns 0/0:
            # under `inertiafromgeom="auto"` a body with no explicit `<inertial>`
            # takes its inertia from geoms, and if NONE qualify there is nothing
            # to take, so it stays massless.
            #
            # "None qualify" is not the same as "no geoms". `inertiagrouprange`
            # (default "0 5") filters by geom GROUP, so a body can be covered in
            # geometry and still weigh nothing. MetaWorld's sawyer sets
            # `inertiagrouprange="4 5"`, which excludes the group-1 visual shells:
            # `hand` (the mocap weld's target), `rightpad` and `leftpad` each own
            # a group-1 box and so each carried a PHANTOM KILOGRAM.
            #
            # That is defect 28. Three phantom kg at the wrist, ~1.15 m from the
            # j0 axis, added 3 x 1.15^2 = 3.97 kg m^2 to the base joint —
            # `dof_invweight0[j0]` was 0.4664x MuJoCo's, and the error compounded
            # down the chain to 0.0249x at the hand. body_invweight0 feeds
            # diagApprox, so the weld's R was 4.4x too large and the constraint
            # 4.4x too soft: the commanded hand pose sagged 77.6 mm where MuJoCo
            # tracks to 3.8 mm. It reached the weld through the constraint
            # regularizer, but the phantom mass was in the MASS MATRIX — every
            # force, acceleration and contact on those bodies was wrong too.
            body_mass[body_id] = Scalar[DTYPE](0)
            body_inv_mass[body_id] = Scalar[DTYPE](0)
            for c in range(3):
                body_inertia[body_id * 3 + c] = Scalar[DTYPE](0)
                body_inv_inertia[body_id * 3 + c] = Scalar[DTYPE](0)

            # ...and its INERTIAL FRAME is the body's own frame, not the origin.
            #
            # `mjCBody::Compile` (user_objects.cc:2738): once `InertiaFromGeom`
            # has run and left `ipos` undefined, MuJoCo copies `pos`/`quat` into
            # `ipos`/`iquat`. Note it copies the PARENT-frame pos into a
            # BODY-frame field — quirk faithfully reproduced, because the value
            # is observable.
            #
            # It is observable through `xipos`, which `mj_jacBodyCom` uses as
            # the Jacobian reference point for `body_invweight0`. Zeroing the
            # mass without this left the hand's translational weight at 2.168
            # against MuJoCo's 6.106 — still 2.8x off, in the opposite
            # direction, i.e. a weld 2.8x too STIFF where it had been 4.4x too
            # soft. Verified across every zero-mass body in sawyer: ipos == pos
            # and iquat == quat for all twelve of them.
            body_ipos[body_id * 3 + 0] = bodies[bo + BODY_IDX_POS_X]
            body_ipos[body_id * 3 + 1] = bodies[bo + BODY_IDX_POS_Y]
            body_ipos[body_id * 3 + 2] = bodies[bo + BODY_IDX_POS_Z]
            body_iquat[body_id * 4 + 0] = bodies[bo + BODY_IDX_QUAT_X]
            body_iquat[body_id * 4 + 1] = bodies[bo + BODY_IDX_QUAT_Y]
            body_iquat[body_id * 4 + 2] = bodies[bo + BODY_IDX_QUAT_Z]
            body_iquat[body_id * 4 + 3] = bodies[bo + BODY_IDX_QUAT_W]
            continue

        if num_contributing == 1:
            for g in range(NGEOM):
                var go = g * MODEL_GEOM_SIZE
                var ggrp1 = geom_group[g]
                if ggrp1 < INERTIA_GROUP_MIN or ggrp1 > INERTIA_GROUP_MAX:
                    continue
                if Int(geoms[go + GEOM_IDX_BODY]) == body_id:
                    var gmi = _geom_mass_and_inertia[DTYPE](
                        g,
                        Int(geoms[go + GEOM_IDX_TYPE]),
                        geom_mass[g],
                        geoms[go + GEOM_IDX_RADIUS],
                        geoms[go + GEOM_IDX_HALF_LENGTH],
                        geoms[go + GEOM_IDX_HALF_X],
                        geoms[go + GEOM_IDX_HALF_Y],
                        geoms[go + GEOM_IDX_HALF_Z],
                        geom_mesh_vol,
                        geom_mesh_eig,
                    )
                    var gm = gmi[0]
                    if gm > Scalar[DTYPE](1e-10):
                        body_ipos[body_id * 3 + 0] = geoms[go + GEOM_IDX_POS_X]
                        body_ipos[body_id * 3 + 1] = geoms[go + GEOM_IDX_POS_Y]
                        body_ipos[body_id * 3 + 2] = geoms[go + GEOM_IDX_POS_Z]
                        body_iquat[body_id * 4 + 0] = geoms[
                            go + GEOM_IDX_QUAT_X
                        ]
                        body_iquat[body_id * 4 + 1] = geoms[
                            go + GEOM_IDX_QUAT_Y
                        ]
                        body_iquat[body_id * 4 + 2] = geoms[
                            go + GEOM_IDX_QUAT_Z
                        ]
                        body_iquat[body_id * 4 + 3] = geoms[
                            go + GEOM_IDX_QUAT_W
                        ]
                        body_mass[body_id] = gm
                        body_inv_mass[body_id] = Scalar[DTYPE](1.0) / gm
                        var inertia = InlineArray[Scalar[DTYPE], 3](
                            fill=Scalar[DTYPE](0)
                        )
                        inertia[0] = gmi[1]
                        inertia[1] = gmi[2]
                        inertia[2] = gmi[3]
                        body_inertia[body_id * 3 + 0] = inertia[0]
                        body_inertia[body_id * 3 + 1] = inertia[1]
                        body_inertia[body_id * 3 + 2] = inertia[2]
                        body_inv_inertia[body_id * 3 + 0] = (
                            Scalar[DTYPE](1.0) / inertia[0]
                        )
                        body_inv_inertia[body_id * 3 + 1] = (
                            Scalar[DTYPE](1.0) / inertia[1]
                        )
                        body_inv_inertia[body_id * 3 + 2] = (
                            Scalar[DTYPE](1.0) / inertia[2]
                        )
                        break
        else:
            var com_x = Scalar[DTYPE](0)
            var com_y = Scalar[DTYPE](0)
            var com_z = Scalar[DTYPE](0)
            for g in range(NGEOM):
                var go = g * MODEL_GEOM_SIZE
                var ggrp2 = geom_group[g]
                if ggrp2 < INERTIA_GROUP_MIN or ggrp2 > INERTIA_GROUP_MAX:
                    continue
                if Int(geoms[go + GEOM_IDX_BODY]) == body_id:
                    var gmi = _geom_mass_and_inertia[DTYPE](
                        g,
                        Int(geoms[go + GEOM_IDX_TYPE]),
                        geom_mass[g],
                        geoms[go + GEOM_IDX_RADIUS],
                        geoms[go + GEOM_IDX_HALF_LENGTH],
                        geoms[go + GEOM_IDX_HALF_X],
                        geoms[go + GEOM_IDX_HALF_Y],
                        geoms[go + GEOM_IDX_HALF_Z],
                        geom_mesh_vol,
                        geom_mesh_eig,
                    )
                    var gm = gmi[0]
                    if gm > Scalar[DTYPE](1e-10):
                        com_x += gm * geoms[go + GEOM_IDX_POS_X]
                        com_y += gm * geoms[go + GEOM_IDX_POS_Y]
                        com_z += gm * geoms[go + GEOM_IDX_POS_Z]
            com_x /= total_mass
            com_y /= total_mass
            com_z /= total_mass

            body_ipos[body_id * 3 + 0] = com_x
            body_ipos[body_id * 3 + 1] = com_y
            body_ipos[body_id * 3 + 2] = com_z

            body_mass[body_id] = total_mass
            body_inv_mass[body_id] = Scalar[DTYPE](1.0) / total_mass

            var toti = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))

            for g in range(NGEOM):
                var go = g * MODEL_GEOM_SIZE
                var ggrp3 = geom_group[g]
                if ggrp3 < INERTIA_GROUP_MIN or ggrp3 > INERTIA_GROUP_MAX:
                    continue
                if Int(geoms[go + GEOM_IDX_BODY]) == body_id:
                    var gmi = _geom_mass_and_inertia[DTYPE](
                        g,
                        Int(geoms[go + GEOM_IDX_TYPE]),
                        geom_mass[g],
                        geoms[go + GEOM_IDX_RADIUS],
                        geoms[go + GEOM_IDX_HALF_LENGTH],
                        geoms[go + GEOM_IDX_HALF_X],
                        geoms[go + GEOM_IDX_HALF_Y],
                        geoms[go + GEOM_IDX_HALF_Z],
                        geom_mesh_vol,
                        geom_mesh_eig,
                    )
                    var gm = gmi[0]
                    if gm > Scalar[DTYPE](1e-10):
                        var diag = InlineArray[Scalar[DTYPE], 3](
                            fill=Scalar[DTYPE](0)
                        )
                        diag[0] = gmi[1]
                        diag[1] = gmi[2]
                        diag[2] = gmi[3]

                        var inert_global = InlineArray[Scalar[DTYPE], 6](
                            fill=Scalar[DTYPE](0)
                        )
                        globalinertia(
                            diag[0],
                            diag[1],
                            diag[2],
                            geoms[go + GEOM_IDX_QUAT_X],
                            geoms[go + GEOM_IDX_QUAT_Y],
                            geoms[go + GEOM_IDX_QUAT_Z],
                            geoms[go + GEOM_IDX_QUAT_W],
                            inert_global,
                        )

                        var dx = geoms[go + GEOM_IDX_POS_X] - com_x
                        var dy = geoms[go + GEOM_IDX_POS_Y] - com_y
                        var dz = geoms[go + GEOM_IDX_POS_Z] - com_z
                        var inert_offset = InlineArray[Scalar[DTYPE], 6](
                            fill=Scalar[DTYPE](0)
                        )
                        offcenter(gm, dx, dy, dz, inert_offset)

                        for j in range(6):
                            toti[j] += inert_global[j] + inert_offset[j]

            var eig = eig3_symmetric(toti)
            body_inertia[body_id * 3 + 0] = eig[0]
            body_inertia[body_id * 3 + 1] = eig[1]
            body_inertia[body_id * 3 + 2] = eig[2]
            body_inv_inertia[body_id * 3 + 0] = Scalar[DTYPE](1.0) / eig[0]
            body_inv_inertia[body_id * 3 + 1] = Scalar[DTYPE](1.0) / eig[1]
            body_inv_inertia[body_id * 3 + 2] = Scalar[DTYPE](1.0) / eig[2]
            body_iquat[body_id * 4 + 0] = eig[3]
            body_iquat[body_id * 4 + 1] = eig[4]
            body_iquat[body_id * 4 + 2] = eig[5]
            body_iquat[body_id * 4 + 3] = eig[6]


def build_model_fields_from_flat[
    DTYPE: DType,
    # ⚠ THE FlatModelDef DIMS USED TO BE PARAMETERS HERE — all fourteen of
    # them. They are gone: `FlatModelDef` is non-generic (List-backed) since
    # 2026-08-05, so its counts come from the Lists themselves.
    #
    # What remains below is the MODEL side, and it is deliberate: these size
    # `fields.Model`'s tensors, which are the hot path. Do not confuse the two
    # — de-genericizing the PARSER did not make the ENGINE runtime-dimensioned.
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    # Model dims (record capacities)
    MAX_EQUALITY: Int,
    MAX_TENDON: Int,
    NSITE: Int,
    NEXCLUDE: Int,
    NMESH_VERTS: Int,
    # <compiler> build modes
    IFG_MODE: Int,  # 0=off, 1=true, 2=auto
    IGR_MIN: Int,
    IGR_MAX: Int,
    SETTOTALMASS: Float64,
    # Appended rather than grouped with NEXCLUDE — see `fields.Model`.
    NPAIR: Int = 0,
](
    fmd: FlatModelDef,
    mut mf: Model[
        DTYPE,
        NV,
        NBODY,
        NJOINT,
        NGEOM,
        MAX_EQUALITY,
        MAX_TENDON,
        NSITE,
        NEXCLUDE,
        NMESH_VERTS,
        NPAIR,
    ],
) raises:
    """Fill every `mf` record tensor from the parsed `fmd` — see module
    docstring. Does NOT compute invweight0 and does NOT upload."""

    # ── meta ─────────────────────────────────────────────────────────────
    mf.meta.data[MODEL_META_IDX_NBODY] = Scalar[DTYPE](NBODY)
    mf.meta.data[MODEL_META_IDX_NJOINT] = Scalar[DTYPE](NJOINT)
    mf.meta.data[MODEL_META_IDX_GRAVITY_X] = Scalar[DTYPE](fmd.gravity_x)
    mf.meta.data[MODEL_META_IDX_GRAVITY_Y] = Scalar[DTYPE](fmd.gravity_y)
    mf.meta.data[MODEL_META_IDX_GRAVITY_Z] = Scalar[DTYPE](fmd.gravity_z)
    mf.meta.data[MODEL_META_IDX_TIMESTEP] = Scalar[DTYPE](fmd.timestep)
    mf.meta.data[MODEL_META_IDX_DENSITY] = Scalar[DTYPE](fmd.opt_density)
    mf.meta.data[MODEL_META_IDX_VISCOSITY] = Scalar[DTYPE](fmd.opt_viscosity)
    mf.meta.data[MODEL_META_IDX_IMPRATIO] = Scalar[DTYPE](1.0)
    mf.meta.data[MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](
        len(fmd.equalities) if len(fmd.equalities) < MAX_EQUALITY
        else MAX_EQUALITY
    )
    # Honest tendon count. This used to be hardcoded 0, which made every
    # tendon record dead. `_tendon_env` treats a record as a BILATERAL
    # EQUALITY, so waking it up is safe only because that pass now also
    # requires TENDON_IDX_IS_EQUALITY — humanoid declares two <fixed> tendons
    # that MuJoCo constrains in no way.
    mf.meta.data[MODEL_META_IDX_NTENDON] = Scalar[DTYPE](
        len(fmd.tendons) if len(fmd.tendons) < MAX_TENDON
        else MAX_TENDON
    )
    mf.meta.data[MODEL_META_IDX_NEXCLUDE] = Scalar[DTYPE](len(fmd.excludes))
    mf.meta.data[MODEL_META_IDX_NPAIR] = Scalar[DTYPE](len(fmd.pairs))

    # Contact solref/solimp: MuJoCo model defaults, then geom[0]'s parsed
    # values (floor / first worldbody geom inherits <default><geom>).
    mf.meta.data[MODEL_META_IDX_SOLREF_CONTACT_0] = Scalar[DTYPE](0.02)
    mf.meta.data[MODEL_META_IDX_SOLREF_CONTACT_1] = Scalar[DTYPE](1.0)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_0] = Scalar[DTYPE](0.9)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_1] = Scalar[DTYPE](0.95)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_2] = Scalar[DTYPE](0.001)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_3] = Scalar[DTYPE](0.5)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_4] = Scalar[DTYPE](2.0)
    comptime if NGEOM > 0:
        var g0 = fmd.geoms[0]
        mf.meta.data[MODEL_META_IDX_SOLREF_CONTACT_0] = Scalar[DTYPE](
            g0.solref_0
        )
        mf.meta.data[MODEL_META_IDX_SOLREF_CONTACT_1] = Scalar[DTYPE](
            g0.solref_1
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_0] = Scalar[DTYPE](
            g0.solimp_0
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_1] = Scalar[DTYPE](
            g0.solimp_1
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_2] = Scalar[DTYPE](
            g0.solimp_2
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_3] = Scalar[DTYPE](
            g0.solimp_3
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_4] = Scalar[DTYPE](
            g0.solimp_4
        )

    # Limit solref/solimp defaults (overridden from joint 0 after the joint
    # fill, matching the legacy model-level sync).
    var def_solref_limit_0 = Scalar[DTYPE](0.02)
    var def_solref_limit_1 = Scalar[DTYPE](1.0)
    var def_solimp_limit_0 = Scalar[DTYPE](0.9)
    var def_solimp_limit_1 = Scalar[DTYPE](0.95)
    var def_solimp_limit_2 = Scalar[DTYPE](0.001)
    var def_solimp_limit_3 = Scalar[DTYPE](0.5)
    var def_solimp_limit_4 = Scalar[DTYPE](2.0)
    mf.meta.data[MODEL_META_IDX_SOLREF_LIMIT_0] = def_solref_limit_0
    mf.meta.data[MODEL_META_IDX_SOLREF_LIMIT_1] = def_solref_limit_1
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_0] = def_solimp_limit_0
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_1] = def_solimp_limit_1
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_2] = def_solimp_limit_2
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_3] = def_solimp_limit_3
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_4] = def_solimp_limit_4

    # ── body staging (mass/inertia block; inertiafromgeom + settotalmass
    #    mutate it before the record write) ─────────────────────────────────
    var body_mass = List[Scalar[DTYPE]](length=NBODY, fill=Scalar[DTYPE](0))
    var body_inv_mass = List[Scalar[DTYPE]](
        length=NBODY, fill=Scalar[DTYPE](0)
    )
    var body_inertia = List[Scalar[DTYPE]](
        length=NBODY * 3, fill=Scalar[DTYPE](0)
    )
    var body_inv_inertia = List[Scalar[DTYPE]](
        length=NBODY * 3, fill=Scalar[DTYPE](0)
    )
    var body_ipos = List[Scalar[DTYPE]](
        length=NBODY * 3, fill=Scalar[DTYPE](0)
    )
    var body_iquat = List[Scalar[DTYPE]](
        length=NBODY * 4, fill=Scalar[DTYPE](0)
    )
    var body_has_explicit_inertia = List[Bool](length=NBODY, fill=False)
    var body_parent = List[Int](length=NBODY, fill=0)

    # Worldbody (index 0): mass/inertia zero, identity iquat.
    body_iquat[3] = Scalar[DTYPE](1)

    # Bodies 1..NBODY-1 from fmd (legacy `set_body` semantics: unguarded 1/x).
    for i in range(len(fmd.bodies)):
        var b = fmd.bodies[i]
        var bi = i + 1
        body_mass[bi] = Scalar[DTYPE](b.mass)
        body_inv_mass[bi] = Scalar[DTYPE](1.0) / Scalar[DTYPE](b.mass)
        body_inertia[bi * 3 + 0] = Scalar[DTYPE](b.ixx)
        body_inertia[bi * 3 + 1] = Scalar[DTYPE](b.iyy)
        body_inertia[bi * 3 + 2] = Scalar[DTYPE](b.izz)
        body_inv_inertia[bi * 3 + 0] = Scalar[DTYPE](1.0) / Scalar[DTYPE](
            b.ixx
        )
        body_inv_inertia[bi * 3 + 1] = Scalar[DTYPE](1.0) / Scalar[DTYPE](
            b.iyy
        )
        body_inv_inertia[bi * 3 + 2] = Scalar[DTYPE](1.0) / Scalar[DTYPE](
            b.izz
        )
        body_ipos[bi * 3 + 0] = Scalar[DTYPE](b.ipos_x)
        body_ipos[bi * 3 + 1] = Scalar[DTYPE](b.ipos_y)
        body_ipos[bi * 3 + 2] = Scalar[DTYPE](b.ipos_z)
        body_iquat[bi * 4 + 0] = Scalar[DTYPE](b.iquat_x)
        body_iquat[bi * 4 + 1] = Scalar[DTYPE](b.iquat_y)
        body_iquat[bi * 4 + 2] = Scalar[DTYPE](b.iquat_z)
        body_iquat[bi * 4 + 3] = Scalar[DTYPE](b.iquat_w)
        body_has_explicit_inertia[bi] = b.has_explicit_inertia
        body_parent[bi] = b.parent

        # Non-staged body columns straight into the record.
        var o = bi * MODEL_BODY_SIZE
        mf.bodies.data[o + BODY_IDX_POS_X] = Scalar[DTYPE](b.pos_x)
        mf.bodies.data[o + BODY_IDX_POS_Y] = Scalar[DTYPE](b.pos_y)
        mf.bodies.data[o + BODY_IDX_POS_Z] = Scalar[DTYPE](b.pos_z)
        mf.bodies.data[o + BODY_IDX_QUAT_X] = Scalar[DTYPE](b.quat_x)
        mf.bodies.data[o + BODY_IDX_QUAT_Y] = Scalar[DTYPE](b.quat_y)
        mf.bodies.data[o + BODY_IDX_QUAT_Z] = Scalar[DTYPE](b.quat_z)
        mf.bodies.data[o + BODY_IDX_QUAT_W] = Scalar[DTYPE](b.quat_w)
        mf.bodies.data[o + BODY_IDX_PARENT] = Scalar[DTYPE](b.parent)
        mf.bodies.data[o + BODY_IDX_MOCAP] = Scalar[DTYPE](
            1.0 if b.is_mocap else 0.0
        )

    # Worldbody record: pos 0, quat identity, parent 0, mocap 0.
    mf.bodies.data[BODY_IDX_QUAT_W] = Scalar[DTYPE](1)

    # body_rootid (root = child of worldbody).
    var body_rootid = List[Int](length=NBODY, fill=0)
    for bi in range(1, NBODY):
        var p = body_parent[bi]
        if p == 0:
            body_rootid[bi] = bi
        else:
            body_rootid[bi] = body_rootid[p]
        mf.bodies.data[bi * MODEL_BODY_SIZE + BODY_IDX_ROOTID] = Scalar[DTYPE](
            body_rootid[bi]
        )

    # ── joints ───────────────────────────────────────────────────────────
    var qpos_adr = 0
    var dof_adr = 0
    for j in range(len(fmd.joints)):
        var jd = fmd.joints[j]
        var o = j * MODEL_JOINT_SIZE

        mf.joints.data[o + JOINT_IDX_TYPE] = Scalar[DTYPE](jd.jnt_type)
        mf.joints.data[o + JOINT_IDX_BODY_ID] = Scalar[DTYPE](jd.body_id)
        mf.joints.data[o + JOINT_IDX_QPOS_ADR] = Scalar[DTYPE](qpos_adr)
        mf.joints.data[o + JOINT_IDX_DOF_ADR] = Scalar[DTYPE](dof_adr)

        if jd.jnt_type == JNT_HINGE or jd.jnt_type == JNT_SLIDE:
            # Legacy create_hinge/create_slide: normalized axis, parsed
            # pos/range/dynamics, tau_limit default 1000.
            var ax = Scalar[DTYPE](jd.axis_x)
            var ay = Scalar[DTYPE](jd.axis_y)
            var az = Scalar[DTYPE](jd.axis_z)
            var length = sqrt(ax * ax + ay * ay + az * az)
            if length > Scalar[DTYPE](1e-10):
                ax = ax / length
                ay = ay / length
                az = az / length
            mf.joints.data[o + JOINT_IDX_POS_X] = Scalar[DTYPE](jd.pos_x)
            mf.joints.data[o + JOINT_IDX_POS_Y] = Scalar[DTYPE](jd.pos_y)
            mf.joints.data[o + JOINT_IDX_POS_Z] = Scalar[DTYPE](jd.pos_z)
            mf.joints.data[o + JOINT_IDX_AXIS_X] = ax
            mf.joints.data[o + JOINT_IDX_AXIS_Y] = ay
            mf.joints.data[o + JOINT_IDX_AXIS_Z] = az
            mf.joints.data[o + JOINT_IDX_TAU_LIMIT] = Scalar[DTYPE](1000.0)
            mf.joints.data[o + JOINT_IDX_RANGE_MIN] = Scalar[DTYPE](
                jd.range_min
            )
            mf.joints.data[o + JOINT_IDX_RANGE_MAX] = Scalar[DTYPE](
                jd.range_max
            )
            mf.joints.data[o + JOINT_IDX_ARMATURE] = Scalar[DTYPE](jd.armature)
            mf.joints.data[o + JOINT_IDX_DAMPING] = Scalar[DTYPE](jd.damping)
            mf.joints.data[o + JOINT_IDX_STIFFNESS] = Scalar[DTYPE](
                jd.stiffness
            )
            mf.joints.data[o + JOINT_IDX_SPRINGREF] = Scalar[DTYPE](
                jd.springref
            )
            mf.joints.data[o + JOINT_IDX_FRICTIONLOSS] = Scalar[DTYPE](
                jd.frictionloss
            )
        else:
            # JNT_FREE (legacy create_free + armature/damping overrides);
            # JNT_BALL was never wired on the XML path.
            mf.joints.data[o + JOINT_IDX_POS_X] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_POS_Y] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_POS_Z] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_AXIS_X] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_AXIS_Y] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_AXIS_Z] = Scalar[DTYPE](1)
            mf.joints.data[o + JOINT_IDX_TAU_LIMIT] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_RANGE_MIN] = Scalar[DTYPE](-1e10)
            mf.joints.data[o + JOINT_IDX_RANGE_MAX] = Scalar[DTYPE](1e10)
            mf.joints.data[o + JOINT_IDX_ARMATURE] = Scalar[DTYPE](jd.armature)
            mf.joints.data[o + JOINT_IDX_DAMPING] = Scalar[DTYPE](jd.damping)
            mf.joints.data[o + JOINT_IDX_STIFFNESS] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_SPRINGREF] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_FRICTIONLOSS] = Scalar[DTYPE](0)

        # qpos0 = joint ref value (MuJoCo: displacement = qpos - qpos0).
        mf.joints.data[o + JOINT_IDX_QPOS0] = Scalar[DTYPE](jd.ref_val)

        # Per-joint limit solref/solimp: parsed value if >= 0, else the model
        # defaults (which at legacy fill time were the MuJoCo defaults).
        mf.joints.data[o + JOINT_IDX_SOLREF_LIMIT_0] = (
            Scalar[DTYPE](jd.solref_limit_0)
            if jd.solref_limit_0 >= 0.0
            else def_solref_limit_0
        )
        mf.joints.data[o + JOINT_IDX_SOLREF_LIMIT_1] = (
            Scalar[DTYPE](jd.solref_limit_1)
            if jd.solref_limit_1 >= 0.0
            else def_solref_limit_1
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_0] = (
            Scalar[DTYPE](jd.solimp_limit_0)
            if jd.solimp_limit_0 >= 0.0
            else def_solimp_limit_0
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_1] = (
            Scalar[DTYPE](jd.solimp_limit_1)
            if jd.solimp_limit_1 >= 0.0
            else def_solimp_limit_1
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_2] = (
            Scalar[DTYPE](jd.solimp_limit_2)
            if jd.solimp_limit_2 >= 0.0
            else def_solimp_limit_2
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_3] = (
            Scalar[DTYPE](jd.solimp_limit_3)
            if jd.solimp_limit_3 >= 0.0
            else def_solimp_limit_3
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_4] = (
            Scalar[DTYPE](jd.solimp_limit_4)
            if jd.solimp_limit_4 >= 0.0
            else def_solimp_limit_4
        )

        qpos_adr += _jnt_qpos_size(jd.jnt_type)
        dof_adr += _jnt_qvel_size(jd.jnt_type)

    # Model-level limit meta sync from joint 0 (legacy CPU/GPU consistency
    # sync — uniform joint solimp across all current models).
    comptime if NJOINT > 0:
        mf.meta.data[MODEL_META_IDX_SOLREF_LIMIT_0] = mf.joints.data[
            JOINT_IDX_SOLREF_LIMIT_0
        ]
        mf.meta.data[MODEL_META_IDX_SOLREF_LIMIT_1] = mf.joints.data[
            JOINT_IDX_SOLREF_LIMIT_1
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_0] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_0
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_1] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_1
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_2] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_2
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_3] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_3
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_4] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_4
        ]

    # body_weldid: bodies with joints weld to themselves, jointless bodies
    # inherit the parent's weldid (MuJoCo convention).
    var body_has_joint = List[Bool](length=NBODY, fill=False)
    for j in range(len(fmd.joints)):
        body_has_joint[fmd.joints[j].body_id] = True
    var body_weldid = List[Int](length=NBODY, fill=0)
    for bi in range(1, NBODY):
        if body_has_joint[bi]:
            body_weldid[bi] = bi
        else:
            body_weldid[bi] = body_weldid[body_parent[bi]]
        mf.bodies.data[bi * MODEL_BODY_SIZE + BODY_IDX_WELDID] = Scalar[DTYPE](
            body_weldid[bi]
        )

    # ── mesh frames + unitless principal moments (MuJoCo `mjCMesh::Compute`)
    #
    # Computed BEFORE the geom loop because both consumers need it: the geom's
    # pos/quat are composed with the mesh frame right below, and
    # `load_mesh_hull` bakes the same frame into the vertices further down. The
    # two are only equivalent TOGETHER — moving the vertices without composing
    # the frame collides the mesh in the wrong place, and vice versa.
    #
    # ⚠ COMPUTED ONLY WHERE IT IS USED: a mesh geom that is neither collidable
    # nor a source of body inertia (its body carries an explicit `<inertial>`)
    # gets nothing, and keeps its declared frame. That is dog — 162 visual-only
    # meshes whose frames reach nothing but the renderer — and it is what keeps
    # this off dog's build path entirely. The same filter §22 applied to hull
    # construction, for the same reason.
    var mesh_inertia_cache = List[MeshInertia[DTYPE]]()
    var mesh_inertia_valid = List[Bool](
        length=fmd.num_mesh_assets, fill=False
    )
    for _ in range(fmd.num_mesh_assets):
        mesh_inertia_cache.append(MeshInertia[DTYPE]())
    for i in range(len(fmd.geoms)):
        var gd_m = fmd.geoms[i]
        if gd_m.geom_type != GEOM_MESH:
            continue
        if gd_m.mesh_id < 0 or gd_m.mesh_filename.byte_length() == 0:
            continue
        if mesh_inertia_valid[gd_m.mesh_id]:
            continue
        var collidable = gd_m.contype != 0 or gd_m.conaffinity != 0
        var needs_inertia = not body_has_explicit_inertia[gd_m.body_id]
        if not (collidable or needs_inertia):
            continue
        try:
            mesh_inertia_cache[gd_m.mesh_id] = mesh_inertia_from_file[DTYPE](
                gd_m.mesh_filename
            )
            mesh_inertia_valid[gd_m.mesh_id] = True
        except:
            print(
                "Warning: failed to compute mesh inertia:",
                gd_m.mesh_filename,
            )

    # ── geoms (+ build-only mass/group staging for inertiafromgeom) ───────
    var geom_mass = List[Scalar[DTYPE]](length=NGEOM, fill=Scalar[DTYPE](0))
    var geom_group = List[Int](length=NGEOM, fill=0)
    # Per-geom mesh volume + UNITLESS principal moments, staged for the inertia
    # assembly. `geom_inertia` cannot compute these — it takes scalar dims and
    # has no way to reach a vertex list — so the mesh case is resolved here and
    # read back as a table.
    var geom_mesh_vol = List[Scalar[DTYPE]](
        length=NGEOM, fill=Scalar[DTYPE](0)
    )
    var geom_mesh_eig = List[Scalar[DTYPE]](
        length=NGEOM * 3, fill=Scalar[DTYPE](0)
    )
    for i in range(len(fmd.geoms)):
        var gd = fmd.geoms[i]
        var o = i * MODEL_GEOM_SIZE
        mf.geoms.data[o + GEOM_IDX_TYPE] = Scalar[DTYPE](gd.geom_type)
        mf.geoms.data[o + GEOM_IDX_BODY] = Scalar[DTYPE](gd.body_id)
        mf.geoms.data[o + GEOM_IDX_POS_X] = Scalar[DTYPE](gd.pos_x)
        mf.geoms.data[o + GEOM_IDX_POS_Y] = Scalar[DTYPE](gd.pos_y)
        mf.geoms.data[o + GEOM_IDX_POS_Z] = Scalar[DTYPE](gd.pos_z)
        mf.geoms.data[o + GEOM_IDX_QUAT_X] = Scalar[DTYPE](gd.quat_x)
        mf.geoms.data[o + GEOM_IDX_QUAT_Y] = Scalar[DTYPE](gd.quat_y)
        mf.geoms.data[o + GEOM_IDX_QUAT_Z] = Scalar[DTYPE](gd.quat_z)
        mf.geoms.data[o + GEOM_IDX_QUAT_W] = Scalar[DTYPE](gd.quat_w)
        mf.geoms.data[o + GEOM_IDX_RADIUS] = Scalar[DTYPE](gd.radius)
        mf.geoms.data[o + GEOM_IDX_HALF_LENGTH] = Scalar[DTYPE](gd.half_length)
        mf.geoms.data[o + GEOM_IDX_HALF_X] = Scalar[DTYPE](gd.half_x)
        mf.geoms.data[o + GEOM_IDX_HALF_Y] = Scalar[DTYPE](gd.half_y)
        mf.geoms.data[o + GEOM_IDX_HALF_Z] = Scalar[DTYPE](gd.half_z)
        mf.geoms.data[o + GEOM_IDX_FRICTION] = Scalar[DTYPE](gd.friction)
        mf.geoms.data[o + GEOM_IDX_CONTYPE] = Scalar[DTYPE](gd.contype)
        mf.geoms.data[o + GEOM_IDX_CONAFFINITY] = Scalar[DTYPE](
            gd.conaffinity
        )
        mf.geoms.data[o + GEOM_IDX_CONDIM] = Scalar[DTYPE](gd.condim)
        mf.geoms.data[o + GEOM_IDX_FRICTION_SPIN] = Scalar[DTYPE](
            gd.friction_spin
        )
        mf.geoms.data[o + GEOM_IDX_FRICTION_ROLL] = Scalar[DTYPE](
            gd.friction_roll
        )
        mf.geoms.data[o + GEOM_IDX_PRIORITY] = Scalar[DTYPE](gd.priority)
        mf.geoms.data[o + GEOM_IDX_SOLREF_0] = Scalar[DTYPE](gd.solref_0)
        mf.geoms.data[o + GEOM_IDX_SOLREF_1] = Scalar[DTYPE](gd.solref_1)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_0] = Scalar[DTYPE](gd.solimp_0)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_1] = Scalar[DTYPE](gd.solimp_1)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_2] = Scalar[DTYPE](gd.solimp_2)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_3] = Scalar[DTYPE](gd.solimp_3)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_4] = Scalar[DTYPE](gd.solimp_4)
        mf.geoms.data[o + GEOM_IDX_MARGIN] = Scalar[DTYPE](gd.margin)
        mf.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[DTYPE](gd.mesh_id)
        geom_mass[i] = Scalar[DTYPE](gd.mass)
        geom_group[i] = gd.group

        # Compose the mesh frame into the geom frame, as MuJoCo's compiler
        # does: `geom_pos = declared_pos + R(declared_quat) * mesh_pos` and
        # `geom_quat = declared_quat (x) mesh_quat`. Measured on Jaco, whose
        # mesh geoms declare no pos/quat: `geom_pos == mesh_pos` and
        # `geom_quat == mesh_quat` for all 13 of them.
        if (
            gd.geom_type == GEOM_MESH
            and gd.mesh_id >= 0
            and mesh_inertia_valid[gd.mesh_id]
        ):
            var mi_g = mesh_inertia_cache[gd.mesh_id]
            var rm = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
            quat_to_mat[DTYPE](
                Scalar[DTYPE](gd.quat_x),
                Scalar[DTYPE](gd.quat_y),
                Scalar[DTYPE](gd.quat_z),
                Scalar[DTYPE](gd.quat_w),
                rm,
            )
            mf.geoms.data[o + GEOM_IDX_POS_X] = Scalar[DTYPE](gd.pos_x) + (
                rm[0] * mi_g.com_x + rm[1] * mi_g.com_y + rm[2] * mi_g.com_z
            )
            mf.geoms.data[o + GEOM_IDX_POS_Y] = Scalar[DTYPE](gd.pos_y) + (
                rm[3] * mi_g.com_x + rm[4] * mi_g.com_y + rm[5] * mi_g.com_z
            )
            mf.geoms.data[o + GEOM_IDX_POS_Z] = Scalar[DTYPE](gd.pos_z) + (
                rm[6] * mi_g.com_x + rm[7] * mi_g.com_y + rm[8] * mi_g.com_z
            )
            var cq = _mulquat_xyzw[DTYPE](
                Scalar[DTYPE](gd.quat_x),
                Scalar[DTYPE](gd.quat_y),
                Scalar[DTYPE](gd.quat_z),
                Scalar[DTYPE](gd.quat_w),
                mi_g.qx,
                mi_g.qy,
                mi_g.qz,
                mi_g.qw,
            )
            mf.geoms.data[o + GEOM_IDX_QUAT_X] = cq[0]
            mf.geoms.data[o + GEOM_IDX_QUAT_Y] = cq[1]
            mf.geoms.data[o + GEOM_IDX_QUAT_Z] = cq[2]
            mf.geoms.data[o + GEOM_IDX_QUAT_W] = cq[3]
            geom_mesh_vol[i] = mi_g.volume
            geom_mesh_eig[i * 3 + 0] = mi_g.eig0
            geom_mesh_eig[i * 3 + 1] = mi_g.eig1
            geom_mesh_eig[i * 3 + 2] = mi_g.eig2

        # Bounding sphere radius for broad-phase (legacy per-type formulas).
        var rbound = Scalar[DTYPE](gd.radius)
        if gd.geom_type == GEOM_PLANE:
            rbound = Scalar[DTYPE](1e10)  # planes are infinite
        elif gd.geom_type == GEOM_SPHERE:
            rbound = Scalar[DTYPE](gd.radius)
        elif gd.geom_type == GEOM_CAPSULE:
            rbound = Scalar[DTYPE](gd.radius + gd.half_length)
        elif gd.geom_type == GEOM_CYLINDER:
            rbound = Scalar[DTYPE](
                sqrt(gd.half_length * gd.half_length + gd.radius * gd.radius)
            )
        elif gd.geom_type == GEOM_BOX:
            rbound = Scalar[DTYPE](
                sqrt(
                    gd.half_x * gd.half_x
                    + gd.half_y * gd.half_y
                    + gd.half_z * gd.half_z
                )
            )
        elif gd.geom_type == GEOM_ELLIPSOID:
            # `max(size)` — mjCGeom::GetRBound (user_objects.cc:3345). Without
            # this case `rbound` fell through to `gd.radius`, which the parser
            # sets to size[0] for an ellipsoid; any ellipsoid whose LARGEST
            # semi-axis is not the first would get a bounding sphere smaller
            # than itself and the broad phase would silently drop its
            # contacts. Harmless until the ellipsoid narrow phase landed
            # (2026-07-31), because a colliding ellipsoid was rejected at
            # build time; a real bug from that point on. quadruped's torso
            # (.3 .27 .2) happens to be ordered largest-first and so would
            # NOT have exposed it.
            var mx = gd.half_x
            if gd.half_y > mx:
                mx = gd.half_y
            if gd.half_z > mx:
                mx = gd.half_z
            rbound = Scalar[DTYPE](mx)
        # GEOM_MESH: refined from hull vertices below.
        mf.geoms.data[o + GEOM_IDX_RBOUND] = rbound

    # ── mesh convex hulls (STL → dedup → hull; shared meshes remapped) ────
    var mesh_vert = List[Scalar[DTYPE]]()
    var mesh_vertadr = List[Int]()
    var mesh_vertnum = List[Int]()
    var num_meshes = 0
    # Polygon topology, accumulated across meshes exactly as `mesh_polyvert`
    # and `mesh_polymap` are in `mjModel`. See `collision/mesh_polygons.mojo`.
    var mesh_polyadr = List[Int]()
    var mesh_polynum = List[Int]()
    var poly_vert = List[Int]()
    var poly_vertadr = List[Int]()
    var poly_vertnum = List[Int]()
    var poly_normal = List[Scalar[DTYPE]]()
    var polymap = List[Int]()
    var polymap_adr = List[Int]()
    var polymap_num = List[Int]()
    var loaded_mesh_ids = List[Int](length=fmd.num_mesh_assets, fill=-1)
    for i in range(len(fmd.geoms)):
        var gd = fmd.geoms[i]
        var o = i * MODEL_GEOM_SIZE
        # ⚠ COLLIDABLE MESHES ONLY. This used to build a hull for EVERY mesh
        # geom carrying a filename, ignoring `contype`/`conaffinity`. Dog has
        # 162 mesh geoms and ALL of them are visual (`contype=0
        # conaffinity=0`), so every dog model build ran the exact incremental
        # hull — O(n*h) — over 162 meshes and threw the whole result away, then
        # tripped the capacity message on vertices nothing could ever collide.
        #
        # A geom with neither bit set cannot pair with anything (MuJoCo's
        # filter is `contype1 & conaffinity2 || contype2 & conaffinity1`), so
        # its hull is dead weight in both time and capacity. Skipping it is
        # also what lets the capacity check below be a hard error: otherwise
        # dog would fail a check for a feature it does not use.
        var mesh_collidable = gd.contype != 0 or gd.conaffinity != 0
        # ⚠ A SKIPPED MESH GEOM MUST NOT KEEP ITS ASSET INDEX. The main geom
        # loop writes `GEOM_IDX_MESH_ID = gd.mesh_id`, which is the index into
        # the XML's mesh ASSETS; the loop below overwrites it with the
        # COMPACTED runtime id for meshes it actually loads. Skipping the
        # visual ones leaves that raw asset index in place, pointing past the
        # end of `mesh_meta` — sawyer keeps 10 visual mesh geoms carrying ids
        # up to 13 while only 2 meshes are loaded. Nothing dereferences it
        # today (the collision filter rejects those geoms first), but a stale
        # index that is only safe because of a check somewhere else is the
        # shape of the `mesh_vertadr` unit bug. Make it unusable instead.
        if gd.geom_type == GEOM_MESH and not mesh_collidable:
            mf.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[DTYPE](-1)
        if (
            gd.geom_type == GEOM_MESH
            and gd.mesh_id >= 0
            and gd.mesh_filename.byte_length() > 0
            and mesh_collidable
        ):
            if loaded_mesh_ids[gd.mesh_id] >= 0:
                var mid = loaded_mesh_ids[gd.mesh_id]
                mf.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[DTYPE](mid)
                # `mesh_vertadr` is a VERTEX index (MuJoCo's convention, and
                # what every collision consumer expects); this helper walks the
                # FLAT scalar list, so convert. See `load_mesh_hull`.
                mf.geoms.data[o + GEOM_IDX_RBOUND] = (
                    compute_bounding_radius_at[DTYPE](
                        mesh_vert, mesh_vertadr[mid] * 3, mesh_vertnum[mid]
                    )
                )
            else:
                try:
                    var result = load_mesh_hull[DTYPE](
                        gd.mesh_filename,
                        mesh_vert,
                        mesh_vertadr,
                        mesh_vertnum,
                        num_meshes,
                        mesh_polyadr,
                        mesh_polynum,
                        poly_vert,
                        poly_vertadr,
                        poly_vertnum,
                        poly_normal,
                        polymap,
                        polymap_adr,
                        polymap_num,
                        mesh_inertia_cache[gd.mesh_id],
                    )
                    var mesh_id = result[0]
                    mf.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[DTYPE](
                        mesh_id
                    )
                    mf.geoms.data[o + GEOM_IDX_RBOUND] = result[1]
                    loaded_mesh_ids[gd.mesh_id] = mesh_id
                except:
                    print("Warning: failed to load mesh:", gd.mesh_filename)

    for m in range(num_meshes):
        if m >= MAX_GPU_MESHES:
            break
        mf.mesh_meta.data[m * MODEL_MESH_META_SIZE + 0] = Scalar[DTYPE](
            mesh_vertadr[m]
        )
        mf.mesh_meta.data[m * MODEL_MESH_META_SIZE + 1] = Scalar[DTYPE](
            mesh_vertnum[m]
        )
        mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_POLYADR
        ] = Scalar[DTYPE](mesh_polyadr[m])
        mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_POLYNUM
        ] = Scalar[DTYPE](mesh_polynum[m])
    # ⚠ CAPACITY IS ANNOUNCED, NOT SILENTLY TRUNCATED. This loop used to just
    # `break` at the cap, which drops hull vertices from the LAST meshes and
    # shrinks their collision shape — an error with one sign, invisible to
    # every gate, and exactly the failure mode that let `mesh_vertadr`'s unit
    # bug hide. Exact hulls need roughly 10x what support sampling did (sawyer:
    # ~5.6k vertices against the 648 the sampler kept), so an under-sized
    # NMESHV is now easy to hit.
    # ⚠⚠ THIS RAISES NOW, AND THE PRINT IT REPLACED IS WHY. Every environment
    # ran with `NMESH_VERTS = 0` — `Phyics3dEnv` hardcoded it — so every mesh
    # geom in every env emitted NO CONTACT (both narrow phases guard the mesh
    # branch with `comptime if NMESH_VERTS > 0`). This message printed on every
    # sawyer construction the whole time and nothing read it. A printed "ERROR"
    # that no assert consumes is a comment, and it hid the entire mesh collider
    # from the env layer while every gate stayed green.
    #
    # Only safe because the loop above now skips non-collidable meshes: dog
    # carries 162 VISUAL meshes and would otherwise fail here for a feature it
    # does not use.
    if len(mesh_vert) > NMESH_VERTS * 3:
        raise Error(
            String("mesh vertex capacity exceeded — NMESH_VERTS = ")
            + String(NMESH_VERTS) + " holds " + String(NMESH_VERTS * 3)
            + " scalars but the COLLIDABLE hulls need " + String(len(mesh_vert))
            + ". Truncating would shrink those collision shapes silently, so"
            " this is fatal. Raise the config's NMESH_VERTS to at least "
            + String((len(mesh_vert) + 2) // 3) + " vertices."
        )
    for i in range(len(mesh_vert)):
        if i >= NMESH_VERTS * 3:
            break
        mf.mesh_verts.data[i] = mesh_vert[i]

    # ── mesh polygon topology ────────────────────────────────────────────
    #
    # The capacities here are Euler's formula on NMESH_VERTS (see
    # `mesh_max_poly` / `mesh_max_polyvert`), so a convex hull that fits in the
    # vertex budget cannot overflow them. They are still checked, because the
    # bound holds for a CONVEX hull and a builder bug is exactly what would
    # break that assumption — and a silent overrun here would corrupt the
    # tensor allocated next to it.
    comptime NMESH_POLY = mesh_max_poly(NMESH_VERTS)
    comptime NMESH_POLYVERT = mesh_max_polyvert(NMESH_VERTS)
    if len(poly_vertadr) > NMESH_POLY or len(poly_vert) > NMESH_POLYVERT:
        raise Error(
            String("mesh POLYGON capacity exceeded: ")
            + String(len(poly_vertadr)) + " polygons (cap " + String(NMESH_POLY)
            + ") and " + String(len(poly_vert)) + " polygon-vertices (cap "
            + String(NMESH_POLYVERT) + ") for NMESH_VERTS = "
            + String(NMESH_VERTS) + ". These caps are Euler's formula for a"
            " CONVEX hull, so exceeding them means the hull or the polygon"
            " merge is wrong, not that the budget is too small."
        )
    for p in range(len(poly_vertadr)):
        var o = p * MODEL_MESH_POLY_SIZE
        mf.mesh_polys.data[o + MESH_POLY_IDX_VERTADR] = Scalar[DTYPE](
            poly_vertadr[p]
        )
        mf.mesh_polys.data[o + MESH_POLY_IDX_VERTNUM] = Scalar[DTYPE](
            poly_vertnum[p]
        )
        mf.mesh_polys.data[o + MESH_POLY_IDX_NX] = poly_normal[p * 3 + 0]
        mf.mesh_polys.data[o + MESH_POLY_IDX_NY] = poly_normal[p * 3 + 1]
        mf.mesh_polys.data[o + MESH_POLY_IDX_NZ] = poly_normal[p * 3 + 2]
    for k in range(len(poly_vert)):
        mf.mesh_polyvert.data[k] = Scalar[DTYPE](poly_vert[k])
    for k in range(len(polymap)):
        mf.mesh_polymap.data[k] = Scalar[DTYPE](polymap[k])
    for v in range(len(polymap_adr)):
        if v >= NMESH_VERTS:
            break
        mf.mesh_vert_polymap.data[v * 2 + 0] = Scalar[DTYPE](polymap_adr[v])
        mf.mesh_vert_polymap.data[v * 2 + 1] = Scalar[DTYPE](polymap_num[v])

    # ── sites (INTENTIONAL FIX: legacy load_from_model left these zero) ───
    for i in range(NSITE):
        var sd = fmd.sites[i]
        var o = i * MODEL_SITE_SIZE
        mf.sites.data[o + SITE_IDX_BODY] = Scalar[DTYPE](sd.body_id)
        mf.sites.data[o + SITE_IDX_POS_X] = Scalar[DTYPE](sd.pos_x)
        mf.sites.data[o + SITE_IDX_POS_Y] = Scalar[DTYPE](sd.pos_y)
        mf.sites.data[o + SITE_IDX_POS_Z] = Scalar[DTYPE](sd.pos_z)
        mf.sites.data[o + SITE_IDX_TYPE] = Scalar[DTYPE](sd.site_type)
        mf.sites.data[o + SITE_IDX_SIZE_0] = Scalar[DTYPE](sd.size_0)
        mf.sites.data[o + SITE_IDX_SIZE_1] = Scalar[DTYPE](sd.size_1)
        mf.sites.data[o + SITE_IDX_SIZE_2] = Scalar[DTYPE](sd.size_2)
        mf.sites.data[o + SITE_IDX_QUAT_X] = Scalar[DTYPE](sd.quat_x)
        mf.sites.data[o + SITE_IDX_QUAT_Y] = Scalar[DTYPE](sd.quat_y)
        mf.sites.data[o + SITE_IDX_QUAT_Z] = Scalar[DTYPE](sd.quat_z)
        mf.sites.data[o + SITE_IDX_QUAT_W] = Scalar[DTYPE](sd.quat_w)

    # ── tendons ──────────────────────────────────────────────────────────
    #
    # NTENDON_P == MAX_TENDON by construction (ModelDefFromXML passes
    # `max_tendon` for both), so this is a straight copy. INVWEIGHT0 is left
    # zero here and filled by the invweight pass, which needs FK at qpos0.
    for i in range(len(fmd.tendons)):
        if i >= MAX_TENDON:
            break
        var td = fmd.tendons[i]
        var o = i * MODEL_TENDON_SIZE
        mf.tendons.data[o + TENDON_IDX_KIND] = Scalar[DTYPE](td.kind)
        mf.tendons.data[o + TENDON_IDX_IS_EQUALITY] = Scalar[DTYPE](
            td.is_equality
        )
        mf.tendons.data[o + TENDON_IDX_NUM_JOINTS] = Scalar[DTYPE](
            td.num_joints
        )
        # ⚠ LOOPED, NOT UNROLLED. These were four explicit writes each, which
        # is how the 4-wrap cap got baked into a fourth place: widening the
        # record without touching this would have left wraps 4..15 as
        # uninitialised garbage. Driving both from `TENDON_MAX_WRAPS` means the
        # cap moves in one edit.
        for k in range(TENDON_MAX_WRAPS):
            mf.tendons.data[o + TENDON_IDX_JOINT_0 + k] = Scalar[DTYPE](
                td.joint_ids[k]
            )
            mf.tendons.data[o + TENDON_IDX_COEF_0 + k] = Scalar[DTYPE](
                td.coefs[k]
            )
        mf.tendons.data[o + TENDON_IDX_LENGTH_REF] = Scalar[DTYPE](
            td.length_ref
        )
        mf.tendons.data[o + TENDON_IDX_NUM_SITES] = Scalar[DTYPE](td.num_sites)
        for k in range(TENDON_MAX_WRAPS):
            mf.tendons.data[o + TENDON_IDX_SITE_0 + k] = Scalar[DTYPE](
                td.site_ids[k]
            )
        mf.tendons.data[o + TENDON_IDX_LIMITED] = Scalar[DTYPE](td.limited)
        mf.tendons.data[o + TENDON_IDX_RANGE_MIN] = Scalar[DTYPE](td.range_min)
        mf.tendons.data[o + TENDON_IDX_RANGE_MAX] = Scalar[DTYPE](td.range_max)
        mf.tendons.data[o + TENDON_IDX_MARGIN] = Scalar[DTYPE](td.margin)
        mf.tendons.data[o + TENDON_IDX_SOLREF_LIM_0] = Scalar[DTYPE](
            td.solref_lim_0
        )
        mf.tendons.data[o + TENDON_IDX_SOLREF_LIM_1] = Scalar[DTYPE](
            td.solref_lim_1
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_0] = Scalar[DTYPE](
            td.solimp_lim_0
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_1] = Scalar[DTYPE](
            td.solimp_lim_1
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_2] = Scalar[DTYPE](
            td.solimp_lim_2
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_3] = Scalar[DTYPE](
            td.solimp_lim_3
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_4] = Scalar[DTYPE](
            td.solimp_lim_4
        )
        # The EQUALITY pair, slots 10..16. `constraints/equality_tendon.mojo`
        # has always read these; until 2026-07-31 nothing wrote them, so every
        # tendon-equality row would have been built on a zero solref timeconst
        # (had any row ever been built — `is_equality` was never set either).
        mf.tendons.data[o + TENDON_IDX_SOLREF_0] = Scalar[DTYPE](td.solref_eq_0)
        mf.tendons.data[o + TENDON_IDX_SOLREF_1] = Scalar[DTYPE](td.solref_eq_1)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_0] = Scalar[DTYPE](td.solimp_eq_0)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_1] = Scalar[DTYPE](td.solimp_eq_1)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_2] = Scalar[DTYPE](td.solimp_eq_2)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_3] = Scalar[DTYPE](td.solimp_eq_3)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_4] = Scalar[DTYPE](td.solimp_eq_4)

    # ── equality constraints (legacy add_connect/add_weld semantics:
    #    solimp[3]=0.5 / solimp[4]=2.0 hardcoded, parsed values dropped) ────
    var num_eq = 0
    for i in range(len(fmd.equalities)):
        if num_eq >= MAX_EQUALITY:
            break
        var ed = fmd.equalities[i]
        if (
            ed.eq_type != _EQ_CONNECT
            and ed.eq_type != _EQ_WELD
            and ed.eq_type != _EQ_JOINT
        ):
            continue
        var o = num_eq * MODEL_EQ_SIZE
        mf.equality.data[o + EQ_IDX_TYPE] = Scalar[DTYPE](ed.eq_type)
        mf.equality.data[o + EQ_IDX_OBJTYPE] = Scalar[DTYPE](ed.objtype)
        mf.equality.data[o + EQ_IDX_BODY_A] = Scalar[DTYPE](ed.body_a)
        mf.equality.data[o + EQ_IDX_BODY_B] = Scalar[DTYPE](ed.body_b)
        mf.equality.data[o + EQ_IDX_ANCHOR_AX] = Scalar[DTYPE](ed.anchor_a_x)
        mf.equality.data[o + EQ_IDX_ANCHOR_AY] = Scalar[DTYPE](ed.anchor_a_y)
        mf.equality.data[o + EQ_IDX_ANCHOR_AZ] = Scalar[DTYPE](ed.anchor_a_z)
        mf.equality.data[o + EQ_IDX_ANCHOR_BX] = Scalar[DTYPE](ed.anchor_b_x)
        mf.equality.data[o + EQ_IDX_ANCHOR_BY] = Scalar[DTYPE](ed.anchor_b_y)
        mf.equality.data[o + EQ_IDX_ANCHOR_BZ] = Scalar[DTYPE](ed.anchor_b_z)
        if ed.eq_type == _EQ_WELD:
            mf.equality.data[o + EQ_IDX_RELPOSE_X] = Scalar[DTYPE](
                ed.relpose_x
            )
            mf.equality.data[o + EQ_IDX_RELPOSE_Y] = Scalar[DTYPE](
                ed.relpose_y
            )
            mf.equality.data[o + EQ_IDX_RELPOSE_Z] = Scalar[DTYPE](
                ed.relpose_z
            )
            mf.equality.data[o + EQ_IDX_RELPOSE_W] = Scalar[DTYPE](
                ed.relpose_w
            )
        else:
            mf.equality.data[o + EQ_IDX_RELPOSE_X] = Scalar[DTYPE](0)
            mf.equality.data[o + EQ_IDX_RELPOSE_Y] = Scalar[DTYPE](0)
            mf.equality.data[o + EQ_IDX_RELPOSE_Z] = Scalar[DTYPE](0)
            mf.equality.data[o + EQ_IDX_RELPOSE_W] = Scalar[DTYPE](1)
        mf.equality.data[o + EQ_IDX_SOLREF_0] = Scalar[DTYPE](ed.solref_0)
        mf.equality.data[o + EQ_IDX_SOLREF_1] = Scalar[DTYPE](ed.solref_1)
        mf.equality.data[o + EQ_IDX_SOLIMP_0] = Scalar[DTYPE](ed.solimp_0)
        mf.equality.data[o + EQ_IDX_SOLIMP_1] = Scalar[DTYPE](ed.solimp_1)
        mf.equality.data[o + EQ_IDX_SOLIMP_2] = Scalar[DTYPE](ed.solimp_2)
        mf.equality.data[o + EQ_IDX_SOLIMP_3] = Scalar[DTYPE](0.5)
        mf.equality.data[o + EQ_IDX_SOLIMP_4] = Scalar[DTYPE](2.0)
        # Weld only (MuJoCo's eq_data[10]); connect has no orientation rows,
        # so 1 keeps the multiply a no-op rather than zeroing anything.
        mf.equality.data[o + EQ_IDX_TORQUESCALE] = Scalar[DTYPE](
            ed.torquescale if ed.eq_type == _EQ_WELD else 1.0
        )
        num_eq += 1
    mf.meta.data[MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](num_eq)

    # ── contact exclusion pairs ────────────────────────────────────────────
    for i in range(len(fmd.excludes)):
        var ex = fmd.excludes[i]
        mf.excludes.data[i * 2 + 0] = Scalar[DTYPE](ex.body1)
        mf.excludes.data[i * 2 + 1] = Scalar[DTYPE](ex.body2)

    # ── predefined contact pairs ───────────────────────────────────────────
    # ⚠ RAISES rather than clamping. `NPAIR` is the sole knob and is derived
    # from the same `<contact>` text `_fill_pairs` walks, so a mismatch means
    # the two disagree about the model — and a clamp would drop the tail
    # pairs, which is invisible: the model still simulates, just without some
    # of the collisions it declared. Compare the tendon count above, which
    # clamps to `MAX_TENDON`.
    if len(fmd.pairs) > NPAIR:
        raise Error(
            "physics3d: parsed "
            + String(len(fmd.pairs))
            + " <contact><pair> records but NPAIR is "
            + String(NPAIR)
            + ". Pass npair=<count> to ModelDefFromXML."
        )
    for i in range(len(fmd.pairs)):
        var pr = fmd.pairs[i]
        var o = i * MODEL_PAIR_SIZE
        mf.pairs.data[o + PAIR_IDX_GEOM1] = Scalar[DTYPE](pr.geom1)
        mf.pairs.data[o + PAIR_IDX_GEOM2] = Scalar[DTYPE](pr.geom2)
        mf.pairs.data[o + PAIR_IDX_CONDIM] = Scalar[DTYPE](pr.condim)
        mf.pairs.data[o + PAIR_IDX_FRICTION] = Scalar[DTYPE](pr.friction)
        mf.pairs.data[o + PAIR_IDX_FRICTION_SPIN] = Scalar[DTYPE](
            pr.friction_spin
        )
        mf.pairs.data[o + PAIR_IDX_FRICTION_ROLL] = Scalar[DTYPE](
            pr.friction_roll
        )
        mf.pairs.data[o + PAIR_IDX_SOLREF_0] = Scalar[DTYPE](pr.solref_0)
        mf.pairs.data[o + PAIR_IDX_SOLREF_1] = Scalar[DTYPE](pr.solref_1)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_0] = Scalar[DTYPE](pr.solimp_0)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_1] = Scalar[DTYPE](pr.solimp_1)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_2] = Scalar[DTYPE](pr.solimp_2)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_3] = Scalar[DTYPE](pr.solimp_3)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_4] = Scalar[DTYPE](pr.solimp_4)
        mf.pairs.data[o + PAIR_IDX_MARGIN] = Scalar[DTYPE](pr.margin)

    # ── <compiler inertiafromgeom> + settotalmass (staging mutations) ─────
    comptime if IFG_MODE == 1:
        _inertia_from_geoms_staging[
            DTYPE, NBODY, NGEOM, IGR_MIN, IGR_MAX, False
        ](
            mf.geoms.data,
            mf.bodies.data,
            geom_mass,
            geom_group,
            geom_mesh_vol,
            geom_mesh_eig,
            body_has_explicit_inertia,
            body_mass,
            body_inv_mass,
            body_inertia,
            body_inv_inertia,
            body_ipos,
            body_iquat,
        )
    comptime if IFG_MODE == 2:
        _inertia_from_geoms_staging[
            DTYPE, NBODY, NGEOM, IGR_MIN, IGR_MAX, True
        ](
            mf.geoms.data,
            mf.bodies.data,
            geom_mass,
            geom_group,
            geom_mesh_vol,
            geom_mesh_eig,
            body_has_explicit_inertia,
            body_mass,
            body_inv_mass,
            body_inertia,
            body_inv_inertia,
            body_ipos,
            body_iquat,
        )
    # ── <compiler boundmass / boundinertia> ──────────────────────────────
    #
    # `mjCBody::Compile`: for EVERY body with id > 0, after the inertial frame
    # is set, `mass = max(mass, boundmass)` and the same per principal moment.
    # ⚠ Applied to every body, not only the ones `inertiafromgeom` touched —
    # a body with an explicit `<inertial>` is bounded too, and so is one with
    # no geoms at all. That last case is the whole point on composer models:
    # `jaco_arm/` and `jaco_arm/jaco_hand/` are pure attachment frames carrying
    # NO geoms, and `b_6`'s only geom has mass 1e-9. All three take their
    # entire mass (1e-05) and inertia (1e-11) from these two numbers.
    #
    # ⚠ Runs AFTER the inertia staging and BEFORE settotalmass, matching
    # MuJoCo's order — settotalmass rescales whatever the bound produced.
    if fmd.boundmass > 0.0 or fmd.boundinertia > 0.0:
        var bm = Scalar[DTYPE](fmd.boundmass)
        var bi = Scalar[DTYPE](fmd.boundinertia)
        for i in range(1, NBODY):
            if body_mass[i] < bm:
                body_mass[i] = bm
                body_inv_mass[i] = Scalar[DTYPE](1.0) / bm
            for k in range(3):
                if body_inertia[i * 3 + k] < bi:
                    body_inertia[i * 3 + k] = bi
                    body_inv_inertia[i * 3 + k] = Scalar[DTYPE](1.0) / bi

    comptime if IFG_MODE > 0:
        comptime if SETTOTALMASS > 0.0:
            var total_mass = Scalar[DTYPE](0)
            for i in range(1, NBODY):
                total_mass += body_mass[i]
            if total_mass > Scalar[DTYPE](0):
                var scale = Scalar[DTYPE](SETTOTALMASS) / total_mass
                for i in range(1, NBODY):
                    body_mass[i] *= scale
                    body_inv_mass[i] = Scalar[DTYPE](1.0) / body_mass[i]
                    for k in range(3):
                        body_inertia[i * 3 + k] *= scale
                        body_inv_inertia[i * 3 + k] = (
                            Scalar[DTYPE](1.0) / body_inertia[i * 3 + k]
                        )

    # ── body mass/inertia record write (post ifg/settotalmass) ────────────
    for b in range(len(fmd.bodies) + 1):
        var o = b * MODEL_BODY_SIZE
        mf.bodies.data[o + BODY_IDX_MASS] = body_mass[b]
        mf.bodies.data[o + BODY_IDX_INV_MASS] = body_inv_mass[b]
        mf.bodies.data[o + BODY_IDX_IXX] = body_inertia[b * 3 + 0]
        mf.bodies.data[o + BODY_IDX_IYY] = body_inertia[b * 3 + 1]
        mf.bodies.data[o + BODY_IDX_IZZ] = body_inertia[b * 3 + 2]
        mf.bodies.data[o + BODY_IDX_INV_IXX] = body_inv_inertia[b * 3 + 0]
        mf.bodies.data[o + BODY_IDX_INV_IYY] = body_inv_inertia[b * 3 + 1]
        mf.bodies.data[o + BODY_IDX_INV_IZZ] = body_inv_inertia[b * 3 + 2]
        mf.bodies.data[o + BODY_IDX_IPOS_X] = body_ipos[b * 3 + 0]
        mf.bodies.data[o + BODY_IDX_IPOS_Y] = body_ipos[b * 3 + 1]
        mf.bodies.data[o + BODY_IDX_IPOS_Z] = body_ipos[b * 3 + 2]
        mf.bodies.data[o + BODY_IDX_IQUAT_X] = body_iquat[b * 4 + 0]
        mf.bodies.data[o + BODY_IDX_IQUAT_Y] = body_iquat[b * 4 + 1]
        mf.bodies.data[o + BODY_IDX_IQUAT_Z] = body_iquat[b * 4 + 2]
        mf.bodies.data[o + BODY_IDX_IQUAT_W] = body_iquat[b * 4 + 3]
