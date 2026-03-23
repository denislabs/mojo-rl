"""Compare HalfCheetah XML vs Def model GPU buffers to find discrepancies.

Initialises both model GPU buffers, copies them back to host, and prints a
side-by-side diff for every body, joint, geom, and metadata field.
Fields with |xml - def| > 1e-4 are marked  *** DIFF ***

Run with:
    pixi run -e apple  mojo run tests/test_compare_halfcheetah_models.mojo
    pixi run -e nvidia mojo run -I . tests/test_compare_halfcheetah_models.mojo
"""

from std.collections import InlineArray
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.gpu.constants import (
    model_size_with_invweight,
    model_body_offset,
    model_joint_offset,
    model_geom_offset,
    model_metadata_offset,
    model_body_invweight0_offset,
    model_dof_invweight0_offset,
    # Body indices
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
    # Joint indices
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
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_TAU_LIMIT,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_QPOS0,
    # Geom indices
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RBOUND,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONDIM,
    GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2,
    # Metadata indices
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
)

# Import the two model defs directly
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from mojo_rl.envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel as HalfCheetahModelDef,
)

comptime DTYPE = DType.float32  # same as GPU training

comptime NBODY = HalfCheetahModel.NBODY  # 8
comptime NJOINT = HalfCheetahModel.NJOINT  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NGEOM = HalfCheetahModel.NGEOM  # 9
comptime MAX_CONTACTS = HalfCheetahModel.MAX_CONTACTS  # 20

comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()

comptime BODY_NAMES = [
    "world",
    "torso",
    "bthigh",
    "bshin",
    "bfoot",
    "fthigh",
    "fshin",
    "ffoot",
]
comptime JOINT_NAMES = [
    "rootx",
    "rootz",
    "rooty",
    "bthigh",
    "bshin",
    "bfoot",
    "fthigh",
    "fshin",
    "ffoot",
]
comptime GEOM_NAMES = [
    "floor",
    "torso",
    "head",
    "bthigh",
    "bshin",
    "bfoot",
    "fthigh",
    "fshin",
    "ffoot",
]


def marker(a: Float32, b: Float32, tol: Float32 = 1e-4) -> String:
    if abs(a - b) > tol:
        return "  *** DIFF ***"
    return ""


def compare_models(xml: List[Float32], def_: List[Float32]):
    var joint_names = materialize[JOINT_NAMES]()
    var geom_names = materialize[GEOM_NAMES]()
    var body_names = materialize[BODY_NAMES]()
    # ── Bodies ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(
        "BODIES  (mass, inv_mass, ixx/iyy/izz, inv_ixx/iyy/izz, local_pos,"
        " local_quat, parent, ipos, iquat)"
    )
    print("=" * 70)
    for b in range(NBODY):
        var off = model_body_offset(b)
        var m_x = xml[off + BODY_IDX_MASS]
        var m_d = def_[off + BODY_IDX_MASS]
        var im_x = xml[off + BODY_IDX_INV_MASS]
        var im_d = def_[off + BODY_IDX_INV_MASS]
        var ix_x = xml[off + BODY_IDX_IXX]
        var ix_d = def_[off + BODY_IDX_IXX]
        var iy_x = xml[off + BODY_IDX_IYY]
        var iy_d = def_[off + BODY_IDX_IYY]
        var iz_x = xml[off + BODY_IDX_IZZ]
        var iz_d = def_[off + BODY_IDX_IZZ]
        var iix_x = xml[off + BODY_IDX_INV_IXX]
        var iix_d = def_[off + BODY_IDX_INV_IXX]
        var iiy_x = xml[off + BODY_IDX_INV_IYY]
        var iiy_d = def_[off + BODY_IDX_INV_IYY]
        var iiz_x = xml[off + BODY_IDX_INV_IZZ]
        var iiz_d = def_[off + BODY_IDX_INV_IZZ]
        var px_x = xml[off + BODY_IDX_POS_X]
        var px_d = def_[off + BODY_IDX_POS_X]
        var py_x = xml[off + BODY_IDX_POS_Y]
        var py_d = def_[off + BODY_IDX_POS_Y]
        var pz_x = xml[off + BODY_IDX_POS_Z]
        var pz_d = def_[off + BODY_IDX_POS_Z]
        var bqx_x = xml[off + BODY_IDX_QUAT_X]
        var bqx_d = def_[off + BODY_IDX_QUAT_X]
        var bqy_x = xml[off + BODY_IDX_QUAT_Y]
        var bqy_d = def_[off + BODY_IDX_QUAT_Y]
        var bqz_x = xml[off + BODY_IDX_QUAT_Z]
        var bqz_d = def_[off + BODY_IDX_QUAT_Z]
        var bqw_x = xml[off + BODY_IDX_QUAT_W]
        var bqw_d = def_[off + BODY_IDX_QUAT_W]
        var par_x = xml[off + BODY_IDX_PARENT]
        var par_d = def_[off + BODY_IDX_PARENT]
        var cx_x = xml[off + BODY_IDX_IPOS_X]
        var cx_d = def_[off + BODY_IDX_IPOS_X]
        var cy_x = xml[off + BODY_IDX_IPOS_Y]
        var cy_d = def_[off + BODY_IDX_IPOS_Y]
        var cz_x = xml[off + BODY_IDX_IPOS_Z]
        var cz_d = def_[off + BODY_IDX_IPOS_Z]
        var qx_x = xml[off + BODY_IDX_IQUAT_X]
        var qx_d = def_[off + BODY_IDX_IQUAT_X]
        var qy_x = xml[off + BODY_IDX_IQUAT_Y]
        var qy_d = def_[off + BODY_IDX_IQUAT_Y]
        var qz_x = xml[off + BODY_IDX_IQUAT_Z]
        var qz_d = def_[off + BODY_IDX_IQUAT_Z]
        var qw_x = xml[off + BODY_IDX_IQUAT_W]
        var qw_d = def_[off + BODY_IDX_IQUAT_W]
        print("\n  [body", b, body_names[b], "]")
        print("    mass     xml=", m_x, " def=", m_d, marker(m_x, m_d))
        print("    inv_mass xml=", im_x, " def=", im_d, marker(im_x, im_d))
        print("    ixx      xml=", ix_x, " def=", ix_d, marker(ix_x, ix_d))
        print("    iyy      xml=", iy_x, " def=", iy_d, marker(iy_x, iy_d))
        print("    izz      xml=", iz_x, " def=", iz_d, marker(iz_x, iz_d))
        print("    inv_ixx  xml=", iix_x, " def=", iix_d, marker(iix_x, iix_d))
        print("    inv_iyy  xml=", iiy_x, " def=", iiy_d, marker(iiy_x, iiy_d))
        print("    inv_izz  xml=", iiz_x, " def=", iiz_d, marker(iiz_x, iiz_d))
        print(
            "    pos    xml=(",
            px_x,
            py_x,
            pz_x,
            ") def=(",
            px_d,
            py_d,
            pz_d,
            ")",
            marker(px_x, px_d) + marker(py_x, py_d) + marker(pz_x, pz_d),
        )
        print(
            "    quat   xml=(",
            bqx_x,
            bqy_x,
            bqz_x,
            bqw_x,
            ") def=(",
            bqx_d,
            bqy_d,
            bqz_d,
            bqw_d,
            ")",
            marker(bqx_x, bqx_d)
            + marker(bqy_x, bqy_d)
            + marker(bqz_x, bqz_d)
            + marker(bqw_x, bqw_d),
        )
        print("    parent xml=", par_x, " def=", par_d, marker(par_x, par_d))
        print(
            "    ipos   xml=(",
            cx_x,
            cy_x,
            cz_x,
            ") def=(",
            cx_d,
            cy_d,
            cz_d,
            ")",
            marker(cx_x, cx_d) + marker(cy_x, cy_d) + marker(cz_x, cz_d),
        )
        print(
            "    iquat  xml=(",
            qx_x,
            qy_x,
            qz_x,
            qw_x,
            ") def=(",
            qx_d,
            qy_d,
            qz_d,
            qw_d,
            ")",
            marker(qx_x, qx_d)
            + marker(qy_x, qy_d)
            + marker(qz_x, qz_d)
            + marker(qw_x, qw_d),
        )

    # ── Joints ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(
        "JOINTS  (type, body_id, qpos_adr, dof_adr, pos, axis,"
        " armature, damping, stiffness, range, tau_limit, solimp_lim, qpos0)"
    )
    print("=" * 70)
    for j in range(NJOINT):
        var off = model_joint_offset[NBODY](j)
        var jtype_x = xml[off + JOINT_IDX_TYPE]
        var jtype_d = def_[off + JOINT_IDX_TYPE]
        var jbody_x = xml[off + JOINT_IDX_BODY_ID]
        var jbody_d = def_[off + JOINT_IDX_BODY_ID]
        var jqadr_x = xml[off + JOINT_IDX_QPOS_ADR]
        var jqadr_d = def_[off + JOINT_IDX_QPOS_ADR]
        var jdadr_x = xml[off + JOINT_IDX_DOF_ADR]
        var jdadr_d = def_[off + JOINT_IDX_DOF_ADR]
        var jpx_x = xml[off + JOINT_IDX_POS_X]
        var jpx_d = def_[off + JOINT_IDX_POS_X]
        var jpy_x = xml[off + JOINT_IDX_POS_Y]
        var jpy_d = def_[off + JOINT_IDX_POS_Y]
        var jpz_x = xml[off + JOINT_IDX_POS_Z]
        var jpz_d = def_[off + JOINT_IDX_POS_Z]
        var jax_x = xml[off + JOINT_IDX_AXIS_X]
        var jax_d = def_[off + JOINT_IDX_AXIS_X]
        var jay_x = xml[off + JOINT_IDX_AXIS_Y]
        var jay_d = def_[off + JOINT_IDX_AXIS_Y]
        var jaz_x = xml[off + JOINT_IDX_AXIS_Z]
        var jaz_d = def_[off + JOINT_IDX_AXIS_Z]
        var arm_x = xml[off + JOINT_IDX_ARMATURE]
        var arm_d = def_[off + JOINT_IDX_ARMATURE]
        var dmp_x = xml[off + JOINT_IDX_DAMPING]
        var dmp_d = def_[off + JOINT_IDX_DAMPING]
        var stf_x = xml[off + JOINT_IDX_STIFFNESS]
        var stf_d = def_[off + JOINT_IDX_STIFFNESS]
        var rn_x = xml[off + JOINT_IDX_RANGE_MIN]
        var rn_d = def_[off + JOINT_IDX_RANGE_MIN]
        var rx_x = xml[off + JOINT_IDX_RANGE_MAX]
        var rx_d = def_[off + JOINT_IDX_RANGE_MAX]
        var tau_x = xml[off + JOINT_IDX_TAU_LIMIT]
        var tau_d = def_[off + JOINT_IDX_TAU_LIMIT]
        var s0_x = xml[off + JOINT_IDX_SOLIMP_LIMIT_0]
        var s0_d = def_[off + JOINT_IDX_SOLIMP_LIMIT_0]
        var s1_x = xml[off + JOINT_IDX_SOLIMP_LIMIT_1]
        var s1_d = def_[off + JOINT_IDX_SOLIMP_LIMIT_1]
        var s2_x = xml[off + JOINT_IDX_SOLIMP_LIMIT_2]
        var s2_d = def_[off + JOINT_IDX_SOLIMP_LIMIT_2]
        var q0_x = xml[off + JOINT_IDX_QPOS0]
        var q0_d = def_[off + JOINT_IDX_QPOS0]

        print("\n  [joint", j, joint_names[j], "]")
        print(
            "    type      xml=",
            jtype_x,
            " def=",
            jtype_d,
            marker(jtype_x, jtype_d),
        )
        print(
            "    body_id   xml=",
            jbody_x,
            " def=",
            jbody_d,
            marker(jbody_x, jbody_d),
        )
        print(
            "    qpos_adr  xml=",
            jqadr_x,
            " def=",
            jqadr_d,
            marker(jqadr_x, jqadr_d),
        )
        print(
            "    dof_adr   xml=",
            jdadr_x,
            " def=",
            jdadr_d,
            marker(jdadr_x, jdadr_d),
        )
        print(
            "    pos       xml=(",
            jpx_x,
            jpy_x,
            jpz_x,
            ") def=(",
            jpx_d,
            jpy_d,
            jpz_d,
            ")",
            marker(jpx_x, jpx_d) + marker(jpy_x, jpy_d) + marker(jpz_x, jpz_d),
        )
        print(
            "    axis      xml=(",
            jax_x,
            jay_x,
            jaz_x,
            ") def=(",
            jax_d,
            jay_d,
            jaz_d,
            ")",
            marker(jax_x, jax_d) + marker(jay_x, jay_d) + marker(jaz_x, jaz_d),
        )
        print("    armature  xml=", arm_x, " def=", arm_d, marker(arm_x, arm_d))
        print("    damping   xml=", dmp_x, " def=", dmp_d, marker(dmp_x, dmp_d))
        print("    stiffness xml=", stf_x, " def=", stf_d, marker(stf_x, stf_d))
        print(
            "    range     xml=(",
            rn_x,
            rx_x,
            ") def=(",
            rn_d,
            rx_d,
            ")",
            marker(rn_x, rn_d) + marker(rx_x, rx_d),
        )
        print("    tau_limit xml=", tau_x, " def=", tau_d, marker(tau_x, tau_d))
        print(
            "    solimp_lim xml=(",
            s0_x,
            s1_x,
            s2_x,
            ") def=(",
            s0_d,
            s1_d,
            s2_d,
            ")",
            marker(s0_x, s0_d) + marker(s1_x, s1_d) + marker(s2_x, s2_d),
        )
        print("    qpos0     xml=", q0_x, " def=", q0_d, marker(q0_x, q0_d))

    # ── Geoms ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(
        "GEOMS  (type, body, pos, quat, rbound, radius, half_length,"
        " contype, conaffinity, condim, friction, solref, solimp)"
    )
    print("=" * 70)
    for g in range(NGEOM):
        var off = model_geom_offset[NBODY, NJOINT](g)
        var gt_x = xml[off + GEOM_IDX_TYPE]
        var gt_d = def_[off + GEOM_IDX_TYPE]
        var gb_x = xml[off + GEOM_IDX_BODY]
        var gb_d = def_[off + GEOM_IDX_BODY]
        var gpx_x = xml[off + GEOM_IDX_POS_X]
        var gpx_d = def_[off + GEOM_IDX_POS_X]
        var gpy_x = xml[off + GEOM_IDX_POS_Y]
        var gpy_d = def_[off + GEOM_IDX_POS_Y]
        var gpz_x = xml[off + GEOM_IDX_POS_Z]
        var gpz_d = def_[off + GEOM_IDX_POS_Z]
        var gqx_x = xml[off + GEOM_IDX_QUAT_X]
        var gqx_d = def_[off + GEOM_IDX_QUAT_X]
        var gqy_x = xml[off + GEOM_IDX_QUAT_Y]
        var gqy_d = def_[off + GEOM_IDX_QUAT_Y]
        var gqz_x = xml[off + GEOM_IDX_QUAT_Z]
        var gqz_d = def_[off + GEOM_IDX_QUAT_Z]
        var gqw_x = xml[off + GEOM_IDX_QUAT_W]
        var gqw_d = def_[off + GEOM_IDX_QUAT_W]
        var rb_x = xml[off + GEOM_IDX_RBOUND]
        var rb_d = def_[off + GEOM_IDX_RBOUND]
        var r_x = xml[off + GEOM_IDX_RADIUS]
        var r_d = def_[off + GEOM_IDX_RADIUS]
        var hl_x = xml[off + GEOM_IDX_HALF_LENGTH]
        var hl_d = def_[off + GEOM_IDX_HALF_LENGTH]
        var ct_x = xml[off + GEOM_IDX_CONTYPE]
        var ct_d = def_[off + GEOM_IDX_CONTYPE]
        var ca_x = xml[off + GEOM_IDX_CONAFFINITY]
        var ca_d = def_[off + GEOM_IDX_CONAFFINITY]
        var cd_x = xml[off + GEOM_IDX_CONDIM]
        var cd_d = def_[off + GEOM_IDX_CONDIM]
        var fr_x = xml[off + GEOM_IDX_FRICTION]
        var fr_d = def_[off + GEOM_IDX_FRICTION]
        var sr0_x = xml[off + GEOM_IDX_SOLREF_0]
        var sr0_d = def_[off + GEOM_IDX_SOLREF_0]
        var sr1_x = xml[off + GEOM_IDX_SOLREF_1]
        var sr1_d = def_[off + GEOM_IDX_SOLREF_1]
        var si0_x = xml[off + GEOM_IDX_SOLIMP_0]
        var si0_d = def_[off + GEOM_IDX_SOLIMP_0]
        var si1_x = xml[off + GEOM_IDX_SOLIMP_1]
        var si1_d = def_[off + GEOM_IDX_SOLIMP_1]
        var si2_x = xml[off + GEOM_IDX_SOLIMP_2]
        var si2_d = def_[off + GEOM_IDX_SOLIMP_2]
        print("\n  [geom", g, geom_names[g], "]")
        print("    type        xml=", gt_x, " def=", gt_d, marker(gt_x, gt_d))
        print("    body        xml=", gb_x, " def=", gb_d, marker(gb_x, gb_d))
        print(
            "    pos         xml=(",
            gpx_x,
            gpy_x,
            gpz_x,
            ") def=(",
            gpx_d,
            gpy_d,
            gpz_d,
            ")",
            marker(gpx_x, gpx_d) + marker(gpy_x, gpy_d) + marker(gpz_x, gpz_d),
        )
        print(
            "    quat        xml=(",
            gqx_x,
            gqy_x,
            gqz_x,
            gqw_x,
            ") def=(",
            gqx_d,
            gqy_d,
            gqz_d,
            gqw_d,
            ")",
            marker(gqx_x, gqx_d)
            + marker(gqy_x, gqy_d)
            + marker(gqz_x, gqz_d)
            + marker(gqw_x, gqw_d),
        )
        print("    rbound      xml=", rb_x, " def=", rb_d, marker(rb_x, rb_d))
        print("    radius      xml=", r_x, " def=", r_d, marker(r_x, r_d))
        print("    half_length xml=", hl_x, " def=", hl_d, marker(hl_x, hl_d))
        print("    contype     xml=", ct_x, " def=", ct_d, marker(ct_x, ct_d))
        print("    conaffinity xml=", ca_x, " def=", ca_d, marker(ca_x, ca_d))
        print("    condim      xml=", cd_x, " def=", cd_d, marker(cd_x, cd_d))
        print("    friction    xml=", fr_x, " def=", fr_d, marker(fr_x, fr_d))
        print(
            "    solref      xml=(",
            sr0_x,
            sr1_x,
            ") def=(",
            sr0_d,
            sr1_d,
            ")",
            marker(sr0_x, sr0_d) + marker(sr1_x, sr1_d),
        )
        print(
            "    solimp      xml=(",
            si0_x,
            si1_x,
            si2_x,
            ") def=(",
            si0_d,
            si1_d,
            si2_d,
            ")",
            marker(si0_x, si0_d) + marker(si1_x, si1_d) + marker(si2_x, si2_d),
        )

    # ── Metadata ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("METADATA  (gravity, timestep, solref/solimp contact+limit)")
    print("=" * 70)
    var meta = model_metadata_offset[NBODY, NJOINT]()
    var gz_x = xml[meta + MODEL_META_IDX_GRAVITY_Z]
    var gz_d = def_[meta + MODEL_META_IDX_GRAVITY_Z]
    var ts_x = xml[meta + MODEL_META_IDX_TIMESTEP]
    var ts_d = def_[meta + MODEL_META_IDX_TIMESTEP]
    var r0c_x = xml[meta + MODEL_META_IDX_SOLREF_CONTACT_0]
    var r0c_d = def_[meta + MODEL_META_IDX_SOLREF_CONTACT_0]
    var r1c_x = xml[meta + MODEL_META_IDX_SOLREF_CONTACT_1]
    var r1c_d = def_[meta + MODEL_META_IDX_SOLREF_CONTACT_1]
    var i0c_x = xml[meta + MODEL_META_IDX_SOLIMP_CONTACT_0]
    var i0c_d = def_[meta + MODEL_META_IDX_SOLIMP_CONTACT_0]
    var i1c_x = xml[meta + MODEL_META_IDX_SOLIMP_CONTACT_1]
    var i1c_d = def_[meta + MODEL_META_IDX_SOLIMP_CONTACT_1]
    var i2c_x = xml[meta + MODEL_META_IDX_SOLIMP_CONTACT_2]
    var i2c_d = def_[meta + MODEL_META_IDX_SOLIMP_CONTACT_2]
    var r0l_x = xml[meta + MODEL_META_IDX_SOLREF_LIMIT_0]
    var r0l_d = def_[meta + MODEL_META_IDX_SOLREF_LIMIT_0]
    var r1l_x = xml[meta + MODEL_META_IDX_SOLREF_LIMIT_1]
    var r1l_d = def_[meta + MODEL_META_IDX_SOLREF_LIMIT_1]
    var i0l_x = xml[meta + MODEL_META_IDX_SOLIMP_LIMIT_0]
    var i0l_d = def_[meta + MODEL_META_IDX_SOLIMP_LIMIT_0]
    var i1l_x = xml[meta + MODEL_META_IDX_SOLIMP_LIMIT_1]
    var i1l_d = def_[meta + MODEL_META_IDX_SOLIMP_LIMIT_1]
    var i2l_x = xml[meta + MODEL_META_IDX_SOLIMP_LIMIT_2]
    var i2l_d = def_[meta + MODEL_META_IDX_SOLIMP_LIMIT_2]
    print("  gravity_z      xml=", gz_x, " def=", gz_d, marker(gz_x, gz_d))
    print("  timestep       xml=", ts_x, " def=", ts_d, marker(ts_x, ts_d))
    print(
        "  solref_contact xml=(",
        r0c_x,
        r1c_x,
        ") def=(",
        r0c_d,
        r1c_d,
        ")",
        marker(r0c_x, r0c_d) + marker(r1c_x, r1c_d),
    )
    print(
        "  solimp_contact xml=(",
        i0c_x,
        i1c_x,
        i2c_x,
        ") def=(",
        i0c_d,
        i1c_d,
        i2c_d,
        ")",
        marker(i0c_x, i0c_d) + marker(i1c_x, i1c_d) + marker(i2c_x, i2c_d),
    )
    print(
        "  solref_limit   xml=(",
        r0l_x,
        r1l_x,
        ") def=(",
        r0l_d,
        r1l_d,
        ")",
        marker(r0l_x, r0l_d) + marker(r1l_x, r1l_d),
    )
    print(
        "  solimp_limit   xml=(",
        i0l_x,
        i1l_x,
        i2l_x,
        ") def=(",
        i0l_d,
        i1l_d,
        i2l_d,
        ")",
        marker(i0l_x, i0l_d) + marker(i1l_x, i1l_d) + marker(i2l_x, i2l_d),
    )


def main():
    print("Comparing HalfCheetah XML vs Def GPU model buffers")
    print(
        "MODEL_SIZE=",
        MODEL_SIZE,
        " NBODY=",
        NBODY,
        " NJOINT=",
        NJOINT,
        " NV=",
        NV,
        " NGEOM=",
        NGEOM,
    )

    var ctx = DeviceContext()

    # Allocate and initialise both model buffers on GPU
    var xml_dev = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu[DTYPE](ctx, xml_dev)

    var def_dev = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModelDef.init_model_gpu[DTYPE](ctx, def_dev)

    ctx.synchronize()

    # Copy both back to host (device → host)
    var xml_host = ctx.enqueue_create_host_buffer[DTYPE](MODEL_SIZE)
    var def_host = ctx.enqueue_create_host_buffer[DTYPE](MODEL_SIZE)
    xml_dev.enqueue_copy_to(xml_host)
    def_dev.enqueue_copy_to(def_host)
    ctx.synchronize()

    # Load into plain lists for easy indexing
    var xml_vals = List[Float32](capacity=MODEL_SIZE)
    var def_vals = List[Float32](capacity=MODEL_SIZE)
    for i in range(MODEL_SIZE):
        xml_vals.append(Float32(xml_host[i]))
        def_vals.append(Float32(def_host[i]))

    compare_models(xml_vals, def_vals)

    # ── invweight0 section ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INVWEIGHT0  (body_invweight0[NBODY*2] + dof_invweight0[NV])")
    print("=" * 70)
    comptime INVW_BODY_OFF = model_body_invweight0_offset[
        NBODY, NJOINT, NGEOM
    ]()
    comptime INVW_DOF_OFF = model_dof_invweight0_offset[NBODY, NJOINT, NGEOM]()
    print(
        "  body_invweight0_offset =",
        INVW_BODY_OFF,
        "  dof_invweight0_offset =",
        INVW_DOF_OFF,
    )
    var bnames = materialize[BODY_NAMES]()
    var jnames = materialize[JOINT_NAMES]()
    for b in range(NBODY):
        var t_x = xml_vals[INVW_BODY_OFF + b * 2 + 0]
        var t_d = def_vals[INVW_BODY_OFF + b * 2 + 0]
        var r_x = xml_vals[INVW_BODY_OFF + b * 2 + 1]
        var r_d = def_vals[INVW_BODY_OFF + b * 2 + 1]
        print(
            "  body[",
            b,
            bnames[b],
            "] trans xml=",
            t_x,
            " def=",
            t_d,
            marker(t_x, t_d),
            "  rot xml=",
            r_x,
            " def=",
            r_d,
            marker(r_x, r_d),
        )
    for v in range(NV):
        var vw_x = xml_vals[INVW_DOF_OFF + v]
        var vw_d = def_vals[INVW_DOF_OFF + v]
        print(
            "  dof[",
            v,
            jnames[v],
            "] xml=",
            vw_x,
            " def=",
            vw_d,
            marker(vw_x, vw_d),
        )

    # ── Raw buffer diff — catch any remaining differences ─────────────────────
    print("\n" + "=" * 70)
    print("RAW BUFFER DIFF (all indices with |xml - def| > 1e-4)")
    print("=" * 70)
    var ndiff = 0
    for i in range(MODEL_SIZE):
        var a = xml_vals[i]
        var b = def_vals[i]
        if abs(a - b) > Float32(1e-4):
            print("  buf[", i, "]  xml=", a, " def=", b, " diff=", a - b)
            ndiff += 1
    if ndiff == 0:
        print(
            "  (no differences found — buffers are identical within tolerance)"
        )
    else:
        print("\n  Total differing indices:", ndiff)

    print("\nDone.")
