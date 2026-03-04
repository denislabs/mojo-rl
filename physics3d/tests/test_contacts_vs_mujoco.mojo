"""Test Contact Detection against MuJoCo reference.

Compares our contact detection output (positions, normals, distances, body pairs)
with MuJoCo's contact data for the HalfCheetah model at multiple configurations.

This is the #1 suspect for full-step divergence. Wrong contact positions
lead to wrong Jacobians, which lead to wrong constraint forces.

MuJoCo reference: mj_data.contact (after mj_step1 or mj_forward)
  - ncon: number of contacts
  - contact[i].pos: contact position (3D world)
  - contact[i].frame: contact frame (3x3 row-major: normal, tangent1, tangent2)
  - contact[i].dist: signed penetration distance (negative = penetration)
  - contact[i].geom: geom pair indices [geom1, geom2]

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_contacts_vs_mujoco.mojo
"""

from python import Python, PythonObject
from std.math import abs, sqrt
from std.collections import InlineArray
from testing import assert_true, TestSuite

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.collision.contact_detection import detect_contacts
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY  # 7
comptime NJOINT = HalfCheetahModel.NJOINT  # 9
comptime NGEOM = HalfCheetahModel.NGEOM  # 9
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS  # 20

# Tolerances
comptime POS_TOL: Float64 = 1e-3  # Contact position
comptime DIST_TOL: Float64 = 1e-3  # Penetration distance
comptime NORMAL_DOT_MIN: Float64 = 0.99  # Normal direction (dot product)


# =============================================================================
# Helpers
# =============================================================================


fn _geom_body_from_mujoco(mj_model: PythonObject, geom_id: Int) raises -> Int:
    """Map MuJoCo geom index to our body index.

    Both MuJoCo and our engine use 0-indexed bodies with worldbody=0.
    Direct mapping: MuJoCo body N = our body N.
    """
    return Int(py=mj_model.geom_bodyid[geom_id])


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_contacts(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
) raises:
    """Detect contacts in both engines with identical state, compare."""
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model, data)
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])

    # Run FK (needed for contact detection)
    forward_kinematics(model, data)

    # Run contact detection
    detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model, data
    )

    var our_ncon = data.num_contacts
    print("  Our contacts:", our_ncon)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]

    # mj_step1 runs FK + collision detection + constraint setup (NO solver)
    mujoco.mj_step1(mj_model, mj_data)

    var mj_ncon = Int(py=mj_data.ncon)
    print("  MJ  contacts:", mj_ncon)

    # Print our contacts
    for c in range(our_ncon):
        var ci = data.contacts[c]
        print(
            "  Our[",
            c,
            "] body(",
            ci.body_a,
            ",",
            ci.body_b,
            ") pos(",
            Float64(ci.pos_x),
            ",",
            Float64(ci.pos_y),
            ",",
            Float64(ci.pos_z),
            ") dist=",
            Float64(ci.dist),
            " n(",
            Float64(ci.normal_x),
            ",",
            Float64(ci.normal_y),
            ",",
            Float64(ci.normal_z),
            ") fric=",
            Float64(ci.friction),
        )

    # Print MuJoCo contacts
    for c in range(mj_ncon):
        var mj_pos = mj_data.contact[c].pos.flatten().tolist()
        var mj_frame = mj_data.contact[c].frame.flatten().tolist()
        var mj_dist = Float64(py=mj_data.contact[c].dist)
        var mj_geom = mj_data.contact[c].geom.flatten().tolist()
        var geom1 = Int(py=mj_geom[0])
        var geom2 = Int(py=mj_geom[1])
        var body1 = _geom_body_from_mujoco(mj_model, geom1)
        var body2 = _geom_body_from_mujoco(mj_model, geom2)
        var nx = Float64(py=mj_frame[0])
        var ny = Float64(py=mj_frame[1])
        var nz = Float64(py=mj_frame[2])
        print(
            "  MJ [",
            c,
            "] body(",
            body1,
            ",",
            body2,
            ") geom(",
            geom1,
            ",",
            geom2,
            ") pos(",
            Float64(py=mj_pos[0]),
            ",",
            Float64(py=mj_pos[1]),
            ",",
            Float64(py=mj_pos[2]),
            ") dist=",
            mj_dist,
            " n(",
            nx,
            ",",
            ny,
            ",",
            nz,
            ")",
        )

    # === Compare ===
    var all_pass = True

    # 1. Check contact count
    if our_ncon != mj_ncon:
        print(
            "  FAIL: contact count mismatch! ours=", our_ncon, " mj=", mj_ncon
        )
        # Don't return False yet — continue to show details for debugging
        all_pass = False

    # 2. Match contacts by body pair
    # For each MuJoCo contact, find the closest matching contact in ours
    var matched = InlineArray[Int, MAX_CONTACTS](
        fill=-1
    )  # our idx matched to each mj contact

    for mc in range(mj_ncon):
        var mj_pos = mj_data.contact[mc].pos.flatten().tolist()
        var mj_frame = mj_data.contact[mc].frame.flatten().tolist()
        var mj_dist = Float64(py=mj_data.contact[mc].dist)
        var mj_geom = mj_data.contact[mc].geom.flatten().tolist()
        var mj_body1 = _geom_body_from_mujoco(mj_model, Int(py=mj_geom[0]))
        var mj_body2 = _geom_body_from_mujoco(mj_model, Int(py=mj_geom[1]))
        var mj_px = Float64(py=mj_pos[0])
        var mj_py = Float64(py=mj_pos[1])
        var mj_pz = Float64(py=mj_pos[2])
        var mj_nx = Float64(py=mj_frame[0])
        var mj_ny = Float64(py=mj_frame[1])
        var mj_nz = Float64(py=mj_frame[2])

        # Find best matching contact from ours
        var best_idx = -1
        var best_pos_err: Float64 = 1e10

        for oc in range(our_ncon):
            # Check if already matched
            var already = False
            for k in range(mc):
                if matched[k] == oc:
                    already = True
                    break
            if already:
                continue

            var ci = data.contacts[oc]
            # Match by body pair (order-independent)
            var our_b1 = ci.body_a
            var our_b2 = ci.body_b
            var body_match = (our_b1 == mj_body1 and our_b2 == mj_body2) or (
                our_b1 == mj_body2 and our_b2 == mj_body1
            )
            if not body_match:
                continue

            # Among body-matching contacts, pick closest by position
            var dx = Float64(ci.pos_x) - mj_px
            var dy = Float64(ci.pos_y) - mj_py
            var dz = Float64(ci.pos_z) - mj_pz
            var pos_err = sqrt(dx * dx + dy * dy + dz * dz)
            if pos_err < best_pos_err:
                best_pos_err = pos_err
                best_idx = oc

        if best_idx < 0:
            print(
                "  FAIL: no matching contact for MJ[",
                mc,
                "] body(",
                mj_body1,
                ",",
                mj_body2,
                ")",
            )
            all_pass = False
            continue

        matched[mc] = best_idx
        var ci = data.contacts[best_idx]

        # 3. Compare position
        var pos_err = best_pos_err
        if pos_err > POS_TOL:
            print(
                "  FAIL pos[",
                mc,
                "] err=",
                pos_err,
                " ours=(",
                Float64(ci.pos_x),
                ",",
                Float64(ci.pos_y),
                ",",
                Float64(ci.pos_z),
                ") mj=(",
                mj_px,
                ",",
                mj_py,
                ",",
                mj_pz,
                ")",
            )
            all_pass = False

        # 4. Compare normal direction (dot product)
        var our_nx = Float64(ci.normal_x)
        var our_ny = Float64(ci.normal_y)
        var our_nz = Float64(ci.normal_z)
        var dot = our_nx * mj_nx + our_ny * mj_ny + our_nz * mj_nz
        if dot < NORMAL_DOT_MIN:
            print(
                "  FAIL normal[",
                mc,
                "] dot=",
                dot,
                " ours=(",
                our_nx,
                ",",
                our_ny,
                ",",
                our_nz,
                ") mj=(",
                mj_nx,
                ",",
                mj_ny,
                ",",
                mj_nz,
                ")",
            )
            all_pass = False

        # 5. Compare penetration distance
        var dist_err = abs(Float64(ci.dist) - mj_dist)
        if dist_err > DIST_TOL:
            print(
                "  FAIL dist[",
                mc,
                "] err=",
                dist_err,
                " ours=",
                Float64(ci.dist),
                " mj=",
                mj_dist,
            )
            all_pass = False

    if all_pass:
        print("  ALL OK  ncon=", our_ncon)
    else:
        print("  FAILED")

    assert_true(all_pass, "Contacts mismatch for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


fn test_high_pose() raises:
    """Robot high above ground (rootz=0.5) — no contacts expected."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.5  # rootz high
    compare_contacts("High pose (no contacts)", qpos)


fn test_default_pose() raises:
    """Default pose (rootz=0.7, which is body_pos offset) — may or may not contact.
    """
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7  # rootz = MuJoCo default
    compare_contacts("Default pose (rootz=0.7)", qpos)


fn test_low_pose() raises:
    """Robot low (rootz=-0.3) — feet should be in contact with ground."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    compare_contacts("Low pose (rootz=-0.3)", qpos)


fn test_very_low_pose() raises:
    """Robot very low (rootz=-0.45) — multiple body parts in contact."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45  # rootz very low
    compare_contacts("Very low pose (rootz=-0.45)", qpos)


fn test_bent_legs() raises:
    """Bent legs with non-default joint angles — different contact geometry."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low enough for contact
    qpos[3] = -0.5  # bthigh bent
    qpos[4] = 0.8  # bshin extended
    qpos[6] = 0.5  # fthigh bent
    qpos[7] = -0.8  # fshin extended
    compare_contacts("Bent legs (various joint angles)", qpos)


fn test_tilted_body() raises:
    """Tilted body (rooty rotation) — asymmetric contacts."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.2  # rootz slightly low
    qpos[2] = 0.3  # rooty tilted forward
    compare_contacts("Tilted body (rooty=0.3)", qpos)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
