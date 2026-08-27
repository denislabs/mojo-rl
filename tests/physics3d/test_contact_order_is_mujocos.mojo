"""Our contact ARRAY is in MuJoCo's order — compared INDEX BY INDEX.

`mj_broadphase` ends by sorting its bodyflex pair list
(`bfsort(bfpair, buf, npair, NULL)`, `engine_collision_driver.c:1683`) by the
signature `(min(b1,b2)<<16) + max(b1,b2)`, and the narrow phase then runs body
pair by body pair in that order. See `collision/contact_order.mojo`.

WHY AN ORDER NEEDS ITS OWN GATE

The primal Newton solve is a global minimisation and is order-INDEPENDENT, so
a permuted contact set produces the same `qacc` and every trajectory gate stays
green. Three things are not order-independent — `mj_solNoSlip` and `solPGS`
are Gauss-Seidel, and so is any index-by-index comparison with MuJoCo.

⚠⚠ AND EVERY EXISTING CONTACT GATE WAS BLIND TO IT. `csweep.py` scores
multiplicity and `|d(dist)|`; `psweep.py` and `cpos.py` match each of our
contacts to MuJoCo's NEAREST one. A contact set that is correct but PERMUTED
scores perfectly clean in all three — `hello_robot_stretch_3` did, at
`dpos <= 7e-16` on every contact, while its board row sat at 4.566e-05 and the
whole of that row was the permutation. Matching by proximity cannot see an
order; only comparing the sequences can.

WHY THE FIXTURE IS A MENAGERIE MODEL AND NOT AN INLINE XML

The defect needs a scene where the SAP sweep's AABB order disagrees with the
body-pair order, which takes three bodies contacting a fourth at particular
positions. Hand-writing an XML that reliably produces that is guesswork, and a
fixture that happened not to produce it would pass while testing nothing.
`hello_robot_stretch_3` IS the case, in the pose the board draws:

    MuJoCo   (9,87) (9,89) (27,87) (34,87) (34,87) (34,87)
    before   (9,87) (9,89) (34,87) (34,87) (34,87) (27,87)

`(27,87)` is `link_aruco_right_base x link_DW3_wrist_pitch`, body pair (5,19),
and it belongs between the (1,19) pairs and the (8,19) ones.

WHAT IS ASSERTED, AND THE NON-VACUITY

  1. our body-pair sequence equals MuJoCo's, index by index, with MuJoCo's
     EXCLUDED contacts filtered out (they are reported but generate no row —
     `csweep.py` documents the same filter, and without it ~30 false rows);
  2. the fixture actually discriminates: it must carry at least three DISTINCT
     body pairs, or an order gate on it means nothing;
  3. the sequence is non-decreasing in the body-pair key — the invariant the
     sort exists to establish, asserted directly so a future change that
     removes the sort fails here even if (1) is somehow still satisfied.

⚠ THE WORLD BODY IS SPELLED TWO WAYS. `detect_contacts` writes 0 and the SAP
path writes -1; both mean world, and MuJoCo's signature uses 0. The key
normalises a negative id to 0, and so does this file.

Run with:
    pixi run mojo run -I . tests/physics3d/test_contact_order_is_mujocos.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims, init_hfield_data
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.studio.stepping import StudioImpFastEll
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE,
    CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    KEY_META_SIZE, KEY_IDX_NQPOS,
)

comptime DT = DType.float64

comptime STRETCH3 = String(
    "references/mujoco_menagerie-main/hello_robot_stretch_3/scene.xml"
)


def _mj_pairs(mut lo_out: List[Int], mut hi_out: List[Int]) raises:
    """MuJoCo's contact body pairs, in ITS order, excluded contacts dropped.

    ⚠ `mj_forward` AT THE KEYFRAME WITH ZERO CONTROL, which is the pose the
    contact set is a property of. The board applies a random `ctrl` and steps,
    but `ctrl` cannot move contacts within the step that detects them — they
    are a function of `qpos` alone — so the simpler state is the honest one
    and it keeps this file independent of the board's RNG.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(STRETCH3)
    var d = mujoco.MjData(m)
    if Int(py=m.nkey) > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for i in range(Int(py=d.ncon)):
        var c = d.contact[i]
        # ⚠ EXCLUDED CONTACTS ARE REPORTED BUT GENERATE NO ROW. Keeping them
        # would shift every index by one and fail this file for bookkeeping.
        if Int(py=c.exclude) != 0:
            continue
        var b1 = Int(py=m.geom_bodyid[c.geom1])
        var b2 = Int(py=m.geom_bodyid[c.geom2])
        lo_out.append(b1 if b1 < b2 else b2)
        hi_out.append(b2 if b1 < b2 else b1)


def _our_pairs(mut lo_out: List[Int], mut hi_out: List[Int]) raises:
    """Our contact body pairs, in OUR array order, after one step."""
    var src = read_model_source(STRETCH3)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)

    var d = Data[DT, DynDims, 1](dims)
    # ⚠ THE HEIGHTFIELD GRID IS STATE — a `Data` that skips this holds a grid
    # of ZEROS, a flat terrain that collides perfectly happily and is not the
    # surface the model declared. `drive.mojo` carries the same line.
    init_hfield_data(d, m)
    var nq = dims.get_nq()
    # ⚠ `qpos0` FIRST, THEN THE KEYFRAME OVER IT. A keyframe may be SHORTER
    # than `nq` (`KEY_IDX_NQPOS`), and the dofs it does not mention must hold
    # the model's rest pose rather than whatever a fresh `Data` had. Reading
    # `key_qpos` from index 0 without the `keyidx * nq` stride is the other
    # half of the same mistake; both were in the first draft of this file and
    # together they produced 19 contacts against MuJoCo's 8.
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    if dims.get_nkey() > 0:
        var nqp = Int(Float64(sf.key_meta.data[0 * KEY_META_SIZE + KEY_IDX_NQPOS]))
        for i in range(min(nqp, nq)):
            d.qpos.data[i] = sf.key_qpos.data[0 * nq + i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)

    # ZERO CONTROL, matching `mj_forward` on the reference side. The contact
    # set is a function of `qpos` alone, so this only has to agree.
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact if nact > 0 else 1, fill=0.0)
    var act = List[Scalar[DT]](length=nact if nact > 0 else 1, fill=Scalar[DT](0))

    # `<option cone="elliptic" integrator="implicitfast">` — the model's own.
    var integ = StudioImpFastEll(dims)
    for i in range(dims.get_nv()):
        d.qfrc.data[i] = Scalar[DT](0)
    apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
    integ.step["cpu"](d, m)

    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    for k in range(nc):
        var a = Int(Float64(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_A]))
        var b = Int(Float64(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_B]))
        if a < 0:
            a = 0
        if b < 0:
            b = 0
        lo_out.append(a if a < b else b)
        hi_out.append(b if a < b else a)


def test_contact_order_matches_mujoco() raises:
    """The sequences, index by index. See the module docstring."""
    print("=== contact ORDER: hello_robot_stretch_3, ours vs MuJoCo ===")
    var mj_lo = List[Int]()
    var mj_hi = List[Int]()
    _mj_pairs(mj_lo, mj_hi)
    var ou_lo = List[Int]()
    var ou_hi = List[Int]()
    _our_pairs(ou_lo, ou_hi)
    var n_mj = len(mj_lo)
    print("  MuJoCo contacts (excluded dropped):", n_mj,
          "   ours:", len(ou_lo))

    for k in range(min(n_mj, len(ou_lo))):
        print("   [", k, "] mj (", mj_lo[k], ",", mj_hi[k],
              ")   ours (", ou_lo[k], ",", ou_hi[k], ")")

    assert_true(
        n_mj == len(ou_lo),
        String(
            "different NUMBER of contacts, so the order cannot be compared:"
            " MuJoCo "
        )
        + String(n_mj)
        + " (excluded dropped) vs ours "
        + String(len(ou_lo))
        + " — this file gates the ORDER, so fix the SET first",
    )

    # (2) NON-VACUITY: an order gate needs more than one body pair to order.
    var distinct = 0
    for k in range(len(ou_lo)):
        var seen = False
        for j in range(k):
            if ou_lo[j] == ou_lo[k] and ou_hi[j] == ou_hi[k]:
                seen = True
        if not seen:
            distinct += 1
    print("  distinct body pairs:", distinct)
    assert_true(
        distinct >= 3,
        String(
            "VACUOUS: only "
        )
        + String(distinct)
        + " distinct body pair(s) in this contact set. An ordering gate needs"
        " a scene whose sweep order can disagree with the body-pair order;"
        " with one pair every permutation is the identity. Pick a pose where"
        " stretch_3's wrist touches the base, the aruco mount AND a wheel.",
    )

    # (1) THE SEQUENCES.
    for k in range(n_mj):
        assert_true(
            mj_lo[k] == ou_lo[k] and mj_hi[k] == ou_hi[k],
            String("contact ")
            + String(k)
            + " is body pair ("
            + String(ou_lo[k]) + "," + String(ou_hi[k])
            + ") for us and ("
            + String(mj_lo[k]) + "," + String(mj_hi[k])
            + ") for MuJoCo. The SET may still be right — every existing"
            " contact gate matches by proximity and would call this clean."
            " `mj_broadphase` sorts body pairs by (min<<16)+max"
            " (engine_collision_driver.c:1683); see"
            " `collision/contact_order.mojo`.",
        )

    # (3) THE INVARIANT the sort exists to establish, asserted directly.
    for k in range(1, len(ou_lo)):
        var ok = (
            ou_lo[k - 1] < ou_lo[k]
            or (ou_lo[k - 1] == ou_lo[k] and ou_hi[k - 1] <= ou_hi[k])
        )
        assert_true(
            ok,
            String("the contact array is not sorted by body pair at index ")
            + String(k)
            + ": ("
            + String(ou_lo[k - 1]) + "," + String(ou_hi[k - 1])
            + ") before ("
            + String(ou_lo[k]) + "," + String(ou_hi[k])
            + "). `sort_contacts_mujoco_order` is either not being called on"
            " this path or is not stable.",
        )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
