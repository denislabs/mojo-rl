"""AUD-32 — a heightfield contact with `margin > 0` is inflated on BOTH sides.

    pixi run mojo run -I . tests/physics3d/test_hfield_margin_vs_mujoco.mojo

`mjc_ConvexHField` raises every prism top by `margin` AND sets
`obj2.margin = margin`, so the geom's support grows by `margin/2`; the convex
query then runs with margin 0 and its distance is stored as is. A sphere of
radius 0.1 with margin 0.04 whose true distance to the field is -0.001 is
reported at -0.061 by MuJoCo. Ours inflated the prism only: -0.041, half a
margin too shallow on every heightfield contact, and the in-band cutoff was
`true < margin` where MuJoCo's is `true < 1.5 * margin`.

Three heights of the same sphere over the same flat-ish cell: penetrating,
inside the band, and beyond it (1.5 * margin = 0.06 above the surface gives
no contact). Count, distance and position are compared per height.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.physics3d.fields import Data, Model, DynDims, init_hfield_data
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE, CONTACT_IDX_POS_X, CONTACT_IDX_POS_Y, CONTACT_IDX_POS_Z,
    CONTACT_IDX_DIST, META_IDX_NUM_CONTACTS,
)

comptime DT = DType.float64
comptime MARGIN = 0.04

comptime HF_XML = String(
    """
<mujoco model="hfield margin">
  <option timestep="0.002"/>
  <asset>
    <hfield name="terrain" file="tests/physics3d/assets/hf_8x8.bin" size="0.5 0.5 0.2 0.1"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" hfield="terrain" pos="0 0 0"/>
    <body name="ba" pos="0.1 0.05 0.3"><freejoint/><geom name="ga" type="sphere" size="0.1" margin="0.04"/></body>
  </worldbody>
</mujoco>
"""
)


struct Built(Movable):
    var m: Model[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var dims: DynDims

    def __init__(out self) raises:
        var fmd = parse_xml_full(HF_XML, String("."))
        var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=64)
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var sf = spec_fields_runtime[DT](fmd, dims, m)
        var d = Data[DT, DynDims, 1](dims)
        init_hfield_data(d, m)
        for i in range(dims.get_nq()):
            d.qpos.data[i] = sf.qpos0.data[i]
        for i in range(dims.get_nv()):
            d.qvel.data[i] = Scalar[DT](0)
        self.m = m^
        self.d = d^
        self.dims = dims


def test_hfield_margin_matches_mujoco() raises:
    print("=== hfield + geom margin: distance and cutoff vs MuJoCo ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(HF_XML)
    var dat = mujoco.MjData(m)
    var b = Built()

    # the surface height under the sphere centre, from MuJoCo with margin 0:
    # place the sphere high, read the field's own answer through a probe run
    var heights = List[Float64]()
    heights.append(0.0)      # set below: penetrating by 1 mm
    heights.append(0.0)      # in the band
    heights.append(0.0)      # beyond 1.5 * margin
    # find the contact height by bisection on MuJoCo's ncon with margin 0
    m.geom_margin[1] = 0.0
    var lo = 0.05
    var hi = 0.4
    for _ in range(60):
        var mid = 0.5 * (lo + hi)
        dat.qpos[2] = mid
        mujoco.mj_forward(m, dat)
        if Int(py=dat.ncon) > 0:
            lo = mid
        else:
            hi = mid
    var z_touch = 0.5 * (lo + hi)
    m.geom_margin[1] = MARGIN
    print("  sphere centre height at first touch (margin 0):", z_touch)
    heights[0] = z_touch - 0.001
    heights[1] = z_touch + 0.035
    heights[2] = z_touch + 0.07

    var seen_band = False
    for h in range(3):
        var z = heights[h]
        dat.qpos[2] = z
        mujoco.mj_forward(m, dat)
        var mjn = Int(py=dat.ncon)
        b.d.qpos.data[2] = Scalar[DT](z)
        forward_kinematics["cpu", DT, DynDims, 1](b.d, b.m)
        detect_contacts["cpu", DT, DynDims, 1](b.d, b.m)
        var nc = Int(b.d.meta.data[META_IDX_NUM_CONTACTS])
        var worst_pos = Float64(0)
        var worst_dist = Float64(0)
        var used = List[Int](length=nc, fill=0)
        for i in range(mjn):
            var rx = Float64(py=dat.contact[i].pos[0])
            var ry = Float64(py=dat.contact[i].pos[1])
            var rz = Float64(py=dat.contact[i].pos[2])
            var rd = Float64(py=dat.contact[i].dist)
            var best = -1
            var bd = Float64(1e30)
            for k in range(nc):
                if used[k] == 1:
                    continue
                var o = k * CONTACT_SIZE
                var e = (
                    abs(Float64(b.d.contacts.data[o + CONTACT_IDX_POS_X]) - rx)
                    + abs(Float64(b.d.contacts.data[o + CONTACT_IDX_POS_Y]) - ry)
                    + abs(Float64(b.d.contacts.data[o + CONTACT_IDX_POS_Z]) - rz)
                )
                if e < bd:
                    bd = e
                    best = k
            if best < 0:
                continue
            used[best] = 1
            var dd = abs(Float64(b.d.contacts.data[best * CONTACT_SIZE + CONTACT_IDX_DIST]) - rd)
            if bd > worst_pos:
                worst_pos = bd
            if dd > worst_dist:
                worst_dist = dd
        var mjd = Float64(py=dat.contact[0].dist) if mjn > 0 else 0.0
        print("  height", h, " z =", z, " ncon MuJoCo", mjn, " ours", nc,
              " MuJoCo dist[0] =", mjd, " worst |d pos|", worst_pos,
              " worst |d dist|", worst_dist)
        assert_true(
            mjn == nc,
            "height " + String(h) + ": ours " + String(nc) + " contacts vs MuJoCo's "
            + String(mjn) + " — the in-band cutoff is not 1.5 * margin",
        )
        if mjn > 0:
            assert_true(
                worst_dist < 1e-6 and worst_pos < 1e-5,
                "height " + String(h) + ": dist off by " + String(worst_dist)
                + " (margin/2 = 0.02 is the old prism-only inflation), pos by "
                + String(worst_pos),
            )
        if h == 1 and mjn > 0:
            seen_band = True
    assert_true(seen_band, "the in-band height produced no MuJoCo contact; the fixture is vacuous")
    assert_true(Int(py=dat.ncon) == 0, "beyond 1.5 * margin MuJoCo must report nothing")
    _ = b^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
