"""`<geom gap>` — the band where a contact EXISTS and the solver cannot see it.

WHY THIS EXISTS
===============
MuJoCo 3.10.0 hands the narrowphase `margin + gap` as its cutoff
(`collisionTask`, engine_collision_driver.c:1871) and then passes `margin`
ALONE to `mj_setContact`, which sets `con->exclude = (con->dist >=
includemargin)`. So a contact whose distance falls in [margin, margin + gap) is
DETECTED, reported, and generates no constraint rows at all — `efc_address` is
-1 and the solver never sees it.

⚠⚠ THAT BAND IS NOT DECORATION: `<adhesion>` ACTS ON IT. `mj_transmission`'s
body arm averages the normal Jacobians of every contact touching the body,
in-gap ones included (it computes theirs directly, precisely because they have
no `efc` rows to read). flybody's eight claw and labrum pads all carry
`margin="0.0005" gap="0.0005"`, and at its keyframe one pad's ONLY contact sits
in the band at dist 9.880e-04. Without gap that pad had no contact and pulled
on nothing — the whole of a 0.985 discrepancy in `qfrc_actuator`.

⚠⚠ THIS WAS REFUSED OUTRIGHT UNTIL THE RULE WAS MEASURED, and the reason was
version drift, not laziness: `mj_setContact` gets `margin - gap` in the 3.3.6,
3.6.0 and main trees, `margin` in the 3.10.0 RUNTIME, and 3.11.0 differs again.
Transcribing from the wrong tree moves which contacts the solver sees on every
model that has a margin at all. The goldens below are the runtime's.

⚠ BOTH ARE SUMS OVER THE PAIR. `getMargin` returns
`geom_margin[g1] + geom_margin[g2]` and `getGap` the same for gaps
(engine_collision_driver.c:161/170) — not maxima. flybody's labrum pair, both
geoms at margin 5e-04, reports `includemargin` 1e-03, which is what pins it.

FOUR ARMS:

  1. the RECORD — eight geoms carry `gap`, and they carry it from a
     `<default class="adhesion-collision">` rather than from any element.

  2. the SPLIT, which is the whole feature. On a purpose-built model — a
     sphere held at dist 7.5e-04 over a plane, `margin` 5e-04 `gap` 5e-04 —
     the contact must be DETECTED (cutoff 1.0e-03) and must record an
     `includemargin` of 5.0e-04, i.e. `dist >= includemargin`. ⚠ ITS CONTROL
     IS THE SAME MODEL WITH `gap="0"`: there the cutoff is 5e-04, the contact
     is not detected at all, and `ncon` is 0. A build that passed the cutoff
     through as the includemargin would pass a count-only arm and fail this.

  3. the EXCLUSION IS REAL — 60 steps of that model against MuJoCo. The sphere
     falls THROUGH the band and settles below it; a build that solved the
     in-gap contact would have caught it at its starting height instead.

  4. flybody, where the feature was actually needed: six contacts where five
     were detected before, exactly one of them in the band.

  5. ⚠ A CONTROL MODEL WITH NO GAP, so the change cannot have been a blanket
     widening: `unitree_go1` must still record `includemargin` equal to its
     margin sum and exclude nothing.

Run: pixi run mojo run -I . tests/physics3d/test_geom_gap_vs_mujoco.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_auto
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr, STUDIO_DT
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE, GEOM_IDX_MARGIN, GEOM_IDX_GAP,
    CONTACT_SIZE, CONTACT_IDX_DIST, CONTACT_IDX_INCLUDEMARGIN,
    METADATA_SIZE, META_IDX_NUM_CONTACTS, KEY_IDX_NQPOS,
)


comptime DT = STUDIO_DT
comptime FLYBODY = String(
    "references/mujoco_menagerie-main/flybody/scene.xml"
)
comptime NOGAP_MODEL = String(
    "references/mujoco_menagerie-main/unitree_go1/scene.xml"
)


def _band_xml(gap: String) -> String:
    """A sphere parked at dist 7.5e-04 over a plane.

    ⚠ THE HEIGHT IS THE FIXTURE. `margin` is 5.0e-04 and `gap` 5.0e-04, so the
    narrowphase cutoff is 1.0e-03 and the includemargin 5.0e-04 — 7.5e-04 is
    the only interval that separates them. Any distance below 5.0e-04 would be
    solved either way and any above 1.0e-03 detected by neither.
    """
    return String(
        "<mujoco><option timestep='0.002' gravity='0 0 -9.81'/>"
        "<worldbody>"
        "<geom name='floor' type='plane' size='5 5 0.1'/>"
        "<body name='b' pos='0 0 0.10075'><freejoint/>"
        "<geom name='s' type='sphere' size='0.1' margin='0.0005' gap='"
        + gap + "'/>"
        "</body></worldbody></mujoco>"
    )


def _mj_band_qpos_60() -> List[Float64]:
    """MuJoCo `qpos` after 60 steps of `_band_xml("0.0005")`.

    z falls from 0.10075 to 0.100132 — THROUGH the band and to rest just
    inside the solved region. A build that solved the in-gap contact would
    have stopped it at 0.10075.
    """
    return [
        +4.77935402827917518e-19, -5.61117188579005940e-19,
        +1.00132407902851828e-01, +1.00000000000000000e+00,
        +3.96315138989570945e-18, +0.00000000000000000e+00,
        +0.00000000000000000e+00,
    ]


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def eq(mut self, got: Int, want: Int, msg: String):
        self.checks += 1
        if got == want:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want)

    def close(mut self, got: Float64, want: Float64, tol: Float64, msg: String):
        self.checks += 1
        if abs(got - want) <= tol:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want)

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def main() raises:
    var t = Tally()
    print("=== <geom gap> vs MuJoCo 3.10.0 ===")

    # ── 2. the split, on a purpose-built pair ────────────────────────────
    print("--- a sphere at dist 7.5e-04: margin 5e-04, gap 5e-04 ---")
    for k in range(2):
        var gap_s = String("0.0005") if k == 0 else String("0")
        var fmd = parse_xml_full(
            expand_mjcf(_band_xml(gap_s), String("")), String("")
        )
        var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var sf = spec_fields_runtime[DT](fmd, dims, m)
        var d = Data[DT, DynDims, 1](dims)
        for i in range(dims.get_nq()):
            d.qpos.data[i] = sf.qpos0.data[i]
        for i in range(dims.get_nv()):
            d.qvel.data[i] = Scalar[DT](0)
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        detect_contacts_auto["cpu", DT, BATCH=1](d, m, None)
        var ncon = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
        if k == 0:
            t.eq(ncon, 1, "gap=5e-04: the contact IS detected (cutoff 1e-03)")
            if ncon == 1:
                var cd = Float64(d.contacts.data[CONTACT_IDX_DIST])
                var ci = Float64(
                    d.contacts.data[CONTACT_IDX_INCLUDEMARGIN]
                )
                print("      dist =", cd, " includemargin =", ci)
                t.close(cd, 7.5e-04, 1e-12, "    its dist")
                # ⚠ 5.0e-04, NOT the 1.0e-03 cutoff. This is the arm that
                # separates the two values; passing the cutoff through here
                # would make `dist < includemargin` and the solver would take
                # the contact.
                t.close(ci, 5.0e-04, 1e-15,
                        "    its includemargin is the MARGIN, not the cutoff")
                t.truth(cd >= ci,
                        "    dist >= includemargin — the solver excludes it")
        else:
            # ⚠ THE CONTROL. With `gap="0"` the cutoff falls back to 5e-04 and
            # 7.5e-04 is simply out of range: nothing is detected. Without
            # this arm, arm 2 would pass on a build that widened the cutoff
            # unconditionally.
            t.eq(ncon, 0, "gap=0: the SAME pose detects nothing at all")

    # ── 3. the exclusion is real, over 60 steps ──────────────────────────
    print("--- 60 steps: the sphere falls THROUGH the band ---")
    var fmd_b = parse_xml_full(
        expand_mjcf(_band_xml(String("0.0005")), String("")), String("")
    )
    var dims_b = dims_from_flat(fmd_b, max_contacts=16, nmesh_verts=0)
    var m_b = Model[DT, DynDims](dims_b)
    build_model_runtime[DT](fmd_b, dims_b, m_b)
    var sf_b = spec_fields_runtime[DT](fmd_b, dims_b, m_b)
    var d_b = Data[DT, DynDims, 1](dims_b)
    for i in range(dims_b.get_nq()):
        d_b.qpos.data[i] = sf_b.qpos0.data[i]
    for i in range(dims_b.get_nv()):
        d_b.qvel.data[i] = Scalar[DT](0)
    var integ = StudioIntegPyr(dims_b)
    for _ in range(60):
        integ.step["cpu"](d_b, m_b)
    var want_b = _mj_band_qpos_60()
    var worst_b = 0.0
    for i in range(dims_b.get_nq()):
        var e = abs(Float64(d_b.qpos.data[i]) - want_b[i])
        if e > worst_b:
            worst_b = e
    print("    worst |d qpos| =", worst_b, "  z =",
          Float64(d_b.qpos.data[2]))
    t.truth(worst_b < 1e-12, "the 60-step trajectory matches MuJoCo")
    # ⚠ NON-VACUITY: it must have MOVED. A build that solved the in-gap
    # contact would have held it at its starting 0.10075.
    t.truth(Float64(d_b.qpos.data[2]) < 0.1005,
            "and it fell BELOW the band — an in-gap contact carried no force")

    # ── 1/4. flybody: the record and the contact set ─────────────────────
    print("--- flybody: eight geoms with a class-set gap ---")
    var src = read_model_source(FLYBODY)
    var ffmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var fdims = dims_from_flat(ffmd, max_contacts=128, nmesh_verts=65536)
    var fm = Model[DT, DynDims](fdims)
    build_model_runtime[DT](ffmd, fdims, fm)
    var fsf = spec_fields_runtime[DT](ffmd, fdims, fm)
    var n_gap = 0
    var n_margin = 0
    for g in range(fdims.get_ngeom()):
        var o = g * MODEL_GEOM_SIZE
        if Float64(fm.geoms.data[o + GEOM_IDX_GAP]) != 0.0:
            n_gap += 1
            t.close(Float64(fm.geoms.data[o + GEOM_IDX_GAP]), 5.0e-04,
                    0.0, "geom " + String(g) + " gap")
        if Float64(fm.geoms.data[o + GEOM_IDX_MARGIN]) != 0.0:
            n_margin += 1
    t.eq(n_gap, 8, "geoms carrying a gap (MuJoCo reports 8)")
    t.eq(n_margin, 8, "and the same eight carry a margin")

    var fd = Data[DT, DynDims, 1](fdims)
    var nqp = Int(Float64(fsf.key_meta.data[KEY_IDX_NQPOS]))
    for i in range(fdims.get_nq()):
        fd.qpos.data[i] = fsf.qpos0.data[i]
    for i in range(min(nqp, fdims.get_nq())):
        fd.qpos.data[i] = fsf.key_qpos.data[i]
    for i in range(fdims.get_nv()):
        fd.qvel.data[i] = Scalar[DT](0)
    forward_kinematics["cpu", DT, DynDims, 1](fd, fm)
    detect_contacts_auto["cpu", DT, BATCH=1](fd, fm, None)
    var fncon = Int(Float64(fd.meta.data[META_IDX_NUM_CONTACTS]))
    t.eq(fncon, 6, "flybody's contact count at its keyframe (was 5)")
    var in_band = 0
    for c in range(fncon):
        var co = c * CONTACT_SIZE
        if (
            Float64(fd.contacts.data[co + CONTACT_IDX_DIST])
            >= Float64(fd.contacts.data[co + CONTACT_IDX_INCLUDEMARGIN])
        ):
            in_band += 1
            print("      in-band contact", c, " dist",
                  Float64(fd.contacts.data[co + CONTACT_IDX_DIST]),
                  " includemargin",
                  Float64(fd.contacts.data[co + CONTACT_IDX_INCLUDEMARGIN]))
    t.eq(in_band, 1, "exactly one of the six is in the gap band")

    # ── 5. the control: a model with no gap is untouched ─────────────────
    print("--- unitree_go1, which declares no gap at all ---")
    var gsrc = read_model_source(NOGAP_MODEL)
    var gfmd = parse_xml_full(expand_mjcf(gsrc[0], gsrc[1]), gsrc[1])
    var gdims = dims_from_flat(gfmd, max_contacts=128, nmesh_verts=65536)
    var gm = Model[DT, DynDims](gdims)
    build_model_runtime[DT](gfmd, gdims, gm)
    var gsf = spec_fields_runtime[DT](gfmd, gdims, gm)
    var ggap = 0
    for g in range(gdims.get_ngeom()):
        if Float64(gm.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_GAP]) != 0.0:
            ggap += 1
    t.eq(ggap, 0, "go1 carries no gap")
    var gd = Data[DT, DynDims, 1](gdims)
    var gnqp = Int(Float64(gsf.key_meta.data[KEY_IDX_NQPOS]))
    for i in range(gdims.get_nq()):
        gd.qpos.data[i] = gsf.qpos0.data[i]
    for i in range(min(gnqp, gdims.get_nq())):
        gd.qpos.data[i] = gsf.key_qpos.data[i]
    for i in range(gdims.get_nv()):
        gd.qvel.data[i] = Scalar[DT](0)
    forward_kinematics["cpu", DT, DynDims, 1](gd, gm)
    detect_contacts_auto["cpu", DT, BATCH=1](gd, gm, None)
    var gncon = Int(Float64(gd.meta.data[META_IDX_NUM_CONTACTS]))
    t.eq(gncon, 4, "go1's contact count (MuJoCo reports 4)")
    var g_excluded = 0
    for c in range(gncon):
        var co = c * CONTACT_SIZE
        # MuJoCo reports includemargin 1.0e-03 on every one of these.
        t.close(
            Float64(gd.contacts.data[co + CONTACT_IDX_INCLUDEMARGIN]),
            1.0e-03, 1e-15, "go1 contact " + String(c) + " includemargin",
        )
        if (
            Float64(gd.contacts.data[co + CONTACT_IDX_DIST])
            >= Float64(gd.contacts.data[co + CONTACT_IDX_INCLUDEMARGIN])
        ):
            g_excluded += 1
    t.eq(g_excluded, 0, "and nothing on it is excluded")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_geom_gap_vs_mujoco: " + String(t.fails) + " failed"
        )
