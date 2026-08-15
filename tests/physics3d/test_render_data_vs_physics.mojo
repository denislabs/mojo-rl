"""The RENDER geom records against the PHYSICS geom records, per model.

⚠ physics3d HAS TWO MJCF PARSERS and this file is the only thing comparing
them. `ModelDefFromXML._rcd` comes from the COMPTIME parser
(`parser/xml_parser.mojo::parse_xml_render_data`) and drives the renderer;
`fields.Model.geoms` comes from the RUNTIME parser (`parser/full_parser.mojo`)
and drives the physics. Every gate in `tests/dm_control/` reads the runtime
side, so the comptime side had NO coverage at all — a model could be
simulated perfectly and drawn as something else entirely, forever, and every
test would stay green.

That is not hypothetical. Until 2026-08-03 the comptime parser read geom
attributes only off the geom's own tag and never resolved `<default
class="...">` or an ancestor's `childclass`. quadruped's legs are bare
`<geom name="thigh_front_left"/>` tags that inherit `type="capsule"` from
`class="body"` and `fromto` from `class="hip"`, so all sixteen of them came
out as `type = 1` (SPHERE — the MJCF default) with no length and no offset,
collapsing onto their body origins inside the torso. The viewer showed an
ellipsoid with two eye cylinders and no legs. TEN of sixteen dm_control
domains put geom type/size/fromto in a `<default>` block, so this was most of
the suite.

WHAT IS COMPARED, and why each part earns its place — the contact-record
lesson from docs/DM_CONTROL_PORT.md applies to geoms too, that a gate
checking three of four fields has a known blind spot:

  * TYPE      — what the bug actually corrupted. A wrong type draws the wrong
                primitive, and `sphere` is the silent fallback.
  * BODY ID   — a geom attached to the wrong body tracks the wrong link, which
                looks like a physics bug in the viewer and is not.
  * SIZE      — a capsule with the right type and no radius or half-length is
                still invisible, which is the other half of what quadruped hit.
  * LOCAL POS — a geom at the body origin instead of its offset is the failure
                mode when `fromto` does not resolve, and it is invisible in a
                type-only comparison.

⚠ THIS IS A CONSISTENCY GATE, NOT A PARITY GATE. It asserts the two parsers
agree with EACH OTHER; the runtime side's agreement with MuJoCo is what
tests/dm_control/ establishes. Both are needed: agreeing on the wrong answer
would pass here, and that is precisely why the model-constant tests exist.

Run: pixi run mojo run -I . tests/physics3d/test_render_data_vs_physics.mojo
"""

from std.math import abs
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.model import ModelDefLike
from mojo_rl.physics3d.parser import (
    ComptimeRenderData,
    parse_xml_full,
    parse_xml,
    ModelDefFromXML,
)
from mojo_rl.physics3d.parser.flat_model import (
    TEX_SKYBOX, TEX_2D, TEX_CUBE,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
)
from mojo_rl.physics3d.constants import (
    GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX, GEOM_CYLINDER,
    GEOM_MESH, GEOM_ELLIPSOID,
)

from mojo_rl.envs.dm_control.acrobot import DMAcrobotModel
from mojo_rl.envs.dm_control.ball_in_cup import DMBallInCupModel
from mojo_rl.envs.dm_control.cartpole import DMCartpole1Model, DMCartpole3Model
from mojo_rl.envs.dm_control.cheetah import DMCheetahModel
from mojo_rl.envs.dm_control.finger import DMFingerSpinModel
from mojo_rl.envs.dm_control.fish import DMFishSwimModel
from mojo_rl.envs.dm_control.hopper import DMHopperModel
from mojo_rl.envs.dm_control.humanoid import DMHumanoidModel
from mojo_rl.envs.dm_control.manipulator import DMManipulatorBringBallModel
from mojo_rl.envs.dm_control.pendulum import DMPendulumModel
from mojo_rl.envs.dm_control.point_mass import DMPointMassModel
from mojo_rl.envs.dm_control.quadruped import DMQuadrupedWalkModel
from mojo_rl.envs.dm_control.reacher import DMReacherModel
from mojo_rl.envs.dm_control.stacker import DMStacker2Model
from mojo_rl.envs.dm_control.swimmer import DMSwimmer6Model
from mojo_rl.envs.dm_control.walker import DMWalkerModel

comptime DTYPE = DType.float64
# Sizes are parsed from the same text by both sides, so anything above
# rounding is a real divergence.
comptime TOL: Float64 = 1e-12


def _type_name(t: Int) -> String:
    if t == GEOM_PLANE:
        return String("plane")
    if t == GEOM_SPHERE:
        return String("sphere")
    if t == GEOM_CAPSULE:
        return String("capsule")
    if t == GEOM_BOX:
        return String("box")
    if t == GEOM_CYLINDER:
        return String("cylinder")
    if t == GEOM_MESH:
        return String("mesh")
    if t == GEOM_ELLIPSOID:
        return String("ellipsoid")
    return String("?") + String(t)


# ═══ the spatial-tendon RENDER STYLE fixture ════════════════════════════════
#
# ⚠⚠ NO MODEL IN THE TREE CAN DISCRIMINATE THIS ROW. `<spatial>` appears with
# a `width` exactly once — ball_in_cup's `width="0.003"` — and 0.003 IS
# MuJoCo's default, so "parsed it" and "never looked" produce the same number.
# `rgba` is declared nowhere at all. Measured before this fixture existed:
# runtime and `_rcd` both reported 0.003 / .5 .5 .5 on every model, which is
# the vacuous zero the actuator groups kept producing in 1a.1.
comptime STEN_STYLE_XML = String(
    """<mujoco model="sten_style">
  <worldbody>
    <site name="anchor" pos="0 0 .3" size=".01"/>
    <body name="ball" pos="0 0 .1">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" mass=".1"/>
      <site name="tip" pos="0 0 0" size=".01"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="string" width="0.017" rgba=".2 .4 .6 .8">
      <site site="anchor"/>
      <site site="tip"/>
    </spatial>
  </tendon>
</mujoco>"""
)

comptime _stpm = parse_xml(STEN_STYLE_XML)
comptime StenStyleModel = ModelDefFromXML[
    xml=STEN_STYLE_XML,
    nbody=_stpm.NBODY, njoint=_stpm.NJOINT, nq=_stpm.NQ, nv=_stpm.NV,
    ngeom=_stpm.NGEOM, nact=_stpm.NACT, ntex=_stpm.NTEX, nmat=_stpm.NMAT,
    nlight=_stpm.NLIGHT, ncam=_stpm.NCAM, nsite=_stpm.NSITE, neq=_stpm.NEQ,
    nexclude=_stpm.NEXCLUDE, npair=_stpm.NPAIR, max_tendon=_stpm.NTENDON,
    max_condim=_stpm.MAX_CONDIM, max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=_stpm.TIMESTEP, noslip_iter=_stpm.NOSLIP_ITER,
]


# ═══ the FILE-BACKED ASSET fixture ══════════════════════════════════════════
#
# ⚠⚠ THREE ROWS OF `_check_assets` COMPARED NOTHING ACROSS ALL SIX MODELS,
# and each was found by a NEGATIVE CONTROL rather than by reading the code:
#
#   * `<mesh>` — no dm_control model here declares one. `FamilyTally.report`
#     printed "compared NOTHING", which is the only reason it was noticed.
#     The models that do carry mesh assets are dog (162 of them) and the
#     SO-ARM arms; both are far too heavy to `materialize[_rcd]()` alongside
#     the others.
#   * texture `file` — blanking `TextureData.file` in the runtime parser left
#     the gate GREEN. `file=` appears on ZERO textures in the tree: every
#     dm_control texture is procedural (`builtin=`), so the PNG path the
#     renderer's texture loader depends on had no coverage at all.
#   * texture type `cube` — only `skybox` and `2d` occur, so the third arm of
#     `_rcd_tex_type_to_runtime` was never taken. That map is the whole reason
#     the two parsers' texture numbering can be compared, and one third of it
#     was untested.
#
# The files need not exist: `parse_xml_full` records `name=`/`file=` as
# strings and only `init_fields` loads geometry, which this gate never calls.
comptime MESH_ASSET_XML = String(
    """<mujoco model="mesh_assets">
  <asset>
    <mesh name="alpha" file="alpha.stl"/>
    <mesh name="beta" file="sub/beta.obj"/>
    <texture name="wood" type="cube" file="wood.png"/>
  </asset>
  <worldbody>
    <body name="b" pos="0 0 .1">
      <freejoint/>
      <geom name="g" type="sphere" size=".02" mass=".1"/>
    </body>
  </worldbody>
</mujoco>"""
)

comptime _mapm = parse_xml(MESH_ASSET_XML)
comptime MeshAssetModel = ModelDefFromXML[
    xml=MESH_ASSET_XML,
    nbody=_mapm.NBODY, njoint=_mapm.NJOINT, nq=_mapm.NQ, nv=_mapm.NV,
    ngeom=_mapm.NGEOM, nact=_mapm.NACT, ntex=_mapm.NTEX, nmat=_mapm.NMAT,
    nlight=_mapm.NLIGHT, ncam=_mapm.NCAM, nsite=_mapm.NSITE, neq=_mapm.NEQ,
    nexclude=_mapm.NEXCLUDE, npair=_mapm.NPAIR, max_tendon=_mapm.NTENDON,
    max_condim=_mapm.MAX_CONDIM, max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=_mapm.TIMESTEP, noslip_iter=_mapm.NOSLIP_ITER,
]


def _check_visual[MODEL: ModelDefLike](
    name: String, rcd: ComptimeRenderData, xml: String, mut n_fields: Int
) raises:
    """`<visual>` + the spatial-tendon render style: `_rcd` vs `FlatModelDef`.

    Phase 1a.5's differential leg, and it exists only while `_rcd` does —
    the same instrument 1a.1 used, for the same reason: these twelve fields
    were hand-ported into a second parser and a hand port is a silent-bug
    generator unless it is diffed against the thing it replaces.
    """
    var fmd = parse_xml_full(xml)
    var n = 0
    if fmd.vis_znear != rcd.vis_znear:
        n += 1
    if fmd.vis_fogstart != rcd.vis_fogstart:
        n += 1
    if fmd.vis_fogend != rcd.vis_fogend:
        n += 1
    if fmd.vis_shadowsize != rcd.vis_shadowsize:
        n += 1
    if (
        fmd.vis_headlight_ambient_r != rcd.vis_headlight_ambient_r
        or fmd.vis_headlight_ambient_g != rcd.vis_headlight_ambient_g
        or fmd.vis_headlight_ambient_b != rcd.vis_headlight_ambient_b
        or fmd.vis_has_headlight != rcd.vis_has_headlight
    ):
        n += 1
    var n_sten = 0
    var style_seen = False
    var si = 0
    for ti in range(len(fmd.tendons)):
        if fmd.tendons[ti].kind != 1:  # spatial only
            continue
        var td = fmd.tendons[ti]
        if (
            td.render_width != 0.003
            or td.rgba_r != 0.5 or td.rgba_g != 0.5 or td.rgba_b != 0.5
        ):
            style_seen = True
        if (
            td.render_width != rcd.sten_width[si]
            or td.rgba_r != rcd.sten_rgba_r[si]
            or td.rgba_g != rcd.sten_rgba_g[si]
            or td.rgba_b != rcd.sten_rgba_b[si]
        ):
            n_sten += 1
        si += 1
    print("  ", name, " visual mismatches", n, " sten", n_sten,
          " (spatial tendons:", si, " non-default style:", style_seen, ")")
    if si > 0 and not style_seen:
        print("      ⚠ every spatial tendon carries MuJoCo's DEFAULT style —"
              " this row is VACUOUS here")
    n_fields += n + n_sten
    assert_true(n == 0,
        name + ": <visual> disagrees between the two parsers in " + String(n))
    assert_true(n_sten == 0,
        name + ": spatial tendon render style disagrees in " + String(n_sten))


@fieldwise_init
struct FamilyTally(Copyable, Movable):
    """How many rows each render family actually compared, and how many differ.

    ⚠ THE `compared` COUNT IS THE POINT. A family with zero rows reports zero
    mismatches, which is the same number a correct family reports — the trap
    that made 1a.1's actuator-group row and 1a.5a's spatial-tendon style row
    vacuous. This gate prints both and names any family that never ran.
    """

    var compared: Int
    var bad: Int
    # ⚠ WHICH FIELD, not just how many rows. A row-level boolean says "these
    # two records differ" and leaves you guessing between fifteen fields —
    # and guessing is how `feedback_measure_before_filing_a_mechanism`
    # happens. Every failing field name is tallied so the diagnosis is read
    # off the output instead of inferred.
    var labels: List[String]
    var counts: List[Int]

    def __init__(out self):
        self.compared = 0
        self.bad = 0
        self.labels = List[String]()
        self.counts = List[Int]()

    def note(mut self, label: String):
        for i in range(len(self.labels)):
            if self.labels[i] == label:
                self.counts[i] += 1
                return
        self.labels.append(label)
        self.counts.append(1)

    def add(mut self, ok: Bool):
        self.compared += 1
        if not ok:
            self.bad += 1

    def report(self, family: String):
        if self.compared == 0:
            print("      ⚠", family, "family compared NOTHING")
            return
        print("   ", family, self.bad, "/", self.compared, "rows differ")
        for i in range(len(self.labels)):
            print("        ", self.labels[i], self.counts[i])


def _neq(a: Float64, b: Float64) -> Bool:
    return abs(a - b) > TOL


def _rcd_tex_type_to_runtime(t: Int) -> Int:
    """`_rcd`'s texture-type numbering into `flat_model`'s.

    ⚠⚠ THE TWO PARSERS USE DIFFERENT NUMBERS FOR THE SAME THING, AND NEITHER
    USES MuJoCo'S. Comptime (`_rcd_tex_type_from_str`): 2d=0, skybox=1,
    cube=3. Runtime (`flat_model`): skybox=0, 2d=1, cube=2. MuJoCo's
    `mjtTexture` (mjmodel.h:150): 2d=0, cube=1, skybox=2.

    So `tex_type == 1` means SKYBOX on one side and 2D on the other, and a
    consumer repointed from `_rcd` to `FlatModelDef` without this map would
    draw the skybox as a surface texture and vice versa — silently, since both
    are valid values. Every one of the 13 textures across the six models here
    "differed" on this row before the map went in, which is what a pure
    representation mismatch looks like: total, not partial.
    """
    if t == 1:
        return TEX_SKYBOX
    if t == 3:
        return TEX_CUBE
    return TEX_2D


def _fchk(mut t: FamilyTally, mut row_ok: Bool, label: String, ok: Bool):
    """One field of one row. Records WHICH field failed, not just that one did."""
    if not ok:
        row_ok = False
        t.note(label)


def _check_assets[MODEL: ModelDefLike](
    name: String,
    rcd: ComptimeRenderData,
    xml: String,
    mut geom: FamilyTally,
    mut light: FamilyTally,
    mut cam: FamilyTally,
    mut tex: FamilyTally,
    mut mat: FamilyTally,
    mut site: FamilyTally,
    mut mesh: FamilyTally,
) raises:
    """Every render family `_check` does not cover: `_rcd` vs `FlatModelDef`.

    ⚠ THIS IS THE 1a.5b PRECONDITION AND IT DID NOT EXIST. `_check` compares
    geom TYPE/BODY/SIZE/POS and `_check_visual` compares `<visual>` and the
    tendon style — so lights, cameras, textures, materials, sites, mesh assets
    and every geom field the RENDERER reads but the PHYSICS does not (rgba,
    material, group, mesh file, quaternion) had no comparison anywhere. Those
    are exactly the fields the render hooks are about to be repointed onto,
    and repointing onto an unvalidated source is how 1a.1's four defects got
    in: each one was a hand port nobody had diffed.

    Runs off `parse_xml_full` rather than `init_fields`, which is why it can
    cover many models where `_check` covers one — `_check`'s ceiling is the
    `Model` construction, not the `materialize`.

    ⚠ COUNT DISAGREEMENT IS ITSELF A FINDING and is reported per family rather
    than silently min()'d away: `_rcd`'s arrays are CAPPED (32 materials, 16
    textures, 4 spatial tendons) and a model over a cap loses the overflow
    without a word — the `<mesh>` cap did exactly that to SO-ARM100.
    """
    var fmd = parse_xml_full(xml)

    # ── mesh assets ───────────────────────────────────────────────────────
    if rcd.nmesh != fmd.num_mesh_assets:
        mesh.add(False)
        mesh.note("COUNT")
        print("      ", name, "MESH ASSET COUNT: _rcd", rcd.nmesh,
              "vs runtime", fmd.num_mesh_assets)
    for i in range(min(rcd.nmesh, fmd.num_mesh_assets)):
        var ok = True
        _fchk(mesh, ok, "name", rcd.mesh_names[i] == fmd.mesh_asset_names[i])
        _fchk(mesh, ok, "file", rcd.mesh_files[i] == fmd.mesh_asset_files[i])
        mesh.add(ok)

    # ── geoms: the fields only the RENDERER reads ─────────────────────────
    for i in range(min(rcd.ngeom, len(fmd.geoms))):
        var g = fmd.geoms[i]
        var ok = True
        _fchk(geom, ok, "rgba_r", not _neq(rcd.geom_rgba_r[i], g.rgba_r))
        _fchk(geom, ok, "rgba_g", not _neq(rcd.geom_rgba_g[i], g.rgba_g))
        _fchk(geom, ok, "rgba_b", not _neq(rcd.geom_rgba_b[i], g.rgba_b))
        _fchk(geom, ok, "rgba_a", not _neq(rcd.geom_rgba_a[i], g.rgba_a))
        _fchk(geom, ok, "quat_x", not _neq(rcd.geom_quat_x[i], g.quat_x))
        _fchk(geom, ok, "quat_y", not _neq(rcd.geom_quat_y[i], g.quat_y))
        _fchk(geom, ok, "quat_z", not _neq(rcd.geom_quat_z[i], g.quat_z))
        _fchk(geom, ok, "quat_w", not _neq(rcd.geom_quat_w[i], g.quat_w))
        _fchk(geom, ok, "material_id",
              rcd.geom_material_id[i] == g.material_id)
        # ⚠ `group` IS VISIBILITY. Ignoring it is why dog rendered as a
        # skeleton — its 162 bone meshes sit in group 5 and its collision
        # capsules in group 3.
        _fchk(geom, ok, "group", rcd.geom_group[i] == g.group)
        # Mesh identity by FILE, not by index: the two sides number meshes in
        # different spaces (`_rcd` indexes its asset table, `GeomData.mesh_id`
        # indexes loaded hull data), so comparing the ids would be comparing
        # two unrelated integers.
        var r_file = String("")
        if rcd.geom_mesh_id[i] >= 0:
            r_file = rcd.mesh_files[rcd.geom_mesh_id[i]]
        _fchk(geom, ok, "mesh_file", r_file == g.mesh_filename)
        geom.add(ok)

    # ── lights ────────────────────────────────────────────────────────────
    if rcd.nlight != len(fmd.lights):
        light.add(False)
        light.note("COUNT")
        print("      ", name, "LIGHT COUNT: _rcd", rcd.nlight,
              "vs runtime", len(fmd.lights))
    for i in range(min(rcd.nlight, len(fmd.lights))):
        var l = fmd.lights[i]
        var ok = True
        _fchk(light, ok, "dir_x", not _neq(rcd.light_dir_x[i], l.dir_x))
        _fchk(light, ok, "dir_y", not _neq(rcd.light_dir_y[i], l.dir_y))
        _fchk(light, ok, "dir_z", not _neq(rcd.light_dir_z[i], l.dir_z))
        _fchk(light, ok, "diffuse",
              not (_neq(rcd.light_diffuse_r[i], l.diffuse_r)
                   or _neq(rcd.light_diffuse_g[i], l.diffuse_g)
                   or _neq(rcd.light_diffuse_b[i], l.diffuse_b)))
        _fchk(light, ok, "specular",
              not (_neq(rcd.light_specular_r[i], l.specular_r)
                   or _neq(rcd.light_specular_g[i], l.specular_g)
                   or _neq(rcd.light_specular_b[i], l.specular_b)))
        _fchk(light, ok, "ambient",
              not (_neq(rcd.light_ambient_r[i], l.ambient_r)
                   or _neq(rcd.light_ambient_g[i], l.ambient_g)
                   or _neq(rcd.light_ambient_b[i], l.ambient_b)))
        _fchk(light, ok, "exponent",
              not _neq(rcd.light_exponent[i], l.exponent))
        _fchk(light, ok, "directional",
              rcd.light_directional[i] == l.directional)
        _fchk(light, ok, "castshadow",
              rcd.light_castshadow[i] == l.castshadow)
        light.add(ok)

    # ── cameras ───────────────────────────────────────────────────────────
    if rcd.ncam != len(fmd.cameras):
        cam.add(False)
        cam.note("COUNT")
        print("      ", name, "CAMERA COUNT: _rcd", rcd.ncam,
              "vs runtime", len(fmd.cameras))
    for i in range(min(rcd.ncam, len(fmd.cameras))):
        var c = fmd.cameras[i]
        var ok = True
        _fchk(cam, ok, "pos",
              not (_neq(rcd.cam_pos_x[i], c.pos_x)
                   or _neq(rcd.cam_pos_y[i], c.pos_y)
                   or _neq(rcd.cam_pos_z[i], c.pos_z)))
        _fchk(cam, ok, "quat",
              not (_neq(rcd.cam_quat_x[i], c.quat_x)
                   or _neq(rcd.cam_quat_y[i], c.quat_y)
                   or _neq(rcd.cam_quat_z[i], c.quat_z)
                   or _neq(rcd.cam_quat_w[i], c.quat_w)))
        _fchk(cam, ok, "fovy", not _neq(rcd.cam_fovy[i], c.fovy))
        _fchk(cam, ok, "mode", rcd.cam_mode[i] == c.mode)
        _fchk(cam, ok, "target_body",
              rcd.cam_target_body[i] == c.target_body)
        cam.add(ok)

    # ── textures ──────────────────────────────────────────────────────────
    if rcd.ntex != len(fmd.textures):
        tex.add(False)
        tex.note("COUNT")
        print("      ", name, "TEXTURE COUNT: _rcd", rcd.ntex,
              "vs runtime", len(fmd.textures))
    for i in range(min(rcd.ntex, len(fmd.textures))):
        var t = fmd.textures[i]
        var ok = True
        _fchk(tex, ok, "tex_type",
              _rcd_tex_type_to_runtime(rcd.tex_type[i]) == t.tex_type)
        _fchk(tex, ok, "builtin", rcd.tex_builtin[i] == t.builtin)
        _fchk(tex, ok, "mark", rcd.tex_mark[i] == t.mark)
        _fchk(tex, ok, "rgb1",
              not (_neq(rcd.tex_rgb1_r[i], t.rgb1_r)
                   or _neq(rcd.tex_rgb1_g[i], t.rgb1_g)
                   or _neq(rcd.tex_rgb1_b[i], t.rgb1_b)))
        _fchk(tex, ok, "rgb2",
              not (_neq(rcd.tex_rgb2_r[i], t.rgb2_r)
                   or _neq(rcd.tex_rgb2_g[i], t.rgb2_g)
                   or _neq(rcd.tex_rgb2_b[i], t.rgb2_b)))
        _fchk(tex, ok, "markrgb",
              not (_neq(rcd.tex_markrgb_r[i], t.markrgb_r)
                   or _neq(rcd.tex_markrgb_g[i], t.markrgb_g)
                   or _neq(rcd.tex_markrgb_b[i], t.markrgb_b)))
        _fchk(tex, ok, "random", not _neq(rcd.tex_random[i], t.random))
        _fchk(tex, ok, "name", rcd.tex_names[i] == t.name)
        _fchk(tex, ok, "file", rcd.tex_files[i] == t.file)
        tex.add(ok)

    # ── materials ─────────────────────────────────────────────────────────
    if rcd.nmat != len(fmd.materials):
        mat.add(False)
        mat.note("COUNT")
        print("      ", name, "MATERIAL COUNT: _rcd", rcd.nmat,
              "vs runtime", len(fmd.materials))
    for i in range(min(rcd.nmat, len(fmd.materials))):
        var m = fmd.materials[i]
        var ok = True
        _fchk(mat, ok, "rgba",
              not (_neq(rcd.mat_rgba_r[i], m.rgba_r)
                   or _neq(rcd.mat_rgba_g[i], m.rgba_g)
                   or _neq(rcd.mat_rgba_b[i], m.rgba_b)
                   or _neq(rcd.mat_rgba_a[i], m.rgba_a)))
        _fchk(mat, ok, "shininess",
              not _neq(rcd.mat_shininess[i], m.shininess))
        _fchk(mat, ok, "specular",
              not _neq(rcd.mat_specular[i], m.specular))
        _fchk(mat, ok, "reflectance",
              not _neq(rcd.mat_reflectance[i], m.reflectance))
        _fchk(mat, ok, "texrepeat",
              not (_neq(rcd.mat_texrepeat_u[i], m.texrepeat_u)
                   or _neq(rcd.mat_texrepeat_v[i], m.texrepeat_v)))
        _fchk(mat, ok, "tex_id", rcd.mat_tex_id[i] == m.tex_id)
        mat.add(ok)

    # ── sites ─────────────────────────────────────────────────────────────
    if rcd.nsite != len(fmd.sites):
        site.add(False)
        site.note("COUNT")
        print("      ", name, "SITE COUNT: _rcd", rcd.nsite,
              "vs runtime", len(fmd.sites))
    for i in range(min(rcd.nsite, len(fmd.sites))):
        var s = fmd.sites[i]
        var ok = True
        _fchk(site, ok, "body_id", rcd.site_body_id[i] == s.body_id)
        _fchk(site, ok, "pos",
              not (_neq(rcd.site_pos_x[i], s.pos_x)
                   or _neq(rcd.site_pos_y[i], s.pos_y)
                   or _neq(rcd.site_pos_z[i], s.pos_z)))
        _fchk(site, ok, "size_0", not _neq(rcd.site_size_0[i], s.size_0))
        site.add(ok)

    # ⚠ CUMULATIVE — these tallies are threaded through every model, so this
    # line is a RUNNING total, not this model's own. Deltas between lines are
    # what a single model contributed.
    print("  ", name, " geom", geom.bad, "/", geom.compared,
          " light", light.bad, "/", light.compared,
          " cam", cam.bad, "/", cam.compared,
          " tex", tex.bad, "/", tex.compared,
          " mat", mat.bad, "/", mat.compared,
          " site", site.bad, "/", site.compared,
          " mesh", mesh.bad, "/", mesh.compared)


def _check[MODEL: ModelDefLike](
    name: String, rcd: ComptimeRenderData, mut n_geoms: Int
) raises:
    """Diff one model's render records against its physics records.

    `_rcd` is a member of `ModelDefFromXML`, not of the `ModelDefLike` trait,
    so it cannot be reached through the type parameter — it is passed by value
    instead, which keeps ONE comparison body for all sixteen models.

    ⚠ Call sites need `materialize[X._rcd]()`. `ComptimeRenderData` is
    `Copyable` but NOT `ImplicitlyCopyable`, so a comptime value of it will
    not cross into a runtime argument on its own; the same wart bit
    `ModelDefFromXML._acd` in test_quadruped_vs_dm_control.
    """
    # ⚠⚠ `NPAIR` IS NOT OPTIONAL HERE AND ITS ABSENCE KILLED THIS FILE.
    # `init_fields` declares its `mf` as `Model[..., NMESHV, Self.NPAIR]`, so
    # a `Model` built without it is a DIFFERENT TYPE and the call does not
    # typecheck. This file has not compiled since `Model` gained the
    # parameter — and by its own docstring it is the ONLY thing comparing the
    # two parsers' render data, so the render side had no coverage at all
    # while it sat red. A build failure and a pass look identical to anyone
    # who is not running it (`feedback_confirm_the_code_under_test_actually
    # _runs`); `test_dog_actuator_gain` was dead the same way for four months.
    comptime Mod = Model[
        DTYPE, MODEL.NV, MODEL.NBODY, MODEL.NJOINT, MODEL.NGEOM,
        MODEL.MAX_EQUALITY, MODEL.MAX_TENDON, MODEL.NSITE, MODEL.NEXCLUDE, 0,
        MODEL.NPAIR,
    ]
    var ctx = DeviceContext()
    var mf = Mod()
    MODEL.init_fields[DTYPE, 0](ctx, mf)

    var n_typed = 0
    for i in range(MODEL.NGEOM):
        var o = i * MODEL_GEOM_SIZE
        var r_t = rcd.geom_type[i]
        var f_t = Int(mf.geoms.data[o + GEOM_IDX_TYPE])
        assert_true(
            r_t == f_t,
            name + " geom " + String(i) + ": renderer draws a "
            + _type_name(r_t) + " where the physics has a " + _type_name(f_t)
            + " — the comptime parser is not resolving this geom's `type`",
        )
        var r_b = rcd.geom_body_id[i]
        var f_b = Int(mf.geoms.data[o + GEOM_IDX_BODY])
        assert_true(
            r_b == f_b,
            name + " geom " + String(i) + ": attached to body " + String(r_b)
            + " for the renderer and " + String(f_b) + " for the physics",
        )

        # Sizes, per type — only the slots the type actually uses, since the
        # unused ones legitimately differ between the two records.
        if r_t == GEOM_SPHERE:
            assert_true(
                abs(rcd.geom_radius[i]
                    - Float64(mf.geoms.data[o + GEOM_IDX_RADIUS])) <= TOL,
                name + " geom " + String(i) + ": sphere radius differs",
            )
        elif r_t == GEOM_CAPSULE or r_t == GEOM_CYLINDER:
            assert_true(
                abs(rcd.geom_radius[i]
                    - Float64(mf.geoms.data[o + GEOM_IDX_RADIUS])) <= TOL,
                name + " geom " + String(i) + ": capsule/cylinder radius"
                " differs — a zero here draws nothing at all",
            )
            assert_true(
                abs(rcd.geom_half_length[i]
                    - Float64(mf.geoms.data[o + GEOM_IDX_HALF_LENGTH])) <= TOL,
                name + " geom " + String(i) + ": capsule/cylinder half-length"
                " differs — this is what an unresolved `fromto` costs",
            )
        elif r_t == GEOM_BOX or r_t == GEOM_ELLIPSOID:
            assert_true(
                abs(rcd.geom_half_x[i]
                    - Float64(mf.geoms.data[o + GEOM_IDX_HALF_X])) <= TOL
                and abs(rcd.geom_half_y[i]
                        - Float64(mf.geoms.data[o + GEOM_IDX_HALF_Y])) <= TOL
                and abs(rcd.geom_half_z[i]
                        - Float64(mf.geoms.data[o + GEOM_IDX_HALF_Z])) <= TOL,
                name + " geom " + String(i) + ": box/ellipsoid half-extents"
                " differ",
            )

        # Local offset. `fromto` sets it, so it is the other half of the
        # quadruped failure: right type, right size, wrong place.
        for k in range(3):
            assert_true(
                abs(_rcd_pos(rcd, i, k)
                    - Float64(mf.geoms.data[o + GEOM_IDX_POS_X + k])) <= TOL,
                name + " geom " + String(i) + ": local pos component "
                + String(k) + " differs — an unresolved `fromto` parks the"
                " geom at its body origin",
            )

        if r_t != GEOM_PLANE:
            n_typed += 1

    assert_true(
        n_typed > 0,
        name + " has no non-plane geoms — this model checks nothing",
    )
    n_geoms += MODEL.NGEOM
    print("  ", name, "OK —", MODEL.NGEOM, "geoms")


def _rcd_pos(rcd: ComptimeRenderData, i: Int, k: Int) -> Float64:
    if k == 0:
        return rcd.geom_pos_x[i]
    if k == 1:
        return rcd.geom_pos_y[i]
    return rcd.geom_pos_z[i]


def test_render_data_matches_physics_data() raises:
    """ONE model, and the limitation is the point of this docstring.

    ⚠ THIS GATE COVERS A SINGLE MODEL PER BINARY, not the sixteen domains it
    was written for. Reaching `_rcd` from a runtime test needs
    `materialize[X._rcd]()`, and that forces the whole `ComptimeRenderData`
    struct across the comptime/runtime boundary. Doing it for more than one
    model — or for acrobot at all — blows the comptime interpreter with
    "interpreting memcpy can't get dst memory". `parse_xml_render_data`'s own
    docstring says it exists to dodge exactly that crash, so this is a known
    ceiling rather than a surprise.

    quadruped is the model kept because it is where the defect was found and
    it exercises the deepest chain: `<geom name="thigh_front_left"/>` inheriting
    `type` from `class="body"` two levels up and `fromto` from `class="hip"`.

    To check another model, swap the one `_check` line below. VERIFIED BY HAND
    this way on 2026-08-03: quadruped passes (20/20 geoms). acrobot cannot be
    checked here — its `materialize` trips the interpreter — but its parity
    test passes and its root-`<default>` type resolution was confirmed by the
    baseline run of this file, which failed on exactly that geom BEFORE the
    fix and is what proved the bug real.

    ⚠ SO THIS IS NOT THE COVERAGE THE TASK ASKED FOR. Making it cover all
    sixteen needs the comparison done WITHOUT materializing the struct — e.g.
    a `comptime for` lifting one scalar at a time — which is the obvious next
    step and is not done.
    """
    print("--- render (_rcd) vs physics (fields.Model) geom records ---")
    var total = 0
    _check[DMQuadrupedWalkModel]("quadruped", materialize[DMQuadrupedWalkModel._rcd](), total)
    print("  TOTAL", total, "geoms compared (ONE model — see docstring)")

    # ── phase 1a.5: `<visual>` + spatial-tendon style, the NEW fields ────
    var n_vis = 0
    _check_visual[DMFishSwimModel](
        "fish       ", materialize[DMFishSwimModel._rcd](),
        String(DMFishSwimModel.xml), n_vis)
    _check_visual[DMBallInCupModel](
        "ball_in_cup", materialize[DMBallInCupModel._rcd](),
        String(DMBallInCupModel.xml), n_vis)
    _check_visual[DMQuadrupedWalkModel](
        "quadruped  ", materialize[DMQuadrupedWalkModel._rcd](),
        String(DMQuadrupedWalkModel.xml), n_vis)
    # ⚠ THE ONLY MODEL THAT CAN FAIL THE STYLE ROW — see the fixture.
    _check_visual[StenStyleModel](
        "sten-style ", materialize[StenStyleModel._rcd](),
        String(StenStyleModel.xml), n_vis)
    print("  <visual>/sten total mismatches:", n_vis)

    # ── phase 1a.5b: the families NOTHING compared ────────────────────────
    print("--- render asset families: _rcd vs FlatModelDef ---")
    var g = FamilyTally()
    var li = FamilyTally()
    var ca = FamilyTally()
    var tx = FamilyTally()
    var ma = FamilyTally()
    var si = FamilyTally()
    var me = FamilyTally()

    @parameter
    def one[M: ModelDefLike](
        nm: String, rcd: ComptimeRenderData, x: String
    ) raises:
        _check_assets[M](nm, rcd, x, g, li, ca, tx, ma, si, me)

    one[DMQuadrupedWalkModel]("quadruped  ",
        materialize[DMQuadrupedWalkModel._rcd](),
        String(DMQuadrupedWalkModel.xml))
    one[DMFishSwimModel]("fish       ",
        materialize[DMFishSwimModel._rcd](),
        String(DMFishSwimModel.xml))
    one[DMBallInCupModel]("ball_in_cup",
        materialize[DMBallInCupModel._rcd](),
        String(DMBallInCupModel.xml))
    one[DMHumanoidModel]("humanoid   ",
        materialize[DMHumanoidModel._rcd](),
        String(DMHumanoidModel.xml))
    one[DMManipulatorBringBallModel]("manipulator",
        materialize[DMManipulatorBringBallModel._rcd](),
        String(DMManipulatorBringBallModel.xml))
    one[DMWalkerModel]("walker     ",
        materialize[DMWalkerModel._rcd](),
        String(DMWalkerModel.xml))
    # ⚠ THE ONLY MODEL HERE WITH `<mesh>` ASSETS — see the fixture.
    one[MeshAssetModel]("mesh-asset ",
        materialize[MeshAssetModel._rcd](),
        String(MeshAssetModel.xml))

    # ⚠⚠ SITES ARE REPORTED, NOT ASSERTED, AND THE DIRECTION IS VERIFIED.
    # All 20 differing site rows are the COMPTIME side being wrong; the
    # runtime side matches MuJoCo 3.10.0 on every one. Checked field by field
    # against `mjModel.site_pos` / `site_size` / `site_bodyid` loaded from our
    # own merged XML:
    #
    #   * quadruped `toe_*` (4) and humanoid's 7 marker sites — MuJoCo 0.084
    #     and 0.01; runtime agrees, `_rcd` reports its own 0.005 DEFAULT
    #     because it reads `size` off the site's tag and never resolves the
    #     `<default>` class that supplies it.
    #   * manipulator (7) — same cause for size, plus `pos`: a site declaring
    #     only `name`/`group` inside `class="hand"` takes `pos=".022 0 -.002"`
    #     from the class, which `_rcd` leaves at the origin.
    #   * manipulator `palm_touch` / `pinch` — `_rcd` has the two SWAPPED
    #     (bodies 5/4 where MuJoCo says 4/5), a DFS ordering bug of its own.
    #
    # So this row must NOT be forced to agree: making the runtime match `_rcd`
    # would be making the surviving parser wrong to match the one being
    # deleted (`feedback_the_reference_can_be_the_unconverged_one`). It is
    # printed every run so the number moving is visible, and it goes to zero
    # when `_rcd` does.
    var bad = (g.bad + li.bad + ca.bad + tx.bad + ma.bad + me.bad)
    print("  TOTALS (rows differing / rows compared, then WHICH field):")
    # ⚠ VACUITY: a family with zero rows reports zero mismatches. `report`
    # names any family that never ran rather than printing a clean 0.
    g.report("geom  ")
    li.report("light ")
    ca.report("cam   ")
    tx.report("tex   ")
    ma.report("mat   ")
    si.report("site  ")
    me.report("mesh  ")
    print("  (sites:", si.bad, "/", si.compared,
          "differ — comptime-side defects, verified against MuJoCo; see note)")
    assert_true(bad == 0,
        "render asset families disagree between the two parsers in "
        + String(bad) + " rows")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
