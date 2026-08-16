"""`RenderFields` against MuJoCo — the render records the viewer draws from.

⚠ THIS FILE REPLACES A CONSISTENCY GATE WITH A PARITY GATE, and that is the
whole point of it. Until phase 1a.5c, `test_render_data_vs_physics.mojo`
compared our two MJCF parsers against EACH OTHER; when `_rcd` was deleted
that gate lost its oracle, and deleting it outright would have left the render
path with no coverage at all — the exact state 1a.5a found it in, dead and
green-looking, over sixteen models.

Two parsers agreeing proves nothing about either. The consistency gate could
not see a default both sides got wrong, and it demonstrably did not: MuJoCo's
`<texture type>` default is `cube` and BOTH our parsers returned `2d`, so
every texture without an explicit type was mistyped and both sides agreed
perfectly about it. This gate found that on its first run.
See `feedback_a_gate_that_shares_its_reference_implementation_is_blind`.

WHAT IS COMPARED, and what is NOT:

  * GEOMS      — type, body, pos, quat, the size slots the type uses, rgba,
                 material, group, mesh. The renderer reads every one.
  * LIGHTS     — dir, diffuse, specular, ambient, directional, castshadow,
                 exponent.
  * CAMERAS    — pos, quat, fovy, mode, targetbody.
  * MATERIALS  — rgba, shininess, specular, reflectance, texrepeat, texid.
  * SITES      — body, pos, size[0]. This family is why the gate matters:
                 `_rcd`'s site records were WRONG and the consistency gate
                 had to exempt them, which meant reporting rather than
                 asserting. Here they are asserted.
  * TENDONS    — spatial `width` / `rgba`, the render style.
  * <visual>   — znear, shadowsize, headlight ambient.

  ⚠ TEXTURE PARAMETERS CANNOT BE GATED HERE and this is a real hole, not an
  omission. MuJoCo does not RETAIN `builtin`, `rgb1`, `rgb2`, `mark`,
  `markrgb` or `random`: `mjCTexture::Compile` consumes them to generate
  `tex_data` and only the generated pixels survive into `mjModel`
  (`hasattr(m, 'tex_builtin')` is False). Only `tex_type` is retained and
  only that is checked. The procedural skybox and checker colours therefore
  have NO parity coverage anywhere — they had none before either, since the
  consistency gate compared them against a second copy of our own reader.

Run: pixi run mojo run -I . tests/physics3d/test_render_fields_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml_full


def _read(path: String) raises -> String:
    """The model's MJCF, from the asset — no embedded copy exists any more."""
    with open(path, "r") as f:
        return f.read()
from mojo_rl.physics3d.parser.render_fields import (
    RenderFields,
    build_render_fields,
)
from mojo_rl.physics3d.parser.flat_model import (
    TEX_SKYBOX, TEX_2D, TEX_CUBE,
)
from mojo_rl.physics3d.constants import (
    GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX, GEOM_CYLINDER,
    GEOM_MESH, GEOM_ELLIPSOID,
)

from mojo_rl.envs.dm_control.ball_in_cup import DMBallInCupModel
from mojo_rl.envs.dm_control.cheetah import DMCheetahModel
from mojo_rl.envs.dm_control.fish import DMFishSwimModel
from mojo_rl.envs.dm_control.humanoid import DMHumanoidModel
from mojo_rl.envs.dm_control.manipulator import DMManipulatorBringBallModel
from mojo_rl.envs.dm_control.quadruped import DMQuadrupedWalkModel
from mojo_rl.envs.dm_control.walker import DMWalkerModel

# Both sides parse the same text, so anything above rounding is real. The
# quaternion path is the one place arithmetic happens (xyaxes/euler/zaxis →
# quat), and it agrees to ~1e-16.
comptime TOL: Float64 = 1e-12

# ⚠⚠ MuJoCo STORES COLOURS AS float32 AND GEOMETRY AS float64, AND A SINGLE
# TOLERANCE CANNOT SPAN BOTH. `geom_rgba`, `mat_rgba`, `light_diffuse`,
# `light_specular`, `mat_reflectance`, `vis.map.znear` and
# `vis.headlight.ambient` are `float`; `geom_pos`, `geom_quat`, `geom_size`
# and `site_pos` are `mjtNum` (double). `float32(0.7)` is 0.699999988079071,
# which is 1.2e-8 away from our parsed 0.7 — so at 1e-12 every colour row in
# the model failed and every geometry row passed, which is what a tolerance
# below the float32 noise floor looks like rather than a defect.
# See `feedback_a_tolerance_below_the_float32_noise_floor`.
comptime TOL_F32: Float64 = 1e-6



# ═══ FIXTURES CARRIED OVER FROM THE CONSISTENCY GATE ════════════════════════
#
# ⚠⚠ WITHOUT THESE, THREE ROWS OF THIS GATE COMPARE NOTHING OR NOTHING THAT
# CAN FAIL. Each was found by a negative control against the gate this file
# replaces, and deleting that file without carrying them would have quietly
# restored the blind spots:
#
#   * spatial tendon STYLE — `<spatial>` carries a `width` exactly once in
#     the whole tree (ball_in_cup's `0.003`) and that IS MuJoCo's default, so
#     "parsed it" and "never looked" produce the same number. `rgba` is
#     declared nowhere at all.
#   * `<mesh>` ASSETS — no dm_control model declares one, so the mesh row had
#     no rows.
#   * texture `file=` and `type="cube"` — every dm_control texture is
#     procedural and typed `2d` or `skybox`, so the PNG path and the third
#     arm of the type map were never taken.
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

# ⚠ The mesh FILES must exist here, unlike in the consistency gate: MuJoCo
# compiles the asset and fails on a missing file, where our parser only
# records the string. `references/` ships STLs, so one of those is used.
comptime MESH_ASSET_XML = String(
    """<mujoco model="mesh_assets">
  <compiler meshdir="references/mujoco-3.6.0/model/humanoid"/>
  <asset>
    <texture name="wood" type="cube" builtin="flat" rgb1=".7 .5 .3"
             width="8" height="8"/>
    <material name="woodmat" texture="wood"/>
  </asset>
  <worldbody>
    <body name="b" pos="0 0 .1">
      <freejoint/>
      <geom name="g" type="sphere" size=".02" mass=".1" material="woodmat"/>
    </body>
  </worldbody>
</mujoco>"""
)

@fieldwise_init
struct Tally(Copyable, Movable):
    """Rows compared and rows differing, per family, with the failing FIELD.

    ⚠ `compared` IS LOAD-BEARING. A family with zero rows reports zero
    mismatches, which is the number a correct family reports. `report` names
    any family that never ran rather than printing a clean 0 — three rows of
    the gate this replaces turned out to be vacuous and only a negative
    control found them.
    """

    var compared: Int
    var bad: Int
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
            print("      ⚠", family, "compared NOTHING")
            return
        print("   ", family, self.bad, "/", self.compared, "rows differ")
        for i in range(len(self.labels)):
            print("        ", self.labels[i], self.counts[i])


def _f(mut t: Tally, mut ok: Bool, label: String, good: Bool):
    if not good:
        ok = False
        t.note(label)


def _neq(a: Float64, b: Float64) -> Bool:
    return abs(a - b) > TOL


def _neq32(a: Float64, b: Float64) -> Bool:
    """For a field MuJoCo stores as `float`. See `TOL_F32`."""
    return abs(a - b) > TOL_F32


def _geom_type_from_mujoco(t: Int) raises -> Int:
    """`mjtGeom` → `physics3d.constants`. ⚠ A FOURTH NUMBERING.

    MuJoCo (mjmodel.h): plane 0, hfield 1, sphere 2, capsule 3, ellipsoid 4,
    cylinder 5, box 6, mesh 7.
    Ours: plane 0, sphere 1, capsule 2, box 3, cylinder 4, mesh 5,
    ellipsoid 6.

    Only `plane` shares a number, so a missing map would fail loudly rather
    than silently — unlike `tex_type`, where comptime 2d=0/skybox=1/cube=3,
    runtime skybox=0/2d=1/cube=2 and MuJoCo 2d=0/cube=1/skybox=2 all overlap
    on values that MEAN different things.
    """
    if t == 0:
        return GEOM_PLANE
    if t == 2:
        return GEOM_SPHERE
    if t == 3:
        return GEOM_CAPSULE
    if t == 4:
        return GEOM_ELLIPSOID
    if t == 5:
        return GEOM_CYLINDER
    if t == 6:
        return GEOM_BOX
    if t == 7:
        return GEOM_MESH
    raise Error("unmapped mjtGeom " + String(t) + " (hfield?)")


def _tex_type_from_mujoco(t: Int) -> Int:
    """`mjtTexture` (2d=0, cube=1, skybox=2) → `flat_model`'s numbering."""
    if t == 0:
        return TEX_2D
    if t == 1:
        return TEX_CUBE
    return TEX_SKYBOX


def _q(o: PythonObject) raises -> Float64:
    return Float64(py=o)


def _i(o: PythonObject) raises -> Int:
    return Int(py=o)



def _qmul(
    aw: Float64, ax: Float64, ay: Float64, az: Float64,
    bw: Float64, bx: Float64, by: Float64, bz: Float64,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Hamilton product, (w, x, y, z) in and out."""
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qrot(
    w: Float64, x: Float64, y: Float64, z: Float64,
    vx: Float64, vy: Float64, vz: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """Rotate (vx, vy, vz) by the quaternion (w, x, y, z)."""
    var t = _qmul(w, x, y, z, 0.0, vx, vy, vz)
    var c = _qmul(t[0], t[1], t[2], t[3], w, -x, -y, -z)
    return (c[1], c[2], c[3])


def _check(
    name: String,
    path: String,
    xml: String,
    mut geom: Tally,
    mut light: Tally,
    mut cam: Tally,
    mut mat: Tally,
    mut site: Tally,
    mut tex: Tally,
    mut sten: Tally,
    mut vis: Tally,
) raises:
    """One model: `RenderFields` vs `mjModel`, family by family.

    ⚠ `path` IS THE MODEL FILE, and both sides need it. Asset paths inside a
    model are relative to that file (§10.5 decision 1), so MuJoCo must load by
    path and `parse_xml_full` must be told the same base directory — otherwise
    SO-ARM100's meshes resolve against the CWD and the mesh row, the only one
    with mesh geoms at all, dies with "Error opening file so_arm100/...".
    An inline FIXTURE passes "" and keeps the CWD behaviour it was written for.
    """
    var mujoco = Python.import_module("mujoco")
    var m: PythonObject
    var base = String("")
    if path.byte_length() > 0:
        m = mujoco.MjModel.from_xml_path(path)
        var cut = path.rfind("/")
        if cut > 0:
            base = String(path[byte=0:cut])
    else:
        m = mujoco.MjModel.from_xml_string(xml)
    var rf = build_render_fields(parse_xml_full(xml, base))

    # ── geoms ─────────────────────────────────────────────────────────────
    var ngeom = _i(m.ngeom)
    if ngeom != len(rf.geom_type):
        geom.add(False)
        geom.note("COUNT " + String(ngeom) + "!=" + String(len(rf.geom_type)))
    for i in range(min(ngeom, len(rf.geom_type))):
        var ok = True
        var mt = _geom_type_from_mujoco(_i(m.geom_type[i]))
        _f(geom, ok, "type", mt == rf.geom_type[i])
        _f(geom, ok, "body", _i(m.geom_bodyid[i]) == rf.geom_body_id[i])
        # ⚠⚠ FOR A MESH GEOM, MuJoCo'S FRAME IS NOT THE XML'S. `mjCMesh`
        # recenters and reorients each mesh asset to its inertial frame and
        # compensates by baking the transform into the geom:
        #     geom_pos  = xml_pos  + R(xml_quat) · mesh_pos
        #     geom_quat = xml_quat ⊗ mesh_quat
        # (verified directly: SO-ARM100's mesh geoms carry no `pos`/`quat`,
        # and MuJoCo reports geom_quat == mesh_quat and geom_pos == mesh_pos
        # for exactly those.) Our record keeps the XML values and our loader
        # keeps the raw STL vertices, so the WORLD placement is identical —
        # only the split between asset and geom differs.
        #
        # The bake is UNDONE here rather than the rows exempted. Skipping 23
        # of SO-ARM100's 33 geoms would have left mesh placement untested on
        # the only model in this gate that has any.
        var mpx = _q(m.geom_pos[i][0])
        var mpy = _q(m.geom_pos[i][1])
        var mpz = _q(m.geom_pos[i][2])
        var mqw = _q(m.geom_quat[i][0])
        var mqx = _q(m.geom_quat[i][1])
        var mqy = _q(m.geom_quat[i][2])
        var mqz = _q(m.geom_quat[i][3])
        var did = _i(m.geom_dataid[i])
        if mt == GEOM_MESH and did >= 0:
            var kw = _q(m.mesh_quat[did][0])
            var kx = _q(m.mesh_quat[did][1])
            var ky = _q(m.mesh_quat[did][2])
            var kz = _q(m.mesh_quat[did][3])
            # xml_quat = geom_quat ⊗ conj(mesh_quat)
            var xq = _qmul(mqw, mqx, mqy, mqz, kw, -kx, -ky, -kz)
            mqw = xq[0]
            mqx = xq[1]
            mqy = xq[2]
            mqz = xq[3]
            # xml_pos = geom_pos − R(xml_quat) · mesh_pos
            var off = _qrot(mqw, mqx, mqy, mqz,
                            _q(m.mesh_pos[did][0]),
                            _q(m.mesh_pos[did][1]),
                            _q(m.mesh_pos[did][2]))
            mpx -= off[0]
            mpy -= off[1]
            mpz -= off[2]
        _f(geom, ok, "pos",
           not (_neq(mpx, rf.geom_pos_x[i])
                or _neq(mpy, rf.geom_pos_y[i])
                or _neq(mpz, rf.geom_pos_z[i])))
        # ⚠ MuJoCo stores quaternions (w, x, y, z); ours are (x, y, z, w).
        # A quaternion and its negation are the SAME rotation, so compare up
        # to sign — the parsers pick a hemisphere independently and neither
        # choice is wrong.
        var qw = mqw
        var qx = mqx
        var qy = mqy
        var qz = mqz
        var same = (
            not _neq(qx, rf.geom_quat_x[i]) and not _neq(qy, rf.geom_quat_y[i])
            and not _neq(qz, rf.geom_quat_z[i])
            and not _neq(qw, rf.geom_quat_w[i])
        )
        var flipped = (
            not _neq(-qx, rf.geom_quat_x[i])
            and not _neq(-qy, rf.geom_quat_y[i])
            and not _neq(-qz, rf.geom_quat_z[i])
            and not _neq(-qw, rf.geom_quat_w[i])
        )
        _f(geom, ok, "quat", same or flipped)

        # Sizes, per type — only the slots the type uses. The others hold
        # `RenderFields`' 0.0 by construction and MuJoCo's own padding
        # otherwise, and comparing them would compare two kinds of nothing.
        var s0 = _q(m.geom_size[i][0])
        var s1 = _q(m.geom_size[i][1])
        var s2 = _q(m.geom_size[i][2])
        if mt == GEOM_SPHERE:
            _f(geom, ok, "sphere.radius", not _neq(s0, rf.geom_radius[i]))
        elif mt == GEOM_CAPSULE or mt == GEOM_CYLINDER:
            _f(geom, ok, "cap.radius", not _neq(s0, rf.geom_radius[i]))
            _f(geom, ok, "cap.half_length",
               not _neq(s1, rf.geom_half_length[i]))
        elif mt == GEOM_BOX or mt == GEOM_ELLIPSOID:
            _f(geom, ok, "box.half",
               not (_neq(s0, rf.geom_half_x[i])
                    or _neq(s1, rf.geom_half_y[i])
                    or _neq(s2, rf.geom_half_z[i])))
        elif mt == GEOM_PLANE:
            _f(geom, ok, "plane.half",
               not (_neq(s0, rf.geom_half_x[i])
                    or _neq(s1, rf.geom_half_y[i])))

        # ⚠ MuJoCo DOES NOT RESOLVE THE MATERIAL COLOUR INTO `geom_rgba` —
        # it keeps both and picks at draw time, applying the material's
        # colour unless the geom's own rgba differs from the "0.5 0.5 0.5 1"
        # default (XMLreference.rst:2623). We resolve at parse time, so the
        # expected value is MuJoCo's rule applied to MuJoCo's own two
        # records. Transcribed from the documentation, not from our
        # implementation — sharing the implementation is what makes a gate
        # blind.
        var gr = _q(m.geom_rgba[i][0])
        var gg = _q(m.geom_rgba[i][1])
        var gb = _q(m.geom_rgba[i][2])
        var ga = _q(m.geom_rgba[i][3])
        var matid = _i(m.geom_matid[i])
        var is_default = (
            not _neq32(gr, 0.5) and not _neq32(gg, 0.5)
            and not _neq32(gb, 0.5) and not _neq32(ga, 1.0)
        )
        if matid >= 0 and is_default:
            gr = _q(m.mat_rgba[matid][0])
            gg = _q(m.mat_rgba[matid][1])
            gb = _q(m.mat_rgba[matid][2])
            ga = _q(m.mat_rgba[matid][3])
        _f(geom, ok, "rgba",
           not (_neq32(gr, rf.geom_rgba_r[i])
                or _neq32(gg, rf.geom_rgba_g[i])
                or _neq32(gb, rf.geom_rgba_b[i])
                or _neq32(ga, rf.geom_rgba_a[i])))
        _f(geom, ok, "material", _i(m.geom_matid[i]) == rf.geom_material_id[i])
        # Mesh identity. MuJoCo's `geom_dataid` indexes its mesh array and
        # ours indexes `mesh_asset_files`; both are the `<asset>` declaration
        # order, so the integers line up. -1 on both sides means "not a mesh".
        _f(geom, ok, "mesh", _i(m.geom_dataid[i]) == rf.geom_mesh_id[i])
        # ⚠ VISIBILITY. Ignoring `group` is why dog rendered as a skeleton.
        _f(geom, ok, "group", _i(m.geom_group[i]) == rf.geom_group[i])
        geom.add(ok)

    # ── lights ────────────────────────────────────────────────────────────
    var nlight = _i(m.nlight)
    if nlight != len(rf.light_dir_x):
        light.add(False)
        light.note("COUNT " + String(nlight) + "!="
                   + String(len(rf.light_dir_x)))
    for i in range(min(nlight, len(rf.light_dir_x))):
        var ok = True
        _f(light, ok, "dir",
           not (_neq(_q(m.light_dir[i][0]), rf.light_dir_x[i])
                or _neq(_q(m.light_dir[i][1]), rf.light_dir_y[i])
                or _neq(_q(m.light_dir[i][2]), rf.light_dir_z[i])))
        _f(light, ok, "diffuse",
           not (_neq32(_q(m.light_diffuse[i][0]), rf.light_diffuse_r[i])
                or _neq32(_q(m.light_diffuse[i][1]), rf.light_diffuse_g[i])
                or _neq32(_q(m.light_diffuse[i][2]), rf.light_diffuse_b[i])))
        _f(light, ok, "specular",
           not (_neq32(_q(m.light_specular[i][0]), rf.light_specular_r[i])
                or _neq32(_q(m.light_specular[i][1]), rf.light_specular_g[i])
                or _neq32(_q(m.light_specular[i][2]), rf.light_specular_b[i])))
        _f(light, ok, "ambient",
           not (_neq32(_q(m.light_ambient[i][0]), rf.light_ambient_r[i])
                or _neq32(_q(m.light_ambient[i][1]), rf.light_ambient_g[i])
                or _neq32(_q(m.light_ambient[i][2]), rf.light_ambient_b[i])))
        _f(light, ok, "exponent",
           not _neq32(_q(m.light_exponent[i]), rf.light_exponent[i]))
        # ⚠ MuJoCo 3.10 REPLACED `light_directional` WITH `light_type`.
        # `mjtLightType` is spot=0, directional=1, point=2, image=3
        # (mjmodel.h:142); the boolean we model is "type == directional".
        # The reference trees still declare the old bool, so this is a case
        # where the RUNTIME disagrees with every checked-out header
        # (`feedback_reference_tree_version_drift`).
        _f(light, ok, "directional",
           (_i(m.light_type[i]) == 1) == rf.light_directional[i])
        _f(light, ok, "castshadow",
           (_i(m.light_castshadow[i]) != 0) == rf.light_castshadow[i])
        light.add(ok)

    # ── cameras ───────────────────────────────────────────────────────────
    var ncam = _i(m.ncam)
    if ncam != len(rf.cam_fovy):
        cam.add(False)
        cam.note("COUNT " + String(ncam) + "!=" + String(len(rf.cam_fovy)))
    for i in range(min(ncam, len(rf.cam_fovy))):
        var ok = True
        _f(cam, ok, "pos",
           not (_neq(_q(m.cam_pos[i][0]), rf.cam_pos_x[i])
                or _neq(_q(m.cam_pos[i][1]), rf.cam_pos_y[i])
                or _neq(_q(m.cam_pos[i][2]), rf.cam_pos_z[i])))
        var cqw = _q(m.cam_quat[i][0])
        var cqx = _q(m.cam_quat[i][1])
        var cqy = _q(m.cam_quat[i][2])
        var cqz = _q(m.cam_quat[i][3])
        var csame = (
            not _neq(cqx, rf.cam_quat_x[i]) and not _neq(cqy, rf.cam_quat_y[i])
            and not _neq(cqz, rf.cam_quat_z[i])
            and not _neq(cqw, rf.cam_quat_w[i])
        )
        var cflip = (
            not _neq(-cqx, rf.cam_quat_x[i])
            and not _neq(-cqy, rf.cam_quat_y[i])
            and not _neq(-cqz, rf.cam_quat_z[i])
            and not _neq(-cqw, rf.cam_quat_w[i])
        )
        # ⚠ THE ROW THAT CAUGHT THE CONJUGATE. `_xyaxes_to_quat` returned the
        # INVERSE rotation, which is NOT the sign flip allowed above — a
        # conjugate negates the vector part only, and that is a different
        # rotation.
        _f(cam, ok, "quat", csame or cflip)
        _f(cam, ok, "fovy", not _neq(_q(m.cam_fovy[i]), rf.cam_fovy[i]))
        _f(cam, ok, "mode", _i(m.cam_mode[i]) == rf.cam_mode[i])
        _f(cam, ok, "targetbody",
           _i(m.cam_targetbodyid[i]) == rf.cam_target_body[i])
        cam.add(ok)

    # ── materials ─────────────────────────────────────────────────────────
    var nmat = _i(m.nmat)
    if nmat != len(rf.mat_rgba_r):
        mat.add(False)
        mat.note("COUNT " + String(nmat) + "!=" + String(len(rf.mat_rgba_r)))
    for i in range(min(nmat, len(rf.mat_rgba_r))):
        var ok = True
        _f(mat, ok, "rgba",
           not (_neq32(_q(m.mat_rgba[i][0]), rf.mat_rgba_r[i])
                or _neq32(_q(m.mat_rgba[i][1]), rf.mat_rgba_g[i])
                or _neq32(_q(m.mat_rgba[i][2]), rf.mat_rgba_b[i])
                or _neq32(_q(m.mat_rgba[i][3]), rf.mat_rgba_a[i])))
        _f(mat, ok, "shininess",
           not _neq32(_q(m.mat_shininess[i]), rf.mat_shininess[i]))
        _f(mat, ok, "specular",
           not _neq32(_q(m.mat_specular[i]), rf.mat_specular[i]))
        _f(mat, ok, "reflectance",
           not _neq32(_q(m.mat_reflectance[i]), rf.mat_reflectance[i]))
        _f(mat, ok, "texrepeat",
           not (_neq32(_q(m.mat_texrepeat[i][0]), rf.mat_texrepeat_u[i])
                or _neq32(_q(m.mat_texrepeat[i][1]), rf.mat_texrepeat_v[i])))
        # ⚠ `mat_texid` IS (nmat, mjNTEXROLE) IN MuJoCo 3.x, not a scalar.
        # A `<material texture="...">` binds the RGB role (index 1); we model
        # one texture per material, so that is the slot to compare.
        _f(mat, ok, "texid", _i(m.mat_texid[i][1]) == rf.mat_tex_id[i])
        mat.add(ok)

    # ── sites ─────────────────────────────────────────────────────────────
    # ⚠ ASSERTED HERE, EXEMPTED BEFORE. The consistency gate had to report
    # this family instead of asserting it, because `_rcd`'s site records were
    # the wrong ones — it read `pos`/`size` off the tag and never resolved the
    # `<default>` class. Against MuJoCo there is nothing to exempt.
    var nsite = _i(m.nsite)
    if nsite != len(rf.site_pos_x):
        site.add(False)
        site.note("COUNT " + String(nsite) + "!=" + String(len(rf.site_pos_x)))
    for i in range(min(nsite, len(rf.site_pos_x))):
        var ok = True
        _f(site, ok, "body", _i(m.site_bodyid[i]) == rf.site_body_id[i])
        _f(site, ok, "pos",
           not (_neq(_q(m.site_pos[i][0]), rf.site_pos_x[i])
                or _neq(_q(m.site_pos[i][1]), rf.site_pos_y[i])
                or _neq(_q(m.site_pos[i][2]), rf.site_pos_z[i])))
        _f(site, ok, "size0",
           not _neq(_q(m.site_size[i][0]), rf.site_size_0[i]))
        site.add(ok)

    # ── textures: TYPE ONLY, see the module docstring ─────────────────────
    var ntex = _i(m.ntex)
    if ntex != rf.ntex:
        tex.add(False)
        tex.note("COUNT " + String(ntex) + "!=" + String(rf.ntex))
    for i in range(min(ntex, rf.ntex)):
        var ok = True
        _f(tex, ok, "type",
           _tex_type_from_mujoco(_i(m.tex_type[i])) == rf.tex_type[i])
        tex.add(ok)

    # ── spatial tendon render style ───────────────────────────────────────
    # MuJoCo indexes ALL tendons; ours records only the spatial ones, in the
    # same relative order. `tendon_num > 0` with wrap objects is the spatial
    # marker — a fixed tendon has no width/rgba worth drawing.
    var s_i = 0
    for ti in range(_i(m.ntendon)):
        if _i(m.tendon_num[ti]) == 0:
            continue
        # A fixed tendon wraps JOINTS (objtype mjOBJ_JOINT); a spatial one
        # wraps sites. Only the latter is drawn, so only it is recorded.
        var adr = _i(m.tendon_adr[ti])
        if _i(m.wrap_type[adr]) != _i(mujoco.mjtWrap.mjWRAP_SITE):
            continue
        if s_i >= len(rf.sten_width):
            break
        var ok = True
        _f(sten, ok, "width",
           not _neq32(_q(m.tendon_width[ti]), rf.sten_width[s_i]))
        _f(sten, ok, "rgba",
           not (_neq32(_q(m.tendon_rgba[ti][0]), rf.sten_rgba_r[s_i])
                or _neq32(_q(m.tendon_rgba[ti][1]), rf.sten_rgba_g[s_i])
                or _neq32(_q(m.tendon_rgba[ti][2]), rf.sten_rgba_b[s_i])))
        sten.add(ok)
        s_i += 1

    # ── <visual> ──────────────────────────────────────────────────────────
    var okv = True
    _f(vis, okv, "znear", not _neq32(_q(m.vis.map.znear), rf.vis_znear))
    _f(vis, okv, "shadowsize",
       _i(m.vis.quality.shadowsize) == rf.vis_shadowsize)
    _f(vis, okv, "headlight_ambient",
       not (_neq32(_q(m.vis.headlight.ambient[0]), rf.vis_headlight_ambient_r)
            or _neq32(_q(m.vis.headlight.ambient[1]),
                    rf.vis_headlight_ambient_g)
            or _neq32(_q(m.vis.headlight.ambient[2]),
                    rf.vis_headlight_ambient_b)))
    vis.add(okv)

    print("  ", name, "ok")


def test_render_fields_match_mujoco() raises:
    print("--- RenderFields vs mjModel ---")
    var geom = Tally()
    var light = Tally()
    var cam = Tally()
    var mat = Tally()
    var site = Tally()
    var tex = Tally()
    var sten = Tally()
    var vis = Tally()

    _check("quadruped  ", DMQuadrupedWalkModel.xml_path, DMQuadrupedWalkModel.xml_text(),
           geom, light, cam, mat, site, tex, sten, vis)
    _check("fish       ", DMFishSwimModel.xml_path, DMFishSwimModel.xml_text(),
           geom, light, cam, mat, site, tex, sten, vis)
    _check("ball_in_cup", DMBallInCupModel.xml_path, DMBallInCupModel.xml_text(),
           geom, light, cam, mat, site, tex, sten, vis)
    _check("humanoid   ", DMHumanoidModel.xml_path, DMHumanoidModel.xml_text(),
           geom, light, cam, mat, site, tex, sten, vis)
    _check("manipulator", DMManipulatorBringBallModel.xml_path, DMManipulatorBringBallModel.xml_text(),
           geom, light, cam, mat, site, tex, sten, vis)
    _check("walker     ", DMWalkerModel.xml_path, DMWalkerModel.xml_text(),
           geom, light, cam, mat, site, tex, sten, vis)
    _check("cheetah    ", DMCheetahModel.xml_path, DMCheetahModel.xml_text(),
           geom, light, cam, mat, site, tex, sten, vis)
    # ⚠ THE ONLY ROW THAT CAN FAIL THE TENDON STYLE — see the fixture.
    _check("sten-style ", "", STEN_STYLE_XML,
           geom, light, cam, mat, site, tex, sten, vis)
    # ⚠ THE ONLY `type="cube"` TEXTURE — the third arm of the type map.
    _check("cube-tex   ", "", MESH_ASSET_XML,
           geom, light, cam, mat, site, tex, sten, vis)
    # ⚠ THE ONLY MODEL HERE WITH MESH GEOMS. Every dm_control domain is
    # primitives-only (dog's 162 meshes are baked out of the port), so
    # without SO-ARM100's 18 mesh assets the `mesh` row above compares
    # nothing but -1 == -1.
    _check("so_arm100  ", "mojo_rl/envs/robots/assets/so_arm100.xml",
           _read("mojo_rl/envs/robots/assets/so_arm100.xml"),
           geom, light, cam, mat, site, tex, sten, vis)

    print("  TOTALS (rows differing / rows compared, then WHICH field):")
    geom.report("geom ")
    light.report("light")
    cam.report("cam  ")
    mat.report("mat  ")
    site.report("site ")
    tex.report("tex  ")
    sten.report("sten ")
    vis.report("vis  ")

    var bad = (
        geom.bad + light.bad + cam.bad + mat.bad + site.bad + tex.bad
        + sten.bad + vis.bad
    )
    assert_true(bad == 0,
        "RenderFields disagrees with MuJoCo in " + String(bad) + " rows")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
