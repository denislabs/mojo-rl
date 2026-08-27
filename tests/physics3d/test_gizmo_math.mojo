"""The transform gizmo's arithmetic — the half of it that is not C++.

WHY THIS EXISTS
===============
ImGuizmo hands back a 4x4. Everything between that matrix and a `pos=` in the
document can be wrong in a way that LOOKS right on screen, and none of it
needs a window to test:

  1. THE LAYOUT. ImGuizmo takes column-major `float[16]`; this project's
     `Mat4` is row-major. A transposed matrix does not fail — it draws a
     plausible gizmo in the wrong place. Arm 1 compares `mat4_to_cm` against
     `render.gpu_types.mat4_to_gpu_f32`, which is an INDEPENDENT transpose
     written for the GPU uniform, element for element.

  2. THE FRAME. The gizmo works in world space; MJCF stores a LOCAL frame.
     Arm 4 checks the composition against the engine's own forward
     kinematics — two different routes to one body's world pose — and arm 5
     checks that un-composing is exactly the inverse.

  3. `float32` NOISE. The matrix crosses the FFI as `float32`, so grabbing a
     handle without moving it re-quantises a `float64` record. Arm 6 is the
     NEGATIVE CONTROL: an unchanged matrix must produce ZERO edits, on every
     element, in both modes. Without it the studio would log an undo step and
     rewrite the document every frame the pointer rests on a handle.

⚠⚠ AND THE ARMS THAT MAKE THE OTHERS MEAN ANYTHING. The fixture's bodies are
ROTATED (30 degrees, then 20) precisely so that a world-space delta and a
local-space delta are DIFFERENT NUMBERS. On a model whose frames are all
axis-aligned, a gizmo that ignored the parent rotation entirely would pass
every arm below. Arm 7 asserts the two differ before it checks which one was
written.

⚠ THE FIXTURE ALSO CARRIES THE TWO SPELLINGS THAT BITE. One geom states its
orientation as `euler=` and one capsule as `fromto=`; MuJoCo REFUSES a tag
carrying two orientation attributes, and `fromto` OVERRIDES both `pos` and
`size`. Arm 9 re-parses the written document, so a rotation that was accepted
into the record and dropped on the way to the file fails here rather than on
the next reload. Whether the result is legal MJCF is MuJoCo's question and is
asked by `scripts/check_gizmo_vs_mujoco.py`.

Run: pixi run mojo run -I . tests/physics3d/test_gizmo_math.mojo
"""

from std.math import pi, sqrt

from mojo_rl.math3d import Vec3 as Vec3G, Quat as QuatG, Mat4 as Mat4G
from mojo_rl.render.gpu_types import mat4_to_gpu_f32
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.studio.remap import joint_qpos_adr
from mojo_rl.physics3d.studio.gizmo import (
    Frame, frame_to_cm, cm_to_frame, mat4_to_cm, parent_frame, local_frame,
    edit_frame, frame_drift, gizmo_edits, edits_from_frame,
    GIZMO_MOVE, GIZMO_TURN,
)
from mojo_rl.physics3d.studio.edit import (
    Edit, apply_edit, apply_edit_to_document, field_name,
    TARGET_GEOM, TARGET_BODY,
    F_POS_X, F_POS_Y, F_POS_Z, F_QUAT_W, F_QUAT_X, F_QUAT_Y, F_QUAT_Z,
    is_pos_field, is_quat_field,
)

comptime DT = DType.float64
comptime Vec3 = Vec3G[DT]
comptime Quat = QuatG[DT]
comptime Mat4 = Mat4G[DT]

comptime BASE = String("mojo_rl/envs/ant/assets")
"""Any directory with no meshes to resolve — the fixture references none."""

# ⚠⚠ THE ROTATIONS ARE THE POINT. `root` is turned 30 degrees about Z and
# `link` a further 20 about Y, so a world-space displacement and the local
# `pos=` that produces it are different numbers in every component. An
# axis-aligned fixture would let a gizmo that never applied the parent
# transform pass every arm in this file.
#
# ⚠ `mid` HAS A SLIDE JOINT so that displacing `qpos` MOVES its origin.
# A hinge anchored at the body origin rotates without translating, and
# `frame_drift` — which answers "how far is the gizmo from the drawn pose",
# in metres — would read 0 on a body whose joint had plainly moved.
comptime FIXTURE = String(
    '<mujoco model="gizmofix">\n'
    '  <compiler angle="degree"/>\n'
    '  <worldbody>\n'
    '    <body name="root" pos="0.1 0.2 0.3" euler="0 0 30">\n'
    '      <joint name="spin" type="hinge" axis="0 0 1"/>\n'
    '      <geom name="plain" type="box" pos="0.05 0 0"'
    ' size="0.02 0.03 0.04"/>\n'
    '      <geom name="turned" type="box" pos="0 0.07 0"'
    ' size="0.01 0.01 0.05" euler="0 45 0"/>\n'
    '      <geom name="seg" type="capsule" fromto="0 0 0 0 0 0.2"'
    ' size="0.01"/>\n'
    '      <body name="mid" pos="0 0 0.2" euler="0 20 0">\n'
    '        <joint name="rise" type="slide" axis="0 0 1"/>\n'
    '        <geom name="tip" type="sphere" pos="0 0 0.05" size="0.02"/>\n'
    '      </body>\n'
    '    </body>\n'
    '  </worldbody>\n'
    '</mujoco>\n'
)


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _f(v: Float64) -> String:
    var s = Int(v * 1000000.0 + (0.5 if v >= 0 else -0.5))
    return String(Float64(s) / 1000000.0)


struct Built(Movable):
    """A parsed fixture plus a stepped-free `Data`, and its draw poses."""

    var fmd: FlatModelDef
    var dims: DynDims
    var m: Model[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var positions: List[Vec3]
    var quats: List[Quat]
    var body_parent: List[Int]

    def __init__(out self, xml: String) raises:
        self.fmd = parse_xml_full(xml, BASE)
        self.dims = dims_from_flat(self.fmd)
        self.m = Model[DT, DynDims](self.dims)
        build_model_runtime[DT](self.fmd, self.dims, self.m)
        var sf = spec_fields_runtime[DT](self.fmd, self.dims, self.m)
        self.d = Data[DT, DynDims, 1](self.dims)
        for i in range(self.dims.get_nq()):
            self.d.qpos.data[i] = sf.qpos0.data[i]
        self.positions = List[Vec3]()
        self.quats = List[Quat]()
        self.body_parent = List[Int]()
        for b in self.fmd.bodies:
            self.body_parent.append(b.parent)
        self.sync()

    def sync(mut self) raises:
        forward_kinematics["cpu", DT, DynDims, 1](self.d, self.m)
        self.positions.clear()
        self.quats.clear()
        for b in range(self.dims.get_nbody()):
            self.positions.append(Vec3(
                Float64(self.d.xpos.data[b * 3 + 0]),
                Float64(self.d.xpos.data[b * 3 + 1]),
                Float64(self.d.xpos.data[b * 3 + 2]),
            ))
            self.quats.append(Quat(
                Float64(self.d.xquat.data[b * 4 + 3]),
                Float64(self.d.xquat.data[b * 4 + 0]),
                Float64(self.d.xquat.data[b * 4 + 1]),
                Float64(self.d.xquat.data[b * 4 + 2]),
            ))

    def frame_of(self, target: Int, index: Int) raises -> Frame:
        return edit_frame(self.fmd, self.positions, self.quats,
                          self.body_parent, target, index)

    def parent_of(self, target: Int, index: Int) raises -> Frame:
        return parent_frame(self.fmd, self.positions, self.quats,
                            self.body_parent, target, index)

    def edits_for(
        self, target: Int, index: Int, op: Int, world: List[Float32]
    ) raises -> List[Edit]:
        return gizmo_edits(self.fmd, self.positions, self.quats,
                           self.body_parent, target, index, op, world, 1.0)

    def geom_named(self, n: String) -> Int:
        for i in range(len(self.fmd.geom_names)):
            if self.fmd.geom_names[i] == n:
                return i
        return -1

    def body_named(self, n: String) -> Int:
        for i in range(len(self.fmd.body_names)):
            if self.fmd.body_names[i] == n:
                return i
        return -1


def _quat_gap(a: Quat, b: Quat) -> Float64:
    """Distance between two ROTATIONS, not between two quadruples.

    ⚠ `q` AND `-q` ARE THE SAME ROTATION. A component-wise difference reports
    2.0 for two identical rotations that happened to be written with opposite
    signs, which would fail a correct result and pass nothing.
    """
    var same = (abs(a.w - b.w) + abs(a.x - b.x) + abs(a.y - b.y)
                + abs(a.z - b.z))
    var flip = (abs(a.w + b.w) + abs(a.x + b.x) + abs(a.y + b.y)
                + abs(a.z + b.z))
    return same if same < flip else flip


def main() raises:
    var t = Tally()
    print("=== the transform gizmo's arithmetic ===")

    var B = Built(FIXTURE)
    print("  fixture:", len(B.fmd.bodies), "bodies,", len(B.fmd.geoms),
          "geoms, nq =", B.dims.get_nq())

    # ── arm 0: the fixture is actually rotated ────────────────────────────
    # ⚠⚠ FIRST, BECAUSE IT LICENSES ARMS 5 AND 7. On an axis-aligned model a
    # gizmo that never applied the parent transform passes everything below.
    print("--- arm 0: the fixture's frames are NOT axis-aligned ---")
    var root = B.body_named(String("root"))
    var mid = B.body_named(String("mid"))
    t.truth(root > 0 and mid > 0, "the fixture parsed with both bodies")
    var rq = B.quats[root]
    t.truth(_quat_gap(rq, Quat(1.0, 0.0, 0.0, 0.0)) > 0.1,
            String("'root' is rotated in world (quat w=", _f(rq.w),
                   " z=", _f(rq.z), ")"))
    var mq = B.quats[mid]
    t.truth(_quat_gap(mq, rq) > 0.1, "'mid' is rotated relative to 'root'")

    # ── arm 1: the layout, against an independent transpose ───────────────
    print("--- arm 1: column-major layout vs the GPU uniform's transpose ---")
    var probe = Mat4.from_quat(
        Quat.from_axis_angle(Vec3(0.3, -0.5, 0.81), 1.1).normalized(),
        Vec3(1.5, -2.25, 0.125),
    )
    var mine = mat4_to_cm(probe)
    var theirs = mat4_to_gpu_f32(probe)
    var layout_bad = 0
    for i in range(16):
        if mine[i] != theirs[i]:
            layout_bad += 1
    t.truth(layout_bad == 0,
            String("all 16 elements agree with mat4_to_gpu_f32 (",
                   layout_bad, " differ)"))
    # ⚠ AND THE TRANSPOSE IS NOT A NO-OP on this matrix, or the arm above
    # would pass for a `mat4_to_cm` that simply copied the fields in order.
    t.truth(mine[1] != mine[4] and mine[12] != 0.0,
            "the probe matrix is asymmetric and translated (control)")

    # ── arm 2: frame_to_cm agrees with a Mat4 built independently ─────────
    print("--- arm 2: frame_to_cm == mat4_to_cm(Mat4.from_quat) ---")
    var pf = Frame(Vec3(0.11, -0.22, 0.33),
                   Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), 0.6))
    var a = frame_to_cm(pf)
    var b = mat4_to_cm(Mat4.from_quat(pf.quat, pf.pos))
    var worst2 = 0.0
    for i in range(16):
        var dv = abs(Float64(a[i]) - Float64(b[i]))
        if dv > worst2:
            worst2 = dv
    t.truth(worst2 < 1e-6,
            String("the two constructions agree (worst ", _f(worst2), ")"))

    # ── arm 3: the round trip ─────────────────────────────────────────────
    print("--- arm 3: frame -> float[16] -> frame ---")
    var worst3 = 0.0
    for g in range(len(B.fmd.geoms)):
        var f0 = B.frame_of(TARGET_GEOM, g)
        var f1 = cm_to_frame(frame_to_cm(f0))
        var dp = (f1.pos - f0.pos).length()
        var dq = _quat_gap(f1.quat, f0.quat)
        if dp > worst3:
            worst3 = dp
        if dq > worst3:
            worst3 = dq
    t.truth(worst3 < 1e-6,
            String("every geom frame survives the float32 trip (worst ",
                   _f(worst3), ")"))

    # ── arm 4: the composition agrees with forward kinematics ─────────────
    # ⚠ TWO INDEPENDENT ROUTES TO ONE POSE. `edit_frame` walks the PARSER's
    # records (parent world ∘ body pos/quat); `xpos`/`xquat` come out of the
    # ENGINE's forward kinematics. At the reference pose they must agree, and
    # a composition applied in the wrong order agrees on neither position nor
    # orientation.
    print("--- arm 4: edit_frame(body) == forward kinematics, at qpos0 ---")
    var worst4 = 0.0
    for bi in range(1, B.dims.get_nbody()):
        var ef = B.frame_of(TARGET_BODY, bi)
        var dp = (ef.pos - B.positions[bi]).length()
        var dq = _quat_gap(ef.quat, B.quats[bi])
        if dp > worst4:
            worst4 = dp
        if dq > worst4:
            worst4 = dq
    t.truth(worst4 < 1e-12,
            String("every body's edit frame is its FK pose (worst ",
                   _f(worst4), ")"))

    # ── arm 5: un-composing is exactly the inverse ────────────────────────
    print("--- arm 5: parent.inverse() undoes parent.compose() ---")
    var worst5 = 0.0
    for g in range(len(B.fmd.geoms)):
        var par = B.parent_of(TARGET_GEOM, g)
        var loc = local_frame(B.fmd, TARGET_GEOM, g)
        var back = par.inverse().compose(par.compose(loc))
        var dp = (back.pos - loc.pos).length()
        var dq = _quat_gap(back.quat, loc.quat)
        if dp > worst5:
            worst5 = dp
        if dq > worst5:
            worst5 = dq
    t.truth(worst5 < 1e-12,
            String("local -> world -> local is exact (worst ", _f(worst5),
                   ")"))

    # ── arm 6: THE NEGATIVE CONTROL ───────────────────────────────────────
    # ⚠⚠ A GRAB THAT DID NOT MOVE MUST WRITE NOTHING. The matrix crosses the
    # FFI as float32, so every component comes back perturbed; without the
    # noise floor this arm reports one edit per component per element.
    print("--- arm 6: an UNMOVED gizmo emits zero edits ---")
    var spurious = 0
    for g in range(len(B.fmd.geoms)):
        var same = frame_to_cm(B.frame_of(TARGET_GEOM, g))
        spurious += len(B.edits_for(TARGET_GEOM, g, GIZMO_MOVE, same))
        spurious += len(B.edits_for(TARGET_GEOM, g, GIZMO_TURN, same))
    for bi in range(1, B.dims.get_nbody()):
        var sameb = frame_to_cm(B.frame_of(TARGET_BODY, bi))
        spurious += len(B.edits_for(TARGET_BODY, bi, GIZMO_MOVE, sameb))
        spurious += len(B.edits_for(TARGET_BODY, bi, GIZMO_TURN, sameb))
    t.truth(spurious == 0,
            String("no element emitted an edit for standing still (",
                   spurious, " did)"))

    # ── arm 7: a MOVE, through the parent's rotation ──────────────────────
    print("--- arm 7: a world-space move lands where it was asked to ---")
    var gi = B.geom_named(String("turned"))
    t.truth(gi >= 0, "geom 'turned' is in the model")
    var before7 = B.frame_of(TARGET_GEOM, gi)
    var want7 = Frame(before7.pos + Vec3(0.05, 0.0, 0.0), before7.quat)
    var ed7 = B.edits_for(TARGET_GEOM, gi, GIZMO_MOVE, frame_to_cm(want7))
    var npos7 = 0
    var nquat7 = 0
    for i in range(len(ed7)):
        if is_pos_field(ed7[i].field):
            npos7 += 1
        if is_quat_field(ed7[i].field):
            nquat7 += 1
    t.truth(nquat7 == 0,
            String("a MOVE emitted no orientation edit (", nquat7, ")"))
    t.truth(npos7 >= 2,
            String("the local displacement is not one-dimensional (", npos7,
                   " components moved) — the parent rotation was applied"))
    # ⚠ AND THE NUMBERS DIFFER FROM THE WORLD DELTA. If the gizmo ignored the
    # parent, `pos[0]` would move by exactly 0.05 and the others not at all.
    var lx = local_frame(B.fmd, TARGET_GEOM, gi).pos.x
    var got_lx = lx
    for i in range(len(ed7)):
        if ed7[i].field == F_POS_X:
            got_lx = ed7[i].value
    t.truth(abs((got_lx - lx) - 0.05) > 1e-3,
            String("the LOCAL delta is not the world delta (",
                   _f(got_lx - lx), " vs 0.05)"))
    for i in range(len(ed7)):
        apply_edit(B.fmd, B.m, ed7[i])
    var after7 = B.frame_of(TARGET_GEOM, gi)
    t.truth((after7.pos - want7.pos).length() < 1e-6,
            String("the geom ended up at the requested world position (off"
                   " by ", _f((after7.pos - want7.pos).length()), ")"))
    t.truth(_quat_gap(after7.quat, before7.quat) < 1e-12,
            "and its orientation was not touched")

    # ── arm 8: a TURN ─────────────────────────────────────────────────────
    print("--- arm 8: a world-space turn lands where it was asked to ---")
    var before8 = B.frame_of(TARGET_GEOM, gi)
    var spin = Quat.from_axis_angle(Vec3(0.0, 0.0, 1.0), pi / 6.0)
    var want8 = Frame(before8.pos, (spin * before8.quat).normalized())
    var ed8 = B.edits_for(TARGET_GEOM, gi, GIZMO_TURN, frame_to_cm(want8))
    var npos8 = 0
    var nquat8 = 0
    for i in range(len(ed8)):
        if is_pos_field(ed8[i].field):
            npos8 += 1
        if is_quat_field(ed8[i].field):
            nquat8 += 1
    t.truth(npos8 == 0, String("a TURN emitted no position edit (", npos8,
                               ")"))
    t.truth(nquat8 == 4,
            String("all FOUR quaternion components were written (", nquat8,
                   ") — three would not be a rotation"))
    for i in range(len(ed8)):
        apply_edit(B.fmd, B.m, ed8[i])
    var after8 = B.frame_of(TARGET_GEOM, gi)
    t.truth(_quat_gap(after8.quat, want8.quat) < 1e-6,
            String("the geom ended up at the requested world orientation"
                   " (off by ", _f(_quat_gap(after8.quat, want8.quat)), ")"))
    t.truth((after8.pos - before8.pos).length() < 1e-12,
            "and its position was not touched")

    # ── arm 9: the DOCUMENT carries both edits ────────────────────────────
    # ⚠⚠ 'turned' STATES ITS ORIENTATION AS `euler=`. MuJoCo refuses a tag
    # with two orientation attributes, so writing `quat` beside it produces a
    # file that will not load — from a studio showing the new rotation
    # happily. Re-parsing is what catches a rotation that reached the record
    # and not the file.
    print("--- arm 9: the edits survive a write and a re-parse ---")
    var doc = FIXTURE
    for i in range(len(ed7)):
        doc = apply_edit_to_document(B.fmd, B.m, doc, ed7[i])
    for i in range(len(ed8)):
        doc = apply_edit_to_document(B.fmd, B.m, doc, ed8[i])
    t.truth(doc.find(' euler="0 45 0"') == -1,
            "the rival `euler=` spelling was removed from 'turned'")
    t.truth(doc != FIXTURE, "the document actually changed (control)")
    var B2 = Built(doc)
    var gi2 = B2.geom_named(String("turned"))
    var re = B2.frame_of(TARGET_GEOM, gi2)
    t.truth((re.pos - want7.pos).length() < 1e-6,
            String("the re-parsed geom is at the moved position (off by ",
                   _f((re.pos - want7.pos).length()), ")"))
    t.truth(_quat_gap(re.quat, want8.quat) < 1e-6,
            String("and at the turned orientation (off by ",
                   _f(_quat_gap(re.quat, want8.quat)), ")"))

    # ── arm 10: `fromto` is materialised rather than fought ───────────────
    print("--- arm 10: a `fromto` capsule can be turned at all ---")
    var si = B.geom_named(String("seg"))
    t.truth(si >= 0, "geom 'seg' is in the model")
    var b10 = B.frame_of(TARGET_GEOM, si)
    var want10 = Frame(
        b10.pos,
        (Quat.from_axis_angle(Vec3(1.0, 0.0, 0.0), 0.4) * b10.quat)
        .normalized(),
    )
    var ed10 = B.edits_for(TARGET_GEOM, si, GIZMO_TURN, frame_to_cm(want10))
    for i in range(len(ed10)):
        apply_edit(B.fmd, B.m, ed10[i])
    var doc10 = FIXTURE
    for i in range(len(ed10)):
        doc10 = apply_edit_to_document(B.fmd, B.m, doc10, ed10[i])
    t.truth(doc10.find("fromto=") == -1,
            "`fromto` was replaced by explicit pos/quat/size")
    var B3 = Built(doc10)
    var r10 = B3.frame_of(TARGET_GEOM, B3.geom_named(String("seg")))
    t.truth(_quat_gap(r10.quat, want10.quat) < 1e-6,
            String("the re-parsed capsule kept the rotation (off by ",
                   _f(_quat_gap(r10.quat, want10.quat)), ")"))
    t.truth((r10.pos - b10.pos).length() < 1e-6,
            String("and stayed where it was (off by ",
                   _f((r10.pos - b10.pos).length()), ")"))

    # ── arm 11: the drift the panel reports ───────────────────────────────
    # ⚠⚠ A JOINTED BODY IS DRAWN SOMEWHERE ITS `pos=` DOES NOT DESCRIBE, and
    # the studio says so rather than leaving a handle floating beside the
    # part. `rise` is a SLIDE joint, so the gap is a number this arm knows.
    print("--- arm 11: frame_drift, on a body whose joint has moved ---")
    var C = Built(FIXTURE)
    var g_drift = 0.0
    for g in range(len(C.fmd.geoms)):
        var dg = frame_drift(C.fmd, C.positions, C.quats, C.body_parent,
                             TARGET_GEOM, g)
        if dg > g_drift:
            g_drift = dg
    t.truth(g_drift == 0.0, "every GEOM reports zero drift, by construction")
    var mid_c = C.body_named(String("mid"))
    var d0 = frame_drift(C.fmd, C.positions, C.quats, C.body_parent,
                         TARGET_BODY, mid_c)
    t.truth(d0 < 1e-12,
            String("at the reference pose the body reports none (", _f(d0),
                   ")"))
    # Displace the slide joint by a known amount.
    # ⚠ THE ADDRESS COMES FROM `joint_qpos_adr`, the same helper the state
    # remap uses — re-deriving it here would be a second answer to a question
    # that already has one.
    var adrs = joint_qpos_adr(C.fmd)
    var qadr = -1
    for j in range(len(C.fmd.joints)):
        if C.fmd.joint_names[j] == String("rise"):
            qadr = adrs[j]
    t.truth(qadr >= 0, "the slide joint has a qpos address")
    C.d.qpos.data[qadr] = Scalar[DT](0.07)
    C.sync()
    var d1 = frame_drift(C.fmd, C.positions, C.quats, C.body_parent,
                         TARGET_BODY, mid_c)
    t.truth(abs(d1 - 0.07) < 1e-9,
            String("after sliding 0.07 m the drift is 0.07 m (", _f(d1),
                   ")"))

    # ── arm 12: the worldbody has no editable frame ───────────────────────
    print("--- arm 12: body 0 is refused, not written past ---")
    var w = C.frame_of(TARGET_BODY, 0)
    t.truth(w.pos.length() == 0.0 and w.quat.w == 1.0,
            "edit_frame(body 0) is the identity, not fmd.bodies[-1]")
    var ew = edits_from_frame(
        TARGET_BODY, 0, GIZMO_MOVE, Frame.identity(),
        Frame(Vec3(1.0, 2.0, 3.0), Quat(1.0, 0.0, 0.0, 0.0)),
    )
    t.truth(len(ew) == 3,
            String("edits_from_frame itself is honest about a real move (",
                   len(ew), ") — the guard is at the call site"))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_gizmo_math: " + String(t.fails) + " failed")
