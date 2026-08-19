"""physics3d studio — open ANY MJCF, inspect it, and open another without relaunching.

    pixi run build-imgui                                   # ONCE, for the UI
    pixi run mojo build -I . -o /tmp/studio examples/physics3d/physics_studio.mojo
    /tmp/studio                                            # then File > Open
    /tmp/studio mojo_rl/envs/humanoid/assets/humanoid.xml sweep 0.4

⚠ BUILD ONCE AND RUN THE BINARY. `mojo run` recompiles, and this file takes
~2 min to build. It does NOT grow with the number of models it can open —
that is the whole point, and it is the number `docs/PHYSICS3D_STUDIO_PLAN.md`
§4 was written to make visible. `dm_viewer_imgui`, which bakes 47 models into
its type, takes ~8 minutes.

WHAT THIS IS
============
Slices S0 + S1 of the studio: one binary, any file, on screen, driven, at
60 Hz, with a MuJoCo-`simulate`-shaped UI — menu bar, left Options panel,
right Explorer/Inspector tabs — and click-to-select with a yellow outline.

⚠⚠ **A MODEL SWAP IS JUST REBUILDING VALUES.** `File > Open` does not
relaunch anything: `Model`, `Data` and `EulerIntegrator` are all
`[DT, DynDims]` — ONE type, whatever the model — so swapping is assignment,
not instantiation. The window survives via `RendererHandoff`. That is the
runtime-dims migration paying off in the most visible possible way: on the
comptime path each model is a distinct type and this feature cannot exist.

⚠ CPU ONLY, ONE ENV, ON PURPOSE. A runtime dims provider captured by a GPU
kernel reads 0 and silently zeroes its output, so there is no runtime GPU path
and this tool does not pretend to have one. One model at 60 Hz needs no GPU;
see the plan's §8 for the "bake and train" answer.

⚠ NOT BUILT ON `Phyics3dEnv`. An env is the RL contract — obs, reward, done,
action space — and a scene is not a task. See `studio/panel.mojo`'s header and
the plan's §5.1.

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window.

THE TWO ARGUMENTS THAT ARE NOT IN THE FILE
==========================================
`max_contacts` and `nmesh_verts` are workspace budgets, not model properties
(see `dims_from_flat`). Too small a `max_contacts` SILENTLY DROPS contacts, so
the Options panel shows the high-water mark as a bar. `nmesh_verts` cannot be
derived before the meshes load, but the builder raises WITH THE NUMBER IT
NEEDS, so `Loaded._build` retries on the raise and the user never has to know.
That is §1.2 of the plan, implemented.
"""

from std.sys import argv
from std.random import seed, random_float64
from std.math import sin
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.math3d import Vec3 as Vec3G, Quat as QuatG
from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import Data, Model, DynDims, SpecFields
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.parser.render_fields import (
    RenderFields, build_render_fields,
)
from mojo_rl.physics3d.parser.model_def_from_xml import RfOnlyModelDef
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.model.model_renderer import ModelRenderer, OverlayLine
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, MODEL_MESH_META_SIZE, MODEL_GEOM_SIZE,
    GEOM_IDX_MESH_ID, MAX_GPU_MESHES, MODEL_BODY_SIZE, BODY_IDX_MASS,
)
from mojo_rl.physics3d.studio.scene import SceneDoc, scene_from_base
from mojo_rl.physics3d.studio.writer import to_mjcf as export_flat_mjcf
from mojo_rl.physics3d.studio import (
    Ray, ray_through_pixel, pick_geom, outline_geom, outline_body,
    StudioPanel, PanelOut, build_ui, SIDEBAR_W, RIGHT_W,
)
from mojo_rl.physics3d.studio.panel import SEL_BODY, SEL_GEOM, SEL_NONE
from mojo_rl.physics3d.studio.edit import (
    Edit, EditLog, apply_edit, needs_rebuild,
    TARGET_GEOM, TARGET_BODY,
    F_POS_X, F_POS_Y, F_POS_Z, F_SIZE_0, F_SIZE_1, F_SIZE_2,
    F_RGBA_R, F_RGBA_G, F_RGBA_B, F_RGBA_A, F_FRICTION, F_MASS,
)
from mojo_rl.render.imgui import ig_want_mouse

comptime DT = DType.float64
comptime Vec3 = Vec3G[DT]
comptime Quat = QuatG[DT]

comptime DRIVE_ZERO: Int = 0
comptime DRIVE_RANDOM: Int = 1
comptime DRIVE_SWEEP: Int = 2

comptime MAX_CONTACTS: Int = 128
"""Generous, because a composed scene is where this tool is going and a short
budget drops contacts with no error. 128 * 30 scalars is ~30 kB."""


def parse_drive(s: String) -> Int:
    if s == "zero":
        return DRIVE_ZERO
    if s == "random":
        return DRIVE_RANDOM
    return DRIVE_SWEEP


# ═══════════════════════════════════════════════════════════════════════════
# one loaded model
# ═══════════════════════════════════════════════════════════════════════════


struct Loaded(Movable):
    """Everything one MJCF turns into. Replaced wholesale on File > Open.

    ⚠⚠ EVERY FIELD HAS ONE TYPE FOR EVERY MODEL — `[DT, DynDims]` — which is
    why a swap is an assignment rather than a relaunch. On the comptime path
    each model is a distinct `ModelDims[...]`, so a struct like this could not
    be written at all, and `dm_viewer` has to tear the whole env down and
    rebuild it per task through 47 separate instantiations. This is the
    runtime-dims migration's payoff in one struct.
    """

    var path: String
    var fmd: FlatModelDef
    var dims: DynDims
    var m: Model[DT, DynDims]
    var sf: SpecFields[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var integ: EulerIntegrator[DT, DynDims, BATCH=1, MAX_CONDIM=3]
    var rf: RenderFields
    var flat: String
    var base_dir: String
    var body_parent: List[Int]
    var geom_body: List[Int]
    var joint_body: List[Int]
    var mesh_half: List[Float64]
    """Real half-extents per geom, 3 each — see `_measure_meshes`."""
    var hull_verts: Int
    """Hull vertices actually LOADED, as opposed to the budget allocated.

    ⚠ THE TWO ARE NOT THE SAME NUMBER AND THE DIFFERENCE LOOKS LIKE A CAP.
    `nmesh_verts` is a workspace budget this loader DISCOVERS by doubling
    (0 → 4096 → 8192 …) until the builder stops raising, so it reports the
    first rung that fit, not what the model needs. ToddlerBot showed
    "mesh verts 4096" while loading two collision hulls of a few hundred
    vertices — a round power of two that reads as a limit being hit. Only
    COLLIDABLE hulls are loaded; the 45 visual meshes go straight from STL to
    the GPU and never enter this budget at all."""

    def __init__(out self, path: String, xml: String,
                 base_dir: String) raises:
        """Build from scene TEXT the studio generated, not from a file.

        ⚠ THE SCENE IS NEVER WRITTEN TO DISK TO BE READ BACK. A structural
        edit regenerates the document in memory and rebuilds from it, so an
        edit cannot fail on a filesystem error and the user's file is not
        touched until they ask. `path` is kept only for display and for undo's
        re-parse.
        """
        self.path = path
        # ⚠ EXPANDED BEFORE PARSING. `<include>`/`<attach>`/`<frame>` become
        # flat text so the ONE existing parser reads it — the studio must not
        # become a second model path (plan §10 risk 2). A file using none of
        # the three passes through untouched.
        self.flat = expand_mjcf(xml, base_dir)
        self.base_dir = base_dir
        self.fmd = parse_xml_full(self.flat, base_dir)

        # ⚠ THE MESH VERTEX BUDGET CANNOT BE DERIVED HERE, and that is not an
        # oversight — `dims_from_flat`'s docstring explains it: the hull vertex
        # count is known only once the meshes load, INSIDE the builder, after
        # this point. Loading them twice to count them would double the most
        # expensive stage of the load.
        #
        # So: guess, and let the builder tell us. It raises naming the number
        # it needs, so doubling on the raise converges in a handful of tries
        # and the user never has to know the number. §1.2's "retry-on-raise
        # load loop", and it is why dropping in a mesh PROP will need no
        # codegen later.
        var verts = 0
        var dims = dims_from_flat(
            self.fmd, max_contacts=MAX_CONTACTS, nmesh_verts=verts
        )
        var m = Model[DT, DynDims](dims)
        var tries = 0
        while True:
            try:
                build_model_runtime[DT](self.fmd, dims, m)
                break
            except e:
                # ⚠ ONLY THE VERTEX-BUDGET RAISE IS RETRYABLE. Anything else
                # is a real load failure and must reach the user unchanged —
                # swallowing it here turns "your MJCF is malformed" into an
                # infinite loop.
                if String(e).find("mesh vertex capacity") == -1:
                    raise e
                tries += 1
                if tries > 24:
                    raise e
                verts = 4096 if verts == 0 else verts * 2
                dims = dims_from_flat(
                    self.fmd, max_contacts=MAX_CONTACTS, nmesh_verts=verts
                )
                m = Model[DT, DynDims](dims)

        self.dims = dims
        self.m = m^
        self.sf = spec_fields_runtime[DT](self.fmd, self.dims)
        self.d = Data[DT, DynDims, 1](self.dims)
        self.integ = EulerIntegrator[DT, DynDims, BATCH=1, MAX_CONDIM=3](
            self.dims
        )
        # ⚠ THE **EXPANDED** TEXT, NOT THE SOURCE. `RenderFields.xml_text` is
        # what `render_skin` and `body_names_of` scan, and after an `<attach>`
        # the source names none of the spliced bodies — the scene file is a
        # floor and two `<attach/>` tags. Handing it the pre-expansion text
        # would give a skin that binds no bones, silently.
        self.rf = build_render_fields(self.fmd, self.flat, self.base_dir)

        # Flat index maps, so `panel.mojo` never sees a `FlatModelDef`.
        self.body_parent = List[Int]()
        for b in self.fmd.bodies:
            self.body_parent.append(b.parent)
        self.geom_body = List[Int]()
        for g in self.fmd.geoms:
            self.geom_body.append(g.body_id)
        self.joint_body = List[Int]()
        for j in self.fmd.joints:
            self.joint_body.append(j.body_id)
        self.mesh_half = List[Float64]()
        self.hull_verts = 0
        self.mesh_half = self._measure_meshes()
        for mi in range(MAX_GPU_MESHES):
            self.hull_verts += Int(Float64(
                self.m.mesh_meta.data[mi * MODEL_MESH_META_SIZE + 1]
            ))

        self.reset()


    def __init__(out self, path: String) raises:
        """Parse, size, build. Raises with a readable message on a bad file.

        ⚠ THE PARSE IS THE EXPENSIVE HALF and it is per-ASSET, not per-edit:
        dog is 9.0 ms of text parsing, so_arm100 is 11.2 ms of mesh loading,
        while `dims_from_flat` and `spec_fields_runtime` are under 0.01 ms.
        S2's composer caches this `FlatModelDef` per asset file for exactly
        that reason; the studio loads one model at a time, so it just keeps it.
        """
        # ⚠ DELEGATES, because Mojo forbids calling a method on `self` before
        # every field is initialised — so the shared tail cannot be a helper
        # method. One constructor holds the body and the other hands it text.
        var src = read_model_source(path)
        self = Self(path, src[0], src[1])

    def _measure_meshes(self) -> List[Float64]:
        """Per-geom half-extents, measured from the LOADED hull vertices.

        ⚠⚠ WITHOUT THIS, A MESH GEOM OUTLINES AS A ONE-METRE CUBE. A `<geom
        mesh="...">` normally carries no `size` attribute — the mesh defines
        the shape — so `GeomData`'s defaults survive: `half_x/y/z` are 0 and
        `radius` is 0.5. Drawing a box from `radius` therefore boxed every
        part of a 30 cm arm in a 1 m cube, which reads as a broken outline
        rather than as empty size fields.

        The vertices exist only AFTER the build (`fields_build` loads the
        STLs), which is why this runs here and not in `build_render_fields` —
        the parser has the filename, not the geometry.

        ⚠ THE HULL IS ALREADY IN THE GEOM'S OWN FRAME, so this is a plain
        min/max over the vertex block; no pose is applied. Applying the body
        transform here would make the box grow as the robot moves.
        """
        var out = List[Float64](length=len(self.fmd.geoms) * 3, fill=0.0)
        var nverts = self.dims.get_nmesh_verts()
        if nverts <= 0:
            return out^
        var biggest = 0.0
        var unmeasured = List[Int]()
        for g in range(len(self.fmd.geoms)):
            # ⚠⚠ THE MESH ID MUST COME FROM THE **MODEL**, NOT THE PARSE.
            # `FlatModelDef.geoms[g].mesh_id` is an index into the XML's ASSET
            # table (ToddlerBot declares 47); `mesh_meta` is keyed by the
            # LOADED-hull index, and only collidable meshes are loaded, capped
            # at `MAX_GPU_MESHES` (16). Using the asset index read
            # `mesh_meta[43]` out of a 16-row table — "index 172 is out of
            # bounds, valid range is 0 to 63". It went unnoticed on so_arm101
            # only because its asset and hull indices happen to overlap in the
            # low range. `fields_build` does the remap; read its result.
            var mid = Int(Float64(
                self.m.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_MESH_ID]
            ))
            if mid < 0 or mid >= MAX_GPU_MESHES:
                unmeasured.append(g)
                continue
            var adr = Int(Float64(
                self.m.mesh_meta.data[mid * MODEL_MESH_META_SIZE + 0]
            ))
            var n = Int(Float64(
                self.m.mesh_meta.data[mid * MODEL_MESH_META_SIZE + 1]
            ))
            if n <= 0 or adr < 0 or adr + n > nverts:
                # ⚠ A VISUAL-ONLY MESH HAS NO VERTICES HERE, and that is by
                # design: `fields_build` loads the COLLIDABLE hulls only —
                # so_arm101 has 30 mesh geoms and one of them measures 0, and
                # sawyer keeps 10 such visual geoms. The RENDERER still draws
                # the shape (it loads the STL by filename), so the part is on
                # screen with no measurable bound in the model. Collected and
                # given the model's own scale below rather than the meaningless
                # 0.5 default.
                unmeasured.append(g)
                continue
            var hx = 0.0
            var hy = 0.0
            var hz = 0.0
            for v in range(adr, adr + n):
                var x = abs(Float64(self.m.mesh_verts.data[v * 3 + 0]))
                var y = abs(Float64(self.m.mesh_verts.data[v * 3 + 1]))
                var z = abs(Float64(self.m.mesh_verts.data[v * 3 + 2]))
                if x > hx:
                    hx = x
                if y > hy:
                    hy = y
                if z > hz:
                    hz = z
            out[g * 3 + 0] = hx
            out[g * 3 + 1] = hy
            out[g * 3 + 2] = hz
            if hx > biggest:
                biggest = hx
            if hy > biggest:
                biggest = hy
            if hz > biggest:
                biggest = hz

        # ⚠ THE UNMEASURABLE ONES GET THE MODEL'S OWN SCALE, AND IT IS A
        # MARKER RATHER THAN A BOUND. The alternative is `radius`, which for a
        # mesh geom is `GeomData`'s untouched default of **0.5** — measured on
        # so_arm101, whose real parts are 0.012-0.050 m, so that default boxes
        # a 3 cm part in a 1 m cube. Wrong by 20-40x reads as a broken
        # outline; wrong by a little reads as an approximate one, which is
        # what it is.
        if biggest <= 0.0:
            biggest = 0.02
        for i in range(len(unmeasured)):
            var g = unmeasured[i]
            out[g * 3 + 0] = biggest
            out[g * 3 + 1] = biggest
            out[g * 3 + 2] = biggest
        return out^

    def reset(mut self):
        """The reference pose, and zero velocity."""
        for i in range(self.dims.get_nq()):
            self.d.qpos.data[i] = self.sf.qpos0.data[i]
        for i in range(self.dims.get_nv()):
            self.d.qvel.data[i] = Scalar[DT](0)

    def describe(self) raises:
        print("  ", self.path)
        print("    nbody", self.dims.get_nbody(),
              " ngeom", self.dims.get_ngeom(),
              " nq", self.dims.get_nq(), " nv", self.dims.get_nv(),
              " nact", self.dims.get_nact())
        print("    nsite", self.dims.get_nsite(),
              " ntendon", self.dims.get_ntendon(),
              " nequality", self.dims.get_nequality())
        # ⚠ USED vs BUDGET, because a lone round number reads as a cap.
        print("    collidable hull verts", self.hull_verts, "used /",
              self.dims.get_nmesh_verts(), "budgeted",
              " (visual meshes are drawn from STL and use neither)")


# ═══════════════════════════════════════════════════════════════════════════
# the inspector's record — the STUDIO owns this, not the panel
# ═══════════════════════════════════════════════════════════════════════════


def _record(
    L: Loaded,
    p: StudioPanel,
    positions: List[Vec3],
    quats: List[Quat],
    mut keys: List[String],
    mut vals: List[Float64],
    mut editable: List[Int],
):
    """Flatten the selection into the (key, value) pair `ui_inspector` shows.

    ⚠ THE STUDIO OWNS THIS, NOT THE PANEL, and the split is the whole reason
    the UI compiles once. `Data[DT, DynDims, 1]` and `FlatModelDef` are exactly
    the types `panel.mojo` must not name; handing it two plain `List`s keeps
    every widget line generic-free.

    Field names follow MuJoCo's `simulate` inspector, so a number here can be
    compared with the reference tool without translating.
    """
    keys.clear()
    vals.clear()
    # ⚠ PARALLEL TO `keys`/`vals`, AND THE STUDIO OWNS IT. The panel cannot
    # know which record slot a row maps to — that knowledge is exactly what
    # would make `panel.mojo` generic — so it gets an edit-field id per row,
    # or -1 for read-only, and hands back which row moved.
    editable.clear()
    if p.sel_kind == SEL_BODY:
        var b = p.sel_index
        if b < 0 or b >= len(positions):
            return
        keys.append(String("pos[0]")); editable.append(-1); vals.append(positions[b].x)
        keys.append(String("pos[1]")); editable.append(-1); vals.append(positions[b].y)
        keys.append(String("pos[2]")); editable.append(-1); vals.append(positions[b].z)
        keys.append(String("quat[0]")); editable.append(-1); vals.append(quats[b].w)
        keys.append(String("quat[1]")); editable.append(-1); vals.append(quats[b].x)
        keys.append(String("quat[2]")); editable.append(-1); vals.append(quats[b].y)
        keys.append(String("quat[3]")); editable.append(-1); vals.append(quats[b].z)
        # ⚠ body 0 IS THE WORLDBODY and is absent from `fmd.bodies`, so the
        # record index is b-1. Reading `fmd.bodies[b]` instead would report
        # every body's mass as its CHILD's — off by one and entirely
        # plausible on screen.
        if b > 0 and b - 1 < len(L.fmd.bodies):
            var bd = L.fmd.bodies[b - 1]
            # ⚠⚠ THE **MODEL**'S MASS, NOT THE RECORD'S. `BodyData.mass` is
            # what the XML said, and a body with no explicit mass keeps the
            # default 1.0 — every dropped-in prop showed as 1 kg while the sim
            # used the value DERIVED from its shape and density (a 5 cm box is
            # 1.68). The derivation runs in the builder and lands here.
            keys.append(String("mass")); editable.append(F_MASS)
            vals.append(Float64(
                L.m.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS]
            ))
            keys.append(String("ipos[0]")); editable.append(-1); vals.append(bd.ipos_x)
            keys.append(String("ipos[1]")); editable.append(-1); vals.append(bd.ipos_y)
            keys.append(String("ipos[2]")); editable.append(-1); vals.append(bd.ipos_z)
            keys.append(String("inertia[0]")); editable.append(-1); vals.append(bd.ixx)
            keys.append(String("inertia[1]")); editable.append(-1); vals.append(bd.iyy)
            keys.append(String("inertia[2]")); editable.append(-1); vals.append(bd.izz)
    elif p.sel_kind == SEL_GEOM:
        var g = p.sel_index
        if g < 0 or g >= len(L.fmd.geoms):
            return
        var gd = L.fmd.geoms[g]
        var bid = gd.body_id
        var wc = Vec3(gd.pos_x, gd.pos_y, gd.pos_z)
        var wq = Quat(gd.quat_w, gd.quat_x, gd.quat_y, gd.quat_z)
        if bid >= 0 and bid < len(positions):
            # WORLD pose: the local pos is what the MJCF says, the world pose
            # is what the user just clicked on.
            wc = positions[bid] + quats[bid].rotate_vec(wc)
            wq = quats[bid] * wq
        keys.append(String("type")); editable.append(-1); vals.append(Float64(gd.geom_type))
        keys.append(String("pos[0]")); editable.append(-1); vals.append(wc.x)
        keys.append(String("pos[1]")); editable.append(-1); vals.append(wc.y)
        keys.append(String("pos[2]")); editable.append(-1); vals.append(wc.z)
        keys.append(String("quat[0]")); editable.append(-1); vals.append(wq.w)
        keys.append(String("quat[1]")); editable.append(-1); vals.append(wq.x)
        keys.append(String("quat[2]")); editable.append(-1); vals.append(wq.y)
        keys.append(String("quat[3]")); editable.append(-1); vals.append(wq.z)
        # ⚠ SIZE IS PER-TYPE. Printing the raw slots would show a capsule's box
        # half-extents (0.5, `GeomData`'s default) as if they were its
        # dimensions — `build_render_fields`' mapping 4 documents the same
        # hazard from the other side.
        var gt = gd.geom_type
        if gt == 3:
            keys.append(String("half_x")); editable.append(F_SIZE_0); vals.append(gd.half_x)
            keys.append(String("half_y")); editable.append(F_SIZE_1); vals.append(gd.half_y)
            keys.append(String("half_z")); editable.append(F_SIZE_2); vals.append(gd.half_z)
        elif gt == 2 or gt == 4:
            keys.append(String("radius")); editable.append(F_SIZE_0); vals.append(gd.radius)
            keys.append(String("half_len")); editable.append(F_SIZE_1); vals.append(gd.half_length)
        else:
            keys.append(String("radius")); editable.append(F_SIZE_0); vals.append(gd.radius)
        keys.append(String("rgba[0]")); editable.append(F_RGBA_R); vals.append(gd.rgba_r)
        keys.append(String("rgba[1]")); editable.append(F_RGBA_G); vals.append(gd.rgba_g)
        keys.append(String("rgba[2]")); editable.append(F_RGBA_B); vals.append(gd.rgba_b)
        keys.append(String("rgba[3]")); editable.append(F_RGBA_A); vals.append(gd.rgba_a)
        keys.append(String("group")); editable.append(-1); vals.append(Float64(gd.group))
        keys.append(String("condim")); editable.append(-1); vals.append(Float64(gd.condim))
        keys.append(String("friction")); editable.append(F_FRICTION); vals.append(gd.friction)
        keys.append(String("margin")); editable.append(-1); vals.append(gd.margin)
        keys.append(String("mass")); editable.append(-1); vals.append(gd.mass)


def _outline_of(
    L: Loaded, p: StudioPanel, positions: List[Vec3], quats: List[Quat]
) -> List[OverlayLine]:
    if p.sel_kind == SEL_GEOM:
        return outline_geom(L.rf, p.sel_index, positions, quats, 1.0,
                            L.mesh_half)
    if p.sel_kind == SEL_BODY:
        return outline_body(L.rf, p.sel_index, positions, quats, 1.0,
                            L.mesh_half)
    return List[OverlayLine]()


# ═══════════════════════════════════════════════════════════════════════════
# the loop
# ═══════════════════════════════════════════════════════════════════════════


def _label_of(names: List[String], i: Int) -> String:
    if i >= 0 and i < len(names):
        return names[i].copy()
    return String("")


def run_studio(
    first: String, drive: Int, scale: Float64, max_frames: Int = 0,
    swap_to: String = String(""),
) raises:
    """`swap_to` is the SMOKE PATH for File > Open.

    ⚠ IT EXISTS BECAUSE A MENU CANNOT BE CLICKED FROM A SCRIPT, and the swap
    is the riskiest code in this file: it detaches a live window, rebuilds
    every container, and adopts the window back. Untested, that is the kind of
    thing that works once by hand and leaks a window forever after. With
    `frames` set, the studio requests the swap at the halfway mark and the run
    ends having exercised the real path — same `PanelOut.open_path`, same
    detach/adopt — rather than a parallel copy of it written for the test.
    """
    print("=" * 70)
    print("physics3d studio")
    print("=" * 70)

    var L = Loaded(first)
    L.describe()

    var panel = StudioPanel(drive, scale, String("mojo_rl/envs"))
    panel.remember(first)

    var renderer = ModelRenderer[RfOnlyModelDef](
        width=1600, height=900, visual_radius_scale=1.0,
        show_velocity=False, title=String("physics3d studio"),
        adopt_rf=Optional[RenderFields](L.rf.copy()),
    )
    renderer.init(None)

    var have_ui = renderer.imgui_init()
    if have_ui:
        # ⚠ THE SCENE VIEWPORT IS BETWEEN THE TWO PANELS. `ui_sidebar_width`
        # reserves the LEFT strip, and the ray-pick below subtracts the RIGHT
        # one itself — an unprojection that ignored either would be biased by
        # half the missing strip, a symptom that reads as a projection bug.
        renderer.set_ui_sidebar_width(Int(SIDEBAR_W))
        renderer.set_show_hud(False)
    else:
        print("  (no ImGui shim — run `pixi run build-imgui` for the UI;")
        print("   the built-in HUD is up instead)")

    var actions = List[Float64]()
    var act = List[Scalar[DT]]()
    var positions = List[Vec3]()
    var quats = List[Quat]()
    var keys = List[String]()
    var vals = List[Float64]()
    var editable = List[Int]()
    var log = EditLog()

    # ⚠ THE SCENE DOCUMENT IS BUILT AROUND WHATEVER WAS OPENED. A plain model
    # becomes an ASSET of a one-instance scene, so a robot and a prop are
    # placed, moved and duplicated by the same machinery — there is no
    # "scene mode". The opened file is never rewritten; the scene is a new
    # document that REFERENCES it.
    var doc = scene_from_base(first)

    var t = 0
    var last_ncon = 0
    var step_ns = 0
    var last_us = Float64(0)
    var t0 = perf_counter_ns()

    while renderer.is_open():
        if renderer.check_quit():
            break
        if max_frames > 0 and t >= max_frames:
            break

        var nbody = L.dims.get_nbody()
        var nact = L.dims.get_nact()
        # Sized here rather than once, because a model SWAP changes them and
        # a stale length is a silent out-of-bounds on the next step.
        while len(actions) < nact:
            actions.append(0.0)
            act.append(Scalar[DT](0))
        while len(positions) < nbody:
            positions.append(Vec3(0.0, 0.0, 0.0))
            quats.append(Quat(1.0, 0.0, 0.0, 0.0))

        if have_ui:
            renderer.imgui_new_frame()

        # ⚠ THE PICK, THE OUTLINE AND THE DRAW MUST SEE THE SAME POSES.
        # Rebuilding these inside the picker would let a click resolve against
        # the pose from before this frame's step, so a fast-moving geom would
        # be selectable only where it WAS.
        for b in range(nbody):
            positions[b] = Vec3(
                Float64(L.d.xpos.data[b * 3 + 0]),
                Float64(L.d.xpos.data[b * 3 + 1]),
                Float64(L.d.xpos.data[b * 3 + 2]),
            )
            quats[b] = Quat(
                Float64(L.d.xquat.data[b * 4 + 3]),
                Float64(L.d.xquat.data[b * 4 + 0]),
                Float64(L.d.xquat.data[b * 4 + 1]),
                Float64(L.d.xquat.data[b * 4 + 2]),
            )

        # ── ray-pick ──────────────────────────────────────────────────────
        # ⚠⚠ `ig_want_mouse()` GATES IT. Without that, every click on a panel
        # also fires a pick THROUGH the panel into the scene behind it — so
        # selecting a row in the Explorer would instantly reselect whatever
        # geom happens to sit under the sidebar.
        var clicked = renderer.take_click()
        if clicked and not (have_ui and ig_want_mouse()):
            var x0 = Float64(renderer.renderer.ui_sidebar_width)
            var right = Float64(RIGHT_W) if have_ui else 0.0
            var vp_w = Float64(renderer.renderer.width) - x0 - right
            var vp_h = Float64(renderer.renderer.height)
            if vp_w > 1.0 and vp_h > 1.0:
                var cam = renderer.renderer.camera.copy()
                var ray = ray_through_pixel(
                    Float64(renderer.mouse_x()), Float64(renderer.mouse_y()),
                    x0, vp_w, vp_h, cam.eye, cam.target, cam.up, cam.fov,
                )
                var hit = pick_geom(ray, L.rf, positions, quats)
                if hit.geom >= 0:
                    panel.sel_kind = SEL_GEOM
                    panel.sel_index = hit.geom
                else:
                    # ⚠ A MISS CLEARS THE SELECTION. Keeping it is the more
                    # common editor behaviour and the wrong one here: the
                    # inspector would keep describing something the user just
                    # clicked away from, which reads as a stale panel.
                    panel.clear_selection()

        _record(L, panel, positions, quats, keys, vals, editable)
        renderer.set_overlay_lines(_outline_of(L, panel, positions, quats))

        # ── the UI ────────────────────────────────────────────────────────
        var ui = PanelOut()
        if have_ui:
            ui = build_ui(
                panel, L.path, t, last_ncon, MAX_CONTACTS, last_us,
                Float32(renderer.renderer.width),
                Float32(renderer.renderer.height),
                L.fmd.body_names, L.fmd.geom_names, L.fmd.joint_names,
                L.body_parent, L.geom_body, L.joint_body, keys, vals,
                editable, log.can_undo(), log.can_redo(),
            )
        # ⚠ AFTER `build_ui`, WHICH RETURNS A FRESH `PanelOut` — injecting
        # before it would be overwritten and the smoke path would silently
        # never fire, which is exactly the shape of a test that proves
        # nothing.
        if swap_to.byte_length() > 0 and max_frames > 0 \
                and t == max_frames // 2:
            ui.open_path = swap_to
        renderer.set_show_hud(panel.show_hud)
        renderer.set_show_sites(panel.show_sites)
        if ui.quit:
            break
        # ── an inspector edit ─────────────────────────────────────────────
        # ⚠ APPLIED HERE, AFTER `build_ui` AND BEFORE THE STEP. The panel
        # returns a REQUEST precisely so this happens at a defined point: an
        # edit landing between the step and the draw would render a pose that
        # never existed, and one landing inside the panel would need the panel
        # to hold a `Model`, which is what keeps it compiling once.
        if ui.edit_field >= 0 and panel.sel_kind != SEL_NONE:
            var tgt = TARGET_GEOM if panel.sel_kind == SEL_GEOM \
                else TARGET_BODY
            var e = Edit(tgt, panel.sel_index, ui.edit_field, ui.edit_value)
            log.push(e)
            apply_edit(L.fmd, L.m, e)
            # ⚠ THE RENDERER READS `RenderFields`, NOT THE RECORD, so a
            # colour or a size change is invisible until `rf` is rebuilt.
            # Cheap (no re-parse, no mesh load) and it keeps "what you see" and
            # "what you edited" the same thing.
            L.rf = build_render_fields(L.fmd, L.flat, L.base_dir)
            L.mesh_half = L._measure_meshes()
            renderer.set_render_fields(L.rf.copy())
            if needs_rebuild(e):
                # Mass changes the DERIVED inertia and invweight0, so the
                # record is authoritative and the live model must be rebuilt
                # rather than patched. See `needs_rebuild`.
                build_model_runtime[DT](L.fmd, L.dims, L.m)

        # ── save / export ─────────────────────────────────────────────────
        # ⚠ TWO DIFFERENT FILES, AND THE DIFFERENCE MATTERS. The scene
        # DOCUMENT keeps the composition — the asset table and the instance
        # list — so it can be reopened and re-edited. The flattened EXPORT
        # keeps what is being SIMULATED, including the fast-path edits the
        # document has nowhere to store. Offering only one would silently lose
        # something either way.
        if panel.want_save != 0:
            var which = panel.want_save
            panel.want_save = 0
            var out_path = L.path + (
                ".scene.xml" if which == 1 else ".flat.xml"
            )
            try:
                var body = doc.to_mjcf(String("scene")) if which == 1 \
                    else export_flat_mjcf(L.fmd, String("exported"))
                var wf = open(out_path, "w")
                wf.write(body)
                wf.close()
                print("  wrote", out_path)
            except e:
                # ⚠ A REFUSED EXPORT IS AN EXPECTED OUTCOME, not a crash: the
                # flattened writer raises rather than emitting a file that
                # loads and is a DIFFERENT model (a dropped <tendon>, say).
                # The message names the sections.
                print("  save failed:", e)

        if panel.want_undo != 0:
            # ⚠ UNDO IS A REPLAY FROM A FRESH PARSE, not an inverse — see
            # `EditLog`. It costs 0.2-14 ms, which is a click.
            if panel.want_undo == 1:
                log.undo()
            else:
                log.redo()
            panel.want_undo = 0
            try:
                var fresh = Loaded(L.path)
                log.replay(fresh.fmd, fresh.m)
                build_model_runtime[DT](fresh.fmd, fresh.dims, fresh.m)
                fresh.rf = build_render_fields(
                    fresh.fmd, fresh.flat, fresh.base_dir
                )
                fresh.mesh_half = fresh._measure_meshes()
                renderer.set_render_fields(fresh.rf.copy())
                L = fresh^
                positions.clear()
                quats.clear()
            except e:
                print("  undo failed:", e)

        # ── props: a STRUCTURAL edit, so the whole model is rebuilt ───────
        if ui.add_prop >= 0 or ui.dup_prop or ui.del_prop:
            var changed = True
            if ui.add_prop >= 0:
                # In front of the camera, at a size that reads on screen.
                var cam2 = renderer.renderer.camera.copy()
                var fwd = (cam2.target - cam2.eye).normalized()
                var at = cam2.target + fwd * 0.0
                _ = doc.add_prop(ui.add_prop, 0.05, 0.05, 0.05,
                                 at.x, at.y, at.z + 0.3)
            elif ui.dup_prop and panel.sel_kind == SEL_BODY:
                _ = doc.duplicate_prop(
                    _label_of(L.fmd.body_names, panel.sel_index)
                )
            elif ui.del_prop and panel.sel_kind == SEL_BODY:
                doc.remove_prop(_label_of(L.fmd.body_names, panel.sel_index))
            else:
                changed = False
            if changed:
                try:
                    var nxt = Loaded(
                        L.path, doc.to_mjcf(String("scene")), L.base_dir
                    )
                    # ⚠ THE SELECTION CANNOT SURVIVE. Indices shift when a
                    # body is added or removed, so a kept index names a
                    # DIFFERENT part — and the outline would sit on it,
                    # confidently.
                    panel.clear_selection()
                    renderer.set_render_fields(nxt.rf.copy())
                    L = nxt^
                    positions.clear()
                    quats.clear()
                    log = EditLog()
                except e:
                    print("  prop edit failed:", e)

        if ui.reset:
            L.reset()
            t = 0
            step_ns = 0
        if ui.reframe:
            renderer.request_free_camera()

        # ── step ──────────────────────────────────────────────────────────
        # ⚠ BOTH PAUSES ARE HONOURED — the panel's and `Renderer3D`'s SPACE
        # binding. Honouring one leaves the other toggling a flag nothing
        # reads, which is a bug `viewer_core` documents having shipped.
        var frozen = panel.paused or renderer.paused()
        if (not frozen) or ui.step_once:
            var s0 = perf_counter_ns()
            for a in range(nact):
                if panel.drive == DRIVE_ZERO:
                    actions[a] = 0.0
                elif panel.drive == DRIVE_RANDOM:
                    actions[a] = (random_float64() * 2.0 - 1.0) \
                        * Float64(panel.scale)
                else:
                    var ph = Float64(t) * 0.02 + Float64(a) * 0.7
                    actions[a] = Float64(panel.scale) * sin(ph)
            if nact > 0:
                apply_actions_fields[DT](L.sf, L.d, actions, act,
                                         L.fmd.timestep)
            L.integ.step["cpu"](L.d, L.m)
            step_ns += Int(perf_counter_ns() - s0)
            t += 1
            last_us = Float64(step_ns) / 1000.0 / Float64(t)
            last_ncon = Int(Float64(L.d.meta.data[META_IDX_NUM_CONTACTS]))

        var hud = List[String]()
        hud.append(String("file: ", L.path))
        hud.append(String("contacts: ", last_ncon, " / ", MAX_CONTACTS))
        renderer.set_hud_extra(hud)
        renderer.render(positions, quats)

        # ── THE MODEL SWAP, at the END of the frame ───────────────────────
        # ⚠⚠ AFTER `render`, NEVER BEFORE. `imgui_new_frame` opened an ImGui
        # frame that only `end_frame` (inside `render`) closes; tearing the
        # renderer down between them leaves that frame open and the NEXT
        # `NewFrame` asserts. `viewer_core` documents the same constraint for
        # its task switch, and it is the reason the switch cannot simply
        # `break` out of the middle of the loop.
        if ui.open_path.byte_length() > 0 and ui.open_path != L.path:
            print("  opening", ui.open_path)
            try:
                var nxt = Loaded(ui.open_path)
                nxt.describe()
                # ⚠ THE WINDOW CROSSES THE GAP. `detach` gives up this model's
                # GPU caches and hands the window on; the next renderer ADOPTS
                # it, so the swap keeps the monitor, position, size and ImGui
                # state. Expect the CAMERA to reframe — it is the model's —
                # and nothing else to move.
                #
                # ⚠ ADOPTING TAKES OWNERSHIP: exactly one party can free the
                # window at any time, so the handoff goes straight into the
                # new renderer with nothing in between that can raise.
                var handoff = renderer.detach()
                var nr = ModelRenderer[RfOnlyModelDef](
                    width=1600, height=900, visual_radius_scale=1.0,
                    show_velocity=False,
                    title=String("physics3d studio"),
                    adopt_rf=Optional[RenderFields](nxt.rf.copy()),
                )
                nr.init(Optional(handoff^))
                renderer = nr^
                L = nxt^
                if have_ui:
                    renderer.set_ui_sidebar_width(Int(SIDEBAR_W))
                    renderer.set_show_hud(panel.show_hud)
                # ⚠ THE SELECTION DOES NOT SURVIVE. An index into the old
                # model's geoms names a DIFFERENT shape in the new one, and a
                # yellow outline around the wrong part is worse than none.
                panel.clear_selection()
                panel.remember(L.path)
                positions.clear()
                quats.clear()
                actions.clear()
                act.clear()
                t = 0
                step_ns = 0
                last_ncon = 0
            except e:
                # ⚠ A BAD FILE MUST NOT END THE SESSION. The whole point of
                # File > Open is trying models; one that fails to parse should
                # leave the current one running and say why.
                print("  FAILED to open", ui.open_path, "—", e)
                print("  (keeping", L.path, ")")

    renderer.close()
    var wall_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    print("  stepped", t, "times on", L.path)
    if t > 0:
        print("  step cost", last_us, "us")
    print("  wall", wall_ms, "ms")


def main() raises:
    seed(0)
    var args = argv()
    var path = String("mojo_rl/envs/humanoid/assets/humanoid.xml")
    if len(args) > 1:
        path = String(args[1])
    else:
        print("no model given — opening", path)
        print("use File > Open in the window to load another")
    var drive = parse_drive(String(args[2])) if len(args) > 2 else DRIVE_ZERO
    var scale = Float64(1.0)
    if len(args) > 3:
        try:
            scale = Float64(String(args[3]))
        except:
            print("bad scale, using 1.0")
    var frames = 0
    if len(args) > 4:
        try:
            frames = Int(String(args[4]))
        except:
            print("bad frame count, running until the window closes")
    var swap_to = String("")
    if len(args) > 5:
        swap_to = String(args[5])
    run_studio(path, drive, scale, frames, swap_to)
