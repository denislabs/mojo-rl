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
from mojo_rl.physics3d.types import ConeType, IntegratorType
from mojo_rl.physics3d.studio.stepping import (
    StudioIntegPyr, StudioIntegEll, studio_cone_of, studio_solver_warning,
    studio_condim_warning,
    StudioImpFastPyr, StudioImpFastEll, studio_uses_implicit,
    StudioRk4Pyr, studio_integrator_of,
    studio_integrator_warning,
)
from mojo_rl.physics3d.model.model_renderer import ModelRenderer, OverlayLine
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.dynamics.pose_transmission import (
    apply_pose_transmission,
)
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, MODEL_MESH_META_SIZE, MODEL_GEOM_SIZE,
    GEOM_IDX_MESH_ID, MAX_GPU_MESHES, MODEL_BODY_SIZE, BODY_IDX_MASS,
)
from mojo_rl.physics3d.studio.scene import SceneDoc, scene_from_base
from mojo_rl.physics3d.studio.writer import to_mjcf as export_flat_mjcf
from mojo_rl.physics3d.studio import (
    Ray, ray_through_pixel, pick_geom, outline_geom, outline_body,
    StudioPanel, PanelOut, build_ui, SIDEBAR_W,
)
from mojo_rl.physics3d.studio.panel import SEL_BODY, SEL_GEOM, SEL_NONE
from mojo_rl.physics3d.studio.validate import (
    Diagnostic, validate_all, worst_severity, count_at, format_diagnostic,
    SEV_ERROR, SEV_WARN,
)
from mojo_rl.physics3d.studio.structure import (
    delete_body, delete_geom, add_body, add_joint, rename_element,
    reparent_body,
)
from mojo_rl.physics3d.studio.remap import (
    remap_state, pose_snapshot, apply_pose_snapshot,
)
from mojo_rl.physics3d.studio.history import History, edit_key
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.kinematics.mocap import reset_mocap_from_model
from mojo_rl.physics3d.studio.edit import (
    Edit, apply_edit, apply_edit_to_document, needs_rebuild, field_name,
    TARGET_GEOM, TARGET_BODY,
    F_POS_X, F_POS_Y, F_POS_Z, F_SIZE_0, F_SIZE_1, F_SIZE_2,
    F_RGBA_R, F_RGBA_G, F_RGBA_B, F_RGBA_A, F_FRICTION, F_MASS,
)
from mojo_rl.render.imgui import (
    ig_want_mouse, gz_begin_frame, gz_set_rect, gz_set_orthographic,
    gz_set_size, gz_manipulate, gz_is_over, gz_is_using,
    GZ_TRANSLATE, GZ_ROTATE, GZ_LOCAL, GZ_WORLD,
)
from mojo_rl.render.gpu_types import perspective_projection
from mojo_rl.physics3d.studio.gizmo import (
    Frame, frame_to_cm, mat4_to_cm, edit_frame, frame_drift, gizmo_edits,
    gizmo_mode_name, GIZMO_OFF, GIZMO_MOVE, GIZMO_TURN,
)
from mojo_rl.physics3d.studio.mesh_bounds import (
    empty_half_extents, measure_geom_from_file, biggest_half_extent,
)

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
    # ⚠⚠ TWO INTEGRATORS, AND THE MODEL PICKS. `EulerIntegrator`'s own
    # defaults are ELLIPTIC + "pgs"; MuJoCo 3.10.0's are PYRAMIDAL + NEWTON
    # (measured: `m.opt.cone` reads 0 and `m.opt.solver` reads 2 on a model
    # whose `<option>` says nothing). The studio was taking the parameter
    # defaults, so EVERY model opened here simulated with a friction cone and
    # a solver the reference does not use — including the 33 Menagerie scenes
    # that are pyramidal precisely because they say nothing.
    #
    # ⚠ CONE IS WHAT VARIES; SOLVER DOES NOT. Measured across 114 models —
    # all 57 Menagerie scenes and all 57 loadable in-repo ones — the solver is
    # NEWTON in 112 and PGS in 2, while the cone splits 67 pyramidal / 45
    # elliptic. Two instantiations cover the split; a model asking for
    # anything else gets a WARNING naming what it asked for and what it got,
    # rather than a silent substitution.
    #
    # Both are constructed because `EulerIntegrator` owns its scratch and the
    # choice is only known after the parse. The waste is one solver workspace
    # (~440 KB at ms_human_700's nv=85), which is not worth an `Optional`
    # dance in the step loop.
    var integ_pyr: StudioIntegPyr
    var integ_ell: StudioIntegEll
    var imp_pyr: StudioImpFastPyr
    var imp_ell: StudioImpFastEll
    var rk4_pyr: StudioRk4Pyr
    """⚠⚠ AND THE MODEL ASKED FOR `RK4` AND WE WERE RUNNING EULER — 14 of the
    131 models in this tree, including ant, hopper, walker2d, swimmer, both
    humanoids and both inverted pendulums. It was warned about and it was
    still wrong: on `bitcraze_crazyflie_2` the substitution is worth 9.200e-06
    after ONE step against MuJoCo, and stepping RK4 takes that to 3.314e-13.
    In free flight the two differ by a clean factor of two — Euler moves
    `a*dt^2`, RK4 moves half of it — which is what the residual measured."""
    var use_implicit: Bool
    """⚠⚠ THE MODEL ASKED FOR `implicitfast` AND WE WERE RUNNING EULER.
    `spot` and `g1` both declare it, and both have `dof_damping` 0 — their
    only damping is actuator `kv`, which explicit Euler integrates unstably:
    spot flew to 18 m instead of standing at 0.65. Honouring `<option
    integrator=>` is not a refinement here, it is the difference between the
    robot the file describes and a different one."""
    var cone_used: Int
    """`ConeType.PYRAMIDAL` / `ELLIPTIC` — which of the two is stepping."""
    var integ_used: Int
    """`IntegratorType.EULER` / `IMPLICITFAST` / `RK4` — which one is stepping.
    ⚠ FROM `studio_integrator_of`, never re-derived here: the selection and
    the branch that acts on it have to come from one place."""
    var rf: RenderFields
    var flat: String
    var base_dir: String
    var body_parent: List[Int]
    var geom_body: List[Int]
    var joint_body: List[Int]
    var mesh_half: List[Float64]
    """Real half-extents per geom, 3 each — see `studio/mesh_bounds`.

    ⚠ MUTATED AFTER LOAD. The file pass is lazy, so a geom's entry can be 0
    ("not measured yet") until it is first selected."""
    var biggest_half: Float64
    """The largest half-extent any geom measured — the marker of last resort
    for a geom no pass could measure. ⚠ IT IS A SCALE, NOT A BOUND."""
    var dirty: Bool
    """Has the document changed since it was loaded?

    ⚠ THE SCENE FILE READS THIS. A scene REFERENCES its base model by path, so
    a scene written while the robot has been edited would reopen as the
    ORIGINAL robot — the one silent loss `File > Save edited model` does not
    cover. See `SceneDoc.retarget_asset`."""

    var diags: List[Diagnostic]
    """What is wrong with this model, refreshed on load and after a rebuild.

    ⚠ HELD ON `Loaded`, NOT RECOMPUTED PER FRAME. `validate_model` walks every
    body against every joint; at ms_human_700's 81 bodies that is cheap once
    and wasteful sixty times a second. The list changes only when the model
    does, which is exactly where it is refreshed."""

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

    var n_mocap: Int
    """Bodies whose pose is an EXTERNAL input, seeded from the XML at load.

    ⚠ COUNTED, so `describe()` can say so. A model with a mocap body has a
    part that no joint moves and no `qpos` describes; that is worth one line
    of output rather than a mystery about why one geom ignores the sim."""

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
        self.sf = spec_fields_runtime[DT](self.fmd, self.dims, self.m)
        self.d = Data[DT, DynDims, 1](self.dims)
        # ⚠⚠ MOCAP BODIES OTHERWISE SIT AT THE WORLD ORIGIN. `Data` allocates
        # `mocap_pos` zeroed and `forward_kinematics` SKIPS mocap bodies by
        # design — their pose is an external input, which an env facade
        # supplies and a tool does not. Without this, so_arm101's `target`
        # sphere is drawn, picked and outlined at (0,0,0) while `m.bodies`
        # says (0.25, 0, 0.2), and nothing raises. `mj_resetData` seeds them
        # from the XML frame for exactly this reason.
        self.n_mocap = reset_mocap_from_model[DT, DynDims, 1](self.m, self.d)
        self.integ_pyr = StudioIntegPyr(self.dims)
        self.integ_ell = StudioIntegEll(self.dims)
        self.imp_pyr = StudioImpFastPyr(self.dims)
        self.imp_ell = StudioImpFastEll(self.dims)
        self.rk4_pyr = StudioRk4Pyr(self.dims)
        self.cone_used = studio_cone_of(self.fmd)
        self.integ_used = studio_integrator_of(self.fmd)
        self.use_implicit = studio_uses_implicit(self.fmd)
        var solver_note = studio_solver_warning(self.fmd)
        if solver_note.byte_length() > 0:
            print(solver_note)
        var integ_note = studio_integrator_warning(self.fmd)
        if integ_note.byte_length() > 0:
            print(integ_note)
        var condim_note = studio_condim_warning(self.fmd)
        if condim_note.byte_length() > 0:
            print(condim_note)
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
        self.biggest_half = 0.0
        self.hull_verts = 0
        # ⚠ EMPTY FIRST, FILLED BELOW — every field must be initialised before
        # the first method call on `self`.
        self.diags = List[Diagnostic]()
        self.dirty = False
        # ⚠ NOTHING IS MEASURED HERE. Every extent is read from the mesh
        # FILE, lazily, when the geom is first selected — measuring all of
        # them at open cost ~900 ms on go2 and panda to fill slots the outline
        # reads one of. See `studio/mesh_bounds`.
        self.mesh_half = empty_half_extents(len(self.fmd.geoms))
        self.biggest_half = 0.0
        for mi in range(MAX_GPU_MESHES):
            self.hull_verts += Int(Float64(
                self.m.mesh_meta.data[mi * MODEL_MESH_META_SIZE + 1]
            ))

        # ⚠ THE DOCUMENT CHECKS RUN ON THE **EXPANDED** TEXT, for the same
        # reason `build_render_fields` does: a scene file's own text names
        # none of the bodies an `<attach>` brought in, so `dangling_references`
        # on the source would call every reference in the robot dangling.
        self.diags = validate_all(self.flat, self.fmd, self.m)

        self.reset()


    def revalidate(mut self) raises:
        """Recompute the diagnostics after the model changed.

        ⚠ CALLED WHERE THE MODEL IS REBUILT, NOT WHERE AN EDIT IS PUSHED. A
        fast-path edit writes into both the record and the live model, and a
        size or a mass edit can be the thing that makes a body massless — so
        the marker has to follow the BUILD, which is the point where the two
        representations agree again.
        """
        self.diags = validate_all(self.flat, self.fmd, self.m)


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

    def measure_selection(mut self, kind: Int, index: Int) raises:
        """Measure whatever the selection will outline — one geom, or a body's.

        ⚠ THE BODY CASE IS NOT ONE GEOM. `outline_body` draws every visible
        geom of the body, so measuring only the first would leave the rest at
        the fallback and the cage would be half right — the exact half-correct
        state that reads as a rendering glitch rather than a missing measure.
        """
        if kind == SEL_GEOM:
            measure_geom_from_file(self.rf, index, self.mesh_half,
                                   self.biggest_half)
        elif kind == SEL_BODY:
            for g in range(len(self.fmd.geoms)):
                if self.fmd.geoms[g].body_id == index:
                    measure_geom_from_file(self.rf, g, self.mesh_half,
                                           self.biggest_half)
        # ⚠ AFTER, NOT BEFORE. The marker scale is the max of what has been
        # measured, so it only improves once something has been.
        self.biggest_half = biggest_half_extent(self.mesh_half)

    def reset(mut self):
        """The reference pose, and zero velocity."""
        for i in range(self.dims.get_nq()):
            self.d.qpos.data[i] = self.sf.qpos0.data[i]
        for i in range(self.dims.get_nv()):
            self.d.qvel.data[i] = Scalar[DT](0)
        # ⚠ AND THE MOCAP TARGETS, which `qpos` does not describe. Resetting
        # without this leaves a target wherever the last step left it while
        # every jointed body snaps back — a pose that is half reset.
        _ = reset_mocap_from_model[DT, DynDims, 1](self.m, self.d)

    def describe(self) raises:
        print("  ", self.path)
        if self.n_mocap > 0:
            # ⚠ SAID OUT LOUD. A mocap body is moved by neither a joint nor
            # the solver — its pose is an external input — so a user watching
            # it ignore gravity and every actuator has no way to tell that
            # from a bug. It is seeded from the XML frame at load, which is
            # what `mj_resetData` does.
            print("    ", self.n_mocap,
                  "mocap body(ies) — pose is an EXTERNAL input, seeded from"
                  " the XML frame; no joint moves them")
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
        # ⚠ PRINTED, NOT ONLY PANELLED. The window's Problems tab is the
        # place to WORK on these; the banner is what a headless run, a smoke
        # frame count and a bug report all show, and a diagnostic nobody can
        # paste into a message is one that gets described instead of quoted.
        var nerr = count_at(self.diags, SEV_ERROR)
        var nwarn = count_at(self.diags, SEV_WARN)
        if len(self.diags) > 0:
            print("    problems:", nerr, "error(s),", nwarn, "warning(s),",
                  len(self.diags) - nerr - nwarn, "info")
            for d in self.diags:
                if d.severity >= SEV_WARN:
                    print("      ", format_diagnostic(d))


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
    mut L: Loaded, p: StudioPanel, positions: List[Vec3], quats: List[Quat]
) raises -> List[OverlayLine]:
    # ⚠ MEASURE FIRST, AND IT IS CHEAP AFTER THE FIRST FRAME. A geom whose
    # mesh has no loaded collision hull is measured from its FILE the first
    # time it is selected; `measure_geom_from_file` returns immediately once
    # the entry is non-zero, so this costs a bounds check per frame after
    # that. Doing it at model-open instead added ~900 ms to go2 and panda.
    L.measure_selection(p.sel_kind, p.sel_index)
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


def _f4(v: Float64) -> String:
    """Four decimals, for a hint line. `String(v)` prints 17 digits."""
    var scaled = Int(v * 10000.0 + 0.5)
    var whole = scaled // 10000
    var frac = String(scaled % 10000)
    while frac.byte_length() < 4:
        frac = String("0") + frac
    return String(whole, ".", frac)


def _label_of(names: List[String], i: Int) -> String:
    if i >= 0 and i < len(names):
        return names[i].copy()
    return String("")


def _sel_label(L: Loaded, p: StudioPanel) -> String:
    """What the selection is called, for the Undo menu. `#index` when unnamed.

    ⚠ MOST GEOMS IN THIS TREE HAVE NO NAME, and "size on " with nothing after
    it reads as a bug in the menu rather than as an unnamed element.
    """
    var n = _label_of(L.fmd.geom_names, p.sel_index) \
        if p.sel_kind == SEL_GEOM else _label_of(L.fmd.body_names, p.sel_index)
    if n.byte_length() > 0:
        return String("'", n, "'")
    return String("#", p.sel_index)


def _sync_pose(
    mut L: Loaded, mut positions: List[Vec3], mut quats: List[Quat]
) raises:
    """Resize and refill the draw poses after the model was REPLACED.

    ⚠⚠ `positions.clear()` ALONE CRASHES THE SAME FRAME. The lists are filled
    at the TOP of the loop and read by the draw at the BOTTOM; a replacement
    in between leaves the draw reading an empty list, and `ModelRenderer`
    indexes `positions[1]` for the tracking camera — "index 1 is out of
    bounds, valid range is 0 to -1", from a frame that had already done the
    right thing everywhere else.

    ⚠ AND FK IS RUN, NOT SKIPPED. A freshly built `Data` has `xpos` all zero
    until something integrates, so refilling without it draws every body at
    the origin for one frame — which looks exactly like the model collapsing,
    and would be blamed on the edit.
    """
    var nbody = L.dims.get_nbody()
    forward_kinematics["cpu", DT, DynDims, 1](L.d, L.m)
    positions.clear()
    quats.clear()
    for b in range(nbody):
        positions.append(Vec3(
            Float64(L.d.xpos.data[b * 3 + 0]),
            Float64(L.d.xpos.data[b * 3 + 1]),
            Float64(L.d.xpos.data[b * 3 + 2]),
        ))
        quats.append(Quat(
            Float64(L.d.xquat.data[b * 4 + 3]),
            Float64(L.d.xquat.data[b * 4 + 0]),
            Float64(L.d.xquat.data[b * 4 + 1]),
            Float64(L.d.xquat.data[b * 4 + 2]),
        ))


def run_studio(
    first: String, drive: Int, scale: Float64, max_frames: Int = 0,
    swap_to: String = String(""), delete_body_named: String = String(""),
    smoke_add: String = String(""), smoke_undo: Bool = False,
    smoke_gizmo: Bool = False,
) raises:
    """`swap_to` is the SMOKE PATH for File > Open, `delete_body_named` for
    the structural delete, `smoke_undo` for undo/redo across one.

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
    var held_reported = False
    var browser_reported = False
    var frame = 0
    var smoke_nbody0 = 0
    var smoke_nbody1 = 0
    var positions = List[Vec3]()
    var quats = List[Quat]()
    var keys = List[String]()
    var vals = List[Float64]()
    var editable = List[Int]()

    # ⚠⚠ ONE UNDO STACK FOR BOTH TIERS — V2.9. The `EditLog` this replaced
    # replayed dims-preserving edits onto a fresh parse, which cannot express
    # a delete, so every structural edit RESET it: the most destructive
    # operation here had no undo at all. `Loaded` is a pure function of
    # (document, base_dir), so a document snapshot restores either kind.
    var hist = History()

    # ⚠ THE SCENE DOCUMENT IS BUILT AROUND WHATEVER WAS OPENED. A plain model
    # becomes an ASSET of a one-instance scene, so a robot and a prop are
    # placed, moved and duplicated by the same machinery — there is no
    # "scene mode". The opened file is never rewritten; the scene is a new
    # document that REFERENCES it.
    var doc = scene_from_base(first)
    # ⚠ THE STACK IS SEEDED WITH THE FILE AS OPENED, not left empty. An empty
    # stack makes the FIRST edit the floor — undo it and there is nowhere to
    # go, so the state the user started from is the one state unreachable.
    hist.push(L.flat, L.base_dir, doc, String("opened"))

    var t = 0
    var last_ncon = 0
    var step_ns = 0
    var last_us = Float64(0)
    var t0 = perf_counter_ns()

    while renderer.is_open():
        if renderer.check_quit():
            break
        # ⚠⚠ THE BOUND IS ON FRAMES, NOT ON STEPS. `t` counts STEPS, and a
        # sim that is paused — or HELD because the model has an error — never
        # advances it, so a headless run with `frames` set spun forever. The
        # hold is new; the pause has had this since S0.
        frame += 1
        if max_frames > 0 and (t >= max_frames or frame > max_frames * 4):
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

        # ⚠⚠ ONE VIEWPORT, READ BY THE PICK AND BY THE GIZMO. It is the
        # region `Renderer3D` actually draws the scene into —
        # `[ui_sidebar_width, width]`, FULL height (`renderer3d.mojo`'s
        # `set_gpu_viewport`, and `scene_width()` is what the camera's aspect
        # is built from). The right-hand panel is drawn OVER the scene rather
        # than inset from it.
        #
        # ⚠⚠ THE PICK USED TO SUBTRACT `RIGHT_W` HERE AND THAT WAS A BUG, not
        # a style difference. `ray_through_pixel` scales the pixel by the
        # viewport it is told about, so a viewport 340 px narrower than the
        # projection's biases EVERY ray by a constant angle: measured on a
        # 1600x900 window with a 45 degree camera, a click at the exact
        # centre of the drawn scene produced a ray 8.9 degrees off the camera
        # axis, missing by 0.47 m at 3 m. It reads as "picking is off to one
        # side", which is easy to blame on the unprojection. Deriving both
        # numbers here is what stops the gizmo and the pick disagreeing about
        # where the world is.
        var vp_x0 = Float64(renderer.renderer.ui_sidebar_width)
        var vp_w = Float64(renderer.renderer.width) - vp_x0
        var vp_h = Float64(renderer.renderer.height)
        if have_ui:
            renderer.imgui_new_frame()
            # ⚠⚠ THE GIZMO'S FRAME OPENS AFTER ImGui's AND BEFORE ANY WIDGET.
            # `ImGuizmo::BeginFrame` pushes a full-screen `NoInputs` window
            # to draw into; opened after the panels it would sit ON TOP of
            # them and the gizmo would draw over the sidebar.
            gz_begin_frame()
            # ⚠⚠ THE RECT IS THE RENDERER'S SCENE VIEWPORT, WHICH IS
            # `[ui_sidebar_width, width]` — the LEFT strip only. The right
            # panel is drawn OVER the scene rather than inset from it, so
            # subtracting it here (as the ray-pick does) would shift every
            # gizmo handle by half its width against the geometry it sits
            # on. The projection is the authority on where the scene lands,
            # and it is built from `scene_width()`.
            gz_set_rect(Float32(vp_x0), 0.0, Float32(vp_w), Float32(vp_h))
            gz_set_orthographic(False)

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
            if vp_w > 1.0 and vp_h > 1.0:
                var cam = renderer.renderer.camera.copy()
                var ray = ray_through_pixel(
                    Float64(renderer.mouse_x()), Float64(renderer.mouse_y()),
                    vp_x0, vp_w, vp_h, cam.eye, cam.target, cam.up, cam.fov,
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
        # ⚠⚠ THE GIZMO EDITS THE FRAME MJCF STORES, NOT THE POSE ON SCREEN,
        # and for a JOINTED BODY those are two different places. `xpos`
        # carries the joint transform on top of the body's `pos=`; a gizmo
        # dragging the drawn pose would be asking for an edit the document
        # cannot express. So it stays on the frame it edits and the panel
        # SAYS when the two have separated, rather than leaving the user
        # looking at a handle floating beside the part.
        var gizmo_hint = String("")
        if panel.gizmo_mode != GIZMO_OFF and panel.sel_kind == SEL_BODY \
                and panel.sel_index > 0:
            var drift = frame_drift(L.fmd, positions, quats, L.body_parent,
                                    TARGET_BODY, panel.sel_index)
            if drift > 1e-6:
                gizmo_hint = String(
                    "on the model frame, ", _f4(drift),
                    " m from the pose on screen\n(this body's joints have"
                    " moved it — reset to line them up)"
                )
        var ui = PanelOut()
        if have_ui:
            ui = build_ui(
                panel, L.path, t, last_ncon, MAX_CONTACTS, last_us,
                Float32(renderer.renderer.width),
                Float32(renderer.renderer.height),
                L.fmd.body_names, L.fmd.geom_names, L.fmd.joint_names,
                L.body_parent, L.geom_body, L.joint_body, keys, vals,
                editable, L.diags, hist.can_undo(), hist.can_redo(),
                hist.undo_label(), hist.redo_label(), gizmo_hint,
            )
        # ⚠ AFTER `build_ui`, WHICH RETURNS A FRESH `PanelOut` — injecting
        # before it would be overwritten and the smoke path would silently
        # never fire, which is exactly the shape of a test that proves
        # nothing.
        # ⚠ THE BROWSER IS DRAWN FIRST, for the quarter of the run before
        # the swap fires. `test_browser_sort` gates the ORDER and the two
        # derived columns; what it cannot reach is the ImGui call sequence —
        # nested `ig_columns` inside a scroll child, a selectable in column 0,
        # a pushed id per row. Those fail as an assert in the shim, not as a
        # wrong string, so the only way to cover them is to draw the window.
        if swap_to.byte_length() > 0 and max_frames > 0 \
                and frame >= max_frames // 4 and frame < max_frames // 2:
            panel.browser_open = True
        # ⚠ REPORTED, NOT ASSUMED. "did not crash" reads identically on a
        # browser that enumerated nothing, so the smoke prints the row count
        # once — the one number that says the listing ran.
        # ⚠ ON THE FIRST FRAME IT IS AVAILABLE, not on the frame the browser
        # is opened. `browser_open` is set below, AFTER `build_ui` has already
        # run for this frame, so the count only exists from the NEXT one — a
        # print keyed to the opening frame fires while the field is still -1
        # and reports nothing, which is how this was first written.
        if max_frames > 0 and ui.browser_rows >= 0 and not browser_reported:
            browser_reported = True
            print("  smoke: browser listed", ui.browser_rows, "row(s) in",
                  panel.browser_dir)
        if swap_to.byte_length() > 0 and max_frames > 0 \
                and frame == max_frames // 2:
            panel.browser_open = False
            ui.open_path = swap_to
        # ⚠ THE SMOKE PATH FOR THE STRUCTURAL DELETE, and it goes through the
        # SAME `PanelOut.del_element` a click sets — including the selection,
        # which is what names the victim. A test that called `delete_body`
        # directly would prove the library works and nothing about the wiring:
        # the selection lookup, the rebuild, the pose remap and the renderer
        # handoff are all here, not there.
        if delete_body_named.byte_length() > 0 and max_frames > 0 \
                and frame == max_frames // 3:
            var vi = -1
            for bi in range(len(L.fmd.body_names)):
                if L.fmd.body_names[bi] == delete_body_named:
                    vi = bi
            if vi >= 0:
                panel.sel_kind = SEL_BODY
                panel.sel_index = vi
                ui.del_element = True
            else:
                print("  smoke: no body named", delete_body_named)
            # ⚠ CAPTURED BEFORE THE DELETE LANDS, from the LIVE model rather
            # than from the file: an expected value re-derived later would
            # move with whatever bug it was meant to catch.
            smoke_nbody0 = len(L.fmd.bodies) + 1
        # ⚠ AND THE SAVE, in the same run. A structural edit whose result
        # cannot be written back is not an edit anyone can use, and the file
        # this writes is exactly what `test_structural_edit` hands to MuJoCo.
        # ⚠ NOT WHEN THE UNDO SMOKE IS DRIVING — it needs the same frames.
        if delete_body_named.byte_length() > 0 and max_frames > 0 \
                and not smoke_undo \
                and frame == (2 * max_frames) // 3 and panel.want_save == 0:
            panel.want_save = 3
        # ── the UNDO smoke — V2.9 ─────────────────────────────────────────
        # ⚠⚠ THIS DRIVES `PanelOut`/`StudioPanel` EXACTLY AS THE MENU DOES.
        # `test_undo_history` gates the STACK; what it cannot reach is the
        # wiring — that the rebuild uses the entry's own `base_dir`, that the
        # pose is remapped, that the renderer is handed the new
        # `RenderFields`, that `SceneDoc` comes back. Every one of those lives
        # here, and a library test would have proved none of them.
        if smoke_undo and delete_body_named.byte_length() > 0 \
                and max_frames > 0:
            if frame == max_frames // 2:
                print("  smoke: undo")
                panel.want_undo = 1
            elif frame == (2 * max_frames) // 3:
                # ⚠ PRINTED, AND THE EXPECTED VALUE PRINTED BESIDE IT. A
                # smoke that logged only the outcome would read the same
                # whether the undo restored the model or never ran.
                print("  smoke: nbody after undo =", len(L.fmd.bodies) + 1,
                      "(as opened:", smoke_nbody0, ")")
                print("  smoke: redo")
                panel.want_undo = 2
            elif frame == (5 * max_frames) // 6:
                print("  smoke: nbody after redo =", len(L.fmd.bodies) + 1,
                      "(after the delete:", smoke_nbody1, ")")
        # ⚠ ADD AND RENAME GO THROUGH THE SAME `PanelOut` FIELDS a click sets,
        # including the name box — `out.new_name` is what the handler reads,
        # so a smoke that set the name anywhere else would test a path the
        # window does not use.
        if smoke_add.byte_length() > 0 and max_frames > 0:
            if frame == max_frames // 6:
                ui.new_name = smoke_add
                ui.add_body_here = True
            elif frame == max_frames // 2:
                var vi2 = -1
                for bi in range(len(L.fmd.body_names)):
                    if L.fmd.body_names[bi] == smoke_add:
                        vi2 = bi
                if vi2 >= 0:
                    panel.sel_kind = SEL_BODY
                    panel.sel_index = vi2
                    ui.new_name = smoke_add + "_r"
                    ui.rename_here = True
                else:
                    print("  smoke: the added body is not there to rename")

        # ⚠ AND THE SCENE, which is a DIFFERENT question: it references the
        # base model by path, so it has to be re-pointed at the edited copy or
        # it reopens as the original robot.
        if delete_body_named.byte_length() > 0 and max_frames > 0 \
                and not smoke_undo \
                and frame == (5 * max_frames) // 6 and panel.want_save == 0:
            panel.want_save = 1
        renderer.set_show_hud(panel.show_hud)
        renderer.set_show_sites(panel.show_sites)
        if ui.quit:
            break
        # ── the transform gizmo — V2.10 ───────────────────────────────────
        # ⚠ DRAWN AFTER THE PANELS AND BEFORE THE EDIT IS APPLIED. It reads
        # `positions`/`quats`, which are this frame's forward kinematics, and
        # it must not read a record the edit below has already changed —
        # otherwise the handle chases the drag by one frame.
        var gz_out = List[Edit]()
        var gz_target = TARGET_GEOM if panel.sel_kind == SEL_GEOM \
            else TARGET_BODY
        var gz_hot = False
        if have_ui and panel.gizmo_mode != GIZMO_OFF:
            # ⚠ THE WORLDBODY HAS NO EDITABLE FRAME. It is body 0, absent
            # from `fmd.bodies`, and its `pos` is not a thing MJCF spells —
            # a gizmo on it would write into the record one before the list.
            var gz_ok = (panel.sel_kind == SEL_GEOM and panel.sel_index >= 0
                         and panel.sel_index < len(L.fmd.geoms)) \
                or (panel.sel_kind == SEL_BODY and panel.sel_index > 0
                    and panel.sel_index - 1 < len(L.fmd.bodies))
            if gz_ok:
                var gcam = renderer.renderer.camera.copy()
                # ⚠⚠ THE **SAME** PROJECTION THE SCENE IS DRAWN WITH.
                # `Camera3D.get_projection_matrix` builds a DIFFERENT one
                # (`Mat4.perspective`); `_build_scene_uniforms` uses this
                # one. Two perspective matrices that differ only in depth
                # convention put the gizmo a few pixels off the geometry —
                # visible, and easy to misread as a picking bug.
                var gview = mat4_to_cm(gcam.get_view_matrix())
                var gproj = mat4_to_cm(perspective_projection(
                    gcam.fov, gcam.aspect, gcam.near, gcam.far
                ))
                var gmat = frame_to_cm(edit_frame(
                    L.fmd, positions, quats, L.body_parent,
                    gz_target, panel.sel_index,
                ))
                var gop = GZ_TRANSLATE if panel.gizmo_mode == GIZMO_MOVE \
                    else GZ_ROTATE
                var gmode = GZ_WORLD if panel.gizmo_world else GZ_LOCAL
                if gz_manipulate(gview, gproj, gop, gmode, gmat,
                                 panel.gizmo_snap):
                    # ⚠ THE SCALE IS THE CAMERA DISTANCE, and it is what the
                    # noise floor is measured against: one pixel of drag is
                    # worth about `2*dist*tan(fov/2)/height` metres, so the
                    # threshold has to know how far away we are or it means
                    # something different on every model.
                    gz_out = gizmo_edits(
                        L.fmd, positions, quats, L.body_parent,
                        gz_target, panel.sel_index, panel.gizmo_mode, gmat,
                        (gcam.eye - gcam.target).length(),
                    )
                gz_hot = gz_is_over() or gz_is_using()
        # ⚠⚠ THE SMOKE DRIVES `gizmo_edits` WITH A SYNTHESISED MATRIX, and it
        # is honest about what that does and does not cover. A script cannot
        # move a pointer, so ImGuizmo's own hit-test and drag are out of
        # reach of any headless run — what IS reachable, and what breaks
        # silently, is everything downstream: the record write, the document
        # write, the undo push, the `RenderFields` refresh and the rebuild
        # decision. Those all live in this file and a library test proves
        # none of them. The FRAME ALGEBRA is gated separately by
        # `test_gizmo_math` and against MuJoCo by `check_gizmo_vs_mujoco.py`.
        if smoke_gizmo and max_frames > 0 and len(L.fmd.geoms) > 1:
            if frame == max_frames // 5 or frame == (2 * max_frames) // 5:
                var turn = frame != max_frames // 5
                panel.sel_kind = SEL_GEOM
                panel.sel_index = 1
                panel.gizmo_mode = GIZMO_TURN if turn else GIZMO_MOVE
                gz_target = TARGET_GEOM
                var f0 = edit_frame(L.fmd, positions, quats, L.body_parent,
                                    TARGET_GEOM, 1)
                var f1 = Frame(f0.pos + Vec3(0.031, -0.017, 0.023), f0.quat)
                if turn:
                    f1 = Frame(
                        f0.pos,
                        (Quat.from_axis_angle(
                            Vec3(0.37, -0.55, 0.75).normalized(), 0.31
                        ) * f0.quat).normalized(),
                    )
                gz_out = gizmo_edits(
                    L.fmd, positions, quats, L.body_parent, TARGET_GEOM, 1,
                    panel.gizmo_mode, frame_to_cm(f1), 1.0,
                )
                # ⚠ PRINTED WITH THE COUNT. "did not crash" reads identically
                # on a gizmo that emitted nothing at all.
                print("  smoke: gizmo",
                      String("turn") if turn else String("move"),
                      "on geom 1 emitted", len(gz_out), "edit(s)")
        # ⚠⚠ AND THE POINTER ARBITRATION, EVERY FRAME. `ig_want_mouse()` is
        # False while a gizmo handle is dragged — ImGuizmo's window carries
        # `NoInputs` — so without this the same drag moves the part AND
        # orbits the camera behind it. Written unconditionally because it is
        # a level, not an event: latched True would freeze the camera.
        renderer.set_pointer_claimed(gz_hot)

        # ── an inspector edit ─────────────────────────────────────────────
        # ⚠ APPLIED HERE, AFTER `build_ui` AND BEFORE THE STEP. The panel
        # returns a REQUEST precisely so this happens at a defined point: an
        # edit landing between the step and the draw would render a pose that
        # never existed, and one landing inside the panel would need the panel
        # to hold a `Model`, which is what keeps it compiling once.
        # ⚠⚠ ONE LIST, TWO PRODUCERS. The inspector's drag emits ONE edit; a
        # gizmo drag emits up to FOUR (a quaternion is not four independent
        # numbers — see `edits_from_frame`). Routing both through the same
        # list is what keeps the document write, the undo push, the render
        # refresh and the rebuild decision from existing twice.
        var pending = List[Edit]()
        var pending_key = String("")
        var pending_label = String("")
        if ui.edit_field >= 0 and panel.sel_kind != SEL_NONE:
            var tgt = TARGET_GEOM if panel.sel_kind == SEL_GEOM \
                else TARGET_BODY
            pending.append(
                Edit(tgt, panel.sel_index, ui.edit_field, ui.edit_value)
            )
            # ⚠ COALESCED BY (target, index, field). A drag emits a value
            # every frame it is held; one snapshot per frame would be a
            # hundred undo steps that each appear to do nothing, and a
            # hundred copies of the document. The key folds the drag into
            # one step and a DIFFERENT field starts a new one.
            pending_key = edit_key(tgt, panel.sel_index, ui.edit_field)
            pending_label = String(field_name(ui.edit_field), " on ") \
                + _sel_label(L, panel)
        elif len(gz_out) > 0:
            pending = gz_out.copy()
            # ⚠ ONE KEY FOR THE WHOLE GIZMO DRAG, not one per component. The
            # four quaternion edits of a single turn must fold into ONE undo
            # step, and the next frame of the same drag into that same step —
            # `edit_key`'s field slot carries the OPERATION for that reason,
            # offset past every `F_*` so it can never collide with one.
            pending_key = edit_key(gz_target, panel.sel_index,
                                   1000 + panel.gizmo_mode)
            pending_label = String(gizmo_mode_name(panel.gizmo_mode), " ") \
                + _sel_label(L, panel)
        if len(pending) > 0:
            # ⚠ EVERY RECORD WRITE BEFORE ANY DOCUMENT WRITE.
            # `apply_edit_to_document` re-reads the WHOLE attribute off the
            # record (all four quaternion components, all three of `pos`), so
            # interleaving them would write a half-updated quaternion into
            # the file and then correct it — which is fine on the last
            # iteration and wrong if the loop raises in the middle.
            for i in range(len(pending)):
                apply_edit(L.fmd, L.m, pending[i])
            # ⚠⚠ AND INTO THE DOCUMENT — the third copy. Without this the sim
            # and the inspector show the edit and `File > Save edited model`
            # writes the value the file had when it was OPENED. Gated by
            # `test_edit_reaches_the_document`.
            try:
                for i in range(len(pending)):
                    L.flat = apply_edit_to_document(
                        L.fmd, L.m, L.flat, pending[i]
                    )
                L.dirty = True
                hist.push(L.flat, L.base_dir, doc, pending_label,
                          pending_key, pose_snapshot(L.fmd, L.d))
            except de:
                # ⚠ NAMED, NOT SWALLOWED. The locator can fail on an element
                # with no name and no body to count within; the edit is still
                # live in the sim, and the user needs to know it will not be
                # in the file.
                print("  this edit cannot be saved:", de)
            # ⚠ THE RENDERER READS `RenderFields`, NOT THE RECORD, so a
            # colour or a size change is invisible until `rf` is rebuilt.
            # Cheap (no re-parse, no mesh load) and it keeps "what you see" and
            # "what you edited" the same thing.
            L.rf = build_render_fields(L.fmd, L.flat, L.base_dir)
            # ⚠ RE-MEASURED FROM THE HULLS ONLY, and the lazily-read entries
            # are dropped with it. A size edit can change a geom's extents, so
            # a stale `mesh_half` would outline the shape as it was before the
            # drag; re-reading the file for the selected geom costs one load
            # on the next frame, which the selection path does anyway.
            L.mesh_half = empty_half_extents(len(L.fmd.geoms))
            L.biggest_half = 0.0
            renderer.set_render_fields(L.rf.copy())
            var want_rebuild = False
            for i in range(len(pending)):
                if needs_rebuild(pending[i]):
                    want_rebuild = True
            if want_rebuild:
                # Mass changes the DERIVED inertia and invweight0, and a body
                # frame changes `dof_invweight0` and a free joint's `qpos0`,
                # so the record is authoritative and the live model must be
                # rebuilt rather than patched. See `needs_rebuild`.
                build_model_runtime[DT](L.fmd, L.dims, L.m)
            L.revalidate()

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
            var suffix = String(".scene.xml")
            if which == 2:
                suffix = String(".flat.xml")
            elif which == 3:
                suffix = String(".edited.xml")
            var out_path = L.path + suffix
            try:
                # ⚠⚠ OPTION 3 WRITES THE LIVE DOCUMENT VERBATIM, and that is
                # the only lossless way out of a structural edit. `to_mjcf`
                # REFUSES a model with <tendon>/<equality>/<keyframe> rather
                # than drop them, and the scene document cannot express an
                # edited robot tree at all — so before this, deleting a body
                # from softfoot was an edit with no save.
                var body = String("")
                if which == 1:
                    # ⚠⚠ A SCENE REFERENCES ITS BASE BY PATH. Writing one while
                    # the robot has been edited produces a file that reopens as
                    # the ORIGINAL robot — a composition pointing at the wrong
                    # model, with nothing to say so. So the edited model is
                    # written FIRST and the asset entry re-pointed at it:
                    # §11.1's materialize-on-override, at asset granularity.
                    if L.dirty:
                        var side = L.path + ".edited.xml"
                        var sf = open(side, "w")
                        sf.write(L.flat)
                        sf.close()
                        if doc.retarget_asset(L.path, side):
                            print("  wrote", side,
                                  "and pointed the scene at it (the base model"
                                  " has edits an <attach> cannot carry)")
                        else:
                            print("  wrote", side,
                                  "— WARNING: the scene's asset table does not"
                                  " name", L.path,
                                  "so it still references the ORIGINAL")
                    body = doc.to_mjcf(String("scene"))
                elif which == 3:
                    body = L.flat
                else:
                    body = export_flat_mjcf(L.fmd, String("exported"))
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
            # ⚠⚠ UNDO IS A REBUILD FROM A SNAPSHOT — V2.9. The previous
            # version re-parsed `L.path` and REPLAYED an `EditLog` onto it,
            # which can only express dims-preserving edits; a delete, an add,
            # a rename or a reparent reset the log, so the destructive half of
            # the studio was not undoable at all. The document is now
            # authoritative, so one snapshot restores either kind, and the
            # cost is the same rebuild the old path already paid.
            # ⚠ THE OUTGOING POSE GOES IN WITH THE MOVE. It is the only
            # moment this state's joint values exist; one tool call later
            # `L` has been replaced.
            var out_pose = pose_snapshot(L.fmd, L.d)
            var moved = hist.undo(out_pose) if panel.want_undo == 1 \
                else hist.redo(out_pose)
            var what = String("undo") if panel.want_undo == 1 \
                else String("redo")
            panel.want_undo = 0
            # ⚠ NOTHING HAPPENS AT THE FLOOR. Rebuilding on a cursor that did
            # not move would throw the live pose away and re-run the remap for
            # a document identical to the current one — an "undo" that
            # visibly resets the robot while having undone nothing.
            if not moved:
                print("  nothing to", what)
            else:
                try:
                    var fresh = Loaded(L.path, hist.doc(), hist.base_dir())
                    # ⚠ THE POSE IS CARRIED BY NAME, exactly as after a
                    # delete. An undo changes the model's SHAPE — that is the
                    # whole point of this change — so every qpos address may
                    # have moved.
                    var rep = remap_state(L.fmd, L.d, fresh.fmd, fresh.d)
                    # ⚠⚠ AND THEN WHAT THE LIVE STATE COULD NOT ACCOUNT FOR,
                    # FROM THE SNAPSHOT. Undoing a delete brings joints back
                    # that the live model does not have, so `remap_state`
                    # leaves them at `qpos0` and the limb reappears in a
                    # different attitude from the one it was deleted in. The
                    # live state still wins everywhere it has an answer, so
                    # this does not rewind a running sim.
                    apply_pose_snapshot(hist.pose(), fresh.fmd, fresh.d, rep)
                    # ⚠ AND THE SCENE COMES BACK TOO. `SceneDoc` is separate
                    # state and `to_mjcf` is one-way, so restoring the text
                    # without it would leave the next `add prop` regenerating
                    # from a composition the user had just undone.
                    doc = hist.scene()
                    fresh.dirty = hist.can_undo()
                    print(" ", what, "—", hist.label(), "|", rep.summary())
                    # ⚠ THE SELECTION CANNOT SURVIVE: indices shift.
                    panel.clear_selection()
                    renderer.set_render_fields(fresh.rf.copy())
                    L = fresh^
                    _sync_pose(L, positions, quats)
                except e:
                    print(" ", what, "failed:", e)

        # ── add / rename — V2.3, through the same rebuild as a delete ─────
        # ⚠ ONE HANDLER, ONE REBUILD PATH. Each of these regenerates the
        # document and re-parses it; giving any of them its own shortcut is
        # how the studio would grow a second model path (plan §10 risk 2).
        var struct_xml = String("")
        var struct_note = String("")
        if ui.add_body_here and ui.new_name.byte_length() > 0:
            var parent = String("")
            if panel.sel_kind == SEL_BODY and panel.sel_index > 0:
                parent = _label_of(L.fmd.body_names, panel.sel_index)
            try:
                # In front of the camera, so it is visible where it lands.
                var cam3 = renderer.renderer.camera.copy()
                var r = add_body(L.flat, parent, ui.new_name,
                                 cam3.target.x, cam3.target.y,
                                 cam3.target.z + 0.2)
                if r.ok:
                    struct_xml = r.xml
                    struct_note = String("added body '") + ui.new_name + "'"
                else:
                    print("  cannot add:", r.notes[0])
            except e:
                print("  add body failed:", e)
        elif ui.add_joint_here >= 0 and panel.sel_kind == SEL_BODY \
                and panel.sel_index > 0:
            var jt = String("hinge")
            if ui.add_joint_here == 1:
                jt = String("slide")
            elif ui.add_joint_here == 2:
                jt = String("ball")
            elif ui.add_joint_here == 3:
                jt = String("free")
            try:
                var r = add_joint(
                    L.flat, _label_of(L.fmd.body_names, panel.sel_index),
                    ui.new_name, jt, 0.0, 1.0, 0.0,
                )
                if r.ok:
                    struct_xml = r.xml
                    struct_note = String("added a ") + jt + " joint"
                else:
                    print("  cannot add joint:", r.notes[0])
            except e:
                print("  add joint failed:", e)
        elif ui.reparent_here and panel.sel_kind == SEL_BODY \
                and panel.sel_index > 0:
            try:
                var r = reparent_body(
                    L.flat, _label_of(L.fmd.body_names, panel.sel_index),
                    ui.new_name,
                )
                if r.ok:
                    struct_xml = r.xml
                    struct_note = r.notes[0]
                else:
                    print("  cannot reparent:", r.notes[0])
            except e:
                print("  reparent failed:", e)
        elif ui.rename_here and ui.new_name.byte_length() > 0 \
                and panel.sel_kind != SEL_NONE:
            var is_g = panel.sel_kind == SEL_GEOM
            var old = _label_of(L.fmd.geom_names, panel.sel_index) if is_g \
                else _label_of(L.fmd.body_names, panel.sel_index)
            var r = rename_element(
                L.flat, String("geom") if is_g else String("body"),
                old, ui.new_name,
            )
            if r.ok:
                struct_xml = r.xml
                struct_note = String("renamed '") + old + "' to '" \
                    + ui.new_name + "'"
                for note in r.notes:
                    print("   ", note)
            else:
                print("  cannot rename:", r.notes[0])

        if struct_xml.byte_length() > 0:
            try:
                print(" ", struct_note)
                var nxt2 = Loaded(L.path, struct_xml, L.base_dir)
                nxt2.dirty = True
                var out2 = pose_snapshot(L.fmd, L.d)
                var rep2 = remap_state(L.fmd, L.d, nxt2.fmd, nxt2.d)
                print("   ", rep2.summary())
                panel.clear_selection()
                renderer.set_render_fields(nxt2.rf.copy())
                L = nxt2^
                _sync_pose(L, positions, quats)
                hist.push(L.flat, L.base_dir, doc, struct_note,
                          String(""), out2)
            except e:
                print("  the edit did not load:", e)

        # ── delete a body or a geom FROM THE MODEL — V2.1 ─────────────────
        # ⚠⚠ THIS IS NOT `del_prop`. That removes an INSTANCE from the scene
        # document; this edits the robot's own tree, so it goes through the
        # text, is re-parsed, and takes every reference to what it removed.
        if ui.del_element and panel.sel_kind != SEL_NONE:
            var is_geom = panel.sel_kind == SEL_GEOM
            var victim = _label_of(L.fmd.geom_names, panel.sel_index) \
                if is_geom else _label_of(L.fmd.body_names, panel.sel_index)
            try:
                var r = delete_geom(L.flat, victim) if is_geom \
                    else delete_body(L.flat, victim)
                if not r.ok:
                    print("  cannot delete:", r.notes[0])
                else:
                    # ⚠ EVERY PRUNE IS PRINTED. These are edits the user did
                    # not make; discovering later that an actuator vanished is
                    # how an editor loses trust.
                    print("  deleted", victim)
                    for note in r.notes:
                        print("   ", note)
                    var nxt = Loaded(L.path, r.xml, L.base_dir)
                    nxt.dirty = True
                    # ⚠ THE POSE IS CARRIED BY NAME. A positional copy would
                    # take the knee's angle into the ankle — every address
                    # after a removed joint has shifted. See `studio/remap`.
                    var out1 = pose_snapshot(L.fmd, L.d)
                    var rep = remap_state(L.fmd, L.d, nxt.fmd, nxt.d)
                    print("   ", rep.summary())
                    # ⚠ THE SELECTION CANNOT SURVIVE: indices shift, and a
                    # kept index names a DIFFERENT part with full confidence.
                    panel.clear_selection()
                    renderer.set_render_fields(nxt.rf.copy())
                    L = nxt^
                    _sync_pose(L, positions, quats)
                    hist.push(L.flat, L.base_dir, doc,
                              String("deleted '", victim, "'"),
                              String(""), out1)
                    smoke_nbody1 = len(L.fmd.bodies) + 1
            except e:
                print("  delete failed:", e)

        # ── props: a STRUCTURAL edit, so the whole model is rebuilt ───────
        if ui.add_prop >= 0 or ui.dup_prop or ui.del_prop:
            var changed = True
            var prop_note = String("prop edit")
            if ui.add_prop >= 0:
                # In front of the camera, at a size that reads on screen.
                var cam2 = renderer.renderer.camera.copy()
                var fwd = (cam2.target - cam2.eye).normalized()
                var at = cam2.target + fwd * 0.0
                _ = doc.add_prop(ui.add_prop, 0.05, 0.05, 0.05,
                                 at.x, at.y, at.z + 0.3)
                prop_note = String("added a prop")
            elif ui.dup_prop and panel.sel_kind == SEL_BODY:
                _ = doc.duplicate_prop(
                    _label_of(L.fmd.body_names, panel.sel_index)
                )
                prop_note = String("duplicated a prop")
            elif ui.del_prop and panel.sel_kind == SEL_BODY:
                doc.remove_prop(_label_of(L.fmd.body_names, panel.sel_index))
                prop_note = String("removed a prop")
            else:
                changed = False
            if changed:
                try:
                    # ⚠⚠ BASE DIR IS "" — THE CWD — AND NOT THE MODEL'S.
                    # The document's `<asset><model file=>` entries hold the
                    # path the user OPENED, which is relative to the process
                    # CWD; expanding against the loaded model's directory
                    # concatenated the two:
                    #   .../boston_dynamics_spot/references/.../toddlerbot/...
                    # Two different bases for one path, which is the oldest
                    # bug shape in this file.
                    var out3 = pose_snapshot(L.fmd, L.d)
                    var nxt = Loaded(
                        L.path, doc.to_mjcf(String("scene")), String("")
                    )
                    # ⚠ THE SELECTION CANNOT SURVIVE. Indices shift when a
                    # body is added or removed, so a kept index names a
                    # DIFFERENT part — and the outline would sit on it,
                    # confidently.
                    panel.clear_selection()
                    renderer.set_render_fields(nxt.rf.copy())
                    L = nxt^
                    _sync_pose(L, positions, quats)
                    # ⚠ THE SCENE GOES IN WITH THE DOCUMENT. A prop edit is
                    # the one edit that changes `doc` rather than the robot's
                    # own text, and an entry holding one without the other
                    # would restore half of it — which is why `HistoryEntry`
                    # carries a `SceneDoc` at all.
                    hist.push(L.flat, L.base_dir, doc, prop_note,
                              String(""), out3)
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
        # ⚠⚠ AN INVALID MODEL IS NOT STEPPED. A body with a joint and no mass
        # gives a singular mass matrix, and the factorisation fills `qpos`
        # with NaN — the window then shows a robot that has vanished, which
        # reads as a renderer bug rather than as the model defect it is. The
        # Problems tab already names the reason; holding the sim is what makes
        # the state WORKABLE instead of merely visible.
        var invalid = worst_severity(L.diags) >= SEV_ERROR
        if invalid and not held_reported:
            held_reported = True
            print("  SIM HELD —", count_at(L.diags, SEV_ERROR),
                  "error(s) in this model; see the Problems tab. Editing,"
                  " selecting and the camera all still work.")
        if not invalid:
            held_reported = False
        var frozen = panel.paused or renderer.paused() or invalid
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
                # Spatial-tendon actuators and springs — see
                # `dynamics/pose_transmission.mojo`. It refreshes FK/cdof at
                # THIS qpos and returns immediately on a model with none, so
                # the four integrator/cone combinations below can share one
                # call and one scratch.
                # ⚠ ANY OF THE FOUR SCRATCHES WILL DO. All four integrators
                # are sized from the same `dims`, and this call OWNS the
                # `cdof` it writes: it fills it from the current `qpos`,
                # reads it, and whichever integrator steps below recomputes
                # its own from the same pose. Picking one avoids both a
                # fifth allocation and a four-way branch around a call that
                # would then have to stay in step with itself.
                apply_pose_transmission[DT](
                    L.sf, L.m, L.d, L.integ_pyr.scratch, actions, act,
                    L.fmd.timestep,
                )
            # ⚠ THE INTEGRATOR THE FILE ASKED FOR, THEN THE CONE. Four
            # combinations, and the studio must pick the one the MJCF
            # declares — a model stepped with a different integrator is not
            # the model the user opened.
            if L.integ_used == IntegratorType.RK4:
                # ⚠ ONE CONE ON THIS ARM. All 14 RK4 models in the tree are
                # pyramidal; `studio_integrator_warning` names the swap if an
                # elliptic one ever appears.
                L.rk4_pyr.step["cpu"](L.d, L.m)
            elif L.use_implicit:
                if L.cone_used == ConeType.ELLIPTIC:
                    L.imp_ell.step["cpu"](L.d, L.m)
                else:
                    L.imp_pyr.step["cpu"](L.d, L.m)
            elif L.cone_used == ConeType.ELLIPTIC:
                L.integ_ell.step["cpu"](L.d, L.m)
            else:
                L.integ_pyr.step["cpu"](L.d, L.m)
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
                # ⚠ THE DOCUMENT FOLLOWS THE MODEL. `doc` was built from the
                # FIRST file and never rebuilt, so after a File > Open the
                # scene still referenced the previous robot — adding a prop to
                # spot composed toddlerbot plus a box. The document describes
                # what is on screen; a swap replaces what is on screen.
                doc = scene_from_base(L.path)
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
    var del_body = String("")
    if len(args) > 6:
        del_body = String(args[6])
    var smoke_add = String("")
    if len(args) > 7:
        smoke_add = String(args[7])
    var smoke_undo = len(args) > 8 and String(args[8]) == "undo"
    var smoke_gizmo = len(args) > 9 and String(args[9]) == "gizmo"
    run_studio(path, drive, scale, frames, swap_to, del_body, smoke_add,
               smoke_undo, smoke_gizmo)
