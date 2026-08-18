"""physics3d studio, slice S0 — open ANY MJCF file, by path, and simulate it.

    pixi run mojo build -I . -o /tmp/studio examples/physics3d/physics_studio.mojo
    /tmp/studio mojo_rl/envs/humanoid/assets/humanoid.xml
    /tmp/studio mojo_rl/envs/robots/assets/so_arm100.xml sweep 0.4

⚠ BUILD ONCE AND RUN THE BINARY. `mojo run` recompiles, and this file takes
~2 min to build. It does NOT grow with the number of models it can open —
that is the whole point, and it is the number `docs/PHYSICS3D_STUDIO_PLAN.md`
§4 was written to make visible. `dm_viewer_imgui`, which bakes 47 models into
its type, takes ~8 minutes.

WHAT THIS SLICE IS
==================
The viewer half of S0: one binary, any file, on screen, driven, at 60 Hz. No
editing, no outliner, no scene composition — those are S1-S4. What it proves
is that the two halves of "runtime model" now meet:

* **physics** — `parse_model_runtime` -> `dims_from_flat` -> `Model[DTYPE,
  DynDims]` -> `build_model_runtime`, drivable since 3d and no longer
  silently skipping constraint families since the `may_exist` conversion;
* **rendering** — `build_render_fields(fmd, …)` -> `ModelRenderer[
  RfOnlyModelDef](adopt_rf=rf)`. Every render hook is a pure function of
  `rf: RenderFields`, so which model def you instantiate the renderer on does
  not matter. `RfOnlyModelDef` is a model with no bodies whose only job is to
  be that namespace; `scripts/audit_render_hooks_are_rf_pure.py` keeps it
  honest.

⚠ CPU ONLY, ONE ENV, ON PURPOSE. A runtime dims provider captured by a GPU
kernel reads 0 and silently zeroes its output, so there is no runtime GPU
path and this tool does not pretend to have one. One model at 60 Hz needs no
GPU; see the plan's §8 for the "bake and train" answer.

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window.

CONTROLS come from `Renderer3D`: orbit/pan/zoom with the mouse, 1-9 to switch
model cameras, 0 for the free camera, SPACE to pause, . to single-step, R to
record. The HUD lists them.

THE TWO ARGUMENTS THAT ARE NOT IN THE FILE
==========================================
`max_contacts` and `nmesh_verts` are workspace budgets, not model properties
(see `dims_from_flat`). Too small a `max_contacts` SILENTLY DROPS contacts,
so this tool prints the high-water mark next to the budget every second — a
scene composer adds props and raises the contact count with no other signal.
`nmesh_verts` cannot be derived before the meshes load, but the builder
raises WITH THE NUMBER IT NEEDS, so `_load` retries on the raise and the user
never has to know. That is §1.2 of the plan, implemented.
"""

from std.sys import argv
from std.random import seed, random_float64
from std.math import sin
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.math3d import Vec3 as Vec3G, Quat as QuatG
from mojo_rl.physics3d.fields import Data, Model, DynDims, SpecFields
from mojo_rl.physics3d.parser import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.parser.runtime_load import read_model_source
from mojo_rl.physics3d.parser.runtime_load import spec_fields_runtime
from mojo_rl.physics3d.parser.render_fields import (
    RenderFields, build_render_fields,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.model_def_from_xml import RfOnlyModelDef
from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.model.model_renderer import ModelRenderer
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS

comptime DT = DType.float64
comptime Vec3 = Vec3G[DT]
comptime Quat = QuatG[DT]

comptime DRIVE_ZERO: Int = 0
comptime DRIVE_RANDOM: Int = 1
comptime DRIVE_SWEEP: Int = 2

comptime MAX_CONTACTS: Int = 128
"""Generous, because a composed scene is where this tool is going and a short
budget drops contacts with no error. 128 * 30 scalars is ~30 kB."""


def _drive_name(m: Int) -> String:
    if m == DRIVE_ZERO:
        return String("zero")
    if m == DRIVE_RANDOM:
        return String("random")
    return String("sweep")


def parse_drive(s: String) -> Int:
    if s == "zero":
        return DRIVE_ZERO
    if s == "random":
        return DRIVE_RANDOM
    return DRIVE_SWEEP


def _fmt2(v: Float64) -> String:
    """Two decimals without a formatting library (borrowed from viewer_core)."""
    var scaled = Int(v * 100.0 + (0.5 if v >= 0 else -0.5))
    var whole = scaled // 100
    var frac = scaled % 100
    if frac < 0:
        frac = -frac
    var f = String(frac) if frac >= 10 else "0" + String(frac)
    return String(whole) + "." + f


def _dims_with_meshes(fmd: FlatModelDef, want: Int) raises -> DynDims:
    """`dims_from_flat` with a vertex budget that is large enough.

    ⚠ THE BUDGET CANNOT BE DERIVED HERE and that is not an oversight —
    `dims_from_flat`'s docstring explains it: the hull vertex count is known
    only once the meshes load, which happens INSIDE the builder, after this
    point. Loading them twice to count them would double the most expensive
    stage of the load (so_arm100's 11.2 ms build is almost all mesh loading).

    So: guess, and let the builder tell us. It raises naming the number it
    needs, so doubling on the raise converges in a handful of tries and the
    caller never has to know the number. This is the "retry-on-raise load
    loop" the plan calls a legitimate cheap fix (§1.2).
    """
    return dims_from_flat(fmd, max_contacts=MAX_CONTACTS, nmesh_verts=want)


def _outline(fmd: FlatModelDef) raises:
    """The kinematic tree, BY NAME — the read-only half of S1's outliner.

    ⚠ THIS IS WHAT THE NAME TABLES BOUGHT. Before them the parser resolved
    names into indices and dropped the strings, so the best this could print
    was "body 7 (parent 3)". `FlatModelDef.body_names` & co. are indexed in
    MuJoCo element order and gated against `mj_id2name`
    (`test_model_names_vs_mujoco`), so what prints here is what MuJoCo would
    call the same element.

    On stdout rather than in the window because S0 has no sidebar: reusing
    `viewer_core`'s means lifting it out of `run_view`, which is parameterised
    on `MODEL: ModelDefLike` + `CONFIG` and builds a comptime env. That is S1's
    work, and this is the data it will show.

    ⚠ AN EMPTY NAME IS PRINTED AS `<geom 4>`, and the angle brackets are the
    tell: MJCF does not require `name=` and most geoms here have none. The
    table stores "" rather than a synthesised name so that an export cannot
    claim a name the source never had — so the invention happens HERE, where
    it is visibly a display choice.
    """
    print("  ── outline ───────────────────────────────────────────────")
    var nb = len(fmd.body_names)
    for b in range(nb):
        var indent = String("    ")
        # Depth by walking parents; the tree is shallow and this runs once.
        var d = 0
        var cur = b
        while cur > 0 and d < 24:
            cur = fmd.bodies[cur - 1].parent
            d += 1
        for _ in range(d):
            indent += "  "
        print(indent, _label(fmd.body_names, b, "body"))
        for j in range(len(fmd.joints)):
            if fmd.joints[j].body_id == b:
                print(indent, "   joint ",
                      _label(fmd.joint_names, j, "joint"))
        for g in range(len(fmd.geoms)):
            if fmd.geoms[g].body_id == b:
                print(indent, "   geom  ",
                      _label(fmd.geom_names, g, "geom"))
    if len(fmd.actuator_names) > 0:
        print("    actuators:")
        for a in range(len(fmd.actuator_names)):
            print("      ", a, _label(fmd.actuator_names, a, "actuator"))
    print("  ──────────────────────────────────────────────────────────")


def _label(names: List[String], i: Int, kind: String) -> String:
    if i < len(names) and names[i].byte_length() > 0:
        return names[i].copy()
    return String("<", kind, " ", i, ">")


def run_studio(
    path: String, drive: Int, scale: Float64, max_frames: Int = 0
) raises:
    print("=" * 70)
    print("physics3d studio (S0) —", path)
    print("=" * 70)

    # ── parse, ONCE ───────────────────────────────────────────────────────
    # ⚠ THE PARSE IS THE EXPENSIVE HALF and it is per-ASSET, not per-edit:
    # dog is 9.0 ms of text parsing, so_arm100 is 11.2 ms of mesh loading,
    # while `dims_from_flat` and `spec_fields_runtime` are under 0.01 ms.
    # S2's scene composer caches this `FlatModelDef` per asset file for
    # exactly that reason; S0 loads one model, so it just keeps it.
    var src = read_model_source(path)
    var fmd = parse_xml_full(src[0], src[1])

    # ── dimensions, with the mesh budget the builder asks for ─────────────
    var verts = 0
    var dims = _dims_with_meshes(fmd, verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            # ⚠ ONLY THE VERTEX-BUDGET RAISE IS RETRYABLE. Anything else is a
            # real load failure and must reach the user unchanged — swallowing
            # it here would turn "your MJCF is malformed" into an infinite
            # loop.
            if String(e).find("mesh vertex capacity") == -1:
                raise e
            tries += 1
            if tries > 24:
                raise e
            verts = 4096 if verts == 0 else verts * 2
            print("  mesh vertex budget ->", verts)
            dims = _dims_with_meshes(fmd, verts)
            m = Model[DT, DynDims](dims)

    var sf = spec_fields_runtime[DT](fmd, dims)
    var d = Data[DT, DynDims, 1](dims)
    var integ = EulerIntegrator[DT, DynDims, BATCH=1, MAX_CONDIM=3](dims)

    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var nact = dims.get_nact()
    print("  nbody", nbody, " ngeom", dims.get_ngeom(), " nq", nq, " nv", nv)
    print("  nact", nact, " nsite", dims.get_nsite(),
          " ntendon", dims.get_ntendon(), " nequality", dims.get_nequality())
    print("  mesh verts", dims.get_nmesh_verts(),
          " contact budget", MAX_CONTACTS)
    print("  drive:", _drive_name(drive), " scale", _fmt2(scale))
    _outline(fmd)

    # ── the reference pose ────────────────────────────────────────────────
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)

    # ── the renderer, on records rather than on a type ────────────────────
    var rf = build_render_fields(fmd, src[0], src[1])
    var renderer = ModelRenderer[RfOnlyModelDef](
        width=1280,
        height=720,
        visual_radius_scale=1.0,
        show_velocity=False,
        title=String("physics3d studio"),
        adopt_rf=Optional[RenderFields](rf^),
    )
    renderer.init(None)

    var actions = List[Float64](length=nact if nact > 0 else 1, fill=0.0)
    var act = List[Scalar[DT]](length=nact if nact > 0 else 1, fill=Scalar[DT](0))
    var timestep = fmd.timestep

    var positions = List[Vec3](capacity=nbody)
    var quats = List[Quat](capacity=nbody)
    for _ in range(nbody):
        positions.append(Vec3(0, 0, 0))
        quats.append(Quat(1, 0, 0, 0))

    var t = 0
    var max_ncon = 0
    # ⚠ THE STEP CLOCK EXCLUDES DRAWING. What the plan asks this slice to
    # measure is the runtime leg's step rate against the comptime one, and a
    # figure that folds in SDL, the GPU submit and the 60 Hz frame wait would
    # answer a different question — one dominated by the display.
    var step_ns = 0
    var t0 = perf_counter_ns()
    while renderer.is_open():
        if renderer.check_quit():
            break
        # ⚠ A FRAME CAP, NOT A TIME CAP, so the smoke run is deterministic.
        # `max_frames > 0` is what lets this binary be run unattended — it
        # still opens a window (there is no headless path in `Renderer3D`),
        # but it closes itself and prints the summary a CI log can read.
        if max_frames > 0 and t >= max_frames:
            break

        if not renderer.paused():
            # ── drive ──────────────────────────────────────────────────────
            for a in range(nact):
                if drive == DRIVE_ZERO:
                    actions[a] = 0.0
                elif drive == DRIVE_RANDOM:
                    actions[a] = (random_float64() * 2.0 - 1.0) * scale
                else:
                    # A slow phase-shifted sweep: enough to see every joint
                    # move without pretending to be a controller.
                    var ph = Float64(t) * 0.02 + Float64(a) * 0.7
                    actions[a] = scale * sin(ph)
            var s0 = perf_counter_ns()
            if nact > 0:
                apply_actions_fields[DT](sf, d, actions, act, timestep)
            integ.step["cpu"](d, m)
            step_ns += Int(perf_counter_ns() - s0)
            t += 1

            var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
            if nc > max_ncon:
                max_ncon = nc

        # ── draw ───────────────────────────────────────────────────────────
        for b in range(nbody):
            positions[b] = Vec3(
                Float64(d.xpos.data[b * 3 + 0]),
                Float64(d.xpos.data[b * 3 + 1]),
                Float64(d.xpos.data[b * 3 + 2]),
            )
            quats[b] = Quat(
                Float64(d.xquat.data[b * 4 + 3]),
                Float64(d.xquat.data[b * 4 + 0]),
                Float64(d.xquat.data[b * 4 + 1]),
                Float64(d.xquat.data[b * 4 + 2]),
            )

        # ⚠ THE CONTACT BUDGET IS ON SCREEN, NOT IN A LOG. Overflowing it
        # drops contacts with no error and no crash — the model just gets
        # quietly softer. §1.2 of the plan calls this out as one of the two
        # silent-failure surfaces the composer creates, so the number that
        # would warn you lives where you are already looking.
        var hud = List[String]()
        hud.append(String("file: ", path))
        hud.append(String("drive: ", _drive_name(drive),
                          "  scale ", _fmt2(scale)))
        hud.append(String("contacts: ", max_ncon, " peak / ",
                          MAX_CONTACTS, " budget"))
        renderer.set_hud_extra(hud)
        renderer.render(positions, quats)

    renderer.close()
    var wall_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    print("  stepped", t, "times; peak contacts", max_ncon, "/", MAX_CONTACTS)
    if t > 0:
        var us = Float64(step_ns) / 1000.0 / Float64(t)
        print("  step cost", _fmt2(us), "us  (", _fmt2(1.0e6 / us),
              "steps/s, physics only)")
        print("  wall", _fmt2(wall_ms), "ms for", t, "frames")


def main() raises:
    seed(0)
    var args = argv()
    if len(args) < 2:
        print("usage: physics_studio <model.xml> [zero|random|sweep] [scale]"
              " [frames]")
        print("  e.g. mojo_rl/envs/humanoid/assets/humanoid.xml sweep 0.4")
        return
    var path = String(args[1])
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
    run_studio(path, drive, scale, frames)
