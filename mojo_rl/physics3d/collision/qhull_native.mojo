"""The convex-hull FACE LIST in qhull's order — `mjCMesh::MakeGraph`'s.

## Why the engine calls a C library at all

`MakeGraph` (`user_mesh.cc`) does not implement a convex hull: it calls qhull
and walks `FORALLfacets`. That WALK ORDER is load-bearing all the way down —
it is the order `MakePolygons` inserts faces in, which decides where
`MeshPolygon::Paths()` starts each vertex cycle, which decides which three
vertices `MakePolygonNormals` reads. Our own exact hull (`convex_hull.mojo`)
reproduces qhull's vertex SET but not its facet order, and the polygon normals
that fall out land up to **0.26 deg** from `mesh_polynormal` — against the
**0.09167 deg** tolerance `alignedFaces` / `alignedFaceEdge` test with. Two
Menagerie board rows are that gap and nothing else.

⚠⚠ AND NO RULE RECOVERS IT. Scored against `mesh_polynormal` over 2574
polygons of one mesh: first-three-on-our-start 260 beyond tolerance, Newell
(what the engine shipped) 227, largest-triangle 228, largest-fan 221, best-fit
plane 220 — against **4** for MuJoCo's start. The information is not in the
polygon; it is in qhull's execution trace.

Reproducing that trace by reimplementation means reproducing qhull's facet
allocation, merge and triangulate ordering — and the reference does not do that
either. So the faithful port of `MakeGraph` is a CALL.

## The shim, and why it is a dylib

`native/mrl_qhull.c` is a transcription of `MakeGraph`'s extraction loop —
`FORALLfacets` with the `toporient` swap — and nothing else.

⚠ IT IS LOADED WITH `_get_dylib_function`, NOT `external_call`, for the reason
`mojo_rl/io/serial/native.mojo` records at length: **`mojo run`'s JIT does not
honour `-Xlinker` at all**, so a direct `external_call` makes the module
un-runnable under `mojo run` whether or not the call is reached — and every
test in this repo runs that way. Going through the stdlib's own `dlsym`
declares no C symbol and so cannot collide with one either.

**Consequence, stated plainly: a built binary that loads a MESH model needs
`libmrl_qhull.dylib` beside it**, exactly as the imgui viewers need
`libmojo_imgui.dylib`. Build it with `pixi run build-qhull`.

## ⚠ The input is the RAW vertex array

`Process()` builds `dvert` from `vert_` BEFORE the principal frame is baked, so
the hull MuJoCo computes is the hull of the raw deduped vertices. Measured on
`shadow_dexee`'s `Asm-MRH-F-Mid-Visual,00+MagTac,00.stl`: hulling the raw
vertices reproduces MuJoCo's 2580 polygon paths **2580/2580 including the cycle
START**; hulling the principal-frame vertices reproduces 713/2580, because the
float32 round trip perturbs `Qt`'s coplanar tie-breaking. Callers must hull
raw and transform afterwards.
"""

from std.ffi import OwnedDLHandle, _Global, _get_dylib_function
from std.os import abort, getenv
from std.pathlib import Path
from std.sys import CompilationTarget


def _lib_name() -> String:
    comptime if CompilationTarget.is_macos():
        return String("libmrl_qhull.dylib")
    else:
        return String("libmrl_qhull.so")


def _candidates() -> List[String]:
    """Most explicit first — the same shape as `io/serial` and `render/imgui`,
    so all three answer to the same conventions."""
    var name = _lib_name()
    var out = List[String]()
    var override = getenv("MOJO_RL_QHULL_LIB")
    if override.byte_length() > 0:
        out.append(override)
    var root = getenv("PIXI_PROJECT_ROOT")
    if root.byte_length() > 0:
        out.append(root + "/mojo_rl/physics3d/collision/" + name)
    # Relative to CWD, which for this project is the repo root.
    out.append("mojo_rl/physics3d/collision/" + name)
    out.append(name)
    return out^


def qhull_shim_available() -> Bool:
    """True when the shim can be found WITHOUT dlopening it.

    `_Global` aborts the process on a missing library — right for a hard
    dependency, wrong as a first impression. Call this before a mesh build to
    say "run `pixi run build-qhull`" instead of dying in the loader.
    """
    var candidates = _candidates()
    for i in range(len(candidates)):
        if Path(candidates[i]).exists():
            return True
    return False


def _init_qhull_handle() -> OwnedDLHandle:
    """Non-raising, as `_Global` demands; aborts with the paths it tried."""
    var candidates = _candidates()
    for i in range(len(candidates)):
        try:
            return OwnedDLHandle(candidates[i])
        except:
            pass
    var tried = String("")
    for i in range(len(candidates)):
        tried += "\n  - " + candidates[i]
    abort(
        "qhull shim not found. Tried:"
        + tried
        + "\nBuild it with `pixi run build-qhull`, or set"
        + " MOJO_RL_QHULL_LIB=/path/to/"
        + _lib_name()
    )


comptime lib = _Global["MOJO_RL_QHULL", _init_qhull_handle]()


def qhull_max_faces(nvert: Int) -> Int:
    """Upper bound on the facet count: Euler's `F <= 2V - 4` for a simplicial
    3-polytope, which is what `Qt` leaves behind."""
    if nvert < 4:
        return 0
    return 2 * nvert - 4


def qhull_faces(
    verts: Pointer[mut=False, Float64, _],
    nvert: Int,
    maxhullvert: Int,
    faces: Pointer[mut=True, Int32, _],
    maxface: Int,
) raises -> Int:
    """`MakeGraph`'s face list: 3 global point ids per face, qhull's order.

    `verts` is `nvert * 3` doubles — the RAW deduped vertices, see the module
    header. `maxhullvert` is the model's `<mesh maxhullvert>`, or -1; it
    selects MuJoCo's own `Q9 TA<n>` option form. Returns the face count, or
    raises with what the shim reported.
    """
    # ⚠ THE ORIGIN CASTS HAPPEN INSIDE THE CALL, and the public parameters stay
    # GENERIC over the caller's origin. Fixing them at `MutUntrackedOrigin`
    # forces every caller to write the cast itself, and that cast SEVERS the
    # borrow — Mojo then destroys the caller's buffer at its last mention,
    # which is the cast, and qhull reads freed memory. Same reasoning as
    # `io/hdf5/h5s.mojo`, written out there at length.
    var n = _get_dylib_function[
        lib,
        "mrl_qhull_faces",
        def (
            Pointer[Float64, MutUntrackedOrigin],
            Int32,
            Int32,
            Int32,
            Pointer[Int32, MutUntrackedOrigin],
        ) thin -> Int32,
    ]()(
        verts.unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int32(nvert),
        Int32(maxhullvert),
        Int32(maxface),
        faces.unsafe_origin_cast[MutUntrackedOrigin](),
    )
    var r = Int(n)
    if r < 0:
        # ⚠ NAMED, NOT SWALLOWED. A silent 0 here would be a mesh with no
        # collision geometry, which no gate in this tree would catch.
        var why = String("unknown")
        if r == -1:
            why = String("qhull error (longjmp) — degenerate input?")
        elif r == -2:
            why = String("face buffer too small")
        elif r == -3:
            why = String("qhull returned a non-triangle")
        elif r == -4:
            why = String("qhull returned a point id out of range")
        raise Error("mrl_qhull_faces failed: " + why)
    return r
