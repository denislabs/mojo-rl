"""How big a MESH geom is — the number the selection outline draws.

## ⚠⚠ THE BUG THIS EXISTS TO FIX

A `<geom mesh="...">` normally carries **no `size` attribute** — the mesh is
the shape — so `GeomData`'s defaults survive into the record: `half_x/y/z` are
0 and `radius` is its untouched default of **0.5**. An outline falling back to
`radius` draws a ONE-METRE CUBE around every part of a 30 cm arm.

## ⚠⚠ AND WHY IT IS THE FILE, NOT THE LOADED HULL

The obvious source is `Model.mesh_verts` — the collision hull is already in
memory, so measuring it is free. It is also in the **wrong frame**, and that
is not a detail:

* `load_mesh_hull` stores vertices in the mesh's **PRINCIPAL frame**. Its own
  docstring says so: `mi` carries the centre of mass and principal-axis
  rotation MuJoCo bakes into every mesh, they are applied before the hull, and
  "the caller must compose the same `mi` into the geom's `pos`/`quat`".
* `outline_geom` reads **`RenderFields`**, and `build_render_fields` copies
  the geom's **declared** `pos`/`quat` straight off `FlatModelDef` — the frame
  the renderer draws the raw mesh file in.

Those are two different frames. Measured on so_arm101's `sts3215_03a_v1`:

    hull (principal frame) : 0.012400  0.019829  0.024133
    file (render frame)    : 0.022700  0.012400  0.020200

Same solid, axes permuted and up to 6% out. A box built from the first and
drawn in the second is wrong in a way that looks like a slightly sloppy
outline rather than like a frame error — which is why it survived. The fix is
not to rotate one into the other; it is to measure in the frame the box is
drawn in, from the same `load_stl` call the renderer makes. One implementation,
so the outline cannot disagree with the shape it is drawn around.

## ⚠ ON DEMAND, NOT AT LOAD

Reading every mesh at model-open measured **895 ms on `unitree_go2` and 908 ms
on `franka_emika_panda`** — roughly doubling the time to open them, to fill 33
and 56 slots of which the outline reads ONE. So the caller measures the
SELECTION, and a slot left at 0 means "not measured yet".

⚠ ZERO IS THAT MARK, which is why the fallback is applied HERE and not by the
caller: a slot filled with a fallback in advance is indistinguishable from a
measured one, and the lazy pass would never run.

## ⚠ THE MODELS THAT MADE A HULL-BASED IMPLEMENTATION LOOK FINE

`so_arm101` pairs every visual geom with a collision geom over the same mesh,
so 13 of its geoms measured *something*. `unitree_go2` and
`anybotics_anymal_c` collide with PRIMITIVES — not one of their 33 and 50 mesh
geoms has a hull at all, `nmesh_verts` is 0, and the old code returned early on
that and left every extent at 0. Those two are the negative controls in
`test_mesh_bounds` for exactly that reason.
"""

from mojo_rl.render.stl_loader import load_stl
from ..parser.render_fields import RenderFields


comptime FALLBACK_HALF: Float64 = 0.02
"""Last resort when a mesh cannot be read at all — 2 cm.

⚠ IT IS A MARKER, NOT A BOUND, and the alternative is what makes it defensible:
`radius` is 0.5 for a mesh geom, and so_arm101's real parts are 0.012-0.050 m.
Wrong by 20-40x reads as a broken outline; wrong by a little reads as an
approximate one, which is what this is."""


def empty_half_extents(ngeom: Int) -> List[Float64]:
    """Three zeros per geom — "nothing measured yet", the initial state."""
    return List[Float64](length=ngeom * 3, fill=0.0)


def measure_geom_from_file(
    rf: RenderFields, g: Int, mut half: List[Float64],
    marker_scale: Float64 = FALLBACK_HALF,
) raises:
    """Fill geom `g`'s half-extents from the mesh file the renderer draws.

    Idempotent: returns immediately once the entry is non-zero, because the
    caller runs this every frame the selection is up.
    """
    # ⚠ AN OUT-OF-RANGE INDEX IS IGNORED, NOT A TRAP. The selection index can
    # outrun the geom list for a frame after a structural edit.
    if g < 0 or g * 3 + 2 >= len(half):
        return
    if half[g * 3 + 0] > 0.0 or half[g * 3 + 1] > 0.0 or half[g * 3 + 2] > 0.0:
        return

    var mid = -1
    if g < len(rf.geom_mesh_id):
        mid = rf.geom_mesh_id[g]
    var ex = 0.0
    var ey = 0.0
    var ez = 0.0
    if mid >= 0 and mid < rf.nmesh:
        try:
            # ⚠ `rf.mesh_files` IS ALREADY RESOLVED against the asset base —
            # `draw_mesh` passes it straight through — and `load_stl` applies
            # `<mesh scale>` itself. 19 Menagerie robots set one and 44
            # declarations are a MIRROR like `1 -1 1`, whose |extent| is the
            # same either way, so the abs() below is scale-safe.
            var md = load_stl(
                rf.mesh_files[mid],
                rf.geom_mesh_scale[g * 3 + 0],
                rf.geom_mesh_scale[g * 3 + 1],
                rf.geom_mesh_scale[g * 3 + 2],
            )
            for v in range(len(md.vertices)):
                var ax = abs(Float64(md.vertices[v].px))
                var ay = abs(Float64(md.vertices[v].py))
                var az = abs(Float64(md.vertices[v].pz))
                if ax > ex:
                    ex = ax
                if ay > ey:
                    ey = ay
                if az > ez:
                    ez = az
        except:
            # ⚠ A FILE THE LOADER REFUSES STILL GETS A BOX. The renderer failed
            # the same way and drew nothing, or it is a format only it handles;
            # either way a marker is a better answer than a one-metre cube.
            ex = 0.0
            ey = 0.0
            ez = 0.0

    if ex <= 0.0 and ey <= 0.0 and ez <= 0.0:
        var mk = marker_scale
        if mk <= 0.0:
            mk = FALLBACK_HALF
        ex = mk
        ey = mk
        ez = mk
    half[g * 3 + 0] = ex
    half[g * 3 + 1] = ey
    half[g * 3 + 2] = ez


def biggest_half_extent(half: List[Float64]) -> Float64:
    """The largest extent measured SO FAR — the marker scale for a geom whose
    file cannot be read.

    ⚠ IT GROWS AS THINGS ARE MEASURED, because measurement is lazy. Early in a
    session it can be 0, which `measure_geom_from_file` reads as "use
    `FALLBACK_HALF`". That is the right degradation: a marker derived from
    nothing is worse than a fixed small one.
    """
    var b = 0.0
    for i in range(len(half)):
        if half[i] > b:
            b = half[i]
    return b
