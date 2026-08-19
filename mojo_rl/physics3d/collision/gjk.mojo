"""GJK/EPA for mesh collision over per-field tensors (migration P4).

Per-field port of gjk_gpu.mojo — arithmetic verbatim. The only change is
the mesh-vertex operand: instead of the flat model slab
(`model[0, mesh_vert_buf_off + i*3 + k]`) the functions take the
`mesh_verts` record tensor (`[NMESH_VERTS, 3]`, see Model) and a
vertex START index (`vert_adr`, MuJoCo `mesh_vertadr`) so reads become
`mesh_verts[vert_adr + i, k]`. Same floats, same iteration order.
"""

from std.math import sqrt, abs
from layout import Layout, LayoutTensor
from .gjk_support import _closest_point_on_simplex
from .gjk_support import (
    support_sphere,
    support_capsule,
    support_box,
    support_cylinder,
)
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_MESH,
)
from ..kinematics.quat_math import quat_rotate, quat_rotate_inverse
from ..gpu.constants import (
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
    mesh_max_edge,
)

# Reuse CPU GJK parameters (verbatim from gjk_gpu.mojo)
comptime GJK_MAX_ITERATIONS: Int = 100
comptime GJK_TOLERANCE: Float64 = 1e-10
comptime EPA_MAX_VERTS: Int = 69
comptime EPA_MAX_FACES: Int = 384

# ⚠⚠ `EPA_MAX_ITERATIONS` AND `EPA_TOLERANCE` ARE GONE. They were 64 and 1e-8
# — MuJoCo's are `opt.ccd_iterations` = 35 and `opt.ccd_tolerance` = 1e-6, and
# a model that set either was ignored. They now arrive as runtime arguments
# (`ccd_iter`, `ccd_tol`) carried in model META, defaulting to MuJoCo's, which
# is why the constants are DELETED rather than retuned: anything still reading
# them fails to compile instead of quietly keeping the old stopping rule.
#
# ⚠ TIGHTER WAS NOT SAFER. EPA's stopping rule picks WHICH boundary face it
# settles on and the contact NORMAL is that face's, so iterating past the
# reference walks away from it rather than toward it.
comptime EPA_ITER_HARD_CAP: Int = 64

# `mjMINVAL` (`mjtnum.h`).
comptime MJ_MINVAL: Float64 = 1e-15


@always_inline
def _gjk_min_norm2[
    DTYPE: DType
](
    type1: Int,
    type2: Int,
    margin: Scalar[DTYPE],
    ccd_tol: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """MuJoCo's `min_norm2` — the floor on |v|^2 that ends the GJK loop.

        mjtNum tol2 = status->tolerance * status->tolerance;
        mjtNum min_norm2 = discreteGeoms(obj1, obj2) ? mjMINVAL2 : tol2;
        ...
        if ((x_norm = dot3(x_k, x_k)) < min_norm2) break;

    (`engine_collision_gjk.c:212-225`.) It is the SAME `discreteGeoms` switch
    `_epa_tolerance` already implements — that one crossed to EPA and never to
    GJK, and this is the other half.

    ⚠⚠ IT IS A THRESHOLD ON |v| SQUARED, WHICH IS WHY THE OLD CONSTANT LOOKED
    HARMLESS. `GJK_TOLERANCE = 1e-10` reads as a tight tolerance; against a
    SQUARED norm it is a distance floor of `sqrt(1e-10)` = **1e-5 m**. Every
    convex pair separated by less than 10 microns was classified PENETRATING
    and handed to EPA, which returned `-0.0`: an invented contact at a real
    gap. MuJoCo's floor is `ccd_tolerance` = 1e-6 m for a smooth pair and
    `mjMINVAL` = 1e-15 m for a polytope pair — 10x and 1e10x tighter.

    ⚠ MEASURED, dm_control `manipulation/reassemble_5`. The Duplo stud sits
    3.0 microns from the brick flange above it (exact: the stud axis is
    0.00465 from the flange face and its radius is 0.004647). MuJoCo reports
    +3.02e-06 on all 48 such pairs; we reported -6.33e-06 — penetration where
    there is a gap — on all 48, because 3e-06 is under the 1e-5 floor.

    ⚠ THE CLASSIFICATION AFTER THE LOOP MUST USE THE SAME VALUE. `dist_sq >
    min_norm2` is what decides separated-vs-penetrating; if it disagrees with
    the loop's break test, a run that converged to the origin is then reported
    separated at a hair's-breadth distance (see the note at that branch).
    """
    var tol2 = ccd_tol * ccd_tol
    if margin != 0:
        return tol2
    var d1 = type1 == GEOM_BOX or type1 == GEOM_MESH
    var d2 = type2 == GEOM_BOX or type2 == GEOM_MESH
    if d1 and d2:
        return Scalar[DTYPE](MJ_MINVAL) * Scalar[DTYPE](MJ_MINVAL)
    return tol2


@always_inline
def _epa_tolerance[
    DTYPE: DType
](
    type1: Int,
    type2: Int,
    margin: Scalar[DTYPE],
    ccd_tol: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """MuJoCo's `discreteGeoms` switch: `ccd_tolerance` is for SMOOTH pairs.

    `epa()` opens with

        int discrete = discreteGeoms(obj1, obj2);
        if (discrete && sizeof(mjtNum) == sizeof(double)) tolerance = mjMINVAL;

    and `discreteGeoms` is "both geoms are MESH/BOX/HFIELD **and** both margins
    are zero" (`engine_collision_gjk.c:159`). A polytope pair has finitely many
    faces, so EPA lands on the exact one and there is nothing for a tolerance to
    trade off; a curved surface has no exact face, so `ccd_tolerance` decides
    where to stop.

    ⚠⚠ MISSING THIS MAKES `ccd_tolerance` A REGRESSION RATHER THAN A FIX, and
    that is how it was found. Wiring `opt.ccd_tolerance` through and applying it
    to EVERYTHING moved mesh-vs-box depth from 1e-16 to 4.98e-7 against MuJoCo
    — inside `ccd_tolerance` and therefore invisible to any check phrased in
    those terms, but a 5e-7 regression on pairs that were EXACT.
    `test_mesh_manifold_vs_mujoco`'s `TOL_DIST = 1e-9` caught it, and the
    tempting response — relax the gate, the number is under 1e-6 — would have
    locked the regression in.

    ⚠ A NON-ZERO MARGIN MAKES A DISCRETE PAIR SMOOTH. That is MuJoCo's first
    line, not an afterthought: margin inflates each geom by a rounded offset
    surface, so the Minkowski boundary stops being piecewise planar.

    ⚠ DOUBLE PRECISION ONLY, matching the `sizeof(mjtNum)` guard. At float32,
    1e-15 is far below epsilon: the convergence test could never fire and EPA
    would run to its iteration cap on every mesh pair.
    """
    comptime if DTYPE != DType.float64:
        return ccd_tol
    else:
        if margin != 0:
            return ccd_tol
        var d1 = type1 == GEOM_BOX or type1 == GEOM_MESH
        var d2 = type2 == GEOM_BOX or type2 == GEOM_MESH
        if d1 and d2:
            return Scalar[DTYPE](MJ_MINVAL)
        return ccd_tol


@always_inline
def _dot3[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    return ax * bx + ay * by + az * bz


# =============================================================================
# EPA polytope helpers
#
# Fixed-size, non-recursive, no dynamic allocation — all three are hard
# requirements for the GPU path, and MuJoCo's own EPA satisfies none of them
# (`horizonRec` at `engine_collision_gjk.c` recurses, and the polytope is heap
# allocated). The horizon is therefore computed by an UNDIRECTED edge count
# instead of a recursive walk: an edge belonging to exactly one visible face is
# on the horizon. Same horizon, no stack. See the note at the loop for why the
# directed formulation — which needs consistently wound faces — is wrong here.
#
# WHAT THIS IS AND IS NOT. EPA is exact wherever it can seed: on a cylinder
# against a box face, and on the same box expressed as an 8-vertex MESH, it
# reproduces the analytic depth BIT-FOR-BIT at every penetration from 5e-4 to
# 0.03, where the placeholder it replaced returned ~-1.1 throughout. It does
# NOT yet cover pairs whose GJK simplex comes back degenerate — notably
# sawyer's obj against the eGripperBase hull — which still take the old
# estimate. Porting `polytope2/3/4` is what closes that, and until it lands
# this is an improvement with a documented hole, not a finished collider.
#
# ⚠ THE CAPS ARE MEASURED, NOT INHERITED. `EPA_MAX_FACES = 384` /
# `EPA_MAX_VERTS = 69` above are MuJoCo's generous allocation. A reference EPA
# run over the pairs this engine actually collides — cylinder/box, box/box and
# capsule/box across penetrations from 1e-4 to 0.05, plus sawyer's obj against
# the 883-vertex eGripperBase hull — peaks at **32 faces and 18 verts**, so
# these are set to 2x that. Overflow is REPORTED (the routine bails to its best
# face so far) rather than silently truncating the polytope, which would return
# a plausible wrong depth. Re-measure if hfields or much larger hulls arrive.
comptime EPA_V_CAP: Int = 36
comptime EPA_F_CAP: Int = 64


@always_inline
def _gjk_signed_distance[
    DTYPE: DType,
](
    simplex: InlineArray[Scalar[DTYPE], 36],
    i1: Int, i2: Int, i3: Int,
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Signed distance from the origin to the plane (v1, v2, v3), and its normal.

    Port of `signedDistance` (`engine_collision_gjk.c:375`). The normal is
    `cross(v3 - v1, v2 - v1)` normalised, and the distance is `n . v1`, so the
    SIGN carries the tetrahedron's orientation — which is the whole mechanism
    `_gjk_intersect` relies on. A degenerate face returns a huge distance so it
    loses every minimum comparison, exactly as MuJoCo's `mjMAX_LIMIT` does.
    """
    var ax = simplex[i1 * 9 + 0]
    var ay = simplex[i1 * 9 + 1]
    var az = simplex[i1 * 9 + 2]
    var d1x = simplex[i3 * 9 + 0] - ax
    var d1y = simplex[i3 * 9 + 1] - ay
    var d1z = simplex[i3 * 9 + 2] - az
    var d2x = simplex[i2 * 9 + 0] - ax
    var d2y = simplex[i2 * 9 + 1] - ay
    var d2z = simplex[i2 * 9 + 2] - az
    var nx = d1y * d2z - d1z * d2y
    var ny = d1z * d2x - d1x * d2z
    var nz = d1x * d2y - d1y * d2x
    var n2 = nx * nx + ny * ny + nz * nz
    if n2 > Scalar[DTYPE](1e-30) and n2 < Scalar[DTYPE](1e30):
        var inv = Scalar[DTYPE](1) / sqrt(n2)
        nx *= inv
        ny *= inv
        nz *= inv
        return (nx * ax + ny * ay + nz * az, nx, ny, nz)
    return (Scalar[DTYPE](1e30), Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](0))


@always_inline
def _epa_face_normal[
    DTYPE: DType,
](
    ev: InlineArray[Scalar[DTYPE], EPA_V_CAP * 9],
    i: Int,
    j: Int,
    k: Int,
) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]
]:
    """Outward unit normal of face (i, j, k) and its distance to the origin.

    "Outward" means away from the origin, which is inside the polytope
    throughout EPA, so the sign is fixed by `n . a >= 0` rather than by winding.
    A zero-area face returns a zero normal and the caller skips it.
    """
    var ax = ev[i * 9 + 0]
    var ay = ev[i * 9 + 1]
    var az = ev[i * 9 + 2]
    var bx = ev[j * 9 + 0] - ax
    var by = ev[j * 9 + 1] - ay
    var bz = ev[j * 9 + 2] - az
    var cx = ev[k * 9 + 0] - ax
    var cy = ev[k * 9 + 1] - ay
    var cz = ev[k * 9 + 2] - az
    var nx = by * cz - bz * cy
    var ny = bz * cx - bx * cz
    var nz = bx * cy - by * cx
    var ln = sqrt(nx * nx + ny * ny + nz * nz)
    if ln < Scalar[DTYPE](1e-20):
        return (Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0),
                Scalar[DTYPE](0))
    nx /= ln
    ny /= ln
    nz /= ln
    var d = nx * ax + ny * ay + nz * az
    if d < Scalar[DTYPE](0):
        nx = -nx
        ny = -ny
        nz = -nz
        d = -d
    return (nx, ny, nz, d)


@always_inline
def _epa_seed_contains_origin[
    DTYPE: DType,
](
    ev: InlineArray[Scalar[DTYPE], EPA_V_CAP * 9],
    ef: InlineArray[Int, EPA_F_CAP * 3],
    nef: Int,
) -> Int:
    """Seed classification: 1 contains the origin, 0 touching, -1 degenerate.

    ⚠ THE TWO FAILURE MODES ARE NOT THE SAME and must not share an answer.
    A well-formed hull whose nearest face sits on the origin means the geoms
    TOUCH, and zero is the true depth. A hull with a ZERO-AREA face means the
    seed itself is degenerate — GJK routinely ends on a near-flat simplex for
    mesh pairs — and says nothing about depth; the pair may be deeply
    overlapped. Answering "zero" there DROPS a real contact: sawyer's obj is
    27.7 mm inside the gripper hull and vanished entirely when both modes
    returned the same thing.

    EPA assumes it does — every face distance it computes is a lower bound on
    the depth only under that assumption. When the origin sits on the boundary
    the closest-face search can lock onto the FAR side instead: a cylinder
    resting exactly on a box returned -1.1, which is the full Minkowski extent
    along z (2*0.5 + 2*0.05), not a depth.
    """
    if nef <= 0:
        return -1
    var touching = False
    for f in range(nef):
        var fnm = _epa_face_normal[DTYPE](
            ev, ef[f * 3 + 0], ef[f * 3 + 1], ef[f * 3 + 2]
        )
        if fnm[0] == 0 and fnm[1] == 0 and fnm[2] == 0:
            return -1
        if fnm[3] < Scalar[DTYPE](1e-12):
            touching = True
    return 0 if touching else 1


# MuJoCo's `mjMESH_HILLCLIMB_MIN` (`engine_collision_convex.h`): below this many
# vertices the linear scan beats walking the graph, and MuJoCo keeps the scan.
comptime _HILLCLIMB_MIN: Int = 10


@always_inline
def hillclimb_support_index[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
](
    ld_x: Scalar[DTYPE],
    ld_y: Scalar[DTYPE],
    ld_z: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    vert_adr: Int,
    num_verts: Int,
    warm: Int,
) -> Int:
    """LOCAL index of the hull vertex maximising `dot(vertex, ld)`, or -1.

    `ld` is the direction in the MESH'S OWN frame — callers holding a world or
    plane-frame direction must rotate it in first (`quat_rotate_inverse`).

    -1 means "this mesh has no usable graph": either it is below
    `_HILLCLIMB_MIN` vertices or its adjacency is absent, and the caller must
    fall back to a linear scan. Returning a sentinel rather than silently
    scanning here keeps the fallback visible at each call site, which matters
    because the two call sites want DIFFERENT things from it — `_support_mesh`
    needs the point, `_plane_mesh_contacts` needs the index AND the heights it
    computes along the way.

    ⚠⚠ EXTRACTED SO THERE IS ONE HILL CLIMB, NOT TWO. `_plane_mesh_contacts`
    reproduced this function's job as a full linear argmin over every hull
    vertex — the very scan this replaced inside GJK — while its NEIGHBOUR walk
    already used the graph. Measured with per-stage timers, that one scan was
    8.1 µs/step on SO-ARM100 and 11.1 µs/step on SO-ARM101, i.e. 23-25% of the
    whole physics step, and on SO-101 it was a SINGLE call per step. MuJoCo
    calls `mjccd_support` there for exactly this reason
    (`mjc_PlaneConvex`, `engine_collision_convex.c:1010`).
    """
    var graph_head = Int(rebind[Scalar[DTYPE]](mesh_vert_edgeadr[vert_adr]))
    if num_verts < _HILLCLIMB_MIN or graph_head < 0:
        return -1

    # Greedy walk. `imax` is a LOCAL vertex index; `mesh_edges` holds
    # GLOBAL ones, so neighbours are converted on the way in.
    # ⚠ THE GUARD IS NOT PARANOIA ABOUT CALLERS, IT IS THE THING THAT MAKES
    # A CROSSED INDEX HARMLESS. `warm` arrives as -1 on the first call of a
    # pair, and a caller that threaded object 1's index into object 2 would
    # otherwise walk off the end of a smaller mesh into whatever vertices
    # follow it in the model-wide slab. Clamping to 0 turns that into lost
    # speed rather than a support point belonging to another geom.
    var imax = warm if (warm >= 0 and warm < num_verts) else 0
    var best_dot = (
        ld_x * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 0])
        + ld_y * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 1])
        + ld_z * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 2])
    )
    var prev = -1
    # ⚠ THE STEP BUDGET IS A HANG GUARD, NOT AN ALGORITHMIC BOUND. The
    # walk is monotone in `best_dot` and so cannot cycle on a well-formed
    # graph; it terminates in far fewer than `num_verts` steps. A malformed
    # adjacency would otherwise spin forever inside the physics step, and a
    # frozen viewer is the one failure mode here that costs a debugging
    # session rather than a test run.
    var budget = num_verts
    while imax != prev and budget > 0:
        budget -= 1
        prev = imax
        var e = Int(
            rebind[Scalar[DTYPE]](mesh_vert_edgeadr[vert_adr + imax])
        )
        if e < 0:
            break
        while True:
            var nb = Int(rebind[Scalar[DTYPE]](mesh_edges[e]))
            if nb < 0:
                break
            var d = (
                ld_x * rebind[Scalar[DTYPE]](mesh_verts[nb, 0])
                + ld_y * rebind[Scalar[DTYPE]](mesh_verts[nb, 1])
                + ld_z * rebind[Scalar[DTYPE]](mesh_verts[nb, 2])
            )
            if d > best_dot:
                best_dot = d
                imax = nb - vert_adr
            e += 1
    return imax


@always_inline
def _support_mesh[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    vert_adr: Int,
    num_verts: Int,
    mut warm: Int,
) -> InlineArray[Scalar[DTYPE], 3]:
    """Support point on the mesh — HILL CLIMB over the hull edge graph.

    This is the hottest function in mesh collision: GJK and EPA call it 10-30
    times per geom pair per step, and it used to be a LINEAR SCAN over every
    hull vertex. That made narrow-phase cost O(hull size), and the bill arrives
    as soon as a model carries CAD-resolution collision meshes. Two changes,
    each measured on its own (ms per env step, which is `FRAME_SKIP=10` physics
    steps; MuJoCo on the same two XMLs is 0.078 and 0.121 for reference):

                   hull verts    scan     + climb   + warm start
        SO-ARM100      2 551    11.95 ms   5.35 ms    2.82 ms   ->  354 Hz
        SO-ARM101     33 076    76.03 ms  12.79 ms    4.74 ms   ->  211 Hz

    (Both arms have since dropped further, to 1.09 and 1.84 ms, by giving the
    SAP broadphase the bounding-sphere reject it was missing — see
    `broadphase_sap.mojo`. That change removes CALLS to this function; it does
    not make a call cheaper, so the columns above still measure what they say.

    Every column re-measured INTERLEAVED against a pristine worktree of the
    commit before it, reported as the MIN of two rounds — a baseline taken
    earlier in a session has drifted by 1.4-1.7x here before, which would have
    inflated this. The MuJoCo figures are a scale reference, not a like-for-like
    gate: both models sit at `ncon = 0` there, so what is being compared is the
    cost of proving pairs APART, which is where GJK spends its time anyway.)

    ⚠ THE 6.6x IS NOT THE MODEL'S FAULT, AND THAT IS THE MEASUREMENT THAT
    MATTERS. The two arms have IDENTICAL dynamics — `nq = nv = nu = 6`,
    `nbody = 8` — so nothing but collision can differ. MuJoCo steps the very
    same two XMLs at 0.092 ms and 0.136 ms, a ratio of 1.5x against our 6.6x.
    A reference that barely notices 13x the vertices is telling you the scaling
    is the implementation's, not the input's.

    MuJoCo's answer is `mjc_hillclimbSupport` (`engine_collision_convex.c`):
    walk the hull's vertex adjacency, greedily stepping to whichever neighbour
    scores higher on `dir`, until no neighbour improves. On a CONVEX hull the
    dot product has no local maximum other than the global one, so the walk is
    exact — it is not an approximation. It keeps `mjc_meshSupport`'s linear
    scan only below `mjMESH_HILLCLIMB_MIN = 10` vertices, where the graph
    overhead outweighs the scan; `_HILLCLIMB_MIN` below is that constant.

    ⚠ THE ADJACENCY WAS ALREADY THERE. `mesh_vert_edgeadr` / `mesh_edges` are
    built by `build_hull_edge_graph` for the plane-mesh path. Ours needs no
    `vert_globalid` indirection because `mesh_verts` holds hull vertices only —
    `mesh_edges` already stores global vertex ids, `-1` terminated.

    `warm` IS THE WARM START, AND IT IS WHAT MAKES THE WALK CHEAP IN PRACTICE.
    It is a LOCAL vertex index — MuJoCo's `mjCCDObj.meshindex` — carrying the
    vertex the previous call landed on, in and out. GJK and EPA ask for support
    points in directions that move a little at a time, so the previous answer is
    usually a neighbour of this one: from vertex 0 the walk costs O(graph
    diameter), from the previous answer it costs a handful of steps. Callers own
    the state; `gjk_epa_witness` keeps one per object for the whole GJK+EPA run,
    which is exactly the lifetime of MuJoCo's `mjCCDObj`.

    ⚠ THE SEED CANNOT MAKE THE ANSWER WRONG, ONLY SLOW. On a convex hull the
    walk is monotone and has no local maximum to be trapped by, so it converges
    to the extreme vertex from ANY starting vertex. That is why the guard below
    can silently clamp a nonsensical seed instead of trusting the caller: a
    crossed or stale index costs steps, never correctness.

    ⚠ THE SCAN ARM DELIBERATELY IGNORES `warm`. MuJoCo's `mjc_meshSupport` seeds
    its running max from the cache, which only shifts tie-breaking (it still
    scans everything). Leaving it out keeps the scan an INDEPENDENT exhaustive
    reference, which is the only reason the hill climb has anything to be
    checked against in `test_gjk_hillclimb_support.mojo`.

    ⚠ FALLS BACK TO THE SCAN when the graph is absent (`edgeadr < 0`, which is
    how `fields/model.mojo` marks an unbuilt graph) or the mesh is tiny. Both
    paths must return the same point — `test_gjk_hillclimb_support.mojo` pins
    that on real hulls, because a support function that is merely CLOSE yields
    plausible contacts in the wrong place rather than an obvious failure.
    """
    var local_dir = quat_rotate_inverse[DTYPE](
        qx, qy, qz, qw, dir_x, dir_y, dir_z
    )
    var ld_x = local_dir[0]
    var ld_y = local_dir[1]
    var ld_z = local_dir[2]

    var best_x: Scalar[DTYPE] = 0
    var best_y: Scalar[DTYPE] = 0
    var best_z: Scalar[DTYPE] = 0

    var imax = hillclimb_support_index[DTYPE](
        ld_x, ld_y, ld_z,
        mesh_verts, mesh_vert_edgeadr, mesh_edges,
        vert_adr, num_verts, warm,
    )
    if imax >= 0:
        # Hand the landing vertex back so the next call starts here. Dropping
        # this line does not break a single result — it silently reverts the
        # walk to starting from vertex 0 every time, which is why
        # `test_gjk_hillclimb_support.mojo` asserts on `warm` itself.
        warm = imax
        best_x = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 0])
        best_y = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 1])
        best_z = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 2])
    else:
        var best_dot: Scalar[DTYPE] = -1e30
        for i in range(num_verts):
            var vx = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 0])
            var vy = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 1])
            var vz = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 2])
            var d = ld_x * vx + ld_y * vy + ld_z * vz
            if d > best_dot:
                best_dot = d
                best_x = vx
                best_y = vy
                best_z = vz

    var world_pt = quat_rotate[DTYPE](qx, qy, qz, qw, best_x, best_y, best_z)
    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x + world_pt[0]
    result[1] = pos_y + world_pt[1]
    result[2] = pos_z + world_pt[2]
    return result^


@always_inline
def _support[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
](
    geom_type: Int,
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    vert_adr: Int,
    mesh_num_verts: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut warm: Int,
) -> InlineArray[Scalar[DTYPE], 3]:
    """Unified support function — reads mesh verts from the record tensor.

    `warm` is the mesh hill-climb's start vertex, in and out; every other geom
    type leaves it untouched, so one variable per object is enough regardless
    of what that object turns out to be. See `_support_mesh`.
    """
    if geom_type == GEOM_SPHERE:
        return support_sphere[DTYPE](
            dir_x, dir_y, dir_z, pos_x, pos_y, pos_z, radius
        )
    elif geom_type == GEOM_CAPSULE:
        return support_capsule[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            radius,
            half_length,
        )
    elif geom_type == GEOM_BOX:
        return support_box[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            half_x,
            half_y,
            half_z,
        )
    elif geom_type == GEOM_CYLINDER:
        return support_cylinder[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            radius,
            half_length,
        )
    elif geom_type == GEOM_MESH:
        return _support_mesh[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            mesh_verts,
            mesh_vert_edgeadr,
            mesh_edges,
            vert_adr,
            mesh_num_verts,
            warm,
        )
    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x
    result[1] = pos_y
    result[2] = pos_z
    return result^


@always_inline
def _minkowski_support[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
](
    type1: Int,
    p1x: Scalar[DTYPE],
    p1y: Scalar[DTYPE],
    p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE],
    q1y: Scalar[DTYPE],
    q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE],
    hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE],
    hy1: Scalar[DTYPE],
    hz1: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    va1: Int,
    mnv1: Int,
    type2: Int,
    p2x: Scalar[DTYPE],
    p2y: Scalar[DTYPE],
    p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE],
    q2y: Scalar[DTYPE],
    q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE],
    hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE],
    hy2: Scalar[DTYPE],
    hz2: Scalar[DTYPE],
    va2: Int,
    mnv2: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut warm1: Int,
    mut warm2: Int,
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    var s1 = _support[DTYPE](
        type1,
        p1x,
        p1y,
        p1z,
        q1x,
        q1y,
        q1z,
        q1w,
        r1,
        hl1,
        hx1,
        hy1,
        hz1,
        mesh_verts,
        mesh_vert_edgeadr,
        mesh_edges,
        va1,
        mnv1,
        dir_x,
        dir_y,
        dir_z,
        warm1,
    )
    var s2 = _support[DTYPE](
        type2,
        p2x,
        p2y,
        p2z,
        q2x,
        q2y,
        q2z,
        q2w,
        r2,
        hl2,
        hx2,
        hy2,
        hz2,
        mesh_verts,
        mesh_vert_edgeadr,
        mesh_edges,
        va2,
        mnv2,
        -dir_x,
        -dir_y,
        -dir_z,
        warm2,
    )
    return (
        s1[0] - s2[0],
        s1[1] - s2[1],
        s1[2] - s2[2],
        s1[0],
        s1[1],
        s1[2],
        s2[0],
        s2[1],
        s2[2],
    )


def _gjk_intersect[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
](
    mut simplex: InlineArray[Scalar[DTYPE], 36],
    type1: Int,
    p1x: Scalar[DTYPE], p1y: Scalar[DTYPE], p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE], q1y: Scalar[DTYPE], q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE], hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE], hy1: Scalar[DTYPE], hz1: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    va1: Int, mnv1: Int,
    type2: Int,
    p2x: Scalar[DTYPE], p2y: Scalar[DTYPE], p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE], q2y: Scalar[DTYPE], q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE], hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE], hy2: Scalar[DTYPE], hz2: Scalar[DTYPE],
    va2: Int, mnv2: Int,
    mut warm1: Int, mut warm2: Int,
) -> Int:
    """Refine a 4-simplex until it ENCLOSES the origin. MuJoCo's `gjkIntersect`.

    Returns 1 (origin enclosed; `simplex` rewritten as a valid tetrahedron),
    0 (origin proven OUTSIDE the Minkowski difference — no collision), or -1
    (inconclusive; the caller should fall back to its distance subalgorithm).

    ⚠ THIS IS THE PIECE WHOSE ABSENCE BROKE EVERY ATTEMPT TO USE EPA ON
    PENETRATING PRIMITIVES. Without it our GJK terminated on a 2-SIMPLEX for
    pairs with millimetres of genuine overlap — a cylinder 5 mm into the RIM of
    a box — and EPA was then handed a seed that does not contain the origin.
    Two prior fixes failed for the same root reason: `polytope3` reconstructs a
    hexahedron that need not enclose (MuJoCo only ever VALIDATES with
    `testTetra`), and greedily growing the seed toward the most-violated face
    CYCLES, requesting the same support point twice. An inscribed polytope of
    the Minkowski difference need not contain the origin even when the origin
    is inside it, so no amount of patching downstream substitutes for arriving
    with a real enclosing tetrahedron.

    The mechanism: keep the four faces of the current tetrahedron, each with an
    ORIENTED signed distance to the origin. If the smallest is positive the
    origin is inside every face and we are done. Otherwise replace the vertex
    OPPOSITE the most-violated face with the support point along that face's
    normal, which is the only direction that can bring the origin inside. The
    trailing swap keeps the tetrahedron's orientation consistent so the signs
    stay meaningful.

    It also carries a real SEPARATION CERTIFICATE, which the engine has never
    had: if that support point does not reach the origin's side of the face
    plane, the whole Minkowski difference lies beyond it and the geoms provably
    do not overlap.
    """
    var sidx: InlineArray[Int, 4] = [0, 1, 2, 3]

    # ⚠ ORIENT THE TETRAHEDRON FIRST. Every signed distance below carries the
    # simplex's winding, and the whole loop reads those SIGNS to decide whether
    # the origin is inside — so an arbitrarily wound input makes the answer
    # meaningless. MuJoCo can skip this because its `gjk` builds the simplex by
    # appending support points and never reorders them; ours arrives via
    # `_closest_point_on_simplex`, which REDUCES and re-packs the simplex, so
    # the winding is whatever that left behind. Measured symptom: a cylinder
    # touching a box rim reported a confident enclosure with a 0.0999 margin
    # while the origin sat exactly ON the Minkowski boundary — an inscribed
    # tetrahedron cannot strictly contain a boundary point, so the sign was
    # simply being read upside down.
    var o0x = simplex[0 * 9 + 0]
    var o0y = simplex[0 * 9 + 1]
    var o0z = simplex[0 * 9 + 2]
    var e1x = simplex[1 * 9 + 0] - o0x
    var e1y = simplex[1 * 9 + 1] - o0y
    var e1z = simplex[1 * 9 + 2] - o0z
    var e2x = simplex[2 * 9 + 0] - o0x
    var e2y = simplex[2 * 9 + 1] - o0y
    var e2z = simplex[2 * 9 + 2] - o0z
    var e3x = simplex[3 * 9 + 0] - o0x
    var e3y = simplex[3 * 9 + 1] - o0y
    var e3z = simplex[3 * 9 + 2] - o0z
    var tp = (
        (e1y * e2z - e1z * e2y) * e3x
        + (e1z * e2x - e1x * e2z) * e3y
        + (e1x * e2y - e1y * e2x) * e3z
    )
    if tp < Scalar[DTYPE](0):
        sidx[0] = 1
        sidx[1] = 0

    for _ in range(GJK_MAX_ITERATIONS):
        # signed distance to each face; face i is the one OPPOSITE vertex i
        var f0 = _gjk_signed_distance[DTYPE](
            simplex, sidx[2], sidx[1], sidx[3]
        )
        var f1 = _gjk_signed_distance[DTYPE](
            simplex, sidx[0], sidx[2], sidx[3]
        )
        var f2 = _gjk_signed_distance[DTYPE](
            simplex, sidx[1], sidx[0], sidx[3]
        )
        var f3 = _gjk_signed_distance[DTYPE](
            simplex, sidx[0], sidx[1], sidx[2]
        )

        # origin exactly on an affine hull -> the signs cannot converge
        if (
            f0[0] == Scalar[DTYPE](0)
            or f1[0] == Scalar[DTYPE](0)
            or f2[0] == Scalar[DTYPE](0)
            or f3[0] == Scalar[DTYPE](0)
        ):
            return -1

        var i = 0 if f0[0] < f1[0] else 1
        var j = 2 if f2[0] < f3[0] else 3
        var di = f0[0] if i == 0 else f1[0]
        var dj = f2[0] if j == 2 else f3[0]
        var index = i if di < dj else j

        var best = di if di < dj else dj
        # ⚠ ENCLOSURE NEEDS A MARGIN, NOT JUST A POSITIVE SIGN. MuJoCo tests
        # `dist[index] > 0` and separately rejects an EXACTLY zero distance
        # (`!dist[i]`, origin on the affine hull). In floating point an exactly
        # touching pair lands a hair either side of zero instead, and reading
        # that as enclosure sends EPA off to a far face: a cylinder resting
        # exactly on a box rim returned -0.09998 with a sideways normal where
        # the truth is 0. Scale the threshold to the simplex, since these are
        # metres and a fixed epsilon means nothing across models.
        var vscale = Scalar[DTYPE](0)
        for vv in range(4):
            for cc in range(3):
                var av = abs(simplex[sidx[vv] * 9 + cc])
                if av > vscale:
                    vscale = av
        if vscale <= Scalar[DTYPE](0):
            vscale = Scalar[DTYPE](1)
        if best > Scalar[DTYPE](0) and best <= vscale * Scalar[DTYPE](1e-12):
            return -1  # touching: let the distance path report zero depth
        if best > Scalar[DTYPE](0):
            # enclosed: emit the tetrahedron in permutation order
            var out = InlineArray[Scalar[DTYPE], 36](fill=Scalar[DTYPE](0))
            for v in range(4):
                for c in range(9):
                    out[v * 9 + c] = simplex[sidx[v] * 9 + c]
            for c in range(36):
                simplex[c] = out[c]
            return 1

        var nx = Scalar[DTYPE](0)
        var ny = Scalar[DTYPE](0)
        var nz = Scalar[DTYPE](0)
        if index == 0:
            nx = f0[1]
            ny = f0[2]
            nz = f0[3]
        elif index == 1:
            nx = f1[1]
            ny = f1[2]
            nz = f1[3]
        elif index == 2:
            nx = f2[1]
            ny = f2[2]
            nz = f2[3]
        else:
            nx = f3[1]
            ny = f3[2]
            nz = f3[3]

        var w = _minkowski_support[DTYPE](
            type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
            r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
            type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
            r2, hl2, hx2, hy2, hz2, va2, mnv2,
            nx, ny, nz,
            warm1, warm2,
        )
        var tgt = sidx[index]
        simplex[tgt * 9 + 0] = w[0]
        simplex[tgt * 9 + 1] = w[1]
        simplex[tgt * 9 + 2] = w[2]
        simplex[tgt * 9 + 3] = w[3]
        simplex[tgt * 9 + 4] = w[4]
        simplex[tgt * 9 + 5] = w[5]
        simplex[tgt * 9 + 6] = w[6]
        simplex[tgt * 9 + 7] = w[7]
        simplex[tgt * 9 + 8] = w[8]

        # separation certificate
        if nx * w[0] + ny * w[1] + nz * w[2] < Scalar[DTYPE](0):
            return 0

        var a = (index + 1) & 3
        var b = (index + 2) & 3
        var tmp = sidx[a]
        sidx[a] = sidx[b]
        sidx[b] = tmp

    return -1


def gjk_epa_witness[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
](
    type1: Int,
    p1x: Scalar[DTYPE],
    p1y: Scalar[DTYPE],
    p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE],
    q1y: Scalar[DTYPE],
    q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE],
    hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE],
    hy1: Scalar[DTYPE],
    hz1: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    va1: Int,
    mnv1: Int,
    type2: Int,
    p2x: Scalar[DTYPE],
    p2y: Scalar[DTYPE],
    p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE],
    q2y: Scalar[DTYPE],
    q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE],
    hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE],
    hy2: Scalar[DTYPE],
    hz2: Scalar[DTYPE],
    va2: Int,
    mnv2: Int,
    mut wf1: InlineArray[Scalar[DTYPE], 9],
    mut wf2: InlineArray[Scalar[DTYPE], 9],
    mut wx: InlineArray[Scalar[DTYPE], 6],
    mut wf_ok: Int,
    ccd_tol: Scalar[DTYPE] = Scalar[DTYPE](MJ_CCD_TOLERANCE),
    ccd_iter: Int = MJ_CCD_ITERATIONS,
    ccd_margin: Scalar[DTYPE] = Scalar[DTYPE](0),
    # ⚠⚠ OPT-IN, AND THE DEFAULT MUST STAY "DISABLED". Negative means
    # "converge to the true distance", which is what every distance gate in the
    # tree asserts on (`test_gjk_float32_no_phantom_contacts` compares
    # separations of 7-17 cm). Only a caller that uses the result SOLELY for a
    # `dist < margin` contact test may pass a cutoff. See the exit in the loop.
    dist_cutoff: Scalar[DTYPE] = Scalar[DTYPE](-1),
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """GJK distance + EPA penetration depth between two convex shapes
    (verbatim from gjk_epa_gpu; mesh reads via record tensor).

    `wf1` / `wf2` receive the SUPPORT POINTS, on geom 1 and geom 2
    respectively, of the three vertices of the EPA face the answer came from —
    MuJoCo's `pt->verts[face->verts[i]].vert1 / .vert2`. `wx` receives the two
    witness points themselves (`status->x1`, then `status->x2`), whose
    difference is MuJoCo's `dir`. `wf_ok` is set to 1 when they are valid and
    left at 0 on every early return.

    ⚠ `dir` IS NOT `normal * depth` AND MUST NOT BE REBUILT THAT WAY. The
    returned normal has already been negated into the `gi -> gj` convention
    callers expect, while `multicontact` wants `x2 - x1` in the reference's own
    orientation; deriving one from the other is a sign guess in a routine where
    a sign error silently mirrors the whole manifold.

    ⚠ THAT FACE IS THE ONLY THING THAT IDENTIFIES THE CONTACT FEATURE, which is
    why it is plumbed out rather than recomputed. `multicontact` has to know
    whether the contact came from a vertex, an edge or a face of each geom, and
    the ONLY evidence for that is which support points the winning face's three
    vertices resolved to: three distinct points mean a face, two mean an edge,
    one means a vertex. It cannot be recovered from the returned witness point,
    which is already a barycentric BLEND of all three.

    Callers that do not need the feature use `gjk_epa`, which wraps this.
    """
    wf_ok = 0
    # ⚠ ONE HILL-CLIMB SEED PER OBJECT, LIVING FOR THE WHOLE GJK+EPA RUN. This
    # is MuJoCo's `mjCCDObj.meshindex`, whose lifetime is exactly one
    # `mjc_initCCDObj` .. collision call — so the scope here matches by
    # construction rather than by luck. Every support request inside GJK, inside
    # `_gjk_intersect`, and inside EPA resumes from where the last one landed;
    # -1 means "no previous answer", and `_support_mesh` starts at vertex 0.
    # ⚠ THEY MUST NOT BE SWAPPED at a call site: object 1's index is a local
    # vertex id in object 1's mesh and means nothing in object 2's. Crossing
    # them is not a correctness bug (the guard in `_support_mesh` clamps an
    # out-of-range seed, and any in-range seed still converges) but it throws
    # the speed-up away silently, which is the worse failure to debug.
    var warm1 = -1
    var warm2 = -1
    # ===== GJK Phase =====
    var simplex = InlineArray[Scalar[DTYPE], 36](fill=Scalar[DTYPE](0))
    var nsimplex = 0

    var dx = p1x - p2x
    var dy = p1y - p2y
    var dz = p1z - p2z
    var dlen = sqrt(dx * dx + dy * dy + dz * dz)
    if dlen < Scalar[DTYPE](1e-12):
        dx = Scalar[DTYPE](1)
        dy = Scalar[DTYPE](0)
        dz = Scalar[DTYPE](0)
        dlen = Scalar[DTYPE](1)
    dx /= dlen
    dy /= dlen
    dz /= dlen

    var s = _minkowski_support[DTYPE](
        type1,
        p1x,
        p1y,
        p1z,
        q1x,
        q1y,
        q1z,
        q1w,
        r1,
        hl1,
        hx1,
        hy1,
        hz1,
        mesh_verts,
        mesh_vert_edgeadr,
        mesh_edges,
        va1,
        mnv1,
        type2,
        p2x,
        p2y,
        p2z,
        q2x,
        q2y,
        q2z,
        q2w,
        r2,
        hl2,
        hx2,
        hy2,
        hz2,
        va2,
        mnv2,
        dx,
        dy,
        dz,
        warm1,
        warm2,
    )
    simplex[0] = s[0]
    simplex[1] = s[1]
    simplex[2] = s[2]
    simplex[3] = s[3]
    simplex[4] = s[4]
    simplex[5] = s[5]
    simplex[6] = s[6]
    simplex[7] = s[7]
    simplex[8] = s[8]
    nsimplex = 1

    var vx = s[0]
    var vy = s[1]
    var vz = s[2]

    # MuJoCo's `min_norm2` (`engine_collision_gjk.c:218`), NOT a constant — see
    # `_gjk_min_norm2`. The old hardcoded `GJK_TOLERANCE` was a 1e-5 m floor.
    var min_norm2 = _gjk_min_norm2[DTYPE](type1, type2, ccd_margin, ccd_tol)

    for _ in range(GJK_MAX_ITERATIONS):
        var v_dot_v = vx * vx + vy * vy + vz * vz
        if v_dot_v < min_norm2:
            break

        var inv_vlen = Scalar[DTYPE](1) / sqrt(v_dot_v)
        var ndx = -vx * inv_vlen
        var ndy = -vy * inv_vlen
        var ndz = -vz * inv_vlen

        var sn = _minkowski_support[DTYPE](
            type1,
            p1x,
            p1y,
            p1z,
            q1x,
            q1y,
            q1z,
            q1w,
            r1,
            hl1,
            hx1,
            hy1,
            hz1,
            mesh_verts,
            mesh_vert_edgeadr,
            mesh_edges,
            va1,
            mnv1,
            type2,
            p2x,
            p2y,
            p2z,
            q2x,
            q2y,
            q2z,
            q2w,
            r2,
            hl2,
            hx2,
            hy2,
            hz2,
            va2,
            mnv2,
            ndx,
            ndy,
            ndz,
            warm1,
            warm2,
        )

        var w_dot = sn[0] * ndx + sn[1] * ndy + sn[2] * ndz
        var v_dot = vx * ndx + vy * ndy + vz * ndz

        # ── CUTOFF EXIT — MuJoCo's `dist_cutoff` arm of `mj_gjk`
        # (`engine_collision_gjk.c:225`). `nd = -v/|v|`, so `-w_dot` is
        # `dot(w, v)/|v|`, the standard GJK LOWER BOUND on the distance from
        # the origin to the Minkowski difference. Once that bound reaches the
        # cutoff the pair is proven at least `cutoff` apart and no further
        # iteration can change the caller's `dist < margin` answer.
        #
        # ⚠⚠ THIS IS SAFE WHERE THE `gi == 0` CERTIFICATE BELOW WAS NOT, AND
        # THE DIFFERENCE IS THE BOUND. That branch proved "separated" and
        # returned 1e30, which is only equivalent to "no contact" when the
        # margin is 0 — with a margin it silently lost every contact in the
        # band, 0 against MuJoCo's 5 (see the comment there). This exits only
        # when a LOWER BOUND on the true distance has reached the very
        # threshold the caller compares against, so it can cost iterations,
        # never a contact. A penetrating pair has the origin inside, hence
        # `dot(w, v) < 0` and `-w_dot < 0`, so it can never fire on one.
        #
        # ⚠ MuJoCo's OTHER early-out (`!get_dist`, one branch up) returns on
        # ANY separating hyperplane. That one is safe only because
        # `mjc_penetration` INFLATES both geoms by margin first, which we have
        # never ported — do not copy it. See
        # `feedback_copying_control_flow_without_its_precondition`.
        #
        # Measured, SO-ARM101: GJK runs ~15 iterations per call converging to a
        # distance nobody reads; its 4 pairs sit 0.9-7.6 cm apart with margin 0.
        if dist_cutoff >= Scalar[DTYPE](0) and -w_dot >= dist_cutoff:
            wf_ok = 0
            return (-w_dot, Scalar[DTYPE](0), Scalar[DTYPE](0),
                    Scalar[DTYPE](0), ndx, ndy, ndz)
        # ⚠⚠ THE FLOAT32 FLOOR IS NOT A LOOSENING, IT IS WHAT MAKES THE TEST
        # ABLE TO FIRE AT ALL. `w_dot - v_dot` is a difference of two dot
        # products of magnitude |v|, so at float32 its rounding floor is about
        # `1e-7 * |v|` — for robot-scale geometry, HUNDREDS of times above
        # `GJK_TOLERANCE = 1e-10`. Without the relative term the comparison
        # never succeeds, GJK runs to `GJK_MAX_ITERATIONS`, and it returns
        # whatever it is holding at the cap. Measured over a 256-pose sweep of
        # two real hulls, that produced three PHANTOM CONTACTS: `-0.0` returned
        # for pairs float64 places 7.2, 16.5 and 16.9 cm apart. A phantom
        # contact at 17 cm hands the solver a constraint row out of nowhere and
        # is indistinguishable downstream from a real one.
        #
        # ⚠ MUJOCO'S THRESHOLD HERE IS ZERO, AND THAT IS NOT A LOOPHOLE WE CAN
        # COPY. `engine_collision_gjk.c` sets it to 0 for mesh pairs because
        # "if both geoms are discrete, finite convergence is guaranteed" — an
        # exact-arithmetic guarantee that float64 nearly honours and float32
        # does not. It is relative to |v| so it is scale-invariant, unlike the
        # absolute constant it adds to.
        #
        # ⚠ FLOAT64 IS UNTOUCHED — the term is exactly 0 there, so every
        # float64 gate is bit-identical. Which also means none of them covers
        # this; `test_gjk_float32_no_phantom_contacts.mojo` does, and it is RED
        # without the term.
        var gjk_eps = Scalar[DTYPE](GJK_TOLERANCE) + Scalar[DTYPE](
            0.0 if DTYPE == DType.float64 else 1e-6
        ) * sqrt(v_dot_v)
        if w_dot - v_dot < gjk_eps:
            break

        var si = nsimplex * 9
        simplex[si + 0] = sn[0]
        simplex[si + 1] = sn[1]
        simplex[si + 2] = sn[2]
        simplex[si + 3] = sn[3]
        simplex[si + 4] = sn[4]
        simplex[si + 5] = sn[5]
        simplex[si + 6] = sn[6]
        simplex[si + 7] = sn[7]
        simplex[si + 8] = sn[8]
        nsimplex += 1

        # ⚠ THE SIMPLEX IS AT FOUR VERTICES — REFINE IT BEFORE REDUCING IT.
        # MuJoCo calls `gjkIntersect` at exactly this point
        # (`engine_collision_gjk.c:238`, `if (n == 3 && backup_gjk)`), BEFORE
        # the distance subalgorithm runs, because that subalgorithm's job is to
        # find the closest FEATURE and it will happily drop back to a triangle
        # or an edge. For a SEPARATED pair that is what you want; for a
        # PENETRATING one it discards the enclosure and hands EPA a seed that
        # does not contain the origin, which is the single root cause behind
        # three failed attempts to use EPA on penetrating primitives.
        if nsimplex == 4:
            var gi = _gjk_intersect[DTYPE](
                simplex,
                type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
                r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
                r2, hl2, hx2, hy2, hz2, va2, mnv2,
                warm1, warm2,
            )
            if gi == 1:
                # enclosed, and `simplex` now holds a valid tetrahedron
                vx = Scalar[DTYPE](0)
                vy = Scalar[DTYPE](0)
                vz = Scalar[DTYPE](0)
                break
            elif gi == 0:
                # ⚠⚠ PROVEN SEPARATED IS NOT THE SAME AS "FAR ENOUGH TO IGNORE",
                # AND THIS USED TO RETURN 1e30 ON THAT ASSUMPTION. The reasoning
                # was "no collision, so report a separation the caller's
                # `dist < margin` test will reject" — which is right only when
                # `margin` is 0. With a margin, a pair separated by LESS than it
                # is a contact MuJoCo reports and we returned nothing for.
                #
                # Measured, cylinder over a box, `margin=0.01` on each
                # (`includemargin` 0.02):
                #
                #     gap      ours   MuJoCo
                #     -0.0005    5       5
                #      0.0       0       5    <- lost
                #      0.0005    0       5    <- lost
                #      0.001     0       5    <- lost
                #      0.002     1       5
                #      0.02      0       0
                #
                # i.e. it fired exactly in the band that matters and nowhere
                # else — the certificate needs a 4-simplex, which only forms at
                # SMALL separations, so distant pairs never reached this branch
                # and paid nothing for it either.
                #
                # ⚠ MUJOCO NEVER NEEDS THE DISTANCE because `mjc_penetration`
                # sets `dist_cutoff = 0` and INFLATES both geoms by `margin`
                # instead (`mjc_initCCDObj(&obj, m, d, g, margin)`), so a
                # within-margin pair reads as penetrating. We do not inflate, so
                # our caller compares a real distance against `margin` — and
                # that means this path MUST produce one.
                #
                # Falling through costs a few more GJK iterations on
                # near-touching non-colliding pairs and keeps the certificate's
                # real value: it still prevents EPA being handed a non-enclosing
                # seed, because `gi == 1` is what gates that.
                #
                # ⚠ NOTHING IS SAVED OR RESTORED HERE, DELIBERATELY.
                # `_gjk_intersect` rewrites the simplex in place, so the obvious
                # move is to snapshot it and put it back — and a 36-element
                # per-thread `InlineArray` to do that is exactly the shape that
                # silently miscomputes on Metal
                # (`feedback_metal_wide_per_thread_inlinearray_miscompute`).
                # It was written that way first and cost CPU-vs-GPU parity:
                # 9.5e-06 on `test_narrow_phase_pairs_gpu_matches_cpu`, which
                # must be bit-exact. The snapshot is also unnecessary — every
                # vertex `_gjk_intersect` writes is a genuine support point, so
                # what it leaves behind is a valid simplex for the distance
                # subalgorithm, just a different one. Verified: identical
                # contact counts and distances across the whole margin band
                # with and without the restore.
                pass
            # gi == -1: inconclusive, fall through to the subalgorithm below

        # Import and use the CPU closest_point function (works on InlineArray, no List)

        var cp = _closest_point_on_simplex[DTYPE](simplex, nsimplex)
        vx = cp[0]
        vy = cp[1]
        vz = cp[2]
        nsimplex = Int(cp[3])

        if nsimplex == 4:
            vx = Scalar[DTYPE](0)
            vy = Scalar[DTYPE](0)
            vz = Scalar[DTYPE](0)
            break

    # Classify with the SAME quantity the loop converges on. `GJK_TOLERANCE` is
    # a threshold on |v|^2 in the loop above (`v_dot_v < GJK_TOLERANCE`), so
    # comparing |v| against it here meant any run that exited by converging to
    # the origin — |v| anywhere in [1e-10, 1e-5) — was then reported SEPARATED
    # at a hair's-breadth positive distance, and dropped by the caller's
    # `dist < margin` test. A deep penetration would vanish outright: the two
    # exits that mean "origin reached" (tetrahedron enclosed, |v| -> 0) have to
    # agree, or a single ULP decides whether a contact exists at all.
    var dist_sq = vx * vx + vy * vy + vz * vz
    var dist = sqrt(dist_sq)

    if dist_sq > min_norm2:
        # Separated
        var w1x: Scalar[DTYPE] = 0
        var w1y: Scalar[DTYPE] = 0
        var w1z: Scalar[DTYPE] = 0
        var w2x: Scalar[DTYPE] = 0
        var w2y: Scalar[DTYPE] = 0
        var w2z: Scalar[DTYPE] = 0
        for i in range(nsimplex):
            w1x += simplex[i * 9 + 3]
            w1y += simplex[i * 9 + 4]
            w1z += simplex[i * 9 + 5]
            w2x += simplex[i * 9 + 6]
            w2y += simplex[i * 9 + 7]
            w2z += simplex[i * 9 + 8]
        if nsimplex > 0:
            var inv_n = Scalar[DTYPE](1) / Scalar[DTYPE](nsimplex)
            w1x *= inv_n
            w1y *= inv_n
            w1z *= inv_n
            w2x *= inv_n
            w2y *= inv_n
            w2z *= inv_n
        var cx = (w1x + w2x) * Scalar[DTYPE](0.5)
        var cy = (w1y + w2y) * Scalar[DTYPE](0.5)
        var cz = (w1z + w2z) * Scalar[DTYPE](0.5)
        # ⚠ SIGN: this function returns `gi -> gj`, the convention EVERY caller
        # applies (the emit then negates once more to reach the record's
        # `body_b -> body_a`). GJK's `v` and EPA's face normal both point OUT of
        # the Minkowski difference, which is `gj -> gi`, so both return sites
        # negate. See the note at the EPA return.
        var nx = -vx / dist
        var ny = -vy / dist
        var nz = -vz / dist
        return (dist, cx, cy, cz, nx, ny, nz)

    # ===== EPA Phase =====
    # Expanding Polytope Algorithm. Replaces a placeholder that took the
    # CENTRE-LINE direction as the normal and the Minkowski support extent
    # along it as the depth — neither of which is a penetration depth.
    #
    # Measured on a cylinder (r=hl=0.05) against a box face, where the true
    # depth is known analytically:
    #     penetration   placeholder      EPA
    #     0.0005        -1.0995          exact
    #     0.005         -1.095           exact
    #     0.030         -1.07            exact
    # i.e. the placeholder was wrong by 37x at 3 cm and ~2200x at contact,
    # while EPA recovers the analytic depth at every sampled penetration.
    #
    # `simplex` holds `nsimplex` vertices of 9 floats — the Minkowski point
    # (0..2) and the two witness points (3..5, 6..8). EPA needs the witnesses
    # carried through expansion, which is why the vertex stride stays 9.
    var ev = InlineArray[Scalar[DTYPE], EPA_V_CAP * 9](fill=Scalar[DTYPE](0))
    var ef = InlineArray[Int, EPA_F_CAP * 3](fill=0)
    var nev = 0
    var nef = 0

    # ---- seed the polytope -------------------------------------------------
    # GJK exits enclosure with a tetrahedron, which is the seed MuJoCo uses.
    # When it instead converged with |v| -> 0 on a lower-dimensional simplex
    # there is no tetrahedron to hand, and rather than port `polytope2/3/4`
    # (three more degenerate constructions) the polytope is seeded from the six
    # AXIS supports. That octahedron was validated in the same reference run
    # that measured the caps: it converged to the analytic depth on every
    # cylinder/box, box/box and capsule/box case tested.
    # Attempt 1 — GJK's tetrahedron, the seed MuJoCo uses.
    if nsimplex == 4:
        for i in range(4):
            for k in range(9):
                ev[i * 9 + k] = simplex[i * 9 + k]
        nev = 4
        ef[0] = 0
        ef[1] = 1
        ef[2] = 2
        ef[3] = 0
        ef[4] = 1
        ef[5] = 3
        ef[6] = 0
        ef[7] = 2
        ef[8] = 3
        ef[9] = 1
        ef[10] = 2
        ef[11] = 3
        nef = 4

    # Attempt 1b — GJK ended on a TRIANGLE: MuJoCo's `polytope3`.
    #
    # Measured, not guessed: sawyer's obj against the eGripperBase hull leaves
    # GJK with `nsimplex == 3`, and both the tetrahedron seed (absent) and the
    # axis octahedron (degenerate) fail on it, which is why that 27.7 mm
    # contact fell through to the old estimate. `polytope3`
    # (`engine_collision_gjk.c`) builds a hexahedron from the triangle by
    # taking supports along BOTH triangle normals, giving 5 vertices and 6
    # faces that straddle the plane the triangle lies in.
    elif nsimplex == 3:
        for i in range(3):
            for k in range(9):
                ev[i * 9 + k] = simplex[i * 9 + k]
        var e1x = ev[9 + 0] - ev[0]
        var e1y = ev[9 + 1] - ev[1]
        var e1z = ev[9 + 2] - ev[2]
        var e2x = ev[18 + 0] - ev[0]
        var e2y = ev[18 + 1] - ev[1]
        var e2z = ev[18 + 2] - ev[2]
        var tnx = e1y * e2z - e1z * e2y
        var tny = e1z * e2x - e1x * e2z
        var tnz = e1x * e2y - e1y * e2x
        var tln = sqrt(tnx * tnx + tny * tny + tnz * tnz)
        if tln > Scalar[DTYPE](1e-20):
            tnx /= tln
            tny /= tln
            tnz /= tln
            var sp4 = _minkowski_support[DTYPE](
                type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
                r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
                r2, hl2, hx2, hy2, hz2, va2, mnv2,
                tnx, tny, tnz,
                warm1, warm2,
            )
            var sp5 = _minkowski_support[DTYPE](
                type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
                r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
                r2, hl2, hx2, hy2, hz2, va2, mnv2,
                -tnx, -tny, -tnz,
                warm1, warm2,
            )
            ev[27 + 0] = sp4[0]
            ev[27 + 1] = sp4[1]
            ev[27 + 2] = sp4[2]
            ev[27 + 3] = sp4[3]
            ev[27 + 4] = sp4[4]
            ev[27 + 5] = sp4[5]
            ev[27 + 6] = sp4[6]
            ev[27 + 7] = sp4[7]
            ev[27 + 8] = sp4[8]
            ev[36 + 0] = sp5[0]
            ev[36 + 1] = sp5[1]
            ev[36 + 2] = sp5[2]
            ev[36 + 3] = sp5[3]
            ev[36 + 4] = sp5[4]
            ev[36 + 5] = sp5[5]
            ev[36 + 6] = sp5[6]
            ev[36 + 7] = sp5[7]
            ev[36 + 8] = sp5[8]
            nev = 5
            var hex_f: InlineArray[Int, 18] = [
                3, 0, 1, 3, 2, 0, 3, 1, 2, 4, 1, 0, 4, 0, 2, 4, 2, 1,
            ]
            for k in range(18):
                ef[k] = hex_f[k]
            nef = 6

    var seed_code = _epa_seed_contains_origin[DTYPE](ev, ef, nef)

    # Attempt 2 — the six AXIS supports as an octahedron.
    #
    # ⚠ A DEGENERATE TETRAHEDRON IS NOT THE SAME THING AS A TOUCHING PAIR, and
    # conflating them drops real contacts. GJK routinely terminates on a nearly
    # FLAT simplex for mesh pairs (the same coplanar-simplex family as the
    # phantom fixed in 13d7d4bb), which fails the containment test while the
    # geoms are deeply overlapped — sawyer's obj sits 27.7 mm inside the
    # gripper hull there. Rebuilding the seed from axis supports recovers those
    # cases; only when THAT is also degenerate is the origin genuinely on the
    # boundary.
    if seed_code != 1:
        for a in range(6):
            var sdx = Scalar[DTYPE](0)
            var sdy = Scalar[DTYPE](0)
            var sdz = Scalar[DTYPE](0)
            if a == 0:
                sdx = Scalar[DTYPE](1)
            elif a == 1:
                sdx = Scalar[DTYPE](-1)
            elif a == 2:
                sdy = Scalar[DTYPE](1)
            elif a == 3:
                sdy = Scalar[DTYPE](-1)
            elif a == 4:
                sdz = Scalar[DTYPE](1)
            else:
                sdz = Scalar[DTYPE](-1)
            var sp = _minkowski_support[DTYPE](
                type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
                r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
                r2, hl2, hx2, hy2, hz2, va2, mnv2,
                sdx, sdy, sdz,
                warm1, warm2,
            )
            ev[a * 9 + 0] = sp[0]
            ev[a * 9 + 1] = sp[1]
            ev[a * 9 + 2] = sp[2]
            ev[a * 9 + 3] = sp[3]
            ev[a * 9 + 4] = sp[4]
            ev[a * 9 + 5] = sp[5]
            ev[a * 9 + 6] = sp[6]
            ev[a * 9 + 7] = sp[7]
            ev[a * 9 + 8] = sp[8]
        nev = 6
        var oct_f: InlineArray[Int, 24] = [
            0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
            2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
        ]
        for k in range(24):
            ef[k] = oct_f[k]
        nef = 8
        seed_code = _epa_seed_contains_origin[DTYPE](ev, ef, nef)

    # ---- expand ------------------------------------------------------------
    var best_nx = Scalar[DTYPE](0)
    var best_ny = Scalar[DTYPE](0)
    var best_nz = Scalar[DTYPE](1)
    var best_d = Scalar[DTYPE](0)
    var best_face = -1
    var converged = False

    # `ccd_iterations` from model META, but never past what the polytope arrays
    # can hold — MuJoCo grows its polytope on the heap and has no equivalent
    # cap, so a model asking for more than `EPA_ITER_HARD_CAP` gets the arrays'
    # limit and NOT what it asked for. The `nev`/`nef` guard below is the one
    # that actually reports it.
    var _epa_tol = _epa_tolerance[DTYPE](type1, type2, ccd_margin, ccd_tol)
    var _epa_iters = ccd_iter
    if _epa_iters > EPA_ITER_HARD_CAP:
        _epa_iters = EPA_ITER_HARD_CAP
    if _epa_iters < 1:
        _epa_iters = 1
    for _epa_it in range(_epa_iters if seed_code == 1 else 0):
        # closest face to the origin
        best_face = -1
        best_d = Scalar[DTYPE](1e30)
        for f in range(nef):
            var fnm = _epa_face_normal[DTYPE](
                ev, ef[f * 3 + 0], ef[f * 3 + 1], ef[f * 3 + 2]
            )
            if fnm[0] == 0 and fnm[1] == 0 and fnm[2] == 0:
                continue
            if fnm[3] < best_d:
                best_d = fnm[3]
                best_nx = fnm[0]
                best_ny = fnm[1]
                best_nz = fnm[2]
                best_face = f
        if best_face < 0:
            break

        # support along that normal; converged when it adds no depth
        var w = _minkowski_support[DTYPE](
            type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
            r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
            type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
            r2, hl2, hx2, hy2, hz2, va2, mnv2,
            best_nx, best_ny, best_nz,
            warm1, warm2,
        )
        var wd = w[0] * best_nx + w[1] * best_ny + w[2] * best_nz
        if wd - best_d < _epa_tol:
            converged = True
            break

        # ⚠ OVERFLOW IS REPORTED, NOT TRUNCATED. Silently dropping faces would
        # return a plausible but wrong depth from a polytope that stopped
        # growing. Bailing keeps the best face found so far, which is a valid
        # LOWER bound on the depth.
        if nev >= EPA_V_CAP or nef + 3 >= EPA_F_CAP:
            break

        # mark faces the new point can see
        var vis = InlineArray[Bool, EPA_F_CAP](fill=False)
        var nvis = 0
        for f in range(nef):
            var i0 = ef[f * 3 + 0]
            var fnm = _epa_face_normal[DTYPE](
                ev, i0, ef[f * 3 + 1], ef[f * 3 + 2]
            )
            if fnm[0] == 0 and fnm[1] == 0 and fnm[2] == 0:
                continue
            var rx = w[0] - ev[i0 * 9 + 0]
            var ry = w[1] - ev[i0 * 9 + 1]
            var rz = w[2] - ev[i0 * 9 + 2]
            if fnm[0] * rx + fnm[1] * ry + fnm[2] * rz > Scalar[DTYPE](1e-14):
                vis[f] = True
                nvis += 1
        if nvis == 0:
            converged = True
            break

        # Horizon = edges belonging to exactly ONE visible face.
        #
        # ⚠ EDGES ARE UNDIRECTED HERE, DELIBERATELY. The obvious formulation is
        # the directed one — an edge is on the horizon when its REVERSE is not
        # also an edge of a visible face — but that needs consistently wound
        # faces, and these are not wound consistently: `_epa_face_normal`
        # orients the NORMAL by `n . a >= 0` and never reorders the triangle's
        # vertices, so an interior edge can appear as (a0, a1) in both of the
        # faces sharing it. The directed rule then reads it as a horizon edge,
        # stitches a malformed polytope, and EPA stops early — which returns a
        # LOW depth, because face distance climbs monotonically toward the true
        # one. Measured on sawyer's mesh pair: 0.0148582 from the directed rule
        # against 0.0260234 for a correct EPA over the same 81 hull vertices.
        var hor = InlineArray[Int, EPA_F_CAP * 6](fill=0)
        var nhor = 0
        for f in range(nef):
            if not vis[f]:
                continue
            for e in range(3):
                var a0 = ef[f * 3 + e]
                var a1 = ef[f * 3 + (e + 1) % 3]
                var lo = a0 if a0 < a1 else a1
                var hi = a1 if a0 < a1 else a0
                var count = 0
                for g in range(nef):
                    if not vis[g]:
                        continue
                    for e2 in range(3):
                        var b0 = ef[g * 3 + e2]
                        var b1 = ef[g * 3 + (e2 + 1) % 3]
                        var blo = b0 if b0 < b1 else b1
                        var bhi = b1 if b0 < b1 else b0
                        if blo == lo and bhi == hi:
                            count += 1
                if count == 1 and nhor * 2 + 1 < EPA_F_CAP * 6:
                    hor[nhor * 2 + 0] = lo
                    hor[nhor * 2 + 1] = hi
                    nhor += 1
        if nhor < 3 or nef - nvis + nhor > EPA_F_CAP:
            break

        # drop visible faces, keeping the rest compact
        var keep = 0
        for f in range(nef):
            if vis[f]:
                continue
            ef[keep * 3 + 0] = ef[f * 3 + 0]
            ef[keep * 3 + 1] = ef[f * 3 + 1]
            ef[keep * 3 + 2] = ef[f * 3 + 2]
            keep += 1
        nef = keep

        # add w and stitch the horizon to it
        ev[nev * 9 + 0] = w[0]
        ev[nev * 9 + 1] = w[1]
        ev[nev * 9 + 2] = w[2]
        ev[nev * 9 + 3] = w[3]
        ev[nev * 9 + 4] = w[4]
        ev[nev * 9 + 5] = w[5]
        ev[nev * 9 + 6] = w[6]
        ev[nev * 9 + 7] = w[7]
        ev[nev * 9 + 8] = w[8]
        var wi = nev
        nev += 1
        for h in range(nhor):
            ef[nef * 3 + 0] = hor[h * 2 + 0]
            ef[nef * 3 + 1] = hor[h * 2 + 1]
            ef[nef * 3 + 2] = wi
            nef += 1

    # ---- UPPER-BOUND CHECK on the depth ------------------------------------
    # Penetration depth is `min` over all unit directions of the Minkowski
    # support extent h(n) = support_M(n) . n, so EVERY direction sampled gives
    # an upper bound the answer cannot exceed. This is a definition, not an
    # approximation, which makes it a cheap and sound guard against EPA
    # converging on a face that is genuinely on the boundary but is NOT the
    # closest one.
    #
    # ⚠ IT FIRES ON EXACT CONTACT, which is where EPA is least trustworthy and
    # where sawyer's reset pose sits. A cylinder resting exactly on a box
    # returned -0.55 face-on and -0.0999 on the rim, both with confident
    # normals, where the truth is 0 — the sampled bound along the contact
    # normal is 0 and pulls it back. Seven support queries.
    if best_face >= 0:
        var ub = best_d
        var ubx = best_nx
        var uby = best_ny
        var ubz = best_nz
        for ax in range(7):
            var dxx = best_nx
            var dyy = best_ny
            var dzz = best_nz
            if ax == 0:
                dxx = Scalar[DTYPE](1); dyy = Scalar[DTYPE](0); dzz = Scalar[DTYPE](0)
            elif ax == 1:
                dxx = Scalar[DTYPE](-1); dyy = Scalar[DTYPE](0); dzz = Scalar[DTYPE](0)
            elif ax == 2:
                dxx = Scalar[DTYPE](0); dyy = Scalar[DTYPE](1); dzz = Scalar[DTYPE](0)
            elif ax == 3:
                dxx = Scalar[DTYPE](0); dyy = Scalar[DTYPE](-1); dzz = Scalar[DTYPE](0)
            elif ax == 4:
                dxx = Scalar[DTYPE](0); dyy = Scalar[DTYPE](0); dzz = Scalar[DTYPE](1)
            elif ax == 5:
                dxx = Scalar[DTYPE](0); dyy = Scalar[DTYPE](0); dzz = Scalar[DTYPE](-1)
            var sw = _minkowski_support[DTYPE](
                type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
                r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
                r2, hl2, hx2, hy2, hz2, va2, mnv2,
                dxx, dyy, dzz,
                warm1, warm2,
            )
            var hh = sw[0] * dxx + sw[1] * dyy + sw[2] * dzz
            if hh < ub:
                ub = hh
                ubx = dxx
                uby = dyy
                ubz = dzz
        if ub < best_d:
            best_d = ub if ub > Scalar[DTYPE](0) else Scalar[DTYPE](0)
            best_nx = ubx
            best_ny = uby
            best_nz = ubz

    # ---- witness points on the closest face --------------------------------
    # Barycentric coordinates of the origin's projection onto the face, applied
    # to the stored witness points — MuJoCo's `epaWitness`. Degenerate faces
    # fall back to the face centroid, which is what the barycentric solve
    # converges to as the triangle collapses.
    if best_face >= 0 and (converged or best_d < Scalar[DTYPE](1e29)):
        var i0 = ef[best_face * 3 + 0]
        var i1 = ef[best_face * 3 + 1]
        var i2 = ef[best_face * 3 + 2]
        var px = best_nx * best_d
        var py = best_ny * best_d
        var pz = best_nz * best_d
        var v0x = ev[i1 * 9 + 0] - ev[i0 * 9 + 0]
        var v0y = ev[i1 * 9 + 1] - ev[i0 * 9 + 1]
        var v0z = ev[i1 * 9 + 2] - ev[i0 * 9 + 2]
        var v1x = ev[i2 * 9 + 0] - ev[i0 * 9 + 0]
        var v1y = ev[i2 * 9 + 1] - ev[i0 * 9 + 1]
        var v1z = ev[i2 * 9 + 2] - ev[i0 * 9 + 2]
        var v2x = px - ev[i0 * 9 + 0]
        var v2y = py - ev[i0 * 9 + 1]
        var v2z = pz - ev[i0 * 9 + 2]
        var d00 = v0x * v0x + v0y * v0y + v0z * v0z
        var d01 = v0x * v1x + v0y * v1y + v0z * v1z
        var d11 = v1x * v1x + v1y * v1y + v1z * v1z
        var d20 = v2x * v0x + v2y * v0y + v2z * v0z
        var d21 = v2x * v1x + v2y * v1y + v2z * v1z
        var den = d00 * d11 - d01 * d01
        var l0 = Scalar[DTYPE](1) / Scalar[DTYPE](3)
        var l1 = l0
        var l2 = l0
        if abs(den) > Scalar[DTYPE](1e-20):
            l1 = (d11 * d20 - d01 * d21) / den
            l2 = (d00 * d21 - d01 * d20) / den
            l0 = Scalar[DTYPE](1) - l1 - l2
        var w1x = (
            l0 * ev[i0 * 9 + 3] + l1 * ev[i1 * 9 + 3] + l2 * ev[i2 * 9 + 3]
        )
        var w1y = (
            l0 * ev[i0 * 9 + 4] + l1 * ev[i1 * 9 + 4] + l2 * ev[i2 * 9 + 4]
        )
        var w1z = (
            l0 * ev[i0 * 9 + 5] + l1 * ev[i1 * 9 + 5] + l2 * ev[i2 * 9 + 5]
        )
        var w2x = (
            l0 * ev[i0 * 9 + 6] + l1 * ev[i1 * 9 + 6] + l2 * ev[i2 * 9 + 6]
        )
        var w2y = (
            l0 * ev[i0 * 9 + 7] + l1 * ev[i1 * 9 + 7] + l2 * ev[i2 * 9 + 7]
        )
        var w2z = (
            l0 * ev[i0 * 9 + 8] + l1 * ev[i1 * 9 + 8] + l2 * ev[i2 * 9 + 8]
        )

        # The winning face's three vertices, as support points on each geom.
        # Written with CONSTANT indices on the destination and only the polytope
        # vertex id varying, so no per-thread array is indexed by a runtime
        # value on the way out — see
        # `feedback_metal_wide_per_thread_inlinearray_miscompute`.
        wf1[0] = ev[i0 * 9 + 3]
        wf1[1] = ev[i0 * 9 + 4]
        wf1[2] = ev[i0 * 9 + 5]
        wf1[3] = ev[i1 * 9 + 3]
        wf1[4] = ev[i1 * 9 + 4]
        wf1[5] = ev[i1 * 9 + 5]
        wf1[6] = ev[i2 * 9 + 3]
        wf1[7] = ev[i2 * 9 + 4]
        wf1[8] = ev[i2 * 9 + 5]
        wf2[0] = ev[i0 * 9 + 6]
        wf2[1] = ev[i0 * 9 + 7]
        wf2[2] = ev[i0 * 9 + 8]
        wf2[3] = ev[i1 * 9 + 6]
        wf2[4] = ev[i1 * 9 + 7]
        wf2[5] = ev[i1 * 9 + 8]
        wf2[6] = ev[i2 * 9 + 6]
        wf2[7] = ev[i2 * 9 + 7]
        wf2[8] = ev[i2 * 9 + 8]
        wx[0] = w1x
        wx[1] = w1y
        wx[2] = w1z
        wx[3] = w2x
        wx[4] = w2y
        wx[5] = w2z
        wf_ok = 1
        # ⚠ THIS SIGN WAS WRONG AND NOTHING CAUGHT IT FOR THE MESH PATH.
        # `test_narrow_phase_pairs` anchors contact DIRECTION against MuJoCo for
        # the primitive pairs, but no gate covered direction for a MESH pair, so
        # every mesh contact this engine produced pointed the wrong way.
        # Measured on sawyer's obj/eGripperBase contact, where MuJoCo reports
        # `geom1(36) -> geom2(27)` = (-8.6e-05, 1.13e-03, -0.999999) and the
        # record convention makes that the same direction we store: ours came
        # out +z. The re-route of cylinder/box through this same function is
        # what exposed it, via a `dir err 1.9999999999976286` — a full
        # reversal — on a pair that IS anchored.
        #
        # EPA's face normal points out of the Minkowski difference, i.e.
        # `gj -> gi`, so returning it negated gives the `gi -> gj` that callers
        # expect. The previous code negated in the comment's direction but
        # against the wrong baseline.
        return (
            -best_d,
            (w1x + w2x) * Scalar[DTYPE](0.5),
            (w1y + w2y) * Scalar[DTYPE](0.5),
            (w1z + w2z) * Scalar[DTYPE](0.5),
            best_nx,
            best_ny,
            best_nz,
        )

    # EPA produced no valid face. That is not a failure needing a guess: it
    # means the polytope is degenerate because the origin lies ON the boundary
    # of the Minkowski difference, i.e. the geoms TOUCH at zero depth.
    #
    # ⚠ This used to fall through to a centre-line estimate — normal = the line
    # between geom origins, depth = the Minkowski support extent along it —
    # which returned -1.1 for a cylinder resting exactly on a box where the
    # true depth is 0. Sawyer's canonical reset pose is exactly that case (the
    # obj's bottom face sits exactly on the table top), so the guess was not
    # hypothetical. MuJoCo reports NO contact there, and returning 0 reproduces
    # that: the caller admits a contact on `dist < margin`, and margin is 0.
    var fallback_nx = p1x - p2x
    var fallback_ny = p1y - p2y
    var fallback_nz = p1z - p2z
    var fallback_len = sqrt(
        fallback_nx * fallback_nx
        + fallback_ny * fallback_ny
        + fallback_nz * fallback_nz
    )
    if fallback_len < Scalar[DTYPE](1e-10):
        fallback_nx = Scalar[DTYPE](0)
        fallback_ny = Scalar[DTYPE](0)
        fallback_nz = Scalar[DTYPE](1)
    else:
        fallback_nx /= fallback_len
        fallback_ny /= fallback_len
        fallback_nz /= fallback_len
    var pen_depth = Scalar[DTYPE](0)
    # The same upper bound applies here, and this path needs it MORE: the
    # centre-line estimate is not a depth at all, so bounding it by the
    # smallest sampled support extent is the only thing keeping it sane. A
    # cylinder touching a box expressed as a mesh returned -1.1 -- the full
    # Minkowski extent -- where the truth is 0.
    var fb_ub = Scalar[DTYPE](1e30)
    for ax in range(6):
        var dxx = Scalar[DTYPE](0)
        var dyy = Scalar[DTYPE](0)
        var dzz = Scalar[DTYPE](0)
        if ax == 0:
            dxx = Scalar[DTYPE](1)
        elif ax == 1:
            dxx = Scalar[DTYPE](-1)
        elif ax == 2:
            dyy = Scalar[DTYPE](1)
        elif ax == 3:
            dyy = Scalar[DTYPE](-1)
        elif ax == 4:
            dzz = Scalar[DTYPE](1)
        else:
            dzz = Scalar[DTYPE](-1)
        var sw = _minkowski_support[DTYPE](
            type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
            r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
            type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
            r2, hl2, hx2, hy2, hz2, va2, mnv2,
            dxx, dyy, dzz,
            warm1, warm2,
        )
        var hh = sw[0] * dxx + sw[1] * dyy + sw[2] * dzz
        if hh < fb_ub:
            fb_ub = hh
            fallback_nx = dxx
            fallback_ny = dyy
            fallback_nz = dzz
    if seed_code < 0:
        # Degenerate seed: EPA never ran, so fall back to the estimate that
        # shipped before it existed. Wrong depth, but it KEEPS the contact,
        # which is strictly the prior behaviour rather than a new regression.
        var s_fwd = _minkowski_support[DTYPE](
            type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
            r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
            type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
            r2, hl2, hx2, hy2, hz2, va2, mnv2,
            fallback_nx, fallback_ny, fallback_nz,
            warm1, warm2,
        )
        pen_depth = _dot3[DTYPE](
            s_fwd[0], s_fwd[1], s_fwd[2],
            fallback_nx, fallback_ny, fallback_nz,
        )
        if pen_depth > Scalar[DTYPE](0):
            pen_depth = -pen_depth
        if fb_ub < Scalar[DTYPE](1e29):
            var cap = -fb_ub if fb_ub > Scalar[DTYPE](0) else Scalar[DTYPE](0)
            if pen_depth < cap:
                pen_depth = cap

    var contact_x = (p1x + p2x) * Scalar[DTYPE](0.5)
    var contact_y = (p1y + p2y) * Scalar[DTYPE](0.5)
    var contact_z = (p1z + p2z) * Scalar[DTYPE](0.5)
    return (
        pen_depth,
        contact_x,
        contact_y,
        contact_z,
        fallback_nx,
        fallback_ny,
        fallback_nz,
    )


@always_inline
def gjk_epa[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
](
    type1: Int,
    p1x: Scalar[DTYPE], p1y: Scalar[DTYPE], p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE], q1y: Scalar[DTYPE], q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE], hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE], hy1: Scalar[DTYPE], hz1: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    va1: Int, mnv1: Int,
    type2: Int,
    p2x: Scalar[DTYPE], p2y: Scalar[DTYPE], p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE], q2y: Scalar[DTYPE], q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE], hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE], hy2: Scalar[DTYPE], hz2: Scalar[DTYPE],
    va2: Int, mnv2: Int,
    ccd_tol: Scalar[DTYPE] = Scalar[DTYPE](MJ_CCD_TOLERANCE),
    ccd_iter: Int = MJ_CCD_ITERATIONS,
    ccd_margin: Scalar[DTYPE] = Scalar[DTYPE](0),
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """`gjk_epa_witness` for callers that only want `(dist, pos, normal)`.

    Kept as a separate entry point rather than making the witness arguments
    optional so that the sixteen existing call sites — two narrow phases, the
    perturbation loop and ten tests — are untouched by the native
    multi-contact work. The scratch arrays are dead stores everywhere the
    result is discarded.
    """
    var wf1 = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var wf2 = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var wx = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    var wf_ok = 0
    return gjk_epa_witness[DTYPE](
        type1,
        p1x, p1y, p1z, q1x, q1y, q1z, q1w,
        r1, hl1, hx1, hy1, hz1,
        mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
        type2,
        p2x, p2y, p2z, q2x, q2y, q2z, q2w,
        r2, hl2, hx2, hy2, hz2,
        va2, mnv2,
        wf1, wf2, wx, wf_ok,
        ccd_tol, ccd_iter, ccd_margin,
    )
