"""`mj_rayMesh` — ray against a mesh's ORIGINAL triangles.

`engine_ray.c:952`, which is a bounding-box reject followed by `mju_rayTree`
(`:771`) — a BVH walk whose every leaf runs `ray_triangle` and keeps the
nearest. **The BVH is an acceleration structure and nothing else**: culling a
node removes only triangles the ray provably misses, so a linear sweep over the
same triangles returns the identical distance and normal. That is why this is a
loop and not a tree, and it is the one thing to re-check before adding a BVH —
the answer must not move, only the time.

⚠⚠ THE HULL CANNOT ANSWER THIS QUERY, which is why `Model.mesh_tris` had to
exist. `Model.mesh_verts` is the CONVEX HULL — right for collision, and MuJoCo
collides hulls too — but a ray aimed into a bracket's cutout hits hull where
the real part has a hole. The three consumers of this package (the
`rangefinder` sensor, studio picking, and a camera) all care about the hole.

⚠ THE SOUP IS OFF BY DEFAULT. `nmesh_tri` defaults to 0, so a model built
without a triangle budget carries no soup and `ntri` here is 0 — and this
returns NO HIT for every ray. That is a capacity being read as an absence of
geometry, so `ntri` is an explicit parameter rather than something inferred
from a table: a caller that forgets it gets an obviously empty answer at the
call site rather than a silently invisible mesh.
"""

from layout import Layout, LayoutTensor

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from .geom import RAY_NO_HIT, ray_map, ray_box
from .triangle import ray_basis, ray_triangle


@always_inline
def _v3[
    DTYPE: DType, L_TRI: Layout
](
    tri: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin], o: Int
) -> Vec3Generic[DTYPE]:
    """Three consecutive floats as a vertex."""
    return Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](tri[o + 0]),
        rebind[Scalar[DTYPE]](tri[o + 1]),
        rebind[Scalar[DTYPE]](tri[o + 2]),
    )


def ray_mesh[
    DTYPE: DType, L_TRI: Layout
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    half_extents: Vec3Generic[DTYPE],
    tri: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    triadr: Int,
    ntri: Int,
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """Distance to the mesh surface and its world-frame normal.

    `tri` is `Model.mesh_tris` — nine floats per triangle, in the mesh's
    principal frame, the same frame the geom's pose assumes. `triadr` is this
    mesh's first TRIANGLE (not float, not vertex) and `ntri` its count.

    `half_extents` is the geom's bounding box, `mjModel.geom_size` for a mesh.
    ⚠ IT IS A PARAMETER RATHER THAN A DERIVED VALUE ON PURPOSE. The reference
    rejects on it before touching a triangle, so a box that is too SMALL
    silently loses hits — and whether our `geom_size` for a mesh is MuJoCo's
    AABB is a question for the gate to answer, not for this routine to assume.
    Pass a box you trust; passing one that is too large only costs time.
    """
    var zero = Vec3Generic[DTYPE](0, 0, 0)
    if ntri <= 0:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    # The reference's bounding-box reject, in the geom's own frame.
    var bb = ray_box[DTYPE](pos, quat, half_extents, pnt, vec)
    if bb[0] < 0:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    var m = ray_map[DTYPE](pos, quat, pnt, vec)
    var lpnt = m[0]
    var lvec = m[1]
    var basis = ray_basis[DTYPE](lvec)
    var b0 = basis[0]
    var b1 = basis[1]

    var x = Scalar[DTYPE](RAY_NO_HIT)
    var normal_local = zero

    for t in range(ntri):
        var o = (triadr + t) * 9
        var r = ray_triangle[DTYPE](
            _v3[DTYPE, L_TRI](tri, o + 0),
            _v3[DTYPE, L_TRI](tri, o + 3),
            _v3[DTYPE, L_TRI](tri, o + 6),
            lpnt, lvec, b0, b1,
        )
        # ⚠ `>= 0` AND `< x`, in that order, matching the reference. A
        # triangle behind the ray comes back NEGATIVE from `ray_triangle`
        # (the plane intersection is deliberately unclamped) and would
        # otherwise win every comparison against a positive distance.
        if r[0] >= 0 and (x < 0 or r[0] < x):
            x = r[0]
            normal_local = r[1]

    if x < 0:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)
    return (x, quat.rotate_vec(normal_local))
