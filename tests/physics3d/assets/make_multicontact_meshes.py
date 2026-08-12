"""Generate the binary STL fixtures used by the mesh-manifold gates.

Two shapes, chosen so the polygon builder is exercised in both regimes:

  cube.stl    6 QUAD faces, each built by merging TWO coplanar triangles.
              This is the case that matters — MuJoCo's compiler merges
              coplanar triangles into polygons, and a builder that skips the
              merge leaves 12 triangles whose face normals still "work" but
              whose clipped manifold is a triangle, not the square face.
  hex.stl     A hexagonal prism: 2 HEXAGONS + 6 quads, so one mesh carries
              polygons of two different vertex counts and the per-polygon
              addressing is actually exercised (a builder that hardcodes 4
              passes on cube). The hexagon merges SIX coplanar triangles, not
              two, so the merge is exercised beyond a single pair as well.

Both are written as BINARY STL because `mojo_rl/render/stl_loader.mojo` reads
binary only. The facet normals STL stores are IGNORED by our loader (it keeps
positions and rebuilds the hull), so they are written correctly but nothing
depends on them.

⚠ BOTH SHAPES ARE CENTRALLY SYMMETRIC WITH THREE DISTINCT EXTENTS, AND THAT IS
A REQUIREMENT, NOT A STYLE CHOICE. MuJoCo's compiler translates every mesh to
its centre of mass and rotates it to its principal inertia axes, so a mesh whose
CoM is off-origin comes back in a DIFFERENT LOCAL FRAME than the STL declared —
which makes any local-frame comparison of vertices, polygons or normals against
`m.mesh_*` meaningless. The first version of this file used a triangular prism
and MuJoCo moved it. Central symmetry puts the CoM at the origin; three distinct
extents keep the inertia eigenvalues distinct so the principal axes are the
coordinate axes and the rotation is identity. `test_mesh_polygons_vs_mujoco`
asserts the vertices came back unmoved, so a future fixture that violates this
fails loudly instead of silently comparing two different solids.

Run: pixi run python tests/physics3d/assets/make_multicontact_meshes.py
"""
import struct
from pathlib import Path

HERE = Path(__file__).parent


def tri_normal(a, b, c):
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    n = [u[1] * v[2] - u[2] * v[1],
         u[2] * v[0] - u[0] * v[2],
         u[0] * v[1] - u[1] * v[0]]
    ln = sum(x * x for x in n) ** 0.5
    return [x / ln for x in n] if ln else [0.0, 0.0, 0.0]


def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for (a, b, c) in tris:
            n = tri_normal(a, b, c)
            f.write(struct.pack("<3f", *n))
            for v in (a, b, c):
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))
    print(f"wrote {path}  ({len(tris)} triangles, {path.stat().st_size} bytes)")


def cube(h=0.05):
    v = [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)]
    # index = 4*xi + 2*yi + zi
    def i(xi, yi, zi):
        return v[4 * xi + 2 * yi + zi]
    quads = [
        # -x face                      +x face
        [i(0, 0, 0), i(0, 0, 1), i(0, 1, 1), i(0, 1, 0)],
        [i(1, 0, 0), i(1, 1, 0), i(1, 1, 1), i(1, 0, 1)],
        # -y                            +y
        [i(0, 0, 0), i(1, 0, 0), i(1, 0, 1), i(0, 0, 1)],
        [i(0, 1, 0), i(0, 1, 1), i(1, 1, 1), i(1, 1, 0)],
        # -z                            +z
        [i(0, 0, 0), i(0, 1, 0), i(1, 1, 0), i(1, 0, 0)],
        [i(0, 0, 1), i(1, 0, 1), i(1, 1, 1), i(0, 1, 1)],
    ]
    tris = []
    for q in quads:
        tris.append((q[0], q[1], q[2]))
        tris.append((q[0], q[2], q[3]))
    return tris


def hex_prism(rx=0.04, ry=0.06, hz=0.08):
    """Hexagonal prism, extruded along z. Centrally symmetric; rx != ry != hz.

    ⚠ THE EXTENTS ARE CHOSEN SO MuJoCo'S PRINCIPAL-AXIS SORT IS THE IDENTITY,
    and they are not free to change. The compiler orders the principal moments
    DESCENDING along x, y, z; for a prism about z that means

        Ixx ~ ry^2/4 + hz^2/3  >  Iyy ~ rx^2/4 + hz^2/3  >  Izz ~ (rx^2+ry^2)/4

    i.e. ry > rx (first inequality) and hz large enough (second). The natural
    reading — "make it a squat prism, largest extent along x" — gives a 90 deg
    rotation about y, and "largest extent along z" gives 90 deg about z. Both
    were measured. See the module docstring for why a rotated fixture is
    useless here.
    """
    import math
    ring_hi, ring_lo = [], []
    for k in range(6):
        a = math.pi / 3 * k
        x, y = rx * math.cos(a), ry * math.sin(a)
        ring_hi.append((x, y, hz))
        ring_lo.append((x, y, -hz))
    tris = []
    # the two hexagonal caps, fanned from vertex 0 (four coplanar triangles each)
    for k in range(1, 5):
        tris.append((ring_hi[0], ring_hi[k], ring_hi[k + 1]))
        tris.append((ring_lo[0], ring_lo[k + 1], ring_lo[k]))
    # six rectangular sides, each split into two coplanar triangles
    for k in range(6):
        j = (k + 1) % 6
        tris.append((ring_lo[k], ring_lo[j], ring_hi[j]))
        tris.append((ring_lo[k], ring_hi[j], ring_hi[k]))
    return tris


if __name__ == "__main__":
    write_stl(HERE / "mc_cube.stl", cube())
    write_stl(HERE / "mc_hex.stl", hex_prism())
