"""Wavefront OBJ loader — the other half of Menagerie.

`mujoco_menagerie` ships **1184 `.obj` files across 33 models** against 1129
`.stl`, so roughly half the library was unreadable: `load_stl` read bytes
80-84 of an OBJ's TEXT as a little-endian triangle count and reported
"STL file too small: expected 46324738584 bytes, got 2193823". A wrong
diagnosis pointing at the right file, which is worse than none — it sends you
looking at the file's size.

⚠ THE DISPATCH IS IN `load_stl` BY EXTENSION, not at the three call sites
(renderer, mesh inertia, convex hull). One of those would have been missed.

## What this parses, and what it ignores

`v x y z` and `f` — that is all a collision hull or a rendered surface needs.

* **`f` may be a POLYGON, not a triangle.** Menagerie's exports are largely
  quads. ⚠⚠ MuJoCo does NOT fan them — see `_triangulate_face` below. Fanning
  a quad picks the WRONG diagonal about half the time and puts a
  7.5e-04 relative error into the mesh's volume, which reaches the model as a
  slightly wrong body mass.
* **`f` indices are 1-BASED and may be negative** (relative to the end).
  Both forms appear in the wild; getting the sign wrong reads a vertex from
  the far end of the file and produces a mesh that is geometrically absurd
  rather than empty.
* **`v/vt/vn` slashes are stripped** — only the position index is taken.
* `vn` is IGNORED and normals are computed per face, matching what
  `load_stl` does. A shared vertex therefore appears once per face, which is
  what the flat-shaded renderer and the hull builder both expect.
* `mtllib` / `usemtl` / `o` / `g` / `s` are skipped. MJCF carries the material
  separately, so an OBJ's own material would be a second, conflicting source.
"""

from std.math import sqrt, fma, abs as math_abs

from .gpu_types import GPUVertex, MeshData


def _is_ws(c: String) -> Bool:
    return c == " " or c == "\t" or c == "\r"


def _tokens(line: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for i in range(line.byte_length()):
        var c = String(line[byte = i : i + 1])
        if _is_ws(c):
            if cur.byte_length() > 0:
                out.append(cur)
                cur = String("")
        else:
            cur += c
    if cur.byte_length() > 0:
        out.append(cur)
    return out^


def _first_index(tok: String) -> Int:
    """`12`, `12/3`, `12//4`, `12/3/4` -> 12. Empty/invalid -> 0."""
    var e = 0
    var n = tok.byte_length()
    while e < n and String(tok[byte = e : e + 1]) != "/":
        e += 1
    var head = String(tok[byte=0:e])
    if head.byte_length() == 0:
        return 0
    var neg = False
    var start = 0
    if String(head[byte=0:1]) == "-":
        neg = True
        start = 1
    var v = 0
    for i in range(start, head.byte_length()):
        var c = String(head[byte = i : i + 1])
        if c < "0" or c > "9":
            return 0
        v = v * 10 + (ord(c) - ord("0"))
    return -v if neg else v


def _f32(tok: String) raises -> Float32:
    """⚠ A MALFORMED NUMBER IS 0, NOT A RAISE. Menagerie exports are machine
    written, but an OBJ with one bad line should lose that vertex rather than
    the whole robot."""
    if tok.byte_length() == 0:
        return Float32(0)
    try:
        return Float32(Float64(tok))
    except:
        return Float32(0)


@always_inline
def _sq_len_fma(ex: Float32, ey: Float32, ez: Float32) -> Float32:
    """`e.x*e.x + e.y*e.y + e.z*e.z` in **float32, with FMA contraction**.

    ⚠⚠ THE FMA IS NOT AN OPTIMISATION, IT IS PART OF THE ANSWER. This value
    only ever feeds the `sqr02 < sqr13` comparison in `_triangulate_face`, and
    on a quad whose two diagonals are equal (a mirrored face — Menagerie is
    full of them) the plain float32 sum lands one ulp apart and picks the
    other diagonal. Measured over Menagerie's 66 `.obj` meshes with a
    non-triangular face: plain float32 reproduces MuJoCo on 62, float64 on 64,
    and the contracted float32 on **all 66**. clang contracts
    `a*a + b*b + c*c` within the statement by default, which is what the
    shipped `tinyobjloader` is compiled with.
    """
    var t = ex * ex
    t = fma(ey, ey, t)
    t = fma(ez, ez, t)
    return t


@always_inline
def _axis_of(
    vx: List[Float32], vy: List[Float32], vz: List[Float32], i: Int, ax: Int
) -> Float32:
    if ax == 0:
        return vx[i]
    if ax == 1:
        return vy[i]
    return vz[i]


def _pnpoly3(
    px: InlineArray[Float32, 3],
    py: InlineArray[Float32, 3],
    tx: Float32,
    ty: Float32,
) -> Bool:
    """`tinyobj::pnpoly` at nvert = 3 — the crossing-number test, verbatim."""
    var c = False
    var j = 2
    for i in range(3):
        if ((py[i] > ty) != (py[j] > ty)) and (
            tx
            < (px[j] - px[i]) * (ty - py[i]) / (py[j] - py[i]) + px[i]
        ):
            c = not c
        j = i
    return c


def _triangulate_face(
    vx: List[Float32],
    vy: List[Float32],
    vz: List[Float32],
    poly: List[Int],
) -> List[Int]:
    """`tinyobj::exportGroupsToShape`'s triangulation, transcribed.

    MuJoCo loads every `.obj` through the `obj_decoder` plugin, which hands
    the file to `tinyobjloader` with the default `triangulate = true`. So the
    triangles a mesh ends up with are TINYOBJLOADER'S, and they are not a fan:

    * **A quad splits along the SHORTER DIAGONAL** — `[0,1,2] + [0,2,3]` when
      `|v2-v0|^2 < |v3-v1|^2`, and `[0,1,3] + [1,2,3]` otherwise. Note the
      comparison is strict, so an exact tie takes the SECOND form. Fanning
      always takes the first.
    * **Anything larger is EAR-CLIPPED** in a 2D projection onto the two axes
      picked by the first non-degenerate corner. ⚠ The loop gives up after
      `remainingIterations` unproductive passes and the leftover polygon is
      then DROPPED, so an n-gon can yield FEWER than n-2 triangles — MuJoCo
      keeps 23,776 faces for `link_SG3_gripper_body` where a fan gives 23,788.
      Reproducing that is the difference between matching the reference and
      merely being reasonable.

    Verified against `mjModel.mesh_face` on **all 66** Menagerie `.obj` meshes
    that have a non-triangular face (agility_cassie, arx_l5, google_robot,
    hello_robot_stretch_3, trossen_wxai): identical faces, identical order.

    Returns a flat list of 3 indices per triangle.
    """
    var out = List[Int]()
    var n = len(poly)
    if n < 3:
        return out^
    if n == 3:
        out.append(poly[0])
        out.append(poly[1])
        out.append(poly[2])
        return out^
    if n == 4:
        var i0 = poly[0]
        var i1 = poly[1]
        var i2 = poly[2]
        var i3 = poly[3]
        var s02 = _sq_len_fma(
            vx[i2] - vx[i0], vy[i2] - vy[i0], vz[i2] - vz[i0]
        )
        var s13 = _sq_len_fma(
            vx[i3] - vx[i1], vy[i3] - vy[i1], vz[i3] - vz[i1]
        )
        if s02 < s13:
            out.append(i0); out.append(i1); out.append(i2)
            out.append(i0); out.append(i2); out.append(i3)
        else:
            out.append(i0); out.append(i1); out.append(i3)
            out.append(i1); out.append(i2); out.append(i3)
        return out^

    # ── the two axes to clip in: the first corner that is not collinear ─────
    var ax0 = 1
    var ax1 = 2
    comptime F32_EPS = Float32(1.1920929e-07)
    for k in range(n):
        var a = poly[(k + 0) % n]
        var b = poly[(k + 1) % n]
        var c = poly[(k + 2) % n]
        var e0x = vx[b] - vx[a]
        var e0y = vy[b] - vy[a]
        var e0z = vz[b] - vz[a]
        var e1x = vx[c] - vx[b]
        var e1y = vy[c] - vy[b]
        var e1z = vz[c] - vz[b]
        var cx = math_abs(e0y * e1z - e0z * e1y)
        var cy = math_abs(e0z * e1x - e0x * e1z)
        var cz = math_abs(e0x * e1y - e0y * e1x)
        if cx > F32_EPS or cy > F32_EPS or cz > F32_EPS:
            if not (cx > cy and cx > cz):
                ax0 = 0
                if cz > cx and cz > cy:
                    ax1 = 1
            break

    var rem = List[Int]()
    for i in range(n):
        rem.append(poly[i])
    var guess = 0
    # ⚠ The two counters are what bounds a polygon the clipper cannot finish.
    var remaining_iters = n
    var prev_remaining = n
    while len(rem) > 3 and remaining_iters > 0:
        var m = len(rem)
        if guess >= m:
            guess -= m
        if prev_remaining != m:
            prev_remaining = m
            remaining_iters = m
        else:
            remaining_iters -= 1

        var ind = InlineArray[Int, 3](fill=0)
        var px = InlineArray[Float32, 3](fill=Float32(0))
        var py = InlineArray[Float32, 3](fill=Float32(0))
        for k in range(3):
            var vi = rem[(guess + k) % m]
            ind[k] = vi
            px[k] = _axis_of(vx, vy, vz, vi, ax0)
            py[k] = _axis_of(vx, vy, vz, vi, ax1)

        var e0x = px[1] - px[0]
        var e0y = py[1] - py[0]
        var e1x = px[2] - px[1]
        var e1y = py[2] - py[1]
        var cross = e0x * e1y - e0y * e1x
        # ⚠ `area` is tinyobj's own, and it is NOT the triangle's area — it
        # uses only the first two corners. Copy it as written; the sign of
        # `cross * area` is the whole ear test.
        var area = (px[0] * py[1] - py[0] * px[1]) * Float32(0.5)
        if cross * area < Float32(0):
            guess += 1
            continue

        var overlap = False
        for other in range(3, m):
            var o = rem[(guess + other) % m]
            if _pnpoly3(
                px,
                py,
                _axis_of(vx, vy, vz, o, ax0),
                _axis_of(vx, vy, vz, o, ax1),
            ):
                overlap = True
                break
        if overlap:
            guess += 1
            continue

        out.append(ind[0])
        out.append(ind[1])
        out.append(ind[2])
        var r = (guess + 1) % m
        while r + 1 < m:
            rem[r] = rem[r + 1]
            r += 1
        _ = rem.pop()

    # ⚠ ONLY when exactly three are left. A polygon the loop gave up on is
    # dropped, faces and all — that is the reference's behaviour, not a bug.
    if len(rem) == 3:
        out.append(rem[0])
        out.append(rem[1])
        out.append(rem[2])
    return out^


def load_obj(path: String) raises -> MeshData:
    """Positions + faces from a Wavefront OBJ, as flat per-face triangles."""
    var f = open(path, "r")
    var text = f.read()
    f.close()

    var vx = List[Float32]()
    var vy = List[Float32]()
    var vz = List[Float32]()
    var mesh = MeshData()

    var min_x = Float32(1e30)
    var min_y = Float32(1e30)
    var min_z = Float32(1e30)
    var max_x = Float32(-1e30)
    var max_y = Float32(-1e30)
    var max_z = Float32(-1e30)

    var n = text.byte_length()
    var pos = 0
    while pos < n:
        var nl = text.find("\n", pos)
        if nl == -1:
            nl = n
        var line = String(text[byte=pos:nl])
        pos = nl + 1
        if line.byte_length() < 2:
            continue
        var head2 = String(line[byte=0:2])
        # ⚠ `v ` WITH THE SPACE. Bare `startswith("v")` also matches `vt` and
        # `vn`, which would push texture coordinates into the position list and
        # shift every face index after them.
        if head2 == "v ":
            var t = _tokens(line)
            if len(t) >= 4:
                var x = _f32(t[1])
                var y = _f32(t[2])
                var z = _f32(t[3])
                vx.append(x)
                vy.append(y)
                vz.append(z)
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if z < min_z:
                    min_z = z
                if z > max_z:
                    max_z = z
        elif head2 == "f ":
            var t = _tokens(line)
            if len(t) < 4:
                continue
            var nverts = len(vx)
            # ⚠ The whole polygon first — the split is `_triangulate_face`'s
            # to make, and it needs every corner to make it.
            var poly = List[Int]()
            var poly_ok = True
            for k in range(1, len(t)):
                var ii = _first_index(t[k])
                # 1-based, or negative meaning "from the end".
                var vi = (ii - 1) if ii > 0 else (nverts + ii)
                if vi < 0 or vi >= nverts:
                    poly_ok = False
                    break
                poly.append(vi)
            if not poly_ok or len(poly) < 3:
                continue
            var tri_idx = _triangulate_face(vx, vy, vz, poly)
            for ti in range(0, len(tri_idx), 3):
                var a = tri_idx[ti + 0]
                var b = tri_idx[ti + 1]
                var c = tri_idx[ti + 2]
                var ax = vx[a]
                var ay = vy[a]
                var az = vz[a]
                var bx = vx[b]
                var by = vy[b]
                var bz = vz[b]
                var cx = vx[c]
                var cy = vy[c]
                var cz = vz[c]
                # Face normal, like `load_stl`'s stored one.
                var ux = bx - ax
                var uy = by - ay
                var uz = bz - az
                var wx = cx - ax
                var wy = cy - ay
                var wz = cz - az
                var nx = uy * wz - uz * wy
                var ny = uz * wx - ux * wz
                var nz = ux * wy - uy * wx
                var ln = sqrt(nx * nx + ny * ny + nz * nz)
                if ln > Float32(1e-20):
                    nx /= ln
                    ny /= ln
                    nz /= ln
                else:
                    nx = Float32(0)
                    ny = Float32(0)
                    nz = Float32(1)
                var base = UInt32(len(mesh.vertices))
                mesh.vertices.append(
                    GPUVertex(px=ax, py=ay, pz=az, nx=nx, ny=ny, nz=nz)
                )
                mesh.vertices.append(
                    GPUVertex(px=bx, py=by, pz=bz, nx=nx, ny=ny, nz=nz)
                )
                mesh.vertices.append(
                    GPUVertex(px=cx, py=cy, pz=cz, nx=nx, ny=ny, nz=nz)
                )
                mesh.indices.append(base)
                mesh.indices.append(base + 1)
                mesh.indices.append(base + 2)

    if len(mesh.vertices) == 0:
        raise Error(
            "OBJ has no triangles: '" + path + "' (parsed "
            + String(len(vx)) + " vertices but no usable `f` lines)"
        )

    # Box-projected UVs on the two largest axes — the same rule as `load_stl`,
    # so a mesh looks the same whichever format it arrived in.
    var dx = max_x - min_x
    var dy = max_y - min_y
    var dz = max_z - min_z
    if dx < Float32(1e-6):
        dx = Float32(1)
    if dy < Float32(1e-6):
        dy = Float32(1)
    if dz < Float32(1e-6):
        dz = Float32(1)
    for i in range(len(mesh.vertices)):
        ref v = mesh.vertices[i]
        if dz <= dx and dz <= dy:
            v.u = (v.px - min_x) / dx
            v.v = (v.py - min_y) / dy
        elif dy <= dx:
            v.u = (v.px - min_x) / dx
            v.v = (v.pz - min_z) / dz
        else:
            v.u = (v.py - min_y) / dy
            v.v = (v.pz - min_z) / dz
    return mesh^
