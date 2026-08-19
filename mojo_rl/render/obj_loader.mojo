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
  quads. Each face is fanned from its first vertex, which is exact for the
  convex faces an exporter emits.
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

from std.math import sqrt

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
            # Fan from the first vertex: (0,1,2), (0,2,3), ...
            for k in range(2, len(t) - 1):
                var ia = _first_index(t[1])
                var ib = _first_index(t[k])
                var ic = _first_index(t[k + 1])
                # 1-based, or negative meaning "from the end".
                var a = (ia - 1) if ia > 0 else (nverts + ia)
                var b = (ib - 1) if ib > 0 else (nverts + ib)
                var c = (ic - 1) if ic > 0 else (nverts + ic)
                if a < 0 or b < 0 or c < 0 or a >= nverts or b >= nverts \
                        or c >= nverts:
                    continue
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
