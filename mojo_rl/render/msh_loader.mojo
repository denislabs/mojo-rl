"""MuJoCo legacy `.msh` binary mesh loader.

    var m = load_msh("bowl_vis.msh")      # MeshData, flat per-face triangles

The format (MuJoCo `user_mesh.cc`, `LoadMSH`), little-endian:

    int32   nvertex nnormal ntexcoord nface
    float32 vertex  [3 * nvertex]
    float32 normal  [3 * nnormal]      nnormal   is 0 or nvertex
    float32 texcoord[2 * ntexcoord]    ntexcoord is 0 or nvertex
    int32   face    [3 * nface]        0-based vertex indices

## Why this exists

LIBERO ships its visual meshes as `.msh` — 138 `<mesh file=…msh>` references
across 106 asset XMLs (`docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md` §1.3), and
the `.obj` beside each one is the SOURCE, not what MuJoCo loads. Without this
reader every LIBERO object renders as nothing; with `.obj` substituted the
geometry would be the pre-conversion mesh, which is not what the benchmark's
pixels show.

⚠ THE FILE SIZE IS CHECKED AGAINST THE HEADER BEFORE ANY READ. The four
counts fully determine the byte length, so a truncated or non-msh file is
refused by arithmetic rather than by reading garbage — the same rule
`load_stl` applies to the triangle count at byte 80. On the corpus this is
also the vacuity check for the reader: every one of the 88 files satisfies
`size == 16 + 4*(3nv + 3nn + 2nt + 3nf)` (measured), so a header misread
cannot pass silently.

⚠ ORIENTATION AND SCALE ARE NOT APPLIED HERE. `<mesh scale>` is applied by
`load_stl`'s dispatch (`_scale_mesh`), exactly as for `.obj`, so that the
three callers do not each carry a copy. Vertices are returned in the FILE's
frame; the parser composes `mesh_pos`/`mesh_quat` as it does for every other
format.
"""

from std.math import sqrt
from .gpu_types import GPUVertex, MeshData


def _is_msh(path: String) -> Bool:
    var n = path.byte_length()
    if n < 4:
        return False
    var ext = String(path[byte = n - 4 : n])
    return ext == ".msh" or ext == ".MSH"


def load_msh(path: String) raises -> MeshData:
    """A `.msh` as flat per-face triangles with per-corner normals and UVs."""
    var f = open(path, "r")
    var content = f.read_bytes()
    f.close()
    if len(content) < 16:
        raise Error(
            "msh: '" + path + "' is " + String(len(content))
            + " bytes; the 16-byte header does not fit"
        )
    var raw = content.unsafe_ptr()
    var hp = raw.unsafe_bitcast[Int32]()
    var nv = Int(hp[unsafe_offset=0])
    var nn = Int(hp[unsafe_offset=1])
    var nt = Int(hp[unsafe_offset=2])
    var nf = Int(hp[unsafe_offset=3])
    if nv <= 0 or nf <= 0 or nn < 0 or nt < 0:
        raise Error(
            "msh: '" + path + "' header nvertex=" + String(nv) + " nnormal="
            + String(nn) + " ntexcoord=" + String(nt) + " nface="
            + String(nf) + " — not a msh file"
        )
    if nn != 0 and nn != nv:
        raise Error(
            "msh: '" + path + "' has " + String(nn) + " normals for "
            + String(nv) + " vertices; MuJoCo requires 0 or nvertex"
        )
    if nt != 0 and nt != nv:
        raise Error(
            "msh: '" + path + "' has " + String(nt) + " texcoords for "
            + String(nv) + " vertices; MuJoCo requires 0 or nvertex"
        )
    var expected = 16 + 4 * (3 * nv + 3 * nn + 2 * nt + 3 * nf)
    if len(content) != expected:
        raise Error(
            "msh: '" + path + "' is " + String(len(content))
            + " bytes but its header implies " + String(expected)
            + " (nvertex=" + String(nv) + " nnormal=" + String(nn)
            + " ntexcoord=" + String(nt) + " nface=" + String(nf) + ")"
        )

    var vp = raw.unsafe_offset(16).unsafe_bitcast[Float32]()
    var npo = raw.unsafe_offset(16 + 12 * nv).unsafe_bitcast[Float32]()
    var tp = raw.unsafe_offset(16 + 12 * nv + 12 * nn).unsafe_bitcast[Float32]()
    var fp = raw.unsafe_offset(
        16 + 12 * nv + 12 * nn + 8 * nt
    ).unsafe_bitcast[Int32]()

    var mesh = MeshData()
    mesh.vertices.reserve(3 * nf)
    mesh.indices.reserve(3 * nf)
    for t in range(nf):
        var ia = Int(fp[unsafe_offset = 3 * t])
        var ib = Int(fp[unsafe_offset = 3 * t + 1])
        var ic = Int(fp[unsafe_offset = 3 * t + 2])
        if ia < 0 or ib < 0 or ic < 0 or ia >= nv or ib >= nv or ic >= nv:
            raise Error(
                "msh: '" + path + "' face " + String(t) + " indexes vertex "
                + String(max(ia, max(ib, ic))) + " of " + String(nv)
            )
        var ax = vp[unsafe_offset = 3 * ia]
        var ay = vp[unsafe_offset = 3 * ia + 1]
        var az = vp[unsafe_offset = 3 * ia + 2]
        var bx = vp[unsafe_offset = 3 * ib]
        var by = vp[unsafe_offset = 3 * ib + 1]
        var bz = vp[unsafe_offset = 3 * ib + 2]
        var cx = vp[unsafe_offset = 3 * ic]
        var cy = vp[unsafe_offset = 3 * ic + 1]
        var cz = vp[unsafe_offset = 3 * ic + 2]
        # face normal, used when the file carries none
        var ex = bx - ax
        var ey = by - ay
        var ez = bz - az
        var gx = cx - ax
        var gy = cy - ay
        var gz = cz - az
        var fnx = ey * gz - ez * gy
        var fny = ez * gx - ex * gz
        var fnz = ex * gy - ey * gx
        var ln = sqrt(fnx * fnx + fny * fny + fnz * fnz)
        if ln > Float32(1e-20):
            fnx /= ln
            fny /= ln
            fnz /= ln
        else:
            fnx = Float32(0)
            fny = Float32(0)
            fnz = Float32(1)
        var base = UInt32(len(mesh.vertices))
        var idx = List[Int]()
        idx.append(ia)
        idx.append(ib)
        idx.append(ic)
        for k in range(3):
            var vi = idx[k]
            var nx = fnx
            var ny = fny
            var nz = fnz
            if nn > 0:
                nx = npo[unsafe_offset = 3 * vi]
                ny = npo[unsafe_offset = 3 * vi + 1]
                nz = npo[unsafe_offset = 3 * vi + 2]
            var u = Float32(0)
            var v = Float32(0)
            if nt > 0:
                u = tp[unsafe_offset = 2 * vi]
                v = tp[unsafe_offset = 2 * vi + 1]
            mesh.vertices.append(
                GPUVertex(
                    px=vp[unsafe_offset = 3 * vi],
                    py=vp[unsafe_offset = 3 * vi + 1],
                    pz=vp[unsafe_offset = 3 * vi + 2],
                    nx=nx, ny=ny, nz=nz, u=u, v=v,
                )
            )
            mesh.indices.append(base + UInt32(k))
    _ = content^
    return mesh^


def msh_counts(path: String) raises -> List[Int]:
    """`[nvertex, nnormal, ntexcoord, nface]` from the header, after the same
    size check `load_msh` makes. For gates that count without loading."""
    var f = open(path, "r")
    var content = f.read_bytes()
    f.close()
    if len(content) < 16:
        raise Error("msh: '" + path + "' shorter than its header")
    var hp = content.unsafe_ptr().unsafe_bitcast[Int32]()
    var out = List[Int]()
    for i in range(4):
        out.append(Int(hp[unsafe_offset=i]))
    var expected = 16 + 4 * (3 * out[0] + 3 * out[1] + 2 * out[2] + 3 * out[3])
    if len(content) != expected:
        raise Error(
            "msh: '" + path + "' is " + String(len(content))
            + " bytes but its header implies " + String(expected)
        )
    _ = content^
    return out^
