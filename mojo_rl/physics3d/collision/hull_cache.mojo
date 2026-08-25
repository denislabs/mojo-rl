"""On-disk cache for the per-mesh convex hull + polygon topology.

Building a mesh's collision geometry is the single most expensive thing that
happens at model load. Measured on SO-ARM101 (13 meshes, 16 MB of STL) with
macOS `sample`, `init_fields` spends **18.9 s**, split

    ~62%  `_convex_hull_f64`      incremental hull, O(n*h)
    ~38%  `build_mesh_polygons`   face merging + the vertex->polygon map

and `deduplicate_vertices` / `load_stl` do not register. ⚠ THAT SPLIT IS WHY
THE CACHE COVERS ALL OF `load_mesh_hull` AND NOT JUST THE HULL: caching only
`compute_convex_hull` would leave more than a third of the cost on the table
and still stall the viewer for seconds on every env switch.

## What is cached

One file per (mesh, mesh-frame) pair, holding the ENTIRE output of
`load_mesh_hull` for that mesh: hull vertices, the merged-polygon topology, the
vertex->polygon map, the hull edge graph and `rbound`.

## ⚠⚠ THE REBASING IS THE WHOLE RISK

`load_mesh_hull` appends into arrays SHARED BY EVERY MESH IN THE MODEL, so half
of what it writes is positional. Get one of these wrong and the indices point
into a neighbouring mesh's region: a silently wrong collision shape, which is
the worst failure class this file can produce, and one that no compile error
and no crash will surface. The convention is not uniform, so it is written out
here and enforced by `test_hull_cache.mojo`:

    array          values are            cached as        on load
    ------------   -------------------   --------------   -----------------
    hull verts     coordinates           float64          convert to DTYPE
    poly_vert      LOCAL vertex ids      verbatim         verbatim
    poly_vertadr   offset into           rebased to 0     + len(poly_vert)
                   poly_vert (GLOBAL)
    poly_vertnum   counts                verbatim         verbatim
    poly_normal    coordinates           float64          convert to DTYPE
    polymap        LOCAL polygon ids     verbatim         verbatim
    polymap_adr    offset into           rebased to 0     + len(polymap)
                   polymap (GLOBAL)
    polymap_num    counts                verbatim         verbatim
    edge_adr       offset into           rebased to 0     + len(edge_list)
                   edge_list (GLOBAL)
    edge_list      GLOBAL vertex ids,    rebased to 0     + vert_base,
                   -1 terminated                          -1 PASSED THROUGH

The miss path gets this right for free: it builds into FRESH, EMPTY lists with
`vert_base = 0`, so every offset it produces is already relative to zero. The
cache therefore stores exactly what a from-scratch build of a one-mesh model
would produce, and the load path applies the same shift the append path always
applied.

## The key

`(mesh file bytes, mesh principal frame, format version)`.

⚠ THE MESH FRAME IS PART OF THE KEY, NOT JUST THE FILE. `load_mesh_hull` bakes
`mi`'s centre of mass and principal rotation into the vertices BEFORE hulling
(see `transform_verts_to_principal_frame`), so the same STL under a different
`mi` is a different hull. Keying on the file alone would serve one geometry for
the other — and since `mi` is normally derived from the same file, it would do
so only in the rare case, which is the worst way for a cache to be wrong.

⚠ CONTENT HASH, NOT MTIME. A hash means a `git checkout`, a `git stash` or an
rsync that rewrites timestamps does not invalidate anything, and a mesh edited
back to a previous state re-uses its old entry. It costs one pass over the
file: measured 5 ms for the 2.7 MB `wrist_roll_pitch_so101_v2.stl`, ~30 ms for
all of SO-ARM101, against 18.9 s of build. `std.os` exposes no `getmtime`
anyway, so the cheaper key was not actually the cheaper implementation.

⚠⚠ THE RUNTIME DTYPE IS IN THE KEY EXPLICITLY. The first version left it
implicit, reasoning that `mi` is rounded to `DTYPE` before hashing and so
carries it. That is false whenever `mi`'s components round identically at both
dtypes — which a DEFAULT-CONSTRUCTED `MeshInertia` (all zeros, qw=1) always
does — and `test_hull_cache.mojo` failed on its very first run with the float32
build reading the float64 entry. The hull itself is dtype-invariant by
construction, but `poly_normal` and `rbound` are computed AT `DTYPE`, so
sharing one entry reintroduces exactly the divergence
`test_convex_hull_dtype_invariance.mojo` exists to prevent — through the cache
instead of through the hull, where no existing gate was looking.

## Disabling it

`PHYSICS3D_HULL_CACHE=0` turns the cache off entirely — both reads and writes.
That is what lets a gate build the same model cold and warm in one process and
compare, and what to reach for if a cached hull is ever suspected.

`PHYSICS3D_HULL_CACHE_DIR` moves it; the default is `.cache/physics3d_hulls`
under the current directory, which is the repo root by the project's run
convention. A miss, an unreadable entry, a short file, a bad magic or a version
bump all fall back to building — the cache is never load-bearing for
correctness, only for time.
"""

from std.ffi import external_call
from std.memory import bitcast
from std.os import getenv, makedirs
from std.pathlib import Path

from ..model.mesh_inertia import MeshInertia

# ⚠ BUMP ON ANY CHANGE TO WHAT `load_mesh_hull` PRODUCES — a new hull
# tolerance, a different polygon merge, a change to the edge graph. Stale
# entries are not detectable from their contents; the version is the only
# thing standing between a hull-algorithm fix and a cache that keeps serving
# the old geometry.
comptime HULL_CACHE_VERSION: Int = 9  # 9: `<mesh maxhullvert>` decimates the hull

comptime _MAGIC: UInt64 = 0x4D4A48554C4C3031  # "MJHULL01"
comptime _FNV_OFFSET: UInt64 = 14695981039346656037
comptime _FNV_PRIME: UInt64 = 1099511628211
comptime _CHUNK: Int = 1 << 20
comptime _HEADER_WORDS: Int = 9


struct HullPayload(Copyable, Movable):
    """One mesh's `load_mesh_hull` output, with every offset rebased to zero.

    Dtype-independent on purpose: floats are held at float64 so a float32 model
    and a float64 model round-trip through the same representation, and
    narrowing on the way out is exact.
    """

    var hull_vert: List[Float64]
    var poly_vertadr: List[Int]
    var poly_vertnum: List[Int]
    var poly_vert: List[Int]
    var poly_normal: List[Float64]
    var polymap_adr: List[Int]
    var polymap_num: List[Int]
    var polymap: List[Int]
    var edge_adr: List[Int]
    var edge_list: List[Int]
    var rbound: Float64
    var num_hull: Int
    var npoly: Int

    var tri_vert: List[Float64]
    """The mesh's ORIGINAL triangles, 9 floats each, in the principal frame.

    ⚠⚠ THE HULL CANNOT ANSWER A RAY. `mj_rayMesh` walks `mesh_face`, the
    original triangle list, and the hull is a different surface: a ray aimed
    into a bracket's cutout hits hull where the real part has a hole. That is
    fine for collision — MuJoCo collides convex hulls too — and wrong for a
    rangefinder, a picker or a camera, which are the three consumers of
    `physics3d/ray/`.

    ⚠ A SOUP, NOT VERTICES-PLUS-INDICES, and the choice is deliberate. MuJoCo
    stores `mesh_vert` + `mesh_face` and this could too, at 1/3 the bytes. It
    does not, because our `deduplicate_vertices` runs at a different epsilon
    and in a different order from the loader MuJoCo uses, so our index arrays
    would not match `mjModel.mesh_face` element for element anyway — the
    structural resemblance would be cosmetic while costing an index map
    threaded through the dedup, and the RAY ANSWER, which is what is actually
    gated, is identical either way. The cost is ~3.3 MB on the largest robot
    in the tree (SO-101) against ~1.1 MB, once, in a CPU-side model.
    """

    var num_tri: Int

    def __init__(out self):
        self.hull_vert = List[Float64]()
        self.poly_vertadr = List[Int]()
        self.poly_vertnum = List[Int]()
        self.poly_vert = List[Int]()
        self.poly_normal = List[Float64]()
        self.polymap_adr = List[Int]()
        self.polymap_num = List[Int]()
        self.polymap = List[Int]()
        self.edge_adr = List[Int]()
        self.edge_list = List[Int]()
        self.rbound = Float64(0)
        self.num_hull = 0
        self.npoly = 0
        self.tri_vert = List[Float64]()
        self.num_tri = 0


@always_inline
def _fnv(h: UInt64, w: UInt64) -> UInt64:
    """One FNV-1a round over the eight bytes of `w`, low byte first."""
    var acc = h
    for k in range(8):
        acc = (acc ^ ((w >> UInt64(k * 8)) & 0xFF)) * _FNV_PRIME
    return acc


def _i2u(x: Int) -> UInt64:
    """Two's-complement Int -> UInt64. `edge_list` carries -1 terminators."""
    return Scalar[DType.int64](x).cast[DType.uint64]()


def _u2i(w: UInt64) -> Int:
    return Int(w.cast[DType.int64]())


def _f2u(x: Float64) -> UInt64:
    return x.to_bits[DType.uint64]()


def _u2f(w: UInt64) -> Float64:
    return bitcast[DType.float64](w)


def _hex16(w: UInt64) -> String:
    """`w` as 16 lowercase hex digits, high nibble first."""
    comptime DIGITS = String("0123456789abcdef")
    var out = String()
    for i in range(16):
        var nib = Int((w >> UInt64((15 - i) * 4)) & 0xF)
        out += String(DIGITS[byte = nib : nib + 1])
    return out^


def _basename(p: String) -> String:
    """Last path component, for a cache filename a human can recognise."""
    var cut = p.rfind("/")
    if cut < 0:
        return p
    return String(p[byte = cut + 1 : p.byte_length()])


def _dirname(p: String) -> String:
    """Everything before the last "/", or "" when there is none."""
    var cut = p.rfind("/")
    if cut <= 0:
        return String("")
    return String(p[byte=0:cut])


def hull_cache_dir() -> String:
    """Where entries live, or "" when the cache is switched off."""
    var flag = getenv("PHYSICS3D_HULL_CACHE", "1")
    if flag == "0" or flag == "off" or flag == "false":
        return String("")
    var override = getenv("PHYSICS3D_HULL_CACHE_DIR", "")
    if override.byte_length() > 0:
        return override
    return String(".cache/physics3d_hulls")


def hull_cache_path[
    DTYPE: DType
](
    mesh_filename: String,
    mi: MeshInertia[DTYPE],
    sx: Float64 = 1.0,
    sy: Float64 = 1.0,
    sz: Float64 = 1.0,
    maxhullvert: Int = -1,
) raises -> String:
    """Cache file for this (mesh contents, mesh frame, format version).

    Returns "" when the cache is disabled or the mesh cannot be read — both
    are "just build it" for the caller, not errors.
    """
    var dir = hull_cache_dir()
    if dir.byte_length() == 0:
        return String("")

    var h = _FNV_OFFSET
    h = _fnv(h, UInt64(HULL_CACHE_VERSION))

    # ⚠⚠ THE RUNTIME DTYPE IS PART OF THE KEY, EXPLICITLY. It was originally
    # left implicit on the reasoning that `mi` is dtype-rounded and so carries
    # it — and `test_hull_cache.mojo` failed on its first run because that is
    # false whenever `mi`'s components round IDENTICALLY at both dtypes, which
    # is exactly what a default-constructed `MeshInertia` (all zeros, qw=1)
    # does. The float32 build was served the float64 entry, reintroducing the
    # dtype divergence `test_convex_hull_dtype_invariance.mojo` exists to
    # prevent, through the cache rather than through the hull.
    #
    # The HULL is dtype-invariant by construction (built in float64, converted
    # on the way out), but `poly_normal` and `rbound` are computed AT `DTYPE`
    # and genuinely differ, so the entries must not be shared.
    var tag = String(DTYPE)
    var tag_bytes = tag.as_bytes()
    for i in range(len(tag_bytes)):
        h = (h ^ UInt64(tag_bytes[i])) * _FNV_PRIME

    var nbytes = 0
    try:
        with open(mesh_filename, "r") as f:
            while True:
                var chunk = f.read_bytes(_CHUNK)
                if len(chunk) == 0:
                    break
                nbytes += len(chunk)
                for i in range(len(chunk)):
                    h = (h ^ UInt64(chunk[i])) * _FNV_PRIME
    except:
        # Unreadable here means unreadable in `load_stl` a moment later; let
        # that path raise the real diagnostic rather than inventing one.
        return String("")
    h = _fnv(h, UInt64(nbytes))

    # ⚠ The FRAME, not just the file — see the module docstring. Promoted to
    # float64 so the stored bits are dtype-stable for a given `DTYPE` input.
    # ⚠⚠ `<mesh scale>` IS IN THE KEY EXPLICITLY, and the reason is written
    # ten lines up: the dtype was once left implicit on the argument that `mi`
    # already carried it, and that was false often enough to fail a test. The
    # same argument would be available here — a scaled mesh has a scaled CoM —
    # and it is just as unsafe. A hull cached before this feature existed is
    # UNSCALED, and serving it to a `scale="0.001"` model reproduces the exact
    # bug (hulls 1000x oversized) from disk, invisibly.
    h = _fnv(h, _f2u(sx))
    h = _fnv(h, _f2u(sy))
    h = _fnv(h, _f2u(sz))

    h = _fnv(h, _f2u(Float64(mi.com_x)))
    h = _fnv(h, _f2u(Float64(mi.com_y)))
    h = _fnv(h, _f2u(Float64(mi.com_z)))
    h = _fnv(h, _f2u(Float64(mi.qx)))
    h = _fnv(h, _f2u(Float64(mi.qy)))
    h = _fnv(h, _f2u(Float64(mi.qz)))
    h = _fnv(h, _f2u(Float64(mi.qw)))

    # ⚠⚠ `<mesh maxhullvert>` IS IN THE KEY FOR THE SAME REASON `scale` IS.
    # It changes the hull qhull returns — a budgeted run stops adding vertices
    # — so one STL at two budgets is two payloads. A hull cached before the
    # budget was honoured is the UNLIMITED one, and serving it to a
    # `maxhullvert="64"` model reproduces the pre-fix hull from disk, silently
    # and with no warning left to notice it by.
    h = _fnv(h, UInt64(Int64(maxhullvert)))

    return dir + "/" + _basename(mesh_filename) + "-" + _hex16(h) + ".hull"


def hull_cache_load(cache_path: String, mut out: HullPayload) -> Bool:
    """Fill `out` from `cache_path`. False on any doubt whatsoever.

    Every failure mode — absent, truncated, wrong magic, wrong version,
    inconsistent lengths — returns False so the caller rebuilds. A cache that
    guesses is worse than no cache.
    """
    if cache_path.byte_length() == 0:
        return False

    var raw = List[UInt8]()
    try:
        with open(cache_path, "r") as f:
            while True:
                var chunk = f.read_bytes(_CHUNK)
                if len(chunk) == 0:
                    break
                for i in range(len(chunk)):
                    raw.append(chunk[i])
    except:
        return False

    var nwords = len(raw) // 8
    if nwords < _HEADER_WORDS:
        return False
    var w = List[UInt64](length=nwords, fill=UInt64(0))
    for i in range(nwords):
        var acc = UInt64(0)
        for k in range(8):
            acc |= UInt64(raw[i * 8 + k]) << UInt64(k * 8)
        w[i] = acc

    if w[0] != _MAGIC or _u2i(w[1]) != HULL_CACHE_VERSION:
        return False

    var num_hull = _u2i(w[2])
    var npoly = _u2i(w[3])
    var n_poly_vert = _u2i(w[4])
    var n_polymap = _u2i(w[5])
    var n_edge_list = _u2i(w[6])
    var rbound = _u2f(w[7])
    var num_tri = _u2i(w[8])

    if num_hull < 0 or npoly < 0 or n_poly_vert < 0 or n_polymap < 0 or (
        n_edge_list < 0
    ) or num_tri < 0:
        return False

    var want = (
        _HEADER_WORDS
        + num_hull * 3
        + npoly * 2
        + n_poly_vert
        + npoly * 3
        + num_hull * 2
        + n_polymap
        + num_hull
        + n_edge_list
        + num_tri * 9
    )
    if nwords != want:
        return False

    out = HullPayload()
    out.num_hull = num_hull
    out.npoly = npoly
    out.rbound = rbound
    out.num_tri = num_tri

    var o = _HEADER_WORDS
    for i in range(num_hull * 3):
        out.hull_vert.append(_u2f(w[o + i]))
    o += num_hull * 3
    for i in range(npoly):
        out.poly_vertadr.append(_u2i(w[o + i]))
    o += npoly
    for i in range(npoly):
        out.poly_vertnum.append(_u2i(w[o + i]))
    o += npoly
    for i in range(n_poly_vert):
        out.poly_vert.append(_u2i(w[o + i]))
    o += n_poly_vert
    for i in range(npoly * 3):
        out.poly_normal.append(_u2f(w[o + i]))
    o += npoly * 3
    for i in range(num_hull):
        out.polymap_adr.append(_u2i(w[o + i]))
    o += num_hull
    for i in range(num_hull):
        out.polymap_num.append(_u2i(w[o + i]))
    o += num_hull
    for i in range(n_polymap):
        out.polymap.append(_u2i(w[o + i]))
    o += n_polymap
    for i in range(num_hull):
        out.edge_adr.append(_u2i(w[o + i]))
    o += num_hull
    for i in range(n_edge_list):
        out.edge_list.append(_u2i(w[o + i]))
    o += n_edge_list
    for i in range(num_tri * 9):
        out.tri_vert.append(_u2f(w[o + i]))

    return True


def hull_cache_store(cache_path: String, p: HullPayload):
    """Write `p` to `cache_path`, or give up quietly.

    ⚠ WRITE-THEN-RENAME, TO A PID-SUFFIXED TEMP. Two processes building the
    same mesh at once, or a run killed mid-write, must not leave a half-written
    entry that the length check happens to accept. `rename` within one
    directory is atomic, so a reader sees either the old entry or the complete
    new one, and the pid keeps two concurrent writers off each other's scratch
    file.

    Failures are swallowed on purpose: a read-only checkout, a full disk or a
    missing parent must not turn a working model load into a raise.
    """
    if cache_path.byte_length() == 0:
        return

    var w = List[UInt64]()
    w.append(_MAGIC)
    w.append(_i2u(HULL_CACHE_VERSION))
    w.append(_i2u(p.num_hull))
    w.append(_i2u(p.npoly))
    w.append(_i2u(len(p.poly_vert)))
    w.append(_i2u(len(p.polymap)))
    w.append(_i2u(len(p.edge_list)))
    w.append(_f2u(p.rbound))
    # w[8] was reserved and written as 0; version 7 spends it on the triangle
    # count, which is why the version bumped rather than the header growing.
    w.append(_i2u(p.num_tri))

    for i in range(len(p.hull_vert)):
        w.append(_f2u(p.hull_vert[i]))
    for i in range(len(p.poly_vertadr)):
        w.append(_i2u(p.poly_vertadr[i]))
    for i in range(len(p.poly_vertnum)):
        w.append(_i2u(p.poly_vertnum[i]))
    for i in range(len(p.poly_vert)):
        w.append(_i2u(p.poly_vert[i]))
    for i in range(len(p.poly_normal)):
        w.append(_f2u(p.poly_normal[i]))
    for i in range(len(p.polymap_adr)):
        w.append(_i2u(p.polymap_adr[i]))
    for i in range(len(p.polymap_num)):
        w.append(_i2u(p.polymap_num[i]))
    for i in range(len(p.polymap)):
        w.append(_i2u(p.polymap[i]))
    for i in range(len(p.edge_adr)):
        w.append(_i2u(p.edge_adr[i]))
    for i in range(len(p.edge_list)):
        w.append(_i2u(p.edge_list[i]))
    for i in range(len(p.tri_vert)):
        w.append(_f2u(p.tri_vert[i]))

    var bytes = List[UInt8](length=len(w) * 8, fill=UInt8(0))
    for i in range(len(w)):
        var v = w[i]
        for k in range(8):
            bytes[i * 8 + k] = UInt8((v >> UInt64(k * 8)) & 0xFF)

    var dir = _dirname(cache_path)

    try:
        if dir.byte_length() > 0:
            makedirs(Path(dir), exist_ok=True)
        # ⚠ THE TEMP NAME CARRIES THE PID. A viewer and a test run building the
        # same mesh at the same time would otherwise interleave their writes
        # into ONE `.tmp` and rename the mixture into place, where the length
        # check has every chance of accepting it.
        var pid = Int(external_call["getpid", Int32]())
        var tmp = cache_path + ".tmp" + String(pid)
        var dst = cache_path
        with open(tmp, "w") as f:
            var off = 0
            while off < len(bytes):
                var take = len(bytes) - off
                if take > _CHUNK:
                    take = _CHUNK
                f.write_bytes(Span(bytes)[off : off + take])
                off += take
        var rc = external_call["rename", Int32](
            tmp.as_c_string_slice().unsafe_ptr(),
            dst.as_c_string_slice().unsafe_ptr(),
        )
        _ = rc
    except:
        # A cache that cannot be written is a slow model load, not a failure.
        pass
