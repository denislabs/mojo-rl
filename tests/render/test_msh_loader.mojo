"""`.msh` reader — L1 of the LIBERO port.

    pixi run mojo run -I . tests/render/test_msh_loader.mojo

Hermetic: a `.msh` is WRITTEN here byte by byte from the format's definition
(`render/msh_loader.mojo` header) and read back, so the reader is gated
against the specification and not against a file it produced itself. Then,
when `references/LIBERO-master` is present, every `.msh` under its assets is
header-checked — 88 files whose size must equal the header's arithmetic,
which is the check that catches a misread field.

What could be wrong:
* the header fields are read in the wrong order (nv/nn/nt/nf);
* normals or texcoords are read from the wrong offset when one block is
  absent (nn == 0 or nt == 0 shifts everything after it);
* a face index is not bounds-checked;
* a truncated file is read as a mesh;
* `load_stl` does not dispatch `.msh`, so the corpus goes through the STL
  path and fails on the byte-80 triangle count.
"""

from std.os import listdir
from std.os.path import exists
from std.pathlib import Path
from std.time import perf_counter_ns

from noeira.render.msh_loader import load_msh, msh_counts
from noeira.render.stl_loader import load_stl


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    """Checks run and failed. Both printed: 0 failed over 0 checks is not a pass."""

    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if not ok:
            self.failures += 1
            print("  FAIL:", what)


def _u32(mut b: List[UInt8], u: UInt32):
    """Little-endian, one byte at a time."""
    var x = u
    for _ in range(4):
        b.append(UInt8(x & UInt32(0xFF)))
        x = x >> UInt32(8)


def _i32(mut b: List[UInt8], v: Int):
    _u32(b, UInt32(v if v >= 0 else v + (1 << 32)))


def _f32(mut b: List[UInt8], v: Float32):
    _u32(b, v.to_bits[DType.uint32]())


def _write(path: String, b: List[UInt8]) raises:
    var f = open(path, "w")
    f.write_bytes(b)
    f.close()


def _msh_bytes(with_normals: Bool, with_uv: Bool) -> List[UInt8]:
    """Two triangles on 4 vertices: a unit square split along a diagonal."""
    var b = List[UInt8]()
    _i32(b, 4)
    _i32(b, 4 if with_normals else 0)
    _i32(b, 4 if with_uv else 0)
    _i32(b, 2)
    # vertices
    _f32(b, 0.0); _f32(b, 0.0); _f32(b, 0.5)
    _f32(b, 1.0); _f32(b, 0.0); _f32(b, 0.5)
    _f32(b, 1.0); _f32(b, 1.0); _f32(b, 0.5)
    _f32(b, 0.0); _f32(b, 1.0); _f32(b, 0.5)
    if with_normals:
        for _ in range(4):
            _f32(b, 0.0); _f32(b, 0.0); _f32(b, -1.0)   # deliberately -z
    if with_uv:
        _f32(b, 0.0); _f32(b, 0.0)
        _f32(b, 1.0); _f32(b, 0.0)
        _f32(b, 1.0); _f32(b, 1.0)
        _f32(b, 0.0); _f32(b, 1.0)
    _i32(b, 0); _i32(b, 1); _i32(b, 2)
    _i32(b, 0); _i32(b, 2); _i32(b, 3)
    return b^


def main() raises:
    var ta = Tally()
    print("=" * 70)
    print(".msh reader — L1")
    print("=" * 70)
    var work = String("/tmp/noeira_msh_") + String(perf_counter_ns())
    from std.os import makedirs
    makedirs(work, exist_ok=True)

    # ── full file: normals + uv ─────────────────────────────────────────
    var full = work + "/full.msh"
    _write(full, _msh_bytes(True, True))
    var m = load_msh(full)
    ta.check(len(m.vertices) == 6 and len(m.indices) == 6, "2 faces -> 6 corners")
    ta.check(m.vertices[2].px == 1.0 and m.vertices[2].py == 1.0 and m.vertices[2].pz == 0.5,
          "corner 2 is vertex 2 (1,1,0.5)")
    ta.check(m.vertices[5].px == 0.0 and m.vertices[5].py == 1.0, "corner 5 is vertex 3")
    ta.check(m.vertices[0].nz == -1.0, "file normals are used when present (-z)")
    ta.check(m.vertices[2].u == 1.0 and m.vertices[2].v == 1.0, "uv read per corner")
    ta.check(m.vertices[5].u == 0.0 and m.vertices[5].v == 1.0, "uv of vertex 3")
    var c = msh_counts(full)
    ta.check(c[0] == 4 and c[1] == 4 and c[2] == 4 and c[3] == 2, "msh_counts header")

    # ── no normals: computed; uv block must still be found ─────────────
    var nonorm = work + "/nonorm.msh"
    _write(nonorm, _msh_bytes(False, True))
    var m2 = load_msh(nonorm)
    ta.check(m2.vertices[0].nz == 1.0, "no normals -> face normal (+z for ccw square)")
    ta.check(m2.vertices[2].u == 1.0 and m2.vertices[2].v == 1.0,
          "uv read from the right offset when normals are absent")

    # ── no uv: normals present, uv zero ─────────────────────────────────
    var nouv = work + "/nouv.msh"
    _write(nouv, _msh_bytes(True, False))
    var m3 = load_msh(nouv)
    ta.check(m3.vertices[0].nz == -1.0 and m3.vertices[2].u == 0.0, "no uv -> zeros, normals kept")

    # ── dispatch through load_stl ───────────────────────────────────────
    var m4 = load_stl(full, 2.0, 2.0, 2.0)
    ta.check(len(m4.vertices) == 6 and m4.vertices[2].px == 2.0,
          "load_stl dispatches .msh and applies <mesh scale>")

    # ── refusals ────────────────────────────────────────────────────────
    var trunc = _msh_bytes(True, True)
    _ = trunc.pop()
    _write(work + "/trunc.msh", trunc)
    var r1 = False
    try:
        var _t = load_msh(work + "/trunc.msh")
    except:
        r1 = True
    ta.check(r1, "a truncated file is refused")
    var badidx = _msh_bytes(False, False)
    # last face index -> 7 (out of 4)
    var n = len(badidx)
    badidx[n - 4] = 7
    _write(work + "/badidx.msh", badidx)
    var r2 = False
    try:
        var _b = load_msh(work + "/badidx.msh")
    except:
        r2 = True
    ta.check(r2, "an out-of-range face index is refused")
    var r3 = False
    try:
        var _e = load_msh(work + "/missing.msh")
    except:
        r3 = True
    ta.check(r3, "a missing file raises")

    # ── the corpus, when present ────────────────────────────────────────
    var root = String("references/LIBERO-master/libero/libero/assets")
    if Path(root).is_dir():
        var stack = List[String]()
        stack.append(root)
        var n_msh = 0
        var n_corners = 0
        while len(stack) > 0:
            var d = stack.pop()
            for e in listdir(d):
                var pth = d + "/" + String(e)
                if Path(pth).is_dir():
                    stack.append(pth)
                elif pth.endswith(".msh"):
                    var cc = msh_counts(pth)
                    ta.check(cc[0] > 0 and cc[3] > 0, "corpus header sane: " + pth)
                    n_msh += 1
                    n_corners += 3 * cc[3]
        print("  corpus: ", n_msh, ".msh files header-checked,", n_corners, "corners")
        ta.check(n_msh >= 80, "the corpus has 88 .msh; found " + String(n_msh))
        # one full load, the heaviest object in the corpus
        var bowl = root + "/stable_scanned_objects/akita_black_bowl/visual/akita_black_bowl_vis.msh"
        if exists(bowl):
            var bm = load_msh(bowl)
            ta.check(len(bm.vertices) == 3 * 42522,
                  "akita_black_bowl: 42522 faces -> 127566 corners, got "
                  + String(len(bm.vertices)))
    else:
        print("  corpus: SKIPPED (no references/LIBERO-master) — hermetic half ran")

    print()
    print("  passed:", ta.checks - ta.failures, " failed:", ta.failures)
    if ta.failures > 0:
        raise Error(String(ta.failures) + " checks failed")
    print("=== PASS ===")
