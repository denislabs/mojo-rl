"""The selection outline's half-extents, against an INDEPENDENT mesh read.

⚠⚠ WHY THIS IS NOT OPTIONAL HERE. Every internal arm in `test_mesh_bounds`
compares our number against another of our numbers, or against a constant it
must not be. The obvious second opinion — the loaded collision hull — is
disqualified: `load_mesh_hull` stores vertices in the mesh's PRINCIPAL frame
while the outline is drawn in the render frame, so the two legitimately
disagree (measured: axes permuted and up to 6% out on so_arm101). That was the
older bug this work uncovered, and it means the hull cannot be the oracle.

So this parses the STL and OBJ files itself — no shared code with the loader,
no MuJoCo — and compares the axis-aligned half-extents.

⚠ THE SCALE IS APPLIED HERE TOO, and it is where a mirror could hide: 44
Menagerie declarations are something like `1 -1 1`, whose |extent| is unchanged.
If our loader ever forgot the scale entirely, a mirrored model would still
agree — so the models dumped include ones with scale 1, where a forgotten
scale would still be invisible, AND the check reports how many rows carried a
non-unit scale. A run where that count is 0 has not tested the scale path.

Run (after the Mojo test, which writes the dump):
    pixi run mojo run -I . tests/physics3d/test_mesh_bounds.mojo
    pixi run python scripts/check_mesh_bounds_vs_python.py
"""

import os
import struct
import sys

DUMP = "/tmp/mesh_bounds_dump.txt"


def read_stl(path: str) -> list[tuple[float, float, float]]:
    """Binary STL: 80-byte header, uint32 count, then 50 bytes per triangle."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 84:
        return []
    ntri = struct.unpack_from("<I", data, 80)[0]
    if 84 + ntri * 50 != len(data):
        # ASCII STL, or a size we do not understand — skip rather than guess.
        return []
    out = []
    for t in range(ntri):
        base = 84 + t * 50 + 12  # skip the facet normal
        for v in range(3):
            out.append(struct.unpack_from("<3f", data, base + v * 12))
    return out


def read_obj(path: str) -> list[tuple[float, float, float]]:
    out = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            p = line.split()
            if len(p) >= 4:
                out.append((float(p[1]), float(p[2]), float(p[3])))
    return out


def main() -> int:
    if not os.path.exists(DUMP):
        print(f"FAIL: {DUMP} missing — run the Mojo test first")
        return 1

    cache: dict[str, list] = {}
    rows = 0
    scaled_rows = 0
    fails = 0
    skipped = 0
    worst = 0.0
    worst_row = ""

    for line in open(DUMP):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 7:
            continue
        path, sx, sy, sz, ex, ey, ez = parts
        sx, sy, sz = float(sx), float(sy), float(sz)
        ex, ey, ez = float(ex), float(ey), float(ez)
        rows += 1
        if (sx, sy, sz) != (1.0, 1.0, 1.0):
            scaled_rows += 1

        if path not in cache:
            if path.lower().endswith(".obj"):
                cache[path] = read_obj(path)
            else:
                cache[path] = read_stl(path)
        verts = cache[path]
        if not verts:
            skipped += 1
            continue

        wx = max(abs(v[0] * sx) for v in verts)
        wy = max(abs(v[1] * sy) for v in verts)
        wz = max(abs(v[2] * sz) for v in verts)

        for got, want, axis in ((ex, wx, "x"), (ey, wy, "y"), (ez, wz, "z")):
            scale = max(abs(got), abs(want), 1e-12)
            rel = abs(got - want) / scale
            if rel > worst:
                worst = rel
                worst_row = f"{os.path.basename(path)} {axis}: {got} vs {want}"
            # float32 vertices round-trip through the loader, so the tolerance
            # is float32 epsilon, not float64.
            if rel > 1e-6:
                print(f"  FAIL {os.path.basename(path)} {axis}: "
                      f"mojo {got} vs python {want} (rel {rel:.3e})")
                fails += 1

    print(f"  rows compared      : {rows}")
    print(f"  meshes read        : {len(cache)}")
    print(f"  rows with a scale  : {scaled_rows}")
    print(f"  rows skipped       : {skipped} (format this checker does not read)")
    print(f"  worst relative gap : {worst:.3e}  [{worst_row}]")

    # ⚠ NON-VACUITY. A dump of zero rows, or one where every mesh was skipped,
    # would print "0 failures" and mean nothing.
    if rows < 100:
        print(f"FAIL: only {rows} rows — the dump is too small to mean anything")
        fails += 1
    if len(cache) - skipped < 10:
        print("FAIL: too few meshes actually read")
        fails += 1
    # ⚠ THE SCALE PATH MUST BE EXERCISED, and this is where the docstring
    # above stops being a promise and becomes a check. A dump of only
    # scale-1 models agrees with a loader that ignores `<mesh scale>`
    # entirely — op3, whose STLs are in millimetres, is what makes that
    # disagree by a factor of a thousand.
    if scaled_rows == 0:
        print("FAIL: no row carried a <mesh scale> — the scale path was not "
              "tested; add a model that sets one (op3 is 0.001)")
        fails += 1

    print(f"=== {rows} rows, {fails} failures ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
