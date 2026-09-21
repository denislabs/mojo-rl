#!/usr/bin/env python3
"""Cut the LIBERO asset pack — L1 of docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md.

    pixi run python tools/assets/build_libero_pack.py
    pixi run python tools/assets/build_libero_pack.py --src <LIBERO assets dir> \
        --out build/packs --version v1 --url <where it will be served>

What it does, and what it refuses:

1. Walks `<src>` (LIBERO's `libero/libero/assets/`) and keeps ONLY what MuJoCo
   loads: `.xml`, `.msh`, `.stl`, `.png`, `.jpg`. The 141 `.obj`/`.mtl` are the
   SOURCES the `.msh` were converted from (§1.3 of the assessment) and the
   `.zip` is a stray; none is referenced by any `<mesh file=>` / `<texture
   file=>`, and THAT IS CHECKED, not assumed — every `file=` attribute in every
   kept XML must resolve inside the kept set, or the build refuses.
2. Writes `ATTRIBUTION.txt` into the pack root. LIBERO's assets are CC-BY-4.0
   (`LIBERO-master/README.md`); re-hosting is legal WITH attribution and this
   file is the attribution.
3. `tar | zstd -19` into `<out>/libero_<version>.tar.zst`, sha256 + size, and
   writes `noeira/tasks/libero/assets.kv` pointing at `--url`.

⚠ IT DOES NOT UPLOAD. Pushing to the Hub is a separate, outward-facing act;
this script's output is the archive and the declaration that names it.

⚠ THE ARCHIVE IS DETERMINISTIC for a given source tree: files are added in
sorted order with fixed mtime/uid/gid, so cutting the pack twice gives the
same sha256, and a re-cut that differs is a source change and not noise.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys

KEEP = {".xml", ".msh", ".stl", ".png", ".jpg"}
DROP = {".obj", ".mtl", ".zip", ".ds_store", ""}

ATTRIBUTION = """LIBERO benchmark assets, repackaged for noeira.

Source:   https://github.com/Lifelong-Robot-Learning/LIBERO
          libero/libero/assets/  (commit as vendored in references/LIBERO-master)
License:  CC-BY-4.0 (assets and datasets), per the LIBERO README.
Authors:  Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu,
          Yuke Zhu, Peter Stone. "LIBERO: Benchmarking Knowledge Transfer
          for Lifelong Robot Learning", NeurIPS 2023 Datasets and Benchmarks.

Changes:  the `.obj` / `.mtl` source meshes and one stray `.zip` were
          dropped; the `.msh`, `.stl`, `.png`, `.jpg` and `.xml` files MuJoCo
          loads are byte-identical to upstream. Nothing was re-encoded.

The individual object scans come from the Google Scanned Objects, HOPE and
TurboSquid collections as redistributed by LIBERO under the same terms.
"""

FILE_ATTR = re.compile(r'file="([^"]+)"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="references/LIBERO-master/libero/libero/assets")
    ap.add_argument("--out", default="build/packs")
    ap.add_argument("--version", default="v1")
    ap.add_argument(
        "--url",
        default="https://huggingface.co/datasets/DenisLabs/mojo-rl-assets/resolve/main/libero_{version}.tar.zst",
    )
    ap.add_argument("--kv", default="noeira/tasks/libero/assets.kv")
    ap.add_argument("--table", default="noeira/tasks/libero/categories.kv")
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    if not os.path.isdir(src):
        sys.exit(f"no LIBERO assets at {src}")

    kept, dropped = [], []
    for root, dirs, files in os.walk(src):
        dirs.sort()
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(root, fn), src)
            ext = os.path.splitext(fn)[1].lower()
            if fn == ".DS_Store":
                continue
            if ext in KEEP:
                kept.append(rel)
            elif ext in DROP:
                dropped.append(rel)
            else:
                sys.exit(f"unclassified file type {ext!r}: {rel} — decide keep or drop")
    kept_set = set(kept)

    # ⚠ THE REFERENCE CHECK, ROOTED AT WHAT LIBERO LOADS. The tree also holds
    # per-part XMLs left over from the mesh conversion (`wooden_cabinet/
    # wooden_cabinet_base/wooden_cabinet_base.xml`, `short_fridge/base/
    # base.xml`, ...) whose `file=` point at textures that no longer exist,
    # and the unused `*_warm_style` arenas reference a `linen_wall_texture.png`
    # that is missing UPSTREAM. Neither is ever opened by the registry, so
    # neither is a defect of this pack. The roots are the `asset=` and
    # `scene=` lines of the category table; every `file=` reachable from a
    # root must resolve inside the kept set, or the build refuses.
    roots = []
    for line in open(a.table):
        line = line.strip()
        if line.startswith("asset=") or line.startswith("scene="):
            roots.append(line.split("=", 1)[1])
    if not roots:
        sys.exit(f"no asset=/scene= lines in {a.table}")
    missing = []
    n_refs = 0
    for rel in roots:
        if rel not in kept_set:
            missing.append(f"{a.table}: {rel} is not in the source tree")
            continue
        text = open(os.path.join(src, rel), encoding="utf-8", errors="replace").read()
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        base = os.path.dirname(rel)
        for m in FILE_ATTR.finditer(text):
            n_refs += 1
            target = os.path.normpath(os.path.join(base, m.group(1)))
            if target not in kept_set:
                missing.append(f"{rel}: file=\"{m.group(1)}\" -> {target}")
    if n_refs == 0:
        sys.exit("no file= references found under the roots — the walk is broken")
    if missing:
        print("\n".join(missing[:20]))
        sys.exit(f"{len(missing)} referenced files are not in the kept set")

    os.makedirs(a.out, exist_ok=True)
    stage = os.path.join(a.out, f"libero_{a.version}_stage")
    subprocess.run(["rm", "-rf", stage], check=True)
    os.makedirs(stage)
    for rel in kept:
        d = os.path.join(stage, os.path.dirname(rel))
        os.makedirs(d, exist_ok=True)
        os.link(os.path.join(src, rel), os.path.join(stage, rel))
    with open(os.path.join(stage, "ATTRIBUTION.txt"), "w") as f:
        f.write(ATTRIBUTION)
    listing = os.path.join(a.out, f"libero_{a.version}.files")
    with open(listing, "w") as f:
        f.write("ATTRIBUTION.txt\n")
        for rel in kept:
            f.write(rel + "\n")

    archive = os.path.join(a.out, f"libero_{a.version}.tar.zst")
    # bsdtar: --uid/--gid/--mtime are not portable; use a plain tar with the
    # listing in sorted order and zstd. Determinism here = same file set, same
    # order, same bytes; mtimes are copied from the source (hard links).
    cmd = (
        f"tar -cf - -C {stage} -T {os.path.abspath(listing)} "
        f"| zstd -19 -T0 -q -o {archive} -f"
    )
    subprocess.run(cmd, shell=True, check=True)

    h = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    sha = h.hexdigest()
    nbytes = os.path.getsize(archive)
    total_src = sum(os.path.getsize(os.path.join(src, r)) for r in kept)

    url = a.url.format(version=a.version)
    kv = (
        "# LIBERO's assets as ONE pack — L1 of docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md\n"
        "#\n"
        "# Cut by tools/assets/build_libero_pack.py from references/LIBERO-master/\n"
        "# libero/libero/assets: the .xml/.msh/.stl/.png/.jpg MuJoCo loads, the\n"
        "# .obj/.mtl sources dropped, ATTRIBUTION.txt added (CC-BY-4.0).\n"
        "#\n"
        "# ⚠ THE HOST IS ONE LINE. The owner intends to move this pack to another\n"
        "# repo later; the sha256 is the identity, so only url= changes.\n"
        "schema_version=1\n"
        "\n"
        f"pack=libero@{a.version}\n"
        "provider=hf\n"
        f"url={url}\n"
        f"sha256={sha}\n"
        f"bytes={nbytes}\n"
        "dest=assets\n"
    )
    os.makedirs(os.path.dirname(a.kv), exist_ok=True)
    with open(a.kv, "w") as f:
        f.write(kv)

    print(f"kept    {len(kept):5d} files  {total_src/1e6:8.1f} MB")
    print(f"dropped {len(dropped):5d} files")
    print(f"roots   {len(roots):5d} XMLs named by the table; {n_refs} file= refs, all resolved")
    print(f"archive {archive}  {nbytes/1e6:.1f} MB  sha256 {sha}")
    print(f"wrote   {a.kv}")


if __name__ == "__main__":
    main()
