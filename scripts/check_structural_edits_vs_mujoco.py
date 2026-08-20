#!/usr/bin/env python3
"""Load every STRUCTURALLY EDITED document in MuJoCo and diff it against ours.

    pixi run mojo run -I . tests/physics3d/test_structural_edit.mojo
    pixi run python scripts/check_structural_edits_vs_mujoco.py

⚠⚠ WHY TWO STEPS. The Mojo gate checks: nothing dangles, our loader takes the
result, the counts dropped. All three are a CLOSED LOOP — a reference we prune
wrongly in a way we also READ wrongly cancels out perfectly, and the edit stays
green while the file is not MJCF anyone else would accept. Only MuJoCo can see
that, and Mojo cannot call it.

The Mojo half writes each edited document to /tmp/physics3d_structural/N.xml
and a manifest line carrying OUR counts, precisely so this can run afterwards.
"""
import sys
from pathlib import Path

OUT = Path("/tmp/physics3d_structural")
FIELDS = ("nbody", "njnt", "ngeom", "nu", "nq", "nv", "nsite", "neq", "ntendon")


def main() -> int:
    manifest = OUT / "manifest.txt"
    if not manifest.exists():
        print(f"no manifest at {manifest} — run the Mojo gate first")
        return 2
    import mujoco

    rows = [ln.split() for ln in manifest.read_text().splitlines() if ln.strip()]
    if not rows:
        print("manifest is EMPTY — the Mojo gate wrote nothing to judge")
        return 2

    fails = 0
    n_expected_refusals = 0
    for row in rows:
        path, expect_load, ours = row[0], row[1] == "1", [int(x) for x in row[2:]]
        try:
            m = mujoco.MjModel.from_xml_path(path)
            loaded = True
            err = ""
        except Exception as e:
            loaded = False
            err = str(e).strip().splitlines()[0]

        # ⚠⚠ AN EDIT IS ALLOWED TO BREAK THE MODEL, and the Mojo half says
        # which ones do. Deleting the only geom of a moving body is a
        # legitimate step in a repair and MuJoCo refuses the result; the
        # requirement is that the two sides AGREE about which case it is,
        # not that every edit stays loadable.
        if not expect_load:
            n_expected_refusals += 1
            if loaded:
                fails += 1
                print(f"FAIL {path}: MuJoCo ACCEPTS a document our validator"
                      " called an error — one of the two is wrong")
            else:
                print(f"ok   {path}: MuJoCo refuses it, as the validator said")
                print(f"     {err}")
            continue

        if not loaded:
            fails += 1
            print(f"FAIL {path}: MuJoCo REFUSES the edited document")
            print(f"     {err}")
            continue
        theirs = [getattr(m, f) for f in FIELDS]
        if ours != theirs:
            fails += 1
            print(f"FAIL {path}:")
            for f, a, b in zip(FIELDS, ours, theirs):
                mark = "  <-- " if a != b else ""
                print(f"     {f:8s} ours {a:5d}   MuJoCo {b:5d}{mark}")
        else:
            print(f"ok   {path}: " + " ".join(
                f"{f}={v}" for f, v in zip(FIELDS, theirs)
            ))

    # ⚠ NON-VACUITY. An empty or truncated manifest would print "0 failures"
    # and mean nothing; the Mojo half writes six documents.
    if len(rows) < 9:
        fails += 1
        print(f"FAIL: manifest has {len(rows)} rows, expected at least 9 —"
              " the Mojo gate did not write everything it judges")

    # ⚠ NON-VACUITY, BOTH WAYS. A manifest of only-refusals would make the
    # count comparison decoration; a manifest of only-loads would never
    # exercise the disagreement check above.
    n_loads = len(rows) - n_expected_refusals
    if n_loads < 7 or n_expected_refusals < 1:
        fails += 1
        print(f"FAIL: {n_loads} must-load and {n_expected_refusals}"
              " must-refuse rows — the table needs both to mean anything")

    fails += _check_document_edit()

    print(f"\n{len(rows) - fails} / {len(rows)} edited documents agree with"
          f" MuJoCo ({n_loads} load, {n_expected_refusals} refused as"
          " predicted)")
    return 1 if fails else 0


def _check_document_edit() -> int:
    """The values a FAST-PATH edit wrote — does MuJoCo read them back?

    ⚠⚠ `test_edit_reaches_the_document` proves our writer and our PARSER agree.
    That is a closed loop: a wrong spelling read wrongly cancels out exactly.
    `size` in particular is per-type and `fromto` silently overrides it, so the
    interesting failure is a file that reads back with the OLD number — which
    only the reference can see.
    """
    import mujoco

    doc = OUT / "doc_edit.xml"
    expect = OUT / "doc_edit_expect.txt"
    if not doc.exists() or not expect.exists():
        print("SKIP document-edit check — run"
              " tests/physics3d/test_edit_reaches_the_document.mojo first")
        return 1

    gname, size0, posz, bname, mass = expect.read_text().split()
    m = mujoco.MjModel.from_xml_path(str(doc))
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
    if gid < 0 or bid < 0:
        print(f"FAIL document edit: MuJoCo cannot find geom {gname!r}"
              f" / body {bname!r}")
        return 1

    bad = 0
    for label, got, want in (
        (f"geom {gname} size[0]", m.geom_size[gid][0], float(size0)),
        (f"geom {gname} pos[2]", m.geom_pos[gid][2], float(posz)),
        (f"body {bname} mass", m.body_mass[bid], float(mass)),
    ):
        if abs(got - want) > 1e-9:
            bad += 1
            print(f"FAIL document edit: {label} — MuJoCo reads {got},"
                  f" we wrote {want}")
        else:
            print(f"ok   document edit: {label} = {got}")
    return bad


if __name__ == "__main__":
    sys.exit(main())
