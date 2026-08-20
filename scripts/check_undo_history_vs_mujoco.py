"""Every document the undo stack hands back must LOAD IN MuJoCo. V2.9.

⚠⚠ WHY AN EXTERNAL ARM AT ALL. `test_undo_history` compares each restored
document against the original by parsing it with OUR parser — which is the
right internal arm and is blind to exactly one thing: an undo that produces
text MuJoCo refuses. Our parser is more permissive than the reference in
several documented places (a dangling `joint=` on an actuator resolves to -1
and simulates as zero force rather than as a load error), so "we parse it to
the same record" does not imply "it is still a model".

That matters here because the states in the stack are not all files a human
wrote: a delete prunes five sections, a reparent moves a subtree, and an undo
hands back the text from BEFORE those. If any step in the chain leaves the
document malformed, this is what says so.

⚠ AND IT CHECKS THE COUNTS, NOT JUST THE LOAD. A document that loads as an
empty model would pass a load-only check; the "undone" state must report the
SAME nbody as the "opened" state and the edited ones must DIFFER, or the
comparison is vacuous.

Run (after the Mojo test, which writes the directory):
    pixi run mojo run -I . tests/physics3d/test_undo_history.mojo
    pixi run python scripts/check_undo_history_vs_mujoco.py
"""

import os
import sys

import mujoco

DUMP = "/tmp/undo_history"


def main() -> int:
    if not os.path.isdir(DUMP):
        print(f"FAIL: {DUMP} missing — run the Mojo test first")
        return 1
    files = sorted(
        (f for f in os.listdir(DUMP) if f.endswith(".xml")),
        key=lambda f: int(f.split("_", 1)[0]),
    )
    if not files:
        print(f"FAIL: no documents in {DUMP}")
        return 1

    dims: dict[str, tuple[int, int, int, int]] = {}
    fails = 0
    for f in files:
        path = os.path.join(DUMP, f)
        tag = f.split("_", 1)[1][: -len(".xml")]
        try:
            m = mujoco.MjModel.from_xml_path(path)
        except Exception as e:  # noqa: BLE001 — the verdict IS the exception
            print(f"  FAIL: MuJoCo refuses {f}: {str(e).splitlines()[0]}")
            fails += 1
            continue
        d = (m.nbody, m.ngeom, m.njnt, m.nu)
        dims[tag] = d
        print(f"  ok: {f} loads — nbody={d[0]} ngeom={d[1]} njnt={d[2]}"
              f" nu={d[3]}")

    def same(a: str, b: str, why: str) -> None:
        nonlocal fails
        if a not in dims or b not in dims:
            print(f"  FAIL: {why} — {a} or {b} was not dumped")
            fails += 1
        elif dims[a] != dims[b]:
            print(f"  FAIL: {why} — {a}={dims[a]} vs {b}={dims[b]}")
            fails += 1
        else:
            print(f"  ok: {why} ({dims[a]})")

    def differ(a: str, b: str, why: str) -> None:
        nonlocal fails
        if a not in dims or b not in dims:
            print(f"  FAIL: {why} — {a} or {b} was not dumped")
            fails += 1
        elif dims[a] == dims[b]:
            print(f"  FAIL: {why} — both are {dims[a]}, so nothing was tested")
            fails += 1
        else:
            print(f"  ok: {why} ({dims[a]} vs {dims[b]})")

    print("--- the reference agrees about what changed and what came back ---")
    differ("opened", "deleted_arm", "the delete changed the model")
    same("opened", "undone", "undo restored MuJoCo's dims exactly")
    differ("cheetah_opened", "cheetah_no_bthigh", "the cheetah delete landed")
    differ("cheetah_no_bthigh", "cheetah_with_fin", "the add landed")
    same("cheetah_opened", "cheetah_undone_twice",
         "two undos across two structural edits restored the cheetah")

    print(f"=== {len(files)} documents, {fails} failures ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
