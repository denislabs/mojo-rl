#!/usr/bin/env python3
"""Load a studio EXPORT in MuJoCo and diff it against the model it came from.

    pixi run mojo run -I . tests/physics3d/test_export_roundtrip.mojo
    pixi run python scripts/check_export_vs_mujoco.py

The Mojo gate checks writer -> OUR parser -> identical records. That is a
CLOSED LOOP: an attribute we spell wrongly in a way we also read wrongly
cancels out perfectly, and the round trip stays green while the file is not
MJCF anyone else would accept. Only MuJoCo can see that, and Mojo cannot call
it — hence two steps.

The test writes `/tmp/physics3d_export_check.xml` (and names its source in
`/tmp/physics3d_export_source.txt`) precisely so this can run afterwards.
"""
import sys
from pathlib import Path

import numpy as np

EXPORT = Path("/tmp/physics3d_export_check.xml")
SOURCE = Path("/tmp/physics3d_export_source.txt")


def main() -> int:
    if not EXPORT.exists():
        print(f"no export at {EXPORT} — run the Mojo gate first")
        return 2
    import mujoco

    src_path = SOURCE.read_text().strip()
    a = mujoco.MjModel.from_xml_path(src_path)
    b = mujoco.MjModel.from_xml_path(str(EXPORT))

    fails = 0
    for name in ("nbody", "ngeom", "njnt", "nu", "nq", "nv", "nsite"):
        x, y = getattr(a, name), getattr(b, name)
        if x != y:
            fails += 1
            print(f"FAIL {name}: source {x}, export {y}")
        else:
            print(f"ok   {name} = {x}")

    # ⚠ THE EXPORT CARRIES ONE DELIBERATE EDIT, so geom_size is EXPECTED to
    # differ in exactly one row. Everything derived from the tree must not.
    for name, tol in (("body_pos", 1e-12), ("body_quat", 1e-12),
                      ("body_mass", 1e-9), ("jnt_axis", 1e-12)):
        d = float(np.abs(getattr(a, name) - getattr(b, name)).max())
        if d > tol:
            fails += 1
            print(f"FAIL {name}: max diff {d:g} > {tol:g}")
        else:
            print(f"ok   {name} max diff {d:g}")

    nsize = int((np.abs(a.geom_size - b.geom_size) > 1e-12).any(axis=1).sum())
    # ⚠ NON-VACUITY: the edit MUST be visible, or this whole file could be
    # comparing an export that ignored the record and re-read the source.
    if nsize != 1:
        fails += 1
        print(f"FAIL geom_size: {nsize} rows differ, expected exactly 1 (the edit)")
    else:
        print("ok   geom_size differs in exactly 1 row — the edit survived")

    print(f"=== {'FAILED' if fails else 'PASSED'} ({fails} failure(s)) ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
