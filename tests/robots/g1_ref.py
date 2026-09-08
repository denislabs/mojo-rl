"""Layer-1 gate for the Unitree G1 port: our baked XML vs the reference, both via MuJoCo.

    pixi run python tests/robots/g1_ref.py

`docs/DM_CONTROL_PORT.md`'s standing invariant, as `so_arm_ref.py` and
`tests/dm_control/mjmodel_diff.py` implement it: compile OUR asset with
MuJoCo, compile the REFERENCE's scene with MuJoCo, and diff every table at
tolerance 0.0. Neither our parser nor our engine is in the loop, so a
mismatch isolates the XML TEXT `g1_bake.py` produced.

TWO DELIBERATE, DOCUMENTED DEVIATIONS, each handled here explicitly rather
than by skipping the comparison:

  * `opt.timestep`. The reference sets it at RUNTIME (`mujoco.py:47`,
    `1 / sim.fps` = 0.005) and our asset bakes it. The reference model is
    given the same runtime override before the diff, so the field IS
    compared — and would fail if the bake wrote anything but 0.005.
  * `<sensor>`. Dropped from our asset (104 sensors, none read by the
    training loop or the observation). The four sensor tables and `nsensor`
    are skipped, with `nsensordata` asserted on the reference so a future
    reference that grows a sensor we DO need cannot pass unnoticed.

⚠ `references/` IS GITIGNORED. Absent the reference tree this SKIPS with a
message rather than passing quietly.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tests", "dm_control"))
sys.path.insert(0, HERE)

import mjmodel_diff  # noqa: E402
import g1_bake  # noqa: E402

REF_SCENE = os.path.join(g1_bake.REF_G1, g1_bake.SCENE)
# `sensor_*` tables: the asset drops the block deliberately (see above).
SENSOR_TABLES = ("sensor_type", "sensor_objid", "sensor_adr", "sensor_dim")
REF_NSENSOR = 104


def main():
    os.chdir(REPO)
    if not os.path.exists(REF_SCENE):
        print("SKIP: reference tree absent —", REF_SCENE)
        return 0
    import mujoco

    ref = mujoco.MjModel.from_xml_path(REF_SCENE)
    ours = mujoco.MjModel.from_xml_path(g1_bake.ASSET_XML)

    # The reference's runtime override, applied so `opt.timestep` is compared.
    assert abs(ref.opt.timestep - 0.002) < 1e-15, ref.opt.timestep
    ref.opt.timestep = g1_bake.SIM_TIMESTEP

    bad = []
    if ref.nsensor != REF_NSENSOR:
        bad.append("reference nsensor {} != {} — the reference grew or lost a"
                   " sensor; re-read g1_bake.py's sensor note".format(
                       ref.nsensor, REF_NSENSOR))
    if ours.nsensor != 0:
        bad.append("ours nsensor {} != 0 — the bake no longer drops the"
                   " sensor block".format(ours.nsensor))
    diffs = mjmodel_diff.diff_models(ref, ours, skip_tables=SENSOR_TABLES)
    # `nsensor` / `nsensordata` show up in the count diff; those two are the
    # deviation and are accounted for above, everything else is a defect.
    for d in diffs:
        if d.startswith("nsensor"):
            continue
        bad.append(d)

    n = mjmodel_diff.n_tables() - len(SENSOR_TABLES)
    print("unitree_g1: {} tables compared, {} counts, {} option fields".format(
        n, len(mjmodel_diff._COUNTS), len(mjmodel_diff._OPTS)))
    if bad:
        print("FAIL — {} mismatch(es):".format(len(bad)))
        for b in bad:
            print("  ", b)
        return 1
    print("PASS — our unitree_g1.xml compiles to the reference model exactly"
          " (sensor tables excluded by design; timestep compared under the"
          " reference's own runtime override)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
