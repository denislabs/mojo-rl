"""`manipulation/reach_site_features` — the baked Jaco model, as MJCF text.

⚠⚠ THIS FILE IS GENERATED, NOT WRITTEN. dm_control builds this task with
`composer`, which ASSEMBLES entities at runtime (arena + Jaco arm + Jaco hand)
and only then flattens them to MJCF. There is no hand-authored XML upstream to
port: `mjcf.export_with_assets` is the source of truth, and
`tests/dm_control/manipulation_ref.py::bake` is what produced this. Regenerate
by re-running that and pasting, do not hand-edit.

Two edits are applied to the export, and only two:
  * `file="<hash>.stl"` -> `file="mojo_rl/envs/dm_control/assets/jaco/<hash>.stl"`,
    because a comptime model is loaded from the repo root rather than from the
    export directory. The content-hashed basenames are PyMJCF's own.
  * nothing else.

⚠ THE MESHES ARE COMMITTED ALONGSIDE, under `assets/jaco/`. They are the
kinova STLs from `references/dm_control-main/dm_control/third_party/kinova/`,
copied under PyMJCF's content-hashed names so the `file=` attributes here
resolve without a bake step. The reference tree is read-only and not on the
runtime path, so pointing at it directly would tie the env to a checked-out
reference.

⚠ WHY A COMPTIME COPY AT ALL, when the gates parse a freshly baked XML: the
gates use the RUNTIME path (`parse_xml_full` + `build_model_fields_from_flat`),
which carries no actuators. `Phyics3dEnv` is built on `ModelDefFromXML`, the
COMPTIME path, and that is what owns `apply_actions`. Stepping this model at
all requires the comptime route, so the XML has to exist as a comptime String.
"""

