"""`manipulation/stack_2_bricks_moveable_base_features` as a comptime MJCF string.

⚠⚠ THIS FILE IS GENERATED, NOT WRITTEN. dm_control builds this task with
`composer`, which ASSEMBLES entities at runtime and only then flattens them to
MJCF. There is no hand-authored XML upstream to port: `mjcf.export_with_assets`
is the source of truth. Regenerate with the generator that produced it; do not
hand-edit.

One edit is applied to the export, and only one:
  * `file="<hash>.stl"` -> `file="mojo_rl/envs/dm_control/assets/jaco/<hash>.stl"`,
    because a comptime model is loaded from the repo root rather than from the
    export directory. The content-hashed basenames are PyMJCF's own.

⚠ ALL 13 MANIPULATION TASKS SHARE THE SAME NINE MESHES — the Jaco arm and
hand. Every prop in this family (Duplo bricks, the large box, the pedestal and
cradle) is built from PRIMITIVES, so no task adds an asset. They are committed
under `assets/jaco/` already.
"""

