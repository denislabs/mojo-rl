"""Parked-slot pose — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/tasks/gen_park_scenes.py
CI checks it with: pixi run python tools/tasks/gen_park_scenes.py --check

The pose `tools/tasks/gen_park_scenes.py` writes into
`assets/so101_park_k*.xml`, re-exported for the repark hook so the scene and
the hook cannot drift apart. Read that generator's docstring for WHY the pose
is high and lateral rather than the `(0, 0, -2)` of
`docs/TASK_LAYER_PLAN.md` §4.2 — the short version is that the floor is an
INFINITE plane, so below it is a four-contact penetration, not an absence.

Slot poses:
#   slot 0: (10.0, 0.0, 50.0)
#   slot 1: (10.5, 0.0, 50.0)
#   slot 2: (11.0, 0.0, 50.0)
#   slot 3: (11.5, 0.0, 50.0)
#   slot 4: (12.0, 0.0, 50.0)
#   slot 5: (12.5, 0.0, 50.0)
#   slot 6: (13.0, 0.0, 50.0)
#   slot 7: (13.5, 0.0, 50.0)
#   slot 8: (14.0, 0.0, 50.0)
"""

comptime PARK_X: Float64 = 10.0
comptime PARK_Y: Float64 = 0.0
comptime PARK_Z: Float64 = 50.0
comptime PARK_SPACING: Float64 = 0.5

comptime SLOT_HALF_EXTENT: Float64 = 0.02
comptime SLOT_MASS: Float64 = 0.05


@always_inline
fn park_pos_x(slot: Int) -> Float64:
    """The x of parked slot `slot` — the scene's own arithmetic, once."""
    return PARK_X + Float64(slot) * PARK_SPACING
