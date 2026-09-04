"""`mojo_rl.tasks` — declarative tasks over the physics3d scene composer.

See `docs/TASK_LAYER_PLAN.md` (design) and
`docs/TASK_LAYER_IMPLEMENTATION.md` (what is built).

⚠ THE DEPENDENCY IS ONE-WAY, and §7 of the plan says to keep it that way:
`tasks/` CALLS `physics3d/studio`'s composer and never reimplements it, and
`physics3d` never imports `tasks` — or the engine stops being usable without
the task layer.

`tasks/` also produces a VALUE consumed by `Phyics3dEnvConfig`. It does not
implement `Env` and knows nothing about agents.
"""

from .spec import (
    FamilySpec, TaskSpec, SlotSpec, RegionSpec, InitSpec,
    parse_family, parse_task, load_family, load_task,
    validate_task_against_family,
    SLOT_FREE, SLOT_STATIC, slot_kind_name, slot_kind_from_name,
    SCHEMA_VERSION,
)
