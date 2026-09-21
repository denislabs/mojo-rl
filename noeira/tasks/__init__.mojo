"""`noeira.tasks` — declarative tasks over the physics3d scene composer.

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
from .family import (
    compose_family, write_family_scene, scene_path, scene_dir, task_path,
    park_pos,
    SCENE_DIR, BASE_PREFIX, PARK_SPACING,
)
from .predicates import (
    Goal, BoundGoal, GoalTerm, BoundTerm,
    parse_goal, bind_goal, require_tier_a, slot_body_id, site_id,
    joint_id, joint_qpos_addresses,
    op_name, op_arity, op_is_tier_a, op_is_composite, op_reads_contacts,
    MAX_GOAL_TERMS,
    CMP_NONE, CMP_LT, CMP_LE, CMP_GT, CMP_GE, cmp_from_name, cmp_name,
)
from .sampler import (
    Placement, RegionFrame, SampleReport,
    sample_placements, MAX_PLACE_ATTEMPTS, PLACEMENT_SALT,
)
from .eval import (
    eval_goal, region_sites, HostState,
    pred_in_rect, pred_near, pred_above, pred_upright,
    pred_joint, pred_ontop, pred_box_in, pred_box_under,
    body_in_slot, slots_touching,
    region_rects, region_half_heights, region_box_flags,
    region_contact_bodies,
)
from .tape import (
    encode_goal, eval_tape, tape_needs_l3,
    TAPE_WORDS, TERM_WORDS, MAX_TAPE_TERMS,
)
from .gpu_eval import (
    eval_tape_gpu, tape_distance_gpu, region_table_words, require_gpu_regions,
    CUR_IDX_REGION_SITE, CUR_IDX_REGION_X0, CUR_IDX_REGION_Y0,
    CUR_IDX_REGION_X1, CUR_IDX_REGION_Y1, CUR_IDX_REGION_H,
    CUR_IDX_REGION_BOX, CUR_IDX_REGION_CONTACT, REGION_WORDS,
    MAX_CURRICULUM_REGIONS,
)
from .reset import (
    SlotAddress, free_slot_addresses, reset_slots,
    write_free_pose, write_free_vel_zero,
)
from .active import (
    active_mask, mask_slots, MASK_SLOT_LIMIT,
)
from .obs import (
    slot_active, write_free_slot_obs, write_free_slot_obs_host,
    FREE_JOINT_NQ, FREE_JOINT_NV,
)
from .init_table import (
    InitTable, write_init_table, load_init_table, append_init_rows,
    family_key, INIT_COLUMN, TASK_COLUMN, MASK_COLUMN, INIT_TIME_WORDS,
)
from .eval_report import SuccessReport
