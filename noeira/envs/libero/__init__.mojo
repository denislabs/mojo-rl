"""`noeira.envs.libero` — LIBERO on the task layer.

LIBERO's four suites (goal, object, spatial, and the twenty LIBERO-10/90
scene families) expressed as task-layer families: the robosuite Panda under
an OSC_POSE controller, BDDL problems translated into `.family` / `.task`
files, and the demos imported into a `TrajectoryStore`.

This package is a TASK ROOT for `noeira.tasks`: `families/`, `tasks/` and
`scenes/` sit beside the code, and `load_family` finds a family's tasks and
scene through `FamilySpec.root`. The dependency is one-way — this package
imports `noeira.tasks`; the task layer never names it.

Layout:
    bddl, importer, categories, fixtures, init_z, state_remap, visual,
    osc_config, act, demos     the port (Python-free)
    libero_{goal,object,spatial}_config   the three suite env configs
    models/        per-family comptime XML + dims + contact budgets (generated)
    placement/     per-family device placement tables (generated)
    families/ tasks/ scenes/   the task root (generated from LIBERO's BDDL)
    objects/ arenas/ tables/   generated object/arena XML and the .kv tables
    assets/        the asset pack (`pixi run assets-pull libero`; not in git)

Nothing is re-exported here: `models/` carries large comptime XML, and a
consumer imports the module it needs.
"""
