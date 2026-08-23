"""EPA's polytope, in a tensor row instead of on the per-thread stack.

⚠ THIS IS THE REFERENCE'S OWN STORAGE CLASS, not a Mojo workaround.
`mjc_penetration` (`engine_collision_convex.c:98`) hands EPA a
`config->buffer` that is either the thread-local `ccd_buffer` or
`mj_stackAllocByte(d, mjc_ccdSize(...))` — i.e. a slab carved out of
**mjData's arena**, sized by `mjc_ccdSize` (`engine_collision_gjk.c:2283`)
and re-used by every collision in the step. MuJoCo has never put the
polytope on the C stack. Ours did, and that is what pinned heightfields to
the CPU: the prism is a sixth shape for GJK, so `HFIELD_ENABLED` compiles a
SECOND instantiation of `gjk_epa_witness` into the collision kernel, and
two ~7 KB frames overflow the Metal per-thread stack ("Compute function
exceeds available stack space" — the ceiling `MC_MAX_POLYVERT` sits on too).

⚠ ONE ROW PER ENV, AND THAT IS WHAT MAKES IT RACE-FREE. The collision
kernels run one THREAD PER ENV (`_detect_contacts_fields_kernel`,
`_sap_narrow_kernel`), so `ws[env, ...]` is private to a thread by exactly
the same argument that makes `d.contacts[env, ...]` private — MuJoCo's
`ccd_buffer` is `mjTHREADLOCAL` for the same reason. A single shared row
WOULD be a race; the note in `_support` about a scratch tensor being racy
was written against that shape and does not apply here.

The prism itself stays on the stack: `mjc_ConvexHField` rebuilds those six
vertices for every grid cell it walks, so they are per-CALL data, eighteen
floats wide, not model or per-env data.

⚠ THE CAPS ARE UNCHANGED BY THE MOVE. They still bound the polytope and
overflow is still REPORTED rather than truncated; only the storage moved.
Raising them is a separate, measurable change — `EPA_ITER_HARD_CAP` and the
`nev`/`nef` guards are what a model's `ccd_iterations` is actually clamped
by, and MuJoCo's own allocation (`5 + iterations` verts, `6*iterations`
faces) is far larger.
"""

from layout import Layout


# A reference EPA run over the pairs this engine actually collides —
# cylinder/box, box/box and capsule/box across penetrations from 1e-4 to 0.05,
# plus sawyer's obj against the 883-vertex eGripperBase hull — peaks at **32
# faces and 18 verts**, so these are set to 2x that. Re-measure if much larger
# hulls arrive.
comptime EPA_V_CAP: Int = 36
comptime EPA_F_CAP: Int = 64

# ---- row layout ------------------------------------------------------------
# `ev` — polytope vertices, 9 floats each: the Minkowski point (0..2) and the
# two witness points (3..5, 6..8). EPA carries the witnesses through expansion,
# which is why the stride is 9 and not 3.
comptime CCD_WS_EV: Int = 0
# `ef` — faces, three vertex indices each. Stored as DTYPE like every other
# index tensor in the fields path (`mesh_polyvert`, `mesh_edges`); the values
# are bounded by `EPA_V_CAP` = 36 and so are exact in float32 as well.
comptime CCD_WS_EF: Int = CCD_WS_EV + EPA_V_CAP * 9
# `vis` — one 0/1 flag per face: can the new support point see it.
comptime CCD_WS_VIS: Int = CCD_WS_EF + EPA_F_CAP * 3
# `hor` — the horizon, two vertex indices per edge.
comptime CCD_WS_HOR: Int = CCD_WS_VIS + EPA_F_CAP
comptime CCD_WS_SIZE: Int = CCD_WS_HOR + EPA_F_CAP * 6

# The single-row spelling, for host callers that collide one pair at a time
# (every gate and probe in `tests/physics3d`). The engine binds
# `[BATCH, CCD_WS_SIZE]` instead and passes the env index as `wrow`.
comptime L_CCD_WS1 = Layout.row_major(1, CCD_WS_SIZE)
