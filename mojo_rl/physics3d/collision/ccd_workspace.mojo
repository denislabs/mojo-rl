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


# ---- EPA's polytope caps ---------------------------------------------------
# ⚠⚠ THESE ARE MuJoCo'S OWN ALLOCATION FORMULA, NOT NUMBERS WE CHOSE.
# `mjc_ccd` carves the polytope out of `config->buffer` as
#
#     pt.verts = Vertex[5 + N];   pt.faces = Face[6 * N];   pt.map = Face*[6 * N]
#     pt.maxfaces = 6 * N
#
# with `N = m->opt.ccd_iterations` (`engine_collision_gjk.c`, `mjc_ccdSize`).
# `5` is the largest seed (`polytope2` and `polytope3` both leave five
# vertices) and EPA adds exactly one vertex per iteration, so `5 + N` is tight
# rather than generous. Deriving both caps from one iteration bound keeps that
# relationship visible: raising the bound raises both, and the ROW GROWS, which
# is the cost that has to be paid deliberately.
#
# ⚠ FACES ARE NEVER REUSED. `attachFace` takes `pt->faces[pt->nfaces++]` and
# `deleteFace` only marks `index = -2` — the slot stays allocated, because
# `Face::adj` refers to faces by INDEX and compacting the array would rewrite
# every adjacency. So the cap has to cover every face ever created, not the
# live set, which is why it is `6 * N` and not something near the live peak.
comptime EPA_ITER_CAP: Int = 64
comptime EPA_V_CAP: Int = 5 + EPA_ITER_CAP
comptime EPA_F_CAP: Int = 6 * EPA_ITER_CAP

# ---- the multi-contact caps ------------------------------------------------
# MuJoCo's `npolygonmax` / `nmeshdegmax`, which are RUNTIME model fields there
# — sized per model, so the reference has no cap at all. Ours are comptime
# because the offsets above have to be.
#
# `MC_MAX_POLYVERT` is the largest number of vertices in one face polygon;
# `MC_MAX_DEG` the most polygons meeting at one vertex. They live HERE, beside
# the row they size, rather than in `native_multicontact` — a constant and the
# buffer it dimensions drifting apart is exactly how the old "checked at model
# build" comment came to be false.
# ⚠ BOTH ARE MEASURED WORSTS, not round numbers, and they are measured over
# MENAGERIE **AND** THIS REPO'S OWN MODELS. robotiq_2f85's base_mount carries a
# 144-vertex face and 21 scenes have one wider than the 56 the width used to
# be; the degree used to be 48, which covers Menagerie (flexiv_rizon4's 47) and
# does NOT cover `envs/robots/assets/so_arm101.xml`, whose STS3215 servo hulls
# (`sts3215_03a_v1`, `sts3215_03a_no_horn_v1`) each carry a vertex with **50**
# incident polygons and its mirror with 49.
#
# ⚠ SO THE CENSUS HAS TO INCLUDE THE MODELS WE SHIP, NOT ONLY THE REFERENCE
# TREE. The old 48 was Menagerie's worst plus one, and it read as a bound on
# "every model" because nothing had ever measured the other half of the corpus.
# Per-scene, collision meshes only: 96 Menagerie scenes give 144 / 47, and all
# 57 in-repo models give 82 / 50 — worst overall 144 / 50.
#
# ⚠⚠ A WHOLE-DIRECTORY CENSUS OVER MESH **FILES** ANSWERS A DIFFERENT QUESTION.
# Sweeping all 2 149 `.stl`/`.obj` in the tree gives 395 / 187, because most of
# them are visual-only geometry that never reaches a collision routine. The
# number these have to cover is the per-SCENE one.
#
# ⚠ WHAT THE RAISE BUYS IS A BOUND, NOT A MEASURED CONTACT. Pressed on that
# exact corner from 128 distinct orientations, shrinking the cap to **8** —
# dropping 42 of the 50 candidates, not 2 — moves NOTHING: every contact still
# matches MuJoCo to 3.6e-15 in position and 8.5e-07 degrees in normal. The
# reason is structural and worth knowing before anyone tries to gate this: a
# vertex has high degree because it is finely tessellated, so its incident
# normals arrive in near-duplicate clusters (indices 46 and 48 here are 0.1
# degrees apart), `_aligned_faces` takes the FIRST match within `MC_FACE_TOL`
# (0.092 degrees), and an early member of the cluster wins before a late one is
# ever reached. Running a shipped model with a knowingly truncated candidate
# list is still not a thing to leave standing.
comptime MC_MAX_POLYVERT: Int = 144
comptime MC_MAX_DEG: Int = 50
comptime MC_CLIP_CAP: Int = 2 * MC_MAX_POLYVERT

# ---- row layout ------------------------------------------------------------
# The `Polytope` struct of `engine_collision_gjk.c`, one region per field.
# Everything is stored as `DTYPE` — including the indices, exactly as the rest
# of the fields path stores them (`mesh_polyvert`, `mesh_edges`) — and every
# index is bounded by `EPA_F_CAP`, so all of them are exact in float32 too.
#
# `ev` — polytope vertices, `Vertex`. 11 floats: the Minkowski point (0..2),
# the two witness points (3..5, 6..8) and the two SUPPORT INDICES (9, 10).
# EPA carries the witnesses through expansion, and the indices are MuJoCo's
# `index1`/`index2` — the box corner code or the mesh hull vertex, which is
# what the discrete repeated-support-point break compares.
comptime CCD_WS_EV: Int = 0
comptime EPA_V_STRIDE: Int = 11
# `ef` — `Face::verts`, three vertex indices per face.
comptime CCD_WS_EF: Int = CCD_WS_EV + EPA_V_CAP * EPA_V_STRIDE
# `eadj` — `Face::adj`, the face across each edge: [v1,v2], [v2,v3], [v3,v1].
comptime CCD_WS_EADJ: Int = CCD_WS_EF + EPA_F_CAP * 3
# `efv` — `Face::v`, the origin projected onto the face's plane. Doubles as the
# face normal, unnormalised, with |v| the distance to the origin.
comptime CCD_WS_EFV: Int = CCD_WS_EADJ + EPA_F_CAP * 3
# `efd` — `Face::dist2`, the squared norm of `v`.
comptime CCD_WS_EFD: Int = CCD_WS_EFV + EPA_F_CAP * 3
# `efi` — `Face::index`: >= 0 the slot in `map`, -1 not in map, -2 deleted.
comptime CCD_WS_EFI: Int = CCD_WS_EFD + EPA_F_CAP
# `map` — the CANDIDATE face list. A face joins it only when its distance lies
# between the current lower and upper bounds, so this is a strict subset of the
# polytope and NOT the same thing as "every face".
comptime CCD_WS_MAP: Int = CCD_WS_EFI + EPA_F_CAP
# `hor` — the horizon, (face index, edge index) per entry. ⚠ MuJoCo sizes
# `horizon.indices` at 24 and never checks it; ours is `EPA_F_CAP` because the
# horizon cannot exceed the faces it is built from.
comptime CCD_WS_HOR: Int = CCD_WS_MAP + EPA_F_CAP
# `hstk` — the explicit stack for `horizonRec`, (face, edge, state) per frame.
# ⚠ MuJoCo RECURSES. A GPU kernel cannot, so the recursion is unrolled into
# this stack; the traversal ORDER is preserved exactly, because the order the
# horizon edges are added in decides which edge seeds the new face fan.
comptime CCD_WS_HSTK: Int = CCD_WS_HOR + EPA_F_CAP * 2
# `center` — `Polytope::center`, the seed's centroid. `attachFace` orients each
# face's projection away from it.
comptime CCD_WS_CTR: Int = CCD_WS_HSTK + EPA_F_CAP * 3
comptime EPA_WS_SIZE: Int = CCD_WS_CTR + 3


# ---- the multi-contact region ----------------------------------------------
# `native_multicontact`'s polygon buffers, for the same reason and by the same
# mechanism. MuJoCo sizes its equivalents from `npolygonmax` / `nmeshdegmax`,
# which are RUNTIME MODEL FIELDS — it has no cap at all.
#
# ⚠ ONLY THE `MC_MAX_POLYVERT`-SIZED ARRAYS MOVE. The `MC_MAX_DEG` ones
# (`n1`/`n2`/`idx1`/`idx2`/`endverts`, ~4.4 KB together at 50) stay on the
# stack: that axis is small and grows slowly — 48 -> 50 is 240 more bytes per
# frame — so it was never the one that needed unlocking. The width axis is: the
# tree's worst face is robotiq_2f85's 144 vertices, and 21 scenes carry a
# polygon wider than 56. ⚠ The degree arrays being on the STACK is why raising
# `MC_MAX_DEG` still has to re-run the Metal canary
# (`tests/physics3d/test_plane_mesh_fields.mojo`) even though `CCD_WS_SIZE`
# does not move: the ceiling it would hit is the per-thread stack, not the row.
#
# ⚠ THE CAP DEGRADES SILENTLY WHEN IT BITES. `_mesh_face` returns 0 past it,
# which is the routine's own "the features do not line up" answer, so the
# caller emits the single EPA point — the reference's own fallback, reached
# for a reason the reference does not have. It is a LOST MANIFOLD.
comptime MC_WS_FACE1: Int = EPA_WS_SIZE
comptime MC_WS_FACE2: Int = MC_WS_FACE1 + MC_MAX_POLYVERT * 3
# The clipped ring. A clip can reach the sum of the two input sizes, hence
# `MC_CLIP_CAP = 2 * MC_MAX_POLYVERT` rather than `MC_MAX_POLYVERT`.
comptime MC_WS_OUT: Int = MC_WS_FACE2 + MC_MAX_POLYVERT * 3
# `_polygon_clip`'s two working rings and its per-edge plane cache.
comptime MC_WS_POLY: Int = MC_WS_OUT + MC_CLIP_CAP * 3
comptime MC_WS_CLIPPED: Int = MC_WS_POLY + MC_CLIP_CAP * 3
comptime MC_WS_PN: Int = MC_WS_CLIPPED + MC_CLIP_CAP * 3
comptime MC_WS_PD: Int = MC_WS_PN + MC_MAX_POLYVERT * 3

# ⚠ ONE TENSOR, TWO REGIONS, AND THE WHOLE ROW IS ALWAYS ALLOCATED. EPA's
# polytope and the multi-contact polygons are live at DIFFERENT times within
# one collision — the manifold routine runs after `gjk_epa_witness` returns —
# so they could have overlapped. They do not, deliberately: an aliasing bug
# between two regions that are "obviously" disjoint in time is invisible in a
# diff and fires only on the pose where the assumption breaks.
#
# EPA's 964 floats plus 28 * MC_MAX_POLYVERT.
comptime CCD_WS_SIZE: Int = MC_WS_PD + MC_MAX_POLYVERT

# The single-row spelling, for host callers that collide one pair at a time
# (every gate and probe in `tests/physics3d`). The engine binds
# `[BATCH, CCD_WS_SIZE]` instead and passes the env index as `wrow`.
comptime L_CCD_WS1 = Layout.row_major(1, CCD_WS_SIZE)
