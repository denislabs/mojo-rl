"""Record layouts for the tracer's own appearance tables.

`Model` carries what the SOLVER reads. These tables carry what a PICTURE
reads — materials, textures, lights — and they are separate for the reason
`MODEL_GEOM_RGBA_SIZE` is separate, said once more and louder: the geom record
is walked by the broadphase, the narrow phase and the solver on every step,
and a texture id has no business in that cache line.

⚠⚠ THE TABLES ARE INDEXED BY THE **VISUAL** GEOM NUMBER, NOT THE MODEL'S.
`VisualModel` keeps only the geoms a camera can see, in its own order, so
`vis.geoms[i]` and `mf.geoms[i]` are different geoms. The one index that still
means what it always did is `GEOM_IDX_BODY`, which addresses `Data.xpos` /
`Data.xquat` — those are the physics' own and the renderer reads them in place.
"""

# ── materials ────────────────────────────────────────────────────────────────
comptime MAX_VIS_MATERIALS: Int = 256
"""Announced, not truncated — the `MAX_GPU_MESHES` lesson. libero_goal has 74
after the Panda's visual set landed, and a material past the cap would leave a
geom pointing at a row nothing filled, i.e. a black surface that renders."""

comptime VIS_MAT_WORDS: Int = 12
comptime MAT_IDX_R: Int = 0
comptime MAT_IDX_G: Int = 1
comptime MAT_IDX_B: Int = 2
comptime MAT_IDX_A: Int = 3
comptime MAT_IDX_TEXID: Int = 4
"""Index into the texture table, or -1. `mjModel.mat_texid[.., mjTEXROLE_RGB]`
— this engine carries the RGB role only, which is the one MuJoCo's classic
renderer samples for colour."""
comptime MAT_IDX_TEXREPEAT_U: Int = 5
comptime MAT_IDX_TEXREPEAT_V: Int = 6
comptime MAT_IDX_TEXUNIFORM: Int = 7
"""1 when `<material texuniform="true">`. It changes the texture SCALE from
"once per object" to "once per spatial unit", by multiplying the repeat by the
geom's own half-size — `settexture` in `render/classic/render_gl3.c`."""
comptime MAT_IDX_SPECULAR: Int = 8
comptime MAT_IDX_SHININESS: Int = 9
comptime MAT_IDX_REFLECTANCE: Int = 10
comptime MAT_IDX_ACTIVE: Int = 11

# ── textures ─────────────────────────────────────────────────────────────────
comptime MAX_VIS_TEXTURES: Int = 64

comptime VIS_TEX_WORDS: Int = 7
comptime TEX_IDX_ADR: Int = 0
"""Offset of this texture's first BYTE in the atlas — see `VisualModel.texels`.
Three bytes per texel, row-major, row 0 first, exactly as the PNG decodes."""
comptime TEX_IDX_WIDTH: Int = 1
comptime TEX_IDX_HEIGHT: Int = 2
comptime TEX_IDX_TYPE: Int = 3
"""`TEX_2D` / `TEX_CUBE` / `TEX_SKYBOX`, `parser/flat_model.mojo`'s numbering.

⚠ A CUBE TEXTURE LOADED FROM ONE FILE IS SQUARE, NOT SIX-HIGH. MuJoCo's
compiler leaves `height == width` and uploads the SAME image to all six faces
when no `gridlayout` says otherwise (`mjr_uploadTexture`), and the arena's
`tex-table` is exactly that. A sampler that assumed a 1x6 strip would read the
wood grain out of the top sixth of the image and tile it."""
comptime TEX_IDX_ACTIVE: Int = 4
comptime TEX_IDX_NCHAN: Int = 5
comptime TEX_IDX_NLEVELS: Int = 6
"""Mip levels stored for this texture, level 0 included; 0 reads as 1.

⚠ THE LEVELS FOLLOW THE BASE IMAGE IN THE ATLAS, AND THEIR OFFSETS ARE NOT
STORED. Level `k` is `max(1, w >> k)` by `max(1, h >> k)` — OpenGL's
`floor(w / 2^k)` — and starts where level `k-1` ends, so `TEX_IDX_ADR`,
`TEX_IDX_WIDTH` and `TEX_IDX_HEIGHT` are enough to find any of them
(`appearance.mip_level_adr`). Built by `visual.append_mip_chain`, because
`render_context.c` calls `glGenerateMipmap` on every 2D and cube texture and
samples with `GL_LINEAR_MIPMAP_LINEAR`."""

# ── lights ───────────────────────────────────────────────────────────────────
comptime LIGHT_BODY_HEADLIGHT: Int = -2
"""`LIGHT_IDX_BODY` for the HEADLIGHT row.

`mjv_makeLights` makes the headlight `scn->lights[0]` — a directional light
along the camera's gaze with `mjModel.vis.headlight`'s colours (defaults
ambient 0.1, diffuse 0.4, specular 0.5), present in every scene that does not
switch it off. It is a row here rather than a kernel argument because six more
scalars would put the camera kernel over Metal's argument table, and because a
row is what the reference makes it. The shader takes its DIRECTION from the
camera and ignores the row's own.
"""

comptime MAX_VIS_LIGHTS: Int = 8
"""`mjMAXLIGHT` is 100 in MuJoCo and 8 in its classic renderer's fixed
pipeline. Eight is also every scene in this tree with room to spare — the
LIBERO arenas carry two."""

comptime VIS_LIGHT_WORDS: Int = 20
comptime LIGHT_IDX_BODY: Int = 0
comptime LIGHT_IDX_POS_X: Int = 1
comptime LIGHT_IDX_POS_Y: Int = 2
comptime LIGHT_IDX_POS_Z: Int = 3
comptime LIGHT_IDX_DIR_X: Int = 4
comptime LIGHT_IDX_DIR_Y: Int = 5
comptime LIGHT_IDX_DIR_Z: Int = 6
comptime LIGHT_IDX_DIFFUSE_R: Int = 7
comptime LIGHT_IDX_DIFFUSE_G: Int = 8
comptime LIGHT_IDX_DIFFUSE_B: Int = 9
comptime LIGHT_IDX_SPECULAR_R: Int = 10
comptime LIGHT_IDX_SPECULAR_G: Int = 11
comptime LIGHT_IDX_SPECULAR_B: Int = 12
comptime LIGHT_IDX_AMBIENT_R: Int = 13
comptime LIGHT_IDX_AMBIENT_G: Int = 14
comptime LIGHT_IDX_AMBIENT_B: Int = 15
comptime LIGHT_IDX_DIRECTIONAL: Int = 16
comptime LIGHT_IDX_CASTSHADOW: Int = 17
comptime LIGHT_IDX_CUTOFF: Int = 18
comptime LIGHT_IDX_ACTIVE: Int = 19

comptime VIS_UV_WORDS: Int = 6
"""Six floats per triangle in `VisualModel.mesh_uv` — `(u, v)` for each of
`v0`, `v1`, `v2`, in the same record order the triangle arena uses, so it is
indexed by the arena record number `MeshHit.tri` reports and by nothing
else."""


# ── the geom's appearance row ────────────────────────────────────────────────
comptime VIS_GEOM_APPEARANCE: Int = 9
comptime APP_IDX_R: Int = 0
comptime APP_IDX_G: Int = 1
comptime APP_IDX_B: Int = 2
comptime APP_IDX_A: Int = 3
comptime APP_IDX_MATID: Int = 4
"""Material index, or -1 for a geom whose colour is its own rgba.

⚠ ONE ROW RATHER THAN A SECOND TENSOR, and the reason is Metal's argument
table: 29 buffers is a SILENT metallib failure in this tree and 27 ships
(`_metals_limit_is_the_argument_table_not_the_stack`). Folding the material id
into the colour row the kernel already binds costs three unused floats per
visual geom and buys a whole operand back."""
comptime APP_IDX_REFLECT: Int = 6
"""`mat_reflectance`, but **zero on every reflective geom but the first**.

⚠⚠ THAT IS NOT A SIMPLIFICATION, IT IS THE REFERENCE. `mjr_render` runs

    // allow only one reflective geom
    int j = 0;
    for (int i=0; i < ngeom; i++) {
      if (j) { scn->geoms[i].reflectance = 0; }
      else if (isReflective(scn->geoms + i)) { j = 1; }
    }

before it draws anything — "more would result in weird transparency" — and
`isReflective` is `(PLANE or BOX) and not transparent and reflectance > 0`,
so a reflective MESH or CYLINDER is not reflective at all. Storing the rule's
OUTPUT per geom is what lets the shader ask one question (`is my reflectance
positive`) instead of re-deriving a scene-wide precedence per pixel.

On `libero_goal` exactly one geom survives it: `flat_stove_1_base_vis`, a box
with `reflectance="0.5"`. The cabinet's and the objects' materials carry 0.5
too and none of them qualifies — they are meshes.
"""

comptime APP_IDX_UVADR: Int = 5
"""Where this geom's mesh UVs start, in TRIANGLES, or -1 when it has none.
A duplicate of `MESH_META_IDX_TRIADR` for the geom's mesh, hoisted so the
sampler does not need a second mesh-table read after the hit."""

comptime APP_IDX_COND_QADR: Int = 7
"""For a CONDITIONAL SITE row — a row at index `>= VisualModel.ngeom` — the
`qpos` address its visibility reads, or -1 for always visible. Unread on an
ordinary geom row.

⚠ WHY A SITE IS A ROW HERE AT ALL. MuJoCo draws sites, and a benchmark can
make one appear: LIBERO's `FlatStove` shows its red `burner` site only once
the knob is turned on (`set_visualization` toggles `site_rgba[3]` every step
from `turn_on(qpos)`), and `turn_on_the_stove`'s recorded frames end with it
lit. The rows sit AFTER the ordinary geoms so `ray_model`'s loop over
`ngeom` never sees them; `render.trace_visual` tests each one, per lane,
against this lane's `qpos`."""
comptime APP_IDX_COND_MIN: Int = 8
"""Visible when `qpos[COND_QADR] >= COND_MIN`."""
