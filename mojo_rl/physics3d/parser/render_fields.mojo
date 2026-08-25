"""`RenderFields` — the runtime render record, phase 1a.5b.

What `ComptimeRenderData` was, built at RUNTIME from `FlatModelDef` instead of
by interpreting XML in the comptime interpreter. The renderer is its only
reader, exactly as the actuation kernels are `SpecFields`' only readers, so it
is a bundle of its own rather than a family on `fields.Model`.

⚠ THE FIELD NAMES ARE `ComptimeRenderData`'S ON PURPOSE. The thirteen render
hooks in `model_def_from_xml.mojo` open with ~95 `materialize[Self._rcd.X]()`
hoists and then index `_m_X[i]` throughout their bodies. Keeping the names
identical makes the repoint one substitution per hoist with the bodies
untouched — 95 mechanical lines instead of 142 semantic ones. The grouping is
`_rcd`'s too (parallel arrays, not `List[GeomData]`) for the same reason.

⚠ ONE DELIBERATE DEPARTURE: `tex_type` carries `flat_model`'s numbering
(TEX_SKYBOX=0 / TEX_2D=1 / TEX_CUBE=2), NOT `_rcd`'s (2d=0 / skybox=1 /
cube=3). The two disagreed and NEITHER matched MuJoCo's mjtTexture
(2d=0/cube=1/skybox=2), so `tex_type == 1` meant SKYBOX on one side and 2D on
the other. Only two call sites compare it; they were fixed to name
`TEX_SKYBOX` rather than a literal, which kills the third numbering instead of
carrying it forward.

Counts (`ngeom`, `nlight`, `ncam`, `nmat`, `nsite`) are absent because nothing
reads them — the hooks take those from `ModelDefFromXML`'s comptime dimension
parameters. `ntex`, `nmesh` and `nsten` ARE read and so are kept.
"""

from .flat_model import (
    FlatModelDef,
    TEX_SKYBOX,
    TEX_2D,
    TEX_CUBE,
)
from ..gpu.constants import WRAP_SITE

# Geom-type codes, spelled here so the size dispatch below reads like the
# comptime block it mirrors. Same numbering as `physics3d.constants`.
comptime _RF_PLANE: Int = 0
comptime _RF_SPHERE: Int = 1
comptime _RF_CAPSULE: Int = 2
comptime _RF_BOX: Int = 3
comptime _RF_CYLINDER: Int = 4
comptime _RF_ELLIPSOID: Int = 6


struct RenderFields(Copyable, Movable):
    """Per-model render data, one `List` per field. CPU only.

    Built once per renderer by `build_render_fields`, never per frame: the
    hooks that read it run inside `render_frame`.
    """

    var ntex: Int
    var geom_body_id: List[Int]
    var geom_type: List[Int]
    var geom_pos_x: List[Float64]
    var geom_pos_y: List[Float64]
    var geom_pos_z: List[Float64]
    var geom_quat_x: List[Float64]
    var geom_quat_y: List[Float64]
    var geom_quat_z: List[Float64]
    var geom_quat_w: List[Float64]
    var geom_radius: List[Float64]
    var geom_half_length: List[Float64]
    var geom_half_x: List[Float64]
    var geom_half_y: List[Float64]
    var geom_half_z: List[Float64]
    var geom_rgba_r: List[Float64]
    var geom_rgba_g: List[Float64]
    var geom_rgba_b: List[Float64]
    var geom_rgba_a: List[Float64]
    var geom_material_id: List[Int]
    var geom_hfield_id: List[Int]
    """Index into the model's `<hfield>` assets, -1 for a non-hfield geom.

    ⚠ CARRIED BECAUSE THE RENDERER CANNOT GUESS IT. A `type="hfield"` geom's
    SHAPE lives in the asset — nrow/ncol and the `size` 4-vector — not in the
    geom's own record, so a renderer holding only `geom_half_*` has no way to
    draw one. This is the same gap `cam_body` was: a field the parser resolved
    and then DROPPED at the `GeomData` -> `RenderFields` boundary.
    """
    var geom_mesh_id: List[Int]
    var geom_mesh_scale: List[Float64]
    """`<mesh scale>` per GEOM, three each — 1,1,1 for a non-mesh geom.

    ⚠ PER GEOM, NOT PER ASSET, and deliberately so. `geom_mesh_id` resolves
    against the asset table BY FILENAME, and a mirrored pair (`scale="1 1 1"`
    and `scale="1 -1 1"` on the same file, which is how a model builds a left
    part from a right one) is two assets sharing one filename — so an
    asset-indexed scale would give both geoms whichever came first. The geom
    already carries its own resolved scale; read that.
    """
    var geom_group: List[Int]
    var nmesh: Int
    var mesh_names: List[String]
    var mesh_files: List[String]
    var light_dir_x: List[Float64]
    var light_dir_y: List[Float64]
    var light_dir_z: List[Float64]
    var light_diffuse_r: List[Float64]
    var light_diffuse_g: List[Float64]
    var light_diffuse_b: List[Float64]
    var light_specular_r: List[Float64]
    var light_specular_g: List[Float64]
    var light_specular_b: List[Float64]
    var light_ambient_r: List[Float64]
    var light_ambient_g: List[Float64]
    var light_ambient_b: List[Float64]
    var light_directional: List[Bool]
    var light_castshadow: List[Bool]
    var light_exponent: List[Float64]
    var cam_pos_x: List[Float64]
    var cam_pos_y: List[Float64]
    var cam_pos_z: List[Float64]
    var cam_quat_x: List[Float64]
    var cam_quat_y: List[Float64]
    var cam_quat_z: List[Float64]
    var cam_quat_w: List[Float64]
    var cam_fovy: List[Float64]
    var cam_mode: List[Int]
    var cam_target_body: List[Int]
    var cam_body: List[Int]
    """Body each camera is ATTACHED to (`mjModel.cam_bodyid`); 0 = worldbody.

    ⚠ `cam_pos`/`cam_quat` above are expressed in THIS body's frame, exactly as
    MuJoCo stores them. Without this column the render path could not tell a
    world-fixed camera from a wrist camera, so every camera was drawn at its
    LOCAL pose read as a world pose — correct only for the worldbody, where the
    transform is the identity. See `mj_camlight`, which calls `mj_local2Global`
    with `cam_bodyid` for every mode before it dispatches on `cam_mode`.
    """
    var tex_type: List[Int]
    var tex_builtin: List[Int]
    var tex_rgb1_r: List[Float64]
    var tex_rgb1_g: List[Float64]
    var tex_rgb1_b: List[Float64]
    var tex_rgb2_r: List[Float64]
    var tex_rgb2_g: List[Float64]
    var tex_rgb2_b: List[Float64]
    var tex_names: List[String]
    var tex_files: List[String]
    var tex_mark: List[Int]
    var tex_markrgb_r: List[Float64]
    var tex_markrgb_g: List[Float64]
    var tex_markrgb_b: List[Float64]
    var tex_random: List[Float64]
    var mat_rgba_r: List[Float64]
    var mat_rgba_g: List[Float64]
    var mat_rgba_b: List[Float64]
    var mat_rgba_a: List[Float64]
    var mat_shininess: List[Float64]
    var mat_specular: List[Float64]
    var mat_reflectance: List[Float64]
    var mat_tex_id: List[Int]
    var mat_texrepeat_u: List[Float64]
    var mat_texrepeat_v: List[Float64]
    var site_body_id: List[Int]
    var site_pos_x: List[Float64]
    var site_pos_y: List[Float64]
    var site_pos_z: List[Float64]
    var site_size_0: List[Float64]
    var nsten: Int
    var sten_nsite: List[Int]
    var sten_sites: List[Int]
    var sten_width: List[Float64]
    var sten_rgba_r: List[Float64]
    var sten_rgba_g: List[Float64]
    var sten_rgba_b: List[Float64]
    var vis_znear: Float64
    var vis_fogstart: Float64
    var vis_fogend: Float64
    var vis_shadowsize: Int
    var vis_headlight_ambient_r: Float64
    var vis_headlight_ambient_g: Float64
    var vis_headlight_ambient_b: Float64
    var vis_has_headlight: Bool

    # ── the model's SOURCE ────────────────────────────────────────────────
    var xml_text: String
    """The model's MJCF, verbatim. Empty when the caller did not supply it.

    ⚠ THIS IS NOT REDUNDANT WITH THE LISTS ABOVE, and it is here so the render
    hooks can stop being methods on a comptime type. Two of them read the raw
    document rather than the parse — `render_skin` walks
    `<skin file= material=>` -> `<material texture=>` -> `<texture file=>`,
    and `body_names` recovers body names the physics parse discards
    (`FlatModelDef` carries names for textures, meshes and materials, and for
    nothing else). Both used to reach `Self.xml_text()`, which only a
    `ModelDefFromXML` has, and which is exactly what a RUNTIME-loaded model
    cannot offer.

    ⚠ EMPTY IS A LEGAL VALUE and means "no skin, no body names", never a
    parse failure — a caller that has no source text still gets a complete
    renderer for everything the lists describe."""

    var asset_base_dir: String
    """Directory that `file=` attributes resolve against. Empty = cwd."""

    def __init__(out self):
        """Empty — `build_render_fields` fills it."""

        self.xml_text = String("")
        self.asset_base_dir = String("")
        self.ntex = 0
        self.geom_body_id = List[Int]()
        self.geom_type = List[Int]()
        self.geom_pos_x = List[Float64]()
        self.geom_pos_y = List[Float64]()
        self.geom_pos_z = List[Float64]()
        self.geom_quat_x = List[Float64]()
        self.geom_quat_y = List[Float64]()
        self.geom_quat_z = List[Float64]()
        self.geom_quat_w = List[Float64]()
        self.geom_radius = List[Float64]()
        self.geom_half_length = List[Float64]()
        self.geom_half_x = List[Float64]()
        self.geom_half_y = List[Float64]()
        self.geom_half_z = List[Float64]()
        self.geom_rgba_r = List[Float64]()
        self.geom_rgba_g = List[Float64]()
        self.geom_rgba_b = List[Float64]()
        self.geom_rgba_a = List[Float64]()
        self.geom_material_id = List[Int]()
        self.geom_hfield_id = List[Int]()
        self.geom_mesh_id = List[Int]()
        self.geom_mesh_scale = List[Float64]()
        self.geom_group = List[Int]()
        self.nmesh = 0
        self.mesh_names = List[String]()
        self.mesh_files = List[String]()
        self.light_dir_x = List[Float64]()
        self.light_dir_y = List[Float64]()
        self.light_dir_z = List[Float64]()
        self.light_diffuse_r = List[Float64]()
        self.light_diffuse_g = List[Float64]()
        self.light_diffuse_b = List[Float64]()
        self.light_specular_r = List[Float64]()
        self.light_specular_g = List[Float64]()
        self.light_specular_b = List[Float64]()
        self.light_ambient_r = List[Float64]()
        self.light_ambient_g = List[Float64]()
        self.light_ambient_b = List[Float64]()
        self.light_directional = List[Bool]()
        self.light_castshadow = List[Bool]()
        self.light_exponent = List[Float64]()
        self.cam_pos_x = List[Float64]()
        self.cam_pos_y = List[Float64]()
        self.cam_pos_z = List[Float64]()
        self.cam_quat_x = List[Float64]()
        self.cam_quat_y = List[Float64]()
        self.cam_quat_z = List[Float64]()
        self.cam_quat_w = List[Float64]()
        self.cam_fovy = List[Float64]()
        self.cam_mode = List[Int]()
        self.cam_target_body = List[Int]()
        self.cam_body = List[Int]()
        self.tex_type = List[Int]()
        self.tex_builtin = List[Int]()
        self.tex_rgb1_r = List[Float64]()
        self.tex_rgb1_g = List[Float64]()
        self.tex_rgb1_b = List[Float64]()
        self.tex_rgb2_r = List[Float64]()
        self.tex_rgb2_g = List[Float64]()
        self.tex_rgb2_b = List[Float64]()
        self.tex_names = List[String]()
        self.tex_files = List[String]()
        self.tex_mark = List[Int]()
        self.tex_markrgb_r = List[Float64]()
        self.tex_markrgb_g = List[Float64]()
        self.tex_markrgb_b = List[Float64]()
        self.tex_random = List[Float64]()
        self.mat_rgba_r = List[Float64]()
        self.mat_rgba_g = List[Float64]()
        self.mat_rgba_b = List[Float64]()
        self.mat_rgba_a = List[Float64]()
        self.mat_shininess = List[Float64]()
        self.mat_specular = List[Float64]()
        self.mat_reflectance = List[Float64]()
        self.mat_tex_id = List[Int]()
        self.mat_texrepeat_u = List[Float64]()
        self.mat_texrepeat_v = List[Float64]()
        self.site_body_id = List[Int]()
        self.site_pos_x = List[Float64]()
        self.site_pos_y = List[Float64]()
        self.site_pos_z = List[Float64]()
        self.site_size_0 = List[Float64]()
        self.nsten = 0
        self.sten_nsite = List[Int]()
        self.sten_sites = List[Int]()
        self.sten_width = List[Float64]()
        self.sten_rgba_r = List[Float64]()
        self.sten_rgba_g = List[Float64]()
        self.sten_rgba_b = List[Float64]()
        self.vis_znear = 0.0
        self.vis_fogstart = 0.0
        self.vis_fogend = 0.0
        self.vis_shadowsize = 0
        self.vis_headlight_ambient_r = 0.0
        self.vis_headlight_ambient_g = 0.0
        self.vis_headlight_ambient_b = 0.0
        self.vis_has_headlight = False


def build_render_fields(
    fmd: FlatModelDef,
    xml_text: String = String(""),
    asset_base_dir: String = String(""),
) raises -> RenderFields:
    """`FlatModelDef` → `RenderFields`. The runtime replacement for
    `parse_xml_render_data`.

    Every family here is diffed against `_rcd` by
    `tests/physics3d/test_render_data_vs_physics.mojo` while both exist —
    that gate is what licenses this function, and it found four defects in
    the sources it copies from before this was written.

    Three mappings are not straight copies and are the only places a bug can
    hide; each is commented where it happens: the mesh-asset index, the
    spatial-tendon site chain, and the texture numbering.
    """
    var rf = RenderFields()
    rf.xml_text = xml_text
    rf.asset_base_dir = asset_base_dir

    # ── geoms ─────────────────────────────────────────────────────────────
    for i in range(len(fmd.geoms)):
        var g = fmd.geoms[i]
        rf.geom_body_id.append(g.body_id)
        rf.geom_type.append(g.geom_type)
        rf.geom_pos_x.append(g.pos_x)
        rf.geom_pos_y.append(g.pos_y)
        rf.geom_pos_z.append(g.pos_z)
        rf.geom_quat_x.append(g.quat_x)
        rf.geom_quat_y.append(g.quat_y)
        rf.geom_quat_z.append(g.quat_z)
        rf.geom_quat_w.append(g.quat_w)
        # ⚠ MAPPING 4 — AN UNUSED SIZE SLOT MUST BE ZERO, NOT `GeomData`'S 0.5.
        #
        # The per-type FILL is already identical on both sides
        # (`full_parser.mojo:1703-1730` vs `xml_parser.mojo:4790-4812`, same
        # branches, same formulas, including the box's `radius` = the diagonal
        # and the sphere's `half_{x,y,z}` = its radius). What differs is the
        # value a slot keeps when its type never writes one:
        # `ComptimeRenderData` initialises to 0.0, `GeomData.__init__` to 0.5.
        # A capsule therefore carries `half_x = 0.5` in the physics record and
        # `0.0` in the render record, and a straight copy imports the 0.5.
        #
        # ⚠ AND ONE READ IS NOT TYPE-DISPATCHED. `render_ground_geoms` scans
        # `geom_radius` over EVERY geom to size the arena
        # (`max_body_radius`, `model_def_from_xml.mojo:2523`), so a mesh or
        # plane contributing 0.5 instead of 0.0 silently resizes the ground.
        # Every other size read sits inside a `gt == ...` branch.
        #
        # So this copies only the slots the type actually uses and leaves the
        # rest at 0.0. It deliberately does NOT recompute the box diagonal:
        # `GeomData.radius` already holds it, and a second copy of a formula
        # is how the `xyaxes` conjugate survived in one parser after being
        # fixed in the other.
        var r_rad = Float64(0.0)
        var r_hl = Float64(0.0)
        var r_hx = Float64(0.0)
        var r_hy = Float64(0.0)
        var r_hz = Float64(0.0)
        if g.geom_type == _RF_SPHERE:
            r_rad = g.radius
            r_hx = g.half_x
            r_hy = g.half_y
            r_hz = g.half_z
        elif g.geom_type == _RF_CAPSULE or g.geom_type == _RF_CYLINDER:
            r_rad = g.radius
            r_hl = g.half_length
        elif g.geom_type == _RF_BOX or g.geom_type == _RF_ELLIPSOID:
            r_rad = g.radius
            r_hx = g.half_x
            r_hy = g.half_y
            r_hz = g.half_z
        elif g.geom_type == _RF_PLANE:
            r_hx = g.half_x
            r_hy = g.half_y
        else:
            # MESH and anything else: `_rcd`'s trailing `else` sets only
            # `geom_radius`, from `size[0]` when there is one.
            r_rad = g.radius
        rf.geom_radius.append(r_rad)
        rf.geom_half_length.append(r_hl)
        rf.geom_half_x.append(r_hx)
        rf.geom_half_y.append(r_hy)
        rf.geom_half_z.append(r_hz)
        rf.geom_rgba_r.append(g.rgba_r)
        rf.geom_rgba_g.append(g.rgba_g)
        rf.geom_rgba_b.append(g.rgba_b)
        rf.geom_rgba_a.append(g.rgba_a)
        rf.geom_material_id.append(g.material_id)
        rf.geom_group.append(g.group)
        # ⚠ MAPPING 1 — MESH IDENTITY IS BY FILE, NOT BY INDEX. `_rcd`'s
        # `geom_mesh_id` indexes its own asset table; `GeomData.mesh_id`
        # indexes LOADED HULL data, a different space that skips every
        # visual-only mesh (SO-ARM100 loads 8 of 18). Copying it across would
        # be copying an unrelated integer. The asset table is
        # `mesh_asset_files`, whose order the render gate proves identical to
        # `_rcd.mesh_files`, so the filename is the shared key.
        var mid = -1
        if g.mesh_filename.byte_length() > 0:
            for k in range(fmd.num_mesh_assets):
                if fmd.mesh_asset_files[k] == g.mesh_filename:
                    mid = k
                    break
        # ⚠ MAPPING 2 — HFIELD IDENTITY IS BY INDEX, UNLIKE THE MESH ABOVE.
        # `GeomData.hfield_id` already indexes the `<asset><hfield>` table by
        # the order the parser read them, which is the same table the renderer
        # reads its sizes from; there is no second space to translate between.
        rf.geom_hfield_id.append(g.hfield_id)
        rf.geom_mesh_id.append(mid)
        rf.geom_mesh_scale.append(g.mesh_scale_x)
        rf.geom_mesh_scale.append(g.mesh_scale_y)
        rf.geom_mesh_scale.append(g.mesh_scale_z)

    # ── mesh assets ───────────────────────────────────────────────────────
    rf.nmesh = fmd.num_mesh_assets
    for i in range(fmd.num_mesh_assets):
        rf.mesh_names.append(fmd.mesh_asset_names[i])
        rf.mesh_files.append(fmd.mesh_asset_files[i])

    # ── lights ────────────────────────────────────────────────────────────
    for i in range(len(fmd.lights)):
        var l = fmd.lights[i]
        rf.light_dir_x.append(l.dir_x)
        rf.light_dir_y.append(l.dir_y)
        rf.light_dir_z.append(l.dir_z)
        rf.light_diffuse_r.append(l.diffuse_r)
        rf.light_diffuse_g.append(l.diffuse_g)
        rf.light_diffuse_b.append(l.diffuse_b)
        rf.light_specular_r.append(l.specular_r)
        rf.light_specular_g.append(l.specular_g)
        rf.light_specular_b.append(l.specular_b)
        rf.light_ambient_r.append(l.ambient_r)
        rf.light_ambient_g.append(l.ambient_g)
        rf.light_ambient_b.append(l.ambient_b)
        rf.light_directional.append(l.directional)
        rf.light_castshadow.append(l.castshadow)
        rf.light_exponent.append(l.exponent)

    # ── cameras ───────────────────────────────────────────────────────────
    for i in range(len(fmd.cameras)):
        var c = fmd.cameras[i]
        rf.cam_pos_x.append(c.pos_x)
        rf.cam_pos_y.append(c.pos_y)
        rf.cam_pos_z.append(c.pos_z)
        rf.cam_quat_x.append(c.quat_x)
        rf.cam_quat_y.append(c.quat_y)
        rf.cam_quat_z.append(c.quat_z)
        rf.cam_quat_w.append(c.quat_w)
        rf.cam_fovy.append(c.fovy)
        rf.cam_mode.append(c.mode)
        rf.cam_target_body.append(c.target_body)
        rf.cam_body.append(c.body_id)

    # ── textures ──────────────────────────────────────────────────────────
    rf.ntex = len(fmd.textures)
    for i in range(len(fmd.textures)):
        var t = fmd.textures[i]
        # ⚠ MAPPING 3 — see the module docstring. This is `flat_model`'s
        # numbering, deliberately NOT `_rcd`'s.
        rf.tex_type.append(t.tex_type)
        rf.tex_builtin.append(t.builtin)
        rf.tex_mark.append(t.mark)
        rf.tex_rgb1_r.append(t.rgb1_r)
        rf.tex_rgb1_g.append(t.rgb1_g)
        rf.tex_rgb1_b.append(t.rgb1_b)
        rf.tex_rgb2_r.append(t.rgb2_r)
        rf.tex_rgb2_g.append(t.rgb2_g)
        rf.tex_rgb2_b.append(t.rgb2_b)
        rf.tex_markrgb_r.append(t.markrgb_r)
        rf.tex_markrgb_g.append(t.markrgb_g)
        rf.tex_markrgb_b.append(t.markrgb_b)
        rf.tex_random.append(t.random)
        rf.tex_names.append(t.name)
        rf.tex_files.append(t.file)

    # ── materials ─────────────────────────────────────────────────────────
    for i in range(len(fmd.materials)):
        var m = fmd.materials[i]
        rf.mat_rgba_r.append(m.rgba_r)
        rf.mat_rgba_g.append(m.rgba_g)
        rf.mat_rgba_b.append(m.rgba_b)
        rf.mat_rgba_a.append(m.rgba_a)
        rf.mat_shininess.append(m.shininess)
        rf.mat_specular.append(m.specular)
        rf.mat_reflectance.append(m.reflectance)
        rf.mat_tex_id.append(m.tex_id)
        rf.mat_texrepeat_u.append(m.texrepeat_u)
        rf.mat_texrepeat_v.append(m.texrepeat_v)

    # ── sites ─────────────────────────────────────────────────────────────
    for i in range(len(fmd.sites)):
        var s = fmd.sites[i]
        rf.site_body_id.append(s.body_id)
        rf.site_pos_x.append(s.pos_x)
        rf.site_pos_y.append(s.pos_y)
        rf.site_pos_z.append(s.pos_z)
        rf.site_size_0.append(s.size_0)

    # ── spatial tendons, for DRAWING only ─────────────────────────────────
    # ⚠ MAPPING 2 — `_rcd` stores the site chain FLAT (`sten_sites` is one
    # array across all tendons, walked with a cursor and `sten_nsite[t]`),
    # while `TendonData` holds each tendon's own `site_ids`. The hooks read
    # the flat form, so it is rebuilt here rather than changed there.
    var nsten = 0
    for ti in range(len(fmd.tendons)):
        var td = fmd.tendons[ti]
        if td.kind != 1:  # spatial only
            continue
        # ⚠ SITES ONLY, AND THE WRAP GEOMS ARE DROPPED ON PURPOSE. The
        # renderer draws a chain of capsules between consecutive points; it
        # has never drawn the ARC round a wrap geom (MuJoCo does). Feeding it
        # geom ids here would index `site_xpos` with a geom id — a chain
        # through arbitrary points. Straight chords through the sites are
        # wrong by the arc's bulge and right everywhere else, which for
        # softfoot's 3.5 mm pulleys is invisible. See `render_spatial_tendons`.
        var nsite_k = 0
        for k in range(td.num_wraps):
            if td.wrap_types[k] == WRAP_SITE:
                nsite_k += 1
        rf.sten_nsite.append(nsite_k)
        for k in range(td.num_wraps):
            if td.wrap_types[k] == WRAP_SITE:
                rf.sten_sites.append(td.wrap_objs[k])
        rf.sten_width.append(td.render_width)
        rf.sten_rgba_r.append(td.rgba_r)
        rf.sten_rgba_g.append(td.rgba_g)
        rf.sten_rgba_b.append(td.rgba_b)
        nsten += 1
    rf.nsten = nsten

    # ── <visual> ──────────────────────────────────────────────────────────
    rf.vis_znear = fmd.vis_znear
    rf.vis_fogstart = fmd.vis_fogstart
    rf.vis_fogend = fmd.vis_fogend
    rf.vis_shadowsize = fmd.vis_shadowsize
    rf.vis_headlight_ambient_r = fmd.vis_headlight_ambient_r
    rf.vis_headlight_ambient_g = fmd.vis_headlight_ambient_g
    rf.vis_headlight_ambient_b = fmd.vis_headlight_ambient_b
    rf.vis_has_headlight = fmd.vis_has_headlight

    return rf^
