"""`base_dir` — what a relative asset path inside an MJCF file resolves against.

⚠ THIS IS THE SEAM FOR THE ASSET-ROOT DECISION (§10.5 open decision 1).
MuJoCo resolves `meshdir` and a bare `file=` against THE DIRECTORY OF THE
MODEL FILE. `parse_xml_full` is handed a string, so it historically had no
such directory and fell back to the process CWD — which is why every ported
model carries repo-root-relative asset paths, why four tests `os.chdir` into
the repo root to load one, and why `mujoco.MjModel.from_xml_path()` cannot
read our own assets at all. Measured, not assumed:

    from_xml_path('mojo_rl/envs/robots/assets/so_arm100.xml')
      -> Error opening file 'mojo_rl/envs/robots/assets/so_arm100/Base.stl'
    from_xml_string(open(same).read())
      -> loads

`base_dir` supplies the missing directory. It is INERT until a caller passes
one: `base_dir=""` is exactly today's CWD behaviour, which is what lets this
land before the models switch over.

WHAT IS PINNED HERE — the four rules, each with a case that would pass under
a wrong implementation of the others:

  1. base_dir + bare file=            (the common case)
  2. base_dir + meshdir + file=       (so_arm100's shape: meshdir is itself
                                       relative to the model directory)
  3. absolute paths ESCAPE at BOTH levels — an absolute `meshdir` and an
     absolute `file=` are used as-is, as MuJoCo does
  4. TEXTURES are prefixed too, and get NO meshdir — MuJoCo would consult
     `texturedir`/`assetdir`, neither of which is handled, so a texture
     resolves against the model directory alone

  ⚠ Rule 3 is the one a naive "just prepend" gets wrong, and rule 4 is the
  one that is easy to forget entirely: textures took no directory treatment
  at all before this, so a texture path is the only asset that changes
  behaviour class rather than just base.

⚠ NON-VACUITY: every assertion below is paired with the SAME model parsed at
`base_dir=""`, and the run fails if the two agree — a base_dir that silently
did nothing would otherwise pass every "starts with the right prefix" check
that a correct one passes.

Run: pixi run mojo run -I . tests/physics3d/test_asset_base_dir.mojo
"""

from std.testing import assert_true

from mojo_rl.physics3d.parser import parse_xml_full


# Rules 1 + 4: bare mesh file=, bare texture file=, no meshdir.
comptime _PLAIN = """<mujoco model="plain">
  <asset>
    <mesh name="m0" file="meshes/thing.stl"/>
    <texture name="t0" type="2d" file="textures/wood.png"/>
  </asset>
  <worldbody>
    <body name="b"><joint type="hinge"/><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>"""

# Rule 2: meshdir is itself relative to the model directory (so_arm100's shape).
comptime _MESHDIR = """<mujoco model="with_meshdir">
  <compiler meshdir="parts/"/>
  <asset><mesh name="m0" file="Base.stl"/></asset>
  <worldbody>
    <body name="b"><joint type="hinge"/><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>"""

# Rule 3: absolute at both levels.
comptime _ABSOLUTE = """<mujoco model="absolute">
  <compiler meshdir="/opt/shared/"/>
  <asset>
    <mesh name="m0" file="Base.stl"/>
    <texture name="t0" type="2d" file="/opt/tex/wood.png"/>
  </asset>
  <worldbody>
    <body name="b"><joint type="hinge"/><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>"""


def main() raises:
    var fails = 0
    var checks = 0
    var differed = 0

    # ── rule 1 + 4 ───────────────────────────────────────────────────────
    var plain = parse_xml_full(String(_PLAIN), "envs/demo")
    var plain0 = parse_xml_full(String(_PLAIN), "")

    checks += 1
    if plain.mesh_asset_files[0] != "envs/demo/meshes/thing.stl":
        fails += 1
        print("  FAIL mesh:", plain.mesh_asset_files[0])
    checks += 1
    if plain.textures[0].file != "envs/demo/textures/wood.png":
        fails += 1
        print("  FAIL texture:", plain.textures[0].file)

    # ⚠ the base_dir="" control — if these AGREE, base_dir did nothing.
    if plain.mesh_asset_files[0] != plain0.mesh_asset_files[0]:
        differed += 1
    if plain.textures[0].file != plain0.textures[0].file:
        differed += 1
    print("  base_dir=\"\" control -> mesh:", plain0.mesh_asset_files[0],
          " texture:", plain0.textures[0].file)

    # ── rule 2: meshdir relative to the model directory ──────────────────
    var md = parse_xml_full(String(_MESHDIR), "envs/robots")
    var md0 = parse_xml_full(String(_MESHDIR), "")
    checks += 1
    if md.mesh_asset_files[0] != "envs/robots/parts/Base.stl":
        fails += 1
        print("  FAIL meshdir:", md.mesh_asset_files[0])
    if md.mesh_asset_files[0] != md0.mesh_asset_files[0]:
        differed += 1
    print("  meshdir control ->", md0.mesh_asset_files[0])

    # ── rule 3: absolute escapes at BOTH levels ──────────────────────────
    var ab = parse_xml_full(String(_ABSOLUTE), "envs/anything")
    checks += 1
    if ab.mesh_asset_files[0] != "/opt/shared/Base.stl":
        fails += 1
        print("  FAIL absolute meshdir:", ab.mesh_asset_files[0])
    checks += 1
    if ab.textures[0].file != "/opt/tex/wood.png":
        fails += 1
        print("  FAIL absolute texture:", ab.textures[0].file)

    # A trailing-slash-free base_dir must join the same way.
    var noslash = parse_xml_full(String(_PLAIN), "envs/demo/")
    checks += 1
    if noslash.mesh_asset_files[0] != plain.mesh_asset_files[0]:
        fails += 1
        print(
            "  FAIL trailing slash:", noslash.mesh_asset_files[0],
            "!=", plain.mesh_asset_files[0],
        )

    print()
    print("checks:", checks, " failures:", fails)
    print("values base_dir actually MOVED:", differed, "of 3")

    assert_true(fails == 0, String(fails) + " base_dir rule(s) wrong")
    assert_true(
        differed == 3,
        "base_dir changed only " + String(differed) + " of 3 paths — a"
        " base_dir that does nothing passes every prefix check a correct one"
        " passes, so this control is the gate",
    )
    print()
    print("PASS")
