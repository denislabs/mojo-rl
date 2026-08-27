"""Bake a manipulation task and emit its MJCF as a Mojo comptime String module.

Mirrors what produced `manipulation_reach_xml.mojo`: export with assets, then
rewrite `file="<hash>.stl"` to point at the committed copies under
`assets/jaco/`. Nothing else is edited.
"""
import os, re, sys, tempfile, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'tests/dm_control')
import manipulation_ref as ref

ASSET_DIR = 'mojo_rl/envs/dm_control/assets/jaco/'


def emit(task, var_name, out_path, title):
    tmp = tempfile.mkdtemp(prefix='bake_')
    xml_path = ref.bake(task, tmp)
    xml = open(xml_path).read()
    # The only edit: point mesh files at the committed assets.
    xml = re.sub(r'file="([^"/]+\.stl)"', lambda m: 'file="%s%s"' % (ASSET_DIR, m.group(1)), xml)
    missing = [f for f in set(re.findall(r'file="([^"]+)"', xml))
               if not os.path.exists(f)]
    if missing:
        raise SystemExit('asset(s) not committed: %s' % missing)
    if '"""' in xml:
        raise SystemExit('XML contains a triple quote; the Mojo literal would break')
    body = '"""%s\n\n%s"""\n\ncomptime %s = """\n%s"""\n' % (title, _HEADER, var_name, xml)
    open(out_path, 'w').write(body)
    print('%-46s -> %-60s %8d bytes' % (task, out_path, len(body)))


_HEADER = '''⚠⚠ THIS FILE IS GENERATED, NOT WRITTEN. dm_control builds this task with
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
'''

if __name__ == '__main__':
    task, var, out, title = sys.argv[1:5]
    emit(task, var, out, title)
