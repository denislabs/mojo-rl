"""Hill-climb walk lengths on the k=0 park scene (CPU, Euler): how many
neighbourhood scans each mesh support call walks, cold vs warm, and the same
walk replayed from (a) MuJoCo 3.12's grid-extrema seed (`mesh_extrema`,
commit 83e621d7) and (b) the vertex the same call landed on in the PREVIOUS
step. PERFORMANCE.md §13.48 has the numbers this printed on 2026-09-08.

    # set `_HILL_PROBE = True` in mojo_rl/physics3d/collision/gjk.mojo, then
    pixi run mojo build -I . -I benchmarks benchmarks/physics3d_cpu/hill_probe.mojo -o <bin>
    <bin>

⚠ THE PROBE IS A COUNTER, NOT A TIMER. A scan is one dependent chain of
global loads on the GPU (edge address, neighbour ids, vertex coordinates),
so scans per step is the quantity a seed can change; what a scan costs is
the box's to say (§13.18's block-kernel bisect: four GJK candidates, 206 of
the 270 µs at k=0).
"""

from mojo_rl.envs.robots.so101_park_config import So101ParkProbeConfig
from mojo_rl.envs.robots.so101_park_xml import SoArm101ParkK0Model
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.physics3d.collision.gjk import hill_probe
from mojo_rl.physics3d.gpu.constants import (
    MODEL_MESH_META_SIZE,
    MESH_META_IDX_VERTADR,
    MESH_META_IDX_VERTNUM,
    MAX_GPU_MESHES,
    META_IDX_NUM_CONTACTS,
)
from physics3d_cpu.harness import _one_step, CTRL

comptime DT = DType.float32
comptime MODEL = SoArm101ParkK0Model
comptime CONFIG = So101ParkProbeConfig[6, 6, 0]


def _f(a: Int, b: Int) -> Float64:
    return Float64(a) / Float64(b if b > 0 else 1)


def report(tag: String, nsteps: Int, ncon: Float64):
    var hp = hill_probe()
    var warm_calls = hp[].calls - hp[].cold_calls
    var warm_scans = hp[].scans - hp[].cold_scans
    var s_warm = hp[].seeded_scans - hp[].seeded_cold_scans
    print("==", tag, " steps", nsteps, " ncon_mean", ncon / Float64(nsteps))
    print("  hill-climb calls/step", _f(hp[].calls, nsteps),
          " linear-scan calls/step", _f(hp[].linear, nsteps))
    print("  scans/call  all", _f(hp[].scans, hp[].calls),
          " cold", _f(hp[].cold_scans, hp[].cold_calls),
          " warm", _f(warm_scans, warm_calls),
          "   (cold calls", hp[].cold_calls, "of", hp[].calls, ")")
    print("  3.12 seed:  scans/call  all", _f(hp[].seeded_scans, hp[].calls),
          " cold", _f(hp[].seeded_cold_scans, hp[].cold_calls),
          " warm", _f(s_warm, warm_calls),
          "   seed beat warm", hp[].seed_beat_warm,
          " landing mismatches", hp[].seed_mismatch,
          " of which NOT ties", hp[].seed_mismatch_nontie)
    print("  cross-step warm: scans/call", _f(hp[].xstep_scans, hp[].xstep_calls),
          " (calls", hp[].xstep_calls, ")")
    print("  scans/step  now", _f(hp[].scans, nsteps),
          " seeded", _f(hp[].seeded_scans, nsteps))


def main() raises:
    var env = Phyics3dEnv[MODEL, CONFIG, DT, False]()
    _ = env.reset()
    for i in range(MODEL.NQ):
        env.d.qpos.data[i] = env.sf.qpos0.data[i]
    for i in range(MODEL.NV):
        env.d.qvel.data[i] = Scalar[DT](0)
        env.d.qacc_warmstart.data[i] = Scalar[DT](0)
    var actions = List[Float64]()
    for _ in range(MODEL.ACTION_DIM):
        actions.append(CTRL)

    # 3.12's mesh_extrema, from the hull vertices the model holds.
    var hp = hill_probe()
    var nmesh = 0
    for r in range(MAX_GPU_MESHES):
        var o = r * MODEL_MESH_META_SIZE
        var va = Int(env.mf.mesh_meta.data[o + MESH_META_IDX_VERTADR])
        var vn = Int(env.mf.mesh_meta.data[o + MESH_META_IDX_VERTNUM])
        if vn <= 0:
            continue
        nmesh += 1
        hp[].ext_keys.append(va)
        for k in range(27):
            var cx = Float64(k // 9 - 1)
            var cy = Float64((k // 3) % 3 - 1)
            var cz = Float64(k % 3 - 1)
            var best = -1e300
            var arg = 0
            for i in range(vn):
                var x = Float64(env.mf.mesh_verts.data[(va + i) * 3 + 0])
                var y = Float64(env.mf.mesh_verts.data[(va + i) * 3 + 1])
                var z = Float64(env.mf.mesh_verts.data[(va + i) * 3 + 2])
                var d = x * cx + y * cy + z * cz
                if d > best:
                    best = d
                    arg = i
            hp[].ext.append(arg)
        print("mesh", nmesh - 1, "vert_adr", va, "hull verts", vn)
    print("meshes with extrema:", nmesh)

    # Phase A: the props in flight (steps 0-500).
    hp[].reset_counts()
    var ncon = 0.0
    for _ in range(500):
        hp[].ordinal = 0
        _one_step[MODEL, CONFIG, DT, True](env, actions)
        ncon += Float64(env.d.meta.data[META_IDX_NUM_CONTACTS])
    report("A flight, steps 0-500", 500, ncon)

    # Phase B: landed (the first prop lands at step 1596): steps 2000-2500.
    for _ in range(1500):
        hp[].ordinal = 0
        _one_step[MODEL, CONFIG, DT, True](env, actions)
    hp[].reset_counts()
    ncon = 0.0
    for _ in range(500):
        hp[].ordinal = 0
        _one_step[MODEL, CONFIG, DT, True](env, actions)
        ncon += Float64(env.d.meta.data[META_IDX_NUM_CONTACTS])
    report("B landed, steps 2000-2500", 500, ncon)
