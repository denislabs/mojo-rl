"""THE DEVICE SAMPLER AGAINST THE HOST ONE — one distribution, two writers.

    pixi run mojo run -I . tests/tasks/test_device_placement.mojo

## ⚠⚠ WHY THERE ARE TWO AT ALL, AND WHY THAT IS THE RISK

`tasks/sampler.sample_placements` places props on the HOST — it is what the
eval, the viewer and the init table use. `So101TabletopConfig.init_qpos_gpu`
places them on the DEVICE, per lane, at every reset, because a reset in a
training run happens between kernel launches and there is no host in the loop.
`sampler.mojo`'s header says it was SHAPED for that: pure geometry,
counter-based Philox, no `Data` and no `Model`.

Two implementations of one distribution is exactly the drift this file exists
to prevent. If they disagree, a policy trains on scenes the eval never shows
it and every other number in the system still agrees.

## WHAT AGREEMENT REQUIRES — five things, and each has broken something

1. the same PHILOX COORDINATES — `seed ^ PLACEMENT_SALT`, subsequence
   `(lane << 16) | axis`, offset `attempt`;
2. the same AXIS NUMBERING — `si * 2` and `si * 2 + 1` where `si` is the
   FAMILY SLOT INDEX, not the free-slot ordinal (they differ from slot 1 on,
   because slot 0 of this family is a static fixture);
3. the same WALK ORDER, because rejection is order-dependent —
   `validate_task_against_family` forces `init=` into family slot order so the
   host's task-order walk and the device's slot-table walk coincide;
4. the same REJECTION RADIUS and the same attempt budget;
5. the same RESTING HEIGHT — `site_z + radius`, not `site_z`.

## ⚠ AND THE RESTATED CONSTANTS, CHECKED AGAINST THE `.family`

`init_qpos_gpu` cannot read `curriculum` or `site_xpos` — it runs before
forward kinematics — so the region table is restated as comptime constants on
the config. Every one of them is asserted here against the loaded `.family`
plus FK on the composed scene, the same way `test_active_mask` asserts the
address table and the park poses.
"""

from mojo_rl.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.family_config import So101TabletopConfig
from mojo_rl.tasks.active import active_mask, init_region_words
from mojo_rl.tasks.eval import region_sites, region_rects
from mojo_rl.tasks.sampler import (
    sample_placements, RegionFrame, SampleReport, MAX_PLACE_ATTEMPTS,
    PLACEMENT_SALT,
)
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_INIT_REGION_0, INIT_REGION_NONE,
    MODEL_JOINT_SIZE, MODEL_BODY_SIZE, MODEL_GEOM_SIZE,
)
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics

from layout import Layout, LayoutTensor
from mojo_rl.nn.core.tensor import TensorImpl


comptime DT = DType.float64
comptime FAMILY = "mojo_rl/tasks/families/so101_tabletop.family"
comptime CFG = So101TabletopConfig
comptime NF = CFG.N_FREE_SLOTS
comptime BATCH = 8
comptime NQ = 27
comptime NV = 24


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def main() raises:
    print("=== device placement vs the host sampler ===")
    var ta = Tally()

    var f = load_family(FAMILY)
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)

    # ── 1. the restated region table matches the .family + FK ─────────────
    var dims = dims_from_flat(
        fmd, max_contacts=32, nmesh_verts=SO_ARM101_NMESH_VERTS
    )
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    forward_kinematics["cpu", DT, DynDims, 1](d, m)

    print("--- 1. the config's restated region table ---")
    ta.check(CFG.N_REGIONS == len(f.regions),
             "N_REGIONS == the family's region count")
    var ok_site = True
    for i in range(len(f.regions)):
        var s = rsites[i]
        var sx = Float64(d.site_xpos.data[s * 3])
        var sy = Float64(d.site_xpos.data[s * 3 + 1])
        var sz = Float64(d.site_xpos.data[s * 3 + 2])
        # ⚠ ONE TRIPLE FOR ALL THREE, which is what the config assumes. If a
        # family ever puts a region on a second fixture this fails HERE, at
        # the assumption, rather than as a misplaced prop.
        if (sx - CFG.REGION_SITE_X) ** 2 + (sy - CFG.REGION_SITE_Y) ** 2 \
                + (sz - CFG.REGION_SITE_Z) ** 2 > 1e-18:
            ok_site = False
            print("      region", i, "site at (", sx, ",", sy, ",", sz,
                  ") but the config restates (", CFG.REGION_SITE_X, ",",
                  CFG.REGION_SITE_Y, ",", CFG.REGION_SITE_Z, ")")
    ta.check(ok_site,
             "every region's site is where the config restates it")

    var cfg_x0 = List[Float64]()
    var cfg_y0 = List[Float64]()
    var cfg_x1 = List[Float64]()
    var cfg_y1 = List[Float64]()
    cfg_x0.append(CFG.REGION_X0_0); cfg_y0.append(CFG.REGION_Y0_0)
    cfg_x1.append(CFG.REGION_X1_0); cfg_y1.append(CFG.REGION_Y1_0)
    cfg_x0.append(CFG.REGION_X0_1); cfg_y0.append(CFG.REGION_Y0_1)
    cfg_x1.append(CFG.REGION_X1_1); cfg_y1.append(CFG.REGION_Y1_1)
    cfg_x0.append(CFG.REGION_X0_2); cfg_y0.append(CFG.REGION_Y0_2)
    cfg_x1.append(CFG.REGION_X1_2); cfg_y1.append(CFG.REGION_Y1_2)
    var ok_rect = True
    for i in range(len(f.regions)):
        if cfg_x0[i] != rects[i][0] or cfg_y0[i] != rects[i][1] \
                or cfg_x1[i] != rects[i][2] or cfg_y1[i] != rects[i][3]:
            ok_rect = False
            print("      region", i, f.regions[i].name, "rect drift")
    ta.check(ok_rect, "every region's rectangle matches the .family")

    ta.check(CFG.MAX_PLACE_ATTEMPTS == MAX_PLACE_ATTEMPTS,
             "the attempt budget matches sampler.MAX_PLACE_ATTEMPTS")
    ta.check(CFG.PLACEMENT_SALT == PLACEMENT_SALT,
             "the placement salt matches sampler.PLACEMENT_SALT")

    # ⚠ THE RADIUS IS READ FROM THE PROP'S OWN ASSET, not from the composed
    # scene. `sampler` uses it for BOTH the pairwise clash test and the
    # RESTING HEIGHT (`z = site_z + radius`), so a value that is not the
    # box's half-size starts every prop either floating or inside the table —
    # and the solver ejects the second kind on step 1.
    var cube = parse_model_runtime("mojo_rl/tasks/assets/props/cube.xml")
    var ok_rad = False
    for i in range(len(cube.geoms)):
        if cube.geoms[i].half_x == CFG.SLOT_RADIUS:
            ok_rad = True
    ta.check(ok_rad,
             "SLOT_RADIUS is the prop asset's own half-size (resting height)")

    # ── 2. the two samplers, on the same (seed, lane) ─────────────────────
    print()
    print("--- 2. device vs host, four tasks x", BATCH, "lanes ---")
    var frames = List[RegionFrame]()
    for i in range(len(f.regions)):
        var s = rsites[i]
        frames.append(RegionFrame(
            Float64(d.site_xpos.data[s * 3]),
            Float64(d.site_xpos.data[s * 3 + 1]),
            Float64(d.site_xpos.data[s * 3 + 2]),
        ))
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(CFG.SLOT_RADIUS)
    var names = List[String]()
    names.append(String("so101_reach_brick"))
    names.append(String("so101_lift_brick"))
    names.append(String("so101_gather_bricks"))
    names.append(String("so101_settle_brick"))

    comptime L_Q = Layout.row_major(BATCH, NQ)
    comptime L_V = Layout.row_major(BATCH, NV)
    comptime L_M = Layout.row_major(BATCH, METADATA_SIZE)
    # ⚠ THE WIDTHS ARE THE MODEL RECORD SIZES, not round numbers — the hook's
    # signature ties each to a `MODEL_*_SIZE` and a mismatch is a compile
    # error naming the expected layout, which is how these were found.
    comptime L_J = Layout.row_major(1, MODEL_JOINT_SIZE)
    comptime L_M3 = Layout.row_major(BATCH, 3)
    comptime L_M4 = Layout.row_major(BATCH, 4)
    comptime L_B = Layout.row_major(1, MODEL_BODY_SIZE)
    comptime L_G = Layout.row_major(1, MODEL_GEOM_SIZE)
    var dj = TensorImpl[DT].alloc(MODEL_JOINT_SIZE)
    var dm3 = TensorImpl[DT].alloc(BATCH * 3)
    var dm4 = TensorImpl[DT].alloc(BATCH * 4)
    var db = TensorImpl[DT].alloc(MODEL_BODY_SIZE)
    var dg = TensorImpl[DT].alloc(MODEL_GEOM_SIZE)

    comptime SEED = 7
    var total_cmp = 0
    var total_bad = 0
    var total_placed = 0

    for n in range(len(names)):
        var t = load_task("mojo_rl/tasks/tasks/" + names[n] + ".task")
        validate_task_against_family(t, f)
        var iw = init_region_words(t, f)

        # the device hook writes into a [BATCH, NQ] / [BATCH, NV] pair and
        # reads [BATCH, METADATA_SIZE]; build them as plain host tensors, the
        # way `test_active_mask` drives the observation hooks.
        var qs = TensorImpl[DT].alloc(BATCH * NQ)
        var vs = TensorImpl[DT].alloc(BATCH * NV)
        var ms = TensorImpl[DT].alloc(BATCH * METADATA_SIZE)
        for i in range(BATCH * NV):
            vs.data[i] = Scalar[DT](0)
        for i in range(BATCH * METADATA_SIZE):
            ms.data[i] = Scalar[DT](0)
        for e in range(BATCH):
            for j in range(NF):
                ms.data[
                    e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j
                ] = Scalar[DT](iw[j])
        var qt = qs.lt["cpu", L_Q]()
        var vt = vs.lt["cpu", L_V]()
        var mt = ms.lt["cpu", L_M]()

        var n_task_placed = 0
        for lane in range(BATCH):
            # ⚠ THE PARK POSE FIRST, exactly as `_reset_env_lane` leaves it
            # (`qpos0` = the composed scene's frame positions). A slot the
            # device declines to place must therefore still read as PARKED,
            # and starting from zeros would make "declined" indistinguishable
            # from "placed at the origin".
            for j in range(NF):
                var qa = (
                    CFG.FREE_QADR_0 if j == 0
                    else (CFG.FREE_QADR_1 if j == 1 else CFG.FREE_QADR_2)
                )
                var si = (
                    CFG.FREE_SLOT_IDX_0 if j == 0
                    else (
                        CFG.FREE_SLOT_IDX_1 if j == 1 else CFG.FREE_SLOT_IDX_2
                    )
                )
                qt[lane, qa + 0] = Scalar[DT](
                    CFG.PARK_X + Float64(si) * CFG.PARK_SPACING
                )
                qt[lane, qa + 1] = Scalar[DT](CFG.PARK_Y)
                qt[lane, qa + 2] = Scalar[DT](CFG.PARK_Z)

            # ⚠ THE FIVE UNUSED TENSORS ARE SIZE-1 SCRATCH. `init_qpos_gpu`
            # ends with `_ = joints` / `_ = bodies` / ...; passing real ones
            # would make this file build a model it never reads.
            CFG.init_qpos_gpu[DT, BATCH, NQ, 1, NV, 1, 1](
                qt, vt,
                dj.lt["cpu", L_J](), dm3.lt["cpu", L_M3](),
                dm4.lt["cpu", L_M4](),
                db.lt["cpu", L_B](), dg.lt["cpu", L_G](),
                mt, lane, SEED,
            )

            # the host's answer for the same (seed, lane)
            var rep = SampleReport()
            var placed = sample_placements(
                t, f, frames, radii, UInt64(SEED), lane, rep
            )
            n_task_placed += len(placed)
            total_placed += len(placed)
            for k in range(len(placed)):
                # family slot index -> free-slot ordinal -> qposadr
                var si2 = placed[k].slot
                var ord = -1
                var seen = 0
                for q in range(len(f.slots)):
                    if f.slots[q].kind == SLOT_FREE:
                        if q == si2:
                            ord = seen
                        seen += 1
                var qa2 = (
                    CFG.FREE_QADR_0 if ord == 0
                    else (CFG.FREE_QADR_1 if ord == 1 else CFG.FREE_QADR_2)
                )
                total_cmp += 3
                # ⚠ `rebind` FIRST. A LayoutTensor element is a SIMD whose
                # width the compiler cannot narrow at the call site, and
                # `Float64(...)` of it fails with "constructing from an
                # `Intable` value requires an integral dtype" — an error that
                # names nothing about layouts.
                var gx = Float64(rebind[Scalar[DT]](qt[lane, qa2 + 0]))
                var gy = Float64(rebind[Scalar[DT]](qt[lane, qa2 + 1]))
                var gz = Float64(rebind[Scalar[DT]](qt[lane, qa2 + 2]))
                var dx = gx - placed[k].x
                var dy = gy - placed[k].y
                var dz = gz - placed[k].z
                if dx * dx + dy * dy + dz * dz > 1e-24:
                    total_bad += 3
                    if total_bad <= 9:
                        print("      ", names[n], "lane", lane, "slot", si2,
                              ": device (", gx, ",", gy, ",", gz,
                              ") host (", placed[k].x, ",", placed[k].y, ",",
                              placed[k].z, ")")
        print("     ", names[n], ":", n_task_placed, "placements over",
              BATCH, "lanes")

    print()
    print("  coordinates compared:", total_cmp, " differing:", total_bad)
    # ⚠⚠ ANTI-VACUITY, AND IT IS NOT DECORATION. "0 differing" is also what a
    # loop that compared nothing reports — and three of the four tasks place
    # two props while `so101_reach_clear` places none, so a bug that skipped
    # every task would print exactly this line with a zero beside it.
    if total_placed == 0:
        raise Error(
            "device placement: the host sampler placed NOTHING across "
            + String(len(names)) + " tasks, so the comparison above compared"
            " nothing and would report 0 differences whatever the device did."
        )
    ta.check(total_bad == 0,
             "the device and host samplers agree on EVERY coordinate")
    ta.check(total_cmp >= 3 * 2 * BATCH,
             "the comparison covered at least two props per lane somewhere")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "device placement: " + String(ta.failures) + " of "
            + String(ta.checks) + " check(s) failed"
        )
    print("=== PASS ===")
