"""Every LIBERO family's device placement table — GENERATED artifacts.

    pixi run gen-placement-tables           # write mojo_rl/tasks/placement/*.mojo
    pixi run gen-placement-tables --check   # CI: fail if one is stale

`placement/table.place_free_slots` is the device twin of
`sampler.sample_placements`, and it reads a family's geometry through a
comptime `PlacementTable` because a kernel cannot read a `.family`. This writes
one per `libero*.family`, from the SAME inputs the host sampler uses: the loaded
spec, `reset.free_slot_addresses`, and the region sites' world frames from
forward kinematics on the composed scene.

⚠ EVERY FLOAT IS `Scalar[DTYPE](<String(Float64)>)`. That decimal is measured
to round-trip to the same bits (see `_f`), so the table matches the frames it
was generated from exactly and `check.placement_table_drift` can demand it; and
it is a `DTYPE` constant in the kernel, because Metal has no `double`.

⚠ A FREE SLOT WITHOUT `slot_geom=` IS REFUSED. The host would use the caller's
radius for it, which is not in the `.family`, so there is nothing to generate
from — `so101_tabletop`, the one family like that, has a hand-written table.

⚠ `region_moves` IS THE SITE'S BODY CHAIN, not a list. A region whose site hangs
under any joint has a frame that depends on this reset's joint draws, and the
kernel runs before FK; `check.require_device_placement` refuses a task that
draws in one.
"""

from std.os import listdir
from std.sys import argv

from mojo_rl.tasks.spec import load_family, FamilySpec, SLOT_FREE
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.reset import free_slot_addresses
from mojo_rl.tasks.eval import region_sites
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics

comptime DT = DType.float64
comptime FAMILY_DIR = "mojo_rl/tasks/families"
comptime OUT_DIR = "mojo_rl/tasks/placement"


def _families() raises -> List[String]:
    """Every `libero*.family`, sorted — enumerated like `gen_family_scene`."""
    var out = List[String]()
    for e in listdir(FAMILY_DIR):
        var n = String(e)
        if n.startswith("libero") and n.endswith(".family"):
            out.append(String(n[byte = 0 : n.byte_length() - 7]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def struct_name(family: String) -> String:
    """`libero_kitchen_scene10` -> `LiberoKitchenScene10Placement`."""
    var out = String("")
    var up = True
    for cp in family.codepoint_slices():
        var c = String(cp)
        if c == "_":
            up = True
            continue
        out += c.upper() if up else c
        up = False
    return out + "Placement"


def _f(v: Float64) -> String:
    """`Scalar[DTYPE](<v>)`, with `v` as Mojo prints it.

    ⚠ MEASURED EXACT BOTH WAYS: `String(Float64)` round-trips on 4000 values
    (random bit patterns included) and the compiler turns those literals back
    into the same bits on 3701. So the decimal IS the bit pattern — and unlike
    a `bitcast` it folds to a `DTYPE` constant, which Metal needs (no double).
    A zero is written `0.0`: the sign of a zero rect bound changes no draw."""
    if v == 0.0:
        return "Scalar[DTYPE](0.0)"
    return "Scalar[DTYPE](" + String(v) + ")"


def _b(v: Bool) -> String:
    return "True" if v else "False"


def _method(
    name: String, arg: String, ret: String, values: List[String]
) -> String:
    """A static method returning `values[arg]` as an `if` chain.

    ⚠ AN `if` CHAIN, NOT A TABLE LITERAL, because this body runs inside a GPU
    kernel: a chain of constant returns lowers on every backend, and a
    comptime aggregate indexed at runtime is one more thing a metallib could
    refuse. Collapsed to one `return` when every entry is the same."""
    var s = "    @staticmethod\n    def " + name + "(" + arg + ": Int) -> " + ret + ":\n"
    if ret == "Float64":
        s = (
            "    @staticmethod\n    def " + name + "[DTYPE: DType](" + arg
            + ": Int) -> Scalar[DTYPE]:\n"
        )
    var same = True
    for i in range(1, len(values)):
        if values[i] != values[0]:
            same = False
    if len(values) == 0:
        # a family with no regions still has to satisfy the trait
        var zero = String("Scalar[DTYPE](0.0)") if ret == "Float64" else (
            String("False") if ret == "Bool" else String("0")
        )
        return s + "        return " + zero + "\n\n"
    if same:
        return s + "        return " + values[0] + "\n\n"
    for i in range(len(values) - 1):
        s += "        if " + arg + " == " + String(i) + ":\n"
        s += "            return " + values[i] + "\n"
    s += "        return " + values[len(values) - 1] + "\n\n"
    return s


def generate(family: String) raises -> String:
    var f = load_family(String(FAMILY_DIR) + "/" + family + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    # ⚠ `fmd.bodies` HAS NO WORLDBODY RECORD: model body id `bi` is
    # `fmd.bodies[bi - 1]` (`fields_build`), while `body_names`, a site's
    # `body_id`, a joint's `body_id` and a body's `parent` all count the
    # worldbody as 0. Checked, because an off-by-one here would walk a
    # NEIGHBOUR's chain and mark the wrong regions as moving.
    if len(fmd.bodies) + 1 != len(fmd.body_names):
        raise Error(
            family + ": " + String(len(fmd.bodies)) + " body records against "
            + String(len(fmd.body_names)) + " names — expected one more name"
            " (the worldbody)"
        )
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    # ⚠ THE HOST'S OWN RESET STATE BEFORE ITS FK: zeros, then `base_qpos`
    # (`test_libero_scenes`). Only a site under a joint could see the
    # difference, and those are marked `moves` and refused.
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    forward_kinematics["cpu", DT, DynDims, 1](d, m)

    var jt = List[Int]()
    var jqn = List[Int]()
    var jvn = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jqn.append(fmd.joints[i].nq)
        jvn.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jqn, jvn)
    var rsites = region_sites(f, fmd.site_names)

    var fslot = List[String]()
    var fqadr = List[String]()
    var fdadr = List[String]()
    var fgeom = List[String]()
    var frest = List[String]()
    var frad = List[String]()
    var fbot = List[String]()
    var ftop = List[String]()
    for si in range(len(f.slots)):
        ref s = f.slots[si]
        if s.kind != SLOT_FREE:
            continue
        if not s.has_geom:
            raise Error(
                family + ": free slot '" + s.name + "' has no slot_geom=. The"
                " host sampler would use the CALLER's radius for it, which the"
                " .family does not hold; write this family's table by hand."
            )
        fslot.append(String(si) + "  # " + s.name)
        fqadr.append(String(addrs[si].qadr))
        fdadr.append(String(addrs[si].dadr))
        fgeom.append(_b(True))
        frest.append(_f(-s.bottom_z))
        frad.append(_f(s.h_radius))
        fbot.append(_f(s.bottom_z))
        ftop.append(_f(s.top_z))

    var rx = List[String]()
    var ry = List[String]()
    var rz = List[String]()
    var rrect = List[String]()
    var rx0 = List[String]()
    var ry0 = List[String]()
    var rx1 = List[String]()
    var ry1 = List[String]()
    var ranch = List[String]()
    var rcg = List[String]()
    var rctop = List[String]()
    var rmove = List[String]()
    var n_moves = 0
    for r in range(len(f.regions)):
        ref reg = f.regions[r]
        var sid = rsites[r]
        rx.append(_f(Float64(d.site_xpos.data[sid * 3])))
        ry.append(_f(Float64(d.site_xpos.data[sid * 3 + 1])))
        rz.append(_f(Float64(d.site_xpos.data[sid * 3 + 2])))
        rrect.append(_b(reg.has_rect) + "  # " + reg.name)
        rx0.append(_f(reg.x_min))
        ry0.append(_f(reg.y_min))
        rx1.append(_f(reg.x_max))
        ry1.append(_f(reg.y_max))
        var anchored = reg.contact.byte_length() > 0
        ranch.append(_b(anchored))
        var cg = False
        var ctop = 0.0
        if anchored:
            var ci = f.slot_index(reg.contact)
            if ci >= 0 and f.slots[ci].has_geom:
                cg = True
                ctop = f.slots[ci].top_z
        rcg.append(_b(cg))
        rctop.append(_f(ctop))
        var moves = False
        var b = fmd.sites[sid].body_id
        var guard = 0
        while b > 0:
            for k in range(len(fmd.joints)):
                if fmd.joints[k].body_id == b:
                    moves = True
            b = fmd.bodies[b - 1].parent
            guard += 1
            if guard > len(fmd.bodies):
                raise Error(family + ": a cycle in the body parent chain")
        if moves:
            n_moves += 1
        rmove.append(_b(moves))

    var name = struct_name(family)
    var o = String("")
    o += '"""`' + family + "`'s device placement table — GENERATED, DO NOT EDIT.\n\n"
    o += "Regenerate with:  pixi run gen-placement-tables\n"
    o += "CI checks it with: pixi run gen-placement-tables --check\n\n"
    o += "From `" + String(FAMILY_DIR) + "/" + family + ".family`,\n"
    o += "`" + scene_path(f) + "` and forward kinematics on it.\n"
    o += String(len(fslot)) + " free slots, " + String(len(f.regions))
    o += " regions, " + String(n_moves) + " of them moving.\n"
    o += "See `placement/table.mojo` for what each method means.\n"
    o += '"""\n\n'
    o += "from mojo_rl.tasks.placement.table import PlacementTable\n\n\n"
    o += "struct " + name + "(PlacementTable):\n"
    o += "    comptime N_SLOTS: Int = " + String(len(f.slots)) + "\n"
    o += "    comptime N_FREE: Int = " + String(len(fslot)) + "\n"
    o += "    comptime N_REGIONS: Int = " + String(len(f.regions)) + "\n"
    o += "    comptime NQ: Int = " + String(nq) + "\n"
    o += "    comptime NV: Int = " + String(nv) + "\n\n"
    o += _method("free_slot", "j", "Int", fslot)
    o += _method("free_qadr", "j", "Int", fqadr)
    o += _method("free_dadr", "j", "Int", fdadr)
    o += _method("free_has_geom", "j", "Bool", fgeom)
    o += _method("free_rest", "j", "Float64", frest)
    o += _method("free_radius", "j", "Float64", frad)
    o += _method("free_bottom_z", "j", "Float64", fbot)
    o += _method("free_top_z", "j", "Float64", ftop)
    o += _method("region_site_x", "r", "Float64", rx)
    o += _method("region_site_y", "r", "Float64", ry)
    o += _method("region_site_z", "r", "Float64", rz)
    o += _method("region_has_rect", "r", "Bool", rrect)
    o += _method("region_x0", "r", "Float64", rx0)
    o += _method("region_y0", "r", "Float64", ry0)
    o += _method("region_x1", "r", "Float64", rx1)
    o += _method("region_y1", "r", "Float64", ry1)
    o += _method("region_anchored", "r", "Bool", ranch)
    o += _method("region_contact_has_geom", "r", "Bool", rcg)
    o += _method("region_contact_top_z", "r", "Float64", rctop)
    o += _method("region_moves", "r", "Bool", rmove)
    # one trailing newline, not two
    return String(o[byte = 0 : o.byte_length() - 1])


def main() raises:
    var args = argv()
    var check = False
    for i in range(len(args)):
        if String(args[i]) == "--check":
            check = True
    var fams = _families()
    if len(fams) == 0:
        raise Error("no libero*.family under " + FAMILY_DIR)
    var stale = 0
    for i in range(len(fams)):
        var text = generate(fams[i])
        var path = String(OUT_DIR) + "/" + fams[i] + ".mojo"
        var old = String("")
        var have = True
        try:
            with open(path, "r") as fh:
                old = fh.read()
        except:
            have = False
        if check:
            if not have or old != text:
                print("  STALE:", path)
                stale += 1
            else:
                print("  up to date:", path)
        else:
            with open(path, "w") as fh:
                fh.write(text)
            print("  wrote", path)
    if check and stale > 0:
        raise Error(
            String(stale) + " placement table(s) stale — run"
            " `pixi run gen-placement-tables`"
        )
    print(len(fams), "placement tables", "checked" if check else "written")
