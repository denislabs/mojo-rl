"""The host half of the device placement — refuse, and diff the table.

    require_device_placement[LiberoGoalPlacement](t, f)      # per task, host
    var drift = placement_table_drift[LiberoGoalPlacement](   # once, at startup
        f, addrs, frames, moves, radii, nq, nv)

`table.place_free_slots` cannot raise, so everything the host sampler raises on
at the SPEC level has to be refused before the words are written, and every
number the table restates has to be compared against what the host would use.
"""

from mojo_rl.tasks.spec import (
    FamilySpec, TaskSpec, SLOT_FREE, INIT_TARGET_SLOT,
)
from mojo_rl.tasks.reset import SlotAddress
from mojo_rl.tasks.sampler import RegionFrame
from mojo_rl.tasks.placement.table import PlacementTable


def require_device_placement[T: PlacementTable](
    t: TaskSpec, f: FamilySpec
) raises:
    """⚠⚠ REFUSE A TASK THE DEVICE CANNOT DRAW THE WAY THE HOST DOES.

    Replaces `gpu_eval.require_gpu_placement`, which refused every STACK
    because the device walked the free-slot table; the kernel now walks
    `spec.order_inits`, so a stack is drawable and what is left to refuse is:

    * **a region whose site MOVES** — hangs under a joint. The host resolves
      frames after this reset's `jinit=` draws and FK; the kernel runs before FK
      and reads the site as a constant. Measured over the corpus: 1 placement
      of 515, `libero_spatial`'s top-drawer region under the drawer's slide.
    * **an `On` an anchored region whose fixture has no `slot_geom=`** — the
      host raises, having no `top_site` to add.
    * **a stack where either slot has no `slot_geom=`** — the host raises.
    * **a table that is not this family's** — counts and free-slot indices.

    ⚠ IT DOES NOT CHECK THE NUMBERS. That is `placement_table_drift`, which
    needs the composed scene; this needs only the spec, so it can run per task.
    """
    var n_free = 0
    for i in range(len(f.slots)):
        if f.slots[i].kind == SLOT_FREE:
            if n_free < T.N_FREE and T.free_slot(n_free) != i:
                raise Error(
                    "device placement: table's free slot " + String(n_free)
                    + " is family slot " + String(T.free_slot(n_free))
                    + " but family '" + f.name + "' has it at " + String(i)
                    + " — the table is not this family's, or is stale."
                )
            n_free += 1
    if (
        T.N_SLOTS != len(f.slots) or T.N_FREE != n_free
        or T.N_REGIONS != len(f.regions)
    ):
        raise Error(
            "device placement: table has " + String(T.N_SLOTS) + " slots / "
            + String(T.N_FREE) + " free / " + String(T.N_REGIONS)
            + " regions, family '" + f.name + "' has " + String(len(f.slots))
            + " / " + String(n_free) + " / " + String(len(f.regions))
            + ". Regenerate: `pixi run gen-placement-tables`."
        )
    for i in range(len(t.inits)):
        ref it = t.inits[i]
        var si = f.slot_index(it.slot)
        if si < 0 or f.slots[si].kind != SLOT_FREE:
            raise Error(
                "task '" + t.name + "': init '" + it.describe() + "' places a"
                " slot that is not FREE; the device init words cover free"
                " slots only."
            )
        var j = _ordinal[T](si)
        if f.init_target_kind(it.region) == INIT_TARGET_SLOT:
            var rsi = f.slot_index(it.region)
            if rsi < 0 or f.slots[rsi].kind != SLOT_FREE:
                raise Error(
                    "task '" + t.name + "': init '" + it.describe() + "'"
                    " stacks on a slot that is not FREE. The host sampler"
                    " raises on it (a stack takes a PLACED reference's pose)."
                )
            if not (T.free_has_geom(j) and T.free_has_geom(_ordinal[T](rsi))):
                raise Error(
                    "task '" + t.name + "': init '" + it.describe() + "'"
                    " stacks without slot_geom= on both slots; the host"
                    " sampler raises on it."
                )
            continue
        var r = f.region_index(it.region)
        if T.region_moves(r):
            raise Error(
                "task '" + t.name + "': init '" + it.describe() + "' draws in"
                " region '" + it.region + "', whose site hangs under a JOINT."
                " The host resolves that frame after this reset's jinit= draws"
                " and forward kinematics; the device kernel runs before FK and"
                " would read the site where the scene composed it. Run this"
                " task on the host path."
            )
        if (
            T.free_has_geom(j) and T.region_anchored(r) and not it.inside
            and not T.region_contact_has_geom(r)
        ):
            raise Error(
                "task '" + t.name + "': init '" + it.describe() + "' is On a"
                " region whose fixture has no slot_geom=; the host sampler"
                " raises on it."
            )


def _ordinal[T: PlacementTable](si: Int) raises -> Int:
    for j in range(T.N_FREE):
        if T.free_slot(j) == si:
            return j
    raise Error(
        "device placement: family slot " + String(si) + " is not in the"
        " table's free slots"
    )


def placement_table_drift[T: PlacementTable](
    f: FamilySpec,
    addrs: List[SlotAddress],
    frames: List[RegionFrame],
    moves: List[Bool],
    radii: List[Float64],
    nq: Int,
    nv: Int,
) raises -> List[String]:
    """Every number `T` restates, against what the HOST sampler would use.

    `addrs` from `reset.free_slot_addresses`, `frames` from forward kinematics
    on the composed scene (the host's own `RegionFrame`s), `moves` per region
    from the site's body chain, `radii` the caller's fallback per family slot.
    Returns one line per disagreement; empty means the table is current.

    ⚠ EXACT EQUALITY, NOT A TOLERANCE. The generator writes each float as
    Mojo's `String(Float64)` — MEASURED round-trip exact on 4000 values, and
    the compiler's literal conversion exact on 3701 — so a table built from
    these very frames matches them to the bit, and any difference at all is a
    stale table or a changed scene. (`Float64(String)` at RUNTIME is not exact;
    that path is not used.)
    """
    var out = List[String]()
    if T.NQ != nq or T.NV != nv:
        out.append(
            "NQ/NV " + String(T.NQ) + "/" + String(T.NV) + " vs scene "
            + String(nq) + "/" + String(nv)
        )
    if T.N_SLOTS != len(f.slots) or T.N_REGIONS != len(f.regions):
        out.append("slot/region counts")
        return out^
    var j = 0
    for si in range(len(f.slots)):
        ref s = f.slots[si]
        if s.kind != SLOT_FREE:
            continue
        if j >= T.N_FREE:
            out.append("more free slots than the table's " + String(T.N_FREE))
            return out^
        var who = "free " + String(j) + " (" + s.name + ")"
        if T.free_slot(j) != si:
            out.append(who + ": slot index")
        if T.free_qadr(j) != addrs[si].qadr or T.free_dadr(j) != addrs[si].dadr:
            out.append(who + ": qpos/qvel address")
        if T.free_has_geom(j) != s.has_geom:
            out.append(who + ": has_geom")
        var rest = -s.bottom_z if s.has_geom else radii[si]
        var rad = s.h_radius if s.has_geom else radii[si]
        if T.free_rest[DType.float64](j) != rest:
            out.append(who + ": rest " + String(T.free_rest[DType.float64](j)) + " vs "
                       + String(rest))
        if T.free_radius[DType.float64](j) != rad:
            out.append(who + ": radius " + String(T.free_radius[DType.float64](j)) + " vs "
                       + String(rad))
        if s.has_geom and (
            T.free_bottom_z[DType.float64](j) != s.bottom_z or T.free_top_z[DType.float64](j) != s.top_z
        ):
            out.append(who + ": bottom_z/top_z")
        j += 1
    if j != T.N_FREE:
        out.append("free slot count " + String(j) + " vs table "
                   + String(T.N_FREE))
    for r in range(len(f.regions)):
        ref reg = f.regions[r]
        var who = "region " + String(r) + " (" + reg.name + ")"
        if (
            T.region_site_x[DType.float64](r) != frames[r].x
            or T.region_site_y[DType.float64](r) != frames[r].y
            or T.region_site_z[DType.float64](r) != frames[r].z
        ):
            out.append(
                who + ": site (" + String(T.region_site_x[DType.float64](r)) + ", "
                + String(T.region_site_y[DType.float64](r)) + ", "
                + String(T.region_site_z[DType.float64](r)) + ") vs FK (" + String(frames[r].x)
                + ", " + String(frames[r].y) + ", " + String(frames[r].z) + ")"
            )
        if T.region_has_rect(r) != reg.has_rect:
            out.append(who + ": has_rect")
        if reg.has_rect and (
            T.region_x0[DType.float64](r) != reg.x_min or T.region_y0[DType.float64](r) != reg.y_min
            or T.region_x1[DType.float64](r) != reg.x_max or T.region_y1[DType.float64](r) != reg.y_max
        ):
            out.append(who + ": rectangle")
        var anchored = reg.contact.byte_length() > 0
        if T.region_anchored(r) != anchored:
            out.append(who + ": anchored")
        if anchored:
            var ci = f.slot_index(reg.contact)
            var cg = ci >= 0 and f.slots[ci].has_geom
            if T.region_contact_has_geom(r) != cg:
                out.append(who + ": contact has_geom")
            if cg and T.region_contact_top_z[DType.float64](r) != f.slots[ci].top_z:
                out.append(who + ": contact top_z")
        if T.region_moves(r) != moves[r]:
            out.append(who + ": moves")
    return out^
