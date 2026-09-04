"""Putting a sampled episode into `qpos` — the reset adapter.

    var adrs = free_slot_addresses(f, joint_names, joints)
    reset_slots(t, f, placed, adrs, qpos, qvel)

The sampler says WHERE each active slot starts; this says where that goes in
the state vector, and parks everything else.

## ⚠ THE ADDRESSES ARE RESOLVED ONCE, THE WRITES ARE SCALAR

`free_slot_addresses` walks joint names and is host-only. `write_free_pose` is
a handful of stores over plain scalars, so P3's reset kernel calls it per lane
unchanged — the same split as `eval.mojo`'s predicates and `sampler.mojo`'s
draws, and for the same reason: a second implementation on the device leg is
how the two stop agreeing.

## ⚠⚠ qvel IS ZEROED, AND THAT IS NOT TIDINESS

A free body carries six velocity dofs that no `qpos` write touches. Reusing a
`Data` across episodes without zeroing them starts the new episode with the old
one's momentum: a prop placed neatly on the table immediately slides off, and
the trajectory looks like a physics bug rather than a reset bug.

⚠ It is also what makes PARKING an invariant rather than an initial condition.
`docs/TASK_LAYER_IMPLEMENTATION.md` Gap D: gravity is a `Model` field shared by
every lane, so a parked body FALLS. Zeroing its velocity at reset does not stop
that — only the per-step repark does — but it does stop the fall compounding
across episodes.

## ⚠ ONLY FREE SLOTS APPEAR HERE

A `static` slot has no joint and therefore no state: it is composed where it
lives (`spec.SlotSpec`) and cannot be moved at reset. Asking for its address
raises rather than returning something that looks like an address.
"""

from .spec import FamilySpec, TaskSpec, SLOT_FREE, SLOT_STATIC
from .sampler import Placement
from .family import park_pos
from mojo_rl.physics3d.joint_types import JNT_FREE


struct SlotAddress(Copyable, ImplicitlyCopyable, Movable):
    """Where a free slot's state lives. `-1` for a slot with no free joint."""

    var qadr: Int
    var dadr: Int

    def __init__(out self, qadr: Int, dadr: Int):
        self.qadr = qadr
        self.dadr = dadr


def free_slot_addresses(
    f: FamilySpec,
    joint_names: List[String],
    joint_types: List[Int],
    joint_nq: List[Int],
    joint_nv: List[Int],
) raises -> List[SlotAddress]:
    """Every slot's `(qpos, qvel)` base address, in family slot order.

    ⚠ TAKES THE JOINT TABLE AS FOUR PARALLEL LISTS rather than
    `List[JointData]`, so `tasks/` does not import a physics3d record type
    into its signature. §7's dependency rule is one-way, and a struct in the
    signature is the kind of coupling that makes it two-way later.

    ⚠ `joint_names` IS 1:1 WITH THE JOINT TABLE — unlike `body_names`, whose
    index 0 is the worldbody. Measured on the composed family: 9 joints, 9
    names, free joints at qpos 6 / 13 / 20. Assuming the body convention here
    would shift every address by one joint and still return a plausible
    number.
    """
    var out = List[SlotAddress]()
    for si in range(len(f.slots)):
        ref s = f.slots[si]
        if s.kind != SLOT_FREE:
            out.append(SlotAddress(-1, -1))
            continue
        var want = s.name + "_"
        var qadr = 0
        var dadr = 0
        var found = -1
        for j in range(len(joint_names)):
            if (
                found < 0
                and joint_types[j] == JNT_FREE
                and String(joint_names[j]).startswith(want)
            ):
                found = j
                break
            qadr += joint_nq[j]
            dadr += joint_nv[j]
        if found < 0:
            raise Error(
                "tasks: free slot '" + s.name + "' has no free joint in the"
                " composed scene — no joint named '" + want + "*' of type"
                " FREE. Either the asset declares no <freejoint>, or the scene"
                " is stale (`pixi run gen-family-scenes`)."
            )
        out.append(SlotAddress(qadr, dadr))
    return out^


@always_inline
def write_free_pose(
    mut qpos: List[Float64], qadr: Int,
    x: Float64, y: Float64, z: Float64,
):
    """A free joint's 7 `qpos`: position then an IDENTITY quaternion.

    ⚠ THE QUATERNION IS NOT OPTIONAL AND IT IS NOT ZERO. `(0,0,0,0)` is a
    DEGENERATE rotation — forward kinematics normalises it and gets a
    division by zero or a NaN pose, depending on the path. Writing only the
    three positions and leaving the quaternion at whatever the buffer held is
    the same trap one step removed.
    """
    qpos[qadr + 0] = x
    qpos[qadr + 1] = y
    qpos[qadr + 2] = z
    qpos[qadr + 3] = 1.0
    qpos[qadr + 4] = 0.0
    qpos[qadr + 5] = 0.0
    qpos[qadr + 6] = 0.0


@always_inline
def write_free_vel_zero(mut qvel: List[Float64], dadr: Int):
    """A free joint's 6 `qvel`. See the module header for why this matters."""
    for k in range(6):
        qvel[dadr + k] = 0.0


def reset_slots(
    t: TaskSpec,
    f: FamilySpec,
    placed: List[Placement],
    addrs: List[SlotAddress],
    mut qpos: List[Float64],
    mut qvel: List[Float64],
) raises:
    """Write this episode: active slots where the sampler put them, every
    other free slot parked. Velocities zeroed either way.

    ⚠ EVERY FREE SLOT IS WRITTEN, not just the active ones. The fixed scene
    budget means an inactive slot still EXISTS and still has state; leaving it
    at whatever the previous episode ended with is how a "parked" object turns
    up in the middle of the table two episodes later.
    """
    for si in range(len(f.slots)):
        if f.slots[si].kind != SLOT_FREE:
            continue
        ref a = addrs[si]
        if a.qadr < 0:
            raise Error(
                "tasks: free slot '" + f.slots[si].name + "' has no address —"
                " `free_slot_addresses` was built from a different family."
            )

        var active = False
        for p in range(len(placed)):
            if placed[p].slot == si:
                active = True
                write_free_pose(
                    qpos, a.qadr, placed[p].x, placed[p].y, placed[p].z
                )
                break
        if not active:
            var pk = park_pos(f, si)
            write_free_pose(qpos, a.qadr, pk[0], pk[1], pk[2])
        write_free_vel_zero(qvel, a.dadr)
    _ = t
