"""Carry the pose across a rebuild — by NAME, because the index changed. V2.2.

## ⚠⚠ THE INDEX IS EXACTLY WHAT A STRUCTURAL EDIT DESTROYS

A dims-preserving edit writes into the live `Model` and nothing moves. A
STRUCTURAL one re-parses, and the parser assigns `qpos` addresses by walking
the tree it is given — so deleting one joint near the root shifts the address
of every joint after it. Copying `qpos` across positionally would take the
knee's angle and write it into the ankle: a pose that is not wrong in any
single number and is wrong everywhere.

⇒ **the JOINT NAME is the key**, which is the reason `FlatModelDef` was taught
to carry names at all (plan §1.3, third bullet).

## ⚠ THE TYPE HAS TO MATCH TOO, AND THAT IS NOT PEDANTRY

`nq` is 7 for a free joint, 4 for a ball, 1 for a hinge or a slide. A name that
survives an edit but changes type — free to hinge is the realistic one, when a
prop is pinned down — would otherwise copy seven numbers into a slot that holds
one, straight over the next six joints. So a slot is carried only when the name
AND the type agree; anything else keeps the reference pose, which is a visible,
recoverable answer rather than a corrupted one.

## ⚠ WHAT IS DELIBERATELY NOT CARRIED

`qacc`, the constraint warm-start and the actuator activation state. They are
derived from a model that no longer exists, and a stale warm-start is a
silent wrong answer for one step rather than a visible one. The pose and the
velocities are what a user recognises as "where I left it".
"""

from ..fields import Data, DynDims
from ..parser.flat_model import FlatModelDef


comptime DT = DType.float64


def joint_qpos_adr(fmd: FlatModelDef) -> List[Int]:
    """`qpos` address per joint, in MuJoCo joint order.

    ⚠ A RUNNING SUM OF `nq`, NOT A STORED FIELD. `FlatModelDef` holds the
    joints in the order the parser emitted them, which IS MuJoCo's order
    (grouped by body), so the cumulative sum is the address by construction.
    Reading it from anywhere else would introduce a second definition of the
    layout.
    """
    var adr = List[Int]()
    var n = 0
    for j in fmd.joints:
        adr.append(n)
        n += j.nq
    return adr^


def joint_dof_adr(fmd: FlatModelDef) -> List[Int]:
    var adr = List[Int]()
    var n = 0
    for j in fmd.joints:
        adr.append(n)
        n += j.nv
    return adr^


struct RemapReport(Copyable, Movable):
    """How much of the pose survived, so the studio can say so."""

    var carried: Int
    """Joints whose value was carried across."""
    var reset: Int
    """Joints that fell back to the reference pose (new, renamed or retyped)."""
    var dropped: Int
    """Joints in the OLD model with no counterpart in the new one."""
    var restored: Int
    """Joints filled from a POSE SNAPSHOT rather than from the live state —
    see `apply_pose_snapshot`. Zero unless the caller offered one."""
    var carried_mask: List[Bool]
    """Per NEW joint: did the live state account for it?

    ⚠ THE MASK IS THE REASON UNDO CAN RESTORE A DELETED LEG'S POSE. "which
    slots are still at the reference" cannot be recovered by inspecting
    `qpos` afterwards — a joint whose live value happens to equal `qpos0` is
    indistinguishable from one that was never written. It has to be recorded
    where it is known."""

    def __init__(out self, carried: Int, reset: Int, dropped: Int,
                 carried_mask: List[Bool]):
        self.carried = carried
        self.reset = reset
        self.dropped = dropped
        self.restored = 0
        self.carried_mask = carried_mask.copy()

    def summary(self) -> String:
        var s = (
            String(self.carried) + " joint(s) kept their pose, "
            + String(self.reset) + " reset to the reference, "
            + String(self.dropped) + " gone with the edit"
        )
        if self.restored > 0:
            s += String(" (", self.restored, " of the reset came back from"
                        " the snapshot)")
        return s^


def _index_of(names: List[String], want: String) -> Int:
    """⚠ "" NEVER MATCHES. An unnamed joint has no identity across a rebuild,
    and matching two of them by their emptiness would pair arbitrary joints."""
    if want.byte_length() == 0:
        return -1
    for i in range(len(names)):
        if names[i] == want:
            return i
    return -1


def remap_state(
    old_fmd: FlatModelDef,
    old_d: Data[DT, DynDims, 1],
    new_fmd: FlatModelDef,
    mut new_d: Data[DT, DynDims, 1],
) -> RemapReport:
    """Copy `qpos`/`qvel` from the old model into the new one, joint by name.

    `new_d` must already hold the reference pose — this only OVERWRITES the
    slots it can account for, so a joint with no counterpart keeps `qpos0`.
    """
    var old_q = joint_qpos_adr(old_fmd)
    var old_v = joint_dof_adr(old_fmd)
    var new_q = joint_qpos_adr(new_fmd)
    var new_v = joint_dof_adr(new_fmd)

    var carried = 0
    var reset = 0
    var matched_old = 0
    var mask = List[Bool]()
    for _ in range(len(new_fmd.joints)):
        mask.append(False)

    for nj in range(len(new_fmd.joints)):
        var name = new_fmd.joint_names[nj] if nj < len(new_fmd.joint_names) \
            else String("")
        var oj = _index_of(old_fmd.joint_names, name)
        if oj == -1 or oj >= len(old_fmd.joints):
            reset += 1
            continue
        # ⚠ THE TYPE GUARD. Same name, different type = different nq, and
        # copying anyway walks over the joints after it.
        if old_fmd.joints[oj].jnt_type != new_fmd.joints[nj].jnt_type:
            reset += 1
            continue
        matched_old += 1
        carried += 1
        mask[nj] = True
        for k in range(new_fmd.joints[nj].nq):
            new_d.qpos.data[new_q[nj] + k] = old_d.qpos.data[old_q[oj] + k]
        for k in range(new_fmd.joints[nj].nv):
            new_d.qvel.data[new_v[nj] + k] = old_d.qvel.data[old_v[oj] + k]

    return RemapReport(carried, reset, len(old_fmd.joints) - matched_old,
                       mask)


# =============================================================================
# The pose of what came BACK — V2.9
# =============================================================================


struct PoseSnapshot(Copyable, Movable):
    """A model's joints and their live values, keyed by name.

    ⚠⚠ WHY THIS EXISTS. Undo restores a deleted body, and `remap_state` fills
    the new model from the LIVE one — which no longer contains the joints that
    were deleted. So the leg came back at `qpos0` while the rest of the robot
    stayed where it was: half the pose from now, half from the reference, and
    a limb visibly popping into a different attitude on undo.

    ⚠ A `FlatModelDef` CANNOT BE THE SNAPSHOT. It is `Movable` and not
    `Copyable` — the same constraint that made `EditLog` a replay rather than
    a snapshot in the first place. All that is needed to place a value is the
    joint's NAME, its TYPE and its width, so that is what this holds.
    """

    var names: List[String]
    var types: List[Int]
    var nq: List[Int]
    var nv: List[Int]
    var qpos: List[Float64]
    var qvel: List[Float64]

    def __init__(out self):
        self.names = List[String]()
        self.types = List[Int]()
        self.nq = List[Int]()
        self.nv = List[Int]()
        self.qpos = List[Float64]()
        self.qvel = List[Float64]()


def pose_snapshot(
    fmd: FlatModelDef, d: Data[DT, DynDims, 1]
) -> PoseSnapshot:
    """Capture every joint's live `qpos`/`qvel`, packed per joint."""
    var s = PoseSnapshot()
    var qa = joint_qpos_adr(fmd)
    var va = joint_dof_adr(fmd)
    for j in range(len(fmd.joints)):
        s.names.append(
            fmd.joint_names[j].copy() if j < len(fmd.joint_names)
            else String("")
        )
        s.types.append(fmd.joints[j].jnt_type)
        s.nq.append(fmd.joints[j].nq)
        s.nv.append(fmd.joints[j].nv)
        for k in range(fmd.joints[j].nq):
            s.qpos.append(Float64(d.qpos.data[qa[j] + k]))
        for k in range(fmd.joints[j].nv):
            s.qvel.append(Float64(d.qvel.data[va[j] + k]))
    return s^


def apply_pose_snapshot(
    snap: PoseSnapshot,
    new_fmd: FlatModelDef,
    mut new_d: Data[DT, DynDims, 1],
    mut rep: RemapReport,
):
    """Fill the slots the LIVE state could not account for, from `snap`.

    ⚠⚠ ONLY THE UNCARRIED SLOTS, AND THAT ORDERING IS THE WHOLE DESIGN. The
    live state wins wherever it has an answer, so an undo does NOT rewind the
    running sim: the robot stays where it is and only the parts that just came
    back from the dead take their old values. Applying the snapshot to
    everything would teleport the entire model to where it stood at the edit,
    which in a running sim is a bigger surprise than the bug it fixes.

    ⚠ NAME **AND** TYPE, as in `remap_state` — the same guard, because the
    same "free joint became a hinge" case writes seven numbers into a slot
    that holds one.
    """
    var qa = joint_qpos_adr(new_fmd)
    var va = joint_dof_adr(new_fmd)
    # Where each snapshot joint's values start, in the packed lists.
    var sq = List[Int]()
    var sv = List[Int]()
    var q = 0
    var v = 0
    for j in range(len(snap.names)):
        sq.append(q)
        sv.append(v)
        q += snap.nq[j]
        v += snap.nv[j]

    for nj in range(len(new_fmd.joints)):
        if nj < len(rep.carried_mask) and rep.carried_mask[nj]:
            continue
        var name = new_fmd.joint_names[nj] if nj < len(new_fmd.joint_names) \
            else String("")
        var sj = _index_of(snap.names, name)
        if sj == -1:
            continue
        if snap.types[sj] != new_fmd.joints[nj].jnt_type:
            continue
        for k in range(new_fmd.joints[nj].nq):
            new_d.qpos.data[qa[nj] + k] = Scalar[DT](snap.qpos[sq[sj] + k])
        for k in range(new_fmd.joints[nj].nv):
            new_d.qvel.data[va[nj] + k] = Scalar[DT](snap.qvel[sv[sj] + k])
        rep.restored += 1
        if nj < len(rep.carried_mask):
            rep.carried_mask[nj] = True
