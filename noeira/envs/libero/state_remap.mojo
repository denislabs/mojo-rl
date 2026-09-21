"""LIBERO's recorded joint order -> ours, read from `state_remap_<family>.kv`.

    var r = load_state_remap("libero_goal")
    var q, v = r.convert(row)        # one recorded `states` row -> our (qpos, qvel)

robosuite merges the robot first and then the objects in `:objects` order; our
composer orders by SLOT. Same `nq`, same `nv`, every address different — so a
recorded state copied straight across puts an object's free-joint pose into a
fixture's slide joints, and the scene still loads. `libero_import.mojo`'s §5.1
note ("converts by writing a header and nothing else") is true only if the order
matches, and it does not.

## ⚠⚠ THE TABLE IS DATA BECAUSE DERIVING IT NEEDS THE DEMONSTRATIONS

The remap is by joint NAME, and THEIR names come from the `model_file` recorded
inside each demo — the ~6 GB of gitignored HDF5. `tools/tasks/libero_init_table.py`
compiles that model with MuJoCo, builds the table, VERIFIES it by driving both
models from all fifty frozen inits and comparing body poses, and only then
writes the `.kv`. Everything downstream — the init freeze, the demo importer —
reads the checked-in file and needs neither MuJoCo nor the demos.

⚠ THE VERIFICATION DOES NOT LIVE HERE AND CANNOT. A reader that re-derived the
mapping would be the second implementation of it
(`_a_gate_that_shares_its_reference_implementation_is_blind`); this file's job
is to refuse a table that does not describe the family it is asked about, which
is the one thing it can check without MuJoCo.
"""

from noeira.core.kv import kv_lines, split_on


struct JointRemap(Copyable, ImplicitlyCopyable, Movable):
    """One joint: where it sits in their state and where it sits in ours."""

    var their_name: String
    var our_name: String
    var their_q: Int
    var our_q: Int
    var nq: Int
    var their_v: Int
    var our_v: Int
    var nv: Int

    def __init__(
        out self, var their_name: String, var our_name: String,
        their_q: Int, our_q: Int, nq: Int, their_v: Int, our_v: Int, nv: Int,
    ):
        self.their_name = their_name^
        self.our_name = our_name^
        self.their_q = their_q
        self.our_q = our_q
        self.nq = nq
        self.their_v = their_v
        self.our_v = our_v
        self.nv = nv


struct StateRemap(Movable & Deinitable):
    var family: String
    var nq: Int
    var nv: Int
    var joints: List[JointRemap]

    def __init__(
        out self, var family: String, nq: Int, nv: Int,
        var joints: List[JointRemap],
    ):
        self.family = family^
        self.nq = nq
        self.nv = nv
        self.joints = joints^

    def __init__(out self, *, deinit move: Self):
        self.family = move.family^
        self.nq = move.nq
        self.nv = move.nv
        self.joints = move.joints^

    def row_words(self) -> Int:
        """A recorded row is `[time, qpos, qvel]`, and their nq/nv equal ours.

        ⚠ THAT EQUALITY IS A FACT ABOUT THIS FAMILY, NOT A DEFINITION. The two
        models declare the same joints in a different ORDER, so the widths match
        and the addresses do not; `load_state_remap` checks the widths are
        covered exactly, which is what would fail if a joint were missing from
        one side."""
        return 1 + self.nq + self.nv

    def convert_into(
        self, row: List[Float64], mut q: List[Float64], mut v: List[Float64]
    ) raises:
        """`convert`, writing into buffers the caller owns.

        ⚠ THE ONE A LOOP SHOULD CALL. The demo importer runs this per ROW over
        ~4 500 rows a task; returning two fresh `List`s each time is two heap
        allocations and two frees per row for a permutation that writes every
        element it reads. `convert` is the convenience form and calls this.

        ⚠ IT OVERWRITES EVERY ADDRESS, so the buffers need no clearing —
        `load_state_remap` refuses a table that does not cover all of `nq` and
        `nv` exactly once, which is what makes that true.
        """
        if len(row) != self.row_words():
            raise Error(
                "libero remap: a recorded row is " + String(len(row))
                + " floats but " + self.family + " is 1 + nq " + String(self.nq)
                + " + nv " + String(self.nv) + " = " + String(self.row_words())
            )
        if len(q) != self.nq or len(v) != self.nv:
            raise Error(
                "libero remap: convert_into was given qpos " + String(len(q))
                + " / qvel " + String(len(v)) + " for nq " + String(self.nq)
                + " / nv " + String(self.nv)
            )
        for j in range(len(self.joints)):
            var r = self.joints[j]
            for k in range(r.nq):
                q[r.our_q + k] = row[1 + r.their_q + k]
            for k in range(r.nv):
                v[r.our_v + k] = row[1 + self.nq + r.their_v + k]

    def convert(
        self, row: List[Float64]
    ) raises -> Tuple[List[Float64], List[Float64]]:
        """One recorded `states` row -> our `(qpos, qvel)`.

        ⚠ THE TIME WORD IS DROPPED. `row[0]` is the recording's simulation
        clock; `Data`'s clock is `META_IDX_SIM_TIME` and a reset zeroes it, so
        restoring it would make two episodes of the same demo differ in a field
        nothing reads. `init_table.INIT_TIME_WORDS` records the same decision.
        """
        var q = List[Float64](length=self.nq, fill=0.0)
        var v = List[Float64](length=self.nv, fill=0.0)
        self.convert_into(row, q, v)
        return (q^, v^)


def load_state_remap(family: String) raises -> StateRemap:
    """`noeira/tasks/libero/state_remap_<family>.kv`, refusing a partial table.

    ⚠⚠ IT REFUSES UNLESS EVERY ADDRESS IS COVERED EXACTLY ONCE, on BOTH sides.
    A joint missing from the table leaves a hole in `qpos` that `convert` fills
    with zero — and zero is a legal joint angle, so the state loads, the scene
    renders, and one body is in the wrong place. A duplicated destination is the
    same defect with two joints fighting over an address. Neither is visible in
    the output, so both are checked here.
    """
    var path = String("noeira/tasks/libero/state_remap_") + family + ".kv"
    var text: String
    with open(path, "r") as fh:
        text = fh.read()
    var lines = kv_lines(text, String("libero state remap"))
    var fam = String("")
    var nq = -1
    var nv = -1
    var joints = List[JointRemap]()
    for i in range(len(lines)):
        var key = String(lines[i].key)
        var val = String(lines[i].value)
        if key == "schema_version":
            if Int(val) != 1:
                raise Error(path + ": schema_version " + val + ", expected 1")
        elif key == "family":
            fam = val^
        elif key == "nq":
            nq = Int(val)
        elif key == "nv":
            nv = Int(val)
        elif key == "tasks":
            pass  # provenance only: how many tasks the verification covered
        elif key == "joint":
            var t = split_on(val, String(" "))
            var f8 = List[String]()
            for k in range(len(t)):
                var s = String(t[k].strip())
                if s.byte_length() > 0:
                    f8.append(s^)
            if len(f8) != 8:
                raise Error(
                    path + ": a `joint=` needs 8 fields (theirs ours their_q"
                    " our_q nq_j their_v our_v nv_j), got " + String(len(f8))
                    + ": " + val
                )
            joints.append(JointRemap(
                String(f8[0]), String(f8[1]), Int(f8[2]), Int(f8[3]),
                Int(f8[4]), Int(f8[5]), Int(f8[6]), Int(f8[7]),
            ))
        else:
            raise Error(path + ": unknown key '" + key + "'")

    if fam != family:
        raise Error(
            path + ": declares family '" + fam + "' and was loaded as '"
            + family + "'"
        )
    if nq < 0 or nv < 0:
        raise Error(path + ": missing nq or nv")
    if len(joints) == 0:
        raise Error(path + ": no `joint=` rows — nothing would be remapped")

    # every address covered exactly once, on both sides
    var tq = List[Int](length=nq, fill=0)
    var oq = List[Int](length=nq, fill=0)
    var tv = List[Int](length=nv, fill=0)
    var ov = List[Int](length=nv, fill=0)
    for j in range(len(joints)):
        var r = joints[j]
        for k in range(r.nq):
            if r.their_q + k >= nq or r.our_q + k >= nq:
                raise Error(
                    path + ": joint '" + r.their_name + "' addresses qpos past"
                    " nq " + String(nq)
                )
            tq[r.their_q + k] += 1
            oq[r.our_q + k] += 1
        for k in range(r.nv):
            if r.their_v + k >= nv or r.our_v + k >= nv:
                raise Error(
                    path + ": joint '" + r.their_name + "' addresses qvel past"
                    " nv " + String(nv)
                )
            tv[r.their_v + k] += 1
            ov[r.our_v + k] += 1
    for k in range(nq):
        if tq[k] != 1 or oq[k] != 1:
            raise Error(
                path + ": qpos address " + String(k) + " is covered "
                + String(tq[k]) + " times on their side and " + String(oq[k])
                + " on ours; every address must be covered exactly once. An"
                " uncovered one converts to 0.0, which is a legal joint angle"
                " and therefore invisible."
            )
    for k in range(nv):
        if tv[k] != 1 or ov[k] != 1:
            raise Error(
                path + ": qvel address " + String(k) + " is covered "
                + String(tv[k]) + " / " + String(ov[k]) + " times"
            )
    return StateRemap(String(family), nq, nv, joints^)
