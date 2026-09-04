"""FROZEN INIT STATES — the thing that makes a success rate mean something. P4a.

    write_init_table(path, f, tasks, rows)      # once, offline
    var tbl = load_init_table(path, f, nq, nv)  # every eval, forever
    tbl.apply(i, qpos, qvel)                    # lane i's episode

`TASK_LAYER_PLAN.md` §6.2. A success rate over states the run sampled for
itself is not comparable with anything — not with last week's run, not with
the paper's. LIBERO's `.pruned_init` files are exactly this mechanism, and
loading one shows the shape: `(50, 79) float64`, fifty initial states of
`[time, qpos, qvel]` flattened. It costs nothing to adopt and it is what makes
two numbers a comparison.

## ⚠ IT IS A `TrajectoryStore`, NOT A NEW FILE TYPE

Three columns, one row per frozen episode:

    init_state   float64  [1 + nq + nv]   time, then qpos, then qvel
    task_index   int32    scalar          which task this episode runs
    active_mask  float64  scalar          which slots that task activates

⚠ `init_state` IS LIBERO'S LAYOUT ON PURPOSE, `[time, qpos, qvel]` and in that
order, so a `.pruned_init` can be converted into one of these by writing a
header and nothing else. P5 is the reason; it costs nothing today.

⚠⚠ THE MASK IS STORED, NOT DERIVED. It is recoverable from `task_index` by
looking the task up and calling `active.active_mask` — and that is exactly the
dependency a frozen table exists to remove. An eval run months later must not
change its answer because someone edited an `active=` line in a `.task` file.
The GOAL still comes from the task spec, deliberately: the goal is the
benchmark's definition and belongs under version control, while the init is a
sample and belongs frozen.

⚠ THE TASK TABLE TRAVELS IN THE MANIFEST. `TrajectoryStoreWriter.add_task`
records `task_index -> instruction` byte-exactly, so a table is self-describing
and a per-task breakdown needs no second file.

## ⚠⚠ THE FAMILY KEY, AND WHY A MISMATCH RAISES

`nq` and `nv` are in the ROW SHAPE. A table frozen for a family with three free
slots is 27+24+1 = 52 wide; one with two is 45. Loading the wrong one is not a
subtle error — but a table from a DIFFERENT family that happens to have the
same `nq`/`nv` would load silently and place props by coordinates that mean
something else. So the key carries the family NAME as well as the dimensions,
`load_init_table` compares the whole string, and a mismatch RAISES.

⚠ IT DOES NOT TRUNCATE OR PAD. §6.2 says so and it is worth restating: a
loader that adapted a mismatched table would produce a run that reports a
number, which is worse than one that fails.
"""

from std.memory import alloc

from mojo_rl.data.column import ColumnSpec
from mojo_rl.data.store import TrajectoryStore, TrajectoryStoreWriter
from .spec import FamilySpec, TaskSpec
from .active import active_mask
from .sampler import sample_placements, RegionFrame, SampleReport
from .reset import SlotAddress, reset_slots


comptime INIT_COLUMN: StaticString = "init_state"
comptime TASK_COLUMN: StaticString = "task_index"
comptime MASK_COLUMN: StaticString = "active_mask"

comptime INIT_TIME_WORDS: Int = 1
"""`init_state[0]` is the simulation time, LIBERO's leading word.

⚠ WE WRITE 0.0 AND `apply` DOES NOT RESTORE IT. `Data`'s clock lives in
`META_IDX_SIM_TIME` and a reset zeroes it; a frozen episode starts at t=0 by
construction. The word is here so the row is LIBERO-shaped, not because
anything reads it — and saying that plainly is better than a reader inferring
that time is being restored."""


def family_key(family: String, nq: Int, nv: Int) -> String:
    """The string a table is keyed by. Compared WHOLE, never parsed.

    ⚠ ONE COMPARISON, NOT THREE. A loader that checked name, then nq, then nv
    has three chances to check two of them; a single string equality has one.
    The pieces are still legible in the error because both strings are printed.
    """
    return (
        String("task_family=") + family + ";nq=" + String(nq)
        + ";nv=" + String(nv)
    )


struct InitTable(Movable & Deinitable):
    """Frozen init states for one family, with each row's task and mask."""

    var key: String
    var nq: Int
    var nv: Int
    var _state: List[Float64]
    """Flat `[n_rows, 1 + nq + nv]`."""
    var task_index: List[Int32]
    var mask: List[Float64]
    var tasks: List[String]
    """`tasks[i]` is the instruction for `task_index == i`, or "" if the
    manifest's table does not name it.

    ⚠ INDEXED BY `task_index`, DENSE AND PADDED. The manifest's table is a
    sparse list of `(index, text)`; this is the lookup a breakdown loop wants,
    and the padding is why an unnamed index reads "" here rather than raising
    the way `Manifest.task_text` does. A BREAKDOWN must still be able to name
    its rows, so `task_label` raises instead."""

    def __init__(
        out self,
        var key: String,
        nq: Int,
        nv: Int,
        var state: List[Float64],
        var task_index: List[Int32],
        var mask: List[Float64],
        var tasks: List[String],
    ):
        self.key = key^
        self.nq = nq
        self.nv = nv
        self._state = state^
        self.task_index = task_index^
        self.mask = mask^
        self.tasks = tasks^

    def __init__(out self, *, deinit move: Self):
        self.key = move.key^
        self.nq = move.nq
        self.nv = move.nv
        self._state = move._state^
        self.task_index = move.task_index^
        self.mask = move.mask^
        self.tasks = move.tasks^

    def row_words(self) -> Int:
        return INIT_TIME_WORDS + self.nq + self.nv

    def n_rows(self) -> Int:
        return len(self.task_index)

    def apply(
        self, i: Int, mut qpos: List[Float64], mut qvel: List[Float64]
    ) raises:
        """Write frozen row `i` into `qpos` / `qvel`.

        ⚠ THE WHOLE VECTORS, NOT JUST THE SLOTS. A frozen episode fixes the
        ARM's pose too — `reset_slots` only touches free slots, so an eval that
        restored only those would start from whatever joint angles the previous
        episode ended at and would not be reproducible at all.
        """
        if i < 0 or i >= self.n_rows():
            raise Error(
                "tasks: init row " + String(i) + " out of range (table has "
                + String(self.n_rows()) + ")"
            )
        if len(qpos) != self.nq or len(qvel) != self.nv:
            raise Error(
                "tasks: init table is nq " + String(self.nq) + " / nv "
                + String(self.nv) + " but was applied to qpos "
                + String(len(qpos)) + " / qvel " + String(len(qvel)) + "."
            )
        var b = i * self.row_words() + INIT_TIME_WORDS
        for k in range(self.nq):
            qpos[k] = self._state[b + k]
        for k in range(self.nv):
            qvel[k] = self._state[b + self.nq + k]

    def sim_time(self, i: Int) raises -> Float64:
        """Row `i`'s stored time word. See `INIT_TIME_WORDS` — nothing applies
        it; it is here so the row is LIBERO-shaped and so a converter can
        round-trip one."""
        if i < 0 or i >= self.n_rows():
            raise Error("tasks: init row " + String(i) + " out of range")
        return self._state[i * self.row_words()]

    def task_label(self, i: Int) raises -> String:
        """The instruction for row `i`'s task.

        ⚠ RAISES ON AN UNNAMED INDEX. A per-task breakdown that fell back to
        "task 3" would still print a table, and the table would be the thing
        someone compares to a paper.
        """
        var ti = Int(self.task_index[i])
        if ti < 0 or ti >= len(self.tasks) or self.tasks[ti] == "":
            raise Error(
                "tasks: init row " + String(i) + " has task_index "
                + String(ti) + ", which the table's manifest does not name."
                " The freeze pass must call `add_task` for every index it"
                " writes."
            )
        return String(self.tasks[ti])


def write_init_table(
    path: String,
    family: String,
    nq: Int,
    nv: Int,
    state: List[Float64],
    task_index: List[Int32],
    mask: List[Float64],
    task_names: List[String],
    seed: Int = 0,
    source_commit: String = String(""),
) raises:
    """Freeze `len(task_index)` init states to `path`.

    `state` is flat `[n, 1 + nq + nv]`. `task_names[i]` is the instruction for
    `task_index == i`; an empty entry is not written, so a sparse list is fine.

    ⚠ ONE EPISODE, WHICH IS A LIE THE FORMAT REQUIRES.
    `TrajectoryStoreWriter.close` refuses a store with no episodes, and the
    rows here are not a trajectory — they are independent samples. They go in
    as a single episode of `n` rows rather than `n` episodes of one, because
    `n` one-row episodes would make `ep_offset` an `n`-long identity map that
    every reader has to walk past. Nothing reads the episode index of an init
    table.
    """
    var n = len(task_index)
    if n == 0:
        raise Error(
            "tasks: refusing to freeze an EMPTY init table. An eval loaded"
            " from it would report a success rate over zero episodes, which"
            " prints as 0.0 and reads as a failing policy."
        )
    var words = INIT_TIME_WORDS + nq + nv
    if len(state) != n * words:
        raise Error(
            "tasks: init state is " + String(len(state)) + " floats but "
            + String(n) + " rows of " + String(words) + " (1 + nq " + String(nq)
            + " + nv " + String(nv) + ") is " + String(n * words) + "."
        )
    if len(mask) != n:
        raise Error(
            "tasks: " + String(len(mask)) + " masks for " + String(n)
            + " rows — every row carries its own active set."
        )

    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String(INIT_COLUMN), DType.float64, words))
    cols.append(ColumnSpec(String(TASK_COLUMN), DType.int32, 1))
    cols.append(ColumnSpec(String(MASK_COLUMN), DType.float64, 1))

    var w = TrajectoryStoreWriter(
        String(path),
        cols^,
        env_id=family_key(family, nq, nv),
        seed=seed,
        source_commit=String(source_commit),
    )
    for i in range(len(task_names)):
        if task_names[i] != "":
            w.add_task(i, String(task_names[i]))

    var sb = alloc[Scalar[DType.float64]](n * words).as_unsafe_any_origin()
    var tb = alloc[Scalar[DType.int32]](n).as_unsafe_any_origin()
    var mb = alloc[Scalar[DType.float64]](n).as_unsafe_any_origin()
    for i in range(n * words):
        sb[i] = Scalar[DType.float64](state[i])
    for i in range(n):
        tb[i] = task_index[i]
        mb[i] = Scalar[DType.float64](mask[i])
    w.append[DType.float64](String(INIT_COLUMN), sb, n)
    w.append[DType.int32](String(TASK_COLUMN), tb, n)
    w.append[DType.float64](String(MASK_COLUMN), mb, n)
    w.end_episode()
    w.close()


def load_init_table(
    path: String, family: String, nq: Int, nv: Int
) raises -> InitTable:
    """Load `path`, REFUSING a table that is not this family's.

    ⚠⚠ THE REFUSAL IS THE FEATURE. §6.2: an init table is only valid for the
    family that produced it. A mismatched one would place props by coordinates
    that mean something else in this scene and the run would still report a
    number.
    """
    var want = family_key(family, nq, nv)
    var st = TrajectoryStore(String(path))
    if st.manifest.env_id != want:
        raise Error(
            "tasks: init table '" + path + "' is keyed '"
            + String(st.manifest.env_id) + "' but this family is '" + want
            + "'. Refused — a table from another family loads as coordinates"
            " that mean something else here. Re-freeze it; do not adapt it."
        )
    var words = INIT_TIME_WORDS + nq + nv
    var spec = st.column(String(INIT_COLUMN))
    if spec.row_dim() != words:
        raise Error(
            "tasks: init table '" + path + "' has " + String(spec.row_dim())
            + "-word rows but this family needs " + String(words)
            + " (1 + nq " + String(nq) + " + nv " + String(nv) + "). The key"
            " matched, so the store was written by a build whose family"
            " dimensions have since changed."
        )

    var state = st.load_column[DType.float64](String(INIT_COLUMN))
    var ti32 = st.load_column[DType.int32](String(TASK_COLUMN))
    var mk = st.load_column[DType.float64](String(MASK_COLUMN))
    var n = st.n_rows()
    if len(ti32) != n or len(mk) != n:
        raise Error(
            "tasks: init table '" + path + "' has " + String(n) + " rows but "
            + String(len(ti32)) + " task indices and " + String(len(mk))
            + " masks."
        )

    var tix = List[Int32]()
    var mask = List[Float64]()
    var hi = -1
    for i in range(n):
        tix.append(ti32[i])
        mask.append(Float64(mk[i]))
        if Int(ti32[i]) > hi:
            hi = Int(ti32[i])
    var names = List[String]()
    for _ in range(hi + 1):
        names.append(String(""))
    for i in range(len(st.manifest.tasks)):
        var e = st.manifest.tasks[i]
        if e.index >= 0 and e.index <= hi:
            names[e.index] = String(e.text)

    var flat = List[Float64]()
    for i in range(len(state)):
        flat.append(Float64(state[i]))
    return InitTable(want^, nq, nv, flat^, tix^, mask^, names^)


def append_init_rows(
    t: TaskSpec,
    f: FamilySpec,
    task_index: Int,
    frames: List[RegionFrame],
    radii: List[Float64],
    addrs: List[SlotAddress],
    base_qpos: List[Float64],
    nq: Int,
    nv: Int,
    n: Int,
    seed: UInt64,
    lane0: Int,
    mut state: List[Float64],
    mut task_ix: List[Int32],
    mut mask: List[Float64],
    mut rep: SampleReport,
) raises:
    """Sample `n` episodes of `t` and append them as frozen rows.

    ⚠ THE LANE IS `lane0 + i`, AND IT IS AN ARGUMENT BECAUSE THE STREAM IS
    KEYED BY IT. `sample_placements` is deterministic in `(seed, lane)`, so
    two tasks frozen with the same `lane0` would draw the SAME placements —
    the two blocks of rows would differ only in their `task_index` and any
    per-task success difference would be an artefact of a shared draw. Give
    each task a disjoint lane range (`ti * n` is the obvious one).

    ⚠ `base_qpos` IS THE SCENE'S REST POSE, `nq` long. Every row starts from
    it and `reset_slots` overwrites only the free slots, so the ARM's pose is
    identical across every frozen episode. That is deliberate for a first
    table — the eval varies the props, not the start posture — and it is the
    line to change when a family wants a randomised arm.
    """
    if len(base_qpos) != nq:
        raise Error(
            "tasks: base_qpos is " + String(len(base_qpos)) + " long but nq is "
            + String(nq)
        )
    if n <= 0:
        raise Error(
            "tasks: refusing to freeze " + String(n) + " episodes of task '"
            + t.name + "'."
        )
    var mk = active_mask(t, f)
    for i in range(n):
        var qpos = List[Float64]()
        for k in range(nq):
            qpos.append(base_qpos[k])
        var qvel = List[Float64]()
        for _ in range(nv):
            qvel.append(0.0)
        var placed = sample_placements(
            t, f, frames, radii, seed, lane0 + i, rep
        )
        reset_slots(t, f, placed, addrs, qpos, qvel)
        state.append(0.0)  # the time word — see INIT_TIME_WORDS
        for k in range(nq):
            state.append(qpos[k])
        for k in range(nv):
            state.append(qvel[k])
        task_ix.append(Int32(task_index))
        mask.append(mk)
