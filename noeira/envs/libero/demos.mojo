"""LIBERO's demonstration HDF5 -> a `TrajectoryStore`, natively — G13.

    var r = import_libero_demos(
        "references/libero_demos/libero_goal", "libero_goal",
        "build/demos/libero_goal.h5",
    )

No Python, no h5py, no MuJoCo: `io/hdf5` is an FFI onto libhdf5 and this walks
their layout with it. Same standing as `data/lerobot.mojo` — the native importer
is the supported path.

## THEIR LAYOUT, AND WHAT IS TAKEN FROM IT

    data/demo_<i>/actions              (T, 7)          float64
    data/demo_<i>/states               (T, 1+nq+nv)    float64
    data/demo_<i>/obs/joint_states     (T, 7)          float64
    data/demo_<i>/obs/gripper_states   (T, 2)          float64
    data/demo_<i>/obs/agentview_rgb    (T, 128,128,3)  uint8
    data/demo_<i>/obs/eye_in_hand_rgb  (T, 128,128,3)  uint8

`joint_states` + `gripper_states` are exactly `cfg.data.obs.modality.low_dim`
and the two `_rgb` keys exactly `.rgb`, so the store carries the policy's whole
observation and nothing else from the recording.

## `qpos` IS THE TWO LOW-DIM COLUMNS AS ONE, PLUS THE TASK — THE NAME THE POLICIES READ

`ACTDataset` and `SmolVLABatchSampler` read the proprioceptive vector from a
column called `qpos` — the LeRobot importer's `observation.state` — and refuse
a store without one. LIBERO's is `joint_states` (7) ++ `gripper_states` (2),
then **their one-step difference** (9 more: row r minus row r-1 of the same
episode, zero on an episode's first row), and then **a one-hot of
`task_index` over the suite's tasks**, so the store carries `18 + n_tasks`
float32 per row and the image policies train on it without a per-dataset
column map.

⚠⚠ THE DIFFERENCE IS THERE BECAUSE ONE POSE DOES NOT CARRY THE PHASE. The
operators disagree by ±10 steps on WHEN the sideways pull at a drawer handle
starts; at one arm pose the demonstrations are half still-approaching and
half already-pulling, and the L1 median of a split set is ~0 — the fitted
ACT descended past the handle and never pulled (0-5/200, three fits). A
nearest-neighbour median chunk from the nine joints alone reproduces that;
the same with the nine one-step differences scores 0.60 on LIBERO's frozen
inits with no network and no camera (`libero_eval_batched --knn --knn-vel`,
5090, 2026-09-20). The difference is what the eval can compute from its own
previous step, which is why it is a difference and not `qvel`.

⚠⚠ THE ONE-HOT IS THERE BECAUSE THE PICTURE DOES NOT CARRY THE TASK. Every
libero_goal task is the SAME scene at the SAME layout; only the instruction
differs. The first ACT fit on a nine-word `qpos` (2026-09-18) predicted the
mean of ten behaviours — position deltas a tenth of the demonstrations', the
gripper hedged at 0.6 where every demo says ±1 — and scored 0/200 with the
ensemble on or off. LIBERO's own BC baselines carry a task embedding for this
reason; a one-hot is that embedding for a fixed set of tasks. The two source
columns stay as recorded: `libero_demo_import.mojo`'s gate reads them and
`task_index` back and refuses a `qpos` that is not their concatenation.

## ⚠⚠ THE STORE'S ROW 0 IS THE TOP OF THE PICTURE — THE RECORDING'S IS THE BOTTOM

LIBERO's `*_rgb` datasets are in OpenGL's row order (robosuite hands
`mjr_readPixels`' buffer through unchanged; `tools/libero/libero_camera_gate.py`
measured it at 26 dB). Their policies train on the picture upside down and
never notice, because a network has no "up". OURS DO: the same policy is
evaluated on frames the batched tracer renders (`raytrace/batch.mojo`, row 0
the top), so a store in the recording's order would train on one orientation
and deploy on the other. The flip is applied HERE, once, at import — every
image in a `TrajectoryStore` of this tree is top row first, whichever camera
or recorder produced it. The gate compares the store's FIRST image row against
the source's LAST, through a different index path, so an importer that stops
flipping fails it.

## ⚠⚠ `states` IS REWRITTEN INTO OUR JOINT ORDER, AND THAT IS THE POINT

An `actions`-and-images store is a dataset any BC codebase could build. The
column only we can produce is `state`: their `(time, qpos, qvel)` remapped by
joint NAME into our composed scene's addresses, so a row can be LOADED into our
engine — for a replay, a re-render through our cameras, or a reset. The remap
comes from `envs/libero/state_remap`, i.e. from the checked-in `.kv` that
`tools/libero/libero_init_table.py` wrote after verifying it against MuJoCo.

⚠ THE TIME WORD IS DROPPED, so `state` is `nq + nv` wide and not `1 + nq + nv`.
`StateRemap.convert` says why; an init TABLE keeps the word because LIBERO's own
file has it and a converter must round-trip one, while a demo store has no such
obligation and a column nothing reads is a column that drifts.

## ⚠ ONE EPISODE PER DEMO, AND THE IMAGES ARE THE WHOLE COST

`TrajectoryStoreWriter` is append-only and row-major, so each demo's rows go in
contiguously and `end_episode` marks the boundary — `ep_offset`/`ep_len` are then
the demo index. The two 128x128 RGB streams are 98 304 bytes per row: a
4 460-row task is 438 MB, a ten-task suite about 4.4 GB. `images=False` writes
the store without them, which is what a state-only replay or a low-dim BC run
wants, and it is a THIRTIETH of the size.

⚠ ROWS ARE COPIED ONE DEMO AT A TIME, NOT ONE FILE AT A TIME. A whole task's
images are 438 MB and a whole suite's 4.4 GB; reading per demo keeps the peak at
one demo's worth (~9 MB) and the store is the only thing that scales.

## ⚠ THE TASK TEXT COMES FROM OUR `.task` FILE, NOT THEIR ATTRIBUTE

Their instruction is in a `problem_info` JSON on an HDF5 ATTRIBUTE, and `io/hdf5`
has no `H5A` binding. It does not need one: the text a policy is conditioned on
must be the `language=` line the goal was written against, and taking it from the
`.task` file means the store and the eval agree by construction. If they ever
disagree, that is a finding about the import in `libero_import.mojo` and belongs
there, not silently split across two spellings of the instruction.
"""

from std.memory.alloc import unsafe_alloc
from std.os import makedirs
from std.os.path import dirname, exists

from noeira.io.hdf5.reader import H5File
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStoreWriter
from noeira.envs.libero.state_remap import load_state_remap, StateRemap


comptime CAM_H: Int = 128
comptime CAM_W: Int = 128
comptime N_CAMS: Int = 2
comptime CAM_ELEMS: Int = 3 * CAM_H * CAM_W
comptime ACTION_DIM: Int = 7
comptime JOINT_DIM: Int = 7
comptime GRIPPER_DIM: Int = 2
comptime QPOS_PROPRIO: Int = JOINT_DIM + GRIPPER_DIM
"""`joint_states` ++ `gripper_states`: the first nine words of `qpos`,
robosuite's `low_dim` modality. The task one-hot follows; the column's width
is `QPOS_WORDS + n_tasks` (the differences sit between) and is read off the
store's manifest."""
comptime QPOS_DQ: Int = QPOS_PROPRIO
"""The one-step difference of the nine proprio words, zero on an episode's
first row; computed in float32 from the float32 words so a gate can compare
it exactly."""
comptime QPOS_WORDS: Int = QPOS_PROPRIO + QPOS_DQ
"""Proprio ++ difference — everything before the task one-hot."""
comptime COL_ACTION: StaticString = "action"
comptime COL_STATE: StaticString = "state"
comptime COL_JOINTS: StaticString = "joint_states"
comptime COL_GRIPPER: StaticString = "gripper_states"
comptime COL_QPOS: StaticString = "qpos"
comptime COL_IMAGES: StaticString = "images"
comptime COL_TASK: StaticString = "task_index"


struct DemoImportReport(Movable & Deinitable):
    var n_tasks: Int
    var n_episodes: Int
    var n_rows: Int
    var with_images: Bool
    var eps_per_task: List[Int]
    """How many episodes each task contributed, in `task_index` order.

    ⚠ NOT DERIVABLE FROM THE STORE. Episodes are laid out task by task, so a
    consumer that wants "episode `k` of task `t`" needs the per-task counts —
    and with `max_demos` they are not all 50. The gate in
    `examples/libero/libero_demo_import.mojo` walks a dump of FIFTY demos against
    a store that may hold two, and without this it reads episode 5 of the store
    (which belongs to task 2) as demo 5 of task 0."""

    def __init__(
        out self, n_tasks: Int, n_episodes: Int, n_rows: Int, im: Bool,
        var eps_per_task: List[Int],
    ):
        self.n_tasks = n_tasks
        self.n_episodes = n_episodes
        self.n_rows = n_rows
        self.with_images = im
        self.eps_per_task = eps_per_task^

    def __init__(out self, *, deinit move: Self):
        self.n_tasks = move.n_tasks
        self.n_episodes = move.n_episodes
        self.n_rows = move.n_rows
        self.with_images = move.with_images
        self.eps_per_task = move.eps_per_task^


def _dset_path(demo: Int, leaf: String) -> String:
    return String("data/demo_") + String(demo) + "/" + leaf


def _count_demos(f: H5File) raises -> Int:
    """How many `data/demo_<i>` groups the file has, by probing for `actions`.

    ⚠ PROBED, NOT READ FROM `num_demos`. That count is an HDF5 ATTRIBUTE and
    `io/hdf5` has no `H5A`; probing also refuses a file whose groups are not
    `0..n-1` contiguous, which is the case a `num_demos` of 50 would hide.
    """
    var n = 0
    while f.has_dataset(_dset_path(n, String("actions"))):
        n += 1
    return n


def import_libero_demos(
    suite_dir: String,
    family: String,
    out_path: String,
    task_names: List[String],
    task_text: List[String],
    demo_files: List[String],
    images: Bool = True,
    max_demos: Int = 0,
    verbose: Bool = True,
) raises -> DemoImportReport:
    """Every demo of every task in `demo_files` into one store at `out_path`.

    `task_names[i]` / `task_text[i]` / `demo_files[i]` describe task `i`, and
    `i` is the `task_index` written per row. `max_demos > 0` caps the demos per
    task, for a smoke run.
    """
    if len(task_names) != len(demo_files) or len(task_text) != len(demo_files):
        raise Error(
            "libero demos: " + String(len(demo_files)) + " files, "
            + String(len(task_names)) + " names and " + String(len(task_text))
            + " instructions — one of each per task"
        )
    if len(demo_files) == 0:
        raise Error(
            "libero demos: no files. An empty store would be written, close"
            " cleanly, and read as a dataset with nothing in it."
        )
    var remap = load_state_remap(family)
    var state_dim = remap.nq + remap.nv
    var row_words = remap.row_words()

    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String(COL_ACTION), DType.float32, ACTION_DIM))
    cols.append(ColumnSpec(String(COL_STATE), DType.float64, state_dim))
    cols.append(ColumnSpec(String(COL_JOINTS), DType.float32, JOINT_DIM))
    cols.append(ColumnSpec(String(COL_GRIPPER), DType.float32, GRIPPER_DIM))
    var qpos_dim = QPOS_WORDS + len(task_names)
    cols.append(ColumnSpec(String(COL_QPOS), DType.float32, qpos_dim))
    cols.append(ColumnSpec(String(COL_TASK), DType.int32, 1))
    if images:
        cols.append(ColumnSpec(String(COL_IMAGES), DType.uint8, N_CAMS * CAM_ELEMS))

    var dd = dirname(out_path)
    if dd.byte_length() > 0:
        makedirs(dd, exist_ok=True)
    var w = TrajectoryStoreWriter(
        String(out_path), cols^,
        env_id=String("libero_demos:") + family,
        seed=0,
        source_commit=String("LIBERO-v1 demonstrations, states remapped by ")
            + "noeira/envs/libero/tables/state_remap_" + family + ".kv",
    )
    for i in range(len(task_names)):
        w.add_task(i, String(task_text[i]))

    var total_rows = 0
    var total_eps = 0
    var eps_per_task = List[Int]()
    for ti in range(len(demo_files)):
        var path = String(demo_files[ti])
        if not exists(path):
            raise Error("libero demos: no file at " + path)
        var f = H5File(path)
        var n_demos = _count_demos(f)
        if n_demos == 0:
            raise Error(
                "libero demos: " + path + " has no `data/demo_0/actions`. A"
                " store written from it would have this task's rows missing and"
                " its `task=` entry present."
            )
        if max_demos > 0 and n_demos > max_demos:
            n_demos = max_demos

        var task_rows = 0
        for di in range(n_demos):
            var d_act = f.open_dataset(_dset_path(di, String("actions")))
            var d_st = f.open_dataset(_dset_path(di, String("states")))
            var d_jn = f.open_dataset(_dset_path(di, String("obs/joint_states")))
            var d_gr = f.open_dataset(_dset_path(di, String("obs/gripper_states")))
            var T = Int(d_act.dims[0])
            if T == 0:
                raise Error(path + " demo_" + String(di) + " has zero rows")
            if d_act.ndim() != 2 or Int(d_act.dims[1]) != ACTION_DIM:
                raise Error(
                    path + " demo_" + String(di) + ": actions are "
                    + String(Int(d_act.dims[1])) + " wide, not "
                    + String(ACTION_DIM)
                )
            if Int(d_st.dims[0]) != T or Int(d_st.dims[1]) != row_words:
                raise Error(
                    path + " demo_" + String(di) + ": states are "
                    + String(Int(d_st.dims[0])) + "x"
                    + String(Int(d_st.dims[1])) + ", expected " + String(T)
                    + "x" + String(row_words) + " (1 + nq " + String(remap.nq)
                    + " + nv " + String(remap.nv) + "). The remap table is for '"
                    + family + "'; this recording is of a different scene."
                )
            if Int(d_jn.dims[0]) != T or Int(d_jn.dims[1]) != JOINT_DIM:
                raise Error(path + " demo_" + String(di) + ": joint_states shape")
            if Int(d_gr.dims[0]) != T or Int(d_gr.dims[1]) != GRIPPER_DIM:
                raise Error(path + " demo_" + String(di) + ": gripper_states shape")

            # ── the float columns ──────────────────────────────────────────
            var raw_a = unsafe_alloc[Scalar[DType.float64]](
                T * ACTION_DIM
            ).as_unsafe_any_origin()
            var raw_s = unsafe_alloc[Scalar[DType.float64]](
                T * row_words
            ).as_unsafe_any_origin()
            var raw_j = unsafe_alloc[Scalar[DType.float64]](
                T * JOINT_DIM
            ).as_unsafe_any_origin()
            var raw_g = unsafe_alloc[Scalar[DType.float64]](
                T * GRIPPER_DIM
            ).as_unsafe_any_origin()
            d_act.read_all[DType.float64](raw_a)
            d_st.read_all[DType.float64](raw_s)
            d_jn.read_all[DType.float64](raw_j)
            d_gr.read_all[DType.float64](raw_g)

            var ab = unsafe_alloc[Scalar[DType.float32]](
                T * ACTION_DIM
            ).as_unsafe_any_origin()
            var jb = unsafe_alloc[Scalar[DType.float32]](
                T * JOINT_DIM
            ).as_unsafe_any_origin()
            var gb = unsafe_alloc[Scalar[DType.float32]](
                T * GRIPPER_DIM
            ).as_unsafe_any_origin()
            var qb = unsafe_alloc[Scalar[DType.float32]](
                T * qpos_dim
            ).as_unsafe_any_origin()
            var sb = unsafe_alloc[Scalar[DType.float64]](
                T * state_dim
            ).as_unsafe_any_origin()
            var tb = unsafe_alloc[Scalar[DType.int32]](T).as_unsafe_any_origin()
            # ⚠ HOISTED OUT OF THE ROW LOOP. A task is ~4 500 rows and these
            # three would otherwise be allocated and freed per row for a
            # permutation that overwrites every element it reads.
            var row = List[Float64](length=row_words, fill=0.0)
            var qo = List[Float64](length=remap.nq, fill=0.0)
            var vo = List[Float64](length=remap.nv, fill=0.0)
            for r in range(T):
                for k in range(ACTION_DIM):
                    ab[unsafe_offset = r * ACTION_DIM + k] = Scalar[
                        DType.float32
                    ](raw_a[unsafe_offset = r * ACTION_DIM + k])
                for k in range(JOINT_DIM):
                    jb[unsafe_offset = r * JOINT_DIM + k] = Scalar[
                        DType.float32
                    ](raw_j[unsafe_offset = r * JOINT_DIM + k])
                for k in range(GRIPPER_DIM):
                    gb[unsafe_offset = r * GRIPPER_DIM + k] = Scalar[
                        DType.float32
                    ](raw_g[unsafe_offset = r * GRIPPER_DIM + k])
                for k in range(JOINT_DIM):
                    qb[unsafe_offset = r * qpos_dim + k] = Scalar[
                        DType.float32
                    ](raw_j[unsafe_offset = r * JOINT_DIM + k])
                for k in range(GRIPPER_DIM):
                    qb[unsafe_offset = r * qpos_dim + JOINT_DIM + k] = Scalar[
                        DType.float32
                    ](raw_g[unsafe_offset = r * GRIPPER_DIM + k])
                # the one-step difference, float32 from the float32 words
                for k in range(QPOS_PROPRIO):
                    var cur = qb[unsafe_offset = r * qpos_dim + k]
                    var prev = (
                        qb[unsafe_offset = (r - 1) * qpos_dim + k]
                        if r > 0 else cur
                    )
                    qb[unsafe_offset = r * qpos_dim + QPOS_PROPRIO + k] = cur - prev
                for k in range(len(task_names)):
                    qb[unsafe_offset = r * qpos_dim + QPOS_WORDS + k] = Scalar[
                        DType.float32
                    ](1.0 if k == ti else 0.0)
                for k in range(row_words):
                    row[k] = Float64(raw_s[unsafe_offset = r * row_words + k])
                remap.convert_into(row, qo, vo)
                for k in range(remap.nq):
                    sb[unsafe_offset = r * state_dim + k] = Scalar[
                        DType.float64
                    ](qo[k])
                for k in range(remap.nv):
                    sb[unsafe_offset = r * state_dim + remap.nq + k] = Scalar[
                        DType.float64
                    ](vo[k])
                tb[unsafe_offset=r] = Int32(ti)

            w.append[DType.float32](String(COL_ACTION), ab, T)
            w.append[DType.float64](String(COL_STATE), sb, T)
            w.append[DType.float32](String(COL_JOINTS), jb, T)
            w.append[DType.float32](String(COL_GRIPPER), gb, T)
            w.append[DType.float32](String(COL_QPOS), qb, T)
            w.append[DType.int32](String(COL_TASK), tb, T)

            # ── the images ─────────────────────────────────────────────────
            if images:
                var d_av = f.open_dataset(
                    _dset_path(di, String("obs/agentview_rgb"))
                )
                var d_eh = f.open_dataset(
                    _dset_path(di, String("obs/eye_in_hand_rgb"))
                )
                if (
                    d_av.ndim() != 4 or Int(d_av.dims[0]) != T
                    or Int(d_av.dims[1]) != CAM_H or Int(d_av.dims[2]) != CAM_W
                    or Int(d_av.dims[3]) != 3
                ):
                    raise Error(
                        path + " demo_" + String(di) + ": agentview_rgb is not "
                        + String(T) + "x" + String(CAM_H) + "x" + String(CAM_W)
                        + "x3"
                    )
                if (
                    d_eh.ndim() != 4 or Int(d_eh.dims[0]) != T
                    or Int(d_eh.dims[1]) != CAM_H or Int(d_eh.dims[2]) != CAM_W
                    or Int(d_eh.dims[3]) != 3
                ):
                    raise Error(
                        path + " demo_" + String(di) + ": eye_in_hand_rgb is not "
                        + String(T) + "x" + String(CAM_H) + "x" + String(CAM_W)
                        + "x3 — the two cameras must agree, the store packs them"
                        " into one row"
                    )
                var im = unsafe_alloc[Scalar[DType.uint8]](
                    T * N_CAMS * CAM_ELEMS
                ).as_unsafe_any_origin()
                var one = unsafe_alloc[Scalar[DType.uint8]](
                    T * CAM_ELEMS
                ).as_unsafe_any_origin()
                # ⚠ HWC ON DISK, CHW IN THE STORE — the transpose is here and
                # not in the loader. `lerobot_v3_to_store.py` writes CHW and
                # every consumer in this tree expects it; a store that carried
                # HWC would train a net whose first conv sees three rows of one
                # image row instead of three planes.
                # ⚠⚠ AND BOTTOM ROW FIRST ON DISK, TOP ROW FIRST IN THE STORE
                # — see the header. `sy` is the source row for store row `y`.
                for cam in range(N_CAMS):
                    if cam == 0:
                        d_av.read_all[DType.uint8](one)
                    else:
                        d_eh.read_all[DType.uint8](one)
                    for r in range(T):
                        var src = r * CAM_ELEMS
                        var dst = r * N_CAMS * CAM_ELEMS + cam * CAM_ELEMS
                        for y in range(CAM_H):
                            var sy = CAM_H - 1 - y
                            for x in range(CAM_W):
                                var s = src + (sy * CAM_W + x) * 3
                                for c in range(3):
                                    im[
                                        unsafe_offset = dst
                                        + c * CAM_H * CAM_W + y * CAM_W + x
                                    ] = one[unsafe_offset = s + c]
                w.append[DType.uint8](String(COL_IMAGES), im, T)

            w.end_episode()
            total_eps += 1
            total_rows += T
            task_rows += T
        eps_per_task.append(n_demos)
        if verbose:
            print(
                "  " + String(task_names[ti]) + ": " + String(n_demos)
                + " demos, " + String(task_rows) + " rows"
            )
    w.close()
    return DemoImportReport(
        len(demo_files), total_eps, total_rows, images, eps_per_task^
    )
