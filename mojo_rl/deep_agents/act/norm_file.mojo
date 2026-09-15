# +--------------------------------------------------------------------------+ #
# | ACT normalization — the few numbers a deployed policy needs from its store
# +--------------------------------------------------------------------------+ #
"""`norm.json`: what a deployment reads from the training store, without the store.

    <run>/checkpoints/norm.json          written by the trainer at startup
    policies/<role>.norm.json            copied there by `project-promote`

⚠⚠ THE STORE IS PART OF THE POLICY, AND IT IS 9 GB. A checkpoint carries no
normalization statistics: `ACTDataset` recomputes them from the store's `qpos`
and `action` columns, and the deploy script used to need the whole `.h5` on the
robot machine for a few kilobytes of numbers — copying 8.9 GB off a rented GPU,
or re-importing 50 episodes on a laptop. This file is those numbers:

| field                          | used by the deployment for                     |
|--------------------------------|------------------------------------------------|
| `qpos_mean` / `qpos_std`       | normalizing the observed joint state           |
| `action_mean` / `action_std`   | denormalizing the policy's output              |
| `action_min` / `action_max`    | the action box (clamp, before its 5% margin)   |
| `qpos_min` / `qpos_max`        | "the arm starts outside what it was shown"     |
| `cameras`                      | which camera fills which slot, in SLOT order   |

⚠ THE STATISTICS ARE COPIED FROM `ACTDataset`, NOT RECOMPUTED HERE. One
implementation of the normalization exists (`data.mojo:_moments`); a second
one here would be the rule written twice, and the failure would be a policy
fed slightly shifted inputs on a real arm.

⚠ `Float32` VALUES ROUND-TRIP EXACTLY. They are written as `Float64` decimals,
and a `Float32` survives a decimal round trip through `Float64` even when the
parse is one `Float64` ULP off — `tests/deep_agents/act/test_act_norm_file.mojo`
asserts bit-equality rather than trusting that argument.
"""

from mojo_rl.io.fileio import write_text_atomic
from mojo_rl.io.json import JsonDoc, JsonWriter, load_json
from mojo_rl.nn.constants import DT


comptime NORM_SCHEMA_VERSION = 1


struct ACTNorm(Movable):
    var qpos_mean: List[Scalar[DT]]
    var qpos_std: List[Scalar[DT]]
    var action_mean: List[Scalar[DT]]
    var action_std: List[Scalar[DT]]
    var qpos_min: List[Float64]
    var qpos_max: List[Float64]
    var action_min: List[Float64]
    var action_max: List[Float64]
    var cameras: List[String]
    """Slot order: alphabetical by feature key, as the importer assigns them.
    Empty when the store's source did not record them."""
    var img_h: Int
    var img_w: Int
    var n_rows: Int
    var n_episodes: Int
    var store: String
    """Provenance only: the path of the store these numbers came from."""

    def __init__(out self):
        self.qpos_mean = List[Scalar[DT]]()
        self.qpos_std = List[Scalar[DT]]()
        self.action_mean = List[Scalar[DT]]()
        self.action_std = List[Scalar[DT]]()
        self.qpos_min = List[Float64]()
        self.qpos_max = List[Float64]()
        self.action_min = List[Float64]()
        self.action_max = List[Float64]()
        self.cameras = List[String]()
        self.img_h = 0
        self.img_w = 0
        self.n_rows = 0
        self.n_episodes = 0
        self.store = String("")

    def __init__(out self, *, deinit move: Self):
        self.qpos_mean = move.qpos_mean^
        self.qpos_std = move.qpos_std^
        self.action_mean = move.action_mean^
        self.action_std = move.action_std^
        self.qpos_min = move.qpos_min^
        self.qpos_max = move.qpos_max^
        self.action_min = move.action_min^
        self.action_max = move.action_max^
        self.cameras = move.cameras^
        self.img_h = move.img_h
        self.img_w = move.img_w
        self.n_rows = move.n_rows
        self.n_episodes = move.n_episodes
        self.store = move.store^

    def qpos_dim(self) -> Int:
        return len(self.qpos_mean)

    def action_dim(self) -> Int:
        return len(self.action_mean)

    def save(self, path: String) raises:
        var w = JsonWriter()
        w.begin_object()
        w.member(String("schema_version"), NORM_SCHEMA_VERSION)
        w.member(String("store"), self.store)
        w.member(String("n_rows"), self.n_rows)
        w.member(String("n_episodes"), self.n_episodes)
        w.member(String("img_h"), self.img_h)
        w.member(String("img_w"), self.img_w)
        w.key(String("cameras"))
        w.begin_array()
        for c in self.cameras:
            w.string(c)
        w.end_array()
        _write_f32(w, String("qpos_mean"), self.qpos_mean)
        _write_f32(w, String("qpos_std"), self.qpos_std)
        _write_f32(w, String("action_mean"), self.action_mean)
        _write_f32(w, String("action_std"), self.action_std)
        _write_f64(w, String("qpos_min"), self.qpos_min)
        _write_f64(w, String("qpos_max"), self.qpos_max)
        _write_f64(w, String("action_min"), self.action_min)
        _write_f64(w, String("action_max"), self.action_max)
        w.end_object()
        write_text_atomic(path, w.done())

    @staticmethod
    def load(path: String, qpos_dim: Int, action_dim: Int) raises -> Self:
        """Read and CHECK against the model the deployment was built for.

        ⚠ A width mismatch is refused, not truncated: a 6-joint norm file fed
        to a 7-joint build would silently normalize with the wrong columns.
        """
        var doc = load_json(path)
        var r = doc.root()
        var out = Self()
        var v = doc.integer(doc.field(r, String("schema_version")))
        if v != NORM_SCHEMA_VERSION:
            raise Error(path + ": schema_version " + String(v) + ", expected " + String(NORM_SCHEMA_VERSION))
        out.store = doc.string(doc.field(r, String("store")))
        out.n_rows = doc.integer(doc.field(r, String("n_rows")))
        out.n_episodes = doc.integer(doc.field(r, String("n_episodes")))
        out.img_h = doc.integer(doc.field(r, String("img_h")))
        out.img_w = doc.integer(doc.field(r, String("img_w")))
        var cams = doc.field(r, String("cameras"))
        for i in range(doc.size(cams)):
            out.cameras.append(doc.string(doc.at(cams, i)))
        out.qpos_mean = _read_f32(doc, r, String("qpos_mean"), qpos_dim, path)
        out.qpos_std = _read_f32(doc, r, String("qpos_std"), qpos_dim, path)
        out.action_mean = _read_f32(doc, r, String("action_mean"), action_dim, path)
        out.action_std = _read_f32(doc, r, String("action_std"), action_dim, path)
        out.qpos_min = _read_f64(doc, r, String("qpos_min"), qpos_dim, path)
        out.qpos_max = _read_f64(doc, r, String("qpos_max"), qpos_dim, path)
        out.action_min = _read_f64(doc, r, String("action_min"), action_dim, path)
        out.action_max = _read_f64(doc, r, String("action_max"), action_dim, path)
        return out^


def act_norm_from(
    ref qpos_raw: List[Scalar[DT]],
    ref action_raw: List[Scalar[DT]],
    n_rows: Int,
    n_episodes: Int,
    ref qpos_mean: List[Scalar[DT]],
    ref qpos_std: List[Scalar[DT]],
    ref action_mean: List[Scalar[DT]],
    ref action_std: List[Scalar[DT]],
    ref cameras: List[String],
    img_h: Int,
    img_w: Int,
    store: String,
) raises -> ACTNorm:
    """Build the file's contents from an `ACTDataset`'s own fields.

    ⚠ The boxes are min/max over EVERY row, exactly the loop the deploy script
    ran over the store — no margin here; the deployment applies its own.
    """
    var out = ACTNorm()
    out.qpos_mean = qpos_mean.copy()
    out.qpos_std = qpos_std.copy()
    out.action_mean = action_mean.copy()
    out.action_std = action_std.copy()
    var qd = len(qpos_mean)
    var ad = len(action_mean)
    out.qpos_min = List[Float64](length=qd, fill=1.0e18)
    out.qpos_max = List[Float64](length=qd, fill=-1.0e18)
    out.action_min = List[Float64](length=ad, fill=1.0e18)
    out.action_max = List[Float64](length=ad, fill=-1.0e18)
    for r in range(n_rows):
        for j in range(ad):
            var v = Float64(action_raw[r * ad + j])
            if v < out.action_min[j]:
                out.action_min[j] = v
            if v > out.action_max[j]:
                out.action_max[j] = v
        for j in range(qd):
            var w = Float64(qpos_raw[r * qd + j])
            if w < out.qpos_min[j]:
                out.qpos_min[j] = w
            if w > out.qpos_max[j]:
                out.qpos_max[j] = w
    out.cameras = cameras.copy()
    out.img_h = img_h
    out.img_w = img_w
    out.n_rows = n_rows
    out.n_episodes = n_episodes
    out.store = store
    return out^


def _write_f32(mut w: JsonWriter, name: String, ref xs: List[Scalar[DT]]) raises:
    w.key(name)
    w.begin_array()
    for x in xs:
        w.number(Float64(x))
    w.end_array()


def _write_f64(mut w: JsonWriter, name: String, ref xs: List[Float64]) raises:
    w.key(name)
    w.begin_array()
    for x in xs:
        w.number(x)
    w.end_array()


def _read_f32(ref doc: JsonDoc, r: Int, name: String, dim: Int, path: String) raises -> List[Scalar[DT]]:
    var node = doc.field(r, name)
    if node < 0 or doc.size(node) != dim:
        raise Error(path + ": '" + name + "' must hold " + String(dim) + " values")
    var out = List[Scalar[DT]]()
    for i in range(dim):
        out.append(Scalar[DT](doc.number(doc.at(node, i))))
    return out^


def _read_f64(ref doc: JsonDoc, r: Int, name: String, dim: Int, path: String) raises -> List[Float64]:
    var node = doc.field(r, name)
    if node < 0 or doc.size(node) != dim:
        raise Error(path + ": '" + name + "' must hold " + String(dim) + " values")
    var out = List[Float64]()
    for i in range(dim):
        out.append(doc.number(doc.at(node, i)))
    return out^
