# +--------------------------------------------------------------------------+ #
# | Write a policy's norm.json from the store it was trained on
# +--------------------------------------------------------------------------+ #
"""The few kilobytes a deployment needs from a 9 GB training store.

    # beside a run's checkpoints — what `project-promote` then carries along
    pixi run mojo run -I . examples/so101/act_so101_export_norm.mojo \\
        --store /workspace/cube-in-bowl_240x320.h5 --run 2026-09-15_act-so101_ab12cd34 \\
        --project so101-tower --dataset cube-in-bowl

    # anywhere
    ... --store S --out /tmp/norm.json

Runs in seconds and loads NO images (`max_image_bytes=0`): only the `qpos` and
`action` columns, which is where the statistics come from.

Camera slot names, first found wins:
  1. `--project P --dataset D`  -> `projects/P/datasets/D/meta/info.json`
  2. the store's side file `<store>.json`
Neither: the file records none, and the deployment prints slot numbers.

⚠ A run trained before the trainer wrote this file itself needs this command
once, on the box that has the store. Runs trained after do not.
"""

from std.os.path import exists
from std.sys import argv

from noeira.core.project import project_dataset_dir, projects_root
from noeira.data.lerobot import LeRobotInfo
from noeira.deep_agents.act.config import (
    SO101_ADIM, SO101_IMG_H, SO101_IMG_W, SO101_N_CAM, SO101_QPOS,
)
from noeira.deep_agents.act.data import ACTDataset
from noeira.deep_agents.act.norm_file import act_norm_from
from noeira.io.json import load_json
from noeira.io.proc import quote_arg, run_capture


comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W


def _opt(name: String) raises -> String:
    var av = argv()
    for i in range(1, len(av) - 1):
        if String(av[i]) == name:
            return String(av[i + 1])
    return String("")


def store_camera_names(store: String, project: String, dataset: String) raises -> List[String]:
    """Slot-ordered camera keys. See the module header for the order of sources."""
    var out = List[String]()
    if project.byte_length() > 0 and dataset.byte_length() > 0:
        var info = LeRobotInfo(project_dataset_dir(project, dataset))
        for c in info.cameras:
            out.append(c)
        return out^
    if store.endswith(".h5"):
        var side = String(store[byte=0 : store.byte_length() - 3]) + ".json"
        if exists(side):
            var doc = load_json(side)
            var cams = doc.field(doc.root(), String("cameras"))
            if cams >= 0:
                for i in range(doc.size(cams)):
                    out.append(doc.string(doc.at(cams, i)))
    return out^


def find_run_dir(rid: String) raises -> String:
    var root = projects_root()
    if exists(String("runs/") + rid + "/run.kv"):
        return String("runs/") + rid
    var txt = run_capture(String("ls -1 ") + quote_arg(root) + " 2>/dev/null", 1 << 16)
    for line in txt.split("\n"):
        var n = String(line.strip())
        if n.byte_length() > 0 and exists(root + "/" + n + "/runs/" + rid + "/run.kv"):
            return root + "/" + n + "/runs/" + rid
    raise Error("no run '" + rid + "' under runs/ or " + root + "/*/runs/")


def main() raises:
    var store = _opt(String("--store"))
    var out = _opt(String("--out"))
    var rid = _opt(String("--run"))
    var project = _opt(String("--project"))
    var dataset = _opt(String("--dataset"))
    if store.byte_length() == 0 or (out.byte_length() == 0) == (rid.byte_length() == 0):
        raise Error(
            "usage: act_so101_export_norm --store S (--out P | --run RUN_ID)"
            " [--project P --dataset D]"
        )
    if not exists(store):
        raise Error("no store at " + store)
    if rid.byte_length() > 0:
        out = find_run_dir(rid) + "/checkpoints/norm.json"

    var ds = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](store.copy(), seed=7, max_image_bytes=0)
    var cams = store_camera_names(store, project, dataset)
    if len(cams) != 0 and len(cams) != N_CAM:
        raise Error(String(len(cams)) + " camera names found for a " + String(N_CAM) + "-camera build")
    var norm = act_norm_from(
        ds.qpos_raw, ds.action_raw, ds.n_rows(), ds.n_episodes(),
        ds.qpos_mean, ds.qpos_std, ds.action_mean, ds.action_std,
        cams, IMG_H, IMG_W, store,
    )
    norm.save(out)
    print("wrote " + out)
    print("  " + String(norm.n_rows) + " rows, " + String(norm.n_episodes) + " episodes, " + String(IMG_W) + "x" + String(IMG_H))
    for i in range(len(cams)):
        print("  camera slot " + String(i) + " = " + cams[i])
    if len(cams) == 0:
        print("  ⚠ no camera names found — pass --project/--dataset; the deployment will print slot numbers")
