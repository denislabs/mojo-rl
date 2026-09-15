# +--------------------------------------------------------------------------+ #
# | `dataset` — mirror a project's recordings on the platform
# +--------------------------------------------------------------------------+ #
"""Push a recording to the platform (also while it is still recording), pull it.

    pixi run dataset-push -- --project so101-tower --dataset cube-in-bowl
    pixi run dataset-push -- --project so101-tower --dataset cube-in-bowl --watch
    pixi run dataset-pull -- --project so101-tower --dataset cube-in-bowl
    pixi run dataset-list -- --project so101-tower

`--watch` re-syncs every `--every` seconds (default 60) until Ctrl-C: run it
in a second terminal during a recording session. It only READS the dataset,
so stopping it at any moment is safe.

The dataset lives at `projects/<project>/datasets/<dataset>/` on both boxes.
The mechanics — which files are held back, why metadata goes last — are in
`mojo_rl/data/dataset_sync.mojo` and `recording_files.mojo`.
"""

from std.os.path import exists
from std.sys import argv
from std.time import sleep

from mojo_rl.core.project import project_dataset_dir, projects_root
from mojo_rl.data.dataset_sync import HashCache, pull_dataset, push_dataset
from mojo_rl.data.remote import RemoteCatalog


def _flag(name: String, dflt: String = String("")) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _has(name: String) -> Bool:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            return True
    return False


def _required(name: String) raises -> String:
    var v = _flag(name)
    if v.byte_length() == 0:
        raise Error(
            "usage: dataset_cli <push|pull|list> --project P [--dataset D]"
            " [--watch] [--every S]   (" + name + " is required)"
        )
    return v^


def cmd_push() raises:
    var project = _required(String("--project"))
    var dataset = _required(String("--dataset"))
    var root = project_dataset_dir(project, dataset)
    var cat = RemoteCatalog.from_env()
    var cache = HashCache()
    var watch = _has(String("--watch"))
    var every = Int(_flag(String("--every"), String("60")))
    while True:
        var rep = push_dataset(cat, project, dataset, root, cache)
        # ⚠ THE UNCHANGED COUNT BESIDE THE UPLOADED ONE. "0 uploaded" is also
        # what a pass that found no files prints.
        print(
            "dataset " + project + "/" + dataset + ": " + String(rep.n_episodes)
            + " episodes — " + String(rep.uploaded) + " uploaded ("
            + String(rep.bytes // 1_000_000) + " MB), " + String(rep.unchanged)
            + " unchanged, " + String(rep.held)
            + " held back (still recording)"
            + (", " + String(rep.passes) + " passes: an episode ended mid-push" if rep.passes > 1 else "")
        )
        if not watch:
            break
        sleep(Float64(every))


def cmd_pull() raises:
    var project = _required(String("--project"))
    var dataset = _required(String("--dataset"))
    var dest = projects_root() + "/" + project + "/datasets/" + dataset
    if not exists(projects_root() + "/" + project + "/project.kv"):
        raise Error(
            "no project '" + project + "' on this box — pull it first:"
            " pixi run project-pull " + project
        )
    var cat = RemoteCatalog.from_env()
    var rep = pull_dataset(cat, project, dataset, dest)
    for r in rep.refused:
        print("  refused  " + r)
    print(
        "dataset " + project + "/" + dataset + " -> " + dest + ": "
        + String(rep.downloaded) + " downloaded, " + String(rep.present)
        + " already here, " + String(rep.pending) + " still uploading on the"
        " platform (not pulled), " + String(len(rep.refused)) + " refused"
    )
    if rep.pending > 0:
        print(
            "  ⚠ some files were mid-upload: run the pull again once the push"
            " has finished, before importing"
        )
    print("next:  pixi run mojo run -I . examples/so101/act_so101_import_dataset.mojo --root " + dest)


def cmd_list() raises:
    var project = _required(String("--project"))
    var cat = RemoteCatalog.from_env()
    var doc = cat.list_datasets(project)
    var root = doc.root()
    for i in range(doc.size(root)):
        var row = doc.at(root, i)
        var name = doc.string(doc.field(row, String("name")))
        var eps_node = doc.field(row, String("nEpisodes"))
        var eps = String("?")
        try:
            eps = String(doc.integer(eps_node))
        except:
            pass
        var files = doc.integer(doc.field(row, String("fileCount")))
        var ready = 0
        try:
            ready = doc.integer(doc.field(row, String("readyCount")))
        except:
            pass
        var mb = doc.integer(doc.field(row, String("sizeBytes"))) // 1_000_000
        print(
            "  " + name + " — " + eps + " episodes, " + String(ready) + "/"
            + String(files) + " files ready, " + String(mb) + " MB"
        )
    print()
    print(" ", doc.size(root), "dataset(s) in project", project)


def main() raises:
    var av = argv()
    var cmd = String(av[1]) if len(av) > 1 else String("")
    if cmd == "push":
        cmd_push()
    elif cmd == "pull":
        cmd_pull()
    elif cmd == "list":
        cmd_list()
    else:
        print("usage: dataset_cli <push|pull|list> --project P [--dataset D] [--watch] [--every S]")
