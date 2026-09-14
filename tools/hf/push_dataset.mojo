# +--------------------------------------------------------------------------+ #
# | Push a recorded LeRobot v3 dataset to the Hugging Face Hub
# +--------------------------------------------------------------------------+ #
"""`record.mojo` / `record_ui.mojo` write a directory; this sends it to the Hub.

    pixi run mojo run -I . tools/hf/push_dataset.mojo \\
        --root ~/datasets/so101-tower/cube-in-bowl --repo DenisLabs/so101-tower-cube-in-bowl

    --root DIR       the dataset directory the recorder wrote (required)
    --repo ORG/NAME  the dataset repo (required)
    --message TEXT   commit message
    --public         create the repo PUBLIC. The default is private.

⚠⚠ PRIVATE UNLESS `--public`. A robot recording is of someone's room, and the
Hub's own default is public. Note the Hub ignores the flag on a repo that
already exists: this never un-publishes anything.

⚠ THIS FILE DID NOT EXIST WHILE THE RECORDER'S HEADER TOLD PEOPLE TO RUN IT.
`push_lerobot_dataset` was a library function only, and the first person to
follow the recording instructions end to end found the command missing.

`HF_TOKEN` is read from the environment, then `.env`.
"""

from std.os import getenv
from std.os.path import exists
from std.sys import argv

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.data.lerobot_push import dataset_files, push_lerobot_dataset


def _flag(name: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return String("")


def _has(name: String) -> Bool:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            return True
    return False


def main() raises:
    var root = _flag(String("--root"))
    var repo = _flag(String("--repo"))
    if root.byte_length() == 0 or repo.byte_length() == 0:
        raise Error(
            "usage: push_dataset --root DIR --repo ORG/NAME [--message TEXT]"
            " [--public]"
        )
    if root.startswith("~/"):
        root = getenv("HOME") + String(root[byte=1:])
    if not exists(root + "/meta/info.json"):
        raise Error(
            root + " is not a LeRobot v3 dataset (no meta/info.json). Did the"
            " recording finish? The writer only completes the metadata when"
            " the recorder exits through `finish`."
        )

    var token = getenv("HF_TOKEN")
    if token.byte_length() == 0:
        try:
            var env = load_dotenv(String(".env"))
            if "HF_TOKEN" in env:
                token = env["HF_TOKEN"]
        except:
            pass
    if token.byte_length() == 0:
        raise Error("HF_TOKEN is not set in the environment or in .env")

    var message = _flag(String("--message"))
    if message.byte_length() == 0:
        message = String("Upload dataset with mojo-rl")
    var private = not _has(String("--public"))

    var files = dataset_files(root)
    print(
        "pushing " + String(len(files)) + " files from " + root + " to " + repo
        + (" (private)" if private else " (PUBLIC)")
    )
    var url = push_lerobot_dataset(root, repo, message, private, token)
    print("done: " + url)
