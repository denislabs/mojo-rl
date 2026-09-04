#!/usr/bin/env python3
"""Pre-tokenise a LeRobot dataset's task strings into a checked-in table.

    pixi run -e act-ref python tools/vla/dump_smolvla_tasks.py \
        --dataset ~/.cache/huggingface/lerobot/DenisLabs/record-test_20260828_092736 \
        --out tools/vla/smolvla_tasks_<name>.tsv

`SmolVLAPrefixEmbed.run` takes PRE-TOKENISED ids, and deliberately so: SmolVLA's
`tokenizer_max_length` is 48 and `pad_language_to` is "longest", so for a fixed
set of instructions the ids are a constant. Shipping a tokeniser into a 50 Hz
Mojo control loop to recompute a constant would be the wrong trade -- and it
would put a 400 MB Python dependency back inside a training binary that has just
been cleared of one.

⚠ **THE TRAILING NEWLINE IS A PIPELINE STEP, NOT PART OF THE STRING.**
`processor_smolvla.py` runs `NewLineTaskProcessorStep()` BEFORE the tokeniser:
the task becomes `"Grab the green cube\\n"`. Without it you get one fewer token,
which shifts the state token, `P`, and every mask derived from `P`. Nothing
raises; the policy just attends to the wrong thing.

⚠ The tokeniser is `config.vlm_model_name`'s -- the SmolVLM2 backbone's -- not
SmolVLA's own repo, which ships no tokeniser files.

Output columns: `task_index<TAB>n_tokens<TAB>id,id,...<TAB>task`
"""

import argparse
import os
import sys

VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MAX_LENGTH = 48  # config.tokenizer_max_length


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="a LeRobot v3 dataset root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=VLM_MODEL)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    path = os.path.join(os.path.expanduser(a.dataset), "meta", "tasks.parquet")
    tbl = pq.read_table(path).to_pydict()
    # `task_index` is the column the data rows join on; `task` is the text.
    pairs = sorted(zip(tbl["task_index"], tbl["task"]))
    print(f"{len(pairs)} task(s) in {path}")

    tok = AutoTokenizer.from_pretrained(a.model)
    lines = []
    lengths = set()
    for idx, task in pairs:
        text = task if task.endswith("\n") else task + "\n"
        ids = tok(text, padding="longest", padding_side="right",
                  max_length=MAX_LENGTH, truncation=True)["input_ids"]
        lengths.add(len(ids))
        pieces = [tok.decode([i]) for i in ids]
        print(f"  [{idx}] {len(ids):2d} tokens  {task!r}")
        print(f"       ids    {ids}")
        print(f"       pieces {pieces}")
        lines.append(f"{idx}\t{len(ids)}\t{','.join(str(i) for i in ids)}\t{task}")

    if len(lengths) > 1:
        # ⚠ `pad_language_to: "longest"` makes N_LANG depend on the batch, while
        # our `SmolVLAPrefixEmbed[N_CAM, N_LANG]` fixes it at compile time. With
        # one length that is safe and stricter; with several it is a decision.
        print(f"\n⚠ tasks tokenise to {sorted(lengths)} tokens — differing lengths.",
              file=sys.stderr)
        print("  `pad_language_to: \"longest\"` pads to the batch maximum, so a",
              file=sys.stderr)
        print("  comptime N_LANG must be pinned to whatever training used.",
              file=sys.stderr)

    with open(a.out, "w") as f:
        f.write("# task_index\tn_tokens\tids\ttask\n")
        f.write(f"# tokenizer: {a.model}\n")
        f.write("# a trailing newline is appended to every task before tokenising\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
