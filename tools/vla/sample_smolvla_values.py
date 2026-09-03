#!/usr/bin/env python3
"""Sample individual tensor values from a safetensors file, by hand.

    pixi run python tools/vla/sample_smolvla_values.py --out tools/vla/smolvla_samples.tsv

The name-map gates compare NAMES and SHAPES. Those cannot catch a wrong
`TN_TRANSPOSE` flag on a SQUARE matrix: `[320, 320]` transposed is still
`[320, 320]`, the element count matches, and the load succeeds with the weights
silently transposed. Only a value at a known (row, col) settles it.

So this reads chosen values straight out of the file — seek to
`8 + header_len + data_offset`, decode BF16 as the top half of an f32 (a shift,
exactly) — with no torch, no safetensors library, and crucially no shared code
with the Mojo reader it is used to check.

Output, one sample per line:

    <name>\t<dtype>\t<shape>\t<row>\t<col>\t<value as float32 hex>\t<value>
"""

import argparse
import json
import struct
import sys

# (name, [(row, col), ...]) — chosen to cover the cases shapes cannot separate.
SAMPLES = [
    # square + F32: a wrong transpose flag is INVISIBLE to any size check.
    ("model.vlm_with_expert.lm_expert.layers.1.self_attn.k_proj.weight",
     [(0, 1), (1, 0), (5, 17), (319, 0), (0, 319)]),
    # non-square BF16, transposed on load
    ("model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.k_proj.weight",
     [(0, 1), (1, 0), (319, 959), (7, 123)]),
    # BF16 square-ish, transposed
    ("model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
     [(0, 1), (1, 0), (767, 0), (0, 767)]),
    # 4-D conv, NOT transposed
    ("model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight",
     [(0, 0), (0, 5), (17, 3)]),
    # F32 with a real bias beside it
    ("model.state_proj.weight", [(0, 0), (1, 2), (959, 31)]),
]

ITEMSIZE = {"F32": 4, "BF16": 2, "F16": 2, "F64": 8}


def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def value_at(f, base, entry, flat):
    dt = entry["dtype"]
    isz = ITEMSIZE[dt]
    f.seek(base + entry["data_offsets"][0] + flat * isz)
    raw = f.read(isz)
    if dt == "F32":
        return struct.unpack("<f", raw)[0]
    if dt == "BF16":
        # bfloat16 IS the top half of an f32 — widening is a shift, exactly.
        bits = struct.unpack("<H", raw)[0] << 16
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    if dt == "F16":
        return struct.unpack("<e", raw)[0]
    if dt == "F64":
        return struct.unpack("<d", raw)[0]
    raise SystemExit(f"unhandled dtype {dt}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="path to model.safetensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = a.file
    if path is None:
        import os
        path = os.path.expanduser(
            "~/.cache/mojo_rl/hub/lerobot__smolvla_base/main/model.safetensors"
        )

    lines = []
    with open(path, "rb") as f:
        hdr, base = read_header(f)
        for name, coords in SAMPLES:
            if name not in hdr:
                raise SystemExit(f"{name!r} not in the file")
            e = hdr[name]
            shape = e["shape"]
            # flat index of (row, col) in the file's own row-major layout;
            # for rank > 2 the trailing dims are folded into the column.
            cols = 1
            for d in shape[1:]:
                cols *= d
            for (r, c) in coords:
                if r >= shape[0] or c >= cols:
                    raise SystemExit(f"{name}: ({r},{c}) out of {shape}")
                v = value_at(f, base, e, r * cols + c)
                bits = struct.unpack("<I", struct.pack("<f", v))[0]
                lines.append(
                    f"{name}\t{e['dtype']}\t{','.join(map(str, shape))}"
                    f"\t{r}\t{c}\t{bits:08x}\t{v!r}"
                )

    with open(a.out, "w") as f:
        f.write("# independent value samples read straight from the file\n")
        f.write("# <name>\\t<dtype>\\t<shape>\\t<row>\\t<col>\\t<f32 hex>\\t<value>\n")
        f.write("\n".join(lines) + "\n")
    print(f"{a.out}: {len(lines)} samples from {len(SAMPLES)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
