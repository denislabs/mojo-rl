#!/usr/bin/env python3
"""Dump the tensor manifest of a SmolVLA checkpoint, without downloading it.

    pixi run python tools/vla/dump_smolvla_manifest.py \
        --repo lerobot/smolvla_base --out tools/vla/smolvla_base_manifest.tsv

A safetensors file states its whole index in a header at the front: eight bytes
of little-endian length, then that many bytes of JSON. Two HTTP Range requests
therefore buy the complete name/dtype/shape listing for **73 KB** instead of the
906,712,520-byte file. The manifest is checked in so `TorchNameMap` work — and
the 500/500 coverage gate over it — needs neither the weights nor the network.

⚠ The manifest is a CHECKLIST, not the weights. It proves a map names every
published tensor with the right shape; it cannot prove the numbers arrive
correctly. That is `LoadTorchNamed.report_exact` against the real file, and then
the layer-by-layer parity gates.

Format, one line per tensor, sorted by name (`refload.mojo`'s convention plus a
dtype column, since SmolVLA is mixed BF16/F32 and the dtype is load-bearing):

    <name>\t<dtype>\t<d0,d1,...>
"""

import argparse
import json
import struct
import sys
import urllib.request

HDR_LEN_BYTES = 8


def _range_get(url: str, first: int, last: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_header(repo: str, filename: str, revision: str) -> dict:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    n = struct.unpack("<Q", _range_get(url, 0, HDR_LEN_BYTES - 1))[0]
    if not (0 < n < (1 << 30)):
        raise SystemExit(f"implausible safetensors header length {n}")
    blob = _range_get(url, HDR_LEN_BYTES, HDR_LEN_BYTES + n - 1)
    return json.loads(blob)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="lerobot/smolvla_base")
    ap.add_argument("--file", default="model.safetensors")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    hdr = fetch_header(a.repo, a.file, a.revision)
    hdr.pop("__metadata__", None)

    lines, total = [], 0
    for name in sorted(hdr):
        e = hdr[name]
        shape = e["shape"]
        n = 1
        for d in shape:
            n *= d
        total += n
        lines.append(f"{name}\t{e['dtype']}\t{','.join(str(d) for d in shape)}")

    with open(a.out, "w") as f:
        f.write(f"# {a.repo}@{a.revision} :: {a.file}\n")
        f.write(f"# {len(lines)} tensors, {total} parameters\n")
        f.write("# <name>\\t<dtype>\\t<shape>\n")
        f.write("\n".join(lines) + "\n")

    print(f"{a.out}: {len(lines)} tensors, {total:,} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
