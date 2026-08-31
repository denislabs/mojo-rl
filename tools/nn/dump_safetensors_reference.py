#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | The reference safetensors implementation, on both ends of ours
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/io/safetensors.mojo` against the library everyone else uses.

    pixi run -e act-ref python tools/nn/dump_safetensors_reference.py \
        --out /tmp/st_ref                       # 1. reference WRITES
    pixi run mojo run -I . tests/io/test_safetensors_reference.mojo \
        /tmp/st_ref                             # 2. we READ it, and we WRITE
    pixi run -e act-ref python tools/nn/dump_safetensors_reference.py \
        --verify /tmp/st_ref                    # 3. reference READS ours

## Why the reference library and not a hand-rolled Python check

A Python re-implementation of the format would be written by the same person,
from the same reading of the same spec, as the Mojo one. Two parsers sharing
one wrong assumption cannot see the assumption — this repo has already shipped
exactly that failure once (two MJCF parsers, one wrong default, one gate that
passed). So both directions go through `safetensors` itself.

## The two directions are NOT the same test

`--out` proves our READER agrees with a file we did not write: real dtype
strings, the producer's own key ordering (alphabetical, not ours), padding
choices we do not make. `--verify` proves our WRITER emits something the
ecosystem accepts — a file only we can read would satisfy a round-trip gate
perfectly and be worthless.

## Values

Every float tensor is `v[i] = (i * 37 % 101) * 0.25 - 12.5`: a multiple of
0.25, so it is exact in f32, f64 AND f16, and a widening bug cannot hide
behind rounding. Expected values are exchanged as f32 BIT PATTERNS
(`expected.txt`), so the gate compares integers and no float text is parsed on
either side.

⚠ `expected.txt` is SPACE-separated, and the empty fields (a rank-0 shape, an
integer tensor's absent values) are written `-`. Not a style choice: Mojo's
`String.splitlines()` breaks on TAB as well as on newline, so a tab-separated
table arrives at the gate as one field per line, and the failure looks like a
malformed file rather than like a string method that does not do what its name
says.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def values(n: int) -> np.ndarray:
    return np.array(
        [((i * 37) % 101) * 0.25 - 12.5 for i in range(n)], dtype=np.float32
    )


def f32_bits(a: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    return ["%08x" % struct.unpack("<I", struct.pack("<f", v))[0] for v in flat]


# ── what the reference writes, for us to read ─────────────────────────────
# Deliberately awkward: a rank-0 scalar and a zero-element tensor are both
# legal and both are where an offset/shape loop divides by or indexes past
# something. `BF16` has no numpy dtype, so it is built by hand from the top
# half of the f32 bits -- which is also exactly the widening our reader does,
# stated here independently.
def build_reference(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    basic = {
        "enc.weight": values(12).reshape(3, 4),
        "enc.bias": values(4),
        "scalar": np.array(values(1)[0], dtype=np.float32),  # rank 0
        "empty": np.zeros((0,), dtype=np.float32),
        "big": values(4096).reshape(64, 64),
    }
    save_file(
        basic,
        str(out / "ref_basic.safetensors"),
        metadata={"format": "pt", "producer": "safetensors-reference"},
    )

    v = values(8)
    bf16 = (np.frombuffer(v.tobytes(), dtype="<u4") >> 16).astype("<u2")
    dtypes = {
        "t_f32": v,
        "t_f64": v.astype(np.float64),
        "t_f16": v.astype(np.float16),
        "t_bf16": bf16,          # written as U16; retyped to BF16 below
        "t_i64": np.arange(3, dtype=np.int64),
        "t_i32": np.arange(3, dtype=np.int32),
        "t_u8": np.arange(3, dtype=np.uint8),
        "t_bool": np.array([True, False, True]),
    }
    p = out / "ref_dtypes.safetensors"
    save_file(dtypes, str(p))
    _retype_u16_to_bf16(p, "t_bf16")

    def row(name, dtype, shape, bits):
        return " ".join(
            [
                name,
                dtype,
                ",".join(str(d) for d in shape) or "-",
                ",".join(bits) or "-",
            ]
        )

    lines = []
    for name, arr in basic.items():
        lines.append(row(name, "F32", arr.shape, f32_bits(arr)))
    # Every entry of ref_dtypes, with the value each one should present as f32.
    # The integer ones carry no values: the gate asserts read_f32 REFUSES them.
    for name, arr in dtypes.items():
        if name == "t_bf16":
            dt, expect = "BF16", f32_bits(
                np.frombuffer(
                    (bf16.astype("<u4") << 16).astype("<u4").tobytes(),
                    dtype="<f4",
                )
            )
        elif name == "t_f32":
            dt, expect = "F32", f32_bits(arr)
        elif name == "t_f64":
            dt, expect = "F64", f32_bits(arr.astype(np.float32))
        elif name == "t_f16":
            dt, expect = "F16", f32_bits(arr.astype(np.float32))
        else:
            dt, expect = {
                "t_i64": "I64",
                "t_i32": "I32",
                "t_u8": "U8",
                "t_bool": "BOOL",
            }[name], []
        lines.append(row(name, dt, arr.shape, expect))
    (out / "expected.txt").write_text("\n".join(lines) + "\n")

    print(f"wrote {out}/ref_basic.safetensors, ref_dtypes.safetensors, expected.txt")
    print(f"  header key order as written: {_header_keys(out / 'ref_basic.safetensors')}")


def _retype_u16_to_bf16(path: Path, name: str) -> None:
    """Rewrite one entry's dtype string in place.

    `safetensors.numpy` cannot write BF16 because numpy has no bfloat16, and
    installing `torch` just to produce eight values is a worse dependency than
    editing one JSON string. The BYTES are already right -- bf16 is the top
    half of the f32 -- so only the declared dtype is wrong, and U16 and BF16
    are both 2 bytes, so no offset moves.
    """
    raw = bytearray(path.read_bytes())
    n = struct.unpack("<Q", raw[:8])[0]
    head = raw[8 : 8 + n].decode("utf-8")
    needle = f'"{name}":{{"dtype":"U16"'
    if needle not in head:
        raise SystemExit(f"{path}: expected {needle!r} in the header")
    head = head.replace(needle, f'"{name}":{{"dtype":"BF16"', 1)
    # BF16 is two characters longer than U16; pad the JSON back to length so
    # every data_offset in the file stays valid.
    head = head + "  " if len(head) < n else head
    if len(head) != n:
        head = head[:n] if len(head) > n else head + " " * (n - len(head))
    raw[8 : 8 + n] = head.encode("utf-8")
    path.write_bytes(bytes(raw))


def _header_keys(path: Path) -> list[str]:
    import json

    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    return [k for k in json.loads(raw[8 : 8 + n]) if k != "__metadata__"]


# ── what we write, for the reference to read ──────────────────────────────
OURS_SPEC = [
    ("mojo.weight", (3, 4)),
    ("mojo.bias", (4,)),
    ("mojo.scalar", ()),
    ("mojo.empty", (0,)),
]


LIN_IN, LIN_OUT = 5, 3


def verify_linear(out: Path) -> int:
    """The `[in, out]` -> `[out, in]` transpose, against `torch.nn.Linear`.

    ResNet18 -- the other `TorchNameMap` -- has no `Linear` before `fc`, which
    the ACT backbone never reaches, so nothing else in the repo exercises this
    path. And it is the one place our layout genuinely differs from torch's:
    `nn.Linear.weight` is `[out, in]` and computes `y = x @ Wt`; ours is
    `[in, out]` and computes `y = x @ W`. Getting it backwards produces a file
    of the right size, with the right names, that computes something else.

    Checked twice, deliberately: once against the transpose recomputed here
    from the known values (does the FILE hold Wt?), and once by loading it into
    a real `nn.Linear` and comparing its output to `x @ W_ours + b` (does torch
    AGREE that this is the same layer?). The first can be satisfied by a
    consistent misunderstanding of torch's convention; the second cannot.
    """
    import torch

    path = out / "ours_linear.safetensors"
    if not path.exists():
        raise SystemExit(f"{path} not found — run the Mojo gate first")
    got = load_file(str(path))
    fails = 0

    if set(got) != {"fc.weight", "fc.bias"}:
        print(f"  FAIL  keys {sorted(got)}")
        return 1
    W, b = got["fc.weight"], got["fc.bias"]
    if W.shape != (LIN_OUT, LIN_IN):
        print(f"  FAIL  fc.weight shape {W.shape}, want {(LIN_OUT, LIN_IN)}")
        return 1
    print(f"  PASS  fc.weight is {W.shape} — torch's [out, in]")

    # Our side's layout, rebuilt from the value formula the Mojo gate uses.
    ours = values(LIN_IN * LIN_OUT).reshape(LIN_IN, LIN_OUT)
    if not np.array_equal(W, ours.T):
        print("  FAIL  the file is not our matrix transposed")
        fails += 1
    else:
        print(f"  PASS  the file holds our [in, out] matrix transposed "
              f"({W.size} values)")
    if not np.array_equal(b, values(LIN_OUT)):
        print("  FAIL  fc.bias values differ")
        fails += 1
    else:
        print(f"  PASS  fc.bias values ({b.size})")

    layer = torch.nn.Linear(LIN_IN, LIN_OUT)
    layer.load_state_dict(
        {"weight": torch.from_numpy(W), "bias": torch.from_numpy(b)}
    )
    x = values(2 * LIN_IN).reshape(2, LIN_IN)
    with torch.no_grad():
        y_torch = layer(torch.from_numpy(x)).numpy()
    y_ours = x @ ours + values(LIN_OUT)
    if not np.allclose(y_torch, y_ours, rtol=0, atol=1e-5):
        print(f"  FAIL  torch computes {y_torch} where we compute {y_ours}")
        fails += 1
    else:
        print("  PASS  nn.Linear(x) equals x @ W_ours + b")

    return 1 if fails else 0


def verify_ours(out: Path) -> int:
    path = out / "ours.safetensors"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run the Mojo gate first:\n"
            f"  pixi run mojo run -I . tests/io/test_safetensors_reference.mojo {out}"
        )
    got = load_file(str(path))
    fails = 0

    names = {n for n, _ in OURS_SPEC}
    if set(got) != names:
        print(f"  FAIL  key set: {sorted(got)} != {sorted(names)}")
        fails += 1

    for name, shape in OURS_SPEC:
        if name not in got:
            continue
        a = got[name]
        n = int(np.prod(shape)) if shape else 1
        want = values(n).reshape(shape)
        if a.dtype != np.float32:
            print(f"  FAIL  {name}: dtype {a.dtype}, want float32")
            fails += 1
        elif a.shape != shape:
            print(f"  FAIL  {name}: shape {a.shape}, want {shape}")
            fails += 1
        elif a.tobytes() != want.tobytes():
            print(f"  FAIL  {name}: values differ")
            fails += 1
        else:
            print(f"  PASS  {name}  {a.dtype} {a.shape}")

    # ⚠ Not a formality. The reference `load_file` is happy to hand back
    # tensors from a file whose header carries junk it ignores; reading the
    # header ourselves is what checks the parts it tolerates.
    raw = path.read_bytes()
    hlen = struct.unpack("<Q", raw[:8])[0]
    if (8 + hlen) % 8 != 0:
        print(f"  WARN  data block starts at {8 + hlen}, not 8-byte aligned")
    import json

    head = json.loads(raw[8 : 8 + hlen])
    meta = head.get("__metadata__", {})
    if meta.get("producer") != "mojo-rl":
        print(f"  FAIL  __metadata__ producer: {meta!r}")
        fails += 1
    else:
        print(f"  PASS  __metadata__  {meta}")
    if list(head) != ["__metadata__"] + [n for n, _ in OURS_SPEC]:
        print(f"  FAIL  header order {list(head)} is not the write order")
        fails += 1
    else:
        print("  PASS  header preserves the write order")

    fails += verify_linear(out)

    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


# ── can torchvision load what we exported? ────────────────────────────────
def verify_resnet18(path: Path) -> int:
    """`load_state_dict` our ResNet18 export into torchvision's own model.

    This is the half `tests/nn/test_safetensors_resnet18_torch.mojo` cannot
    reach. That gate proves our export is bit-identical to the file we read;
    it cannot prove the file is a STATE DICT — that the keys are the ones
    torchvision looks for, and that the shapes are the ones its modules
    declare. Only `load_state_dict` says that, and it is the operation an
    outside consumer will actually perform.

    `strict=False` on purpose, with the tolerated absences ENUMERATED rather
    than waved through: the ACT backbone stops at `layer4`, so there is no
    `fc`, and `num_batches_tracked` is momentum bookkeeping we do not carry.
    Anything else missing, or ANY unexpected key, is a failure.
    """
    import torch
    import torchvision
    from safetensors.torch import load_file as load_torch

    if not path.exists():
        raise SystemExit(f"{path} not found — run the Mojo gate first")

    ours = load_torch(str(path))
    net = torchvision.models.resnet18(weights=None)
    result = net.load_state_dict(ours, strict=False)
    fails = 0

    # ⚠ ONLY `fc`. `num_batches_tracked` is absent from our file too, but it
    # does NOT appear in `missing_keys`: `_NormBase._load_from_state_dict`
    # back-fills it with 0 for compatibility with pre-1.0 checkpoints and
    # drops it from the report. Listing it as tolerated therefore hides a real
    # absence rather than excusing a known one -- which is how this expectation
    # was wrong the first time it was written.
    allowed_missing = {"fc.weight", "fc.bias"}
    # Excluded from the VALUE comparison for a different reason: it is an
    # int64 momentum counter, not a weight, and torch just set ours to 0.
    skip_compare = allowed_missing | {
        k for k in net.state_dict() if k.endswith("num_batches_tracked")
    }
    missing = set(result.missing_keys)
    if result.unexpected_keys:
        print(f"  FAIL  {len(result.unexpected_keys)} unexpected key(s), "
              f"first {result.unexpected_keys[0]!r}")
        fails += 1
    else:
        print(f"  PASS  no unexpected keys ({len(ours)} tensors accepted)")

    if missing != allowed_missing:
        extra = sorted(missing - allowed_missing)
        print(f"  FAIL  {len(extra)} unexpected missing key(s), "
              f"first {extra[0]!r}" if extra else
              f"  FAIL  missing set is {sorted(missing)}")
        fails += 1
    else:
        print(f"  PASS  the only absences are {sorted(missing)}")

    # ⚠ Accepting the keys is not the same as carrying the weights. Compare
    # the loaded model against torchvision's OWN pretrained trunk: if our file
    # reconstitutes it exactly, the export is the pretrained network and not
    # merely the right shape.
    ref = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    rsd, nsd = ref.state_dict(), net.state_dict()
    compared = differing = 0
    for k, want in rsd.items():
        if k in skip_compare:
            continue
        got = nsd[k]
        if got.shape != want.shape:
            print(f"  FAIL  {k}: shape {tuple(got.shape)} != {tuple(want.shape)}")
            fails += 1
            continue
        n = want.numel()
        compared += n
        d = int((got != want).sum())
        if d:
            differing += d
            if differing == d:
                print(f"  FAIL  {k}: {d}/{n} values differ")
    if differing:
        print(f"  FAIL  {differing} of {compared} values differ from "
              f"torchvision's IMAGENET1K_V1")
        fails += 1
    else:
        print(f"  PASS  {compared} values bit-identical to "
              f"torchvision IMAGENET1K_V1")

    # And it runs.
    net.eval()
    with torch.no_grad():
        y = net(torch.zeros(1, 3, 64, 96))
    print(f"  PASS  forward runs, output {tuple(y.shape)}")

    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="write the reference files here")
    ap.add_argument("--verify", type=Path, help="read ours.safetensors from here")
    ap.add_argument(
        "--verify-resnet18",
        type=Path,
        help="load_state_dict our ResNet18 export into torchvision",
    )
    a = ap.parse_args()
    if not a.out and not a.verify and not a.verify_resnet18:
        ap.error("pass --out, --verify or --verify-resnet18")
    rc = 0
    if a.out:
        build_reference(a.out)
    if a.verify:
        rc |= verify_ours(a.verify)
    if a.verify_resnet18:
        rc |= verify_resnet18(a.verify_resnet18)
    return rc


if __name__ == "__main__":
    sys.exit(main())
