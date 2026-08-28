# Test fixtures

## `act_ref_xattn.tgz` (16 KB)

The cross-attention slice of the ACT reference dump — the 18 arrays
`tests/nn/test_cross_attention_gpu.mojo` reads, and nothing else.

    mkdir -p /tmp/act_ref && tar xzf tests/fixtures/act_ref_xattn.tgz -C /tmp/act_ref

**Why it is committed.** Regenerating it needs the `act-ref` pixi environment
(PyTorch + torchvision), whose dependency tree is multi-GB and has filled the
disk on a fresh box. That is a lot of setup for 16 KB of fixed arrays, and it
put the ONLY GPU gate with a prerequisite behind a torch install. The other
three GPU gates (`test_act_gpu_vs_cpu`, `test_resnet18_gpu`,
`test_batch_norm_2d_eval_backward_gpu`) need nothing at all, so with this the
whole GPU suite is `git clone` + run.

**It is a SUBSET, on purpose.** The full dump is ~130 MB, dominated by the
DETRVAE and ResNet18 parameter blobs the CPU reference gates load. Those stay
regenerated rather than committed — they are large, and they must track
`references/act-main/` rather than drift as a stale binary.

    pixi run -e act-ref python tools/act/dump_act_reference.py --out /tmp/act_ref

`manifest.txt` here lists only the `xattn_*` arrays. `RefDump` reads names and
sizes from the manifest and opens a `.bin` only when asked for it, so a subset
manifest is consistent — a gate asking for anything outside it fails loudly with
the missing name rather than reading garbage.
