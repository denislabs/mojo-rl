# max_rl — MAX-as-inference-backend prototype

A small prototype probing **MAX as an *inference* backend for mojo-rl**, driven from Mojo
via Python interop. Companion to `docs/MAX_TRAINING_ASSESSMENT.md` (which covers why
*training* on MAX is blocked today). v1 scope is **MLP inference only**.

It answers three questions for "should mojo-rl incorporate MAX?":

1. How fast is MAX device compute on a realistic RL MLP?
2. What is the host↔device data-transfer cost (H2D / D2H)?
3. What does the Mojo↔Python interop bridge actually cost on the real call path?

## Layout

| File | What |
|---|---|
| `mlp_inference.py` | `MLPInference` — configurable MLP (dims/batch/device all variables) built+compiled once on MAX, plus timing primitives. Inference only; weights are random. |
| `benchmark_interop.mojo` | Mojo driver that imports the package via Python interop and times the MAX decomposition across a batch/shape sweep. |
| `benchmark_nn2_baseline.mojo` | Pure-nn2 native GPU forward on the SAME shapes — the apples-to-apples "why incorporate MAX?" baseline. |
| `graph_mlp_example.py`, `graph_relu_example.py` | Original MAX reference snippets (kept for reference). |

`MLPInference(input_dim, hidden, output_dim, batch, device="gpu", seed=0)` — `hidden` is a
list `[256, 256]` or a Mojo-friendly string `"256,256"`. `device="gpu"` → Metal on Apple,
CUDA on NVIDIA (same Python, different backend); falls back to CPU if no accelerator.

## How to run

**Build to a binary — do NOT `mojo run`.**

```bash
# Apple (Metal)
pixi run -e apple  mojo build -I . max_rl/benchmark_interop.mojo -o /tmp/bench && /tmp/bench
# NVIDIA (CUDA)
pixi run -e nvidia mojo build -I . max_rl/benchmark_interop.mojo -o /tmp/bench && /tmp/bench
```

nn2 baseline (pure nn2, no Python — plain `mojo run` is fine):
```bash
pixi run -e apple  mojo run -I . max_rl/benchmark_nn2_baseline.mojo
pixi run -e nvidia mojo run -I . max_rl/benchmark_nn2_baseline.mojo
```

You can also drive the Python package directly:
```bash
pixi run -e apple python -c "from max_rl import MLPInference; m=MLPInference(17,'256,256',6,64); print(m.info())"
```

## Findings so far (Apple M-series / Metal, 2026-06-03)

### ⚠️ Footgun: `mojo run` + MAX Python engine clash at the runtime-context level
JIT `mojo run` creates an `M::Context` whose `Init::Options` conflict with the one
`max.engine` wants → `LLVM ERROR: Init::getOrCreateContext() requested an M::Context with
different Init::Options`. **A compiled binary has no JIT context, so the Python engine
initializes cleanly.** Always `mojo build` then run the executable. (Worth re-checking on
NVIDIA — this is your part.)

### The Mojo↔Python FFI crossing is essentially free
The per-call interop floor (a Mojo loop over a Python `noop()`) is **~0.15 µs/call**. The
"Python in the hot loop" tax people worry about is *not* the FFI boundary. Mojo-side
end-to-end `infer()` ≈ Python-side end-to-end `full()` to within run-to-run noise.

### The real costs are MAX compute, data transfer, and Python *glue*
For each call the decomposition is: `MAX device compute` + `H2D` + `D2H` + `Python glue`
(numpy contiguity checks, `Buffer` object creation, attribute lookups). On Metal the
**Python glue per call (~130–170 µs)** dwarfs the FFI crossing (0.15 µs) — i.e. *what you
do in Python per call matters far more than crossing the Mojo/Python line.* This is the
argument for the production path (B): Mojo → MAX **C API** on a precompiled MEF, which
removes the Python glue entirely.

### Numbers are path (A), a pessimistic upper bound
This prototype is **path (A): Mojo → CPython → MAX**, Python in the hot loop. The
production-realistic **path (B): Mojo → MAX C API** (`M_executeModelSync` on a precompiled
MEF) removes Python from the loop and is strictly faster. Read (A) as an upper bound: *if
MAX wins even here, path (B) wins by more.*

### Metal compile time is high (~15 s even for a tiny MLP)
One-time per graph shape, excluded from per-call numbers, but relevant for research
iteration that sweeps many shapes. NVIDIA compile times are the ones that matter for you.

### nn2 vs MAX head-to-head (Apple/Metal, µs per call)

| Shape | nn2 forward (compute) | MAX device compute | MAX end-to-end from Mojo |
|---|---|---|---|
| actor-b1    (17→256→256→6, b=1)     | 658  | **284**   | 507   |
| actor-b64   (b=64)                  | 618  | **309**   | 978   |
| actor-b1024 (b=1024)                | **1357** | 1663  | 4484  |
| wide-b1     (256→512→512→64, b=1)   | 699  | **287**   | 868   |
| wide-b1024  (b=1024)                | **8913** | 12455 | 15050 |

Reading (Metal only — **NVIDIA is the decisive run, your part**):
- **MAX raw compute wins at small batch** (~2× faster, 284 vs 658 µs at b=1) but **loses
  end-to-end** once you add H2D+D2H+Python glue — the *delivered* MAX latency to a Mojo
  caller (507–978 µs) is at or above nn2's (618–699 µs).
- **nn2 wins outright at large batch**, on raw compute *and* end-to-end (its native kernels
  beat MAX's here on Metal, and it pays zero transfer/Python tax).
- nn2's number is the **full delivered latency** to a Mojo caller (data already in Mojo GPU
  buffers); MAX must overcome its transfer+glue tax to be worth it. On Metal it generally
  isn't; whether MAX's compute edge widens enough on CUDA to flip the end-to-end verdict is
  exactly what the NVIDIA run answers.
- Caveat: nn2 small-batch numbers include nn2's per-call overhead (Sequential mid-buffer
  handling); both columns are Metal and not the target platform. Treat the *shape of the
  story* (compute vs delivered, small vs large batch) as the takeaway, not absolute µs.

## Not yet done (next steps)
- **NVIDIA run** (the decisive measurement — your part). Re-verify the `mojo run` clash and
  whether it reproduces or differs on CUDA, and whether MAX's compute edge widens enough to
  flip the end-to-end verdict against nn2.
- **Path (B)**: a thin Mojo C-API binding to `M_executeModelSync` on a precompiled MEF, to
  measure the no-Python hot path (removes the Python-glue tax entirely).
- bf16 / fp16, CUDA-graph capture/replay (`M_captureModelSync`).

## Done
- ✅ `MLPInference` MAX package (configurable dims/batch/device) + interop decomposition.
- ✅ nn2 native baseline on identical shapes (`benchmark_nn2_baseline.mojo`).
