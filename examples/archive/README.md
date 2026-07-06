# Archived examples

Unmaintained scripts that reference removed APIs and are superseded by
current code. Kept for reference only — they do NOT compile and are excluded
from the `examples-compile` CI manifest.

| Script | Why archived | Superseded by |
|---|---|---|
| `test_cartpole_gpu.mojo` | imports the removed `mojo_rl.envs.cartpole_gpu` module (GPU kernels were folded into `envs/cartpole.mojo`) | `tests/` CartPole GPU tests + `examples/cartpole/` |
| `test_lunar_lander_v2_gpu_cpu.mojo` | uses the removed `LunarLander.step_kernel` public API | `examples/lunar_lander/` + `tests/envs/test_lunar_lander_copy_semantics.mojo` |
