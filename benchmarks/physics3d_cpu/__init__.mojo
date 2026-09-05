"""CPU step benchmark harness, `physics3d` against MuJoCo on the same XML.

    pixi run bash scripts/physics3d_cpu_vs_mujoco.sh

The package exists so that `harness.mojo` is written ONCE and the three model
groups (`bench_gym`, `bench_so101`, `bench_contact`) are separate binaries. One
binary holding every model would carry twelve comptime XML parses in a single
build, and a `Phyics3dEnv` instantiation is not cheap to elaborate.
"""
