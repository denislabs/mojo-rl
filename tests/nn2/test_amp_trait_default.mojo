"""Probe: does Mojo accept a default on a trait-typed comptime param?

If `def f[T: SomeTrait = SomeImpl](...)` compiles and dispatches correctly
when caller omits T, we can default POLICY = NoAMP in Module trait and
avoid updating every caller in Phase 3.
"""

from mojo_rl.nn2.core import AMPPolicy, NoAMP, Bf16Compute


def show_policy[POLICY: AMPPolicy = NoAMP](label: String) raises:
    print(label + ": compute_dtype = " + String(POLICY.compute_dtype))


def main() raises:
    print("=" * 60)
    print("Probe: default on trait-typed comptime param")
    print("=" * 60)
    # Caller omits POLICY -> default NoAMP -> fp32.
    show_policy("default     ")
    # Caller passes explicit policy.
    show_policy[POLICY=NoAMP]("explicit NoAMP")
    show_policy[POLICY=Bf16Compute]("Bf16Compute  ")
    print("PROBE PASSED")
