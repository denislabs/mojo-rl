"""TargetTag — runtime enum for module misuse detection.

Modules in nn2 carry both CPU and GPU storage slots; `make[target, INIT]`
populates one set and stamps the runtime `_target_tag`. Every method
that touches state opens with a tag check matching its `[target]`
comptime param, catching make-CPU / forward-GPU mistakes at the entry
point with a clear message rather than letting `Optional.value()` panic
on `None`.

Three values:
  - `UNINIT = 0` — default-constructed; no `make` has been called yet.
  - `CPU    = 1`
  - `GPU    = 2`

`target_tag_for[target]()` maps the comptime `StaticString` to the
runtime `Int8` tag. Use it from method bodies:

```mojo
from mojo_rl.nn2.core import target_tag_for, TARGET_CPU
var target_tag = TARGET_CPU
comptime expected = target_tag_for["cpu"]()
if target_tag != expected:
    raise Error("...")
```
"""


comptime TARGET_UNINIT: Int8 = 0
comptime TARGET_CPU: Int8 = 1
comptime TARGET_GPU: Int8 = 2


def target_tag_for[target: StaticString]() -> Int8:
    comptime if target == "cpu":
        return TARGET_CPU
    elif target == "gpu":
        return TARGET_GPU
    else:
        comptime assert False, "target must be 'cpu' or 'gpu'"
