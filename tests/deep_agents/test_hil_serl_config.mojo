"""`HilSerlConfig` — the flags parse as the SAC driver's did, and refuse the same.

    pixi run mojo run -I . tests/deep_agents/test_hil_serl_config.mojo
"""

from std.testing import assert_equal, assert_true, assert_false

from noeira.nn.constants import DT
from noeira.deep_agents.hil_serl import HilSerlConfig
from noeira.deep_agents.demos.filter import DemoFilter


def _raises_with(cfg: HilSerlConfig, text: String) -> Bool:
    try:
        cfg.validate("sac task")
    except e:
        return String(e) == text
    return False


def main() raises:
    var c = HilSerlConfig()
    assert_false(c.active())
    c.validate("sac task")                       # plain SAC: nothing to refuse
    assert_true(c.try_parse("--demos", "a.demo,b.demo", True))
    assert_true(c.try_parse("--demo-filter", "intervened", True))
    assert_true(c.try_parse("--bc-weight", "40", True))
    assert_true(c.try_parse("--bc-only", "", False))
    assert_false(c.try_parse("--steps", "10", True), "not ours")
    assert_false(c.try_parse("--demos", "", False), "a valueless --demos is the caller's error")
    assert_true(c.active())
    assert_equal(c.filter.kind, DemoFilter.INTERVENED)
    assert_equal(c.bc_weight, Scalar[DT](40.0))
    assert_true(c.bc_only)
    var p = c.paths()
    assert_equal(len(p), 2)
    assert_equal(p[1], "b.demo")
    c.validate("sac task")

    var no_demos = HilSerlConfig()
    _ = no_demos.try_parse("--bc-weight", "1", True)
    assert_true(_raises_with(no_demos, "sac task: --bc-weight needs --demos"))

    var only = HilSerlConfig()
    _ = only.try_parse("--demos", "a.demo", True)
    _ = only.try_parse("--bc-only", "", False)
    assert_true(_raises_with(only, "sac task: --bc-only needs --bc-weight > 0"))

    var comma = HilSerlConfig()
    _ = comma.try_parse("--demos", "a.demo,", True)
    var refused = False
    try:
        _ = comma.paths("sac task")
    except e:
        refused = "empty element" in String(e)
    assert_true(refused, "a trailing comma is refused")
    print("hil-serl config: flags, filter, validation and paths as the driver's")
