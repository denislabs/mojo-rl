"""`fit_bc` on a `.demo` recording — BC is not LIBERO-only.

    pixi run -e apple  mojo run -I . tests/deep_agents/test_bc_fit_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/deep_agents/test_bc_fit_gpu.mojo

A DemoSet whose actions are a fixed linear function of the observation (plus
a bias), 5 episodes; `BcDataset.from_demo_set` holds out the last one;
`fit_bc` must beat the training-mean baseline by a wide margin, and its
checkpoint must reload on the CPU to the same score (fit_bc raises if not).
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true

from noeira.deep_agents.demos.file import DemoSet
from noeira.deep_agents.demos.filter import DemoFilter
from noeira.deep_agents.bc.dataset import BcDataset
from noeira.deep_agents.bc.fit import fit_bc
from noeira.deep_agents.bc.policy import load_bc_norm

comptime OBS = 4
comptime ACT = 2
comptime BATCH = 16
comptime OUT = "/tmp/noeira_test_bc_fit_gpu.ckpt"


def _obs(e: Int, t: Int) -> List[Float64]:
    var o = List[Float64]()
    for i in range(OBS):
        o.append(Float64(((e * 97 + t * 31 + i * 17) % 23)) / 11.0 - 1.0)
    return o^


def _act(o: List[Float64]) -> List[Float64]:
    var a = List[Float64]()
    a.append(0.5 * o[0] - 0.25 * o[1] + 0.1)
    a.append(-0.3 * o[2] + 0.2 * o[3] - 0.05)
    return a^


def main() raises:
    comptime if not has_accelerator():
        print("SKIPPED: fit_bc trains on the device (no accelerator)")
        return
    var ds = DemoSet(OBS, ACT)
    for e in range(5):
        ds.begin_episode()
        for t in range(64):
            var o = _obs(e, t)
            var n = _obs(e, t + 1)
            var a = _act(o)
            ds.add(o, a, 0.0, n, 0.0, intervened=(t % 3 == 0))
        ds.end_episode(success=(e != 1))
    var data = BcDataset.from_demo_set(ds, DemoFilter(), 1)
    assert_equal(data.n_tr, 256)
    assert_equal(data.n_va, 64)
    var only_success = BcDataset.from_demo_set(ds, DemoFilter(DemoFilter.SUCCESS), 1)
    assert_equal(only_success.n_tr, 192, "the filter drops the failed episode")

    var rep = fit_bc[OBS, ACT, BATCH](data, 40, 3.0e-3, String(OUT), String("bc test"))
    assert_true(rep.best_val < 0.2 * rep.mse_mean, "a linear map is learnable")
    assert_true(rep.reloaded_val <= rep.best_val * 1.5 + 1.0e-9)
    _ = load_bc_norm(String(OUT) + ".norm", OBS, ACT)
    print("bc fit: val MSE", rep.best_val, "against the mean's", rep.mse_mean)
