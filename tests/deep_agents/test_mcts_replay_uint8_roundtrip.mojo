"""MCTSSequenceReplay uint8 obs ring — bit-lossless k/255 round-trip (CPU).

Phase 1 of `docs/MUZERO_PIXEL_PONG_PLAN.md`: the host MuZero replay gained an
``OBS_STORE_DT`` param so pixel obs can live as uint8 (4× capacity). The arcade
pixel pipeline emits exact ``k/255`` grayscale, so quantizing ``round(x·255)`` on
store and dequantizing ``k/255`` on read must be EXACT for those inputs. This
test stores an episode of ``k/255`` observations into both a uint8 ring and the
default float ring and asserts `read_obs` returns the original values bit-for-bit
from both — so swapping in uint8 storage is lossless for pixels.

Run:
    pixi run mojo run -I . tests/deep_agents/test_mcts_replay_uint8_roundtrip.mojo
"""

from std.memory import alloc
from std.testing import assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import mptr
from mojo_rl.deep_agents.zero.sequence_replay_mcts import MCTSSequenceReplay


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


def main() raises:
    comptime OBS = 8
    comptime ACT = 3
    comptime CAP = 64
    comptime L = 5    # episode length

    # Episode obs: exact k/255 grayscale values, k cycling through [0, 255].
    var ep_obs = _a(L * OBS)
    for i in range(L * OBS):
        var k = (i * 17 + 3) % 256
        ep_obs[i] = Scalar[DT](Float64(k) / 255.0)
    # Minimal other fields (a single env, all legal, to_play 0).
    var ep_act = _a(L)
    var ep_rew = _a(L)
    var ep_val = _a(L)
    var ep_tp = _a(L)
    var ep_pol = _a(L * ACT)
    var ep_legal = _a(L * ACT)
    for i in range(L):
        ep_act[i] = Scalar[DT](i % ACT)
        ep_rew[i] = Scalar[DT](0.0)
        ep_val[i] = Scalar[DT](0.0)
        ep_tp[i] = Scalar[DT](0.0)
        for a in range(ACT):
            ep_pol[i * ACT + a] = Scalar[DT](1.0) / Scalar[DT](ACT)
            ep_legal[i * ACT + a] = Scalar[DT](1.0)

    var rb_u8 = MCTSSequenceReplay[OBS, ACT, CAP, DType.uint8](seed=1)
    var rb_f = MCTSSequenceReplay[OBS, ACT, CAP](seed=1)  # default DT
    rb_u8.store_episode(
        ep_obs, ep_act, ep_rew, ep_pol, ep_val, ep_tp, ep_legal, L
    )
    rb_f.store_episode(
        ep_obs, ep_act, ep_rew, ep_pol, ep_val, ep_tp, ep_legal, L
    )
    assert_equal(rb_u8.num_steps(), L, "uint8 ring step count")
    assert_equal(rb_f.num_steps(), L, "float ring step count")

    var out_u8 = _a(OBS)
    var out_f = _a(OBS)
    var max_abs_err = 0.0
    for off in range(L):
        rb_u8.read_obs(0, off, out_u8)
        rb_f.read_obs(0, off, out_f)
        for j in range(OBS):
            var orig = Float64(ep_obs[off * OBS + j])
            var du8 = Float64(out_u8[j])
            var df = Float64(out_f[j])
            # float ring is a pure copy → exact.
            assert_true(df == orig, "float ring not bit-exact")
            # uint8 ring is k/255 → round(x·255)/255 → exact for k/255 inputs.
            var e = du8 - orig
            if e < 0.0:
                e = -e
            if e > max_abs_err:
                max_abs_err = e
            assert_true(e == 0.0, "uint8 ring not bit-lossless for k/255 obs")
    print("uint8 round-trip max |err| =", max_abs_err, "(expect 0.0)")

    ep_obs.free(); ep_act.free(); ep_rew.free(); ep_val.free(); ep_tp.free()
    ep_pol.free(); ep_legal.free(); out_u8.free(); out_f.free()
    print("MCTSSequenceReplay uint8 round-trip: OK")
