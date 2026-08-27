"""CPU smoke test for the storage-migrated DreamerV3 trainer + agent.

Constructs a tiny DreamerV3Trainer, records synthetic transitions until it can
train, runs a few train_steps and asserts the WM / AC losses are finite +
nonzero. Then constructs a DreamerV3Agent and calls select_action once,
asserting a finite action. Gate for the trainer.mojo / agent.mojo storage port.
"""

from std.math import isfinite
from std.random import random_float64
from std.memory import alloc

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent


comptime OBS = 3
comptime ACT = 1
comptime DETER = 8
comptime H = 8
comptime STOCH = 3
comptime CLASSES = 4
comptime BLOCKS = 2
comptime TOKEN = 6
comptime DEC_U = 8
comptime HU = 8
comptime VU = 8
comptime PU = 8
comptime BINS = 7
comptime B = 4
comptime T = 3
comptime T_IMAG = 3
comptime CAP = 2000


comptime TrainerT = DreamerV3Trainer[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU, PU,
    BINS, B, T, T_IMAG, CAP, False,
]
comptime AgentT = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU, PU,
    BINS, B, T, T_IMAG, CAP, False,
]


def test_trainer_train_step() raises:
    var tr = TrainerT.make(learning_starts=8)

    var obs = alloc[Scalar[DT]](OBS)
    var act = alloc[Scalar[DT]](ACT)

    # record enough synthetic transitions to be able to sample a length-T window
    for s in range(64):
        for i in range(OBS):
            obs[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        for j in range(ACT):
            act[j] = Scalar[DT](random_float64() * 2.0 - 1.0)
        var rew = Scalar[DT](random_float64() - 0.5)
        var done = Scalar[DT](1.0) if (s % 17 == 16) else Scalar[DT](0.0)
        tr.record(
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](obs),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](act),
            rew, done,
        )

    if not tr.can_train():
        raise Error("trainer cannot train after recording 64 transitions")

    var saw_nonzero_wm = False
    var saw_nonzero_ac = False
    for _ in range(5):
        var did = tr.train_step()
        if not did:
            raise Error("train_step returned False after can_train()")
        var wl = tr.last_wm_loss()
        var al = tr.last_ac_loss()
        if not isfinite(Float64(wl)):
            raise Error("wm_loss is not finite")
        if not isfinite(Float64(al)):
            raise Error("ac_loss is not finite")
        if wl != Scalar[DT](0.0):
            saw_nonzero_wm = True
        if al != Scalar[DT](0.0):
            saw_nonzero_ac = True

    if not saw_nonzero_wm:
        raise Error("wm_loss stayed exactly zero across 5 steps")
    if not saw_nonzero_ac:
        raise Error("ac_loss stayed exactly zero across 5 steps")

    obs.free()
    act.free()
    print("test_trainer_train_step: OK (wm/ac losses finite + nonzero)")


def test_agent_select_action() raises:
    var ag = AgentT.make(learning_starts=8)

    var obs = alloc[Scalar[DT]](OBS)
    var out_action = alloc[Scalar[DT]](ACT)
    for i in range(OBS):
        obs[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    for j in range(ACT):
        out_action[j] = Scalar[DT](0.0)

    ag.select_action(
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](obs),
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](out_action),
        explore=True,
    )
    for j in range(ACT):
        if not isfinite(Float64(out_action[j])):
            raise Error("select_action produced a non-finite action")
        if out_action[j] > Scalar[DT](1.0) or out_action[j] < Scalar[DT](-1.0):
            raise Error("select_action action out of [-1,1]")

    obs.free()
    out_action.free()
    print("test_agent_select_action: OK (action finite + in [-1,1])")


def main() raises:
    test_trainer_train_step()
    test_agent_select_action()
    print("ALL DREAMERV3 STORAGE CPU SMOKE TESTS PASSED")
