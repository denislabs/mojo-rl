"""Compile-only smoke for PCDynamicsEnsembleGPU — Phase B1a build check.

Just instantiates the type and references its methods so the compiler
must monomorphize them. No GPU work — caller (`main`) constructs no
DeviceContext, so this just walks the type machinery.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.experimental.pcn import PCDynamicsEnsembleGPU


comptime ENS = PCDynamicsEnsembleGPU[
    OBS_DIM=3, ACTION_DIM=1, HIDDEN_DIM=64,
    NUM_ENSEMBLE=3, NUM_ELITES=2, dtype=dtype,
]
comptime OPT = Adam[LR=0.001]


def main() raises:
    print("PCDynamicsEnsembleGPU compile-only smoke")
    print("  PER_MEMBER_PARAM_SIZE =", ENS.PER_MEMBER_PARAM_SIZE)
    print("  TOTAL_PARAM_SIZE      =", ENS.TOTAL_PARAM_SIZE)
    # Touch select_elites (pure host) to force monomorphization.
    var losses = List[Float64]()
    losses.append(0.5)
    losses.append(0.3)
    losses.append(0.4)
    var elites = List[Int]()
    ENS.select_elites(losses, elites)
    print("  elites:", elites)
    print("=== Compile-only smoke OK ===")
