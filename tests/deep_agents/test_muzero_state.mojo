"""Test MuZero state container and network dimensions."""

from mojo_rl.deep_agents.muzero.state import MuZeroCPUState


def main():
    print("=== MuZero State Tests ===")

    comptime OBS = 4
    comptime ACT = 2

    # Check compile-time dimensions
    comptime StateType = MuZeroCPUState[
        OBS, ACT, LATENT_DIM=64, HIDDEN_DIM=64, NUM_BINS=51
    ]
    print(
        "RepModel: IN=",
        StateType.RepModel.IN_DIM,
        "OUT=",
        StateType.RepModel.OUT_DIM,
        "PARAMS=",
        StateType.RepModel.PARAM_SIZE,
    )
    print(
        "DynModel: IN=",
        StateType.DynModel.IN_DIM,
        "OUT=",
        StateType.DynModel.OUT_DIM,
        "PARAMS=",
        StateType.DynModel.PARAM_SIZE,
    )
    print(
        "PredModel: IN=",
        StateType.PredModel.IN_DIM,
        "OUT=",
        StateType.PredModel.OUT_DIM,
        "PARAMS=",
        StateType.PredModel.PARAM_SIZE,
    )

    # Verify dimension consistency
    if StateType.RepModel.OUT_DIM == 64:
        print("PASS: RepModel output = LATENT_DIM")
    else:
        print("FAIL: RepModel output =", StateType.RepModel.OUT_DIM)

    if StateType.DynModel.IN_DIM == 64 + ACT:
        print("PASS: DynModel input = LATENT + ACT")
    else:
        print("FAIL: DynModel input =", StateType.DynModel.IN_DIM)

    if StateType.DynModel.OUT_DIM == 64 + 51:
        print("PASS: DynModel output = LATENT + BINS")
    else:
        print("FAIL: DynModel output =", StateType.DynModel.OUT_DIM)

    if StateType.PredModel.IN_DIM == 64:
        print("PASS: PredModel input = LATENT")
    else:
        print("FAIL: PredModel input =", StateType.PredModel.IN_DIM)

    if StateType.PredModel.OUT_DIM == ACT + 51:
        print("PASS: PredModel output = ACT + BINS")
    else:
        print("FAIL: PredModel output =", StateType.PredModel.OUT_DIM)

    # Test construction
    print("Creating state...")
    var state = StateType()
    print("State created successfully")
    print("Buffer capacity check:", state.buffer.len() == 0)
    print("is_ready:", state.is_ready())

    print("=== Done ===")
