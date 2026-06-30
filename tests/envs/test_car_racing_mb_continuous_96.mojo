"""CarRacingMB CONTINUOUS action + 96x96 PIXEL obs smoke (P3 of the pixel arc).

Validates the new BoxContinuousActionEnv conformance + parametrized resolution:
  - CarRacingMB[f32, PIXEL_OBS=True, PIX_RES=96] exposes a 4x96x96 = 36864 pixel
    obs (16-divisible → fits the DreamerV3 conv stack, minres 6).
  - step_continuous_vec([steer, gas, brake]) drives the car (full gas → speed↑),
    pixels in [0,1], car sprite visible at the view center.
  - gas/brake remap [-1,1] → [0,1] (Gymnasium continuous convention).
  - clean continuous variant (PIXEL_OBS=False) returns the 13-D obs.

Run: pixi run mojo run -I . tests/envs/test_car_racing_mb_continuous_96.mojo
"""

from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB
from mojo_rl.envs.car_racing.car_racing_pixel import CarRacingPixel

comptime DT = DType.float32


def test_continuous_pixel_96() raises:
    comptime E = CarRacingMB[DT, True, 96]
    comptime P = CarRacingPixel[DT, 96]
    comptime EXP = 4 * 96 * 96  # 36864
    if E.EFF_OBS_DIM != EXP or P.OBS_DIM != EXP:
        raise Error("96px pixel obs dim mismatch")
    if P.OBS_W != 96 or P.OBS_H != 96:
        raise Error("PIX_RES=96 did not set OBS_W/H")

    var env = E()
    var obs = env.reset_obs_list()
    if len(obs) != EXP:
        raise Error("reset pixel obs wrong length")

    # Drive with full gas, straight: action [steer=0, gas=+1 (→1.0), brake=-1 (→0)]
    var act = List[Scalar[DT]]()
    act.append(Scalar[DT](0.0))
    act.append(Scalar[DT](1.0))
    act.append(Scalar[DT](-1.0))
    var done = False
    var last = env.reset_obs_list()
    for _ in range(60):
        var r = env.step_continuous_vec[DT](act)
        last = r[0].copy()
        done = r[2]
        if done:
            break

    # obs bounds + content on the newest frame.
    var mn = Scalar[DT](1e9)
    var mx = Scalar[DT](-1e9)
    for i in range(len(last)):
        if last[i] < mn:
            mn = last[i]
        if last[i] > mx:
            mx = last[i]
    var newest = (P.FRAME_STACK - 1) * P.FRAME_SIZE
    # The car sprite is drawn at screen (CX, CY) (camera follows the car).
    var car_px = last[newest + Int(P.CY) * 96 + Int(P.CX)]
    var n_road = 0
    var n_grass = 0
    var n_car = 0
    for i in range(P.FRAME_SIZE):
        var v = last[newest + i]
        if v > Scalar[DT](0.95):
            n_car += 1
        elif v > Scalar[DT](0.5):
            n_road += 1
        elif v > Scalar[DT](0.2):
            n_grass += 1
    print("  96px obs_dim", E.EFF_OBS_DIM, " range [", mn, ",", mx, "]")
    print("  newest frame: car", n_car, " road", n_road, " grass", n_grass,
          " car-sprite px", car_px)
    if mn < Scalar[DT](0.0) or mx > Scalar[DT](1.0):
        raise Error("pixel obs out of [0,1]")
    if not (n_car > 0 and n_road > 0 and n_grass > 0):
        raise Error("expected car+road+grass pixels in the 96px frame")
    if car_px < Scalar[DT](0.95):
        raise Error("car sprite not at screen (CX,CY) — camera scaling wrong")
    print("test_continuous_pixel_96: OK")


def test_continuous_drives() raises:
    # Clean-obs continuous variant: full gas should raise speed (obs[12]=|v|/100).
    comptime E = CarRacingMB[DT, False]
    var env = E()
    _ = env.reset_obs_list()
    var act = List[Scalar[DT]]()
    act.append(Scalar[DT](0.0))
    act.append(Scalar[DT](1.0))   # gas → 1.0
    act.append(Scalar[DT](-1.0))  # brake → 0.0
    var spd = Scalar[DT](0.0)
    for _ in range(60):
        var r = env.step_continuous_vec[DT](act)
        spd = r[0][12]  # |speed| / 100 (normalized clean obs)
    print("  clean continuous: normalized speed after 60 gas steps =", spd)
    if not (spd > Scalar[DT](0.05)):
        raise Error("continuous full-gas did not accelerate the car")
    print("test_continuous_drives: OK")


def main() raises:
    print("=== CarRacingMB continuous + 96x96 pixel smoke ===")
    test_continuous_pixel_96()
    test_continuous_drives()
    print("=== PASS: CarRacingMB BoxContinuousActionEnv + 96px pixel obs ===")
