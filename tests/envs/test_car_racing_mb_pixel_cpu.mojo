"""CPU pixel-rasterizer test for CarRacingMB (matches the GPU CarRacingPixel).

Validates the CPU pixel path used by the color-scene eval-render: reset_pixel /
step_action_pixel produce a 4x84x84 frame stack in [0,1] whose newest frame
contains the car sprite (center), road, and grass — the same kind of image the
GPU env renders, so a pixel-trained CNN sees in-distribution input.
"""

from mojo_rl.envs.car_racing import CarRacingMB, CarRacingPixel


def main() raises:
    print("=== CarRacingMB CPU pixel-rasterizer test ===")
    comptime P = CarRacingPixel[DType.float32]
    comptime OBS = P.OBS_DIM
    comptime FS = P.FRAME_SIZE
    comptime OW = P.OBS_W

    var env = CarRacingMB[DType.float32](max_steps=1000)
    var obs = env.reset_pixel()
    print("pixel obs_dim =", len(obs), " expected =", OBS)
    if len(obs) != OBS:
        raise Error("pixel obs dim mismatch")

    # Drive forward a bit so the view has road around the car.
    for _ in range(20):
        var r = env.step_action_pixel(3)  # gas
        obs = r[0].copy()

    var car = 0
    var road = 0
    var grass = 0
    var oor = 0
    var nb = 3 * FS  # newest frame
    for p in range(FS):
        var v = Float64(obs[nb + p])
        if v < 0.0 or v > 1.0:
            oor += 1
        if v > 0.95:
            car += 1
        elif v > 0.6 and v < 0.8:
            road += 1
        elif v > 0.25 and v < 0.35:
            grass += 1
    var cx = Int(OW // 2)
    var cy = Int(P.CY)
    var center = Float64(obs[nb + cy * OW + cx])
    print("newest frame: car=", car, " road=", road, " grass=", grass)
    print("center pixel =", center, " (expect ~1.0)")

    if oor > 0:
        raise Error(String("pixels out of [0,1]: ", oor))
    if car <= 0:
        raise Error("no car sprite pixels")
    if road <= 0:
        raise Error("no road pixels — track not visible")
    if grass <= 0:
        raise Error("no grass pixels")
    if center < 0.95:
        raise Error(String("center pixel not car (", center, ")"))

    print("=== PASS: CPU rasterizer renders car + road + grass, obs in [0,1] ===")
