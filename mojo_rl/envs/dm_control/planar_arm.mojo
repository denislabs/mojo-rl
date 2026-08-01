"""The planar arm shared by `manipulator` and `stacker`.

Both domains drive the SAME four-link planar arm — `arm_root`, `arm_shoulder`,
`arm_elbow`, `arm_wrist` plus a two-finger hand on `thumb`/`thumbtip` and
`finger`/`fingertip` — and both ship it as their own copy of the MJCF. Upstream
keeps `manipulator.xml` and `stacker.xml` as two files that happen to agree on
the arm and disagree on one default (`<geom solref>` is `.005 1` there and
`.01 1` here), so the XML stays duplicated in each domain's `*_xml.mojo`: a
shared segment would silently propagate an upstream change to one file into the
other.

What is NOT duplicated is everything below — the index permutations and the
reset-time geometry, which have no XML identity of their own and would be a
silent divergence if the two copies ever drifted. `arm_clearance` in particular
is a correctness argument, not a formula (see its docstring); two copies of an
argument is one copy too many.

⚠ THE OBSERVATION'S JOINT ORDER IS NOT THE MODEL'S. Both tasks declare

    _ARM_JOINTS = [arm_root, arm_shoulder, arm_elbow, arm_wrist,
                   finger, fingertip, thumb, thumbtip]

while the MODEL declares the thumb chain BEFORE the finger chain, so entries
4..7 are transposed in pairs. `arm_joint_obs_order` carries that permutation.
A symmetric hand pose — which both resets produce, since both symmetrise — hides
the difference completely, so it has to be driven asymmetrically to be tested.
"""

from std.math import sin, cos, sqrt


comptime NARM_JOINTS: Int = 8

# world=0, then upper_arm, middle_arm, lower_arm, hand — in both domains, since
# both put the arm first in the worldbody and the props after it.
comptime HAND_BODY_IDX: Int = 4

# Arm sites, OUR order (XML text order). MuJoCo sorts sites by body id, and the
# one place the two orders DIVERGE is `palm_touch`: it is declared after the
# `pinch site` body yet belongs to `hand`, so MuJoCo lists it before `pinch`.
# The parity tests carry that swap as `_our_site_to_mj`.
comptime SITE_GRASP: Int = 0
comptime SITE_PINCH: Int = 1
comptime SITE_PALM_TOUCH: Int = 2
comptime SITE_THUMB_TOUCH: Int = 3
comptime SITE_THUMBTIP_TOUCH: Int = 4
comptime SITE_FINGER_TOUCH: Int = 5
comptime SITE_FINGERTIP_TOUCH: Int = 6
comptime N_ARM_SITES: Int = 7


def arm_joint_obs_order(k: Int) -> Int:
    """`_ARM_JOINTS[k]` as OUR joint index."""
    if k == 4:
        return 6  # finger
    if k == 5:
        return 7  # fingertip
    if k == 6:
        return 4  # thumb
    if k == 7:
        return 5  # thumbtip
    return k  # arm_root, arm_shoulder, arm_elbow, arm_wrist


def touch_site_order(k: Int) -> Int:
    """The k'th touch sensor's zone, as OUR site index.

    `manipulator` reads `sensordata[_TOUCH_SENSORS]` and `stacker` reads all of
    `sensordata`; the `<sensor>` block is identical and declares them in the
    order palm, finger, thumb, fingertip, thumbtip, so the two agree.
    """
    if k == 0:
        return SITE_PALM_TOUCH
    if k == 1:
        return SITE_FINGER_TOUCH
    if k == 2:
        return SITE_THUMB_TOUCH
    if k == 3:
        return SITE_FINGERTIP_TOUCH
    return SITE_THUMBTIP_TOUCH


# ── the arm as a planar chain, for reset-time clearance ─────────────────────
#
# Both resets need the arm's actual pose to place props clear of it, and a reset
# hook runs BEFORE forward kinematics (`_reset_state`: reset_data -> hooks -> FK)
# so `d.xpos` is not yet meaningful. That is fine here because the arm is a
# PLANAR chain with every hinge about `0 -1 0`, so its world geometry is three
# lines of trig rather than a call into FK.
#
# Rotating a local vector by angle `q` about the axis (0,-1,0) maps the local +z
# direction to `(-sin q, cos q)` in world (x, z), and the joints compose, so
# link `i` runs along the cumulative angle `q0 + ... + qi`.
comptime SHOULDER_X: Float64 = 0.0
comptime SHOULDER_Z: Float64 = 0.4  # `<body name="upper_arm" pos="0 0 .4">`

# `fromto` lengths and `size` radii of upper_arm / middle_arm / lower_arm.
comptime LINK_LEN_0: Float64 = 0.18
comptime LINK_LEN_1: Float64 = 0.15
comptime LINK_LEN_2: Float64 = 0.12
comptime LINK_RAD_0: Float64 = 0.02
comptime LINK_RAD_1: Float64 = 0.017
comptime LINK_RAD_2: Float64 = 0.014

# Everything from the wrist outwards — the hand capsule, both palms, both
# fingers and both fingertips — bounded by one disc about the wrist origin. The
# furthest reachable point is palm tip (.054) + thumb (.05 + .01) + tip radius
# (.008) ~ .12; .13 rounds that up. Conservative on purpose: a disc is what makes
# the test a SUBSET of the reference's acceptance region, since the fingers move
# with two joints this test never reads.
comptime HAND_DISC_RAD: Float64 = 0.13


def dist_point_segment(
    px: Float64, pz: Float64,
    ax: Float64, az: Float64,
    bx: Float64, bz: Float64,
) -> Float64:
    """Distance from (px, pz) to the segment a->b, in the x-z plane."""
    var dx = bx - ax
    var dz = bz - az
    var l2 = dx * dx + dz * dz
    var t = 0.0
    if l2 > 0.0:
        t = ((px - ax) * dx + (pz - az) * dz) / l2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
    var cx = px - (ax + t * dx)
    var cz = pz - (az + t * dz)
    return sqrt(cx * cx + cz * cz)


def arm_clearance(
    q0: Float64, q1: Float64, q2: Float64, q3: Float64,
    px: Float64, pz: Float64,
) -> Float64:
    """Gap in metres between the point (px, pz) and the arm's SURFACE.

    Negative means inside. Exact for the three arm capsules (a capsule's surface
    is exactly `distance-to-axis minus radius`) and conservative for everything
    past the wrist, which is bounded by `HAND_DISC_RAD`.

    Both domains' `initialize_episode` rejects a draw while
    `physics.data.ncon > 0`, which needs full collision detection from inside a
    reset hook and is not available there. Comparing a prop's BOUNDING RADIUS
    against this clearance instead makes the accepted region a strict SUBSET of
    the reference's: everything this accepts, the reference would also accept.
    The distinction matters, because an approximate region drifts silently and a
    subset cannot.

    ⚠ A conservative bound is only sound if the region it LEAVES is still
    sampleable, and nothing in the code says which. The first version of this
    rejected anything within ARM_REACH (.62 m — the arm's full extension) of the
    shoulder, which is perfectly sound and accepts 0.13% of draws, so 77% of
    resets exhausted their retry budget and fell through with an arbitrary,
    usually arm-penetrating placement. Testing the arm where it actually IS
    rather than where it could reach takes that to 88% (ball) / 70% (peg).
    Measure the acceptance rate before trusting a rejection sampler.

    `q3` (`arm_wrist`) only rotates the hand about the wrist origin, which the
    disc already covers, so it is accepted and unused — spelled out rather than
    dropped from the signature so a future non-disc hand model has it.
    """
    var x = SHOULDER_X
    var z = SHOULDER_Z
    var th = 0.0
    var best = 1.0e18

    var lens = [LINK_LEN_0, LINK_LEN_1, LINK_LEN_2]
    var rads = [LINK_RAD_0, LINK_RAD_1, LINK_RAD_2]
    var angs = [q0, q1, q2]
    for i in range(3):
        th += angs[i]
        var nx = x - lens[i] * sin(th)
        var nz = z + lens[i] * cos(th)
        var g = dist_point_segment(px, pz, x, z, nx, nz) - rads[i]
        if g < best:
            best = g
        x = nx
        z = nz

    # Wrist origin: the hand assembly, as one disc.
    var hx = px - x
    var hz = pz - z
    var gh = sqrt(hx * hx + hz * hz) - HAND_DISC_RAD
    if gh < best:
        best = gh
    return best
