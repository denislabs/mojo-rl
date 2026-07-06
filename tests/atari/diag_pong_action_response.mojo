"""Diagnostic: does the agent's paddle actually RESPOND to actions? If the
agent can never move its paddle (or never scores), no policy/imagination can
learn Pong — eval stays -21 forever regardless of the world model.

For each of the 6 Pong actions, from a fresh reset, hold that action for N steps
and report the Y-centroid of bright pixels in the LEFT paddle band (opponent) and
RIGHT paddle band (agent). If the RIGHT (agent) centroid is ~identical across all
actions, the agent's paddle doesn't respond → root cause is the control/action
path, not the WM. If actions move it up/down, the control loop is fine.

Run: pixi run -e apple mojo run -I . tests/atari/diag_pong_action_response.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.envs.atari import AtariEnv
from mojo_rl.envs.atari.games.registry import AtariGame

comptime IMG = 96
comptime OBS = IMG * IMG
comptime NSTEP = 30


def _centroid_y(
    obs: List[Scalar[DT]], x0: Int, x1: Int
) -> Tuple[Scalar[DT], Int]:
    # Weighted Y centroid of bright (>0.5) pixels in column band [x0,x1),
    # over the play-area rows [14,90). Returns (centroid_y, bright_count).
    var sy = Scalar[DT](0.0)
    var n = 0
    for y in range(14, 90):
        for x in range(x0, x1):
            if obs[y * IMG + x] > Scalar[DT](0.5):
                sy += Scalar[DT](y)
                n += 1
    if n == 0:
        return (Scalar[DT](-1.0), 0)
    return (sy / Scalar[DT](n), n)


def main() raises:
    print("=" * 64)
    print("Pong action response — does the AGENT paddle move? (hold each")
    print("action", NSTEP, "steps from reset; report L/R paddle Y-centroid)")
    print("=" * 64)
    print("  action   LEFT(opp) y / n     RIGHT(agent) y / n")
    var rys = List[Scalar[DT]]()
    for a in range(6):
        var env = AtariEnv[3, DT](AtariGame.PONG)
        var obs = env.reset_obs_list()
        for _ in range(NSTEP):
            obs = env.step_obs(a)[0].copy()
        var lc = _centroid_y(obs, 4, 16)   # left/opponent paddle band
        var rc = _centroid_y(obs, 80, 92)  # right/agent paddle band
        rys.append(lc[0] if rc[1] == 0 else rc[0])
        print(
            "   ", a, "     ", lc[0], "/", lc[1], "      ", rc[0], "/", rc[1]
        )
        env.close()
        _ = env^

    # Range of the agent-paddle centroid across actions.
    var mn = rys[0]
    var mx = rys[0]
    for i in range(len(rys)):
        if rys[i] >= Scalar[DT](0.0):
            if mn < Scalar[DT](0.0) or rys[i] < mn:
                mn = rys[i]
            if rys[i] > mx:
                mx = rys[i]
    print("-" * 64)
    print("  agent-paddle Y spread across actions =", mx - mn, "px")
    if mx - mn < Scalar[DT](3.0):
        print("  → agent paddle barely moves regardless of action")
        print("    ⇒ ROOT CAUSE is the control/action path (not the WM).")
    else:
        print("  → agent paddle DOES respond to actions (control loop OK);")
        print("    the failure is elsewhere (WM/actor/reward).")
    print("=" * 64)
