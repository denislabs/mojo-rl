# Generic Gymnasium wrapper
from .gymnasium_wrapper import GymnasiumEnv

# Classic Control environments
from .gymnasium_classic_control import (
    CartPoleEnv,
    MountainCarEnv,
    PendulumEnv,
    AcrobotEnv,
    discretize_mountain_car,
    discretize_acrobot,
    discretize_cart_pole,
    get_cart_pole_num_states,
    get_mountain_car_num_states,
    get_acrobot_num_states,
)

# Box2D environments
from .gymnasium_box2d import (
    LunarLanderEnv,
    BipedalWalkerEnv,
    CarRacingEnv,
    discretize_lunar_lander,
)

# Toy Text environments (Gymnasium wrappers)
from .gymnasium_toy_text import (
    GymFrozenLakeEnv,
    GymTaxiEnv,
    BlackjackEnv,
    GymCliffWalkingEnv,
)

# MuJoCo environments
from .gymnasium_mujoco import (
    MuJoCoEnv,
    make_half_cheetah,
    make_ant,
    make_humanoid,
    make_walker2d,
    make_hopper,
    make_swimmer,
    make_inverted_pendulum,
    make_inverted_double_pendulum,
    make_reacher,
    make_pusher,
)
