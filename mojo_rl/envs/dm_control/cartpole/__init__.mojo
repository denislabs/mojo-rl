"""`dm_control` `cartpole` domain.

Tasks: balance, balance_sparse, swingup, swingup_sparse, two_poles, three_poles.
Reference: references/dm_control-main/dm_control/suite/cartpole.py + .xml
"""

from .cartpole import (
    DMCartpoleBalance,
    DMCartpoleBalanceBatched,
    DMCartpoleBalanceSparse,
    DMCartpoleBalanceSparseBatched,
    DMCartpoleSwingup,
    DMCartpoleSwingupBatched,
    DMCartpoleSwingupSparse,
    DMCartpoleSwingupSparseBatched,
    DMCartpoleTwoPoles,
    DMCartpoleTwoPolesBatched,
    DMCartpoleThreePoles,
    DMCartpoleThreePolesBatched,
)
from .cartpole_config import DMCartpoleConfig
from .cartpole_xml import (
    DMCartpole1Model,
    DMCartpole2Model,
    DMCartpole3Model,
    CART_BODY_IDX,
    FIRST_POLE_BODY_IDX,
)
