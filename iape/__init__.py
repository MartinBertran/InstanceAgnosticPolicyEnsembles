# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.


# Algorithms

from iape.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from iape.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from iape.algos.pytorch.sac.sac import sac as sac_pytorch
from iape.algos.pytorch.td3.td3 import td3 as td3_pytorch
from iape.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from iape.algos.pytorch.vpg.vpg import vpg as vpg_pytorch
from iape.iape.deviant import deviant as deviant_pytorch

# Loggers
from iape.iape.backend.logger import EpochLogger,Logger

# Version
from iape.version import __version__