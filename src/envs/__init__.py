from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv
from .one_step_matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt
from .grf import Academy_3_vs_1_with_keeper, Run_pass_and_shoot_with_keeper, Pass_and_shoot_with_keeper

try:
    smac = True
    from .smac_v1 import StarCraft2EnvWrapper
except Exception as e:
    print(e)
    smac = False

try:
    smacv2 = True
    from .smac_v2 import StarCraft2Env2Wrapper
except Exception as e:
    print(e)
    smacv2 = False


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

if smac:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2EnvWrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V1 is not supported...")

if smacv2:
    REGISTRY["sc2_v2"] = partial(env_fn, env=StarCraft2Env2Wrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V2 is not supported...")
    
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["academy_3_vs_1_with_keeper"] = partial(env_fn, env=Academy_3_vs_1_with_keeper),
REGISTRY["run_pass_and_shoot_with_keeper"] = partial(env_fn, env=Run_pass_and_shoot_with_keeper), 
REGISTRY["pass_and_shoot_with_keeper"] = partial(env_fn, env=Pass_and_shoot_with_keeper)
print("Supported environments:", REGISTRY)
