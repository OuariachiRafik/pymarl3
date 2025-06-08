from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv
from .one_step_matrix_game import OneStepMatrixGame
from .mpe.mpe_wrapper import MPEEnv
import envs.mpe.multiagent.scenarios as scenarios

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

def env_fn_mpe(env, **kwargs)-> MultiAgentEnv:
    scenario_name="simple"
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MPEEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MPEEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env    

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
REGISTRY["particle"] = partial(env_fn_mpe, env=MPEEnv)
print("Supported environments:", REGISTRY)
