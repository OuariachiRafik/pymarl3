import numpy as np
from typing import List, Dict, Any
# import your gym-based env
from envs.mpe.multiagent.environment import  MultiAgentEnv 
# import the abstract interface
from envs import MultiAgentEnv as AbstractMultiAgentEnv

import envs.mpe.multiagent.scenarios as scenarios
from gym.spaces import Discrete, MultiDiscrete, Tuple, Box

class MPEWrapper(AbstractMultiAgentEnv):
    def __init__(self, **kwargs):
        # load scenario from script
        benchmark = False
        scenario_name = kwargs.pop('scenario')
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        if benchmark:        
            self._env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
        else:
            self._env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
        # number of agents
        self.n_agents = self._env.n
        # max episode length (for env_info)
        self.episode_limit = 25
        
        init_obs = self._env.reset()
        self._last_obs   = init_obs
        self._last_state = np.concatenate(init_obs, axis=0)
        
        single_obs_space = self._env.observation_space[0]
        self.obs_shape = single_obs_space.shape[0]

        # 4) state_shape: we’ll define the global state as the
        #    concatenation of all agents’ observations
        self.state_shape = sum(
            space.shape[0] for space in self._env.observation_space
        )

        # 5) n_actions: total number of discrete actions per agent
        first_aspace = self._env.action_space[0]
        if hasattr(first_aspace, 'n'):
            # simple Discrete
            self.n_actions = first_aspace.n
        else:
            # if it’s a MultiDiscrete, multiply the sizes of each sub-space
            try:
                sizes = (first_aspace.high - first_aspace.low + 1).astype(int)
                self.n_actions = int(np.prod(sizes))
            except Exception:
                raise NotImplementedError(
                    "Cannot infer n_actions for this action space"
                )

        
    def reset(self) -> (List[np.ndarray], np.ndarray):
        """Returns initial obs list and initial global state."""
        obs_n = self._env.reset()
        self._last_obs = obs_n
        # For a “state” you could concatenate all observations, or
        # call some custom world-state extractor if available.
        self._last_state = np.concatenate(obs_n, axis=0)
        return obs_n, self._last_state

    def step(self, actions: List) -> (List[float], bool, Dict[str, Any]):
        """
        actions: list of per-agent actions
        returns: reward list, terminated flag, info dict
        """
        obs_n, reward_n, done_n, info_n = self._env.step(actions)
        self._last_obs = obs_n
        self._last_state = np.concatenate(obs_n, axis=0)
        # terminated when all agents done or any global condition
        terminated = all(done_n)
        return reward_n, terminated, info_n

    def get_obs(self) -> List[np.ndarray]:
        return self._last_obs

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self._last_obs[agent_id]

    def get_obs_size(self) -> int:
        # assume all obs have same shape
        return self._last_obs[0].shape[0]

    def get_state(self) -> np.ndarray:
        return self._last_state

    def get_state_size(self) -> int:
        return self._last_state.shape[0]

    def get_avail_actions(self) -> List[np.ndarray]:
        """
        Returns, for each agent, a binary mask of available actions.
        Here we assume fully available discrete actions [0..n-1].
        """
        masks = []
        for a in range(self.n_agents):
            ac_space = self._env.action_space[a]
            if hasattr(ac_space, 'n'):
                masks.append(np.ones(ac_space.n, dtype=np.int32))
            else:
                # continuous – treat all as available
                masks.append(None)
        return masks

    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self) -> int:
        ac0 = self._env.action_space[0]
    
        # 1) Discrete
        if isinstance(ac0, Discrete):
            return ac0.n
    
        # 2) MultiDiscrete
        if isinstance(ac0, MultiDiscrete):
            # sizes is array of (high - low + 1)
            sizes = ac0.high - ac0.low + 1
            return int(np.prod(sizes))
    
        # 3) Tuple (mix of Discrete / MultiDiscrete / Box)
        if isinstance(ac0, Tuple):
            total = 1
            for space in ac0.spaces:
                if isinstance(space, Discrete):
                    total *= space.n
                elif isinstance(space, MultiDiscrete):
                    sizes = space.high - space.low + 1
                    total *= int(np.prod(sizes))
                elif isinstance(space, Box):
                    total *= int(np.prod(space.shape))
                else:
                    # unknown sub‐space
                    return -1
            return total
    
        # 4) Box (continuous)
        if isinstance(ac0, Box):
            return int(np.prod(ac0.shape))
    
        # 5) fallback
        return -1

    def render(self) -> Any:
        return self._env.render()

    def close(self):
        # if your gym env has a close method:
        if hasattr(self._env, 'close'):
            self._env.close()

    def seed(self, seed: int):
        if hasattr(self._env, 'seed'):
            self._env.seed(seed)

    def save_replay(self):
        if hasattr(self._env, 'save_replay'):
            self._env.save_replay()

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
