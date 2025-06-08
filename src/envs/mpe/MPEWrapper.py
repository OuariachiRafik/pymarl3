import numpy as np
from typing import List, Dict, Any
# import your gym-based env
from envs.mpe.multiagent import environment as MultiAgentEnv 
# import the abstract interface
from envs import MultiAgentEnv as AbstractMultiAgentEnv

class MPEWrapper(AbstractMultiAgentEnv):
     def __init__(self, **kwargs):
        self._env = MultiAgentEnv(
            **kwargs
        )
        # number of agents
        self.n_agents = self._env.n
        # max episode length (for env_info)
        self.episode_limit = 25
        # storage for the latest step outputs
        self._last_obs: List[np.ndarray] = []
        self._last_state: np.ndarray = None

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
        # Assuming all agents share the same discrete action space:
        ac0 = self._env.action_space[0]
        if hasattr(ac0, 'n'):
            return ac0.n
        else:
            # continuous or tuple; raise or return -1
            raise NotImplementedError("Total actions undefined for non-discrete spaces.")

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

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
