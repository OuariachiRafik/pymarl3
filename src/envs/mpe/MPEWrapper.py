from pettingzoo.mpe import simple_spread_v3, simple_tag_v3, simple_adversary_v3
from envs.multiagentenv import MultiAgentEnv
import numpy as np

class MPEWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.map_name = kwargs.get("map_name", "simple_spread_v3")
        self.episode_limit = kwargs.get("episode_limit", 25)
        env_map = {
        "simple_spread_v3": simple_spread_v3.parallel_env,
        "simple_tag_v3": simple_tag_v3.parallel_env,
        "simple_adversary_v3": simple_adversary_v3.parallel_env,
        }

        self.env = env_map[self.map_name]()
        self.env.reset()
        self.env = simple_spread_v3.parallel_env()
        self.env.reset()
        self.n_agents = len(self.env.agents)
        self.obs_shape = [self.env.observation_space(agent).shape[0] for agent in self.env.agents]
        self.state_shape = sum(self.obs_shape)
        self.n_actions = self.env.action_space(self.env.agents[0]).n

    def step(self, actions):
        action_dict = {agent: action for agent, action in zip(self.env.agents, actions)}
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
        terminated = any(terminations.values())
        obs_list = [obs[agent_id] for agent_id in range(self.n_agents-1)]
        reward_list = [rewards[agent_id] for agent_id in range(self.n_agents-1)]
        return reward_list, terminated, obs_list, infos

    def reset(self):
        obs = self.env.reset()
        return [obs[agent_id] for agent_id in range(len(self.env.agents))]

    def get_obs(self):
        return [self.env.observe(agent) for agent in self.env.agents]

    def get_state(self):
        return np.concatenate(self.get_obs())

    def get_avail_actions(self):
        return [np.ones(self.n_actions) for _ in range(self.n_agents)]

    def get_obs_size(self):
        return self.obs_shape[0]

    def get_state_size(self):
        return self.state_shape

    def get_total_actions(self):
        return self.n_actions

    def close(self):
        self.env.close()
