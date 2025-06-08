from pettingzoo.mpe import simple_spread_v2
from envs.multiagentenv import MultiAgentEnv
import numpy as np

class MPEWrapper(MultiAgentEnv):
    def __init__(self, args):
        self.env = simple_spread_v2.parallel_env()
        self.env.reset()
        self.n_agents = len(self.env.agents)
        self.obs_shape = [self.env.observation_space(agent).shape[0] for agent in self.env.agents]
        self.state_shape = sum(self.obs_shape)
        self.n_actions = self.env.action_space(self.env.agents[0]).n

    def step(self, actions):
        action_dict = {agent: action for agent, action in zip(self.env.agents, actions)}
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
        terminated = any(terminations.values())
        obs_list = [obs[agent] for agent in self.env.agents]
        reward_list = [rewards[agent] for agent in self.env.agents]
        return reward_list, terminated, obs_list, infos

    def reset(self):
        obs = self.env.reset()
        return [obs[agent] for agent in self.env.agents]

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
