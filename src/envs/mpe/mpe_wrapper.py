import numpy as np
from .multiagent.environment import MultiAgentEnv as MPECoreEnv
from .multiagent import scenario
from envs.multiagentenv import MultiAgentEnv


class MPEEnv(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Load scenario
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        scenario_name=args.scenario_name
        scenario = scenario.load(scenario_name + ".py").Scenario()
        world = scenario.make_world()
        self.env = MPECoreEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
        
        self.n_agents = self.env.n
        self.n_actions = self.env.action_space[0].n
        self.episode_limit = args.get("episode_limit", 25)

        self.obs_shape = [obs.shape[0] for obs in self.env.observation_space]
        self.state_shape = sum(self.obs_shape)  # You can customize this

        self._episode_step = 0

    def step(self, actions):
        self._episode_step += 1
        obs, rewards, dones, infos = self.env.step(actions)
        return rewards, all(dones), infos

    def get_obs(self):
        return [self.env._get_obs(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return self.env._get_obs(agent_id)

    def get_obs_size(self):
        return self.obs_shape[0]

    def get_state(self):
        return np.concatenate(self.get_obs(), axis=0)

    def get_state_size(self):
        return self.state_shape

    def get_avail_actions(self):
        return [np.ones(self.n_actions) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.n_actions)

    def get_total_actions(self):
        return self.n_actions

    def reset(self):
        self._episode_step = 0
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def get_env_info(self):
        return {
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "episode_limit": self.episode_limit,
        }
