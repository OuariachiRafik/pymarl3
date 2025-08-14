from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import time
from utils.SMACv2StateSlicer import SMACv2StateSlicer
from modules.encoders.state_semantic_encoder import StateSemanticEncoder

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        if self.batch_size > 1:
            self.batch_size = 1
            logger.console_logger.warning("Reset the `batch_size_run' to 1...")

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        if self.args.evaluate:
            print("Waiting the environment to start...")
            time.sleep(5)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        
        ####hro
        #StateSlicer
        info = infer_state_layout_from_env(self.env)
        self.state_slicer = SMACv2StateSlicer(
            n_allies=info["n_allies"],
            n_enemies=info["n_enemies"],
            ally_feat_dim=info["ally_feat_dim"],
            enemy_feat_dim=info["enemy_feat_dim"],
            # pass these flags if your slicer accounts for extras
            include_last_action=info["state_last_action"],
            include_timestep=info["state_timestep_number"]
            )
        #StateSlicer
        #SemanticEncoder
        self.semantic_encoder = StateSemanticEncoder(
            ally_dim=info["ally_feat_dim"],
            enemy_dim=info["enemy_feat_dim"],
            n_allies=info["n_allies"],
            n_enemies=info["n_enemies"],
            action_dim=(self.n_actions if args.use_last_action_in_semantic else 0),
            out_dim=args.state_semantic_dim
            ).to(self.device)
        #SemanticEncoder
        ####hro

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        if self.args.use_cuda and not self.args.cpu_inference:
            self.batch_device = self.args.device
        else:
            self.batch_device = "cpu" if self.args.buffer_cpu_only else self.args.device
        print(" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& self.batch_device={}".format(
            self.batch_device))
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.batch_device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        if (self.args.use_cuda and self.args.cpu_inference) and str(self.mac.get_device()) != "cpu":
            self.mac.cpu()  # copy model to cpu

        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            ####hro
            # state -> ally/enemy tensors (+ masks)
            ally_feats, enemy_feats, ally_mask, enemy_mask = self.state_slicer(pre_transition_data["state"])  # expects [B,1,S]

            # optional last-action one-hot per ally (maintained on the runner)
            ally_last_act_oh = None
            if self.args.use_last_action_in_semantic:
                # self.last_actions_oh: [B,n_agents,n_actions] kept up to date after each env.step()
                ally_last_act_oh = self.last_actions_oh.unsqueeze(1)  # -> [B,1,n_agents,n_actions]

            # encode to semantic state
            z = self.semantic_encoder(
                ally_feats, enemy_feats, ally_mask, enemy_mask,
                ally_last_act_oh=ally_last_act_oh
            )  # [B,1,Dz]

            # write it with the rest of pre-transition data
            pre_transition_data["state_semantic"] = z
            ####hro

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            if self.args.evaluate:
                time.sleep(1)
                print(self.t, post_transition_data["reward"])

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_min", np.min(returns), self.t_env)
        self.logger.log_stat(prefix + "return_max", np.max(returns), self.t_env)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

####hro
def infer_state_layout_from_env(env):
    """
    Returns: dict with n_allies, n_enemies, ally_feat_dim, enemy_feat_dim,
    and flags about optional extras (last actions, timestep) you might care about.
    Works across Subproc/Parallel wrappers by unwrapping safely.
    """
    # unwrap common wrapper layers
    base = env
    for attr in ("env", "wrapped_env", "unwrapped", "raw_env"):
        base = getattr(base, attr, base)

    # In SMAC/SMACv2 these are standard
    n_allies  = getattr(base, "n_agents")
    n_enemies = getattr(base, "n_enemies")

    # Prefer state-specific sizing if present; otherwise fall back to obs sizes
    get = lambda name: getattr(base, name)() if hasattr(base, name) else None

    ally_feat_dim  = get("get_state_ally_feats_size") or get("get_obs_ally_feats_size")
    enemy_feat_dim = get("get_state_enemy_feats_size") or get("get_obs_enemy_feats_size")

    if ally_feat_dim is None or enemy_feat_dim is None:
        raise RuntimeError("Could not infer per-unit feature sizes from env. "
                           "Check your SMAC/SMACv2 version or wrappers.")

    # Optional extras sometimes included in the global state
    state_last_action    = bool(getattr(base, "state_last_action", False))
    state_timestep_number = bool(getattr(base, "state_timestep_number", False))

    return {
        "n_allies": n_allies,
        "n_enemies": n_enemies,
        "ally_feat_dim": ally_feat_dim,
        "enemy_feat_dim": enemy_feat_dim,
        "state_last_action": state_last_action,
        "state_timestep_number": state_timestep_number,
    }
####hro