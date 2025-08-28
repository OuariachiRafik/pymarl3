import copy
import time

import torch as th
from torch.optim import RMSprop, Adam
import numpy as np

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num

from modules.CMImasker import CMIMasker, CMIMaskerConfig  
from modules.semantic_state import (
    from_state_layout,
    StateBlockEncoder, StateBlockEncoderConfig,
    StateAdapter,
)
#hro

def calculate_target_q(target_mac, batch, enable_parallel_computing=False, thread_num=4):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)
    with th.no_grad():
        # Set target mac to testing mode
        target_mac.set_evaluation_mode()
        target_mac_out = []
        target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
        return target_mac_out


def calculate_n_step_td_target(target_mixer, target_max_qvals, states_for_mixer, rewards, terminated, mask, gamma, td_lambda,
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        target_max_qvals = target_mixer(target_max_qvals, states_for_mixer)

        if q_lambda:
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()


class NQLearner:
    def __init__(self, mac, scheme, logger, args, state_layout):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":  # 31.521K
            self.mixer = Mixer(args)
        else:
            raise "mixer error"

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        self.use_state_blocks = getattr(args, "state_blocks_enabled", True)
        if self.use_state_blocks:
             # parse centralized state layout into semantic slices
             layout = state_layout
             slices = from_state_layout(layout)
             state_dim = args.state_shape

             sb_cfg = StateBlockEncoderConfig(
                 ally_latent_dim=getattr(args, "ally_latent_dim", 16),
                 enemy_latent_dim=getattr(args, "enemy_latent_dim", 16),
                 hist_latent_dim=getattr(args, "hist_latent_dim", 8),
                 comp_latent_dim=getattr(args, "comp_latent_dim", 6),
                 geom_latent_dim=getattr(args, "geom_latent_dim", 8),
                 set_hidden=getattr(args, "state_blocks_set_hidden", 64),
                 mlp_hidden=getattr(args, "state_blocks_mlp_hidden", 64),
                 dropout=getattr(args, "state_blocks_dropout", 0.0),
                 ally_pos_offset=getattr(args, "ally_pos_offset", 2),    # allies (x,y) start index in per-unit vector
                 enemy_pos_offset=getattr(args, "enemy_pos_offset", 1),
                 transition_pretrain=getattr(args, "state_blocks_transition_pretrain", False),
                 transition_lr=getattr(args, "state_blocks_transition_lr", 3e-4),
             )
             self.block_encoder = StateBlockEncoder(slices, sb_cfg).to(self.args.device)
             self.latent_dim = self.block_encoder.latent_dim
             # adapter to map masked latents back to mixer-expected state_dim

             self.state_adapter = StateAdapter(self.latent_dim, state_dim).to(self.args.device)
             self.params += list(self.state_adapter.parameters())
        else:
             self.block_encoder = None
             self.state_adapter = None
             self.latent_dim = args.state_shape



        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.use_cmi_mask = getattr(args, "cmi_mask_enabled", True)
        if self.use_cmi_mask:
             #state_dim = args.state_shape  # int
             state_dim = self.latent_dim
            # joint-action dim: concatenate per-agent one-hots (or logits) across n_agents
             act_dim = args.n_agents * args.n_actions
             print("######################################################## act_dim = ", act_dim, "n_actions = ", args.n_actions)
             cmicfg = CMIMaskerConfig(
                 state_dim=state_dim,
                 act_dim=act_dim,
                 feat_dim=getattr(args, "cmi_feat_dim", 64),
                 head_hidden=getattr(args, "cmi_head_hidden", 64),
                 pool=getattr(args, "cmi_pool", "max"),
                 lr=getattr(args, "cmi_lr", 3e-4),
                 ema_decay=getattr(args, "cmi_ema_decay", 0.999),
                 eval_interval=getattr(args, "cmi_eval_interval", 10),
                 val_split=getattr(args, "cmi_val_split", 0.1),
                 threshold=getattr(args, "cmi_threshold", 1e-1),
                 refresh_stride=getattr(args, "cmi_refresh_stride", 1000),
                 device="cuda" if args.use_cuda else "cpu",
             )
             self.cmi_masker = CMIMasker(cmicfg)
        else:
             self.cmi_masker = None
        
        self.use_intrinsic_rewards = getattr(args, "use_intrinsic_rewards", True)
        self.causal_beta  = getattr(args, "intrinsic_rewards_beta", 0.5)
        self.causal_tau   = getattr(args, "intrinsic_rewards_tau", 1.0)
        self.causal_norm  = getattr(args, "intrinsic_rewards_norm", "ema")
        self.causal_clip  = getattr(args, "intrinsic_rewards_clip", 5.0)

        # EMA stats for normalization
        self._gap_mean = 0.0
        self._gap_std  = 1.0
        self._gap_ema_decay = 0.999

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0

        self.enable_parallel_computing = (not self.args.use_cuda) and getattr(self.args, 'enable_parallel_computing',
                                                                              False)
        # self.enable_parallel_computing = False
        if self.enable_parallel_computing:
            from multiprocessing import Pool
            # Multiprocessing pool for parallel computing.
            self.pool = Pool(1)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        # Get the relevant quantities
        states = batch["state"][:, :-1]            # [B, T, state_dim]
        next_states = batch["state"][:, 1:]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

# -------- Encode centralized state to semantic latents z ----------
        if self.use_state_blocks:
            z_t = self.block_encoder.encode(states)          # [B,T,dz]
            z_tp1 = self.block_encoder.encode(next_states)   # [B,T,dz]
        else:
            z_t = states
            z_tp1 = next_states


        # Optional: pretrain block encoder transitions
        if self.use_state_blocks and getattr(self.args, "state_blocks_transition_pretrain", True):
            B, T, _ = z_t.shape
            n_ag, n_ac = self.args.n_agents, self.args.n_actions
            try:
                Aoh_btna = batch["actions_onehot"][:, :-1]  # [B,T,n_agents,n_actions]
            except KeyError:
                # actions is indices [B,T,n_agents,1] -> one-hot
                Aoh_btna = th.zeros(B, T, n_ag, n_ac, device=actions.device, dtype=th.float32)
                Aoh_btna.scatter_(3, actions.long(), 1.0)
            A_flat = Aoh_btna.reshape(B*T, n_ag * n_ac).float()
            loss_enc = self.block_encoder.step_pretrain(
                z_t.new_tensor(states.reshape(B*T, -1)),
                A_flat,
                z_tp1.new_tensor(next_states.reshape(B*T, -1)),
            )

        # ---- UPDATE CMI MASKER -----------------------------------------
        if self.use_cmi_mask:
            # Build joint action one-hot per step: [B*T, n_agents * n_actions]
            B, T, _ = z_t.shape
            n_ag, n_ac = self.args.n_agents, self.args.n_actions
            try:
                Aoh_btna = batch["actions_onehot"][:, :-1]  # [B,T,n_agents,n_actions]
            except KeyError:
                # actions is indices [B,T,n_agents,1] -> one-hot
                Aoh_btna = th.zeros(B, T, n_ag, n_ac, device=actions.device, dtype=th.float32)
                Aoh_btna.scatter_(3, actions.long(), 1.0)
            A_flat = Aoh_btna.reshape(B*T, n_ag * n_ac).float()
            #A_flat = actions.reshape(B*T, n_ag * n_ac).float()
            #S_flat = states.reshape(B*T, -1).float()
            #Sp_flat = batch["state"][:, 1:].reshape(B*T, -1).float()  # next state
            Z_flat  = z_t.reshape(B*T, -1).float()
            Zp_flat = z_tp1.reshape(B*T, -1).float()
            
            print("######################################################## A_flat shape = ", A_flat.shape)
            print("######################################################## Z_flat shape = ", Z_flat.shape)
            print("######################################################## Zp_flat shape = ", Zp_flat.shape)

            sample_size = 5000
            total_samples = Z_flat.shape[0]
            sample_size = min(sample_size, total_samples)  
            sample_indices = np.random.choice(total_samples, sample_size, replace=False)

            cmi_logs = self.cmi_masker.step_train_minibatch(Z_flat[sample_indices], A_flat[sample_indices], Zp_flat[sample_indices])



        if self.use_state_blocks and self.use_cmi_mask and self.use_intrinsic_rewards:
            with th.no_grad():
                # (i) compute per-transition gap on the whole mini-batch
                gap = self.cmi_masker.prediction_gap(Z_flat, A_flat, Zp_flat, sum_over_k=True)  # [B*T]

                # (ii) normalize & clip
                if getattr(self, "causal_norm", "ema") == "batch":
                    gmu = gap.mean()
                    gst = gap.std(unbiased=False).clamp_min(1e-6)
                    gap = (gap - gmu) / gst
                elif getattr(self, "causal_norm", "ema") == "ema":
                    gmu = float(gap.mean().item())
                    gst = float(gap.std(unbiased=False).clamp_min(1e-6).item())
                    d = getattr(self, "_gap_ema_decay", 0.999)
                    self._gap_mean = d * getattr(self, "_gap_mean", 0.0) + (1 - d) * gmu
                    self._gap_std  = d * getattr(self, "_gap_std", 1.0)  + (1 - d) * gst
                    gap = (gap - self._gap_mean) / max(self._gap_std, 1e-6)

                clip_v = getattr(self, "causal_clip", 5.0)
                if clip_v and clip_v > 0:
                    gap = gap.clamp(min=-clip_v, max=clip_v)

                # (iii) squash (tanh) and scale
                tau  = getattr(self, "causal_tau", 1.0)
                beta = getattr(self, "causal_beta", 0.5)
                r_pd = th.tanh(tau * gap).view(B, T, 1) * beta    # [B,T,1]

            rewards_for_td = rewards + r_pd   # rewards is [B,T,1]
        else:
            rewards_for_td = rewards
            
        if self.enable_parallel_computing:
            target_mac_out = self.pool.apply_async(
                calculate_target_q,
                (self.target_mac, batch, True, self.args.thread_num)
            )

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # TODO: double DQN action, COMMENT: do not need copy
        mac_out[avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            if self.enable_parallel_computing:
                target_mac_out = target_mac_out.get()
            else:
                target_mac_out = calculate_target_q(self.target_mac, batch)
            
            target_avail = batch["avail_actions"][:, 1:]                  # [B, T, n_agents, n_actions]
            target_mac_out[:, 1:][target_avail == 0] = -9999999   
            # Max over target Q-Values/ Double q learning
            # mac_out_detach = mac_out.clone().detach()
            # TODO: COMMENT: do not need copy
            mac_out_detach = mac_out
            # mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]

            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)


            # Masking
            if self.use_cmi_mask:
                #M = self.cmi_masker.get_state_mask().detach()  # [state_dim]
                #M = M.view(1, 1, -1)  # broadcast over [B,T,state_dim]
                M = self.cmi_masker.get_state_mask().detach().view(1,1,-1)  # [1,1,dz]
                print("Causal_Mask:", M)
                z_masked = z_t * M
                #states_masked = states * M  # [B,T,state_dim]
            else:
                z_masked = z_t
                #states_masked = states

            if self.use_state_blocks:
                states_masked = self.state_adapter(z_masked)  # [B,T,state_dim]
                chosen_action_qvals = self.mixer(chosen_action_qvals, states_masked)
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals, states) #hro
            

            if self.use_state_blocks:
                z_masked_tp1 = (z_tp1 * M) if (self.use_cmi_mask) else z_tp1
                states_masked_tp1 = self.state_adapter(z_masked_tp1)  # [B,T,state_dim] aligned with target_max_qvals
            else:
                states_masked_tp1 = next_states
                
            assert getattr(self.args, 'q_lambda', False) == False

            if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
                targets = self.pool.apply_async(
                    calculate_n_step_td_target,
                    (self.target_mixer, target_max_qvals, states_masked_tp1, rewards_for_td, terminated, mask, self.args.gamma,
                     self.args.td_lambda, True, self.args.thread_num, False, None)
                )
            else:
                targets = calculate_n_step_td_target(
                    self.target_mixer, target_max_qvals, states_masked_tp1, rewards_for_td, terminated, mask, self.args.gamma,
                    self.args.td_lambda
                )

        # Set mixing net to training mode
        self.mixer.train()

        if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
            targets = targets.get()

        td_error = (chosen_action_qvals - targets)
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        mask_elems = mask.sum()
        loss = masked_td_error.sum() / mask_elems

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # For log
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            #self.logger.log_stat("blocks/transition_loss", loss_enc, t_env)
            #self.logger.log_stat("cmi_logs", cmi_logs, t_env)
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("target_mean", target_mean, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def __del__(self):
        if self.enable_parallel_computing:
            self.pool.close()
