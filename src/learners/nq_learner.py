import copy
import time
import math #CausalHRO


import numpy as np #CausalHRO

import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num

#from envs.one_step_matrix_game import print_matrix_status #CausalHRO
from utils.causal_weight import get_a2s_weight,get_s2s_weight, mask_irrelevant_states, mask_irrelevant_actions #CausalHRO

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


def calculate_n_step_td_target(target_mixer, target_max_qvals, batch, rewards, terminated, mask, gamma, td_lambda,
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        target_max_qvals = target_mixer(target_max_qvals, batch["state"])

        if q_lambda:
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()


class NQLearner:
    def __init__(self, mac, scheme, logger, args):
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

        self.entropy_coef = 0.03 #CausalHRO

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0

        #CausalHRO
        self.weight_s2s = None
        self.weight_a2s = None
        
        self.n_agents = args.n_agents
        self.causal_default_weight = np.ones(self.n_agents, dtype=np.float32)
        #CausalHRO

        self.enable_parallel_computing = (not self.args.use_cuda) and getattr(self.args, 'enable_parallel_computing',
                                                                              False)
        # self.enable_parallel_computing = False
        if self.enable_parallel_computing:
            from multiprocessing import Pool
            # Multiprocessing pool for parallel computing.
            self.pool = Pool(1)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, causal_update):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        #CausalHRO
        #Print Causal Update
        print('causal_update', causal_update)
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1].to(self.device)
        actions = batch["actions"][:, :-1].to(self.device)
        terminated = batch["terminated"][:, :-1].float().to(self.device)
        mask = batch["filled"][:, :-1].float().to(self.device)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]).to(self.device)
        avail_actions = batch["avail_actions"].to(self.device)

        #CausalHRO
        if causal_update==1000:#causal_update % 1000 == 0 and causal_update >= 1000:
            per_agent_weights = []
            total_time = 0.0
            s2s, t_s = get_s2s_weight(batch)
            a2s, t_a = get_a2s_weight(batch)
            total_time = t_s + t_a
            self.weight_s2s = s2s
            self.weight_a2s = a2s
            print("Causal action to state matrix: ", a2s, "Causal state to state matrix: ", s2s, "Total time: ", total_time)              

        dead_onehot = th.zeros_like(avail_actions[0,0,0])
        dead_onehot[0] = 1.
        dead_onehot = dead_onehot.int()
        dead_allies = avail_actions.clone() == dead_onehot
        dead_allies = dead_allies.all(-1).float()
        #CausalHRO

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
        mac_out = self.mixer.func_g(mac_out, batch["state"], t_env) #CausalHRO

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

            # Max over target Q-Values/ Double q learning
            # mac_out_detach = mac_out.clone().detach()
            # TODO: COMMENT: do not need copy
            #mac_out_detach = mac_out #Commented by HRO

            #CausalHRO
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach = self.mixer.func_f(mac_out_detach, batch["state"], t_env)
            mac_out_detach = mac_out_detach / self.entropy_coef
            mac_out_detach[avail_actions == 0] = -9999999
            
            if self.weight_a2s is not None: 
                a2s_mask = mask_irrelevant_actions(batch["actions"])
                mac_out_detach = mac_out_detach.masked_fill(a2s_mask == 0, -9999999)

            # actions_pdf 是基于Q值的softmax计算出的动作概率分布 即智能体在当前状态下采取不同动作的概率分布。
            actions_pdf = th.softmax(mac_out_detach, dim=-1)
            rand_idx = th.rand(actions_pdf[:,:,:,:1].shape).to(actions_pdf.device)
            actions_cdf = th.cumsum(actions_pdf, -1)
            rand_idx = th.clamp(rand_idx, 1e-6, 1-1e-6)
            picked_actions = th.searchsorted(actions_cdf, rand_idx)
            target_qvals = th.gather(target_mac_out.clone(), 3, picked_actions).squeeze(3)
            
            target_logp = th.log(actions_pdf)

            target_logp = th.gather(target_logp, 3, picked_actions).squeeze(3)
            causal_weight = th.from_numpy(self.causal_default_weight).to(target_logp.device).clone().detach()
            target_logp = target_logp * causal_weight.unsqueeze(0)


            target_entropy = - target_logp.sum(-1, keepdim=True)

            
            # logp_inf2zero = th.where(th.log(actions_pdf)==-th.inf, 0, th.log(actions_pdf))
            # target_entropy = -actions_pdf * logp_inf2zero
            # target_entropy = target_entropy.sum(-1).sum(-1, keepdim=True)

            if self.weight_s2s is not None:
                state = mask_irrelevant_states(batch["semantic_state"], self.weight_s2s)
                target_qvals = self.target_mixer(target_qvals, state, target_mac_out)
            else:
                target_qvals = self.target_mixer(target_qvals, batch["state"], target_mac_out)
            #CausalHRO

            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]

            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])
                if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
                    targets = self.pool.apply_async(
                        build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
                )
                    '''
                    targets = self.pool.apply_async(
                        calculate_n_step_td_target,
                        (self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                        self.args.td_lambda, True, self.args.thread_num, False, None) 
                    )
                    '''
                else:
                    targets = build_q_lambda_targets(
                        rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda
                                    )
                    '''
                    targets = calculate_n_step_td_target(
                        self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                        self.args.td_lambda
                    )'''
            else:
                if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
                    targets = self.pool.apply_async(
                        build_td_lambda_targets(rewards, terminated, mask, target_qvals, target_entropy*self.entropy_coef,
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)
                    )
                else:
                    targets = build_td_lambda_targets(
                        rewards, terminated, mask, target_qvals, target_entropy*self.entropy_coef,
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda
                                                    )


        # Set mixing net to training mode
        self.mixer.train()
        # Mixer

        #CausalHRO
        naive_sum = chosen_action_qvals.clone().detach().sum(-1, keepdim=True)
        chosen_aq_clone = chosen_action_qvals.clone().detach()
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], dead_allies[:,:-1])
        #CausalHRO

        if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
            targets = targets.get()

        td_error = (chosen_action_qvals - targets)
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        mask_elems = mask.sum()
        loss_td = masked_td_error.sum() / mask_elems

        #CausalHRO
        # beta loss
        affine_aq = self.mixer.func_f(chosen_aq_clone, batch["state"][:, :-1], t_env, dead_allies[:,:-1])
        approx_error = chosen_action_qvals.detach() - affine_aq.sum(-1, keepdim=True)
        beta_error = 0.5 * approx_error.pow(2)
        masked_beta_error = beta_error * mask
        L_beta = masked_beta_error.sum() / mask.sum()
        
        gopt_mask = (((approx_error>0.).float() + (td_error<0.).float()) != 1).float()
        weight_td_error = masked_td_error * gopt_mask * 0.5 + masked_td_error * (1-gopt_mask)
        mask_sum = mask * gopt_mask * 0.5 + mask * (1-gopt_mask)
        L_wtd = weight_td_error.sum() / mask_sum.sum()

        loss = L_wtd + L_beta
        #CausalHRO
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
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            #CausalHRO
            self.logger.log_stat("loss_wtd", L_wtd.item(), t_env)
            self.logger.log_stat("loss_beta", L_beta.item(), t_env)
            self.logger.log_stat("err_mask", (mask_sum.sum()/mask.sum()).item(), t_env)
            #CausalHRO
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("target_mean", target_mean, t_env)
            #CausalHRO
            self.logger.log_stat("entropy", target_entropy.mean().item(), t_env)
            self.logger.log_stat("entropy_coef", self.entropy_coef, t_env)
            self.logger.log_stat("naive_sum", naive_sum.mean().item(), t_env)
            #CausalHRO
            self.log_stats_t = t_env

            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)
                
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