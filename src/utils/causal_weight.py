import os
import pickle
import numpy as np
import pandas as pd
import time
from causallearnmain.causallearn.search.FCMBased import lingam
from causallearnmain.causallearn.search.ConstraintBased import PC
from causallearnmain.causallearn.utils.cit import fisherz, kci, chisq, gsq
from causallearnmain.causallearn.search.ScoreBased.GES import ges
import sys
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


def get_sa2r_weight_ges( memory,  sample_size=5000, causal_method='DirectLiNGAM'):
    states, actions, rewards, next_states, dones = memory.sample(sample_size)
    rewards = np.squeeze(rewards[:sample_size])
    rewards = np.reshape(rewards, (sample_size, 1))
    # X_ori = np.hstack((states[:sample_size, :], actions[:sample_size, :], rewards))
    X_ori = np.hstack((actions[:sample_size, :], rewards))
    # X_ori += 1e-6
    # print('X_ori', X_ori)
    np.savetxt("temp_data.txt", X_ori, delimiter=" ")
    # 再用 np.loadtxt 读取，仿照默认的 PC 方法
    data = np.loadtxt("temp_data.txt", skiprows=0)
    X = pd.DataFrame(X_ori, columns=list(range(np.shape(X_ori)[1])))

    if causal_method == 'DirectLiNGAM':
        start_time = time.time()
        res_map = ges(data, score_func='local_score_BDeu', maxP=None, parameters=None)
        print('cg.G.graph', res_map['G'].graph)
        end_time = time.time()
        _running_time = end_time - start_time
        weight_r = res_map['G'].graph[-1, 0:(0 + np.shape(actions)[1])]

    # softmax weight_r
    weight = F.softmax(torch.Tensor(weight_r), 0)
    weight = weight.numpy()
    # * multiply by action size
    weight = weight * weight.shape[0]
    return weight, _running_time


def get_sa2r_weight_pc( memory,  sample_size=5000, causal_method='DirectLiNGAM'):
    states, actions, rewards, next_states, dones = memory.sample(sample_size)
    rewards = np.squeeze(rewards[:sample_size])
    rewards = np.reshape(rewards, (sample_size, 1))
    X_ori = np.hstack( (actions[:sample_size, :], rewards))
    # X_ori += 1e-6
    # print('X_ori', X_ori)
    np.savetxt("temp_data_2.txt", X_ori, delimiter=" ")
    # 再用 np.loadtxt 读取，仿照默认的 PC 方法
    data = np.loadtxt("temp_data_2.txt", skiprows=0)
    X = pd.DataFrame(X_ori, columns=list(range(np.shape(X_ori)[1])))

    if causal_method == 'DirectLiNGAM':
        start_time = time.time()
        cg = PC.pc(data, 0.05, chisq)
        print('cg.G.graph', cg.G.graph)
        end_time = time.time()
        _running_time = end_time - start_time
        weight_r = cg.G.graph[-1, 0:(0 + np.shape(actions)[1])]

    # softmax weight_r
    weight = F.softmax(torch.Tensor(weight_r), 0)
    weight = weight.numpy()
    # * multiply by action size
    weight = weight * weight.shape[0]
    return weight, _running_time



def get_sa2r_weight(batch, sample_size=1000, causal_method='DirectLiNGAM'):
    states = batch["state"][:, :-1].cpu().numpy()  # (128, 85, 120)
    actions = batch["actions"][:, :-1].squeeze(-1).cpu().numpy()  # (128, 85, 5)
    rewards = batch["reward"][:, :-1].squeeze(-1).cpu().numpy()  # (128, 85)

    batch_size, seq_len, state_dim = states.shape
    _, _, action_dim = actions.shape

    # 展平成 (batch_size * seq_len, feature_dim)
    states = states.reshape(-1, state_dim)  # (128*85, 120)
    actions = actions.reshape(-1, action_dim)  # (128*85, 5)
    rewards = rewards.reshape(-1, 1)  # (128*85, 1)

    total_samples = states.shape[0]
    sample_size = min(sample_size, total_samples)  # 避免超过总样本数

    # 随机采样 `sample_size` 个索引
    sample_indices = np.random.choice(total_samples, sample_size, replace=False)

    # 选取采样数据
    sampled_states = states[sample_indices]
    sampled_actions = actions[sample_indices]
    sampled_rewards = rewards[sample_indices]

    # 组合数据并转换为 DataFrame
    X_ori = np.hstack((sampled_states, sampled_actions, sampled_rewards))

    # **检查 NaN 和 Inf**
    if np.isnan(X_ori).any() or np.isinf(X_ori).any():
        raise ValueError("X contains NaN or Inf values, check preprocessing!")

    # **归一化**
    # scaler = StandardScaler()
    # X_ori = scaler.fit_transform(X_ori)

    X = pd.DataFrame(X_ori, columns=list(range(X_ori.shape[1])))

    if causal_method == 'DirectLiNGAM':
        start_time = time.time()
        model = lingam.DirectLiNGAM()
        model.fit(X)
        end_time = time.time()
        model._running_time = end_time - start_time
        # state -> reward 权重
        weight_ss2r =  model.adjacency_matrix_[-1, :state_dim]
        # 提取 action -> reward 的因果权重
        weight_r = model.adjacency_matrix_[-1, state_dim:(state_dim + action_dim)]

    # softmax 归一化
    weight = F.softmax(torch.Tensor(weight_r), dim=0).numpy()
    weight = weight * weight.shape[0]  # * action_dim
    weight_ss2r = F.softmax(torch.Tensor(weight_ss2r), dim=0).numpy() * state_dim

    return weight, weight_ss2r, model._running_time

def get_sa2r_weight_peragent(batch,agent_id, sample_size=1000, causal_method='DirectLiNGAM'):
    # Step 1: Extract batch data up to timestep T-1
    observations = batch["obs"][:, :-1].cpu().numpy()  # (batch_size, episode_len, n_agents, obs_dim)
    actions = batch["actions"][:, :-1].cpu().numpy()   # (batch_size, episode_len, n_agents, 1)

    # Step 2: Slice to keep only the desired agent (preserving singleton dimension)
    agent_observations = observations[:, :, agent_id:agent_id+1]  # (batch_size, episode_len, 1, obs_dim)
    agent_actions = actions[:, :, agent_id:agent_id+1]            # (batch_size, episode_len, 1, 1)

    # Step 3: Rewards for the agent (preserving singleton dimension)
    rewards = batch["reward"][:, :-1, agent_id:agent_id+1].cpu().numpy()  # (batch_size, episode_len, 1, 1)


    batch_size, seq_len, n_agents, state_dim = agent_observations.shape
    _, _, n_agents, agent_action_dim = agent_actions.shape

    print("batch_size, seq_len, n_agents, state_dim = ",agent_observations.shape)
    print("_, _, n_agents, agent_action_dim = ",agent_actions.shape)
    
    # 展平成 (batch_size * seq_len, feature_dim)
    agent_observations = agent_observations.reshape(-1, state_dim)  # (128*85, 120)
    agent_actions = agent_actions.reshape(-1, agent_action_dim)  # (128*85, 1)
    rewards = rewards.reshape(-1, 1)  # (128*85, 1)

    total_samples = agent_observations.shape[0]
    sample_size = min(sample_size, total_samples)  # 避免超过总样本数

    # 随机采样 `sample_size` 个索引
    sample_indices = np.random.choice(total_samples, sample_size, replace=False)

    # 选取采样数据
    sampled_observations = agent_observations[sample_indices]
    sampled_agent_actions = agent_actions[sample_indices]
    sampled_rewards = rewards[sample_indices]

    # 组合数据并转换为 DataFrame
    X_ori = np.hstack((sampled_observations, sampled_agent_actions, sampled_rewards))
    
    # **检查 NaN 和 Inf**
    if np.isnan(X_ori).any() or np.isinf(X_ori).any():
        raise ValueError("X contains NaN or Inf values, check preprocessing!")

    # **归一化**
    # scaler = StandardScaler()
    # X_ori = scaler.fit_transform(X_ori)

    X = pd.DataFrame(X_ori, columns=list(range(X_ori.shape[1])))
    print("X=", X)
    if causal_method == 'DirectLiNGAM':
        start_time = time.time()
        model = lingam.DirectLiNGAM()
        model.fit(X)
        end_time = time.time()
        model._running_time = end_time - start_time
        # state -> reward 权重
        weight_ss2r =  model.adjacency_matrix_[-1, :state_dim]
        # 提取 action -> reward 的因果权重
        weight_r = model.adjacency_matrix_[-1, state_dim:(state_dim + agent_action_dim)]

    # softmax 归一化
    weight = F.softmax(torch.Tensor(weight_r), dim=0).numpy()
    weight = weight * weight.shape[0]  # * action_dim
    weight_ss2r = F.softmax(torch.Tensor(weight_ss2r), dim=0).numpy() * state_dim

    return weight, weight_ss2r, model._running_time

import numpy as np


def mask_irrelevant_states(states, weight_ss2r, top_k=80):
    """
    使用 state-reward 因果矩阵对状态维度进行筛选，mask 掉最不相关的维度

    :param states: 原始状态矩阵, shape (batch_size, seq_len, state_dim)
    :param weight_ss2r: state-reward 因果权重, shape (state_dim,)
    :param threshold: 低于该值的状态维度将被 mask 掉
    :param top_k: 如果设定，则只保留 top_k 个最相关的维度
    :return: masked_states, mask
    """
    state_dim = states.shape[-1]  # 获取状态维度

    # 确保权重是 numpy 数组
    weight_ss2r = np.array(weight_ss2r)

    # # **方法 1：使用阈值筛选**
    # if top_k is None:
    #     mask = (weight_ss2r >= threshold).astype(float)  # 相关维度为 1，不相关为 0

    # **方法 2：选取前 top_k 个最相关维度**

    top_indices = np.argsort(weight_ss2r)[-top_k:]  # 选取 top_k 个最大的索引
    mask = np.zeros(state_dim)  # 初始化为全 0
    mask[top_indices] = 1  # 仅保留前 top_k 维度

    # **应用 mask**
    masked_states = states * mask  # 按位乘法，屏蔽无关维度

    return masked_states


def get_sa2r_weight_ace( memory,  sample_size=5000, causal_method='DirectLiNGAM'):
    states, actions, rewards, next_states, dones = memory.sample(sample_size)
    rewards = np.squeeze(rewards[:sample_size])
    rewards = np.reshape(rewards, (sample_size, 1))
    X_ori = np.hstack((states[:sample_size, :], actions[:sample_size, :], rewards))
    X = pd.DataFrame(X_ori, columns=list(range(np.shape(X_ori)[1])))

    if causal_method == 'DirectLiNGAM':
        start_time = time.time()
        model = lingam.DirectLiNGAM()
        model.fit(X)
        end_time = time.time()
        model._running_time = end_time - start_time
        weight_r = model.adjacency_matrix_[-1, np.shape(states)[1]:(np.shape(states)[1] + np.shape(actions)[1])]

    # softmax weight_r
    weight = F.softmax(torch.Tensor(weight_r), 0)
    weight = weight.numpy()
    # * multiply by action size
    weight = weight * weight.shape[0]
    return weight, model._running_time
