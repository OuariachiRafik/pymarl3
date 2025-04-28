import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class Mixer(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        assert self.qmix_pos_func == "abs"
        
        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))
        #CausalHRO
        # order preserving transformation
        self.hyper_w3 = nn.Sequential(
            nn.Linear(self.input_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim//2)
        )
        self.hyper_b3 = nn.Sequential(nn.Linear(self.input_dim, self.n_agents*self.embed_dim//2))
        self.hyper_w4 = nn.Sequential(
            nn.Linear(self.input_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents*self.embed_dim//2)
        )
        self.hyper_b4 = nn.Sequential(
            nn.Linear(self.input_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents)
        )
        
        
        # linear transformation
        self.hyper_w5 = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents)
        )
        self.hyper_b5 = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents)
        )
        #CausalHRO
        if getattr(args, "use_orthogonal", False):
            raise NotImplementedError
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()
        
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        
        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1, 1)
        
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
        # print(w1.mean(), w1.var())
        # print(w2.mean(), w2.var())

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1) # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        return y.view(b, t, -1)

    #CausalHRO
    def func_f(self, qvals, states, t_env, death_mask=None):
        
        qval_shape = qvals.shape
        states = states.reshape(-1, self.state_dim)
        
        if qval_shape[-2] == self.n_agents:
            self.dim_idx = -3
            qvals = qvals.reshape(-1, self.n_agents, qvals.shape[-1])
            w = self.hyper_w5(states).view(-1, self.n_agents, 1)
            b = self.hyper_b5(states).view(-1, self.n_agents, 1)
        if qval_shape[-1] == self.n_agents:
            self.dim_idx = -2
            qvals = qvals.reshape(-1, self.n_agents)
            w = self.hyper_w5(states).view(-1, self.n_agents)
            b = self.hyper_b5(states).view(-1, self.n_agents)
            
        if self.abs:
            w = w.abs()
        
        y = qvals * w + b 
        
        return y.reshape(qval_shape)
    
    def func_g(self, qvals, states, t_env, death_mask=None):
        
        qval_shape = qvals.shape
        states = states.reshape(-1, self.state_dim)
        
        qvals = qvals.reshape(-1, 1, self.n_agents, qval_shape[-1])
        w1 = self.hyper_w3(states).view(-1, self.embed_dim//2, self.n_agents, 1)
        b1 = self.hyper_b3(states).view(-1, self.embed_dim//2, self.n_agents, 1)
        w2 = self.hyper_w4(states).view(-1, self.embed_dim//2, self.n_agents, 1)
        b2 = self.hyper_b4(states).view(-1, 1, self.n_agents, 1)
        
        if self.abs:
            w1 = w1.abs()
            w2 = w2.abs()
        
        y = F.elu(qvals * w1 + b1)
        y = (y * w2).sum(dim=-3, keepdim=True) + b2
        y = y + qvals
        
        return y.reshape(qval_shape)
    
    #CausalHRO
    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)
        
