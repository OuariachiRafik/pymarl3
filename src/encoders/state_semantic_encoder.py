import torch as th, torch.nn as nn, torch.nn.functional as F

class SetMLP(nn.Module):
    def __init__(self, in_dim, hid=128, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, out), nn.ReLU()
        )
    def forward(self, x, mask=None):           # x: [B,T,N,D]
        h = self.net(x)
        if mask is not None: h = h * mask      # mask: [B,T,N,1] with {0,1}
        return h.sum(2)                         # PI pooling → [B,T,out]

class StateSemanticEncoder(nn.Module):
    """
    Build z_t from SMAC-v2 state by encoding ally/enemy sets + (optional) last action.
    ally_feats:  [B,T,N_allies, D_ally]
    enemy_feats: [B,T,N_enemies,D_enemy]
    ally_last_act_oh (optional): [B,T,N_allies,A]  (last action one-hot per ally)
    """
    def __init__(self, ally_feats_dim, enemy_feats_dim,
                 action_dim=0, out_dim=128):
        super().__init__()
        self.ally_dim, self.enemy_dim = ally_feats_dim, enemy_feats_dim
        print("ally_dim =", self.ally_dim)
        print("enemy_dim=", self.enemy_dim)
        self.ally_dim = self.ally_dim + (action_dim if action_dim>0 else 0)
        self.ally_enc  = SetMLP(self.ally_dim, out=out_dim//2)
        self.enemy_enc = SetMLP(self.enemy_dim, out=out_dim//2)
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU())
        self.action_dim = action_dim

    def forward(self, ally_feats, enemy_feats,
                ally_last_act_oh=None):
        print("ally_feats_dimensions=", ally_feats.size)
        print("enemy_feats_dimensions=", enemy_feats.size)
        ally_feats = th.tensor(ally_feats, dtype=th.float32)
        enemy_feats = th.tensor(enemy_feats, dtype=th.float32)
        if self.action_dim and ally_last_act_oh is not None:
            ally_feats = th.cat([ally_feats, ally_last_act_oh], dim=-1)
        a = self.ally_enc(ally_feats)      # [B,T,out/2]
        e = self.enemy_enc(enemy_feats)   # [B,T,out/2]
        z = th.cat([a, e], dim=-1)                    # [B,T,out]
        return self.proj(z)
