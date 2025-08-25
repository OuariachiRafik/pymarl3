from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import BlockSlices


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):  # [*, in_dim]
        return self.net(x)


class DeepSet(nn.Module):
    """Permutation-invariant set encoder: rho(sum(phi(x_i)))."""
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, X):  # X: [B,T,N,in_dim]
        B, T, N, D = X.shape
        h = self.phi(X.reshape(B*T*N, D)).reshape(B, T, N, -1)  # [B,T,N,H]
        pooled = h.sum(dim=2)  # [B,T,H]
        out = self.rho(pooled) # [B,T,out_dim]
        return out


@dataclass
class StateBlockEncoderConfig:
    # dims for each latent block
    ally_latent_dim: int = 16
    enemy_latent_dim: int = 16
    dyn_latent_dim: int = 8     # cooldown+energy
    hist_latent_dim: int = 8    # last actions
    comp_latent_dim: int = 6    # type histograms
    geom_latent_dim: int = 8    # centroids/variances
    # MLP/DeepSet widths
    set_hidden: int = 64
    mlp_hidden: int = 64
    dropout: float = 0.0
    # race channels, etc.
    race_type_dim: int = 3
    # training knobs (optional)
    transition_pretrain: bool = False
    transition_lr: float = 3e-4


class StateBlockEncoder(nn.Module):
    """
    Splits flat centralized state into semantic blocks (allies, enemies, cooldown/energy, last actions),
    encodes each block to a small latent, and concatenates them into z_t.
    Optionally learns a simple transition model z' ~ f([z, A]) for pretraining.
    """
    def __init__(self, slices: BlockSlices, cfg: StateBlockEncoderConfig, n_actions: int):
        super().__init__()
        self.slices = slices
        self.cfg = cfg
        self.race_type_dim = cfg.race_type_dim
        self.d_unit = slices.d_unit
        self.U_A = slices.U_A
        self.U_E = slices.U_E
        self.n_actions = n_actions

        # Encoders
        self.enc_ally = DeepSet(in_dim=self.d_unit, hidden=cfg.set_hidden, out_dim=cfg.ally_latent_dim, dropout=cfg.dropout)
        self.enc_enemy = DeepSet(in_dim=self.d_unit, hidden=cfg.set_hidden, out_dim=cfg.enemy_latent_dim, dropout=cfg.dropout)
        dyn_in = 2*self.U_A if (slices.ally_cooldown and slices.ally_energy) else 0
        hist_in = self.U_A * self.n_actions if slices.ally_last_actions else 0

        self.enc_dyn = MLP(in_dim=max(1,dyn_in), out_dim=cfg.dyn_latent_dim, hidden=cfg.mlp_hidden, dropout=cfg.dropout) if dyn_in>0 else None
        self.enc_hist = MLP(in_dim=max(1,hist_in), out_dim=cfg.hist_latent_dim, hidden=cfg.mlp_hidden, dropout=cfg.dropout) if hist_in>0 else None
        self.enc_comp = MLP(in_dim=2*self.race_type_dim, out_dim=cfg.comp_latent_dim, hidden=cfg.mlp_hidden, dropout=cfg.dropout)
        self.enc_geom = MLP(in_dim=6, out_dim=cfg.geom_latent_dim, hidden=cfg.mlp_hidden, dropout=cfg.dropout)

        # optional transition predictor for pretraining: z' = g([z, A_flat])
        self.transition = None
        if cfg.transition_pretrain:
            dz = self.latent_dim
            self.transition = MLP(in_dim=dz + self.U_A*self.n_actions, out_dim=dz, hidden=cfg.mlp_hidden, dropout=cfg.dropout)
            self.opt = torch.optim.Adam(self.parameters(), lr=cfg.transition_lr)

    @property
    def latent_dim(self) -> int:
        dz = self.cfg.ally_latent_dim + self.cfg.enemy_latent_dim + self.cfg.comp_latent_dim + self.cfg.geom_latent_dim
        if self.enc_dyn is not None:
            dz += self.cfg.dyn_latent_dim
        if self.enc_hist is not None:
            dz += self.cfg.hist_latent_dim
        return dz

    def _split_blocks(self, S: torch.Tensor) -> Dict[str, torch.Tensor]:
        # S: [B,T,state_dim]
        a = self.slices
        out = {}
        out["ally_units"] = S[..., a.ally_units]
        out["enemy_units"] = S[..., a.enemy_units]
        if a.ally_cooldown is not None:
            out["ally_cooldown"] = S[..., a.ally_cooldown]
        if a.ally_energy is not None:
            out["ally_energy"] = S[..., a.ally_energy]
        if a.ally_last_actions is not None:
            out["ally_last_actions"] = S[..., a.ally_last_actions]
        return out

    def _unit_reshape(self, X: torch.Tensor, U: int) -> torch.Tensor:
        # X: [B,T, U*d_unit] -> [B,T,U,d_unit]
        B,T,D = X.shape
        return X.view(B, T, U, self.d_unit)

    def _type_hist(self, units: torch.Tensor) -> torch.Tensor:
        # units: [B,T,U,d_unit], last race_type_dim positions are type one-hot
        types = units[..., -self.race_type_dim:]          # [B,T,U,R]
        return types.sum(dim=2)                           # [B,T,R]

    def _centroid_var(self, units: torch.Tensor) -> torch.Tensor:
        # coords assumed at positions 0:2 (x,y)
        pos = units[..., 0:2]                             # [B,T,U,2]
        mean = pos.mean(dim=2)                            # [B,T,2]
        var = pos.var(dim=2, unbiased=False).sum(dim=-1, keepdim=True)  # [B,T,1] sum of variances
        return mean, var                                  # [B,T,2], [B,T,1]

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        S: [B,T,state_dim] -> z: [B,T,latent_dim]
        """
        blocks = self._split_blocks(S)
        ally = self._unit_reshape(blocks["ally_units"], self.U_A)
        enemy = self._unit_reshape(blocks["enemy_units"], self.U_E)

        z_ally = self.enc_ally(ally)     # [B,T,dA]
        z_enemy = self.enc_enemy(enemy)  # [B,T,dB]

        # composition histograms (ally/enemy)
        hist_a = self._type_hist(ally)   # [B,T,R]
        hist_e = self._type_hist(enemy)  # [B,T,R]
        comp = torch.cat([hist_a, hist_e], dim=-1)        # [B,T,2R]
        z_comp = self.enc_comp(comp)

        # geometry
        c_a, v_a = self._centroid_var(ally)               # [B,T,2],[B,T,1]
        c_e, v_e = self._centroid_var(enemy)
        geom = torch.cat([c_a, v_a, c_e, v_e], dim=-1)    # [B,T,6]
        z_geom = self.enc_geom(geom)

        zs = [z_ally, z_enemy, z_comp, z_geom]

        if self.enc_dyn is not None:
            dyn = torch.cat([blocks["ally_cooldown"], blocks["ally_energy"]], dim=-1)  # [B,T,2U_A]
            zs.append(self.enc_dyn(dyn))
        if self.enc_hist is not None:
            hist = blocks["ally_last_actions"]            # [B,T,U_A*n_actions]
            zs.append(self.enc_hist(hist))

        z = torch.cat(zs, dim=-1)                         # [B,T,latent_dim]
        return z

    @torch.no_grad()
    def encode(self, S: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(S)

    def step_pretrain(self, S_t: torch.Tensor, A_flat: torch.Tensor, S_tp1: torch.Tensor) -> float:
        """
        Optional: pretrain encoders + transition with z' ~ f([z, A]).
        S_t,S_tp1: [B*T,state_dim], A_flat: [B*T, U_A*n_actions]
        """
        if self.transition is None:
            return 0.0
        self.train()
        B_T = S_t.size(0)
        # reshape to [B,T,dim] with T=1 for simplicity
        z_t = self.forward(S_t.view(B_T,1,-1))[:,0,:]        # [B*T, dz]
        with torch.no_grad():
            z_tp1 = self.forward(S_tp1.view(B_T,1,-1))[:,0,:]  # target detached
        pred = self.transition(torch.cat([z_t, A_flat], dim=-1))
        loss = F.mse_loss(pred, z_tp1)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())
