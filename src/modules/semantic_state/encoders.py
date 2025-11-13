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
    # ---- latent dims per semantic block (set small if you want micro-latents) ----
    ally_latent_dim: int = 16
    enemy_latent_dim: int = 16
    hist_latent_dim: int = 8    # last actions tail (if present)
    comp_latent_dim: int = 6    # type histograms (if unit types are present)
    geom_latent_dim: int = 8    # centroids/variances (if positions are known)

    # ---- model widths ----
    set_hidden: int = 64
    mlp_hidden: int = 64
    dropout: float = 0.0

    # ---- unit-type and coordinate assumptions (configurable & validated at runtime) ----
    race_type_dim: int = 3      # number of unit-type one-hot channels per unit (0 if not observed)
    ally_pos_offset: int = 2    # start index of (x,y) in ally per-unit vector (SMAC canonical: hp, cd/energy, x, y, ...)
    enemy_pos_offset: int = 1   # start index of (x,y) in enemy per-unit vector (SMAC canonical: hp, x, y, ...)

    # ---- optional pretrain of a latent transition head ----
    transition_pretrain: bool = False
    transition_lr: float = 3e-4


class StateBlockEncoder(nn.Module):
    """
    Splits flat centralized state into semantic blocks (allies, enemies, optional last_actions),
    encodes each block to a small latent, and concatenates them into z_t.
    Optionally learns a simple transition model z' ~ f([z, A]) for pretraining.
    """
    def __init__(self, slices: BlockSlices, cfg: StateBlockEncoderConfig):
        super().__init__()
        self.slices = slices
        self.cfg = cfg

        # Shapes from deterministic layout
        self.d_unit_ally  = slices.d_unit_ally
        self.d_unit_enemy = slices.d_unit_enemy
        self.U_A = slices.U_A
        self.U_E = slices.U_E
        self.n_actions = slices.n_actions

        # -----------------------
        # Encoders (always-on)
        # -----------------------
        self.enc_ally = DeepSet(
            in_dim=self.d_unit_ally,
            hidden=cfg.set_hidden,
            out_dim=cfg.ally_latent_dim,
            dropout=cfg.dropout
        )
        self.enc_enemy = DeepSet(
            in_dim=self.d_unit_enemy,
            hidden=cfg.set_hidden,
            out_dim=cfg.enemy_latent_dim,
            dropout=cfg.dropout
        )

        # -----------------------
        # Optional encoders (enabled only if inputs are valid)
        # -----------------------

        # Last-actions tail
        self.has_hist = (slices.last_actions is not None)
        self.enc_hist: Optional[nn.Module] = None
        if self.has_hist:
            hist_in = self.U_A * self.n_actions
            self.enc_hist = MLP(in_dim=hist_in, out_dim=cfg.hist_latent_dim,
                                hidden=cfg.mlp_hidden, dropout=cfg.dropout)

        # Composition (unit-type histograms from the last race_type_dim channels)
        self.has_types = (cfg.race_type_dim > 0
                          and self.d_unit_ally  >= cfg.race_type_dim
                          and self.d_unit_enemy >= cfg.race_type_dim)
        self.enc_comp: Optional[nn.Module] = None
        if self.has_types:
            self.enc_comp = MLP(in_dim=2*cfg.race_type_dim, out_dim=cfg.comp_latent_dim,
                                hidden=cfg.mlp_hidden, dropout=cfg.dropout)

        # Geometry (centroids/dispersion from (x,y) columns)
        self.has_pos_ally  = (self.d_unit_ally  >= (cfg.ally_pos_offset  + 2))
        self.has_pos_enemy = (self.d_unit_enemy >= (cfg.enemy_pos_offset + 2))
        self.has_geom = (self.has_pos_ally and self.has_pos_enemy)
        self.enc_geom: Optional[nn.Module] = None
        if self.has_geom:
            self.enc_geom = MLP(in_dim=6, out_dim=cfg.geom_latent_dim,
                                hidden=cfg.mlp_hidden, dropout=cfg.dropout)

        # -----------------------
        # Optional transition predictor for pretraining: z' = g([z, A_flat])
        # -----------------------
        self.transition: Optional[nn.Module] = None
        if cfg.transition_pretrain:
            dz = self.latent_dim  # computed from what is actually enabled
            self.transition = MLP(in_dim=dz + self.U_A*self.n_actions, out_dim=dz,
                                  hidden=cfg.mlp_hidden, dropout=cfg.dropout)
            self.opt = torch.optim.Adam(self.parameters(), lr=cfg.transition_lr)

    @property
    def latent_dim(self) -> int:
        dz = self.cfg.ally_latent_dim + self.cfg.enemy_latent_dim
        if self.enc_comp is not None:
            dz += self.cfg.comp_latent_dim
        if self.enc_geom is not None:
            dz += self.cfg.geom_latent_dim
        if self.enc_hist is not None:
            dz += self.cfg.hist_latent_dim
        return dz

    # ---- slicing helpers ----
    def _split_blocks(self, S: torch.Tensor) -> Dict[str, torch.Tensor]:
        # S: [B,T,state_dim]
        a = self.slices
        out = {
            "ally_units":  S[..., a.ally_units],
            "enemy_units": S[..., a.enemy_units],
        }
        if a.last_actions is not None:
            out["last_actions"] = S[..., a.last_actions]
        return out

    def _unit_reshape(self, X: torch.Tensor, U: int, d_unit: int) -> torch.Tensor:
        # X: [B,T, U*d_unit] -> [B,T,U,d_unit]
        B, T, D = X.shape
        assert D == U * d_unit, f"reshape mismatch: {D} != {U}*{d_unit}"
        return X.view(B, T, U, d_unit)

    # ---- optional feature builders ----
    def _type_hist(self, units: torch.Tensor, R: int) -> torch.Tensor:
        # units: [B,T,U,d_unit]; take last R dims as type one-hot
        types = units[..., -R:]              # [B,T,U,R]
        return types.sum(dim=2)              # [B,T,R]

    def _centroid_var(self, units: torch.Tensor, pos_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # units: [B,T,U,d_unit]; take (x,y) at pos_offset:pos_offset+2
        pos = units[..., pos_offset:pos_offset+2]               # [B,T,U,2]
        mean = pos.mean(dim=2)                                  # [B,T,2]
        var  = pos.var(dim=2, unbiased=False).sum(dim=-1, keepdim=True)  # [B,T,1] sum of variances
        return mean, var

    # ---- forward ----
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        S: [B,T,state_dim] -> z: [B,T,latent_dim]
        """
        blocks = self._split_blocks(S)

        ally  = self._unit_reshape(blocks["ally_units"],  self.U_A, self.d_unit_ally)    # [B,T,U_A,dA]
        enemy = self._unit_reshape(blocks["enemy_units"], self.U_E, self.d_unit_enemy)   # [B,T,U_E,dE]

        z_ally  = self.enc_ally(ally)     # [B,T, d_ally]
        z_enemy = self.enc_enemy(enemy)   # [B,T, d_enemy]

        zs = [z_ally, z_enemy]

        # composition (if available)
        if self.enc_comp is not None:
            R = self.cfg.race_type_dim
            hist_a = self._type_hist(ally,  R)   # [B,T,R]
            hist_e = self._type_hist(enemy, R)   # [B,T,R]
            comp   = torch.cat([hist_a, hist_e], dim=-1)        # [B,T,2R]
            zs.append(self.enc_comp(comp))

        # geometry (if available)
        if self.enc_geom is not None:
            c_a, v_a = self._centroid_var(ally,  self.cfg.ally_pos_offset)   # [B,T,2],[B,T,1]
            c_e, v_e = self._centroid_var(enemy, self.cfg.enemy_pos_offset)  # [B,T,2],[B,T,1]
            geom = torch.cat([c_a, v_a, c_e, v_e], dim=-1)                   # [B,T,6]
            zs.append(self.enc_geom(geom))

        # last actions tail (if present)
        if self.enc_hist is not None:
            hist = blocks["last_actions"]            # [B,T,U_A*n_actions]
            zs.append(self.enc_hist(hist))

        z = torch.cat(zs, dim=-1)                    # [B,T,latent_dim]
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
        z_t = self.forward(S_t.view(B_T, 1, -1))[:, 0, :]        # [B*T, dz]
        with torch.no_grad():
            z_tp1 = self.forward(S_tp1.view(B_T, 1, -1))[:, 0, :]  # target detached
        pred = self.transition(torch.cat([z_t, A_flat], dim=-1))
        loss = F.mse_loss(pred, z_tp1)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())
