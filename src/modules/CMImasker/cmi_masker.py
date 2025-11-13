# src/components/cmi_masker/cmi_masker.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .masked_predictor import ChildMaskedPredictor


@dataclass
class CMIMaskerConfig:
    state_dim: int
    act_dim: int
    feat_dim: int = 64
    head_hidden: int = 64
    pool: str = "max"
    lr: float = 3e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.999
    eval_interval: int = 10
    val_split: float = 0.1
    threshold: float = 1e-3  # epsilon for deciding action→state dependence
    refresh_stride: int = 1000
    device: str = "cuda"


class CMIMasker(nn.Module):
    """
    Trains per-child masked predictors to estimate:
      log p(s'_j | S, A)  and  log p(s'_j | S)
    and computes CMI_j = E[ log p(s'_j | S, A) - log p(s'_j | S) ] on held-out,
    then thresholds to produce a binary mask M over state dims for the mixer input.
    """
    def __init__(self, cfg: CMIMaskerConfig):
        super().__init__()
        self.cfg = cfg
        self.J = cfg.state_dim
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # one masked predictor per next-state component
        print(cfg)
        self.children_ = nn.ModuleList([
            ChildMaskedPredictor(cfg.state_dim, cfg.act_dim, cfg.feat_dim, cfg.pool, cfg.head_hidden)
            for _ in range(self.J)
        ]).to(self.device)

        # --- NEW (CDL) full parent->child CMI matrix (rows: parents z dims + action, cols: children z' dims) ---
        self.register_buffer("cmi_matrix", torch.zeros(self.J + 1, self.J, device=self.device))


        self.opt = optim.Adam(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # state for CMI estimation
        self._val_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._ema_cmi = torch.zeros(self.J, device=self.device)
        self._steps = 0
        self._mask = torch.ones(self.J, device=self.device)  # start unmasked

    # ---- public API -------------------------------------------------------

    @torch.no_grad()
    def get_state_mask(self) -> torch.Tensor:
        """Return current binary mask M in [0,1]^{state_dim} as a 1D tensor on device."""
        return self._mask.clone()

    def step_train_minibatch(self, s_t: torch.Tensor, a_t: torch.Tensor, s_tp1: torch.Tensor) -> Dict[str, float]:
        """
        One training step on a minibatch of transitions, following the paper:
        maximize log p(s'_j | S, A) + log p(s'_j | S \ s_i) + (optional) parent-only term (omitted here).
        For mask building we only need the two conditionals used in the CMI ratio.
        """
        self.train()
        self._steps += 1
        s_t = s_t.to(self.device)              # [B, J]
        a_t = a_t.to(self.device)              # [B, act_dim]
        s_tp1 = s_tp1.to(self.device)          # [B, J]
        B = s_t.size(0)

        # create validation split (first call caches it)
        if self._val_cache is None:
            n_val = max(1, int(B * self.cfg.val_split))
            idx = torch.randperm(B, device=self.device)
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            self._val_cache = (s_t[val_idx], a_t[val_idx], s_tp1[val_idx])
            s_t, a_t, s_tp1 = s_t[tr_idx], a_t[tr_idx], s_tp1[tr_idx]

        # training loss (sum over children)
        loss = 0.0
        total = 0
        # sample a single leave-one-state-out index per child (uniform), as in paper
        loso_idx = torch.randint(low=0, high=self.J, size=(self.J,), device=self.device)

        for j in range(self.J):
            child = self.children_[j]

            # FULL conditional: p(s'j | S, A)  (no mask)
            mask_full = torch.ones(self.J + 1, device=self.device, dtype=torch.bool)  # +1 for action block
            mu_f, logstd_f = child(s_t, a_t, mask_full)
            nll_f = child.gaussian_nll(s_tp1[:, j], mu_f, logstd_f)  # [B]

            # LOSO conditional: p(s'j | S \ s_i, A)  (mask one state scalar)
            mask_loso = mask_full.clone()
            i = loso_idx[j].item()
            mask_loso[i] = False
            mu_l, logstd_l = child(s_t, a_t, mask_loso)
            nll_l = child.gaussian_nll(s_tp1[:, j], mu_l, logstd_l)

            # total loss accumulates both terms
            loss += (nll_f.mean() + nll_l.mean())
            total += 2

        loss = loss / max(1, total)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.opt.step()

        # periodic CMI evaluation on held-out
        logs = {"cmi_masker/train_loss": float(loss.item())}
        print("steps=",self._steps, "eval_interval=",self.cfg.eval_interval)
        print("refresh stride=", self.cfg.refresh_stride)
        if self._steps > 1000 and (self._steps % self.cfg.eval_interval) == 0:
            cmi = self._compute_cmi_on_val()  # [J]
            # EMA
            self._ema_cmi = self.cfg.ema_decay * self._ema_cmi + (1 - self.cfg.ema_decay) * cmi
            logs["cmi_masker/cmi_mean"] = float(self._ema_cmi.mean().item())
            logs["cmi_masker/cmi_max"] = float(self._ema_cmi.max().item())

            # (optional) refresh mask every refresh_stride
            if (self._steps % self.cfg.refresh_stride) == 0:
                self._refresh_mask()

        return logs

    # ---- internals --------------------------------------------------------

    @torch.no_grad()
    def _compute_cmi_on_val(self) -> torch.Tensor:
        """CMI_j = E_val[ log p(s'j | S, A) - log p(s'j | S) ]."""
        assert self._val_cache is not None, "no val split cached yet"
        self.eval()
        s_v, a_v, sp_v = self._val_cache
        s_v = s_v.to(self.device)
        a_v = a_v.to(self.device)
        sp_v = sp_v.to(self.device)

        B = s_v.size(0)
        cmi = torch.zeros(self.J, device=self.device)

        full_mask = torch.ones(self.J + 1, device=self.device, dtype=torch.bool)
        # action-masked conditional: p(s'j | S)  (mask the action block only)
        mask_no_act = full_mask.clone()
        mask_no_act[-1] = False  # last input is action block

        for j in range(self.J):
            child = self.children_[j]

            mu_f, logstd_f = child(s_v, a_v, full_mask)
            mu_n, logstd_n = child(s_v, a_v, mask_no_act)

            # log-likelihoods
            ll_full = -child.gaussian_nll(sp_v[:, j], mu_f, logstd_f)  # [B]
            ll_noact = -child.gaussian_nll(sp_v[:, j], mu_n, logstd_n)

            cmi[j] = (ll_full - ll_noact).mean()

        return cmi  # [J]

    @torch.no_grad()
    def _refresh_mask(self):
        """Binary mask by thresholding EMA CMI."""
        M = (self._ema_cmi >= self.cfg.threshold).float()
        # Avoid empty mask: keep top-1 if all zero
        if M.sum() == 0:
            topk = torch.topk(self._ema_cmi, k=1).indices
            M[topk] = 1.0
        self._mask = M

    @torch.no_grad()
    def prediction_gap(self, Z, A, Zp, sum_over_k: bool = True):
        """
        Z, Zp: [N, z_dim], A: [N, a_dim]
        Returns:
          g: [N] if sum_over_k else [N, z_dim]
             where g[n] = sum_k ( log p_f - log p_m )  at transition n
        """
        self.eval()
        N, dz = Z.shape
        gaps = []
        for k in range(dz):
            y = Zp[:, k:k+1]  # target coord
            # full vs action-masked
            lp_full = self.children_[k].logp(y, Z, A, mask_use_action=True)   # [N]
            lp_mask = self.children_[k].logp(y, Z, A, mask_use_action=False)  # [N]
            gaps.append(lp_full - lp_mask)  # [N]
        G = torch.stack(gaps, dim=1)  # [N, dz]
        return G.sum(dim=1) if sum_over_k else G


    @torch.no_grad()
    def step_train_cdl(self, Z: torch.Tensor, A: torch.Tensor, Zp: torch.Tensor):
        """CDL-style CMI over latents: fill self.cmi_matrix[parent, child]."""
        self.eval()
        Z = Z.to(self.device)
        A = A.to(self.device)
        Zp = Zp.to(self.device)
        N, dz = Z.shape
        assert dz == self.J, f"latent dim {dz} != expected {self.J}"
        for k in range(self.J):
            child = self.children_[k]
            full_mask = torch.ones(self.J + 1, device=self.device, dtype=torch.bool)
            mu_full, logstd_full = child(Z, A, full_mask)
            logp_full = -0.5 * ((Zp[:, k] - mu_full) ** 2) * torch.exp(-2 * logstd_full) - logstd_full  # [N]
            for i in range(self.J + 1):
                drop_mask = full_mask.clone()
                drop_mask[i] = False
                mu_drop, logstd_drop = child(Z, A, drop_mask)
                logp_drop = -0.5 * ((Zp[:, k] - mu_drop) ** 2) * torch.exp(-2 * logstd_drop) - logstd_drop
                gap = (logp_full - logp_drop).mean()
                old = self.cmi_matrix[i, k]
                self.cmi_matrix[i, k] = self.cfg.ema_decay * old + (1.0 - self.cfg.ema_decay) * gap

    def get_cdl_mask(self, threshold: float = None) -> torch.Tensor:
        """Reduce parent->child scores to a 1D child mask (max over parents)."""
        if threshold is None:
            threshold = self.cfg.threshold
        scores_per_child = self.cmi_matrix.max(dim=0).values
        M = (scores_per_child >= threshold).float()
        if M.sum() == 0:
            topk = torch.topk(scores_per_child, k=1).indices
            M[topk] = 1.0
        return M
