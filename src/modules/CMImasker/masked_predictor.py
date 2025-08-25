# src/components/cmi_masker/masked_predictor.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHead(nn.Module):
    """Outputs mean, log_std for a single target scalar."""
    def __init__(self, in_dim: int, hidden: int = 64, min_log_std: float = -6.9, max_log_std: float = -4.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)  
        return mu, log_std


class PerInputEmbed(nn.Module):
    """Embeds a single input feature/block to a common feature size."""
    def __init__(self, in_dim: int, feat_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ChildMaskedPredictor(nn.Module):
    """
    One predictor per next-state dimension s'_{j}.
    - Has an embedding for each state scalar s_t[k] and one for the (joint) action block A_t.
    - Applies mask by setting masked feature vectors to -inf then elementwise max across inputs.
    - Feeds the pooled vector to a Gaussian head predicting p(s'_{j} | .).
    """
    def __init__(self, state_dim: int, action_dim: int, feat_dim: int = 64, pool: str = "max", head_hidden: int = 64):
        super().__init__()
        assert pool in {"max", "logsumexp"}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pool = pool

        # one embed per state scalar (treat each as 1D) + one embed for joint action
        self.state_embeds = nn.ModuleList([PerInputEmbed(1, feat_dim) for _ in range(state_dim)])
        self.action_embed = PerInputEmbed(action_dim, feat_dim)

        # combine pooled feature to predict scalar next-state component
        self.head = GaussianHead(in_dim=feat_dim, hidden=head_hidden)

        # constant used for masking
        self.neg_inf = -999999

    def _pool(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, num_inputs, feat_dim]
        if self.pool == "max":
            return feats.max(dim=1).values  # [B, feat_dim]
        else:
            return torch.logsumexp(feats, dim=1)  # smooth alternative

    def forward(
        self,
        s_t: torch.Tensor,          # [B, state_dim]
        a_t: torch.Tensor,          # [B, act_dim] (use logits/probs or one-hot concat per agent)
        mask_keep: torch.Tensor,    # [num_inputs] boolean: which inputs are active (True = keep)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mu, log_std) for the single next-state dimension this module models.
        num_inputs = state_dim + 1 (joint action block).
        """
        B = s_t.size(0)
        device = s_t.device
        inputs = []

        # state features first
        for k in range(self.state_dim):
            x = s_t[:, k:k+1]
            inputs.append(self.state_embeds[k](x))  # [B, feat_dim]

        # joint action block
        inputs.append(self.action_embed(a_t))  # [B, feat_dim]

        feats = torch.stack(inputs, dim=1)  # [B, num_inputs, feat_dim]
        # apply mask: set masked features to -inf so max-pool ignores them
        num_inputs = feats.size(1)
        assert mask_keep.numel() == num_inputs
        mk = mask_keep.to(device).float().view(1, num_inputs, 1)
        masked_feats = feats * mk + (1.0 - mk) * self.neg_inf

        pooled = self._pool(masked_feats)  # [B, feat_dim]
        mu, log_std = self.head(pooled)    # each [B, 1]
        return mu.squeeze(-1), log_std.squeeze(-1)

    @staticmethod
    def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        # per-sample negative log-likelihood (scalar target)
        var = torch.exp(2.0 * log_std)
        return 0.5 * (math.log(2.0 * math.pi) + 2.0 * log_std + (y - mu) ** 2 / var)
