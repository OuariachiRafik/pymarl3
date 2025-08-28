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


def gaussian_log_prob(y, mu, log_std):
    """
    Elementwise log N(y | mu, sigma^2) up to the additive constant -0.5*log(2*pi).
    Shapes: y, mu, log_std can be [N] or [N,1]; will broadcast.
    """
    return -0.5 * ((y - mu) ** 2) * torch.exp(-2 * log_std) - log_std


class ChildMaskedPredictor(nn.Module):
    """
    One predictor per target scalar (e.g., next-latent coordinate z'_{k}).
    - Embeds each state scalar s_t[k] (1D) and the joint action block A_t (multi-D).
    - Applies a boolean mask over inputs; masked embeddings are set to -inf so max-pool ignores them.
    - Feeds pooled feature to a Gaussian head predicting p(y | inputs).
    """
    def __init__(self, state_dim: int, action_dim: int, feat_dim: int = 64, pool: str = "max", head_hidden: int = 64):
        super().__init__()
        assert pool in {"max", "logsumexp"}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pool = pool

        # one embed per state scalar + one embed for the joint action block
        self.state_embeds = nn.ModuleList([PerInputEmbed(1, feat_dim) for _ in range(state_dim)])
        self.action_embed = PerInputEmbed(action_dim, feat_dim)

        self.head = GaussianHead(in_dim=feat_dim, hidden=head_hidden)

        # numeric -inf used for masking
        self.neg_inf = -1e9

    def _pool(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, num_inputs, feat_dim]
        if self.pool == "max":
            return feats.max(dim=1).values  # [B, feat_dim]
        else:
            return torch.logsumexp(feats, dim=1)  # smooth alternative

    def forward(
        self,
        s_t: torch.Tensor,          # [B, state_dim]
        a_t: torch.Tensor,          # [B, action_dim] (one-hot/probs/logits per joint action)
        mask_keep: torch.Tensor,    # [num_inputs] boolean: which inputs are active (True=keep)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mu, log_std) for the single next-target dimension this module models.
        num_inputs = state_dim + 1 (the +1 is the joint action block).
        """
        B = s_t.size(0)
        device = s_t.device
        inputs = []

        # state scalars
        for k in range(self.state_dim):
            x = s_t[:, k:k+1]                         # [B,1]
            inputs.append(self.state_embeds[k](x))    # [B, feat_dim]

        # joint action block
        inputs.append(self.action_embed(a_t))         # [B, feat_dim]

        feats = torch.stack(inputs, dim=1)            # [B, num_inputs, feat_dim]

        # apply mask: masked features -> -inf so pooling ignores them
        num_inputs = feats.size(1)
        assert mask_keep.numel() == num_inputs, f"mask_keep length {mask_keep.numel()} != num_inputs {num_inputs}"
        mk = mask_keep.to(device).float().view(1, num_inputs, 1)
        masked_feats = feats * mk + (1.0 - mk) * self.neg_inf

        pooled = self._pool(masked_feats)             # [B, feat_dim]
        mu, log_std = self.head(pooled)               # each [B,1]
        return mu.squeeze(-1), log_std.squeeze(-1)    # -> [B], [B]

    def _build_mask(self, use_action: bool, device) -> torch.Tensor:
        """
        Build [state_dim + 1] boolean mask: keep all state scalars; keep or drop the action block.
        """
        mask = torch.ones(self.state_dim + 1, dtype=torch.bool, device=device)
        if not use_action:
            mask[-1] = False  # drop the action block for the denominator p(y | s_t)
        return mask

    @torch.no_grad()
    def logp(self, y: torch.Tensor, s_t: torch.Tensor, a_t: torch.Tensor, mask_use_action: bool) -> torch.Tensor:
        """
        Per-sample log-likelihood under the chosen mask.
        - y: [B] or [B,1]
        - s_t: [B, state_dim]
        - a_t: [B, action_dim]
        - mask_use_action: True -> p(y | s_t, a_t) ; False -> p(y | s_t)
        Returns: [B]
        """
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)  # [B]
        mask_keep = self._build_mask(mask_use_action, s_t.device)  # [state_dim+1]
        mu, log_std = self.forward(s_t, a_t, mask_keep=mask_keep)  # [B], [B]
        return gaussian_log_prob(y, mu, log_std)                   # [B]

    @staticmethod
    def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """
        Per-sample negative log-likelihood with the full constant term.
        y, mu, log_std are [B] or [B,1]; returns [B].
        """
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)
        if mu.dim() == 2 and mu.size(-1) == 1:
            mu = mu.squeeze(-1)
        if log_std.dim() == 2 and log_std.size(-1) == 1:
            log_std = log_std.squeeze(-1)
        var = torch.exp(2.0 * log_std)
        return 0.5 * (math.log(2.0 * math.pi)) + log_std + 0.5 * (y - mu) ** 2 / var
