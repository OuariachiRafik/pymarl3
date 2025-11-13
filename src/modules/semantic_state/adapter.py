import torch
import torch.nn as nn


class StateAdapter(nn.Module):
    """
    Maps masked latent z_t [B,T,dz] back to the original state_dim expected by the mixer.
    If dz == state_dim, acts as identity.
    """
    def __init__(self, latent_dim: int, state_dim: int, hidden: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        if latent_dim == state_dim:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, state_dim),
            )

    def forward(self, z_masked: torch.Tensor) -> torch.Tensor:
        return self.net(z_masked)
