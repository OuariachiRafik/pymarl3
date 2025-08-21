import torch as th

class SMACv2StateSlicer:
    """
    Splits flat SMAC(v2) global state -> ally/enemy entity blocks.
    """
    def __init__(self, n_allies, n_enemies, ally_dim, enemy_dim, misc_dim=0, device="cpu"):
        self.n_allies, self.ally_dim = ally_dim
        self.n_enemies, self.enemy_dim =  enemy_dim
        self.misc_dim = misc_dim
        self.device = device

        self._i0 = 0
        self._i1 = self._i0 + self.n_allies * self.ally_dim            # allies span
        self._i2 = self._i1 + self.n_enemies * self.enemy_dim          # enemies span
        self._i3 = self._i2
        self._iend = self._i2 + self.misc_dim                     # optional misc tail

    @th.no_grad()
    def __call__(self, state_bt):
        """
        state_bt: [B, 1, S] float tensor (flat state at time t)
        returns: ally_feats [B,1,Na,Da], enemy_feats [B,1,Ne,De], ally_mask [B,1,Na,1], enemy_mask [B,1,Ne,1]
        """
        state_bt = th.tensor(state_bt, dtype=th.float32)
        B = state_bt.size(0)
        s = state_bt.view(B, -1)[:, self._i0:self._iend]     # [B, S_used]

        a_flat = s[:, self._i0:self._i1].view(B, self.n_allies, self.ally_dim)
        e_flat = s[:, self._i1:self._i2].view(B, self.n_enemies, self.enemy_dim)

        ally_feats  = a_flat.unsqueeze(1)                    # [B,1,Na,Da]
        enemy_feats = e_flat.unsqueeze(1)                    # [B,1,Ne,De]

        # simple “alive” masks: non-zero row -> 1.0
        ally_mask  = (ally_feats.abs().sum(-1, keepdim=True) > 1e-8).float()
        enemy_mask = (enemy_feats.abs().sum(-1, keepdim=True) > 1e-8).float()
        return ally_feats, enemy_feats, ally_mask, enemy_mask
