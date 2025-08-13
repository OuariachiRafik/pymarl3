import numpy as np
from causallearn.search.FCMBased import lingam  # causal-learn's LiNGAM family

def build_rows_from_batch(sample_batch, use_semantic=True):
    """
    sample_batch: EpisodeBatch (already truncated with max_t_filled()).
    Returns X (rows of [a_t | z_t | z_{t+1}]), (Dz, A_dim).
    """
    key = "state_semantic" if use_semantic else "state"
    z = sample_batch[key].cpu().numpy()                 # [B, T+1, Dz]
    a = sample_batch["actions_onehot"].cpu().numpy()    # [B, T+1, N, A]
    mask = sample_batch["filled"].cpu().numpy().astype(bool)[:, :-1, 0]  # [B, T]

    B, Tp1, Dz = z.shape
    T = Tp1 - 1
    A_dim = a.shape[-2] * a.shape[-1]

    rows = []
    for b in range(B):
        for t in range(T):
            if not mask[b, t]:  # only valid transitions
                continue
            a_t   = a[b, t].reshape(-1)  # [A_dim]
            z_t   = z[b, t]              # [Dz]
            z_tp1 = z[b, t+1]            # [Dz]
            rows.append(np.concatenate([a_t, z_t, z_tp1], axis=0))

    if rows:
        X = np.stack(rows, 0)  # [Nrows, A+Dz+Dz]
    else:
        X = np.zeros((0, A_dim + Dz + Dz), dtype=np.float32)
    return X, Dz, A_dim

def discover_graph_with_lingam_cl(sample_batch, use_semantic=True, measure="pwling",
                                  soft=False, thr=1e-3):
    """
    Run DirectLiNGAM on rows [a_t | z_t | z_{t+1}] and return:
      S2S_bin: [Dz, Dz] (z_t -> z_{t+1}),  A2S_bin: [Dz, A] (a_t -> z_{t+1})
    """
    X, Dz, A_dim = build_rows_from_batch(sample_batch, use_semantic=use_semantic)
    if X.shape[0] == 0:
        return (np.zeros((Dz, Dz), np.float32),
                np.zeros((Dz, A_dim), np.float32))

    n_vars = A_dim + Dz + Dz
    idx_a   = np.arange(0, A_dim)
    idx_zt  = np.arange(A_dim, A_dim + Dz)
    idx_zt1 = np.arange(A_dim + Dz, A_dim + Dz + Dz)

    # Prior knowledge matrix M ∈ {-1,0,1}^{n×n}
    # 0: no directed path i→j, 1: there is a directed path, -1: unknown
    M = -np.ones((n_vars, n_vars), dtype=int)
    # forbid paths *from* future to past (z_{t+1} → {a_t,z_t})
    M[np.ix_(idx_zt1, np.concatenate([idx_a, idx_zt]))] = 0
    # actions are exogenous: forbid any path into actions
    M[:, idx_a] = 0

    model = lingam.DirectLiNGAM(prior_knowledge=M,
                                apply_prior_knowledge_softly=soft,
                                measure=measure)
    model.fit(X)

    # adjacency_matrix_[i,j] is j -> i
    Bmat = model.adjacency_matrix_.astype(np.float32)
    S2S = Bmat[np.ix_(idx_zt1, idx_zt)]    # [Dz, Dz]
    A2S = Bmat[np.ix_(idx_zt1, idx_a)]     # [Dz, A]

    S2S_bin = (np.abs(S2S) > thr).astype(np.float32)
    A2S_bin = (np.abs(A2S) > thr).astype(np.float32)
    return S2S_bin, A2S_bin
