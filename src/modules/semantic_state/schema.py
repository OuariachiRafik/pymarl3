from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any
import numpy as np

# -----------------------------
# Public dataclass used by encoders
# -----------------------------
@dataclass
class BlockSlices:
    # contiguous index ranges in the flat centralized state vector (end-exclusive)
    ally_units: slice
    enemy_units: slice
    last_actions: Optional[slice]
    timestep: Optional[slice]

    # meta (kept by encoders)
    d_unit_ally: int
    d_unit_enemy: int
    U_A: int
    U_E: int
    n_actions: int

    # internal: canonical reorder used to build the contiguous layout (original -> contiguous)
    # not used by encoders.py, but useful if callers want to apply the same reindex
    _order: Optional[List[int]] = None
    _tails: Optional[Dict[str, slice]] = None  # geometry/composition/history slices (contiguous)


# -----------------------------
# Helpers for spans/indices
# -----------------------------
def _flatten_spans(spans: List[Tuple[int, int]]) -> List[int]:
    idx: List[int] = []
    for a, b in spans:
        a, b = int(a), int(b)
        if b > a:
            idx.extend(range(a, b))
    return idx

def _indices_from(layout: Dict[str, Any],
                  contig_key: str,
                  spans_key: str,
                  indices_key: str) -> List[int]:
    """Prefer explicit indices, then spans, finally contiguous [start,end)."""
    if indices_key in layout and layout[indices_key]:
        return [int(i) for i in layout[indices_key]]
    if spans_key in layout and layout[spans_key]:
        return _flatten_spans([(int(a), int(b)) for a, b in layout[spans_key]])
    if contig_key in layout and layout[contig_key] not in (None, [-1, -1]):
        s = layout[contig_key]
        return list(range(int(s[0]), int(s[1])))
    return []

def _tail_range(t: Dict[str, Any]) -> List[int]:
    return list(range(int(t["start"]), int(t["end"])))


# -----------------------------
# Public: gather a block from a 1D numpy vector
# -----------------------------
def gather_block(vec: np.ndarray,
                 contig: Optional[slice] = None,
                 spans: Optional[List[Tuple[int, int]]] = None,
                 indices: Optional[List[int]] = None) -> np.ndarray:
    """
    Returns a 1D contiguous view/array by:
      indices > spans > contig
    """
    if indices is not None:
        return vec[np.asarray(indices, dtype=np.int64)]
    if spans:
        parts = [vec[slice(a, b)] for a, b in spans if (b > a)]
        return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=vec.dtype)
    if contig is not None:
        return vec[contig]
    return np.empty((0,), dtype=vec.dtype)


# -----------------------------
# Build a canonical reorder and a contiguous BlockSlices the encoder expects
# -----------------------------
def from_state_layout(layout_dict: Dict[str, Any]) -> BlockSlices:
    """
    Accept a (possibly non-contiguous) layout from the env and
    construct a contiguous canonical layout for encoders:

      [ALLY (U_A*dA), ENEMY (U_E*dE), GEOMETRY, COMPOSITION, HISTORY]

    Returns BlockSlices with ally_units/enemy_units as contiguous slices
    into that canonical order. Also stores the reorder vector in the
    private field `_order` for callers who want to reindex original states.
    """
    # ---- collect indices for each semantic block from the ORIGINAL vector ----
    ally_idx  = _indices_from(layout_dict, "ally_slice",  "ally_spans",  "ally_indices")
    enemy_idx = _indices_from(layout_dict, "enemy_slice", "enemy_spans", "enemy_indices")

    # tails by name (keep order geometry -> composition -> history if present)
    tails = {t["name"]: _tail_range(t) for t in layout_dict.get("tails", [])}
    geom_idx = tails.get("geometry", [])
    comp_idx = tails.get("composition", [])
    hist_idx = tails.get("history", [])

    # ---- canonical order for the encoder ----
    order: List[int] = ally_idx + enemy_idx + geom_idx + comp_idx + hist_idx

    # lengths
    ally_len = len(ally_idx)
    enemy_len = len(enemy_idx)
    geom_len = len(geom_idx)
    comp_len = len(comp_idx)
    hist_len = len(hist_idx)

    # ---- contiguous slices in the CANONICAL (reindexed) space ----
    ally_start, ally_end   = 0, ally_len
    enemy_start, enemy_end = ally_end, ally_end + enemy_len
    geom_start,  geom_end  = enemy_end, enemy_end + geom_len
    comp_start,  comp_end  = geom_end,  geom_end  + comp_len
    hist_start,  hist_end  = comp_end,  comp_end  + hist_len

    # ---- optional tails for convenience (not used by encoders.py) ----
    tails_contig: Dict[str, slice] = {}
    if geom_len: tails_contig["geometry"] = slice(geom_start, geom_end)
    if comp_len: tails_contig["composition"] = slice(comp_start, comp_end)
    if hist_len or ("history" in tails): tails_contig["history"] = slice(hist_start, hist_end)

    # ---- meta copied as-is (encoders rely on these) ----
    d_unit_ally  = int(layout_dict["d_unit_ally"])
    d_unit_enemy = int(layout_dict["d_unit_enemy"])
    U_A = int(layout_dict["U_A"])
    U_E = int(layout_dict["U_E"])
    n_actions = int(layout_dict["n_actions"])

    # Sanity: ally/enemy unit dims must match
    assert ally_len == U_A * d_unit_ally, f"Ally length {ally_len} != U_A*dA {U_A*d_unit_ally}"
    assert enemy_len == U_E * d_unit_enemy, f"Enemy length {enemy_len} != U_E*dE {U_E*d_unit_enemy}"

    return BlockSlices(
        ally_units=slice(ally_start, ally_end),
        enemy_units=slice(enemy_start, enemy_end),
        last_actions=None,         # Simple115 baseline has no history tail
        timestep=None,             # not used in Simple115
        d_unit_ally=d_unit_ally,
        d_unit_enemy=d_unit_enemy,
        U_A=U_A,
        U_E=U_E,
        n_actions=n_actions,
        _order=order,
        _tails=tails_contig,
    )


# -----------------------------
# Public: apply the canonical reorder to numpy arrays
# -----------------------------
def rearrange_state_np(x: np.ndarray, slices: BlockSlices) -> np.ndarray:
    """
    Reindex the ORIGINAL Simple115 vector(s) into the canonical contiguous layout
    expected by the returned BlockSlices.

    x can be shape [D] or [N,D]; returns same shape with columns permuted by slices._order.
    """
    order = np.asarray(slices._order or [], dtype=np.int64)
    if order.size == 0:
        return x  # nothing to do

    if x.ndim == 1:
        return x[order]
    elif x.ndim == 2:
        return x[:, order]
    else:
        raise ValueError(f"Unsupported ndim for numpy array: {x.ndim}")

