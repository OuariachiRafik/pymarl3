from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any
import numpy as np

@dataclass
class BlockSlices:
    ally_units: Optional[slice]
    enemy_units: Optional[slice]
    ally_spans: Optional[List[Tuple[int, int]]]
    enemy_spans: Optional[List[Tuple[int, int]]]
    ally_indices: Optional[List[int]]
    enemy_indices: Optional[List[int]]
    tails: Dict[str, slice]
    last_actions: Optional[slice]
    timestep: Optional[slice]
    d_unit_ally: int
    d_unit_enemy: int
    U_A: int
    U_E: int
    n_actions: int


def _to_slice(pair: List[int]) -> slice:
    return slice(int(pair[0]), int(pair[1]))


def from_state_layout(layout_dict: Dict[str, Any]) -> BlockSlices:
    
    ally_sl = None
    enemy_sl = None
    if "ally_slice" in layout_dict and layout_dict["ally_slice"] not in (None, [-1, -1]):
        s = layout_dict["ally_slice"]
        ally_sl = _to_slice(s)
    if "enemy_slice" in layout_dict and layout_dict["enemy_slice"] not in (None, [-1, -1]):
        s = layout_dict["enemy_slice"]
        enemy_sl = _to_slice(s)

    ally_spans = None
    enemy_spans = None
    if "ally_spans" in layout_dict and layout_dict["ally_spans"]:
        ally_spans = [(int(a), int(b)) for a, b in layout_dict["ally_spans"]]
    if "enemy_spans" in layout_dict and layout_dict["enemy_spans"]:
        enemy_spans = [(int(a), int(b)) for a, b in layout_dict["enemy_spans"]]

    ally_indices = layout_dict.get("ally_indices")
    enemy_indices = layout_dict.get("enemy_indices")

    tails_dict: Dict[str, slice] = {}
    last_actions_sl: Optional[slice] = None
    timestep_sl: Optional[slice] = None
    for t in layout_dict.get("tails", []):
        name = t["name"]
        sl = slice(int(t["start"]), int(t["end"]))
        tails_dict[name] = sl
        if name == "last_actions":
            last_actions_sl = sl
        elif name == "timestep":
            timestep_sl = sl

    return BlockSlices(
        ally_units=ally_sl,
        enemy_units=enemy_sl,
        ally_spans=ally_spans,
        enemy_spans=enemy_spans,
        ally_indices=ally_indices,
        enemy_indices=enemy_indices,
        tails=tails_dict,
        last_actions=last_actions_sl,
        timestep=timestep_sl,
        d_unit_ally=int(layout_dict["d_unit_ally"]),
        d_unit_enemy=int(layout_dict["d_unit_enemy"]),
        U_A=int(layout_dict["U_A"]),
        U_E=int(layout_dict["U_E"]),
        n_actions=int(layout_dict["n_actions"]),
    )


def gather_block(vec: np.ndarray,
                 contig: Optional[slice] = None,
                 spans: Optional[List[Tuple[int, int]]] = None,
                 indices: Optional[List[int]] = None) -> np.ndarray:
                     
    if indices is not None:
        return vec[np.asarray(indices, dtype=int)]
    if spans:
        parts = [vec[slice(a, b)] for a, b in spans]
        return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=vec.dtype)
    if contig is not None:
        return vec[contig]
    return np.empty((0,), dtype=vec.dtype)
