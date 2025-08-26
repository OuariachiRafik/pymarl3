from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

@dataclass
class BlockSlices:
    # contiguous index ranges in the flat centralized state vector
    ally_units: slice
    enemy_units: slice
    last_actions: Optional[slice]
    timestep: Optional[slice]
    # per-unit dims can differ in SMAC/SMACv2
    d_unit_ally: int
    d_unit_enemy: int
    U_A: int
    U_E: int
    n_actions: int

def from_state_layout(layout_dict: Dict) -> BlockSlices:
    """
    Build BlockSlices from env_wrapper.get_state_layout().to_dict()
    """
    ally_s   = layout_dict["ally_slice"]
    enemy_s  = layout_dict["enemy_slice"]
    ally_sl  = slice(ally_s[0], ally_s[1])
    enemy_sl = slice(enemy_s[0], enemy_s[1])

    last_actions_sl: Optional[slice] = None
    timestep_sl: Optional[slice] = None
    for t in layout_dict["tails"]:
        if t["name"] == "last_actions":
            last_actions_sl = slice(t["start"], t["end"])
        elif t["name"] == "timestep":
            timestep_sl = slice(t["start"], t["end"])

    return BlockSlices(
        ally_units=ally_sl,
        enemy_units=enemy_sl,
        last_actions=last_actions_sl,
        timestep=timestep_sl,
        d_unit_ally=int(layout_dict["d_unit_ally"]),
        d_unit_enemy=int(layout_dict["d_unit_enemy"]),
        U_A=int(layout_dict["U_A"]),
        U_E=int(layout_dict["U_E"]),
        n_actions=int(layout_dict["n_actions"]),
    )
