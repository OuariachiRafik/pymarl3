from dataclasses import dataclass
from typing import Optional


@dataclass
class BlockSlices:
    # contiguous index ranges in the flat centralized state vector
    ally_units: slice
    enemy_units: slice
    ally_cooldown: Optional[slice]
    ally_energy: Optional[slice]
    ally_last_actions: Optional[slice]
    # bookkeeping
    d_unit: int
    U_A: int
    U_E: int
    n_actions: int


def infer_block_slices(
    state_dim: int,
    n_agents: int,
    n_actions: int,
    race_type_dim: int = 3,  # number of unit-type channels per race
    enemies_hint: Optional[int] = None,
    has_cooldown_energy: bool = True,
    has_last_actions: bool = True,
) -> BlockSlices:
    """
    Heuristically parse SMAC/SMACv2 centralized state into contiguous blocks:
      [ allies | enemies | ally_cooldown | ally_energy | ally_last_actions ]
    Falls back to a conservative parse if exact match doesn't hold.
    """
    d_unit = 4 + race_type_dim   # [x,y,health,shield] + unit-type one-hot(R)
    U_A = n_agents

    rem = state_dim
    d_allies = U_A * d_unit
    if rem < d_allies:
        raise ValueError(f"state_dim too small: {state_dim} < allies {d_allies}")
    rem -= d_allies

    # If we know U_E, use it; else try to infer later.
    U_E = enemies_hint if enemies_hint is not None else -1

    # Reserve optional tails if configured
    d_cool = U_A if has_cooldown_energy else 0
    d_energy = U_A if has_cooldown_energy else 0
    d_last = U_A * n_actions if has_last_actions else 0

    # If enemies unknown, infer from remaining dimensions
    if U_E < 0:
        # Assume enemies block is multiple of d_unit, prior to tails
        rem_no_tails = rem - (d_cool + d_energy + d_last)
        if rem_no_tails < 0 or rem_no_tails % d_unit != 0:
            # Fallback: put as many enemies as possible; clamp >=0
            U_E = max(0, rem // d_unit)
        else:
            U_E = rem_no_tails // d_unit

    d_enemies = U_E * d_unit

    # Recompute tails check
    calc = d_allies + d_enemies + d_cool + d_energy + d_last
    if calc != state_dim:
        # If mismatch, try dropping tails gracefully
        d_cool = d_energy = d_last = 0
        calc = d_allies + d_enemies
    if calc != state_dim:
        # Final fallback: cap enemies so that allies+enemies == state_dim
        U_E = max(0, (state_dim - d_allies) // d_unit)
        d_enemies = U_E * d_unit
        calc = d_allies + d_enemies
        if calc != state_dim:
            # Give up: treat everything after allies as enemies
            U_E = (state_dim - d_allies) // d_unit
            d_enemies = U_E * d_unit

    off = 0
    ally_units = slice(off, off + d_allies); off += d_allies
    enemy_units = slice(off, off + d_enemies); off += d_enemies

    ally_cooldown = ally_energy = ally_last_actions = None
    if has_cooldown_energy and off + U_A <= state_dim:
        ally_cooldown = slice(off, off + U_A); off += U_A
        if off + U_A <= state_dim:
            ally_energy = slice(off, off + U_A); off += U_A
    if has_last_actions and off + U_A * n_actions <= state_dim:
        ally_last_actions = slice(off, off + U_A * n_actions); off += U_A * n_actions

    return BlockSlices(
        ally_units=ally_units,
        enemy_units=enemy_units,
        ally_cooldown=ally_cooldown,
        ally_energy=ally_energy,
        ally_last_actions=ally_last_actions,
        d_unit=d_unit,
        U_A=U_A,
        U_E=U_E,
        n_actions=n_actions,
    )
