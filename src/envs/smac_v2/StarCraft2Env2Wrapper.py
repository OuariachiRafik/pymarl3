#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：API-Network 
@File    ：StarCraft2EnvWrapper.py
@Author  ：Hao Xiaotian
@Date    ：2022/6/13 16:26 
'''

from .official.wrapper import StarCraftCapabilityEnvWrapper

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List

class StarCraft2Env2Wrapper(StarCraftCapabilityEnvWrapper):

    # Add new functions to support permutation operation
    def get_obs_component(self):
        move_feats_dim = self.env.get_obs_move_feats_size()
        enemy_feats_dim = self.env.get_obs_enemy_feats_size()
        ally_feats_dim = self.env.get_obs_ally_feats_size()
        own_feats_dim = self.env.get_obs_own_feats_size()
        obs_component = [move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim]
        return obs_component

    def get_state_component(self):
        if self.env.obs_instead_of_state:
            return [self.env.get_obs_size()] * self.env.n_agents

        nf_al = self.env.get_ally_num_attributes()
        nf_en = self.env.get_enemy_num_attributes()

        enemy_state = self.env.n_enemies * nf_en
        ally_state = self.env.n_agents * nf_al

        size = [ally_state, enemy_state]

        if self.env.state_last_action:
            size.append(self.env.n_agents * self.env.n_actions)
        if self.env.state_timestep_number:
            size.append(1)
        return size

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.env.n_agents,
            "n_enemies": self.env.n_enemies,
            "episode_limit": self.env.episode_limit,

            # New features we added.
            "n_normal_actions": self.env.n_actions_no_attack,
            "n_allies": self.env.n_agents - 1,
            "state_ally_feats_size": self.env.get_ally_num_attributes(),
            "state_enemy_feats_size": self.env.get_enemy_num_attributes(),
            "obs_component": self.get_obs_component(),
            "state_component": self.get_state_component(),
            "map_type": self.env.map_type,
        }
        print(env_info)
        return env_info

    def _get_medivac_ids(self):
        medivac_ids = []
        for al_id, al_unit in self.env.agents.items():
            if self.env.map_type == "MMM" and al_unit.unit_type == self.env.medivac_id:
                medivac_ids.append(al_id)
        print(medivac_ids)  # [9]
        return medivac_ids

class TailSlice:
    name: str
    start: int
    end: int
    shape: Tuple[int, ...]  # e.g., (n_agents, n_actions) or (1,)

@dataclass
class StateLayout:
    total_dim: int
    U_A: int
    U_E: int
    n_actions: int
    d_unit_ally: int
    d_unit_enemy: int
    ally_slice: slice
    enemy_slice: slice
    tails: List[TailSlice]

    def to_dict(self) -> Dict:
        return {
            "total_dim": self.total_dim,
            "U_A": self.U_A,
            "U_E": self.U_E,
            "n_actions": self.n_actions,
            "d_unit_ally": self.d_unit_ally,
            "d_unit_enemy": self.d_unit_enemy,
            "ally_slice": (self.ally_slice.start, self.ally_slice.stop),
            "enemy_slice": (self.enemy_slice.start, self.enemy_slice.stop),
            "tails": [asdict(t) for t in self.tails],
        }

def get_state_layout(self) -> StateLayout:
    """
    Deterministic global-state schema: exact slices for [allies | enemies | tails...].
    Uses only env_info your wrapper already exposes.
    """
    info = self.get_env_info()
    state_dim      = int(info["state_shape"])
    U_A            = int(info["n_agents"])
    U_E            = int(info["n_enemies"])
    n_actions      = int(info["n_actions"])
    d_unit_ally    = int(info["state_ally_feats_size"])
    d_unit_enemy   = int(info["state_enemy_feats_size"])
    component_list = list(info["state_component"])  # ordered chunk lengths

    # Sanity: first two entries must be ally_state and enemy_state (your get_state_component guarantees this)
    ally_len  = U_A * d_unit_ally
    enemy_len = U_E * d_unit_enemy
    assert component_list[0] == ally_len,  f"ally_state len mismatch: {component_list[0]} vs {ally_len}"
    assert component_list[1] == enemy_len, f"enemy_state len mismatch: {component_list[1]} vs {enemy_len}"

    idx = 0
    ally_slice  = slice(idx, idx + ally_len);  idx += ally_len
    enemy_slice = slice(idx, idx + enemy_len); idx += enemy_len

    tails: List[TailSlice] = []
    # Optional third chunk: last actions (n_agents * n_actions)
    if len(component_list) >= 3 and component_list[2] > 0:
        la_len = component_list[2]
        assert la_len == U_A * n_actions, f"last_actions len mismatch: {la_len} vs {U_A*n_actions}"
        tails.append(TailSlice("last_actions", idx, idx + la_len, (U_A, n_actions)))
        idx += la_len

    # Optional fourth chunk: timestep scalar
    if len(component_list) >= 4 and component_list[3] > 0:
        ts_len = component_list[3]
        assert ts_len == 1, f"timestep chunk expected to be 1, got {ts_len}"
        tails.append(TailSlice("timestep", idx, idx + 1, (1,)))
        idx += 1

    # Final sanity: consumed exactly the state vector
    assert idx == state_dim, f"State layout mismatch: consumed {idx} of {state_dim}"

    return StateLayout(
        total_dim=state_dim,
        U_A=U_A, U_E=U_E, n_actions=n_actions,
        d_unit_ally=d_unit_ally, d_unit_enemy=d_unit_enemy,
        ally_slice=ally_slice, enemy_slice=enemy_slice, tails=tails,
    )

    # def reward_battle(self):
    #     """Reward function when self.reward_spare==False.
    #
    #     Fix the **REWARD FUNCTION BUG** of the original starcraft2.py.
    #
    #     We carefully check the code and indeed find some code error in starcraft2.py.
    #     The error is caused by the incorrect reward calculation for the shield regeneration process and this error will
    #     only occur for scenarios where the enemies are Protoss units.
    #
    #     (1) At line 717 of reward_battle() of starcraft2.py, the reward is computed as: reward = abs(delta_enemy).
    #         Normally, when the agents attack the enemies, delta_enemy will > 0 and thus the agents will be rewarded for attacking enemies.
    #
    #     (2) For Protoss enemies, delta_enemy can < 0 due to the shield regeneration. However, due to the abs() taken over delta_enemy,
    #         the agents will still be rewarded when the enemies' shields regenerate. This incorrect reward will lead to undesired behaviors,
    #         e.g., attacking the enemies but not killing them and waiting their shields regenerating.
    #
    #     (3) Due to the PI/PE design and the improved representational capacity, HPN-QMIX is more sensitive to such
    #         incorrect rewards and sometimes learn strange behaviors.
    #
    #     Returns accumulative hit/shield point damage dealt to the enemy
    #     + reward_death_value per enemy unit killed, and, in case
    #     self.reward_only_positive == False, - (damage dealt to ally units
    #     + reward_death_value per ally unit killed) * self.reward_negative_scale
    #     """
    #     if self.reward_sparse:
    #         return 0
    #
    #     reward = 0
    #     delta_deaths = 0
    #     delta_ally = 0
    #     delta_enemy = 0
    #
    #     neg_scale = self.reward_negative_scale
    #
    #     # update deaths
    #     for al_id, al_unit in self.agents.items():
    #         if not self.death_tracker_ally[al_id]:
    #             # did not die so far
    #             prev_health = (
    #                     self.previous_ally_units[al_id].health
    #                     + self.previous_ally_units[al_id].shield
    #             )
    #             if al_unit.health == 0:
    #                 # just died
    #                 self.death_tracker_ally[al_id] = 1
    #                 if not self.reward_only_positive:
    #                     delta_deaths -= self.reward_death_value * neg_scale
    #                 delta_ally += prev_health * neg_scale
    #             else:
    #                 # still alive
    #                 delta_ally += neg_scale * (
    #                         prev_health - al_unit.health - al_unit.shield
    #                 )
    #
    #     for e_id, e_unit in self.enemies.items():
    #         if not self.death_tracker_enemy[e_id]:
    #             prev_health = (
    #                     self.previous_enemy_units[e_id].health
    #                     + self.previous_enemy_units[e_id].shield
    #             )
    #             if e_unit.health == 0:
    #                 self.death_tracker_enemy[e_id] = 1
    #                 delta_deaths += self.reward_death_value
    #                 delta_enemy += prev_health
    #             else:
    #                 delta_enemy += prev_health - e_unit.health - e_unit.shield
    #
    #     if self.reward_only_positive:
    #         ###### reward = abs(delta_enemy + delta_deaths)  # shield regeneration (the original wrong implementation)
    #         # reward = max(delta_enemy, 0) + delta_deaths  # only consider the shield damage
    #         reward = delta_enemy + delta_deaths  # consider the `+shield-damage` and the `-shield-regeneration`
    #     else:
    #         reward = delta_enemy + delta_deaths - delta_ally
    #
    #     return reward
