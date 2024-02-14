import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import MultiBinary, MultiDiscrete, Discrete
from fapaihime.allkeys import *
from fapaihime.games.ruler import Ruler
from agents.random_agent import RandomAgent
from fapaihime.AssertHu import AssertHuForTrain, AssertShanten
from copy import deepcopy

map_fan, rev_map_fan, map_34, rev_map_34, map_action, rev_map_action = get_keys()

rev_new_action_map = {
    0: "PASS",
    1: "PLAY",
    2: "HU",
    3: "PENG",
    4: "GANG",
    5: "ANGANG",
    6: "BUGANG",
    7: "CHI"
}

class FapaiHimeEnv(gym.Env):
    def __init__(self, other_agents = [RandomAgent() for i in range(4)]):
        self.observation_space = spaces.Dict({
            "hand_feature": MultiBinary(136),
            "table_played_feature": MultiBinary(136),
            "table_who_feature": MultiBinary([4, 136]),
            "table_whos": MultiBinary([4, 136]),
            "time_feature": MultiDiscrete([200 for i in range(136)]),

            "is_peng": MultiBinary(136),
            "is_gang": MultiBinary(136),
            "is_chi": MultiBinary(136),

            "draw_tile": MultiBinary(34),
            "other_new_played_tile": MultiBinary(34),
            "who_is_play": MultiBinary(4),
                
            "who_am_i": Discrete(4),
            "wind": Discrete(4),
            "is_last": Discrete(2)
                
        })

        self.action_space = spaces.Dict({
            "my_played_output": MultiBinary(34),
            "action_type_output": MultiBinary(8)
        })

        self.other_agents = other_agents
        self.ruler = Ruler()

        self.wind = -1
        self.dealer = -1 


        self.terminated = False

    def _zero(self):

        # Observation

        self.hand_feature = np.zeros(136, dtype=bool)  # maintain
        self.table_played_feature = np.zeros(136, dtype=bool)  # not include hand maintain
        self.table_who_feature = np.zeros((4, 136), dtype=bool)  # maintain
        self.table_whos = np.zeros((4, 136), dtype=bool)  # maintain
        self.time_feature = np.zeros(136, dtype=int)  # not one-hot maintain

        self.is_peng = np.zeros(136, dtype=bool)  # maintain
        self.is_gang = np.zeros(136, dtype=bool)  # maintain
        self.is_chi = np.zeros(136, dtype=bool)  # maintain

        self.draw_tile = np.zeros(34, dtype=bool)  # refresh
        self.other_new_played_tile = np.zeros(34, dtype=bool)  # refresh
        self.who_is_play = np.zeros(4, dtype=bool)  # refresh

        self.who_am_i = 0
        self.wind = (self.wind + 1) % 16

        # Info
        self.opponent_hand_features = np.zeros((4, 136), dtype=bool)
        self.opponent_draw_tiles = np.zeros((4, 34), dtype=bool)
        
        self.opponent_is_pengs = np.zeros((4, 136), dtype=bool)
        self.opponent_is_gangs = np.zeros((4, 136), dtype=bool)
        self.opponent_is_chis = np.zeros((4, 136), dtype=bool)

        self.mountain = np.zeros(136, dtype=int)
        self.mountain_ptr = 0 
        self.time_idx = -1 
        self.dealer = (self.dealer + 1) % 4
        self.last_tile = 0
        self.win_tiles = np.zeros(4, dtype=int) 
        self.win_types = np.zeros(4, dtype=int)

        self.special_play_action = [None for i in range(4)]

        self.terminated = False

        self.other_action = np.zeros((4, 2, 34), dtype=bool)

    def _fapai(self):
        self.mountain = np.arange(136)
        np.random.shuffle(self.mountain)

        for i in range(13):
            self.hand_feature[self.mountain[i]] = 1
        for i in range(4):
            if i != self.who_am_i:
                for j in range(13):
                    self.opponent_hand_features[i][self.mountain[i*13+j]] = 1

        self.who_is_play[(self.dealer+4-1)%4] = 1 
        self.mountain_ptr = 52

        self._no_cpg_action()

    def _get_obs(self):
        self.other_new_played_tile = np.zeros(34, dtype=bool)
        self.other_new_played_tile[self.last_tile] = True
        return {"hand_feature": self.hand_feature,
            "table_played_feature": self.table_played_feature,
            "table_who_feature": self.table_who_feature,
            "table_whos": self.table_whos,
            "time_feature": self.time_feature,

            "is_peng": self.is_peng,
            "is_gang": self.is_gang,
            "is_chi": self.is_chi,

            "draw_tile": self.draw_tile,
            "other_new_played_tile": self.other_new_played_tile,
            "who_is_play": self.who_is_play,

            "who_am_i": self.who_am_i,
            "wind": self.wind,
            "is_last": self.mountain_ptr == 136
        } 

    def _other_get_obs(self, player):
        assert player != self.who_am_i
        obs = deepcopy(self._get_obs())

        obs["hand_feature"] = self.opponent_hand_features[player]

        obs["is_peng"] = self.opponent_is_pengs[player]
        obs["is_gang"] = self.opponent_is_gangs[player]
        obs["is_chi"] = self.opponent_is_chis[player]

        obs["draw_tile"] = self.opponent_draw_tiles[player]
        obs["other_new_played_tile"] = np.zeros(34, dtype=bool)
        obs["other_new_played_tile"][self.last_tile] = 1

        obs["who_am_i"] = player

        return obs

    def _get_info(self):
        return {
                "opponent_hand_features": self.opponent_hand_features, 
                "mountain": self.mountain, 
                "mountain_ptr": self.mountain_ptr, 
                
                "opponent_is_pengs": self.opponent_is_pengs,
                "opponent_is_gangs": self.opponent_is_gangs,
                "opponent_is_chis": self.opponent_is_chis,

                "time_idx": self.time_idx,
                "dealer": self.dealer, 
                "last_tile": self.last_tile,
                "win_tiles": self.win_tiles,
                "win_types": self.win_types
            }

    def reset(self, seed=None, options=None, other_agents=None):
        super().reset(seed=seed)

        self._zero()
        self._fapai()
        if other_agents != None:
            assert len(other_agents) == 4
            self.other_agents = other_agents

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_payoff(self):
        reward = 0
        print("cong!")
        for player in range(4):
            if player == self.who_am_i:
                fan = AssertHuForTrain(self, rev_map_34[self.win_tiles[player]], (player+4-self.dealer)%4, self.wind, self.mountain_ptr == 136)
                if fan >= 8:
                    if self.win_types[player] == 2:
                        reward += 8 * 3 + fan
                    else:
                        reward += (8 + fan) * 3
            else:
                self.hand_feature = self.opponent_hand_features[player]
                fan = AssertHuForTrain(self, rev_map_34[self.win_tiles[player]], (player+4-self.dealer)%4, self.wind, self.mountain_ptr == 136)
                if fan >= 8:
                    if self.win_types[player] == 2:
                        if np.argwhere(self.who_is_play!=0)[0][0] == self.who_am_i:
                            reward -= (8 + fan)
                        else:
                            reward -= 8
                    else:
                        reward -= (8 + fan)
                        
        return reward

    def _get_shanten(self):
        return AssertShanten(self)

    def _other_agents_draw_action(self, player):
            if self.mountain_ptr >= 136:
                self.terminated = True
                return -1
            tile = self.mountain[self.mountain_ptr] // 4
            self.mountain_ptr += 1
                    
            self.win_tiles[player] = tile 

            self.opponent_draw_tiles = np.zeros((4, 34), dtype=bool)
            self.opponent_draw_tiles[player][tile] = 1

            self.ruler.update_draw_tile(tile, self.opponent_hand_features, player)

    def _other_agents_play_action(self, player, action):
        tile = np.argwhere(action["my_played_output"]!=0)[0][0]

        self.last_tile = tile
        self.other_new_played_tile = np.zeros(34, dtype=bool)
        self.other_new_played_tile[tile] = 1

        self.time_idx += 1

        self.ruler.update_play_tile(tile, self.time_idx, player, 
                self.opponent_hand_features, self.table_played_feature, 
                self.table_who_feature, self.table_whos, self.time_feature, True)

    def _boss_agent_draw_action(self):
        if self.mountain_ptr >= 136:
            self.terminated = True
            return -1 
        tile = self.mountain[self.mountain_ptr] // 4
        self.mountain_ptr += 1

        self.win_tiles[self.who_am_i] = tile 

        self.draw_tile = np.zeros(34, dtype=bool) 
        self.draw_tile[tile] = 1

        self.ruler.update_draw_tile(tile, self.hand_feature) 
        

    def _no_cpg_action(self):
        player = np.argwhere(self.who_is_play!=0)[0][0]
        self.who_is_play[player] = 0
        player = (player + 1) % 4 
        self.who_is_play[player] = 1

        if player != self.who_am_i: 
            return_value = self._other_agents_draw_action(player) 
            if return_value == -1:
                return -1

            obs = self._other_get_obs(player)

            # Give other player last turn self action
            tile_output = np.zeros(34, dtype=bool) 
            action_output = np.zeros(8, dtype=bool)
            for i in range(len(self.other_action[player][0])):
                if self.other_action[player][0][i] == 1:
                    action_output[i] = 1
                
            for i in range(len(self.other_action[player][1])):
                if self.other_action[player][1][i] == 1:
                    tile_output[i] = 1

            pre_action = {"action_type_output": action_output, "my_played_output": tile_output}

            action = self.other_agents[player].step(obs, pre_action)
            action_type = action["action_type_output"]

            # Maintain last action if have
            self.other_action[player] = np.zeros((2, 34), dtype=bool)
            for i in range(len(action_type)):
                if action_type[i] == 1:
                    self.other_action[player][0][i] = 1
                
            for i in range(len(action["my_played_output"])):
                if action["my_played_output"][i] == 1:
                    self.other_action[player][1][i] = 1


            assert action_type[1] == True or action_type[2] == True or action_type[5] == True or action_type[6] == True 
            if action_type[1] == True:
                self._other_agents_play_action(player, action)

            elif action_type[2] == True:
                self.terminated = True
                if sum(self.opponent_hand_features[player]) % 3 == 1: 
                    raise ValueError
                else:
                    self.win_types[player] = 1

            else: 
                return_value = self._other_agents_confirmed_cpgh_action(player, action)
                if return_value == -1:
                    return -1


        else: 
            return_value = self._boss_agent_draw_action()
            if return_value == -1:
                return -1

    def _other_agents_cpgh_action(self): 
        for player in range(4):
            if player != self.who_am_i: 
                obs = self._other_get_obs(player)

                # Give other player last trun self action
                tile_output = np.zeros(34, dtype=bool) 
                action_output = np.zeros(8, dtype=bool)
                for i in range(len(self.other_action[player][0])):
                    if self.other_action[player][0][i] == 1:
                        action_output[i] = 1
                
                for i in range(len(self.other_action[player][1])):
                    if self.other_action[player][1][i] == 1:
                        tile_output[i] = 1

                pre_action = {"action_type_output": action_output, "my_played_output": tile_output}

                action = self.other_agents[player].step(obs, pre_action)
                action_type = action["action_type_output"]

                # Maintain last action if have
                self.other_action[player] = np.zeros((2, 34), dtype=bool)
                for i in range(len(action_type)):
                    if action_type[i] == 1:
                        self.other_action[player][0][i] = 1
                
                for i in range(len(action["my_played_output"])):
                    if action["my_played_output"][i] == 1:
                        self.other_action[player][1][i] = 1
                assert action_type[1] == False 

                if action_type[2] == True: 
                    self.terminated = True
                    if sum(self.opponent_hand_features[player]) % 3 == 1:
                        self.win_tiles[player] = self.last_tile
                        self.win_types[player] = 2
                    else: 
                        raise ValueError

                elif action_type[0] == False: 
                    if action_type[7] == False or player == (np.argwhere(self.who_is_play!=0)[0][0] + 1) % 4:
                        self.special_play_action[player] = action 
                
    def _other_agents_confirmed_cpgh_action(self, player, action):
        action_type = action["action_type_output"]

        tile = np.argwhere(action["my_played_output"])[0][0]


        self.time_idx += 1 

        self.other_new_played_tile = np.zeros(34, dtype=bool) 
        self.ruler.update_special_play(tile, self.time_idx, player, 
                action_type, self.opponent_hand_features, self.table_played_feature, 
                self.table_who_feature, self.table_whos, self.time_feature, 
                self.opponent_is_pengs, self.opponent_is_gangs, self.opponent_is_chis, self.last_tile, True)

        if action_type[4] == True or action_type[5] == True or action_type[6] == True:
            return_value = self._other_agents_draw_action(player)
            if return_value == -1:
                return -1

        obs = self._other_get_obs(player)

        # Give other player last trun self action
        tile_output = np.zeros(34, dtype=bool) 
        action_output = np.zeros(8, dtype=bool)
        for i in range(len(self.other_action[player][0])):
            if self.other_action[player][0][i] == 1:
                action_output[i] = 1
    
        for i in range(len(self.other_action[player][1])):
            if self.other_action[player][1][i] == 1:
                tile_output[i] = 1
        pre_action = {"action_type_output": action_output, "my_played_output": tile_output}

        action = self.other_agents[player].step(obs, pre_action)
        action_type = action["action_type_output"]

        # Maintain last action if have
        self.other_action[player] = np.zeros((2, 34), dtype=bool)
        for i in range(len(action_type)):
            if action_type[i] == 1:
                self.other_action[player][0][i] = 1
    
        for i in range(len(action["my_played_output"])):
            if action["my_played_output"][i] == 1:
                self.other_action[player][1][i] = 1


        assert action_type[1] == True or action_type[2] == True 

        if action_type[1] == True:
            self._other_agents_play_action(player, action)

        else: 
            self.terminated = True
            if sum(self.opponent_hand_features[player]) % 3 == 1: 
                raise ValueError
            else:
                self.win_types[player] = 1


    def _turn(self, action):

        action_type = action["action_type_output"]


        if action_type[0] == False: 
            self.who_is_play = np.zeros(4, dtype=bool)
            self.who_is_play[self.who_am_i] = True
            player = self.who_am_i

            if action_type[1] == True: 
                tile = np.argwhere(action["my_played_output"])[0][0]

                
                self.last_tile = tile 

                self.time_idx += 1 

                self.ruler.update_play_tile(tile, self.time_idx, player, 
                        self.hand_feature, self.table_played_feature, 
                        self.table_who_feature, self.table_whos, self.time_feature)

            elif action_type[2] == True:
                self.terminated = True
                if sum(self.hand_feature) % 3 == 1:
                    self.win_tiles[player] = self.last_tile
                    self.win_types[player] = 2
                else:
                    self.win_types[player] = 1

            else: 
                self.special_play_action[player] = action

        self._other_agents_cpgh_action()

        res = 0
        res_player = -1
        for player in range(4): 
            if self.special_play_action[player] != None:
                special_play_type = np.argwhere(self.special_play_action[player]["action_type_output"])[0][0]
                if res == 0 and special_play_type != 0:
                    res = special_play_type
                    res_player = player
                elif res == 7 and special_play_type >= 3 and special_play_type <= 6: 
                    res = special_play_type
                    res_player = player

        if not self.terminated: 
            player = res_player
            if player != -1:
                action = self.special_play_action[player]
                action_type = action["action_type_output"]
                self.who_is_play = np.zeros(4, dtype=bool)
                self.who_is_play[player] = True

                if player == self.who_am_i: 
                    tile = np.argwhere(action["my_played_output"])[0][0] 

                    self.time_idx += 1 

                    self.ruler.update_special_play(tile, self.time_idx, player, 
                            action_type, self.hand_feature, self.table_played_feature, 
                            self.table_who_feature, self.table_whos, self.time_feature,
                            self.is_peng, self.is_gang, self.is_chi, self.last_tile, False)

                    if action_type[4] == True or action_type[5] == True or action_type[6] == True:
                        return_value = self._boss_agent_draw_action()
                        if return_value == -1:
                            return 0, self._get_obs(), self._get_info()

                else: 
                    return_value = self._other_agents_confirmed_cpgh_action(player, action)
                    if return_value == -1:
                        return 0, self._get_obs(), self._get_info()

            else: 
                return_value = self._no_cpg_action()
                if return_value == -1:
                    return 0, self._get_obs(), self._get_info()

        self.special_play_action = [None for player in range(4)]

        if self.terminated:
            reward = 6 * (self._get_payoff() + 0.1)
        elif action_type[3] == True or action_type[7] == True:
            reward = 0 + 0.1
        else:
            reward = 6 - self._get_shanten() + 0.1
            
        return reward, self._get_obs(), self._get_info()

    def step(self, action):
        reward, observation, info = self._turn(action)

        return observation, reward, self.terminated, False, info

    def close(self):
        pass


    

