# -*- coding: utf-8 -*-

import numpy as np
import random

class Drone():
    
    def __init__(self, x, y):
        
        #----- Coordinates -----
        
        self.x = x
        self.y = y
        
        self.list_of_move_actions = [0, 1, 2, 3, 4] #up, down, left, right, stay
        
        self.max_stay = 1
        self.stay_counter = 0
        
        #----- Store values for plotting -----
        
        self.list_of_observations     = []
        self.list_of_actions_taken    = []
        
        self.list_of_visited_nodes    = [[x,y]]
        self.history = []
        
        self.list_of_visited_importance = []
        
        #----- Zig zag states -----
        
        self.state_zz    = 'up'
        self.move_action = 0
        self.counter_zz  = 0
        self.sweep_dir   = 'right'

        #----- LSTM values -----
        
        self.state_history  = []
        self.action_history = []
        self.reward_history = []
    
    def random_init(self, Map):
        
        x_size = Map.x_size
        y_size = Map.y_size
        
        self.x = np.random.randint(0, high=x_size)
        self.y = np.random.randint(0, high=y_size)
        
    def info_adjacent_nodes(self, t, x, y, Map, visit_mask, T):
        
        temporal_importance_values = []
        real_temporal_importance_values = []
        
        adjacent_nodes = [[x-1,y], [x+1,y], [x,y-1], [x,y+1], [x,y]] # same convention [up, down, left, right, stay]
        
        for given_x, given_y in adjacent_nodes:
            
            if 0 <= given_x <= Map.x_size-1 and 0 <= given_y <= Map.y_size-1:
                
                m = Map.dynamic_importance(t, given_x, given_y, T)

                temporal_importance_values.append(m * visit_mask[given_x, given_y] * Map.not_obstacles_mask[given_x, given_y])
                real_temporal_importance_values.append(m)
                
            else:
                
                temporal_importance_values.append(0.0)
                real_temporal_importance_values.append(0.0)
                
        return adjacent_nodes, temporal_importance_values, real_temporal_importance_values
    
    def get_observation(self, t, x, y, Map, visit_mask, T, shared_info=None):
        
        observation = {}
        observation["x"] = x
        observation["y"] = y
        
        adjacent_nodes, temporal_importance_values, real_temporal_importance_values = self.info_adjacent_nodes(t, x, y, Map, visit_mask, T)
        
        observation["adjacent_nodes"] = adjacent_nodes
        observation["temporal_importance_values"] = temporal_importance_values 
        observation["real_temporal_importance_values"] = real_temporal_importance_values
        
        observation["shared_info"] = shared_info
        
        return observation
    
    def feasible_actions(self, Map):
        
        given_x = self.x
        given_y = self.y

        feasible_actions = [4]
                
        if 0 <= given_x - 1:
            feasible_actions.append(0)
            
        if given_x + 1 <= Map.x_size - 1:
            feasible_actions.append(1)
            
        if 0 <= given_y - 1:
            feasible_actions.append(2)
            
        if given_y + 1 <= Map.y_size - 1:
            feasible_actions.append(3)
            
        return feasible_actions

#----- Different drone behaviours -----
    
    def random_agent_choose_action(self, observation):
        
        adjacent_temporal_importance_values = observation["temporal_importance_values"]
        
        feasible_actions = []
        
        for i in range(len(adjacent_temporal_importance_values)):
            
            ativ   = adjacent_temporal_importance_values[i]
            action = self.list_of_move_actions[i]

            print(f"Action: {action} in random_agent_choose_action")
            
            if not ativ == 0.0:
                print("Went in if not ativ == 0 in random_agent_choose_action")
                
                feasible_actions.append(action)
        
        if self.stay_counter == self.max_stay:

            print("Went in the if in random_agent_choose_action")
            
            self.stay_counter = 0
            
            feasible_actions = [action for action in feasible_actions if action!=4]

        print(f"list of feasible action (random_agent_choose_action): {feasible_actions}")
            
        move_action = random.choice(feasible_actions)
        if move_action == 4:
            self.stay_counter += 1
        
        return move_action

    def greedy_agent_choose_action(self, observation):
        
        adjacent_temporal_importance_values = observation["temporal_importance_values"]
        
        feasible_actions       = []
        feasible_action_values = []
        
        for i in range(len(adjacent_temporal_importance_values)):
            
            ativ   = adjacent_temporal_importance_values[i]
            action = self.list_of_move_actions[i]
            
            if not ativ == 0.0:
                
                feasible_actions.append(action)
                feasible_action_values.append(ativ)
        
        if self.stay_counter == self.max_stay:
            
            self.stay_counter = 0
            
            feasible_actions = [action for action in feasible_actions if action!=4]
            feasible_action_values.pop()
            
        max_index   = feasible_action_values.index(max(feasible_action_values))    
        move_action = feasible_actions[max_index]
    
        if move_action == 4:
            self.stay_counter += 1
        
        return move_action

    def zig_zag_agent_choose_action(self, observation, max_c=1):
        
        adjacent_temporal_importance_values = observation["temporal_importance_values"]
        
        #----- State machine -----
        
        if (self.state_zz == 'up' and adjacent_temporal_importance_values[0] == 0.0) or (self.state_zz == 'down' and adjacent_temporal_importance_values[1] == 0.0):
            
            if self.sweep_dir == 'right' and adjacent_temporal_importance_values[3] > 0.0:
                
                self.move_action = 3
                self.counter_zz  = 0
                self.state_zz    = 'right'
                
            elif self.sweep_dir == 'right':
                
                self.move_action = 2
                self.counter_zz  = 0
                self.state_zz    = 'left'
                self.sweep_dir   = 'left'
                
            elif self.sweep_dir == 'left' and adjacent_temporal_importance_values[2] > 0.0:
                
                self.move_action = 2
                self.counter_zz  = 0
                self.state_zz    = 'left'
                
            elif self.sweep_dir == 'left':
                
                self.move_action = 3
                self.counter_zz  = 0
                self.state_zz    = 'right'
                self.sweep_dir   = 'right'
                
        if (self.state_zz == 'right' and self.counter_zz == max_c) or (self.state_zz == 'left' and self.counter_zz == max_c):
            
            if adjacent_temporal_importance_values[0] > 0.0:
                
                self.state_zz = 'up'
                self.move_action = 0
                
            elif adjacent_temporal_importance_values[1] > 0.0:
                
                self.state_zz = 'down'
                self.move_action = 1
        
        elif self.state_zz == 'right':
            
            if adjacent_temporal_importance_values[3] > 0.0:
                
                self.move_action = 3
                self.counter_zz += 1
                
            elif adjacent_temporal_importance_values[1] > 0.0:
                
                self.state_zz = 'down'
                self.move_action = 1

            elif adjacent_temporal_importance_values[0] > 0.0:
                
                self.state_zz = 'up'
                self.move_action = 0                
            
        elif self.state_zz == 'left':
 
            if adjacent_temporal_importance_values[2] > 0.0:
                
                self.move_action = 2
                self.counter_zz += 1

            elif adjacent_temporal_importance_values[1] > 0.0:
                
                self.state_zz = 'down'
                self.move_action = 1

            elif adjacent_temporal_importance_values[0] > 0.0:
                
                self.state_zz = 'up'
                self.move_action = 0                   
            
        move_action = self.move_action
        
        return move_action

#----- Move drone functions -----
            
    def choose_action(self, observation, agent_type='random'):
        
        if agent_type == 'random':
            
            move_action = self.random_agent_choose_action(observation)
            
        elif agent_type == 'greedy'   :
            
            move_action = self.greedy_agent_choose_action(observation)
            
        elif agent_type == 'zz'   :
            
            move_action = self.zig_zag_agent_choose_action(observation)
            
        return move_action
    
    def update_drone(self, observation, move_action):
        
        if move_action == - np.inf:
            
            self.x = self.x
            self.y = self.y
            
        elif move_action == 0:
            self.x -= 1
        elif move_action == 1:
            self.x += 1 
        elif move_action == 2:
            self.y -= 1
        elif move_action == 3:
            self.y += 1
        
        self.list_of_observations.append(observation)
        self.list_of_actions_taken.append(move_action)
        self.list_of_visited_nodes.append([self.x, self.y])
        self.history.append((self.x,self.y))