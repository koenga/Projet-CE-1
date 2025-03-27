# -*- coding: utf-8 -*-

from RegionMap import RegionMap
from Fleet import Monitoring_Fleet

import numpy as np
import os 
from datetime import datetime

if __name__ == '__main__':

    N_episodes = 1000
    ep_len     = 5
    
    buffer_size = 6000
    batch_size = 32
    
    # ----- Create a map -----
    
    x_size = 20
    y_size = 30
    
    list_of_small_pertb = [[0,4], [13,5], [17,25]]
    list_of_big_pert = [[6, 6], [6, 27], [18, 6], [18, 27]]
    
    Map = RegionMap(x_size, y_size, list_of_small_pertb, list_of_big_pert)
    Map.initialize_importance_map()
    
    # ----- Create a fleet -----
    
    drone_init_pos = [[1,1], [10, 1], [17, 1], [2,15], [11,15], [18, 15], [1,28], [9,28], [18, 28]] 
  
    state_shape    = [ep_len, 13]    
 
    pretrained_folder = None    
    num_stacked = 1
    alpha = 1e-4
 
    F = Monitoring_Fleet(Map, buffer_size, state_shape, pretrained_folder, False, num_stacked, alpha)
    F.add_drones(drone_init_pos)    
    
    #----- Create save folder paths -----
    
    np.set_printoptions(threshold=np.inf)
    
    current_folder = os.getcwd()
    save_folder = current_folder + "/Results"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)    
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")    
    
    current_results = save_folder + "/" + date_time
    if not os.path.exists(current_results):
        os.makedirs(current_results)     
    
    #----- Simultaion -----
    
    t_curr = 0
    
    ep_curr = 0 
    
    T = int(N_episodes * ep_len)
    
    while ep_curr < N_episodes:
        
        print('Episode: ',ep_curr)
        
        ep_step = 0
        
        while ep_step < ep_len:          
        
            # ------------------------------------------------------------------------------------------
            
            print('Time | ep_step: ', t_curr, ep_step)
            
            if t_curr % 50 == 0:
                F.plot_fleets_trajectories(t_curr, T, masked=True)
            
            if t_curr == 0:
            
                list_of_observations, list_of_drone_positions = F.get_fleet_info(t_curr, T)
                list_of_drone_states = F.get_drone_states(list_of_observations, list_of_drone_positions)
                visit_matrix = F.visit_matrix
                
                F.update_drone_state_history(list_of_drone_states)                   
                
            #------------- MOVE -------------------------------------------------------------------------------------
        
            list_of_actions = F.move_drones_random(list_of_drone_states, list_of_observations, discount_param=0.3, recover_param=0.025)

            #--------------------------------------------------------------------------------------------------------------------
            
            F.update_drone_action_history(list_of_actions)
            
            list_of_observations_, list_of_drone_positions_ = F.get_fleet_info(t_curr+1, T)
            list_of_drone_states_ = F.get_drone_states(list_of_observations_, list_of_drone_positions_)
            visit_matrix_ = F.visit_matrix
            
            F.update_drone_state_history(list_of_drone_states_)
            
            #----- Reward calculation -----
        
            list_of_rewards = F.get_reward(list_of_observations, list_of_observations_, visit_matrix, visit_matrix_, alpha1=1.0, alpha2=0.5)
            
            F.update_drone_reward_history(list_of_rewards)
            
            #----- Re-assign states for next iteration -----
        
            list_of_observations    = list_of_observations_
            list_of_drone_positions = list_of_drone_positions_
            list_of_drone_states    = list_of_drone_states_
        
            t_curr  += 1 
            ep_step +=1

        #----- Store transitions in the buffer -----
            
        ep_curr += 1
        
        print("-----------")
    
    F.create_data_set()
    
    F.buffer.save(current_results)
     