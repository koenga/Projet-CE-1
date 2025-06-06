# -*- coding: utf-8 -*-

from RegionMap import RegionMap
from Fleet import Monitoring_Fleet

import numpy as np
import os 
from datetime import datetime

if __name__ == '__main__':

    T = 3000
    buffer_size = 6000
    batch_size = 32
    
    # ----- Create a map -----
    
    x_size = 20
    y_size = 30
    
    list_of_small_pertb = [[0,4], [13,5], [17,25]]
    list_of_big_pert = [[6,9], [17,25], [19, 2], [3, 27]]
    
    Map = RegionMap(x_size, y_size, list_of_small_pertb, list_of_big_pert)
    Map.initialize_importance_map()
    
    # ----- Create a fleet -----
    
    drone_init_pos = [[1,1], [10, 1], [17, 1], [2,15], [11,15], [18, 15], [1,28], [9,28], [18, 28]]  
  
    state_shape    = [13]    
 
    pretrained_folder = None    
 
    F = Monitoring_Fleet(Map, buffer_size, state_shape, pretrained_folder)
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
    
    while t_curr < T:
        
        print('Time: ',t_curr)
        
        # ------------------------------------------------------------------------------------------
        
        if t_curr % 50 == 0:
            F.plot_fleets_trajectories(t_curr, T, masked=True)
            
        if t_curr == 0:
            
            list_of_observations, list_of_drone_positions = F.get_fleet_info(t_curr, T)
            list_of_drone_states = F.get_drone_states(list_of_observations, list_of_drone_positions)
            visit_matrix = F.visit_matrix
        
        #------------- MOVE -------------------------------------------------------------------------------------
        
        list_of_actions = F.move_drones_random(list_of_drone_states, list_of_observations, discount_param=0.3, recover_param=0.025)

        #--------------------------------------------------------------------------------------------------------------------

        list_of_observations_, list_of_drone_positions_ = F.get_fleet_info(t_curr+1, T)
        list_of_drone_states_ = F.get_drone_states(list_of_observations_, list_of_drone_positions_)
        visit_matrix_ = F.visit_matrix
        
        list_of_rewards = F.get_reward(list_of_observations, list_of_observations_, visit_matrix, visit_matrix_, alpha1=1.0, alpha2=0.5)
        
        #----- Store transitions in the buffer -----
        
        for state, state_, reward, action in zip(list_of_drone_states, list_of_drone_states_, list_of_rewards, list_of_actions): 
            
            F.buffer.store_transition(state, action, reward, state_)         
        
        #----- Re-assign states for next iteration -----
        
        list_of_observations    = list_of_observations_
        list_of_drone_positions = list_of_drone_positions_
        list_of_drone_states    = list_of_drone_states_
        
        t_curr += 1  
        
    F.buffer.save(current_results)
     