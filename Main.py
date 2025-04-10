# -*- coding: utf-8 -*-

from RegionMap import RegionMap
from Fleet import Monitoring_Fleet

import numpy as np
import os 
from datetime import datetime
import pandas as pd

if __name__ == '__main__':

    N_episodes = 300 
    ep_len     = 5
    
    buffer_size = 6000
    batch_size = 32
    
    # ----- Create a map -----
    
    x_size = 20
    y_size = 30
    
    list_of_small_pertb = [[0,4], [13,5], [17,25]]
    list_of_big_pert = [[4,22],[13,15]]

    link = r"D:\EPFL\MA2\Projet\Code\Data\datasets\simbarca\all_agg"
    df_link = pd.read_csv(r"D:\EPFL\MA2\Projet\Code\Data\datasets\simbarca\all_agg\metadata\link_bboxes_clustered.csv")
    listFileNumbers = ['000']
    id = 'ld_speed'

    # Map = RegionMap(y_size, x_size, [], [], 1, df_link = df_link, link = link, listFileNumbers=listFileNumbers)
    # Map.initialize_better_importance_map(id)
    
    Map = RegionMap(x_size, y_size, list_of_small_pertb, list_of_big_pert)
    Map.initialize_importance_map()
    
    # ----- Create a fleet -----
    
    drone_init_pos = [[3,3], [3,26], [16, 3], [16,26]]

    state_shape    = [ep_len, 13]    
    num_stacked = 2    
    alpha = 1e-3
 
    pretrained_folder = None     
    # pretrained_folder = "/Results/11_20_2024_02_14_49_pretraining/Saved_models/policy_network_49999.pt"
 
    F = Monitoring_Fleet(Map, buffer_size, state_shape, pretrained_folder, True, num_stacked, alpha)
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
    
    T = int(N_episodes * ep_len)  # total number of steps chosen for the simulation

    list_of_importance_metric = []
    list_of_patrol_metric     = []
    list_of_reward_metric     = []
    
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
            
            if t_curr > ep_len:
                list_of_actions = F.move_drones_LSTM(list_of_drone_states, list_of_observations, discount_param=0.1, recover_param=0.025)
                
            else:
                list_of_actions = F.move_drones_random(list_of_drone_states, list_of_observations, discount_param=0.1, recover_param=0.025)

            #--------------------------------------------------------------------------------------------------------------------
            
            F.update_drone_action_history(list_of_actions)
            
            list_of_observations_, list_of_drone_positions_ = F.get_fleet_info(t_curr+1, T)
            list_of_drone_states_ = F.get_drone_states(list_of_observations_, list_of_drone_positions_)
            visit_matrix_ = F.visit_matrix
            
            F.update_drone_state_history(list_of_drone_states_)
            
            #----- Reward calculation -----
        
            list_of_rewards = F.get_reward(list_of_observations, list_of_observations_, visit_matrix, visit_matrix_, alpha1=1.0, alpha2=0.5)
            
            F.update_drone_reward_history(list_of_rewards)
            
            #----- Collect metrics -----
        
            importance_metric, patrol_metric, reward_metric = F.get_metrics(list_of_rewards)
        
            list_of_importance_metric.append(importance_metric)
            list_of_patrol_metric.append(patrol_metric)
            list_of_reward_metric.append(reward_metric)  
            
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

    #----- Save metrics -----
    
    np.save(current_results + "/list_of_importance_metric.npy", np.array(list_of_importance_metric))
    np.save(current_results + "/list_of_patrol_metric.npy"    , np.array(list_of_patrol_metric))
    np.save(current_results + "/list_of_reward_metric.npy"    , np.array(list_of_reward_metric))
    np.save(current_results + "/list_of_perc_visited.npy"     , np.array(F.procentage_visited))
     