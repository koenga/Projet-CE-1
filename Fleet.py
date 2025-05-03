# -*- coding: utf-8 -*-
import numpy as np
import os 
import matplotlib.pyplot as plt 

from Drone import Drone
from Buffer import ReplayBuffer
from Neural_net import *

from LSTM_net import *
from GRU_net import *

import torch
import random

class Monitoring_Fleet():
    
    def __init__(self, Map, buffer_size=None, state_shape=None, pretrained_folder=None, lstm=True, num_stacked=2, alpha=1e-4):
        
        self.N = 0
        self.list_of_drones = []
        
        self.Map = Map
        self.y_size = Map.y_size
        self.x_size = Map.x_size
        
        self.state_shape = state_shape
       
        # ----- Shared info -----
        
        self.shared_info = {}
        
        # ----- Matrices for tracking the movement -----
        
        self.visit_matrix = np.ones((self.x_size, self.y_size))
        
        # ----- RL components -----
        
        self.alpha = alpha
        self.gamma = 0.9
        
        self.buffer = ReplayBuffer(buffer_size, state_shape)
        
        self.lstm = None #lstm
        self.gru = True
        
        if lstm:

            self.policy_network = LSTM_model(input_size=state_shape[1], hidden_size=state_shape[1], num_stacked_layers=num_stacked, alpha=self.alpha)
            self.target_network = LSTM_model(input_size=state_shape[1], hidden_size=state_shape[1], num_stacked_layers=num_stacked, alpha=self.alpha)            
        
        elif self.gru:
            self.policy_network = GRU_model(input_size=state_shape[1], hidden_size=state_shape[1], num_stacked_layers=num_stacked, alpha=self.alpha)
            self.target_network = GRU_model(input_size=state_shape[1], hidden_size=state_shape[1], num_stacked_layers=num_stacked, alpha=self.alpha)            
        
        else:
        
            self.policy_network = NN_model(state_shape[0], alpha=self.alpha)
            self.target_network = NN_model(state_shape[0], alpha=self.alpha)     
        
        self.train_counter = 0
        
        self.log_avg_losses = []
        
        self.log_average_score = []
        
        #----- Load pretrained -----
        
        if pretrained_folder is not None:
            
            load_weights = os.getcwd() + pretrained_folder
            self.policy_network.load_checkpoint(load_weights)
            self.target_network.load_checkpoint(load_weights)   
            
        #----- Other params -----
        
        self.list_of_reward_metric = []
        
        self.list_of_visited_cells = []
        
        self.procentage_visited = []
            
    def add_drones(self, drone_init_pos):
        
        self.N += len(drone_init_pos)
        
        for i in range(len(drone_init_pos)):
            
            x = drone_init_pos[i][0]
            y = drone_init_pos[i][1]
            self.list_of_drones.append(Drone(x,y))                                 

    def get_drone_positions(self):
        
        list_of_drone_positions = []
        for drone in self.list_of_drones:
            list_of_drone_positions.append([drone.x, drone.y])
            
        return list_of_drone_positions
    
    def get_fleet_info(self, t, T):
        
        list_of_observations = []
        list_of_drone_positions = self.get_drone_positions()
        
        self.shared_info["visit_matrix"] = self.visit_matrix 
        
        for idx, drone in enumerate(self.list_of_drones):
            
            d_x = drone.x
            d_y = drone.y

            d_observation = drone.get_observation(t, d_x, d_y, self.Map, self.visit_matrix, T, self.shared_info)

            list_of_observations.append(d_observation)
            
        return list_of_observations, list_of_drone_positions 

    def get_masked_dynamic_map(self, t, T):
        
        current_map = self.Map.get_dynamic_map(t, T) # aled peut etre
        
        for x in range(self.x_size):
            for y in range(self.y_size):
                
                current_map [x, y] *= self.visit_matrix[x, y]
                
        return current_map

    def get_metrics(self, list_of_rewards):
        
        importance_metric = 0.0
        patrol_metric     = 0.0
        
        self.list_of_reward_metric.append(np.sum(np.array(list_of_rewards)))
        reward_metric = np.sum(np.array(self.list_of_reward_metric))
        
        for drone in self.list_of_drones:
            
            importance_metric += np.sum(np.array(drone.list_of_visited_importance)) # SUM OF THE IMPORTANCE OF THE VISITED ZONES
            
        patrol_metric = 1.0 -  np.sum(np.sum(self.visit_matrix))/self.x_size/self.y_size # PERCENTAGE YOU HAVE TO VISIT
        
        return importance_metric, patrol_metric, reward_metric

    def update_visit_matrix(self, discount_param=0.3, recover_param=0.025):
        
        list_of_drone_positions = []
        
        b = 1.0
        
        for idx, drone in enumerate(self.list_of_drones):            
            list_of_drone_positions.append([drone.x, drone.y])
            
            if not [drone.x, drone.y] in self.list_of_visited_cells:
                
                self.list_of_visited_cells.append([drone.x, drone.y])
                
        self.procentage_visited.append(len(self.list_of_visited_cells)/self.x_size/self.y_size)
        
        #----- Discount visited fields -----    
        
        for pos in list_of_drone_positions:
            
            self.visit_matrix[pos[0], pos[1]] *= discount_param
            
            x = pos[0]
            y = pos[1]
            
            adjacent_nodes = [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]

            for node in adjacent_nodes:
            
                node_x = node[0]
                node_y = node[1]
            
                if 0 <= node_x <= self.Map.x_size-1 and 0 <= node_y <= self.Map.y_size-1:
                
                    self.visit_matrix[node_x, node_y] *= b
                    
        #----- Refresh other fields -----
            
        for x in range(self.x_size):
            for y in range(self.y_size):
                
                if [x, y] not in list_of_drone_positions:
                    
                    self.visit_matrix[x, y] += recover_param
                    if self.visit_matrix[x, y] > 1.0: self.visit_matrix[x, y] = 1.0 * self.Map.not_obstacles_mask[x, y]
    
    def get_drone_states(self, list_of_observations, list_of_drone_positions):
        
        list_of_drone_states = []
        
        pos_x_sum = 0.0
        pos_y_sum = 0.0
        
        for idx_, drone_ in enumerate(self.list_of_drones):
                
            pos_x_sum += drone_.x
            pos_y_sum += drone_.y
        
        for idx, drone in enumerate(self.list_of_drones):
            
            drone_x = list_of_observations[idx]["x"]
            drone_y = list_of_observations[idx]["y"] 
            
            avg_mask = [0.0, 0.0, 0.0, 0.0] #up, down, left, right, stay
            
            #----- Up -----
            
            n_fields = 0.0
            for index in range(0, drone_x):
                
                if self.Map.not_obstacles_mask[index, drone_y] > 0.0:
                
                    n_fields += 1
                    avg_mask[0] += self.visit_matrix[index, drone_y]  
                
            if n_fields>0:
                
                avg_mask[0] = avg_mask[0]/n_fields
                
            #----- Down -----
            
            n_fields = 0.0
            for index in range(drone_x, self.x_size):

                if self.Map.not_obstacles_mask[index, drone_y] > 0.0:
                
                    n_fields += 1
                    avg_mask[1] += self.visit_matrix[index, drone_y]  
                
            if n_fields>0:
                
                avg_mask[1] = avg_mask[1]/n_fields 
                
            #----- Left -----
            
            n_fields = 0.0
            for index in range(0, drone_y):

                if self.Map.not_obstacles_mask[drone_x, index] > 0.0:
                
                    n_fields += 1
                    avg_mask[2] += self.visit_matrix[drone_x, index]  
                
            if n_fields>0:
                
                avg_mask[2] = avg_mask[2]/n_fields 
                
            #----- Right -----
            
            n_fields = 0.0
            for index in range(drone_y, self.y_size):

                if self.Map.not_obstacles_mask[drone_x, index] > 0.0:
                
                    n_fields += 1
                    avg_mask[3] += self.visit_matrix[drone_x, index]  
                
            if n_fields>0:
                
                avg_mask[3] = avg_mask[3]/n_fields  
                
            #----- Calculate cm of the fleet -----
            
            avg_x = (pos_x_sum - drone_x) / (self.N - 1)
            avg_y = (pos_y_sum - drone_y) / (self.N - 1)
                
            #----- Prepare state -----
            
            coeff1 = 0.1
            coeff2 = 10.0
            
            state = [drone_x * coeff1, drone_y * coeff1]
            
            state += [avg_x * coeff1, avg_y * coeff1]
            
            state += list(np.array(list_of_observations[idx]["temporal_importance_values"]) * coeff2)
            state += avg_mask
            
            list_of_drone_states.append(state)
        
        return list_of_drone_states
        
    def get_reward(self, list_of_observations, list_of_observations_prime, visit_matrix, visit_matrix_prime, alpha1=1.0, alpha2=0.5):
        
        list_of_rewards = []

        coeff = 10.0
        
        for idx in range(len(list_of_observations)):

            adjacent_nodes = list_of_observations[idx]["adjacent_nodes"]
            temporal_importance_values = list_of_observations[idx]["temporal_importance_values"]

            adjacent_nodes_p = list_of_observations_prime[idx]["adjacent_nodes"]
            temporal_importance_values_p = list_of_observations_prime[idx]["temporal_importance_values"]            
            
            #----- Calculate reward components -----
            
            previous_importance = 0.0
            previous_visit_mask = 0.0
            
            for (x, y), importance in zip(adjacent_nodes, temporal_importance_values):
                
                if importance > 0.0: 
                    
                    previous_importance += importance
                    previous_visit_mask += visit_matrix[x, y]
                        
            current_importance = 0.0
            current_visit_mask = 0.0
            
            for (x_p, y_p), importance_p in zip(adjacent_nodes_p, temporal_importance_values_p):
                
                if importance_p > 0.0: 
                    
                    current_importance += importance_p
                    current_visit_mask += visit_matrix_prime[x_p, y_p]
                
            r1_ = current_importance - previous_importance
            r2_ = current_visit_mask - previous_visit_mask
            
            #----- Calculate total reward -----
            
            R = alpha1 * r1_ + alpha2 * r2_
            
            list_of_rewards.append(coeff * R)

        return list_of_rewards
    
#----- LSTM needed functions -----
    
    def update_drone_state_history(self, list_of_drone_states):
        
        for idx, drone in enumerate(self.list_of_drones):
            
            drone.state_history.append(list_of_drone_states[idx])

    def update_drone_action_history(self, list_of_actions):
        
        for idx, drone in enumerate(self.list_of_drones):
            
            drone.action_history.append(list_of_actions[idx])

    def update_drone_reward_history(self, list_of_rewards):
        
        for idx, drone in enumerate(self.list_of_drones):
            
            drone.reward_history.append(list_of_rewards[idx])
            
    def create_data_set(self):
        
        d = self.state_shape[0]
        
        list_of_extended_states  = []
        list_of_extended_states_ = []
        list_of_extended_rewards = []
        list_of_extended_actions = []
    
        for idx, drone in enumerate(self.list_of_drones):
            
            start_id   = len(drone.state_history) - d
            
            r_and_a_id = len(drone.reward_history) - 1
            
            while start_id - 1 >= 0 and r_and_a_id >= 0:
                
                new_state  = drone.state_history[start_id:start_id+d]
                state      = drone.state_history[start_id - 1:start_id - 1 + d]
                reward     = drone.reward_history[r_and_a_id]
                action     = drone.action_history[r_and_a_id]
                
                list_of_extended_states.append(state)
                list_of_extended_states_.append(new_state)
                list_of_extended_rewards.append(reward)
                list_of_extended_actions.append(action)
                
                start_id   -= 1
                r_and_a_id -= 1
        
        #----- Shuffle before adding -----
        
        combined = list(zip(list_of_extended_states, list_of_extended_states_, list_of_extended_rewards, list_of_extended_actions))
        random.shuffle(combined)
        
        l1s, l2s, l3s, l4s = zip(*combined)
        
        list_of_extended_states  = list(l1s)
        list_of_extended_states_ = list(l2s)
        list_of_extended_rewards = list(l3s)
        list_of_extended_actions = list(l4s)
        
        #----- Add -----
        
        for idx in range(len(list_of_extended_states)):
            
            state  = list_of_extended_states [idx]
            action = list_of_extended_actions[idx]
            reward = list_of_extended_rewards[idx]
            state_ = list_of_extended_states_[idx]
            
            self.buffer.store_transition(state, action ,reward, state_)
        
#----- RL functions -----

    def train(self, n_epochs, states_NN, actions_NN, rewards_NN, next_state_NN, online_train_lr=None):
        
        self.policy_network.train()
        
        #----- Online training with a different lr if needed -----
        
        if online_train_lr is not None:
            self.policy_network.optimizer.lr = online_train_lr   
            
        #----- State -----

        state_batch  = states_NN
        
        reward_batch = rewards_NN
        
        action_batch = actions_NN
        
        state_batch_ = next_state_NN
        
        #----- Convert to tensor -----
        
        state_batch   = torch.tensor(state_batch ).float().to(self.policy_network.device)        
        reward_batch  = torch.tensor(reward_batch).float().to(self.policy_network.device)
        action_batch  = torch.tensor(action_batch).to(torch.int64).to(self.policy_network.device)
        state_batch_  = torch.tensor(state_batch_ ).float().to(self.policy_network.device)  
        
        #----- train -----

        list_of_losses_current = []
        
        for ep in range(n_epochs):
            
            state_action_values = self.policy_network(state_batch).gather(1, action_batch)

            with torch.no_grad():
                
                next_state_values = self.target_network(state_batch_).max(1)[0]      
                
            expected_values = reward_batch + self.gamma * next_state_values
            
            expected_values = expected_values.unsqueeze(1)
            
            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_values)
             
            list_of_losses_current.append(loss.detach().cpu().numpy())
            
            #----- Clear gradients -----
            
            self.policy_network.optimizer.zero_grad()
            loss.backward()
            self.policy_network.optimizer.step()
            
            self.train_counter += 1
            
            if self.train_counter % (int(n_epochs/2)) == 0:

                self.target_network.load_state_dict(self.policy_network.state_dict())
            
        avg_loss = np.sum(np.array(list_of_losses_current))/n_epochs
        
        self.log_avg_losses.append(avg_loss) 
        self.log_average_score.append(np.mean(self.log_avg_losses[-100:]))     
        
    def pretrain(self, ckpt_file_name, N_iter=3000, load_data='/Results/06_14_2024_15_48_00', plot_pretrained=True):

        #----- load_data is the location of the buffer data 
        #----- ckpt_file_name is the name of the folder where we save the results after continued training 
        
        folder = os.getcwd() + load_data
        
        ckpt_file_name = ckpt_file_name + '/Saved_models'
        if not os.path.isdir(ckpt_file_name):
            os.makedirs(ckpt_file_name) 

        self.buffer.load(folder)
        
        print("Counter: ", self.buffer.mem_cntr)
        
        batch_size = 32
        n_epochs = 30
        
        for it in range(N_iter):
            
            states_NN, actions_NN, rewards_NN, next_state_NN = self.buffer.sample_buffer(batch_size)
            self.train(n_epochs, states_NN, actions_NN, rewards_NN, next_state_NN, None)
            
            if it % 1000 == 999:
                
                self.policy_network.save_checkpoint(it, ckpt_file_name)

                if plot_pretrained:
            
                    fig1, ax1 = plt.subplots(dpi=180)
            
                    k = np.arange(len(self.log_average_score))
            
                    ax1.set_yscale('log')
                    ax1.plot(k, self.log_average_score)    
                    ax1.grid('on')
                    ax1.legend()
                    ax1.set_xlabel(r'$k$')
                    ax1.set_ylabel(r'$Loss$')
                    
                    fig1.savefig(ckpt_file_name + "/reward"+"_"+str(it)+".jpg", dpi=180)
                    plt.close(fig1)
                
            if it % 100 == 1:
                
                print("Iteration:", it)
                print("Loss: ", self.log_avg_losses[-1])
                
        if plot_pretrained:
            
            fig1, ax1 = plt.subplots(dpi=180)
            
            k = np.arange(len(self.log_average_score))
            
            ax1.set_yscale('log')
            ax1.plot(k, self.log_average_score)    
            ax1.grid('on')
            ax1.legend()
            ax1.set_xlabel(r'$k$')
            ax1.set_ylabel(r'$Loss$')    
            
            fig1.savefig(ckpt_file_name + "/reward.jpg", dpi=180)
            
        np.save(ckpt_file_name + "/log_average_score.npy", np.array(self.log_average_score))
        np.save(ckpt_file_name + "/log_avg_losses.npy"   , np.array(self.log_avg_losses))
        
    def continue_training(self, ckpt_file_name, lr=1e-4, n_lstm=2, N_iter=3000, load_data='/Results/06_19_2024_20_28_04', pretrained_folder="/Results/06_20_2024_01_31_10_pretraining/Saved_models/policy_network_29999.pt", plot_pretrained=True):
        
        #----- load_data is the location of the buffer data 
        #----- ckpt_file_name is the name of the folder where we save the results after continued training 
        #----- pretrained_folder is the folder where the pretrained weights are stored 
        
        folder = os.getcwd() + load_data
        self.buffer.load(folder)
        
        ckpt_file_name = ckpt_file_name + '/Saved_models'
        if not os.path.isdir(ckpt_file_name):
            os.makedirs(ckpt_file_name) 
        
        if self.lstm:

            self.policy_network = LSTM_model(input_size=self.state_shape[1], hidden_size=self.state_shape[1], num_stacked_layers=n_lstm, alpha=lr)
            self.target_network = LSTM_model(input_size=self.state_shape[1], hidden_size=self.state_shape[1], num_stacked_layers=n_lstm, alpha=lr)

        elif self.gru:
            
            self.policy_network = GRU_model(input_size=self.state_shape[1], hidden_size=self.state_shape[1], num_stacked_layers=n_lstm, alpha=lr)
            self.target_network = GRU_model(input_size=self.state_shape[1], hidden_size=self.state_shape[1], num_stacked_layers=n_lstm, alpha=lr)            
            
        else:
            
            self.policy_network = NN_model(self.state_shape[0], alpha=lr)
            self.target_network = NN_model(self.state_shape[0], alpha=lr) 
            
        if pretrained_folder is not None:
            
            load_weights = os.getcwd() + pretrained_folder
            self.policy_network.load_checkpoint(load_weights)
            self.target_network.load_checkpoint(load_weights) 
            
        batch_size = 64
        n_epochs = 30
        
        for it in range(N_iter):
            
            states_NN, actions_NN, rewards_NN, next_state_NN = self.buffer.sample_buffer(batch_size)
            self.train(n_epochs, states_NN, actions_NN, rewards_NN, next_state_NN)
            
            if it % 1000 == 999:
                
                self.policy_network.save_checkpoint(it, ckpt_file_name)

                if plot_pretrained:
            
                    fig1, ax1 = plt.subplots(dpi=180)
            
                    k = np.arange(len(self.log_average_score))
            
                    ax1.set_yscale('log')
                    ax1.plot(k, self.log_average_score)    
                    ax1.grid('on')
                    ax1.legend()
                    ax1.set_xlabel(r'$k$')
                    ax1.set_ylabel(r'$Loss$')
                    
                    fig1.savefig(ckpt_file_name + "/reward"+"_"+str(it)+".jpg", dpi=180)
                    plt.close(fig1)
                
            if it % 200 == 199:
                
                print("Iteration:", it)
                print("Loss: ", self.log_avg_losses[-1])
                
        if plot_pretrained:
            
            fig1, ax1 = plt.subplots(dpi=180)
            
            k = np.arange(len(self.log_average_score))
            
            ax1.set_yscale('log')
            ax1.plot(k, self.log_average_score)    
            ax1.grid('on')
            ax1.legend()
            ax1.set_xlabel(r'$k$')
            ax1.set_ylabel(r'$Loss$')    
            
            fig1.savefig(ckpt_file_name + "/reward.jpg", dpi=180)
            
        np.save(ckpt_file_name + "/log_average_score.npy", np.array(self.log_average_score))
        np.save(ckpt_file_name + "/log_avg_losses.npy"   , np.array(self.log_avg_losses))   
    
#----- Plotting functions -----

    def plot_fleets_trajectories(self, t, T, masked=False):
        
        fig, ax = plt.subplots()
        
        if masked:
            
            im = ax.imshow(self.get_masked_dynamic_map(t, T), cmap='viridis')
            
        else:
        
            im = ax.imshow(self.Map.get_dynamic_map(t, T), cmap='viridis')
        
        list_of_markers = ["o", "v", "s", "p", "D", "8", "P", "X", "d", "4"]
    
        for idx, drone in enumerate(self.list_of_drones):
            
            positions = drone.history
            x_positions = [pos[0] for pos in positions]
            y_positions = [pos[1] for pos in positions]
            ax.plot(y_positions, x_positions)
            
            marker_idx = list_of_markers[idx]
            plt.scatter(drone.y, drone.x, color='black', facecolors='none',linewidth=1,marker=marker_idx,s=200)
            
        plt.xlabel("y-coordinate")
        plt.ylabel("x-coordinate")
        plt.title("Drone Trajectories on Importance Map")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Importance")
        
        plt.show()

#----- Different fleet agents -----
            
    def move_drones_random(self, list_of_drone_states, list_of_observations, discount_param=0.1, recover_param=0.025):
        
        list_of_actions = [] # stores the actions for all drones
                     
        for idx, drone in enumerate(self.list_of_drones):
            
            d_observation = list_of_observations[idx]
            d_action = drone.choose_action(d_observation, agent_type='random')
            list_of_actions.append(d_action)
            drone.update_drone(d_observation, d_action)
            
        self.update_visit_matrix(discount_param, recover_param)
        
        return list_of_actions

    def move_drones_greedy(self, list_of_drone_states, list_of_observations, discount_param=0.1, recover_param=0.025):
        
        list_of_actions = [] # stores the actions for all drones
                     
        for idx, drone in enumerate(self.list_of_drones):
            
            d_observation = list_of_observations[idx]
            d_action = drone.choose_action(d_observation, agent_type='greedy')
            list_of_actions.append(d_action)
            drone.update_drone(d_observation, d_action)
                    
        self.update_visit_matrix(discount_param, recover_param)
            
        return list_of_actions
    
    def move_drones_NN(self, list_of_drone_states, list_of_observations, discount_param=0.1, recover_param=0.025):
        
        list_of_actions = []
        
        thrs = 0.1
        
        p = np.random.rand() 
    
        if p < thrs:
            
            list_of_actions = self.move_drones_random(list_of_drone_states, list_of_observations, discount_param, recover_param)
            
        else:    
            
            self.policy_network.eval()
            
            for idx, drone in enumerate(self.list_of_drones):
                
                d_state = list_of_drone_states[idx]
                d_observation = list_of_observations[idx]
                
                s = np.array(d_state, dtype=np.float32)
                s = torch.tensor(s).float().unsqueeze(0).to(self.policy_network.device)                
            
                q_values_torch = self.policy_network.forward(s).detach()
                q_values = q_values_torch.cpu().numpy().reshape(5)         
                
                feasible_actions = drone.feasible_actions(self.Map)
    
                q_value_max = - np.inf
                idx_max = - np.inf
                
                for idx_q, q_value in enumerate(q_values):
                    
                    if idx_q in feasible_actions:
                        
                        if q_value > q_value_max:
                            
                            idx_max = idx_q
                            q_value_max = q_value
                
                d_action = idx_max
                
                list_of_actions.append(idx_max)
                drone.update_drone(d_observation, d_action)
                
            self.update_visit_matrix(discount_param, recover_param)
            
        return list_of_actions
    
    def move_drones_LSTM(self, list_of_drone_states, list_of_observations, discount_param=0.1, recover_param=0.025):
        
        list_of_actions = []
        
        thrs = 0.1
        
        p = np.random.rand() 
    
        if p < thrs:
            
            list_of_actions = self.move_drones_random(list_of_drone_states, list_of_observations, discount_param, recover_param)
            
        else:    
            
            self.policy_network.eval()
            
            for idx, drone in enumerate(self.list_of_drones):
                
                start_id = len(drone.state_history) - self.state_shape[0]
                d_state  = drone.state_history[start_id:start_id+self.state_shape[0]]
                
                d_observation = list_of_observations[idx]

                s = np.array(d_state, dtype=np.float32)
                s = s.reshape((-1, self.state_shape[0], self.state_shape[1]))
                s = torch.tensor(s).float().to(self.policy_network.device)                
            
                q_values_torch = self.policy_network.forward(s).detach()
                q_values = q_values_torch.cpu().numpy().reshape(5)         
                
                feasible_actions = drone.feasible_actions(self.Map)
    
                q_value_max = - np.inf
                idx_max = - np.inf
                
                for idx_q, q_value in enumerate(q_values):
                    
                    if idx_q in feasible_actions:
                        
                        if q_value > q_value_max:
                            
                            idx_max = idx_q
                            q_value_max = q_value
                
                d_action = idx_max
                
                list_of_actions.append(idx_max)
                drone.update_drone(d_observation, d_action)                                
            
            #----- Update -----
                  
            self.update_visit_matrix(discount_param, recover_param)
            
        return list_of_actions
    
    
    
    
    
    
    