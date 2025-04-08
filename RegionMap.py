# -*- coding: utf-8 -*-
import numpy as np
import os 
from datetime import datetime
import matplotlib.pyplot as plt

class RegionMap():
    
    def __init__(self, v_size, h_size, list_of_small_pertb, list_of_big_pert):
        
       self.y_size = h_size
       self.x_size = v_size
       
       epsilon_init = 0.000001
       
       self.importance_map_s = epsilon_init * np.ones((self.x_size, self.y_size))
       self.importance_map_b = epsilon_init * np.ones((self.x_size, self.y_size))
       
       self.importance_map = np.zeros((self.x_size, self.y_size))
       
       self.list_of_small_pertb = list_of_small_pertb
       self.list_of_big_pert    = list_of_big_pert
       
       #----- Real map params -----
       
       self.prestored = False
       self.prestored_dynamical_importance_map = None
       
       self.not_obstacles_mask = np.ones((self.x_size, self.y_size))
       self.list_of_obstacles  = []
    
    def initialize_importance_map(self): 
        
        small_amplitude = 0.5
        small_sigma     = 1.0
        
        big_amplitude   = 1.0
        big_sigma       = 3.0
        
        for x in range(self.x_size):
            for y in range(self.y_size):        
                
                for small_p in self.list_of_small_pertb:
                    
                    x_p = small_p[0]
                    y_p = small_p[1] 
                    self.importance_map_s[x,y] = self.importance_map_s[x,y] + small_amplitude * np.exp(-((x-x_p)**2+(y-y_p)**2)/small_sigma**2/2.0)

                for big_p in self.list_of_big_pert:
                    
                    x_p = big_p[0]
                    y_p = big_p[1]      
                    self.importance_map_b[x,y] = self.importance_map_b[x,y] + big_amplitude   * np.exp(-((x-x_p)**2+(y-y_p)**2)/big_sigma**  2/2.0)

                #----- Clip the importance of each type of pert to 1 -----
                
                if self.importance_map_s[x, y] > 1.0: self.importance_map_s[x, y] = 1.0
                if self.importance_map_b[x, y] > 1.0: self.importance_map_b[x, y] = 1.0
                
                self.importance_map[x, y] = self.importance_map_b[x, y] + self.importance_map_s[x, y]
    
    def initialize_better_importance_map(self, id, listFileNumbers):
        
        listAllTensor3D, listAddedTimesteps = createAllTensor3D(listFileNumbers, id)
        mean_tensor3D = averageAllTensor(listAllTensor3D, listAddedTimesteps)

        self.importance_map = mean_tensor3D


    def load_stored_dynamical_importance_map(self, folder): # WILL BE USED, without normalisation ?
        
        path = os.getcwd() + folder
        
        self.prestored = True
        
        matrix = np.load(path)
        
        matrix_min = matrix.min()
        matrix_max = matrix.max()
        
        normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
        scaled_matrix = 1.0 * normalized_matrix
        
        self.prestored_dynamical_importance_map = scaled_matrix
        self.importance_map = scaled_matrix[0, :, :].reshape((self.importance_map.shape[0], self.importance_map.shape[1]))
                        
    def write_to_file(self, filename=None):
        
        path = os.getcwd() + '/Map_configurations'
        if not os.path.isdir(path):
            os.makedirs(path)
        
        if filename == None:
            
            now = datetime.now()
            date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
            filename = + date_time + '_' + 'map.npy'
            
        else:
            
            filename += '_map.npy'
        
        name = path + '/' + filename                       
        
        np.save(name, self.importance_map)

    def load_from_file(self, filename):

        path = os.getcwd() + '/Map_configurations'
        file_path = path + '/' + filename
        
        data = np.load(file_path)
        imp_map = np.array(data, dtype=np.float32)

        self.v_size = np.shape(imp_map)[0]
        self.h_size = np.shape(imp_map)[1]
        
        self.importance_map = imp_map

    def plot_map(self, rmap_values=True, t=None):

        
        fig, ax = plt.subplots()
        
        if t is not None:
            
            _ = plt.imshow(self.prestored_dynamical_importance_map[t, :, :].reshape(self.x_size, self.y_size), cmap = 'viridis')
            
        else:
            
            _ = plt.imshow(self.importance_map, cmap = 'viridis')

        # ----- Annotate values on the RegionMap -----
        
        if rmap_values:
            for i in range(self.x_size):
                for j in range(self.y_size):
                    _ = ax.text(i, j, f'{self.importance_map[i, j]:.2f}', ha='center', va='center', color='black')


        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")

        cbar = plt.colorbar()
        cbar.ax.set_ylabel("", rotation=90)
     
    def plot_drones_on_map(self, list_of_drone_positions, t=None):

        fig, ax = plt.subplots()
        
        if t is not None:
            
            _ = plt.imshow(self.prestored_dynamical_importance_map[t, :, :].reshape(self.x_size, self.y_size), cmap = 'viridis')
            
        else:
            
            _ = plt.imshow(self.importance_map, cmap = 'viridis')

        plt.xlabel("y-coordinate")
        plt.ylabel("x-coordinate")

        cbar = plt.colorbar()
        cbar.ax.set_ylabel("", rotation=90)

        list_of_markers = ["o", "v", "s", "p", "D"]
        
        for idx, dpos in enumerate(list_of_drone_positions):
            
            marker_idx = list_of_markers[idx]
            plt.scatter(dpos[1], dpos[0], color='white', facecolors='none',linewidth=1,marker=marker_idx,s=200)
            
    def dynamic_importance(self, t, x, y, T):
        
        if self.prestored:
            
            return self.prestored_dynamical_importance_map[t, x, y]
        
        else:
        
            if np.sin(2 * np.pi * (5/T) * t) > 0:
                
                importance_map_s_2 = np.sin(2 * np.pi * (5/T) * t)* self.importance_map_s[x, y]
                
            else:
                
                importance_map_s_2 = 0.0
                    
            importance_map_b_2 = (np.exp(-t/(0.7*T))) * self.importance_map_b[x, y]
            
            res = importance_map_s_2 + importance_map_b_2
            if res > 1.0 : res = 1.0
            
            return res
    
    def get_dynamic_map(self, t, T):
        
        dynamic_importance_map = np.zeros((self.x_size, self.y_size))
        
        for i in range(self.x_size):
            for j in range(self.y_size):
                
                dynamic_importance_map[i, j] = self.dynamic_importance(t, i, j, T)
                
        return dynamic_importance_map
    
    def create_obstacles_map(self):
        
        obstacle_coords = []
        
        for x in range(self.x_size):
            for y in range(self.y_size):
                
                if np.sum(self.prestored_dynamical_importance_map[:,x,y] > 0.0) == 0.0:
                    
                    obstacle_coords.append([x,y])
                    
        for (x, y) in obstacle_coords:
            
            self.not_obstacles_mask[x, y] = 0.0
    

