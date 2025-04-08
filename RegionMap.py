# -*- coding: utf-8 -*-
import numpy as np
import os 
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

class RegionMap():
    
    def __init__(self, v_size, h_size, list_of_small_pertb, list_of_big_pert, timestep, df_link, link, listFileNumbers):
        
        self.y_size = h_size
        self.x_size = v_size

        self.timestep = timestep
        self.df_link = df_link
        self.link = link
        self.listFileNumbers = listFileNumbers
        
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


    def charger_fichier(self, numero):
        """Charge un fichier .pkl en fonction de son numéro"""
        
        # Construire le nom du fichier avec le bon format
        nom_fichier = f"agg_timeseries_{numero}.pkl"
        chemin_fichier = os.path.join(self.link, nom_fichier)
        
        # Vérifier si le fichier existe
        if not os.path.exists(chemin_fichier):
            print(f"❌ Fichier {nom_fichier} non trouvé !")
            return None

        # Charger le fichier pickle
        with open(chemin_fichier, "rb") as f:
            data = pickle.load(f)
        
        #print(f"✅ Fichier {nom_fichier} chargé avec succès !")
        return data  # Retourne le contenu du fichier

    def gridTensor(self):

        df_pos = self.df_link[['id', 'c_x', 'c_y']].copy()
        self.df_pos = df_pos

        # Définition du nombre de lignes et colonnes de la grille
        n_rows, n_cols = self.y_size, self.x_size

        # Déterminer les bornes des points
        xmin1, ymin1 = df_pos[['c_x', 'c_y']].min()
        xmax1, ymax1 = df_pos[['c_x', 'c_y']].max()

        # Ajuster les bordures
        xmin = xmin1 - 0.05*(xmax1-xmin1)
        xmax = xmax1 + 0.05*(xmax1-xmin1)

        ymin = ymin1 - 0.05*(ymax1-ymin1)
        ymax = ymax1 + 0.05*(ymax1-ymin1)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
        # Calculer la taille des cellules de la grille
        cell_width = (xmax - xmin) / n_cols
        cell_height = (ymax - ymin) / n_rows

        # Fonction pour obtenir l'indice de grille d'un point
        def assign_to_grid(x, y):
            col = int((x - xmin) / cell_width)
            row = int((ymax - y) / cell_height)  # Inversion pour que 0 soit en haut
            col = min(col, n_cols - 1)  # S'assurer que les indices restent dans la grille
            row = min(row, n_rows - 1)
            return (row, col)

        # Appliquer l'assignation de grille à chaque point
        df_pos['grid_cell'] = df_pos.apply(lambda row: assign_to_grid(row['c_x'], row['c_y']), axis=1)

        # Afficher le DataFrame résultant dans la console
        print(df_pos)

    def visualizeGrid(self):
        n_rows, n_cols = self.y_size, self.x_size
        # Visualisation
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        df_pos = self.df_pos
        
        cell_width = (xmax - xmin) / n_cols
        cell_height = (ymax - ymin) / n_rows
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Dessiner la grille
        for i in range(1, n_cols):
            ax.axvline(xmin + i * cell_width, color='gray', linestyle='--')
        for i in range(1, n_rows):
            ax.axhline(ymin + i * cell_height, color='gray', linestyle='--')

        # Afficher les points avec leur ID et leur cellule
        for _, row in df_pos.iterrows():
            ax.scatter(row['c_x'], row['c_y'], color='hotpink')
            #ax.text(row['c_x'], row['c_y'], f"{row['id']}", fontsize=8, verticalalignment='bottom')

        plt.show()

    def getdf_id(self, id, pklidx):

        # Dossier où se trouvent les fichiers
        dossier = self.link
        
        # Construire le nom du fichier avec le bon format
        nom_fichier = f"agg_timeseries_{pklidx}.pkl"
        chemin_fichier = os.path.join(dossier, nom_fichier)

        # Vérifier si le fichier existe
        if not os.path.exists(chemin_fichier):
            print(f"❌ Fichier {nom_fichier} non trouvé !")
            return None

        # Charger le fichier pickle
        with open(chemin_fichier, "rb") as f:
            data = pickle.load(f)

        if id == 'ld_speed':
            self.df_id = data[id]
        else:
            df_copy = data[id].copy()
            for column in data[id].columns.values:
                num_lanes = self.df_link[self.df_link['id'] == column]['num_lanes'].values
                length = self.df_link[self.df_link['id'] == column]['length'].values
                df_copy[column] = data[id][column]/(num_lanes*length)
            self.df_id = df_copy

        return self.df_id

    def createMatrix(self, timestep, id): # df = df_pos
        df_id = self.df_id
        df = self.df_pos
        n_rows, n_cols = self.y_size, self.x_size

        tensor = np.zeros((n_rows, n_cols))

        for i in range(n_rows):
            for j in range(n_cols):
                somme = 0
                target_cell = (i,j)
                ids_in_cell = df[df['grid_cell'] == target_cell]['id'].tolist()
                for idx in ids_in_cell:
                    somme+= df_id.loc[timestep, idx]
                if id == 'ld_speed':
                    if len(ids_in_cell) == 0:
                        tensor[i,j] = 0 
                    else: 
                        tensor[i,j] = somme/len(ids_in_cell)
                else: 
                    if len(ids_in_cell) == 0:
                        tensor[i,j] = 0 
                    else: 
                        tensor[i,j] = somme
        return tensor
    
    
    def getMaxTimestep(self, id):
        listNumberTimestep = []
        for i in self.listFileNumbers:
            df_id = self.charger_fichier(i)[id]
            list_timestep = df_id.index.astype(str).tolist()
            listNumberTimestep.append(len(list_timestep))
        self.maxTimestep = max(listNumberTimestep)

        return max(listNumberTimestep)

    def createTensor3D(self,id,pklidx):
        n_rows, n_cols = self.y_size, self.x_size
        maxTimestep = self.getMaxTimestep(self, id)
        list_timestep = self.getdf_id(id,pklidx).index.astype(str).tolist()
        listTensor = []
        for i in list_timestep:
            tensor = self.createMatrix(i, id)
            listTensor.append(tensor)
        
        if len(listTensor) < maxTimestep:
            addedTimesteps = maxTimestep - len(listTensor)

            zero_matrix = np.zeros((n_rows, n_cols))  # Matrice remplie de zéros
            listTensor.extend([zero_matrix] * addedTimesteps)  # Ajoût de matrices nulles 
        else:
            addedTimesteps = 0

        tensor3D = np.stack(listTensor, axis = 0)

        return tensor3D, addedTimesteps
    
    def createAllTensor3D(self, listFileNumbers, id):
        listAllTensor3D = []
        listAddedTimesteps = []

        for i in listFileNumbers:
            df_id = self.charger_fichier(i)[id]
            list_timestep = df_id.index.astype(str).tolist()
            tensor3D, addedTimesteps = self.createTensor3D(id,i)
            listAllTensor3D.append(tensor3D) 
            listAddedTimesteps.append(addedTimesteps) 

        return listAllTensor3D, listAddedTimesteps
    
    def averageAllTensor(self, listAllTensor3D, listAddedTimesteps):
        # sommer tous les tenseurs pour obtenir un tenseur de forme (maxTimestep, n_rows, n_cols)
        sum_array = np.sum(np.stack(listAllTensor3D), axis=0)
        mean_array = np.zeros(sum_array.shape)
        # liste qui contient le nombre de timestep ne contenant pas de zeros pour chaque pickle file
        tracking = np.ones(len(listAddedTimesteps))*(sum_array.shape[0]) - listAddedTimesteps

        for i in range(sum_array.shape[0]):
            array = sum_array[i,:,:]
            # on divise seulement par le nombre d'entrée de la liste qui sont positifs
            mean_array[i,:,:] = array/sum(x > 0 for x in tracking)
            tracking -= 1

        return mean_array
    
        # Fonction pour visualiser la grille avec une heatmap
    def visualize_grid_with_heatmap(self, tensor3D, n_timestep):
        
        vmax = np.max(tensor3D)
        vmin = np.min(tensor3D)

        n_rows, n_cols = self.y_size, self.x_size
        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = self.ymin, self.ymax
        df = self.df_pos

        # Calculer la taille des cellules de la grille
        cell_width = (xmax - xmin) / n_cols
        cell_height = (ymax - ymin) / n_rows
        
        # Affichage de la heatmap avec échelle fixe
        for i in range(n_timestep):
            
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(tensor3D[i,:,:], cmap='Reds', interpolation='nearest', origin='upper', 
                        extent=[xmin, xmax, ymin, ymax], vmin=vmin, vmax=vmax)

            # Ajouter la barre de couleur
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Valeur du tensor (e.g., vitesse moyenne)")

            # Dessiner la grille
            for i in range(1, n_cols):
                ax.axvline(xmin + i * cell_width, color='gray', linestyle='--', linewidth=0.5)
            for i in range(1, n_rows):
                ax.axhline(ymin + i * cell_height, color='gray', linestyle='--', linewidth=0.5)

            # Afficher les points
            for _, row in df.iterrows():
                ax.scatter(row['c_x'], row['c_y'], color='black', s=10)

            # Supprimer les graduations et labels des axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.title("Grille avec Heatmap des valeurs du tensor")
            plt.show()


    def NormalizeTensor(self, tensor):

        maxValues = np.max(tensor, axis=(1,2))
        maxValues[maxValues == 0] = 1  # Remplace les valeurs nulles par 1
        maxValuesReshaped = maxValues[:, np.newaxis, np.newaxis]
        
        normalizedTensor = tensor / maxValuesReshaped

        return normalizedTensor

    
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
    
    def initialize_better_importance_map(self, id):
        listFileNumbers = self.listFileNumbers
        
        listAllTensor3D, listAddedTimesteps = self.createAllTensor3D(listFileNumbers, id)
        mean_tensor3D = self.averageAllTensor(listAllTensor3D, listAddedTimesteps)

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
    

