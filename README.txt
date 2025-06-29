The files are seperated in two big folders:
- FAKE DATA: contains all the notebooks, python scripts, results and trained model for the synthetic data
- REAL DATA: same but with the Barcelone map, using the array from Map_configurations

First, let's go through FAKE DATA:

The are a lot of files that were not touched, so they won't be mentionned here.

1. Fleet.py: The input of the files are modified so that you can choose between GRU and LSTM when choosing the policy.
             The policy networks are implemented the same way for GRU and LSTM. If you are using GRU, the function used
             for moving the drones is the same as for LSTM, only the policy used changes.
             Otherwise, no changes.

2. Main.ipynb: To choose wether you want gru or lstm, you need to change the variable 'mode' to 'gru' or 'lstm'
               Otherwise it is the same as for the main.py, you gave us.

3. Main_Gather_Pretrain_Data_GRU.ipynb and Main_Pretrain_GRU.ipynb: Follow the same architecture as the corresponding files
                                                                    for LSTM, only adapted for the GRU.

4. trajectories GRU.gif and trajectories LSTM.gif are the gifs that are in the report

Now, let's go through REAL:

0. Folder Map_configurations: contains three arrays that the three map (one for each variable) and the pretraining data 
                              for the GRU in the Map of Barcelona

1. Main.ipynb: You have a run() function with the following parameters
            - link: str, path that should lead to all the pickle files
            - df_link': str, path used to open the file 'link_bboxes_clustered.csv'
            - File: list of string that contains the files you want to use to build the region map, in the code it is 
                    a default value ['000'] because we load the maps from the folder 'Map_configurations'
            - id: str, type of variable you want in your Map ('pred_vdist', 'pred_vtime' or 'ld_speed')
            - mode: str, type of agent you want to use ('random', 'greedy', 'lstm' or 'gru')
            - load: bool, if you want to load stored maps
            - pretrained_folder: str, path that leads to the pretrained agent
            - gif: bool, if you want to store the gif or not.

2. RegionMap.py:
This file is a bit messy, there are a lot of functions that we used to build the maps, functions to build the synthetic
map and load functions. Since you have the maps already, we won't guide through the building functions, but if you need
any information, feel free to send us an email.
We'll focus on the loading part.

- initialize_better_importance_map(id, gather_pretrain): If load = True, then it just loads the map corresponding to the 
                                                         chosen id.
                                                         If load = False, then it creates a map based on files you want.
                                                         (listFileNumbers)
                                                         The gather_pretrain variable is used only if you want to gather 
                                                         pretrain data, in this case we load the data from the folder Map_configurations:
                                                         'pretrain data GRU pred_vdist v2.npy' which is all the maps of the
                                                         differents simulation for the pred_vdist variable before we put them
                                                         all together into one map.

3. Main_Gather_Pretrain_Data_GRU and Main_Pretrain_GRU.ipynb: Same as the one in FAKE DATA, only it uses the real data to train this time.


4. GIFs files: All the different gifs that are in the report, the name are explicit of what they contain.
                                                          


