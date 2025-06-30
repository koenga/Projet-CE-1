# Project Overview

This repository is divided into two main folders:

- FAKE DATA: Contains all notebooks, Python scripts, results, and trained models for synthetic data.
- REAL DATA: Same structure, but applied to the Barcelona map, using arrays from Map_configurations.

---

## FAKE DATA

Many files were not modified and are not described here.

1. Fleet.py

- The input of the file is modified to allow choosing between GRU and LSTM for the policy.
- Policy networks are implemented identically for both GRU and LSTM.
- The function used to move the drones is shared across both; only the policy itself differs.
- No other changes were made to the file.

2. Main.ipynb

- To choose between GRU or LSTM, change the variable 'mode' to 'gru' or 'lstm'.
- Other than that, it behaves the same as the provided main.py.

3. Main_Gather_Pretrain_Data_GRU.ipynb & Main_Pretrain_GRU.ipynb

- Follow the same architecture as the corresponding LSTM files.
- Adapted to work with GRU.

4. trajectories_GRU.gif & trajectories_LSTM.gif

- These are the animated visualizations included in the report.

---

## REAL DATA

0. Folder: Map_configurations

- Contains three arrays (one for each variable).
- Includes pretraining data for GRU on the Barcelona map.

1. Main.ipynb

Defines a run() function with the following parameters:

    run(
        link: str,                  # Path to all the pickle files
        df_link: str,               # Path to 'link_bboxes_clustered.csv'
        File: list[str] = ['000'],  # List of files to build the region map (defaults to loading from Map_configurations)
        id: str,                    # Type of variable in the map: 'pred_vdist', 'pred_vtime', or 'ld_speed'
        mode: str,                  # Agent type: 'random', 'greedy', 'lstm', or 'gru'
        load: bool,                 # Whether to load stored maps
        pretrained_folder: str,     # Path to the pretrained agent
        gif: bool                   # Whether to generate and save the gif
    )

2. RegionMap.py

This file contains a large number of functions used during the map-building process, including both synthetic map creation and data loading utilities. Since the necessary maps are already provided in the repository, we will not detail the map construction functions here. However, if you require any additional information, feel free to reach out by email.

We will instead focus on the map loading functionality:

    initialize_better_importance_map(id, gather_pretrain)

- If load = True:
    → Loads the precomputed map corresponding to the selected id.
- If load = False:
    → Creates a new map using specific files (listFileNumbers).
- The gather_pretrain flag:
    → If set to True, the function loads pretraining data from the Map_configurations folder.
    → Specifically, it loads the file:
      'pretrain data GRU pred_vdist v2.npy',
      which aggregates maps from various simulations for the pred_vdist variable into a single dataset used for pretraining.

3. Main_Gather_Pretrain_Data_GRU.ipynb & Main_Pretrain_GRU.ipynb

- These notebooks are structurally identical to their counterparts in the FAKE DATA section, but use real-world data for model training.

4. GIF Files

- This folder contains all the animated visualizations featured in the report.
- The filenames are descriptive and clearly indicate the content of each GIF.

---

## Contact

If you need any help understanding the code—especially regarding how the maps were built—feel free to reach out via email.
