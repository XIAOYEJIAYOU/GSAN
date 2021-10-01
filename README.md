## Spatio-Temporal Interaction Representation (STIR) 

### Introduction

Code for paper _**GSAN: Graph Self-Attention Network for Learning Spatial-Temporal Interaction Representation in Autonomous Driving**_. Previous code for _**GSAN: Graph Self-Attention Network for Interaction Measurement in Autonomous Driving**_ is moved to `gsan` branch for backup.

#### TODO: 

- Directory listed below has not been built in the repo.
    ```
    ./highD-dataset-v1.0/...
    ./pickle_data/

    new_neighbor_track_pred/data/
    new_neighbor_track_pred/model/

    new_neighbor/new_data/
    new_neighbor/model/
    new_neighbor/temp/right_pic
    ```
    
- Dataset(e.g. high-D) is not included in the repo.

- Some data pre-process pipline are not clear. (Run pre-process code in what order?)

### Quick Start



1. Install/Update python dependency library

    ```
    pip install -r requirements.txt
    ```

2. Build the directory

    ```
    python buildfolder.py
    ```

3. Get the data

    - [Data source](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)

    - Follow this [introduction](https://github.com/nachiket92/conv-social-pooling) to pre-process the data and get following files:
        - TestSet.mat
        - TrainSet.mat
        - ValSet.mat
    
    - Put these 3 files into `data/` folder.
    
4. Format the data to fit GSAN model

    ```
    python datapreprocessing.py
    ```

5. Train & validate

    ```
    python train.py
    ```

6. Test

    ```
    python test.py
    ```

