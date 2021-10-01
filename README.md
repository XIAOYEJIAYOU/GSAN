## GSAN

### Introduction

Code for paper _**Graph Self-Attention Network for Learning Spatial-Temporal Interaction Representation in Autonomous Driving**_, which was published on IEEE Internet of Things Journal.

To reference the code, please cite this publication:

  ```
    @article{ye2021gsan,
      title={GSAN: Graph Self-Attention Network for Learning Spatial-Temporal Interaction Representation in Autonomous Driving},
      author={Ye, Luyao and Wang, Zezhong and Chen, Xinhong and Wang, Jianping and Wu, Kui and Lu, Kejie},
      journal={IEEE Internet of Things Journal},
      year={2021},
      publisher={IEEE}
    }
  ```



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

- Datasets(NGSIM us-101, i-80 and HighD) are not included in the repo, please download by yourself from the official website.

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
