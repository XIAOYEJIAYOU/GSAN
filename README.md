## GSAN

### Introduction

Code for paper _**GSAN: Graph Self-Attention Network for Learning Spatial-Temporal Interaction Representation in Autonomous Driving**_, which was published on IEEE Internet of Things Journal. And the link is [https://ieeexplore.ieee.org/document/9474961](https://ieeexplore.ieee.org/document/9474961).

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

### Datasets
- For lane-changing prediction task, we choose the open-source High-way Drone (HighD) Dataset.
- For trajectory prediction task, we choose NGSIM I-80 and US-101 Dataset.
- Datasets(NGSIM us-101, i-80 and HighD) are not included in the repo, please download by yourself from the official website.


### Quick Start

1. Install/Update python dependency library

    ```
    pip install -r requirements.txt
    ```

2. Build the directory

    ```
    python buildfolder.py
    ```

### Task1: Lane-changing classification
1. Get the data
    - HighD: [https://www.highd-dataset.com/](https://www.highd-dataset.com/)

2. Run all cells in `highD_data_process.ipynb`

### Task2: Trajectory prediction  

1. Get the data

    - NGSIM: [https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)

    - Follow this [introduction](https://github.com/nachiket92/conv-social-pooling) to pre-process the data and get following files:
        - TestSet.mat
        - TrainSet.mat
        - ValSet.mat

    - Put these 3 files into `data/` folder.

2. Format the data to fit GSAN model

    ```
    python datapreprocessing.py
    ```
