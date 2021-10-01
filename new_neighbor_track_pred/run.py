import os

cmd_list = [
    "python train.py -opt=sgd -lr=1e-5 -ep=300",
    "python train.py -opt=sgd -lr=1e-4 -ep=300",
    "python train.py -opt=adam -lr=1e-4 -ep=300",
    "python train.py -opt=adam -lr=1e-5 -ep=300"
]

for cmd in cmd_list:
    os.system(cmd)