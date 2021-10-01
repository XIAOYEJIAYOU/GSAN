import os

opt_list = ["adam","sgd"]
lr_list = [1e-3,1e-4,1e-5]
bs_list = [256,1024,512,128,64]
sd_list = [i for i in range(3)]
max_epoch = 500
for opt in opt_list:
    for lr in lr_list:
        for bs in bs_list:
            for sd in sd_list:
                os.system(f"python train_3cls.py -sd={sd} -lr={lr} -bs={bs} -opt={opt} -ep={max_epoch}")

# sd = 0
# lr = 1e-4
# bs = 256
# opt = "adam"
# max_epoch = 1000
# os.system(f"python train.py -sd={sd} -lr={lr} -bs={bs} -opt={opt} -ep={max_epoch}")