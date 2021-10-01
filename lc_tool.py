import os
import time
import logging
import sys
import torch
import numpy as np
import random

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def setup_logger(root_folder,exp_id):
    if not os.path.exists(root_folder):
        os.system(f"mkdir {root_folder}")
    log_path = os.path.join(root_folder,exp_id)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level=logging.INFO)
    logger.addHandler(file_handler)
    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def getGrad(m):
    grads = {}
    for name, params in m.named_parameters():
        grad = params.grad
        if grad is None:
            continue
        grad_array = grad.data.cpu().numpy()
        grads[name] = np.abs(grad_array).mean()
    return grads