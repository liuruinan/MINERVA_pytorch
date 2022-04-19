import numpy as np
import random
import torch
import time

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

