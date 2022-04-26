import os
import random
import numpy as np
import torch


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
