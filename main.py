import numpy as np
import torch
import random

from generate_alkanes import generate_branched_alkane

import logging
import pickle
logging.basicConfig(level=logging.DEBUG)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    mol = generate_branched_alkane(14)
    