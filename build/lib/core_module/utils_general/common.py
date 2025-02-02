import json
import os
import os.path as osp
import pickle
import numpy as np


def ensure_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


