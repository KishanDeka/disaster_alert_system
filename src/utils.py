import numpy as np 
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib as mpl
import random
import shutil
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn

# Global Constants
CLASS_NAMES = ['Fire', 'Flood', 'Normal', 'Earthquake']
CLASS_COLORS = ['#FF4B4B', '#FF8C00', '#1E90FF', '#2ECC71']

def get_device():
    """Check cuda gpu or cpu"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=123):
    """seed for reproducibilty"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
