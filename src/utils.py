import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
import shutil
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

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
        
def load_dataset_stats(csv_path="data/dataset_stats.csv"):
    """Reads mean and std from CSV and returns them as lists."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Stats CSV file not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    mean = df['mean'].tolist()
    std = df['std'].tolist()
    return mean, std
