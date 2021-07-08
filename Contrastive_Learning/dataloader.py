import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from math import sqrt
import torch


class HandPatchdataset(Dataset):
    def __init__(self,data_path,sample_rate_sk):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return self.data["video"].max()

    def __getitem__(self,idx):
        
        return (torch.Tensor(sampled_frames),torch.Tensor(label))


