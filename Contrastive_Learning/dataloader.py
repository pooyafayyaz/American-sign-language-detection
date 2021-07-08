import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from math import sqrt
import torch


class HandPatchdataset(Dataset):


    def row_to_array(data_str):
    	 return np.array(data_str[1:-1].strip().split()).reshape((-1,2))    	 
    	
    def __init__(self,data_path,sample_rate_sk):
        self.data = pd.read_csv(data_path)
        self.data.iloc[:,2] = self.data.iloc[:,2].apply(row_to_array)
        self.dataset = [] 
        for index, row in data.iterrows():
       	for item in row[2]:
       		self.dataset.add([row[0],row[1],item])
       		
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self,idx):
        return []


